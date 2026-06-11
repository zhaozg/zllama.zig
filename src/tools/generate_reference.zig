//! 参考输出生成器
//!
//! 使用 zllama 引擎为各模型生成参考 logits，用于回归测试。
//! 输出为二进制格式，供 compare_logits 工具使用。
//!
//! 用法：
//!   zllama-gen-ref --model model.gguf --prompt "Hello" --output ref_logits.bin
//!
//! 参考 llama.cpp 的生成参考输出方法。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const log = std.log.scoped(.gen_ref);

// ============================================================================
// 参考输出配置
// ============================================================================

pub const ReferenceConfig = struct {
    pub const OutputFormat = enum { binary, text };

    /// 模型文件路径
    model_path: []const u8 = "",
    /// 输入 prompt（将被 tokenize）
    prompt: []const u8 = "Hello",
    /// 输出文件路径
    output_path: []const u8 = "ref_logits.bin",
    /// 输出格式：binary 或 text
    output_format: OutputFormat = .binary,
    /// 是否输出 token ids
    output_tokens: bool = false,
};

// ============================================================================
// 参考输出生成器
// ============================================================================

pub const ReferenceGenerator = struct {
    allocator: std.mem.Allocator,
    config: ReferenceConfig,

    pub fn init(allocator: std.mem.Allocator, config: ReferenceConfig) ReferenceGenerator {
        return ReferenceGenerator{ .allocator = allocator, .config = config };
    }

    /// 生成参考输出
    /// 返回 logits 和 token ids
    pub fn generate(
        self: *ReferenceGenerator,
        io: std.Io,
    ) !struct { logits: []f32, tokens: []u32 } {
        // 1. 加载 GGUF 文件
        const dir = std.Io.Dir.cwd();
        const file = try dir.openFile(io, self.config.model_path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const file_size = @as(usize, @intCast(stat.size));
        const gguf_data = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(gguf_data);
        const bytes_read = try file.readPositionalAll(io, gguf_data, 0);
        if (bytes_read != file_size) return error.FileReadError;

        var gguf_file = try gguf.parse(gguf_data, self.allocator);
        defer gguf_file.deinit();

        // 2. 检测架构
        const arch = registry.detectArchitecture(&gguf_file) orelse {
            log.err("Could not detect architecture from {s}", .{self.config.model_path});
            return error.UnsupportedArchitecture;
        };
        log.info("Detected architecture: {s}", .{@tagName(arch)});

        // 3. 创建模型
        var model = try registry.createModel(self.allocator, &gguf_file, arch, io);
        defer model.deinit(self.allocator);

        const params = model.getParams();
        log.info("Model params: n_vocab={d}, n_embd={d}, n_layer={d}, n_head={d}",
            .{ params.n_vocab, params.n_embd, params.n_layer, params.n_head });

        // 4. Tokenize prompt
        const tokenizer = @import("tokenizer");
        var tok = try tokenizer.Tokenizer.init(&gguf_file, self.allocator);
        defer tok.deinit();

        var input_token_list = try tok.encode(self.config.prompt, false);
        defer input_token_list.deinit(self.allocator);
        const input_tokens = input_token_list.items;

        log.info("Prompt: \"{s}\" -> {d} tokens", .{ self.config.prompt, input_tokens.len });

        // 5. 构建推理图并执行
        const n_tokens: i32 = @intCast(input_tokens.len);

        const ctx = try ggml.Context.initNoAlloc(256 * 1024 * 1024);
        defer ctx.deinit();

        ctx.setNoAlloc(false);
        const input_tensor = try ctx.newTensor1d(.i32, n_tokens);
        ctx.setNoAlloc(true);

        // 复制输入 token
        const data = input_tensor.dataBytes();
        const dst = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
        for (input_tokens, 0..) |token, j| {
            dst[j] = @as(i32, @intCast(token));
        }

        // 构建计算图
        const graph = try ggml.CGraph.init(ctx);
        var builder = graph_builder.GraphBuilder.init(ctx, graph, params, self.allocator);
        const logits_tensor = try model.buildGraph(&builder, input_tensor, n_tokens, null, 0);

        // 分配并执行
        const buft = ggml.backendCpuBufferType();
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();

        if (!galloc.allocGraph(graph)) {
            return error.GraphAllocationFailed;
        }

        const n_threads = @as(i32, @intCast(@min(4, @max(1, try std.Thread.getCpuCount() - 1))));
        try graph.compute(n_threads);

        // 6. 读取 logits
        const logits_data = logits_tensor.dataBytes();
        const n_logits = @as(usize, @intCast(params.n_vocab));
        const logits = try self.allocator.alloc(f32, n_logits);
        const src = @as([*]f32, @ptrCast(@alignCast(logits_data.ptr)))[0..n_logits];
        @memcpy(logits, src);

        // 7. 保存到文件
        try self.saveOutput(logits, input_tokens, io);

        // 复制 tokens 以便返回
        const tokens_copy = try self.allocator.dupe(u32, input_tokens);

        return .{ .logits = logits, .tokens = tokens_copy };
    }

    /// 保存输出到文件
    fn saveOutput(
        self: *ReferenceGenerator,
        logits: []const f32,
        tokens: []const u32,
        io: std.Io,
    ) !void {
        const dir = std.Io.Dir.cwd();

        switch (self.config.output_format) {
            .binary => {
                // 二进制格式：先写 token 数（u32），再写 token ids（i32），再写 logits（f32）
                const file = try dir.createFile(io, self.config.output_path, .{});
                defer file.close(io);

                // 写入 token 数
                var n_tokens: u32 = @intCast(tokens.len);
                try file.writeStreamingAll(io, std.mem.asBytes(&n_tokens));

                // 写入 token ids
                try file.writeStreamingAll(io, std.mem.sliceAsBytes(tokens));

                // 写入 logits
                try file.writeStreamingAll(io, std.mem.sliceAsBytes(logits));

                log.info("Saved reference output to {s} ({d} tokens, {d} logits)", .{
                    self.config.output_path, tokens.len, logits.len,
                });
            },
            .text => {
                // 文本格式：每行一个值，先写 token ids，再写 logits
                const file = try dir.createFile(io, self.config.output_path, .{});
                defer file.close(io);

                // 逐行写入，使用固定缓冲区
                var line_buf: [1024]u8 = undefined;

                inlineLine: {
                    const line = std.fmt.bufPrint(&line_buf, "# Reference output\n", .{}) catch break :inlineLine;
                    try file.writeStreamingAll(io, line);
                }
                inlineLine2: {
                    const line = std.fmt.bufPrint(&line_buf, "# Model: {s}\n", .{self.config.model_path}) catch break :inlineLine2;
                    try file.writeStreamingAll(io, line);
                }
                inlineLine3: {
                    const line = std.fmt.bufPrint(&line_buf, "# Prompt: {s}\n", .{self.config.prompt}) catch break :inlineLine3;
                    try file.writeStreamingAll(io, line);
                }
                inlineLine4: {
                    const line = std.fmt.bufPrint(&line_buf, "# Tokens: {d}\n", .{tokens.len}) catch break :inlineLine4;
                    try file.writeStreamingAll(io, line);
                }
                inlineLine5: {
                    const line = std.fmt.bufPrint(&line_buf, "# Logits: {d}\n", .{logits.len}) catch break :inlineLine5;
                    try file.writeStreamingAll(io, line);
                }

                if (self.config.output_tokens) {
                    if (std.fmt.bufPrint(&line_buf, "# Token IDs:\n", .{})) |line| {
                        try file.writeStreamingAll(io, line);
                    } else |_| {}
                    for (tokens) |t| {
                        if (std.fmt.bufPrint(&line_buf, "{d}\n", .{t})) |tl| {
                            try file.writeStreamingAll(io, tl);
                        } else |_| continue;
                    }
                }

                {
                    if (std.fmt.bufPrint(&line_buf, "# Logits:\n", .{})) |line| {
                        try file.writeStreamingAll(io, line);
                    } else |_| {}
                }
                for (logits) |l| {
                    if (std.fmt.bufPrint(&line_buf, "{e}\n", .{l})) |tl| {
                        try file.writeStreamingAll(io, tl);
                    } else |_| continue;
                }

                log.info("Saved reference output (text) to {s}", .{self.config.output_path});
            },
        }
    }
};

// ============================================================================
// 命令行入口
// ============================================================================
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    // 简单参数解析
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    var model_path: ?[]const u8 = null;
    var prompt: []const u8 = "Hello";
    var output_path: ?[]const u8 = null;
    var output_format: ReferenceConfig.OutputFormat = .binary;
    var output_tokens: bool = false;

    // 跳过 argv[0]
    _ = args_iter.next();

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            model_path = args_iter.next() orelse {
                std.debug.print("Error: --model requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            prompt = args_iter.next() orelse {
                std.debug.print("Error: --prompt requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--output") or std.mem.eql(u8, arg, "-o")) {
            output_path = args_iter.next() orelse {
                std.debug.print("Error: --output requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--format") or std.mem.eql(u8, arg, "-f")) {
            const fmt = args_iter.next() orelse {
                std.debug.print("Error: --format requires a value (binary|text)\n", .{});
                std.process.exit(1);
            };
            if (std.mem.eql(u8, fmt, "text")) {
                output_format = .text;
            } else if (!std.mem.eql(u8, fmt, "binary")) {
                std.debug.print("Error: unknown format '{s}', expected 'binary' or 'text'\n", .{fmt});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            output_tokens = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            std.debug.print("Usage: zllama-gen-ref [options]\n", .{});
            std.debug.print("Options:\n", .{});
            std.debug.print("  --model, -m <path>     Model GGUF file path (required)\n", .{});
            std.debug.print("  --prompt, -p <text>    Input prompt (default: \"Hello\")\n", .{});
            std.debug.print("  --output, -o <path>    Output file path (default: ref_logits.bin)\n", .{});
            std.debug.print("  --format, -f <fmt>     Output format: binary|text (default: binary)\n", .{});
            std.debug.print("  --tokens               Include token IDs in output\n", .{});
            std.debug.print("  --help, -h             Show this help\n", .{});
            return;
        }
    }

    if (model_path == null) {
        std.debug.print("Error: --model is required\n", .{});
        std.debug.print("Usage: zllama-gen-ref --model <model.gguf> [options]\n", .{});
        std.process.exit(1);
    }

    const config = ReferenceConfig{
        .model_path = model_path.?,
        .prompt = prompt,
        .output_path = output_path orelse "ref_logits.bin",
        .output_format = output_format,
        .output_tokens = output_tokens,
    };

    var gen = ReferenceGenerator.init(allocator, config);
    const result = try gen.generate(io);
    defer allocator.free(result.logits);
    defer allocator.free(result.tokens);

    log.info("Generated reference output: {d} logits, {d} tokens", .{ result.logits.len, result.tokens.len });
    log.info("First 5 logits: {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}", .{
        result.logits[0], result.logits[1], result.logits[2], result.logits[3], result.logits[4],
    });
}


// 测试
// ============================================================================

const testing = std.testing;

test "ReferenceConfig defaults" {
    const config = ReferenceConfig{};
    try testing.expectEqualStrings("Hello", config.prompt);
    try testing.expectEqualStrings("ref_logits.bin", config.output_path);
    try testing.expectEqual(@as(u8, 0), @intFromEnum(config.output_format));
}

test "ReferenceGenerator init" {
    const config = ReferenceConfig{ .model_path = "test.gguf" };
    const gen = ReferenceGenerator.init(testing.allocator, config);
    try testing.expectEqualStrings("test.gguf", gen.config.model_path);
}
