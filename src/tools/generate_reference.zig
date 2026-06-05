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
    /// 模型文件路径
    model_path: []const u8 = "",
    /// 输入 prompt（将被 tokenize）
    prompt: []const u8 = "Hello",
    /// 输出文件路径
    output_path: []const u8 = "ref_logits.bin",
    /// 输出格式：binary 或 text
    output_format: enum { binary, text } = .binary,
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
    ) !struct { logits: []f32, tokens: []i32 } {
        // 1. 加载 GGUF 文件
        const dir = std.Io.Dir.cwd();
        const file = try dir.openFile(io, self.config.model_path, .{ .mode = .read_only });
        defer file.close();

        const file_size = try file.getEndPos();
        const gguf_data = try file.readToEndAlloc(self.allocator, file_size);
        errdefer self.allocator.free(gguf_data);

        var gguf_file = try gguf.parse(gguf_data, self.allocator);
        errdefer gguf_file.deinit();

        // 2. 检测架构
        const arch = registry.detectArchitecture(&gguf_file) orelse {
            log.err("Could not detect architecture from {s}", .{self.config.model_path});
            return error.UnsupportedArchitecture;
        };
        log.info("Detected architecture: {s}", .{@tagName(arch)});

        // 3. 创建模型
        var model = try registry.createModel(self.allocator, &gguf_file, arch, io);
        errdefer model.deinit();

        const params = model.getParams();
        log.info("Model params: n_vocab={d}, n_embd={d}, n_layer={d}, n_head={d}",
            .{ params.n_vocab, params.n_embd, params.n_layer, params.n_head });

        // 4. Tokenize prompt
        const tokenizer = @import("tokenizer");
        var tok = try tokenizer.Tokenizer.init(self.allocator, &gguf_file, io);
        defer tok.deinit();

        const input_tokens = try tok.encode(self.config.prompt, self.allocator);
        defer self.allocator.free(input_tokens);

        log.info("Prompt: \"{s}\" -> {d} tokens", .{ self.config.prompt, input_tokens.len });

        // 5. 构建推理图并执行
        const n_tokens: i32 = @intCast(input_tokens.len);

        const ctx = try ggml.Context.initNoAlloc(256 * 1024);
        defer ctx.deinit();

        ctx.setNoAlloc(false);
        const input_tensor = try ctx.newTensor1d(.i32, n_tokens);
        ctx.setNoAlloc(true);

        // 复制输入 token
        const data = input_tensor.dataBytes();
        const dst = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
        @memcpy(dst, input_tokens);

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
        ggml.backendCpuSetNThreads(n_threads);
        graph.computeWithCtx();

        // 6. 读取 logits
        const logits_data = logits_tensor.dataBytes();
        const n_logits = @as(usize, @intCast(params.n_vocab));
        const logits = try self.allocator.alloc(f32, n_logits);
        const src = @as([*]f32, @ptrCast(@alignCast(logits_data.ptr)))[0..n_logits];
        @memcpy(logits, src);

        // 7. 保存到文件
        try self.saveOutput(logits, input_tokens, io);

        return .{ .logits = logits, .tokens = input_tokens };
    }

    /// 保存输出到文件
    fn saveOutput(
        self: *ReferenceGenerator,
        logits: []const f32,
        tokens: []const i32,
        io: std.Io,
    ) !void {
        const dir = std.Io.Dir.cwd();

        switch (self.config.output_format) {
            .binary => {
                // 二进制格式：先写 token 数（u32），再写 token ids（i32），再写 logits（f32）
                const file = try dir.createFile(io, self.config.output_path, .{});
                defer file.close();

                // 写入 token 数
                var n_tokens: u32 = @intCast(tokens.len);
                try file.writeAll(std.mem.asBytes(&n_tokens));

                // 写入 token ids
                try file.writeAll(std.mem.sliceAsBytes(tokens));

                // 写入 logits
                try file.writeAll(std.mem.sliceAsBytes(logits));

                log.info("Saved reference output to {s} ({d} tokens, {d} logits)", .{
                    self.config.output_path, tokens.len, logits.len,
                });
            },
            .text => {
                // 文本格式：每行一个值，先写 token ids，再写 logits
                const file = try dir.createFile(io, self.config.output_path, .{});
                defer file.close();

                var writer = file.writer();

                try writer.print("# Reference output\n", .{});
                try writer.print("# Model: {s}\n", .{self.config.model_path});
                try writer.print("# Prompt: {s}\n", .{self.config.prompt});
                try writer.print("# Tokens: {d}\n", .{tokens.len});
                try writer.print("# Logits: {d}\n", .{logits.len});

                if (self.config.output_tokens) {
                    try writer.print("# Token IDs:\n", .{});
                    for (tokens) |t| {
                        try writer.print("{d}\n", .{t});
                    }
                }

                try writer.print("# Logits:\n", .{});
                for (logits) |l| {
                    try writer.print("{e}\n", .{l});
                }

                log.info("Saved reference output (text) to {s}", .{self.config.output_path});
            },
        }
    }
};

// ============================================================================
// ============================================================================
// 命令行入口
// ============================================================================

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.debug.print("zllama-gen-ref: Generate reference logits\n", .{});
    std.debug.print("Usage: zllama-gen-ref <model.gguf> [options]\n", .{});
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
