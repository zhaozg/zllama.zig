//! zllama.zig 入口
//!
//! zllama.zig - 多模型本地推理引擎 - 主入口点
//! 处理 CLI 参数、初始化、推理循环
//! 支持多模型架构（Qwen / LLaMA 等）
//! 实现首 token 完整图推理 + 增量解码
//!
//! 解码流程：
//! 1. 编码 prompt -> token IDs（不自动添加 BOS，由模型内部处理）
//! 2. 检测模型架构（从 GGUF 元数据）
//! 3. 首 token 完整前向计算（填充 KV Cache）
//! 4. 增量生成后续 tokens（每次 1 个 token）
//! 5. 所有 token 收集后一次性解码，确保 UTF-8 字节序列正确组合
//! 6. 输出完整文本

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const kv_cache = @import("kv_cache");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = log,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .tokenizer, .level = .info},
        .{ .scope = .ggml, .level = .info },
        .{ .scope = .qwen, .level = .info },
        .{ .scope = .llama, .level = .info },
        .{ .scope = .model, .level = .info },
        .{ .scope = .main, .level = .info },
        .{ .scope = .sampler, .level = .info },
    }
};

const logger = std.log.scoped(.main);

var runtime_log_level: std.log.Level = .info;

pub fn setLogLevel(level: std.log.Level) void {
    runtime_log_level = level;
}

pub fn getLogLevel() std.log.Level {
    return runtime_log_level;
}

pub fn log(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@intFromEnum(level) > @intFromEnum(runtime_log_level)) return;
    std.log.defaultLog(level, scope, format, args);
}

fn currentTimeMs() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) {
        return 0;
    }
    return @as(i64, ts.sec) * 1000 + @as(i64, @divTrunc(ts.nsec, 1000000));
}

const CliArgs = struct {
    model_path: [:0]const u8 = "",
    prompt: []const u8 = "Hello, how are you?",
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    n_threads: i32 = 0,
    verbose: bool = false,
    debug: bool = false,
    help: bool = false,

    pub fn parse(args_it: *std.process.Args.Iterator) !CliArgs {
        var result = CliArgs{};
        _ = args_it.next();
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                result.help = true;
            } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
                result.model_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
                result.prompt = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
                result.max_tokens = std.fmt.parseUnsigned(u32, args_it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--temperature") or std.mem.eql(u8, arg, "-t")) {
                result.temperature = std.fmt.parseFloat(f32, args_it.next() orelse return error.InvalidArgs) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--top-k") or std.mem.eql(u8, arg, "-k")) {
                result.top_k = std.fmt.parseUnsigned(u32, args_it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--top-p") or std.mem.eql(u8, arg, "-tp")) {
                result.top_p = std.fmt.parseFloat(f32, args_it.next() orelse return error.InvalidArgs) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--threads") or std.mem.eql(u8, arg, "-th")) {
                result.n_threads = std.fmt.parseInt(i32, args_it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
                result.verbose = true;
            } else if (std.mem.eql(u8, arg, "--debug") or std.mem.eql(u8, arg, "-d")) {
                result.debug = true;
            } else {
                logger.warn("unknown argument '{s}'", .{arg});
            }
        }
        return result;
    }

    pub fn printHelp() void {
        std.debug.print(
            \\zllama.zig - 多模型本地推理引擎
            \\
            \\用法: zllama [选项]
            \\
            \\选项:
            \\  -h, --help            显示此帮助信息
            \\  -m, --model <路径>     模型文件路径 (GGUF格式)
            \\  -p, --prompt <文本>    输入提示词
            \\  -n, --max-tokens <N>  最大生成token数 (默认: 256)
            \\  -t, --temperature <F> 采样温度 (默认: 0.7)
            \\  -k, --top-k <N>       Top-K 采样 (默认: 40)
            \\  -tp, --top-p <F>      Top-P 采样 (默认: 0.9)
            \\  -th, --threads <N>    线程数 (默认: auto)
            \\  -v, --verbose         详细输出
            \\  -d, --debug           输出调试日志
            \\
        , .{});
    }
};

const InferenceEngine = struct {
    allocator: std.mem.Allocator,
    ctx_weights: *ggml.Context,
    ctx_graph: *ggml.Context,
    ctx_kv_cache: *ggml.Context,
    arch: model_if.Architecture,
    model: model_if.ModelInstance,
    params: model_if.ModelParams,
    tok: tokenizer.Tokenizer,
    sampler_state: sampler.Sampler,
    kv_cache_mgr: kv_cache.KVCache,
    n_threads: i32,
    verbose: bool,
    gguf_data: []u8,

    pub fn init(io: std.Io, allocator: std.mem.Allocator, model_path: [:0]const u8, cli_args: *const CliArgs) !InferenceEngine {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, model_path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const file_size = @as(usize, @intCast(stat.size));
        const gguf_data = try allocator.alloc(u8, file_size);
        errdefer allocator.free(gguf_data);

        const bytes_read = try file.readPositionalAll(io, gguf_data, 0);
        if (bytes_read != file_size) {
            allocator.free(gguf_data);
            return error.FileReadError;
        }

        var gguf_file = try gguf.parse(gguf_data, allocator);
        defer gguf_file.deinit();

        const arch = registry.detectArchitecture(&gguf_file) orelse return error.UnsupportedArchitecture;
        logger.info("Detected architecture: {s}", .{@tagName(arch)});

        var model = try registry.createModel(allocator, &gguf_file, arch, io);
        errdefer model.deinit(allocator);

        const params = model.getParams().*;

        if (cli_args.verbose) {
            logger.info("n_vocab={d}, n_embd={d}, n_head={d}, n_kv_head={d}", .{ params.n_vocab, params.n_embd, params.n_head, params.n_kv_head });
            logger.info("n_layer={d}, n_ff={d}, n_head_dim={d}", .{ params.n_layer, params.n_ff, params.n_head_dim });
            logger.info("max_seq_len={d}, rope_theta={d}, rope_dim={d}", .{ params.max_seq_len, params.rope_theta, params.rope_dim });
        }

        var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
        errdefer tok.deinit();
        logger.info("Tokenizer: {d} tokens", .{tok.vocabSize()});

        const n_threads = if (cli_args.n_threads > 0) cli_args.n_threads else ggml.recommendedThreads();
        const mem_size_estimate = 2 * 1024 * 1024 * 1024;
        logger.info("Estimated memory: {d} MB", .{mem_size_estimate / (1024 * 1024)});
        const ctx_weights = try ggml.Context.initNoAlloc(mem_size_estimate);
        errdefer ctx_weights.deinit();
        const ctx_kv_cache = try ggml.Context.initNoAlloc(mem_size_estimate);
        errdefer ctx_kv_cache.deinit();
        const max_seq_len = @min(params.max_seq_len, 2048);
        var kv_cache_mgr = try kv_cache.KVCache.init(ctx_kv_cache, params.n_layer, params.n_kv_head, params.n_head_dim, max_seq_len, allocator);
        errdefer kv_cache_mgr.deinit(allocator);
        const ctx_graph = try ggml.Context.initNoAlloc(mem_size_estimate);
        errdefer ctx_graph.deinit();
        {
            const buft = ggml.backendCpuBufferType();
            try ggml.backendAllocCtxTensorsFromBuft(ctx_kv_cache, buft);
        }

        const sampler_state = sampler.Sampler.init(.{
            .temperature = cli_args.temperature,
            .top_k = cli_args.top_k,
            .top_p = cli_args.top_p,
        });

        return InferenceEngine{
            .allocator = allocator,
            .ctx_weights = ctx_weights,
            .ctx_graph = ctx_graph,
            .ctx_kv_cache = ctx_kv_cache,
            .arch = arch,
            .model = model,
            .params = params,
            .tok = tok,
            .sampler_state = sampler_state,
            .kv_cache_mgr = kv_cache_mgr,
            .n_threads = n_threads,
            .verbose = cli_args.verbose,
            .gguf_data = gguf_data,
        };
    }

    pub fn deinit(self: *InferenceEngine) void {
        self.kv_cache_mgr.deinit(self.allocator);
        self.tok.deinit();
        self.ctx_graph.deinit();
        self.ctx_kv_cache.deinit();
        // ctx_weights 由 model.deinit() 释放
        self.model.deinit(self.allocator);
        self.allocator.free(self.gguf_data);
    }

    pub fn generate(self: *InferenceEngine, prompt: []const u8, max_tokens: u32) !void {
        // 编码 prompt，添加特殊 token（BOS/EOS）
        var input_tokens = try self.tok.encode(prompt, true);
        // 调试：打印编码后的 token IDs
        logger.debug("Encoded tokens ({d}):", .{input_tokens.items.len});
        for (input_tokens.items, 0..) |t, i| {
            if (i < 20) logger.debug("  [{d}] = {d}", .{ i, t });
        }

        defer input_tokens.deinit(self.allocator);

        const n_prompt_tokens: i32 = @intCast(input_tokens.items.len);

        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_prompt_tokens);
        self.ctx_graph.setNoAlloc(true);

        logger.info("Building forward graph for prompt...", .{});
        var graph = try ggml.CGraph.init(self.ctx_graph);
        var builder = graph_builder.GraphBuilder.init(self.ctx_graph, graph, &self.params, self.allocator);
        const logits = try self.model.buildGraph(&builder, input_tensor, n_prompt_tokens, @ptrCast(&self.kv_cache_mgr), 0);

        logger.info("Allocating graph memory...", .{});
        const buft = ggml.backendCpuBufferType();
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();

        if (!galloc.allocGraph(graph)) {
            logger.err("Failed to allocate graph memory", .{});
            return error.GraphAllocFailed;
        }

        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_prompt_tokens))];
            for (input_tokens.items, 0..) |token, j| {
                slice[j] = @as(i32, @intCast(token));
            }
        }

        logger.info("Computing forward pass...", .{});
        const start_time = currentTimeMs();
        try graph.compute(self.n_threads);
        const end_time = currentTimeMs();
        const elapsed_ms = end_time - start_time;

        logger.info("Forward pass completed in {d} ms ({d:.2} tok/s)", .{
            elapsed_ms,
            @as(f64, @floatFromInt(n_prompt_tokens)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0),
        });

        const first_token = sampler.Sampler.sampleGreedy(logits);

        logger.info("first_token = {d}", .{first_token});

        var output_tokens = std.ArrayList(u32).initCapacity(self.allocator, 0) catch unreachable;
        defer output_tokens.deinit(self.allocator);

        try output_tokens.append(self.allocator, @as(u32, @intCast(first_token)));

        var pos: i32 = n_prompt_tokens;
        var current_token: u32 = @as(u32, @intCast(first_token));
        var gen_token_count: u32 = 0;
        const gen_start_time = currentTimeMs();

        // 增量解码循环
        while (gen_token_count < max_tokens - 1) {
            self.ctx_graph.reset();
            self.model.resetSSMStates();
            self.ctx_graph.setNoAlloc(false);
            const single_input = try self.ctx_graph.newTensor1d(.i32, 1);
            self.ctx_graph.setNoAlloc(true);

            var inc_graph = try ggml.CGraph.init(self.ctx_graph);
            var inc_builder = graph_builder.GraphBuilder.init(self.ctx_graph, inc_graph, &self.params, self.allocator);
            const inc_logits = try self.model.buildGraph(&inc_builder, single_input, 1, @ptrCast(&self.kv_cache_mgr), pos);

            var inc_galloc = try ggml.Gallocr.init(buft);
            defer inc_galloc.free();

            if (!inc_galloc.allocGraph(inc_graph)) {
                logger.err("Failed to allocate incremental graph memory", .{});
                return error.GraphAllocFailed;
            }

            {
                const data = single_input.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..1];
                slice[0] = @as(i32, @intCast(current_token));
            }

            try inc_graph.compute(self.n_threads);

            const next_token = sampler.Sampler.sampleGreedy(inc_logits);
            logger.debug("next_token = {d}", .{next_token});
            const next_token_u32 = @as(u32, @intCast(next_token));
            try output_tokens.append(self.allocator, next_token_u32);
            current_token = next_token_u32;
            pos += 1;
            gen_token_count += 1;
        }

        const gen_end_time = currentTimeMs();
        const gen_elapsed_ms = gen_end_time - gen_start_time;
        if (gen_token_count > 0 and gen_elapsed_ms > 0) {
            logger.info("Generation: {d} tokens in {d} ms ({d:.2} tok/s)", .{
                gen_token_count,
                gen_elapsed_ms,
                @as(f64, @floatFromInt(gen_token_count)) / (@as(f64, @floatFromInt(gen_elapsed_ms)) / 1000.0),
            });
        }

        const decoded_text = try self.tok.decode(output_tokens.items, self.allocator);
        defer self.allocator.free(decoded_text);

        logger.info("output_tokens count: {d}", .{output_tokens.items.len});
        if (output_tokens.items.len > 0) {
            logger.info("first few tokens: {d} {d} {d}", .{ output_tokens.items[0], output_tokens.items[1], output_tokens.items[2] });
        }
        logger.info("decoded text length: {d}", .{decoded_text.len});
        logger.info("decoded text: '{s}'", .{decoded_text});

        if (!std.unicode.utf8ValidateSlice(decoded_text)) {
            logger.warn("Generated text contains invalid UTF-8 sequences", .{});
            if (self.verbose) {
                tokenizer.hexDump(decoded_text[0..@min(decoded_text.len, @as(usize, 64))]);
            }
        }

        std.debug.print("{s}\n", .{decoded_text});

        const total_tokens = output_tokens.items.len + @as(usize, @intCast(n_prompt_tokens));
        logger.info("Generated {d} tokens (total {d})", .{ output_tokens.items.len, total_tokens });
    }
};

pub fn main(init: std.process.Init) !void {
    // 设置 ggml 日志回调
    ggml.logSet();
    const io = init.io;
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();


    const args = CliArgs.parse(&args_iter) catch |err| {
        if (err == error.InvalidArgs) {
            CliArgs.printHelp();
            return;
        }
        return err;
    };

    if (args.help) {
        CliArgs.printHelp();
        return;
    }
    if (args.debug) {
        setLogLevel(.debug);
    } else if (args.verbose) {
        setLogLevel(.info);
    } else {
        setLogLevel(.warn);
    }
    logger.info("zllama.zig v0.1.0 (ggml {s})", .{ggml.version()});

    if (args.model_path.len == 0) {
        logger.err("no model specified. Use --model <path>", .{});
        return;
    }

    logger.info("Loading model: {s}", .{args.model_path});
    var engine = InferenceEngine.init(io, allocator, args.model_path, &args) catch |err| {
        logger.err("Failed to initialize inference engine: {}\n", .{err});
        return;
    };
    defer engine.deinit();

    logger.info("Model loaded successfully.", .{});
    logger.info("Prompt: \"{s}\"", .{args.prompt});
    logger.info("Max tokens: {d}", .{args.max_tokens});

    logger.info("--- Generation ---", .{});
    engine.generate(args.prompt, args.max_tokens) catch |err| {
        logger.err("Generation failed: {}\n", .{err});
        return;
    };
    logger.info("--- Done ---", .{});
}

const testing = std.testing;

test "CliArgs parse" {
    const test_args = CliArgs{};
    try testing.expectEqual(@as(u32, 256), test_args.max_tokens);
    try testing.expectEqual(@as(f32, 0.7), test_args.temperature);
}

test "ggml version available" {
    const v = ggml.version();
    try testing.expect(v.len > 0);
}
// 导入所有测试模块（通过 zig build test 运行）

const test_utils = @import("tests/utils.zig");
const test_layers = @import("tests/test_layers.zig");
const test_gguf = @import("tests/test_gguf.zig");
const test_archs = @import("tests/test_archs.zig");
const test_kv_cache = @import("tests/test_kv_cache.zig");
const test_compare_logits = @import("tests/test_compare_logits.zig");
const test_vocab = @import("tests/test_vocab.zig");
