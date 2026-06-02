//! qwen-engine 入口
//!
//! Qwen 3.5 本地推理引擎 - 主入口点
//! 处理 CLI 参数、初始化、推理循环
//! 实现首 token 完整图推理

const std = @import("std");
const ggml = @import("ggml.zig");
const gguf = @import("gguf.zig");
const model = @import("model.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");
const kv_cache = @import("kv_cache.zig");

const log = std.log.scoped(.main);

// ============================================================================
// 时间工具
// ============================================================================

/// 获取当前毫秒时间戳（使用 POSIX clock_gettime）
fn currentTimeMs() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) {
        return 0;
    }
    return @as(i64, ts.sec) * 1000 + @as(i64, @divTrunc(ts.nsec, 1000000));
}

// ============================================================================
// CLI 参数
// ============================================================================

const CliArgs = struct {
    model_path: [:0]const u8 = "",
    prompt: []const u8 = "Hello, how are you?",
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    n_threads: i32 = 0, // 0 = auto
    verbose: bool = false,
    help: bool = false,

    pub fn parse(args_it: *std.process.Args.Iterator) !CliArgs {
        var result = CliArgs{};

        // Skip program name
        _ = args_it.next();

        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                result.help = true;
            } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
                result.model_path = args_it.next() orelse {
                    std.debug.print("Error: --model requires a path argument\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
                result.prompt = args_it.next() orelse {
                    std.debug.print("Error: --prompt requires a string argument\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
                result.max_tokens = std.fmt.parseUnsigned(u32, args_it.next() orelse {
                    std.debug.print("Error: --max-tokens requires a number\n", .{});
                    return error.InvalidArgs;
                }, 10) catch {
                    std.debug.print("Error: invalid --max-tokens value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--temperature") or std.mem.eql(u8, arg, "-t")) {
                result.temperature = std.fmt.parseFloat(f32, args_it.next() orelse {
                    std.debug.print("Error: --temperature requires a number\n", .{});
                    return error.InvalidArgs;
                }) catch {
                    std.debug.print("Error: invalid --temperature value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--top-k") or std.mem.eql(u8, arg, "-k")) {
                result.top_k = std.fmt.parseUnsigned(u32, args_it.next() orelse {
                    std.debug.print("Error: --top-k requires a number\n", .{});
                    return error.InvalidArgs;
                }, 10) catch {
                    std.debug.print("Error: invalid --top-k value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--top-p") or std.mem.eql(u8, arg, "-tp")) {
                result.top_p = std.fmt.parseFloat(f32, args_it.next() orelse {
                    std.debug.print("Error: --top-p requires a number\n", .{});
                    return error.InvalidArgs;
                }) catch {
                    std.debug.print("Error: invalid --top-p value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--threads") or std.mem.eql(u8, arg, "-th")) {
                result.n_threads = std.fmt.parseInt(i32, args_it.next() orelse {
                    std.debug.print("Error: --threads requires a number\n", .{});
                    return error.InvalidArgs;
                }, 10) catch {
                    std.debug.print("Error: invalid --threads value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
                result.verbose = true;
            } else {
                std.debug.print("Warning: unknown argument '{s}'\n", .{arg});
            }
        }

        return result;
    }

    pub fn printHelp() void {
        const help_text =
            \\Qwen 3.5 本地推理引擎
            \\
            \\用法: qwen [选项]
            \\
            \\选项:
            \\  -h, --help            显示此帮助信息
            \\  -m, --model <路径>     模型文件路径 (GGUF格式)
            \\  -p, --prompt <文本>    输入提示词 (默认: "Hello, how are you?")
            \\  -n, --max-tokens <N>  最大生成token数 (默认: 256)
            \\  -t, --temperature <F> 采样温度 (默认: 0.7)
            \\  -k, --top-k <N>       Top-K 采样 (默认: 40)
            \\  -tp, --top-p <F>      Top-P 采样 (默认: 0.9)
            \\  -th, --threads <N>    线程数 (默认: auto)
            \\  -v, --verbose         详细输出
            \\
        ;
        std.debug.print("{s}", .{help_text});
    }
};

// ============================================================================
// 推理引擎
const InferenceEngine = struct {
    allocator: std.mem.Allocator,
    ctx_weights: *ggml.Context,  // 权重 context（no_alloc=false，权重通过 setDataPtr 指向 GGUF 数据）
    ctx_graph: *ggml.Context,    // 计算图 context（no_alloc=true，由 backend 统一分配内存）
    params: model.ModelParams,
    weights: model.ModelWeights,
    tok: tokenizer.Tokenizer,
    sampler_state: sampler.Sampler,
    kv_cache_mgr: kv_cache.KVCache,
    n_threads: i32,
    verbose: bool,
    // 保存 GGUF 文件数据，生命周期与引擎相同
    gguf_data: []u8,


    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        model_path: [:0]const u8,
        cli_args: *const CliArgs,
    ) !InferenceEngine {
        // 读取 GGUF 文件
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, model_path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const file_size = @as(usize, @intCast(stat.size));

        // 分配并读取整个文件到内存
        // 注意：gguf.parse 返回的 GGUFFile 引用这块内存，所以不能提前释放
        const gguf_data = try allocator.alloc(u8, file_size);

        errdefer allocator.free(gguf_data);

        const bytes_read = try file.readPositionalAll(io, gguf_data, 0);
        if (bytes_read != file_size) {
            allocator.free(gguf_data);
            log.err("Short read: expected {d} bytes, got {d}", .{ file_size, bytes_read });
            return error.FileReadError;
        }

        // 解析 GGUF
        var gguf_file = try gguf.parse(gguf_data, allocator);

        defer gguf_file.deinit();

        // 解析模型参数
        var params = try model.parseParams(&gguf_file, allocator);

        if (cli_args.verbose) {
            log.info("Model parameters:", .{});
            log.info("  n_vocab={d}, n_embd={d}, n_head={d}, n_kv_head={d}", .{
                params.n_vocab, params.n_embd, params.n_head, params.n_kv_head,
            });
            log.info("  n_layer={d}, n_ff={d}, n_head_dim={d}", .{
                params.n_layer, params.n_ff, params.n_head_dim,
            });
            log.info("  max_seq_len={d}, rope_theta={d}, rope_dim={d}", .{
                params.max_seq_len, params.rope_theta, params.rope_dim,
            });
            log.info("  full_attn_interval={d}, ssm_inner={d}", .{
                params.full_attention_interval, params.ssm_inner_size,
            });
        }

        // 初始化分词器
        var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
        log.info("Tokenizer: {d} tokens", .{tok.vocabSize()});

        // 计算 ggml context 大小
        const n_threads = if (cli_args.n_threads > 0) cli_args.n_threads else ggml.recommendedThreads();

        // 估算内存需求
        const mem_size_estimate = blk: {
            const base_mem: usize = 1024 * 1024 * 1024; // 1GB base
            const per_layer: usize = @as(usize, @intCast((params.n_embd * (params.n_ff * 3 + params.n_embd * 2) * 4) +
                (params.ssm_inner_size * params.n_embd * 4 * 3)));
            break :blk base_mem + per_layer * params.n_layer * 2;
        };
        log.info("Estimated memory: {d} MB", .{mem_size_estimate / (1024 * 1024)});

        // ==========================================
        // 创建权重 context（no_alloc=false，权重通过 setDataPtr 指向 GGUF 数据）
        // ==========================================
        const ctx_weights = try ggml.Context.initNoAlloc(mem_size_estimate);

        // 加载权重（使用 setDataPtr 指向 GGUF 文件数据，不额外分配内存）
        const weights = try model.loadWeights(&gguf_file, gguf_data, &params, ctx_weights, allocator);

        // ==========================================
        // 创建计算图 context（no_alloc=true，由 backend 统一分配内存）
        // ==========================================
        const ctx_graph = try ggml.Context.initNoAlloc(mem_size_estimate);

        // 初始化 KV Cache（在 ctx_graph 中创建张量，no_alloc=true）
        const max_seq_len = @min(params.max_seq_len, 2048);
        const kv_cache_mgr = try kv_cache.KVCache.init(
            ctx_graph,
            params.n_layer,
            params.n_kv_head,
            params.n_head_dim,
            max_seq_len,
            allocator,
        );


        // 这包括 KV Cache 张量 (k, v) 以及后续计算图中的中间张量

        log.info("Allocating graph context memory via backend...", .{});
        {
            const buft = ggml.backendCpuBufferType();
            // 为 ctx_graph 中所有未分配的张量分配内存
            // KV Cache 的 k/v 张量此时 data 为 NULL，会被分配
            try ggml.backendAllocCtxTensorsFromBuft(ctx_graph, buft);
            log.info("Graph context memory allocated successfully", .{});
        }



        // 初始化采样器
        const sampler_state = sampler.Sampler.init(.{
            .temperature = cli_args.temperature,
            .top_k = cli_args.top_k,
            .top_p = cli_args.top_p,
        });

        return InferenceEngine{
            .allocator = allocator,
            .ctx_weights = ctx_weights,
            .ctx_graph = ctx_graph,
            .params = params,
            .weights = weights,
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
        self.weights.deinit(self.allocator);
        self.tok.deinit();
        self.ctx_graph.deinit();
        self.ctx_weights.deinit();
        self.allocator.free(self.gguf_data);
    }



    /// 运行推理：编码 prompt -> 首 token -> 增量生成
    /// 运行推理：编码 prompt -> 首 token -> 增量生成
    pub fn generate(self: *InferenceEngine, prompt: []const u8, max_tokens: u32) !void {
        // 1. 编码 prompt
        var input_tokens = try self.tok.encode(prompt);
        defer input_tokens.deinit(self.allocator);

        const n_prompt_tokens: i32 = @intCast(input_tokens.items.len);
        log.info("Prompt: {d} tokens", .{n_prompt_tokens});

        if (self.verbose) {
            std.debug.print("Input tokens: ", .{});
            for (input_tokens.items[0..@min(@as(usize, 10), input_tokens.items.len)]) |t| {
                std.debug.print("{d} ", .{t});
            }
            if (input_tokens.items.len > 10) std.debug.print("...", .{});
            std.debug.print("\n", .{});
        }

        // 2. 构建输入张量（在 ctx_graph 中创建，no_alloc=true，由 gallocr 分配内存）
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_prompt_tokens);
        self.ctx_graph.setNoAlloc(true);

        // 3. 构建首 token 计算图
        log.info("Building forward graph for prompt...", .{});
        var graph = try ggml.CGraph.init(self.ctx_graph);
        const logits = try model.buildForwardGraph(
            self.ctx_graph,
            graph,
            &self.weights,
            input_tensor,
            n_prompt_tokens,
            &self.kv_cache_mgr, // 传入 KV Cache，首 token 填充 Cache
            0, // start_pos = 0
            true, // is_qwen
        );

        // 4. 使用 Graph Allocator 为计算图分配中间张量内存
        log.info("Allocating graph memory...", .{});
        const buft = ggml.backendCpuBufferType();
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();

        if (!galloc.allocGraph(graph)) {
            log.err("Failed to allocate graph memory", .{});
            return error.GraphAllocFailed;
        }

        // 写入输入张量数据（内存已由 gallocr 分配）
        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_prompt_tokens))];
            for (input_tokens.items, 0..) |token, j| {
                slice[j] = @as(i32, @intCast(token));
            }
        }

        // 5. 执行计算
        log.info("Computing forward pass...", .{});
        const start_time = currentTimeMs();
        try graph.compute(self.n_threads);
        const end_time = currentTimeMs();
        const elapsed_ms = end_time - start_time;

        log.info("Forward pass completed in {d} ms ({d:.2} tok/s)", .{
            elapsed_ms,
            @as(f64, @floatFromInt(n_prompt_tokens)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0),
        });

        // 6. 采样首 token
        const first_token = sampler.Sampler.sampleGreedy(logits);
        log.info("First token: {d}", .{first_token});

        // 7. 解码并输出
        var output_tokens = std.ArrayList(u32).empty;
        defer output_tokens.deinit(self.allocator);

        try output_tokens.append(self.allocator, @as(u32, @intCast(first_token)));

        const first_token_u32 = @as(u32, @intCast(first_token));
        const first_text = try self.tok.decode(&.{first_token_u32}, self.allocator);
        defer self.allocator.free(first_text);
        std.debug.print("\n{s}", .{first_text});

        // 8. 增量生成剩余 tokens
        var current_token: u32 = @intCast(first_token);
        var pos: i32 = n_prompt_tokens;

        while (output_tokens.items.len < max_tokens) {
            // 检查 EOS
            if (current_token == self.tok.special.eos) {
                log.info("EOS token reached", .{});
                break;
            }

            // 构建单 token 输入
            self.ctx_graph.setNoAlloc(false);
            const single_input = try self.ctx_graph.newTensor1d(.i32, 1);
            self.ctx_graph.setNoAlloc(true);

            // 构建增量计算图
            var inc_graph = try ggml.CGraph.init(self.ctx_graph);
            const inc_logits = try model.buildForwardGraph(
                self.ctx_graph,
                inc_graph,
                &self.weights,
                single_input,
                1,
                &self.kv_cache_mgr, // 使用已有的 KV Cache
                pos, // start_pos = 当前序列位置
                true, // is_qwen
            );

            // 使用 gallocr 分配增量图的内存
            var inc_galloc = try ggml.Gallocr.init(buft);
            defer inc_galloc.free();

            if (!inc_galloc.allocGraph(inc_graph)) {
                log.err("Failed to allocate incremental graph memory", .{});
                return error.GraphAllocFailed;
            }

            // 写入输入 token 数据
            {
                const data = single_input.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..1];
                slice[0] = @as(i32, @intCast(current_token));
            }

            try inc_graph.compute(self.n_threads);

            // 采样
            const next_token = sampler.Sampler.sampleGreedy(inc_logits);

            // 解码
            const next_token_u32 = @as(u32, @intCast(next_token));
            const text = try self.tok.decode(&.{next_token_u32}, self.allocator);
            defer self.allocator.free(text);
            std.debug.print("{s}", .{text});

            try output_tokens.append(self.allocator, next_token_u32);
            current_token = next_token_u32;
            pos += 1;

            // 重置计算图 context，释放中间张量
            // 注意：ctx_graph 只包含计算图张量和 KV Cache，不包含权重
            // 权重在 ctx_weights 中，不受 reset 影响
            self.ctx_graph.reset();
        }

        std.debug.print("\n", .{});

        const total_tokens = output_tokens.items.len + @as(usize, @intCast(n_prompt_tokens));
        log.info("Generated {d} tokens (total {d})", .{ output_tokens.items.len, total_tokens });
    }

};


pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    // 解析 CLI 参数
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

    // 打印版本信息
    std.debug.print("Qwen Engine v0.1.0 (ggml {s})\n", .{ggml.version()});

    // 检测 CPU 特性
    if (args.verbose) {
        std.debug.print("CPU features:\n", .{});
        std.debug.print("  AVX2:  {}\n", .{ggml.CpuFeatures.hasAvx2()});
        std.debug.print("  AVX:   {}\n", .{ggml.CpuFeatures.hasAvx()});
        std.debug.print("  NEON:  {}\n", .{ggml.CpuFeatures.hasNeon()});
        std.debug.print("  Metal: {}\n", .{ggml.CpuFeatures.hasMetal()});
        std.debug.print("  CUDA:  {}\n", .{ggml.CpuFeatures.hasCuda()});
    }

    // 确定线程数
    const n_threads = if (args.n_threads > 0) args.n_threads else ggml.recommendedThreads();
    std.debug.print("Using {d} threads\n", .{n_threads});

    // 如果没有指定模型路径，显示帮助
    if (args.model_path.len == 0) {
        std.debug.print("Error: no model specified. Use --model <path> to specify a GGUF model file.\n", .{});
        std.debug.print("\nTip: You can also run with --help to see all options.\n", .{});
        return;
    }

    // 初始化推理引擎
    std.debug.print("\nLoading model: {s}\n", .{args.model_path});

    var engine = InferenceEngine.init(io, allocator, args.model_path, &args) catch |err| {
        log.err("Failed to initialize inference engine: {}", .{err});
        return;
    };
    defer engine.deinit();

    std.debug.print("Model loaded successfully.\n", .{});
    std.debug.print("Prompt: \"{s}\"\n", .{args.prompt});
    std.debug.print("Max tokens: {d}\n", .{args.max_tokens});
    std.debug.print("Temperature: {d}\n", .{args.temperature});
    std.debug.print("Top-K: {d}, Top-P: {d}\n", .{ args.top_k, args.top_p });

    // 运行推理
    std.debug.print("\n--- Generation ---\n", .{});
    engine.generate(args.prompt, args.max_tokens) catch |err| {
        log.err("Generation failed: {}", .{err});
        return;
    };
    std.debug.print("\n--- Done ---\n", .{});
}

// ============================================================================
// 测试
// ============================================================================

test "CliArgs parse" {
    const test_args = CliArgs{};
    try std.testing.expectEqual(@as(u32, 256), test_args.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.7), test_args.temperature);
}

test "ggml version available" {
    const v = ggml.version();
    try std.testing.expect(v.len > 0);
}
