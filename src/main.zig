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
const graph_context = @import("graph_context");
const memory = @import("memory");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const kv_cache = @import("kv_cache");
const mm = @import("mm");
const preprocess = @import("preprocess");

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
    benchmark: bool = false,
    chat: bool = false,
    info: bool = false,
    // Multimodal
    mmproj_path: [:0]const u8 = "",
    image_path: [:0]const u8 = "",
    audio_path: [:0]const u8 = "",
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
            } else if (std.mem.eql(u8, arg, "--chat") or std.mem.eql(u8, arg, "-c")) {
                result.chat = true;
            } else if (std.mem.eql(u8, arg, "--mmproj")) {
                result.mmproj_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--image")) {
                result.image_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--audio")) {
                result.audio_path = args_it.next() orelse return error.InvalidArgs;
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
            \\  --benchmark           benchmark 模式
            \\  -c, --chat            交互式聊天模式
            \\  --info                显示模型能力信息后退出
            \\
            \\多模态选项:
            \\  --mmproj <路径>       多模态投影器文件 (GGUF格式, mmproj)
            \\  --image <路径>        输入图像文件 (PPM/JPEG/PNG/BMP/GIF)
            \\  --audio <路径>        输入音频文件 (PCM F32, 16kHz)
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
    benchmark: bool,
    gguf_data: []u8,

    // Graph optimization: gallocr reuse context for incremental decoding
    inc_ctx: graph_context.IncContext,

    // Multimodal support
    mm_manager: ?mm.MultiModalManager = null,
    capabilities: model_if.ModelCapabilities = .{},

    pub fn init(io: std.Io, allocator: std.mem.Allocator, model_path: [:0]const u8, cli_args: *const CliArgs) !InferenceEngine {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, model_path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const file_size = @as(usize, @intCast(stat.size));
        const gguf_data = try allocator.alloc(u8, file_size);
        errdefer allocator.free(gguf_data);

        {
            var offset: u64 = 0;
            const chunk_size: usize = 64 * 1024 * 1024;
            while (offset < file_size) {
                const end = @min(offset + chunk_size, file_size);
                const len = end - offset;
                const bytes_read = try file.readPositionalAll(io, gguf_data[offset..][0..len], offset);
                if (bytes_read != len) {
                    allocator.free(gguf_data);
                    return error.FileReadError;
                }
                offset += bytes_read;
            }
        }

        var gguf_file = try gguf.parse(gguf_data, allocator);
        defer gguf_file.deinit();

        const arch = registry.detectArchitecture(&gguf_file) orelse return error.UnsupportedArchitecture;
        logger.info("Detected architecture: {s}", .{@tagName(arch)});

        var model = try registry.createModel(allocator, &gguf_file, arch, io);
        errdefer model.deinit(allocator);
        var params = model.getParams().*;

        // Copy model_name string from GGUF arena to heap (arena will be freed by gguf_file.deinit)
        if (params.model_name.len > 0) {
            params.model_name = try allocator.dupe(u8, params.model_name);
        }

        if (cli_args.verbose) {
            logger.info("n_vocab={d}, n_embd={d}, n_head={d}, n_kv_head={d}", .{ params.n_vocab, params.n_embd, params.n_head, params.n_kv_head });
            logger.info("n_layer={d}, n_ff={d}, n_head_dim={d}", .{ params.n_layer, params.n_ff, params.n_head_dim });
            logger.info("max_seq_len={d}, rope_theta={d}, rope_dim={d}", .{ params.max_seq_len, params.rope_theta, params.rope_dim });
        }


        // Detect and log model capabilities
        var capabilities = registry.detectCapabilities(&gguf_file, arch);
        if (capabilities.has_vision or capabilities.has_audio) {
            logger.info("Multi-modal: yes", .{});
        } else {
            logger.info("Multi-modal: no (text-only)", .{});
        }
        if (capabilities.has_vision) {
            logger.info("  Vision: yes ({s})", .{capabilities.vision_encoder_type});
        }
        if (capabilities.has_audio) {
            logger.info("  Audio : yes ({s}, {d} Hz)", .{ capabilities.audio_encoder_type, capabilities.audio_sample_rate });
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
        const hdim_kv = params.n_head_dim;
        const hdim_k = if (params.n_head_dim_k > 0) params.n_head_dim_k else hdim_kv;
        const hdim_v = if (params.n_head_dim_v > 0) params.n_head_dim_v else hdim_kv;
        var kv_cache_mgr = try kv_cache.KVCache.initWithKVDim(ctx_kv_cache, params.n_layer, params.n_kv_head, hdim_k, hdim_v, max_seq_len, allocator);
        errdefer kv_cache_mgr.deinit(allocator);
        model.setKVCacheContext(ctx_kv_cache);
        const ctx_graph = try ggml.Context.initNoAlloc(mem_size_estimate);
        // Create incremental decoding context for gallocr reuse
        const inc_ctx_size = 512 * 1024 * 1024; // 512MB
        const inc_ctx = try graph_context.IncContext.init(allocator, &params, inc_ctx_size);

        // Load multimodal encoder if mmproj file is provided
        var mm_manager: ?mm.MultiModalManager = null;
        if (cli_args.mmproj_path.len > 0) {
            mm_manager = try loadMMProj(io, allocator, cli_args.mmproj_path, &capabilities);
            logger.info("Multimodal encoder loaded from: {s}", .{cli_args.mmproj_path});
        } else if (capabilities.has_vision or capabilities.has_audio) {
            logger.warn("Model has multimodal capabilities but no --mmproj file provided", .{});
            logger.warn("  Use --mmproj <path> to load vision/audio encoder weights", .{});
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
            .benchmark = cli_args.benchmark,
            .inc_ctx = inc_ctx,
            .mm_manager = mm_manager,
            .capabilities = capabilities,
        };
    }

    pub fn deinit(self: *InferenceEngine) void {
        if (self.mm_manager) |*m| {
            m.deinit();
        }
        self.inc_ctx.deinit();
        self.ctx_graph.deinit();
        self.ctx_kv_cache.deinit();
        // ctx_weights freed by model.deinit()
        self.model.deinit(self.allocator);
        if (self.params.model_name.len > 0) {
            self.allocator.free(self.params.model_name);
        }
        self.allocator.free(self.gguf_data);
        self.tok.deinit();
        self.kv_cache_mgr.deinit(self.allocator);
    }

    pub fn generate(self: *InferenceEngine, io: std.Io, prompt: []const u8, max_tokens: u32) !void {
        // Encode prompt, adding special tokens (BOS/EOS)
        var input_tokens = try self.tok.encode(prompt, true);
        defer input_tokens.deinit(self.allocator);

        const n_prompt_tokens: i32 = @intCast(input_tokens.items.len);

        // --- Prompt processing (uses ctx_graph) ---
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_prompt_tokens);
        self.ctx_graph.setNoAlloc(true);

        var graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var builder = graph_builder.GraphBuilder.init(self.ctx_graph, graph, &self.params, self.allocator);
        const logits = try self.model.buildGraph(&builder, input_tensor, n_prompt_tokens, @ptrCast(&self.kv_cache_mgr), 0);

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

        // Prompt evaluation timing
        const t_pp_start = currentTimeMs();
        try graph.compute(self.n_threads);
        const first_token = sampler.Sampler.sampleGreedy(logits);
        const t_pp_end = currentTimeMs();
        const pp_time_s = @as(f64, @floatFromInt(t_pp_end - t_pp_start)) / 1000.0;

        // Stream prompt tokens (skip in benchmark mode)
        if (!self.benchmark) {
            for (input_tokens.items) |token_id| {
                var buf: [128]u8 = undefined;
                const n = try self.tok.decodeSingle(token_id, &buf);
                if (n > 0) {
                    const stdout_file = std.Io.File.stdout();
                    try stdout_file.writeStreamingAll(io, buf[0..n]);
                }
            }
        }

        var current_token: i32 = first_token;
        var pos: i32 = n_prompt_tokens;
        var gen_count: u32 = 0;

        // Text generation timing
        const t_tg_start = currentTimeMs();

        // --- Incremental decoding ---
        while (gen_count < max_tokens) {
            if (self.tok.isEog(@intCast(current_token))) break;

            // Stream output immediately (skip in benchmark mode)
            if (!self.benchmark) {
                var buf: [128]u8 = undefined;
                const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
                if (n > 0) {
                    const stdout_file = std.Io.File.stdout();
                    try stdout_file.writeStreamingAll(io, buf[0..n]);
                }
            }

            const step = try self.inc_ctx.beginStep();
            step.setToken(current_token);

            var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
            const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);

            if (!step.galloc.allocGraph(step.graph)) {
                logger.err("Failed to allocate incremental graph memory", .{});
                return error.GraphAllocFailed;
            }

            try step.graph.compute(self.n_threads);

            current_token = sampler.Sampler.sampleGreedy(inc_logits);
            pos += 1;
            gen_count += 1;
        }

        const t_tg_end = currentTimeMs();
        const tg_time_s = @as(f64, @floatFromInt(t_tg_end - t_tg_start)) / 1000.0;

        // Print newline (skip in benchmark mode)
        if (!self.benchmark) {
            const stdout_file = std.Io.File.stdout();
            try stdout_file.writeStreamingAll(io, "\n");
        }

        // Performance stats
        if (self.benchmark) {
            const total_time_s = pp_time_s + tg_time_s;
            const pp_speed = if (pp_time_s > 0.0)
                @as(f64, @floatFromInt(n_prompt_tokens)) / pp_time_s
            else
                0.0;
            const tg_speed = if (tg_time_s > 0.0 and gen_count > 0)
                @as(f64, @floatFromInt(gen_count)) / tg_time_s
            else
                0.0;
            const avg_speed = if (total_time_s > 0.0 and gen_count > 0)
                @as(f64, @floatFromInt(gen_count)) / total_time_s
            else
                0.0;

            std.debug.print(
                \\============ Benchmark Results ============
                \\  Model            : {s}
                \\  Architecture     : {s}
                \\  Threads          : {d}
                \\  Prompt tokens    : {d}
                \\  Output tokens    : {d}
                \\  ------------------------------------------
                \\  PP eval time     : {d:.3} s ({d:.1} tok/s)
                \\  TG time          : {d:.3} s ({d:.1} tok/s)
                \\  Total time       : {d:.3} s ({d:.1} tok/s)
                \\=============================================
                \\
            , .{
                if (self.params.model_name.len > 0) self.params.model_name else @tagName(self.arch),
                @tagName(self.arch),
                self.n_threads,
                n_prompt_tokens,
                gen_count,
                pp_time_s, pp_speed,
                tg_time_s, tg_speed,
                total_time_s, avg_speed,
            });
        } else if (gen_count > 0) {
            const total_time_s = pp_time_s + tg_time_s;
            const avg_speed = @as(f64, @floatFromInt(gen_count)) / total_time_s;
            logger.info("decoded {d} tokens in {d:.2} s, speed: {d:.2} t/s", .{ gen_count, total_time_s, avg_speed });
        }
    }

    /// 交互式聊天循环
    pub fn chatLoop(self: *InferenceEngine, io: std.Io) !void {
        const stdin = std.Io.File.stdin();
        const stdout = std.Io.File.stdout();

        try stdout.writeStreamingAll(io, "zllama chat - Interactive AI\n");
        try stdout.writeStreamingAll(io, "Type /help for commands.\n\n");

        var line_buf: [4096]u8 = undefined;
        var history = std.ArrayListUnmanaged([]const u8){ .items = &.{}, .capacity = 0 };
        defer {
            for (history.items) |item| self.allocator.free(item);
            history.deinit(self.allocator);
        }

        while (true) {
            try stdout.writeStreamingAll(io, ">>> ");

            var reader = stdin.reader(io, &line_buf);
            const line_slice = reader.interface.takeDelimiterExclusive('\n') catch |err| {
                if (err == error.EndOfStream) {
                    try stdout.writeStreamingAll(io, "\n");
                    break;
                }
                return err;
            };
            const line = std.mem.trimEnd(u8, line_slice, "\r");

            if (line.len == 0) continue;

            if (history.items.len == 0 or !std.mem.eql(u8, history.getLast(), line)) {
                const owned = try self.allocator.dupe(u8, line);
                try history.append(self.allocator, owned);
            }

            if (line[0] == '/') {
                if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) {
                    try stdout.writeStreamingAll(io, "Bye.\n");
                    break;
                } else if (std.mem.eql(u8, line, "/help")) {
                    try stdout.writeStreamingAll(io, "Available commands:\n");
                    try stdout.writeStreamingAll(io, "  /help     Show this help\n");
                    try stdout.writeStreamingAll(io, "  /clear    Clear screen\n");
                    try stdout.writeStreamingAll(io, "  /exit     Quit\n");
                    try stdout.writeStreamingAll(io, "  /reset    Reset KV cache\n");
                } else if (std.mem.eql(u8, line, "/clear")) {
                    try stdout.writeStreamingAll(io, "\x1b[2J\x1b[H");
                } else if (std.mem.eql(u8, line, "/reset")) {
                    self.kv_cache_mgr.reset();
                    try stdout.writeStreamingAll(io, "KV cache reset.\n");
                } else {
                    try stdout.writeStreamingAll(io, "Unknown command. Try /help.\n");
                }
                continue;
            }

            try stdout.writeStreamingAll(io, ">>> ");
            self.generate(io, line, 512) catch |err| {
                logger.err("Generation failed: {}\n", .{err});
            };
            try stdout.writeStreamingAll(io, "\n");
        }
    }

    /// Generate with image input (vision + text)
    ///
    /// Workflow:
    /// 1. Load and preprocess image (resize + normalize)
    /// 2. Run vision encoder to get image embeddings
    /// 3. Create prompt with vision tokens + text tokens
    /// 4. Run LLM inference
    pub fn generateWithImage(self: *InferenceEngine, io: std.Io, prompt: []const u8, image_path: [:0]const u8, max_tokens: u32) !void {
        var mm_mgr = self.mm_manager orelse return error.MMProjNotLoaded;
        if (!self.capabilities.has_vision) return error.VisionNotSupported;

        // Only Gemma4 supports vision currently
        if (self.arch != .gemma4) {
            logger.warn("Vision only supported for Gemma4 models. Using text-only generation.", .{});
            return self.generate(io, prompt, max_tokens);
        }
        const gemma4_model: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(self.model.ptr));

        // Step 1: Load and preprocess image (auto-detect format: PPM/JPEG/PNG/GIF/BMP/TGA)
        const target_size: u32 = 896;
        var img = try preprocess.loadImage(self.allocator, io, image_path, target_size, .auto);
        defer img.deinit();

        logger.info("Loaded image: {d}x{d} -> {d}x{d}", .{ img.width, img.height, target_size, target_size });

        // Step 2: Run vision encoder
        self.ctx_graph.setNoAlloc(false);
        var vision_graph = try ggml.CGraph.initReserved(self.ctx_graph, 32768);

        const vision_embeddings = try mm_mgr.encodeMedia(self.ctx_graph, vision_graph, .{
            .media_type = .image,
            .image_data = img.data,
            .image_width = img.width,
            .image_height = img.height,
        });
        self.ctx_graph.setNoAlloc(true);

        // Allocate and compute vision encoder graph
        const buft = ggml.backendCpuBufferType();
        {
            var v_galloc = try ggml.Gallocr.init(buft);
            defer v_galloc.free();
            if (!v_galloc.allocGraph(vision_graph)) {
                return error.GraphAllocFailed;
            }
            try vision_graph.compute(self.n_threads);
        }

        // Log vision encoding results
        const n_vision_tokens: i32 = @intCast(vision_embeddings.ne()[1]);
        const n_embd_val: usize = @intCast(vision_embeddings.ne()[0]);
        logger.info("Vision encoder output: [{d}, {d}] (n_embd x n_tokens)", .{ n_embd_val, n_vision_tokens });

        // Step 3: Tokenize prompt text
        var input_tokens = try self.tok.encode(prompt, true);
        defer input_tokens.deinit(self.allocator);
        const n_text_tokens: i32 = @intCast(input_tokens.items.len);
        const n_total_tokens: i32 = n_vision_tokens + n_text_tokens;

        // Step 4: Create input token tensor (vision positions = PAD, text positions = real tokens)
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_total_tokens);
        self.ctx_graph.setNoAlloc(true);
        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_total_tokens))];
            for (0..@as(usize, @intCast(n_vision_tokens))) |j| {
                slice[j] = 0;
            }
            for (input_tokens.items, 0..) |token, j| {
                slice[@as(usize, @intCast(n_vision_tokens)) + j] = @as(i32, @intCast(token));
            }
        }

        // Step 5: Build LLM graph with vision embedding override
        var graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        const start_pos: i32 = 0;
        const logits = try gemma4_model.forwardWithEmbdOverride(
            self.ctx_graph, graph, input_tensor, n_total_tokens,
            @ptrCast(&self.kv_cache_mgr), start_pos, vision_embeddings,
        );

        // Allocate and compute LLM graph
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();
        if (!galloc.allocGraph(graph)) {
            logger.err("Failed to allocate LLM graph for vision+text", .{});
            return error.GraphAllocFailed;
        }

        const t_start = currentTimeMs();
        try graph.compute(self.n_threads);
        const t_end = currentTimeMs();
        const pp_time_s = @as(f64, @floatFromInt(t_end - t_start)) / 1000.0;

        // Step 6: Sample first token and generate
        var current_token: i32 = sampler.Sampler.sampleGreedy(logits);
        var pos: i32 = n_total_tokens;
        var gen_count: u32 = 0;

        const t_tg_start = currentTimeMs();

        while (gen_count < max_tokens) {
            if (self.tok.isEog(@intCast(current_token))) break;

            if (!self.benchmark) {
                var buf: [128]u8 = undefined;
                const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
                if (n > 0) {
                    const stdout_file = std.Io.File.stdout();
                    try stdout_file.writeStreamingAll(io, buf[0..n]);
                }
            }

            const step = try self.inc_ctx.beginStep();
            step.setToken(current_token);

            var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
            const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);

            if (!step.galloc.allocGraph(step.graph)) {
                logger.err("Failed to allocate incremental graph memory", .{});
                return error.GraphAllocFailed;
            }
            try step.graph.compute(self.n_threads);

            current_token = sampler.Sampler.sampleGreedy(inc_logits);
            pos += 1;
            gen_count += 1;
        }

        const t_tg_end = currentTimeMs();
        const tg_time_s = @as(f64, @floatFromInt(t_tg_end - t_tg_start)) / 1000.0;

        if (!self.benchmark) {
            const stdout_file = std.Io.File.stdout();
            try stdout_file.writeStreamingAll(io, "\n");
        }

        if (gen_count > 0) {
            const total_time_s = pp_time_s + tg_time_s;
            const avg_speed = @as(f64, @floatFromInt(gen_count)) / total_time_s;
            logger.info("Vision generation: {d} tokens in {d:.2}s ({d:.1} t/s)", .{ gen_count, total_time_s, avg_speed });
        }
    }
    /// Workflow:
    /// 1. Load WAV audio file
    /// 2. Compute Mel spectrogram
    /// 3. Run Conformer audio encoder to get audio embeddings
    /// 4. Log dimensions and fall back to text-only (LLM token integration TODO)
    pub fn generateWithAudio(self: *InferenceEngine, io: std.Io, prompt: []const u8, audio_path: [:0]const u8, max_tokens: u32) !void {
        var mm_mgr = self.mm_manager orelse return error.MMProjNotLoaded;
        if (!self.capabilities.has_audio) return error.AudioNotSupported;

        // Only Gemma4 supports audio currently
        if (self.arch != .gemma4) {
            logger.warn("Audio only supported for Gemma4 models. Using text-only generation.", .{});
            return self.generate(io, prompt, max_tokens);
        }
        const gemma4_model: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(self.model.ptr));

        // Step 1: Load WAV file
        const wav_result = try preprocess.loadWav(self.allocator, io, audio_path);
        const wav_samples = wav_result.samples;
        const wav_info = wav_result.info;
        defer self.allocator.free(wav_samples);

        logger.info("Loaded audio: {d:.1}s, {d} Hz, {d} ch", .{
            @as(f32, @floatFromInt(wav_info.num_samples)) / @as(f32, @floatFromInt(wav_info.sample_rate)),
            wav_info.sample_rate,
            wav_info.num_channels,
        });

        // Step 2: Compute Mel spectrogram
        const n_mel_bins: u32 = preprocess.AUDIO_N_MEL_BINS;
        var mel = try preprocess.computeMelSpectrogram(self.allocator, wav_samples, wav_info.sample_rate, n_mel_bins);
        defer mel.deinit();

        logger.info("Mel spectrogram: {d} frames x {d} bins", .{ mel.n_frames, mel.n_mel_bins });

        // Step 3: Run Conformer audio encoder
        self.ctx_graph.setNoAlloc(false);
        var audio_graph = try ggml.CGraph.initReserved(self.ctx_graph, 32768);

        const audio_embeddings = try mm_mgr.encodeMedia(self.ctx_graph, audio_graph, .{
            .media_type = .audio,
            .mel_data = mel.data,
            .mel_bins = mel.n_mel_bins,
            .mel_frames = mel.n_frames,
            .audio_length_sec = @as(f32, @floatFromInt(wav_info.num_samples)) / @as(f32, @floatFromInt(wav_info.sample_rate)),
        });
        self.ctx_graph.setNoAlloc(true);

        // Allocate and compute audio encoder graph
        const buft = ggml.backendCpuBufferType();
        {
            var a_galloc = try ggml.Gallocr.init(buft);
            defer a_galloc.free();
            if (!a_galloc.allocGraph(audio_graph)) {
                return error.GraphAllocFailed;
            }
            try audio_graph.compute(self.n_threads);
        }

        // Log audio encoding results
        const n_audio_tokens: i32 = @intCast(audio_embeddings.ne()[1]);
        const n_embd_val: usize = @intCast(audio_embeddings.ne()[0]);
        logger.info("Audio encoder output: [{d}, {d}] (n_embd x n_tokens)", .{ n_embd_val, n_audio_tokens });

        // Step 4: Tokenize prompt text
        var input_tokens = try self.tok.encode(prompt, true);
        defer input_tokens.deinit(self.allocator);
        const n_text_tokens: i32 = @intCast(input_tokens.items.len);
        const n_total_tokens: i32 = n_audio_tokens + n_text_tokens;

        // Step 5: Create input token tensor (audio positions = PAD, text positions = real tokens)
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_total_tokens);
        self.ctx_graph.setNoAlloc(true);
        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_total_tokens))];
            for (0..@as(usize, @intCast(n_audio_tokens))) |j| {
                slice[j] = 0;
            }
            for (input_tokens.items, 0..) |token, j| {
                slice[@as(usize, @intCast(n_audio_tokens)) + j] = @as(i32, @intCast(token));
            }
        }

        // Step 6: Build LLM graph with audio embedding override
        var graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        const start_pos: i32 = 0;
        const logits = try gemma4_model.forwardWithEmbdOverride(
            self.ctx_graph, graph, input_tensor, n_total_tokens,
            @ptrCast(&self.kv_cache_mgr), start_pos, audio_embeddings,
        );

        // Allocate and compute LLM graph
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();
        if (!galloc.allocGraph(graph)) {
            logger.err("Failed to allocate LLM graph for audio+text", .{});
            return error.GraphAllocFailed;
        }

        const t_start = currentTimeMs();
        try graph.compute(self.n_threads);
        const t_end = currentTimeMs();
        const pp_time_s = @as(f64, @floatFromInt(t_end - t_start)) / 1000.0;

        // Step 7: Sample first token and generate
        var current_token: i32 = sampler.Sampler.sampleGreedy(logits);
        var pos: i32 = n_total_tokens;
        var gen_count: u32 = 0;

        const t_tg_start = currentTimeMs();

        while (gen_count < max_tokens) {
            if (self.tok.isEog(@intCast(current_token))) break;

            if (!self.benchmark) {
                var buf: [128]u8 = undefined;
                const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
                if (n > 0) {
                    const stdout_file = std.Io.File.stdout();
                    try stdout_file.writeStreamingAll(io, buf[0..n]);
                }
            }

            const step = try self.inc_ctx.beginStep();
            step.setToken(current_token);

            var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
            const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);

            if (!step.galloc.allocGraph(step.graph)) {
                logger.err("Failed to allocate incremental graph memory", .{});
                return error.GraphAllocFailed;
            }
            try step.graph.compute(self.n_threads);

            current_token = sampler.Sampler.sampleGreedy(inc_logits);
            pos += 1;
            gen_count += 1;
        }

        const t_tg_end = currentTimeMs();
        const tg_time_s = @as(f64, @floatFromInt(t_tg_end - t_tg_start)) / 1000.0;

        if (!self.benchmark) {
            const stdout_file = std.Io.File.stdout();
            try stdout_file.writeStreamingAll(io, "\n");
        }

        if (gen_count > 0) {
            const total_time_s = pp_time_s + tg_time_s;
            const avg_speed = @as(f64, @floatFromInt(gen_count)) / total_time_s;
            logger.info("Audio generation: {d} tokens in {d:.2}s ({d:.1} t/s)", .{ gen_count, total_time_s, avg_speed });
        }
    }
};

/// Load multimodal projector from separate GGUF file
/// Also detects audio/vision capabilities from the mmproj GGUF file.
fn loadMMProj(io: std.Io, allocator: std.mem.Allocator, mmproj_path: [:0]const u8, capabilities: *model_if.ModelCapabilities) !mm.MultiModalManager {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const file_size = @as(usize, @intCast(stat.size));
    const gguf_data = try allocator.alloc(u8, file_size);
    defer allocator.free(gguf_data);

    {
        var offset: u64 = 0;
        const chunk_size: usize = 64 * 1024 * 1024;
        while (offset < file_size) {
            const end = @min(offset + chunk_size, file_size);
            const len = end - offset;
            const bytes_read = try file.readPositionalAll(io, gguf_data[offset..][0..len], offset);
            if (bytes_read != len) {
                return error.FileReadError;
            }
            offset += bytes_read;
        }
    }

    var gguf_file = try gguf.parse(gguf_data, allocator);
    defer gguf_file.deinit();

    // Detect capabilities from mmproj file
    if (gguf_file.findTensor("a.conv1d.0.weight") != null or
        gguf_file.findTensor("a.pre_encode.out.weight") != null or
        gguf_file.findTensor("mm.a.input_projection.weight") != null)
    {
        capabilities.has_audio = true;
        if (capabilities.audio_encoder_type.len == 0) {
            capabilities.audio_encoder_type = "Conformer (E2B)";
        }
        if (capabilities.audio_sample_rate == 0) {
            capabilities.audio_sample_rate = 16000;
        }
    }
    if (gguf_file.findTensor("v.patch_embd.weight") != null or
        gguf_file.findTensor("mm.input_projection.weight") != null or
        gguf_file.findTensor("mm.soft_emb_norm.weight") != null)
    {
        capabilities.has_vision = true;
        if (capabilities.vision_encoder_type.len == 0) {
            capabilities.vision_encoder_type = "ViT (SigLIP/Gemma4V)";
        }
    }

    logger.info("MMProj capabilities: audio={}, vision={}", .{ capabilities.has_audio, capabilities.has_vision });

    const mem_size = 2 * 1024 * 1024 * 1024;
    const ctx = try ggml.Context.initNoAlloc(mem_size);
    errdefer ctx.deinit();

    const mgr = try mm.MultiModalManager.init(allocator, &gguf_file, ctx, capabilities.*);
    return mgr;
}
pub fn main(init: std.process.Init) !void {
    // Set ggml log callback
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

    if (args.chat) {
        try engine.chatLoop(io);
    } else if (args.image_path.len > 0) {
        logger.info("--- Vision Generation ---", .{});
        engine.generateWithImage(io, args.prompt, args.image_path, args.max_tokens) catch |err| {
            logger.err("Vision generation failed: {}", .{err});
            return;
        };
        logger.info("--- Done ---", .{});
    } else if (args.audio_path.len > 0) {
        logger.info("--- Audio Generation ---", .{});
        engine.generateWithAudio(io, args.prompt, args.audio_path, args.max_tokens) catch |err| {
            logger.err("Audio generation failed: {}", .{err});
            return;
        };
        logger.info("--- Done ---", .{});
    } else {
        logger.info("--- Generation ---", .{});
        engine.generate(io, args.prompt, args.max_tokens) catch |err| {
            logger.err("Generation failed: {}", .{err});
            return;
        };
        logger.info("--- Done ---", .{});
    }
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
