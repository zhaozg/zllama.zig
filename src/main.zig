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
const engine_common = @import("engine_common");
const chat_template = @import("chat_template");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = engine_common.logFilter,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .tokenizer, .level = .info},
        .{ .scope = .ggml, .level = .info },
        .{ .scope = .qwen, .level = .info },
        .{ .scope = .llama, .level = .info },
        .{ .scope = .model, .level = .info },
        .{ .scope = .main, .level = .info },
    }
};

const logger = std.log.scoped(.main);

/// Tokenize a text segment for multimodal placeholder expansion.
/// Used as callback for tokenizeWithPlaceholders.
fn tokenizeTextSegment(ctx: ?*anyopaque, text: []const u8, alloc: std.mem.Allocator) ![]u32 {
    const tok: *tokenizer.Tokenizer = @ptrCast(@alignCast(ctx orelse return error.NullCtx));
    var result = try tok.encode(text, false);
    defer result.deinit(alloc);
    return try result.toOwnedSlice(alloc);
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
    // Embedding
    embed: bool = false,
    pooling: []const u8 = "mean",
    embed_normalize: bool = true,
    // Multimodal
    mmproj_path: [:0]const u8 = "",
    image_path: [:0]const u8 = "",
    audio_path: [:0]const u8 = "",
    // Chat template
    chat_template_name: []const u8 = "",
    system_prompt: []const u8 = "",
    no_chat_template: bool = false,
    no_jinja: bool = false,
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
            } else if (std.mem.eql(u8, arg, "--chat") or std.mem.eql(u8, arg, "-c")) {
                result.chat = true;
            } else if (std.mem.eql(u8, arg, "--mmproj")) {
                result.mmproj_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--image")) {
                result.image_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--audio")) {
                result.audio_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--embed")) {
                result.embed = true;
            } else if (std.mem.eql(u8, arg, "--pooling")) {
                result.pooling = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--embd-normalize")) {
                const val = args_it.next() orelse return error.InvalidArgs;
                result.embed_normalize = std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
            } else if (std.mem.eql(u8, arg, "--chat-template")) {
                result.chat_template_name = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--system-prompt")) {
            } else if (std.mem.eql(u8, arg, "--no-chat-template")) {
                result.no_chat_template = true;
            } else if (std.mem.eql(u8, arg, "--no-jinja")) {
                result.no_jinja = true;
                result.no_chat_template = true;
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
            \\  -v, --verbose         详细日志输出 (info 级别)
            \\  -d, --debug           调试日志输出 (debug 级别)
            \\  --benchmark           benchmark 模式
            \\  -c, --chat            交互式聊天模式
            \\
            \\对话模板选项:
            \\  --chat-template <名称> 指定对话模板 (chatml | llama3 | gemma)
            \\  --system-prompt <文本> 指定系统提示词
            \\  --no-chat-template     禁用对话模板，原始 prompt 透传
            \\
            \\嵌入模式选项:
            \\  --embed               启用嵌入向量生成模式
            \\  --pooling <策略>      池化策略: mean | cls | last (默认: mean)
            \\  --embd-normalize 1    是否 L2 归一化 (默认: 1/true)
            \\
            \\多模态选项:
            \\  --mmproj <路径>       多模态投影器文件 (GGUF格式, mmproj)
            \\  --image <路径>        输入图像文件 (PPM/JPEG/PNG/BMP/GIF)
            \\  --audio <路径>        输入音频文件 (WAV 16-bit PCM)
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

    // Chat template
    chat_template_source: ?chat_template.TemplateSource = null,
    system_prompt: []const u8 = "",
    no_chat_template: bool = false,
    no_jinja: bool = false,

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


        // Detect and log model capabilities (from main GGUF metadata only)
        // Full capabilities are determined after mmproj loading below
        var capabilities = registry.detectCapabilities(&gguf_file, arch);
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
        const hdim_k = @max(params.n_head_dim, params.n_head_dim_k);
        const hdim_v = if (params.n_head_dim_v > 0) @max(params.n_head_dim, params.n_head_dim_v) else hdim_kv;
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
            // Re-check capabilities after mmproj load (updates has_audio/has_vision)
            if (capabilities.has_vision or capabilities.has_audio) {
                logger.info("Multi-modal: yes", .{});
                if (capabilities.has_vision) {
                    logger.info("  Vision: yes ({s})", .{capabilities.vision_encoder_type});
                }
                if (capabilities.has_audio) {
                    logger.info("  Audio : yes ({s}, {d} Hz)", .{ capabilities.audio_encoder_type, capabilities.audio_sample_rate });
                }
            }
        } else if (capabilities.has_vision or capabilities.has_audio) {
            logger.warn("Model has multimodal capabilities but no --mmproj file provided", .{});
            logger.warn("  Use --mmproj <path> to load vision/audio encoder weights", .{});
        }

        // Resolve chat template source
        // Priority: --chat-template > --no-chat-template > GGUF built-in > arch default
        var chat_template_source: ?chat_template.TemplateSource = null;
        var system_prompt: []const u8 = "";
        if (cli_args.system_prompt.len > 0) {
            system_prompt = try allocator.dupe(u8, cli_args.system_prompt);
        }
        if (cli_args.no_chat_template) {
            // Disabled - no template
            chat_template_source = null;
        } else if (cli_args.chat_template_name.len > 0) {
            // User-specified preset
            if (chat_template.TemplateKind.fromString(cli_args.chat_template_name)) |kind| {
                chat_template_source = chat_template.TemplateSource{ .preset = kind };
                logger.info("Chat template: {s} (from --chat-template)", .{cli_args.chat_template_name});
            } else {
                logger.warn("Unknown chat template '{s}', falling back to arch default", .{cli_args.chat_template_name});
            }
        } else if (gguf_file.getString("tokenizer.chat_template")) |tmpl_str| {
            // GGUF built-in template
            const owned = try allocator.dupe(u8, tmpl_str);
            chat_template_source = chat_template.TemplateSource{ .gguf_builtin = owned };
            logger.info("Chat template: from GGUF metadata", .{});
        }
        // If no source resolved, will use arch default in generate()


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
            .chat_template_source = chat_template_source,
            .system_prompt = system_prompt,
            .no_chat_template = cli_args.no_chat_template,
            .no_jinja = cli_args.no_jinja,
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
        // Free chat template source (gguf_builtin variant owns allocated string)
        if (self.chat_template_source) |src| {
            switch (src) {
                .gguf_builtin => |s| self.allocator.free(s),
                .custom => |s| self.allocator.free(s),
                .preset => {},
            }
        }
        // Free system prompt if allocated
        if (self.system_prompt.len > 0) {
            self.allocator.free(self.system_prompt);
        }
    }

    /// Apply chat template to the user prompt.
    /// Returns the formatted prompt (caller owns memory).
    fn applyChatTemplate(self: *InferenceEngine, user_prompt: []const u8) ![]const u8 {
        return self.applyChatTemplateWithMedia(user_prompt, null);
    }

    /// 应用聊天模板，支持可选的媒体数据
    fn applyChatTemplateWithMedia(self: *InferenceEngine, user_prompt: []const u8, media: ?chat_template.Media) ![]const u8 {
        if (self.no_chat_template) {
            return self.allocator.dupe(u8, user_prompt);
        }

        const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
        const source = self.chat_template_source orelse
            chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };

        var tmpl = try chat_template.resolve(self.allocator, source, self.arch, model_name, !self.no_jinja);
        defer tmpl.deinit(self.allocator);

        const messages = if (media) |m| [_]chat_template.ChatMessage{
            chat_template.ChatMessage.withMedia("user", user_prompt, m),
        } else [_]chat_template.ChatMessage{
            chat_template.ChatMessage.init("user", user_prompt),
        };

        const system = if (self.system_prompt.len > 0) self.system_prompt else null;
        return tmpl.apply(self.allocator, &messages, system, true);
    }

    pub fn generate(self: *InferenceEngine, io: std.Io, prompt: []const u8, max_tokens: u32) !void {
        // Apply chat template
        const formatted_prompt = try self.applyChatTemplate(prompt);
        defer self.allocator.free(formatted_prompt);

        // Encode prompt, adding special tokens (BOS/EOS)
        var input_tokens = try self.tok.encode(formatted_prompt, true);
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
        const t_pp_start = engine_common.currentTimeMs();
        try graph.compute(self.n_threads);
        const first_token = sampler.Sampler.sampleGreedy(logits);
        const t_pp_end = engine_common.currentTimeMs();
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
        const t_tg_start = engine_common.currentTimeMs();

        // Buffer for EOG text detection (accumulates decoded output)
        var eog_detect_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
        defer eog_detect_buf.deinit(self.allocator);

        // --- Incremental decoding ---
        while (gen_count < max_tokens) {
            if (self.tok.isEog(@intCast(current_token))) break;


            // Skip control tokens (e.g. <|channel|>, <|channel>) that should
            // be filtered from output but don't stop generation.
            if (self.tok.isSkipToken(@intCast(current_token))) {
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
                continue;
            }

            // Decode token
            var buf: [128]u8 = undefined;
            const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
            const decoded = buf[0..n];

            // Check if decoded text contains EOG token string
            // (handles models that generate EOG token as sub-token sequence)
            if (n > 0) {
                try eog_detect_buf.appendSlice(self.allocator, decoded);
                if (self.tok.isEogText(eog_detect_buf.items)) {
                    // Stream the current token before stopping
                    if (!self.benchmark) {
                        const stdout_file = std.Io.File.stdout();
                        try stdout_file.writeStreamingAll(io, decoded);
                    }
                    break;
                }
            }

            // Stream output immediately (skip in benchmark mode)
            if (!self.benchmark and n > 0) {
                const stdout_file = std.Io.File.stdout();
                try stdout_file.writeStreamingAll(io, decoded);
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

        const t_tg_end = engine_common.currentTimeMs();
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

    /// 嵌入向量生成
    /// 将输入文本编码为固定维度向量
    pub fn generateEmbedding(self: *InferenceEngine, io: std.Io, prompt: []const u8) ![]f32 {
        // Tokenize
        var input_tokens = try self.tok.encode(prompt, true);
        defer input_tokens.deinit(self.allocator);

        const n_tokens: i32 = @intCast(input_tokens.items.len);
        if (n_tokens == 0) {
            return error.EmptyInput;
        }

        // Build input tensor
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_tokens);
        self.ctx_graph.setNoAlloc(true);

        // Fill token data
        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
            for (input_tokens.items, 0..) |token, j| {
                slice[j] = @as(i32, @intCast(token));
            }
        }

        // Build compute graph (bidirectional attention, no KV cache, pooling output)
        var graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var builder = graph_builder.GraphBuilder.init(self.ctx_graph, graph, &self.params, self.allocator);

        // Embedding models ignore KV cache (passed as null context)
        const embedding_vector = try self.model.buildGraph(&builder, input_tensor, n_tokens, null, 0);

        // Allocate & compute
        const buft = ggml.backendCpuBufferType();
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();
        if (!galloc.allocGraph(graph)) {
            return error.GraphAllocFailed;
        }
        try graph.compute(self.n_threads);

        // Extract result
        const result_data = embedding_vector.dataF32();
        const n_embd = @as(usize, @intCast(self.params.n_embd));
        const result = try self.allocator.alloc(f32, n_embd);
        @memcpy(result, result_data[0..n_embd]);

        // Print embedding vector
        const stdout_file = std.Io.File.stdout();
        var buf: [128]u8 = undefined;
        for (result) |v| {
            const line = try std.fmt.bufPrint(&buf, "{d:.6}\n", .{v});
            try stdout_file.writeStreamingAll(io, line);
        }

        return result;
    }

    /// Interactive chat loop with multi-turn context
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

        var chat_history = std.ArrayListUnmanaged(chat_template.ChatMessage){ .items = &.{}, .capacity = 0 };
        defer chat_history.deinit(self.allocator);

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
                    try stdout.writeStreamingAll(io, "  /new      Start new conversation (clear history)\n");
                    try stdout.writeStreamingAll(io, "  /image <path> <prompt>  Attach image and prompt\n");
                    try stdout.writeStreamingAll(io, "  /audio <path> <prompt>  Attach audio and prompt\n");
                } else if (std.mem.eql(u8, line, "/clear")) {
                    try stdout.writeStreamingAll(io, "\x1b[2J\x1b[H");
                } else if (std.mem.eql(u8, line, "/reset")) {
                    self.kv_cache_mgr.reset();
                    try stdout.writeStreamingAll(io, "KV cache reset.\n");
                } else if (std.mem.eql(u8, line, "/new")) {
                    chat_history.clearAndFree(self.allocator);
                    self.kv_cache_mgr.reset();
                    try stdout.writeStreamingAll(io, "New conversation started.\n");
                } else if (std.mem.startsWith(u8, line, "/image ")) {
                    // /image <path> <prompt>
                    const rest = line[7..]; // skip "/image "
                    const space_pos = std.mem.indexOf(u8, rest, " ") orelse {
                        try stdout.writeStreamingAll(io, "Usage: /image <path> <prompt>\n");
                        continue;
                    };
                    const img_path = rest[0..space_pos];
                    const img_prompt = rest[space_pos + 1 ..];
                    try stdout.writeStreamingAll(io, "Processing image...\n");
                    self.generateWithImage(io, img_prompt, @ptrCast(@constCast(img_path)), 256) catch {
                        try stdout.writeStreamingAll(io, "Image processing failed.\n");
                    };
                    try stdout.writeStreamingAll(io, "\n");
                    continue;
                } else if (std.mem.startsWith(u8, line, "/audio ")) {
                    // /audio <path> <prompt>
                    const rest = line[7..]; // skip "/audio "
                    const space_pos = std.mem.indexOf(u8, rest, " ") orelse {
                        try stdout.writeStreamingAll(io, "Usage: /audio <path> <prompt>\n");
                        continue;
                    };
                    const aud_path = rest[0..space_pos];
                    const aud_prompt = rest[space_pos + 1 ..];
                    try stdout.writeStreamingAll(io, "Processing audio...\n");
                    self.generateWithAudio(io, aud_prompt, @ptrCast(@constCast(aud_path)), 256) catch {
                        try stdout.writeStreamingAll(io, "Audio processing failed.\n");
                    };
                    try stdout.writeStreamingAll(io, "\n");
                    continue;
                } else {
                    try stdout.writeStreamingAll(io, "Unknown command. Try /help.\n");
                }
                continue;
            }

            // Add user message to chat history
            try chat_history.append(self.allocator, chat_template.ChatMessage.init("user", line));

            // Format with multi-turn template (or raw prompt if --no-chat-template)
            const formatted_prompt = if (self.no_chat_template) blk: {
                break :blk try self.allocator.dupe(u8, line);
            } else blk: {
                const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
                const source = self.chat_template_source orelse
                    chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };
                var tmpl = try chat_template.resolve(self.allocator, source, self.arch, model_name, !self.no_jinja);
                defer tmpl.deinit(self.allocator);
                const system = if (self.system_prompt.len > 0) self.system_prompt else null;
                break :blk try tmpl.apply(self.allocator, chat_history.items, system, true);
            };
            defer self.allocator.free(formatted_prompt);

            // Reset KV cache for each turn (full prompt processing)
            self.kv_cache_mgr.reset();

            // Encode and generate
            var input_tokens = try self.tok.encode(formatted_prompt, true);
            defer input_tokens.deinit(self.allocator);

            const n_prompt_tokens: i32 = @intCast(input_tokens.items.len);

            // --- Prompt processing ---
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

            try graph.compute(self.n_threads);
            var current_token = sampler.Sampler.sampleGreedy(logits);
            var pos: i32 = n_prompt_tokens;
            var gen_count: u32 = 0;

            // Collect generated tokens for history
            var output_tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
            defer output_tokens.deinit(self.allocator);

            // --- Incremental decoding ---
            while (gen_count < 512) {
                if (self.tok.isEog(@intCast(current_token))) break;

                // Stream output
                var buf: [128]u8 = undefined;
                const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
                if (n > 0) {
                    try stdout.writeStreamingAll(io, buf[0..n]);
                }

                try output_tokens.append(self.allocator, @intCast(current_token));

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

            try stdout.writeStreamingAll(io, "\n");

            // Decode assistant response and add to chat history
            if (output_tokens.items.len > 0) {
                var response_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
                defer response_buf.deinit(self.allocator);
                for (output_tokens.items) |token_id| {
                    var buf: [128]u8 = undefined;
                    const n = try self.tok.decodeSingle(token_id, &buf);
                    if (n > 0) {
                        try response_buf.appendSlice(self.allocator, buf[0..n]);
                    }
                }
                try chat_history.append(self.allocator, chat_template.ChatMessage.init("assistant", response_buf.items));
            }
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

        if (self.arch != .gemma4) {
            logger.warn("Vision only supported for Gemma4 models. Using text-only generation.", .{});
            return self.generate(io, prompt, max_tokens);
        }
        const gemma4_model: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(self.model.ptr));

        const target_size: u32 = 896;
        var img = try preprocess.loadImage(self.allocator, io, image_path, target_size, .auto);
        defer img.deinit();
        logger.info("Loaded image: {d}x{d} -> {d}x{d}", .{ img.width, img.height, target_size, target_size });

        self.ctx_graph.setNoAlloc(false);
        var vision_graph = try ggml.CGraph.initReserved(self.ctx_graph, 32768);

        const vision_embeddings = try mm_mgr.encodeMedia(self.ctx_graph, vision_graph, .{
            .media_type = .image,
            .image_data = img.data,
            .image_width = img.width,
            .image_height = img.height,
        });
        self.ctx_graph.setNoAlloc(true);

        const buft = ggml.backendCpuBufferType();
        var v_galloc = try ggml.Gallocr.init(buft);
        defer v_galloc.free();
        if (!v_galloc.allocGraph(vision_graph)) {
            return error.GraphAllocFailed;
        }
        try vision_graph.compute(self.n_threads);

        const n_vision_tokens: i32 = @intCast(vision_embeddings.ne()[1]);
        const n_embd_val: usize = @intCast(vision_embeddings.ne()[0]);
        logger.info("Vision encoder output: [{d}, {d}] (n_embd x n_tokens)", .{ n_embd_val, n_vision_tokens });

        const model_n_embd: usize = @intCast(self.params.n_embd);
        if (n_embd_val != model_n_embd) {
            logger.err("Vision encoder output dim {d} != model n_embd {d}!", .{ n_embd_val, model_n_embd });
            return error.EmbeddingDimensionMismatch;
        }
        logger.info("Vision embedding dimension check: {d} == model n_embd {d} ✓", .{ n_embd_val, model_n_embd });

        const expected_tokens: i32 = @intCast(@divTrunc(
            (@as(u32, @intCast(@divTrunc(target_size, 14))) * @as(u32, @intCast(@divTrunc(target_size, 14)))),
            4,
        ));
        if (n_vision_tokens != expected_tokens) {
            logger.info("Vision tokens={d} (expected ~{d} for {d}x{d}); mmproj may use different pooling", .{ n_vision_tokens, expected_tokens, target_size, target_size });
        } else {
            logger.info("Vision token count: {d} == expected {d} ✓", .{ n_vision_tokens, expected_tokens });
        }

        // 使用多模态 ChatMessage API
        const image_token_id: u32 = blk: {
            if (self.tok.textToToken("<|image|>")) |id| {
                logger.info("Image placeholder token '<|image|>' -> id={d}", .{id});
                break :blk @as(u32, @intCast(id));
            }
            if (self.tok.textToToken("<image>")) |id| {
                logger.info("Image placeholder token '<image>' -> id={d}", .{id});
                break :blk @as(u32, @intCast(id));
            }
            logger.err("No <|image|> or <image> token in vocabulary!", .{});
            return error.NoImagePlaceholderToken;
        };

        // 使用 ensurePlaceholderInContent 自动插入占位符
        const content_with_placeholder = try chat_template.ensurePlaceholderInContent(prompt, .image, self.allocator);
        defer if (content_with_placeholder.ptr != prompt.ptr) self.allocator.free(content_with_placeholder);

        const formatted_prompt = if (self.no_chat_template) blk: {
            break :blk try self.allocator.dupe(u8, content_with_placeholder);
        } else blk: {
            const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
            const source = self.chat_template_source orelse
                chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };
            var tmpl = try chat_template.resolve(self.allocator, source, self.arch, model_name, !self.no_jinja);
            defer tmpl.deinit(self.allocator);
            const system = if (self.system_prompt.len > 0) self.system_prompt else null;
            const messages = [_]chat_template.ChatMessage{
                chat_template.ChatMessage.init("user", content_with_placeholder),
            };
            break :blk try tmpl.apply(self.allocator, &messages, system, true);
        };
        defer self.allocator.free(formatted_prompt);
        logger.info("Formatted prompt ({d} chars):\n{s}", .{ formatted_prompt.len, formatted_prompt });

        // 使用 tokenizeWithPlaceholders 分段 tokenize + 展开占位符
        var expanded = try chat_template.tokenizeWithPlaceholders(
            self.allocator,
            formatted_prompt,
            @ptrCast(&self.tok),
            tokenizeTextSegment,
            image_token_id,
            0, // audio_token_id (unused for vision)
            @intCast(n_vision_tokens),
            0, // audio_token_count (unused for vision)
        );
        defer expanded.deinit();
        // Determine text prefix and suffix around image placeholder
        // expanded.offsets[0].start is the string position of the first <|image|>
        const first_ph = expanded.offsets[0];
        const text_before = formatted_prompt[0..first_ph.start];
        const text_after_start = first_ph.start + first_ph.length;
        const text_after = if (text_after_start < formatted_prompt.len)
            formatted_prompt[text_after_start..]
        else
            "";

        var prefix_tokens_raw = try self.tok.encode(text_before, true);
        defer prefix_tokens_raw.deinit(self.allocator);
        var suffix_tokens_raw = try self.tok.encode(text_after, true);
        defer suffix_tokens_raw.deinit(self.allocator);

        const prefix_len: i32 = @intCast(prefix_tokens_raw.items.len);
        const suffix_len: i32 = @intCast(suffix_tokens_raw.items.len);

        logger.info("Three-pass prefill: prefix={d}, image={d}, suffix={d}", .{
            prefix_len, n_vision_tokens, suffix_len,
        });

        const kv_cache_ptr: ?*kv_cache.KVCache = @ptrCast(&self.kv_cache_mgr);
        var pp_time_s: f64 = 0.0;

        // Pass 1: Text prefix only (causal attention)
        if (prefix_len > 0) {
            self.ctx_graph.reset();
            self.ctx_graph.setNoAlloc(false);
            const p1_input = try self.ctx_graph.newTensor1d(.i32, prefix_len);
            self.ctx_graph.setNoAlloc(true);
            {
                const data = p1_input.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(prefix_len))];
                for (prefix_tokens_raw.items, 0..) |t, j| {
                    slice[j] = @as(i32, @intCast(t));
                }
            }

            var p1_graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
            var p1_builder = graph_builder.GraphBuilder.init(self.ctx_graph, p1_graph, &self.params, self.allocator);
            _ = try self.model.buildGraph(&p1_builder, p1_input, prefix_len, kv_cache_ptr, 0);

            var p1_galloc = try ggml.Gallocr.init(buft);
            defer p1_galloc.free();
            if (!p1_galloc.allocGraph(p1_graph)) {
                logger.err("Graph alloc failed: text-prefix pass", .{});
                return error.GraphAllocFailed;
            }
            try p1_graph.compute(self.n_threads);
            logger.debug("Pass 1 (text prefix): {d} tokens ✓", .{prefix_len});
        }

        // Pass 2: Image tokens only (non-causal attention)
        {
            self.ctx_graph.reset();
            self.ctx_graph.setNoAlloc(false);
            const p2_input = try self.ctx_graph.newTensor1d(.i32, n_vision_tokens);
            self.ctx_graph.setNoAlloc(true);
            {
                const data = p2_input.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_vision_tokens))];
                @memset(slice, @as(i32, @intCast(image_token_id)));
            }

            var p2_graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
            _ = try gemma4_model.forwardWithEmbdOverride(
                self.ctx_graph, p2_graph, p2_input, n_vision_tokens,
                kv_cache_ptr, prefix_len, vision_embeddings, 0, false,
            );

            var p2_galloc = try ggml.Gallocr.init(buft);
            defer p2_galloc.free();
            if (!p2_galloc.allocGraph(p2_graph)) {
                logger.err("Graph alloc failed: image pass", .{});
                return error.GraphAllocFailed;
            }
            const t2_start = engine_common.currentTimeMs();
            try p2_graph.compute(self.n_threads);
            const t2_end = engine_common.currentTimeMs();
            pp_time_s += @as(f64, @floatFromInt(t2_end - t2_start)) / 1000.0;
            logger.debug("Pass 2 (image): {d} tokens (non-causal) ✓", .{n_vision_tokens});
        }

        // Pass 3: Text suffix only (causal attention) — sample logits from here
        self.ctx_graph.reset();
        self.ctx_graph.setNoAlloc(false);
        const sfx_n: i32 = if (suffix_len > 0) suffix_len else 1;
        const p3_input = try self.ctx_graph.newTensor1d(.i32, sfx_n);
        self.ctx_graph.setNoAlloc(true);
        {
            const data = p3_input.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(sfx_n))];
            if (suffix_len > 0) {
                for (suffix_tokens_raw.items, 0..) |t, j| {
                    slice[j] = @as(i32, @intCast(t));
                }
            } else {
                // No real suffix: use a single dummy token (its KV entry is harmless
                // since we immediately sample and enter incremental decode).
                slice[0] = @as(i32, @intCast(image_token_id));
            }
        }

        const suffix_start_pos: i32 = prefix_len + n_vision_tokens;
        var p3_graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var p3_builder = graph_builder.GraphBuilder.init(self.ctx_graph, p3_graph, &self.params, self.allocator);
        const logits = try self.model.buildGraph(&p3_builder, p3_input, sfx_n, kv_cache_ptr, suffix_start_pos);

        var p3_galloc = try ggml.Gallocr.init(buft);
        defer p3_galloc.free();
        if (!p3_galloc.allocGraph(p3_graph)) {
            logger.err("Graph alloc failed: text-suffix pass", .{});
            return error.GraphAllocFailed;
        }

        const t3_start = engine_common.currentTimeMs();
        try p3_graph.compute(self.n_threads);
        const t3_end = engine_common.currentTimeMs();
        pp_time_s += @as(f64, @floatFromInt(t3_end - t3_start)) / 1000.0;
        logger.debug("Pass 3 (text suffix): {d} tokens ✓", .{sfx_n});

        // Step 7: Sample and generate
        var current_token: i32 = sampler.Sampler.sampleGreedy(logits);
        var pos: i32 = suffix_start_pos + (if (suffix_len > 0) suffix_len else 1);
        var gen_count: u32 = 0;
        logger.info("First generated token (vision): id={d}, is_eog={}, pp_time={d:.3}s", .{
            current_token,
            self.tok.isEog(@intCast(current_token)),
            pp_time_s,
        });


        const t_tg_start = engine_common.currentTimeMs();

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

        const t_tg_end = engine_common.currentTimeMs();
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
    /// 4. Format prompt with chat template and inject audio placeholder tokens
    /// 5. Run LLM with mixed embeddings (audio + text)
    pub fn generateWithAudio(self: *InferenceEngine, io: std.Io, prompt: []const u8, audio_path: [:0]const u8, max_tokens: u32) !void {
        var mm_mgr = self.mm_manager orelse return error.MMProjNotLoaded;
        if (!self.capabilities.has_audio) return error.AudioNotSupported;

        if (self.arch != .gemma4) {
            logger.warn("Audio only supported for Gemma4 models. Using text-only generation.", .{});
            return self.generate(io, prompt, max_tokens);
        }
        const gemma4_model: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(self.model.ptr));

        const wav_result = try preprocess.loadWav(self.allocator, io, audio_path);
        const wav_samples = wav_result.samples;
        const wav_info = wav_result.info;
        defer self.allocator.free(wav_samples);

        logger.info("Loaded audio: {d:.1}s, {d} Hz, {d} ch", .{
            @as(f32, @floatFromInt(wav_info.num_samples)) / @as(f32, @floatFromInt(wav_info.sample_rate)),
            wav_info.sample_rate,
            wav_info.num_channels,
        });

        const preprocess_params = preprocess.AudioPreprocessParams.fromAudioEncoder(
            if (mm_mgr.audio_encoder) |enc| enc.params.n_mel_bins else preprocess.AUDIO_N_MEL_BINS,
        );
        var mel = try preprocess.computeMelSpectrogram(self.allocator, wav_samples, wav_info.sample_rate, preprocess_params);
        defer mel.deinit();
        logger.info("Mel spectrogram: {d} frames x {d} bins", .{ mel.n_frames, mel.n_mel_bins });

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

        const buft = ggml.backendCpuBufferType();
        var a_galloc = try ggml.Gallocr.init(buft);
        defer a_galloc.free();
        if (!a_galloc.allocGraph(audio_graph)) {
            return error.GraphAllocFailed;
        }
        try audio_graph.compute(self.n_threads);

        const n_audio_tokens: i32 = @intCast(audio_embeddings.ne()[1]);
        const n_embd_val: usize = @intCast(audio_embeddings.ne()[0]);
        logger.info("Audio encoder output: [{d}, {d}] (n_embd x n_tokens)", .{ n_embd_val, n_audio_tokens });

        const model_n_embd: usize = @intCast(self.params.n_embd);
        if (n_embd_val != model_n_embd) {
            logger.err("Audio encoder output dim {d} != model n_embd {d}!", .{ n_embd_val, model_n_embd });
            return error.EmbeddingDimensionMismatch;
        }
        logger.info("Audio embedding dimension check: {d} == model n_embd {d} ✓", .{ n_embd_val, model_n_embd });

        // 使用多模态 ChatMessage API
        const audio_token_id: u32 = blk: {
            if (self.tok.textToToken("<|audio|>")) |id| {
                logger.info("Audio placeholder token '<|audio|>' -> id={d}", .{id});
                break :blk @as(u32, @intCast(id));
            }
            if (self.tok.textToToken("<audio>")) |id| {
                logger.info("Audio placeholder token '<audio>' -> id={d}", .{id});
                break :blk @as(u32, @intCast(id));
            }
            logger.err("No <|audio|> or <audio> token in vocabulary!", .{});
            return error.NoAudioPlaceholderToken;
        };

        // 使用 ensurePlaceholderInContent 自动插入占位符
        const content_with_placeholder = try chat_template.ensurePlaceholderInContent(prompt, .audio, self.allocator);
        defer if (content_with_placeholder.ptr != prompt.ptr) self.allocator.free(content_with_placeholder);

        const formatted_prompt = if (self.no_chat_template) blk: {
            break :blk try self.allocator.dupe(u8, content_with_placeholder);
        } else blk: {
            const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
            const source = self.chat_template_source orelse
                chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };
            var tmpl = try chat_template.resolve(self.allocator, source, self.arch, model_name, !self.no_jinja);
            defer tmpl.deinit(self.allocator);
            const system = if (self.system_prompt.len > 0) self.system_prompt else null;
            const messages = [_]chat_template.ChatMessage{
                chat_template.ChatMessage.init("user", content_with_placeholder),
            };
            break :blk try tmpl.apply(self.allocator, &messages, system, true);
        };
        defer self.allocator.free(formatted_prompt);

        logger.info("Formatted prompt ({d} chars):\n{s}", .{ formatted_prompt.len, formatted_prompt });
        // 使用 tokenizeWithPlaceholders 分段 tokenize + 展开占位符
        var expanded = try chat_template.tokenizeWithPlaceholders(
            self.allocator,
            formatted_prompt,
            @ptrCast(&self.tok),
            tokenizeTextSegment,
            0, // image_token_id (unused for audio)
            audio_token_id,
            0, // image_token_count (unused for audio)
            @intCast(n_audio_tokens),
        );
        defer expanded.deinit();

        // Determine text prefix and suffix around audio placeholder
        const first_ph = expanded.offsets[0];
        const text_before = formatted_prompt[0..first_ph.start];
        const text_after_start = first_ph.start + first_ph.length;
        const text_after = if (text_after_start < formatted_prompt.len)
            formatted_prompt[text_after_start..]
        else
            "";

        var prefix_tokens_raw = try self.tok.encode(text_before, true);
        defer prefix_tokens_raw.deinit(self.allocator);
        var suffix_tokens_raw = try self.tok.encode(text_after, true);
        defer suffix_tokens_raw.deinit(self.allocator);

        const prefix_len: i32 = @intCast(prefix_tokens_raw.items.len);
        const suffix_len: i32 = @intCast(suffix_tokens_raw.items.len);

        logger.info("Three-pass prefill (audio): prefix={d}, audio={d}, suffix={d}", .{
            prefix_len, n_audio_tokens, suffix_len,
        });

        const kv_cache_ptr: ?*kv_cache.KVCache = @ptrCast(&self.kv_cache_mgr);
        var pp_time_s: f64 = 0.0;

        // Pass 1: Text prefix only (causal attention)
        if (prefix_len > 0) {
            self.ctx_graph.reset();
            self.ctx_graph.setNoAlloc(false);
            const p1_input = try self.ctx_graph.newTensor1d(.i32, prefix_len);
            self.ctx_graph.setNoAlloc(true);
            {
                const data = p1_input.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(prefix_len))];
                for (prefix_tokens_raw.items, 0..) |t, j| {
                    slice[j] = @as(i32, @intCast(t));
                }
            }

            var p1_graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
            var p1_builder = graph_builder.GraphBuilder.init(self.ctx_graph, p1_graph, &self.params, self.allocator);
            _ = try self.model.buildGraph(&p1_builder, p1_input, prefix_len, kv_cache_ptr, 0);

            var p1_galloc = try ggml.Gallocr.init(buft);
            defer p1_galloc.free();
            if (!p1_galloc.allocGraph(p1_graph)) {
                logger.err("Graph alloc failed: audio text-prefix pass", .{});
                return error.GraphAllocFailed;
            }
            try p1_graph.compute(self.n_threads);
            logger.debug("Pass 1 (audio text prefix): {d} tokens ✓", .{prefix_len});
        }

        // Pass 2: Audio tokens only (non-causal attention)
        {
            self.ctx_graph.reset();
            self.ctx_graph.setNoAlloc(false);
            const p2_input = try self.ctx_graph.newTensor1d(.i32, n_audio_tokens);
            self.ctx_graph.setNoAlloc(true);
            {
                const data = p2_input.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_audio_tokens))];
                @memset(slice, @as(i32, @intCast(audio_token_id)));
            }

            var p2_graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
            _ = try gemma4_model.forwardWithEmbdOverride(
                self.ctx_graph, p2_graph, p2_input, n_audio_tokens,
                kv_cache_ptr, prefix_len, audio_embeddings, 0, false,
            );

            var p2_galloc = try ggml.Gallocr.init(buft);
            defer p2_galloc.free();
            if (!p2_galloc.allocGraph(p2_graph)) {
                logger.err("Graph alloc failed: audio pass", .{});
                return error.GraphAllocFailed;
            }
            const t2_start = engine_common.currentTimeMs();
            try p2_graph.compute(self.n_threads);
            const t2_end = engine_common.currentTimeMs();
            pp_time_s += @as(f64, @floatFromInt(t2_end - t2_start)) / 1000.0;
            logger.debug("Pass 2 (audio): {d} tokens (non-causal) ✓", .{n_audio_tokens});
        }

        // Pass 3: Text suffix only (causal attention) — sample logits from here
        self.ctx_graph.reset();
        self.ctx_graph.setNoAlloc(false);
        const sfx_n: i32 = if (suffix_len > 0) suffix_len else 1;
        const p3_input = try self.ctx_graph.newTensor1d(.i32, sfx_n);
        self.ctx_graph.setNoAlloc(true);
        {
            const data = p3_input.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(sfx_n))];
            if (suffix_len > 0) {
                for (suffix_tokens_raw.items, 0..) |t, j| {
                    slice[j] = @as(i32, @intCast(t));
                }
            } else {
                slice[0] = @as(i32, @intCast(audio_token_id));
            }
        }

        const suffix_start_pos: i32 = prefix_len + n_audio_tokens;
        var p3_graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var p3_builder = graph_builder.GraphBuilder.init(self.ctx_graph, p3_graph, &self.params, self.allocator);
        const logits = try self.model.buildGraph(&p3_builder, p3_input, sfx_n, kv_cache_ptr, suffix_start_pos);

        var p3_galloc = try ggml.Gallocr.init(buft);
        defer p3_galloc.free();
        if (!p3_galloc.allocGraph(p3_graph)) {
            logger.err("Graph alloc failed: audio text-suffix pass", .{});
            return error.GraphAllocFailed;
        }

        const t3_start = engine_common.currentTimeMs();
        try p3_graph.compute(self.n_threads);
        const t3_end = engine_common.currentTimeMs();
        pp_time_s += @as(f64, @floatFromInt(t3_end - t3_start)) / 1000.0;
        logger.debug("Pass 3 (audio text suffix): {d} tokens ✓", .{sfx_n});

        var current_token: i32 = sampler.Sampler.sampleGreedy(logits);
        var pos: i32 = suffix_start_pos + (if (suffix_len > 0) suffix_len else 1);
        var gen_count: u32 = 0;

        logger.info("First generated token (audio): id={d}, is_eog={}, pp_time={d:.3}s", .{
            current_token,
            self.tok.isEog(@intCast(current_token)),
            pp_time_s,
        });


        const t_tg_start = engine_common.currentTimeMs();

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

        const t_tg_end = engine_common.currentTimeMs();
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
        engine_common.setLogLevel(.debug);
    } else if (args.verbose) {
        engine_common.setLogLevel(.info);
    } else {
        engine_common.setLogLevel(.warn);
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

    if (args.embed) {
        logger.info("--- Embedding Generation ---", .{});
        const emb = engine.generateEmbedding(io, args.prompt) catch |err| {
            logger.err("Embedding generation failed: {}", .{err});
            return;
        };
        defer allocator.free(emb);

        logger.info("--- Done (dims={d}) ---", .{emb.len});
    } else if (args.chat) {
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
const test_embed = @import("tests/test_embed.zig");
