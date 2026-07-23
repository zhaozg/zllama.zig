//! Inference engine — orchestrates model loading, prefill, and incremental decode.
//!
//! Uses vtable dispatch to share the decode loop across text-only, vision,
//! and audio generation paths while keeping the prefill/media-specific
//! logic in strategy objects.

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
const mtmd = @import("mtmd");
const debug = @import("debug");
const preprocess = @import("preprocess");
const audio_mod = mtmd.audio_mod;
const engine_common = @import("engine_common");
const prefill_mod = @import("prefill");

const chat_template = @import("chat_template");
const decode_mod = @import("decode");
const verbose_mod = @import("verbose");
const embedding_mod = @import("embedding_gen");
const multimodal_mod = @import("multimodal");

const CliArgs = @import("../cli_args.zig").CliArgs;
const loadMMProj = @import("loader.zig").loadMMProj;

const logger = std.log.scoped(.core_engine);

// ============================================================================
// Memory size estimation helpers
// ============================================================================

/// Estimate the ggml context size needed for KV cache metadata + tensor data.
/// KV cache: 2 (K/V) × n_layer × n_kv_head × head_dim × max_seq_len × sizeof(f32)
/// Plus 50% overhead for ggml metadata and alignment.
fn estimateKVCacheSize(params: *const model_if.ModelParams) usize {
    const kv_per_token: usize = 2 * // K and V
        @as(usize, @intCast(params.n_layer)) *
        @as(usize, @intCast(params.n_kv_head)) *
        @as(usize, @intCast(params.n_head_dim)) *
        @sizeOf(f32);
    const kv_total = kv_per_token * @as(usize, @intCast(params.max_seq_len));
    // Add overhead for ggml metadata (tensor descriptors, alignment)
    const overhead = @as(usize, @intCast(params.n_layer)) * 1024; // ~1KB per layer for metadata
    const total = kv_total + kv_total / 2 + overhead + 64 * 1024 * 1024; // +50% + 64MB safety
    return total;
}

/// Estimate the ggml context size needed for graph building (metadata only, no_alloc).
/// Graph context stores tensor descriptors and graph nodes, not tensor data.
/// For multimodal models with large vision graphs, up to ~2 GB may be needed.
/// P0: Increased from 2 GB to 4 GB to avoid OOM in multimodal prefill.
fn estimateGraphSize(params: *const model_if.ModelParams) usize {
    _ = params;
    // Graph context only stores metadata (no_alloc mode).
    // For large multimodal prefill (794+ tokens, 35 layers), we need ~2 GB of metadata.
    // Each tensor descriptor is ~416 bytes, and a full prefill graph can have 2M+ nodes.
    return 2 * 1024 * 1024 * 1024;
}

// ============================================================================
// Prefill result
// ============================================================================

const PrefillResult = struct {
    logits: []f32,
    pos: i32,
    pp_time_s: f64,
};

// ============================================================================
// ChatTokenCollector — context for chatLoop's afterToken callback
// ============================================================================

/// Collects output token IDs during the chat decode loop so the assistant
/// turn can be reconstructed into a ChatMessage after generation completes.
const ChatTokenCollector = struct {
    tokens: *std.ArrayListUnmanaged(u32),
    allocator: std.mem.Allocator,
};

// ============================================================================
// InferenceEngine
// ============================================================================

pub const InferenceEngine = struct {
    allocator: std.mem.Allocator,
    ctx_graph: *ggml.Context,
    ctx_kv_cache: *ggml.Context,
    arch: model_if.Architecture,
    model: model_if.ModelInstance,
    params: model_if.ModelParams,
    tok: *tokenizer.Tokenizer,
    sampler_state: sampler.Sampler,
    kv_cache_mgr: kv_cache.KVCache,
    n_threads: i32,
    verbose: bool,
    verbose_prompt: bool,
    benchmark: bool,
    gguf_data: []u8,
    mapped_file: ?engine_common.MappedFile = null,

    inc_ctx: graph_context.IncContext,

    /// Gallocr（计算图内存分配器，所有权上移自 engine_common.computeGraph）
    gallocr: *ggml.Gallocr,

    mm_manager: ?*mtmd.MultiModalManager = null,
    mtmd_context: ?*mtmd.MtmdContext = null,
    capabilities: model_if.ModelCapabilities = .{},

    /// GPU backend (Metal/CUDA) — nil if CPU-only
    gpu_backend: ?*ggml.Backend = null,
    gpu_enabled: bool = false,

    chat_template_source: ?chat_template.TemplateSource = null,
    system_prompt: []const u8 = "",
    no_chat_template: bool = false,
    no_jinja: bool = false,
    image_max_pixels: u32 = 0,

    // ========================================================================
    // init / deinit
    // ========================================================================

    pub fn init(io: std.Io, allocator: std.mem.Allocator, model_path: [:0]const u8, cli_args: *const CliArgs) !InferenceEngine {
        // Register ggml CPU backend (and optionally Metal/CUDA) once at engine init.
        ggml.loadBackends();
        if (cli_args.verbose) {
            ggml.logAvailableBackends();
        }

        // Detect GPU backend if requested
        var gpu_backend: ?*ggml.Backend = null;
        var gpu_enabled: bool = cli_args.gpu;
        if (cli_args.gpu) {
            gpu_backend = ggml.detectBestBackend() catch |err| blk: {
                logger.warn("GPU backend init failed ({}), falling back to CPU", .{err});
                break :blk null;
            };
            if (gpu_backend != null and !ggml.backendIsGpu(gpu_backend.?)) {
                logger.info("Best backend is not GPU; using CPU for consistency", .{});
                ggml.backendFree(gpu_backend.?);
                gpu_backend = null;
                gpu_enabled = false;
            }
            if (gpu_backend) |b| {
                logger.info("GPU backend enabled: {s}", .{ggml.backendName(b)});
            }
        }

        var init_arena = std.heap.ArenaAllocator.init(allocator);
        defer init_arena.deinit();
        _ = init_arena.allocator();

        var mapped_file = try engine_common.mmapFile(io, allocator, model_path);
        errdefer mapped_file.deinit(io);
        const gguf_data = mapped_file.data;

        var gguf_file = try gguf.parse(gguf_data, allocator);
        defer gguf_file.deinit();

        const arch = registry.detectArchitecture(&gguf_file) orelse return error.UnsupportedArchitecture;
        logger.info("Detected architecture: {s}", .{@tagName(arch)});

        var model = try registry.createModel(allocator, &gguf_file, arch, io);
        errdefer model.deinit(allocator);
        var params = model.getParams().*;

        if (params.model_name.len > 0) {
            params.model_name = try allocator.dupe(u8, params.model_name);
        }

        if (cli_args.verbose) {
            logger.info("n_vocab={d}, n_embd={d}, n_head={d}, n_kv_head={d}", .{ params.n_vocab, params.n_embd, params.n_head, params.n_kv_head });
            logger.info("n_layer={d}, n_ff={d}, n_head_dim={d}", .{ params.n_layer, params.n_ff, params.n_head_dim });
            logger.info("max_seq_len={d}, rope_theta={d}, rope_dim={d}", .{ params.max_seq_len, params.rope_theta, params.rope_dim });
        }

        var capabilities = registry.detectCapabilities(&gguf_file, arch);
        var tok = try allocator.create(tokenizer.Tokenizer);
        tok.* = try tokenizer.Tokenizer.init(&gguf_file, allocator);
        errdefer tok.deinit();
        logger.info("Tokenizer: {d} tokens", .{tok.vocabSize()});

        const n_threads = if (cli_args.n_threads > 0) cli_args.n_threads else ggml.recommendedThreads();

        // --- Dynamic memory sizing ---
        // ctx_weights is owned by the model itself (created in model.init),
        // so we don't need a separate one here.
        const max_seq_len = params.max_seq_len;
        const hdim_kv = params.n_head_dim;
        const hdim_k = @max(params.n_head_dim, params.n_head_dim_k);
        const hdim_v = if (params.n_head_dim_v > 0) @max(params.n_head_dim, params.n_head_dim_v) else hdim_kv;

        // KV cache context: sized based on actual KV cache requirements
        const kv_cache_size = estimateKVCacheSize(&params);
        logger.info("KV cache context: {d:.1} MB (max_seq_len={d})", .{
            @as(f64, @floatFromInt(kv_cache_size)) / (1024.0 * 1024.0),
            max_seq_len,
        });
        const ctx_kv_cache = try ggml.Context.init(kv_cache_size);
        errdefer ctx_kv_cache.deinit();

        var kv_cache_mgr: kv_cache.KVCache = undefined;
        {
            const per_layer_lens = model.getPerLayerMaxSeqLen(allocator);
            if (per_layer_lens) |lens| {
                defer allocator.free(lens);
                kv_cache_mgr = try kv_cache.KVCache.initWithPerLayerLens(ctx_kv_cache, params.n_layer, params.n_kv_head, hdim_k, hdim_v, lens, allocator);
            } else {
                kv_cache_mgr = try kv_cache.KVCache.initWithKVDim(ctx_kv_cache, params.n_layer, params.n_kv_head, hdim_k, hdim_v, max_seq_len, allocator);
            }
        }
        errdefer kv_cache_mgr.deinit(allocator);
        model.setKVCacheContext(ctx_kv_cache);

        // Graph context: only stores tensor metadata (no_alloc), sized for large vision graphs
        const graph_size = estimateGraphSize(&params);
        logger.info("Graph context: {d:.1} MB", .{
            @as(f64, @floatFromInt(graph_size)) / (1024.0 * 1024.0),
        });
        const ctx_graph = try ggml.Context.initNoAlloc(graph_size);
        errdefer ctx_graph.deinit();

        // Incremental decode context: sized for single-token decode graphs.
        // Gemma4 with per-layer embeddings needs more metadata space
        // due to per-layer injection tensors (n_layer × n_embd_pl views+reshapes).
        const inc_ctx_size = 512 * 1024 * 1024;
        const inc_ctx = try graph_context.IncContext.init(allocator, &params, inc_ctx_size);

        // Gallocr: 计算图内存分配器，由 InferenceEngine 管理生命周期
        const buft = ggml.backendCpuBufferType();
        const gallocr = try ggml.Gallocr.init(buft);
        errdefer gallocr.free();

        var mm_manager: ?*mtmd.MultiModalManager = null;
        var mtmd_context: ?*mtmd.MtmdContext = null;
        if (cli_args.mmproj_path.len > 0) {
            const mm_val = try loadMMProj(io, allocator, cli_args.mmproj_path, &capabilities);
            const mm_ptr = try allocator.create(mtmd.MultiModalManager);
            mm_ptr.* = mm_val;
            mm_manager = mm_ptr;
            logger.info("Multimodal encoder loaded from: {s}", .{cli_args.mmproj_path});
            if (capabilities.has_vision or capabilities.has_audio) {
                logger.info("Multi-modal: yes", .{});
                if (capabilities.has_vision) logger.info("  Vision: yes ({s})", .{capabilities.vision_encoder_type});
                if (capabilities.has_audio) logger.info("  Audio : yes ({s}, {d} Hz)", .{ capabilities.audio_encoder_type, capabilities.audio_sample_rate });
            }
            if (mm_manager) |mgr| {
                mtmd_context = try mtmd.MtmdContext.init(allocator, mgr, @intCast(params.n_embd), mtmd.contextParamsDefault(), tok);
                logger.info("MtmdContext initialized for multimodal processing", .{});
            }
        } else if (capabilities.has_vision or capabilities.has_audio) {
            logger.warn("Model has multimodal capabilities but no --mmproj file provided", .{});
            logger.warn("  Use --mmproj <path> to load vision/audio encoder weights", .{});
        }

        var chat_template_source: ?chat_template.TemplateSource = null;
        var system_prompt: []const u8 = "";
        if (cli_args.system_prompt.len > 0) system_prompt = try allocator.dupe(u8, cli_args.system_prompt);
        if (cli_args.no_chat_template) {
            chat_template_source = null;
        } else if (cli_args.chat_template_name.len > 0) {
            if (chat_template.TemplateKind.fromString(cli_args.chat_template_name)) |kind| {
                chat_template_source = chat_template.TemplateSource{ .preset = kind };
                logger.info("Chat template: {s} (from --chat-template)", .{cli_args.chat_template_name});
            } else {
                const owned = try allocator.dupe(u8, cli_args.chat_template_name);
                chat_template_source = chat_template.TemplateSource{ .custom = owned };
                logger.info("Chat template: custom Jinja ({d} bytes) (from --chat-template)", .{cli_args.chat_template_name.len});
            }
        } else if (gguf_file.getString("tokenizer.chat_template")) |tmpl_str| {
            const owned = try allocator.dupe(u8, tmpl_str);
            chat_template_source = chat_template.TemplateSource{ .gguf_builtin = owned };
            logger.info("Chat template: from GGUF metadata ({d} bytes)", .{tmpl_str.len});
        }

        const sampler_state = sampler.Sampler.init(.{
            .temperature = cli_args.temperature,
            .top_k = cli_args.top_k,
            .top_p = cli_args.top_p,
        });

        return InferenceEngine{
            .allocator = allocator,
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
            .verbose_prompt = cli_args.verbose_prompt,
            .gguf_data = gguf_data,
            .mapped_file = mapped_file,
            .benchmark = cli_args.benchmark,
            .inc_ctx = inc_ctx,
            .gallocr = gallocr,
            .mm_manager = mm_manager,
            .mtmd_context = mtmd_context,
            .capabilities = capabilities,
            .chat_template_source = chat_template_source,
            .system_prompt = system_prompt,
            .no_chat_template = cli_args.no_chat_template,
            .no_jinja = cli_args.no_jinja,
            .image_max_pixels = cli_args.image_max_pixels,
            .gpu_backend = gpu_backend,
            .gpu_enabled = gpu_enabled,
        };
    }

    pub fn deinit(self: *InferenceEngine) void {
        // Gallocr 必须在所有依赖它的张量使用之后释放
        self.gallocr.free();
        if (self.gpu_backend) |b| {
            ggml.backendFree(b);
        }
        if (self.mtmd_context) |ctx| {
            ctx.deinit();
        }
        if (self.mm_manager) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        self.inc_ctx.deinit();
        self.ctx_graph.deinit();
        self.ctx_kv_cache.deinit();
        self.model.deinit(self.allocator);
        if (self.params.model_name.len > 0) self.allocator.free(self.params.model_name);
        if (self.mapped_file) |*mf| {
            if (!mf.is_mmap) {
                self.allocator.free(mf.data);
            }
        } else {
            self.allocator.free(self.gguf_data);
        }
        self.tok.deinit();
        self.allocator.destroy(self.tok);
        self.kv_cache_mgr.deinit(self.allocator);
        if (self.chat_template_source) |src| {
            switch (src) {
                .gguf_builtin => |s| self.allocator.free(s),
                .custom => |s| self.allocator.free(s),
                .preset => {},
            }
        }
        if (self.system_prompt.len > 0) self.allocator.free(self.system_prompt);
    }

    // ========================================================================
    // Chat template API
    // ========================================================================

    pub fn applyChatTemplate(self: *InferenceEngine, user_prompt: []const u8) ![]const u8 {
        return self.applyChatTemplateWithMedia(user_prompt, null);
    }

    pub fn applyChatTemplateWithMedia(self: *InferenceEngine, user_prompt: []const u8, media: ?chat_template.Media) ![]const u8 {
        const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
        const system = if (self.system_prompt.len > 0) self.system_prompt else null;
        return chat_template.applyWithMedia(
            self.allocator,
            self.arch,
            model_name,
            self.chat_template_source,
            self.no_chat_template,
            self.no_jinja,
            user_prompt,
            media,
            system,
        );
    }

    pub fn applyChatTemplateMultiTurn(self: *InferenceEngine, chat_history: []const chat_template.ChatMessage) ![]const u8 {
        if (self.no_chat_template) {
            if (chat_history.len == 0) return self.allocator.dupe(u8, "");
            return self.allocator.dupe(u8, chat_history[chat_history.len - 1].content);
        }
        const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
        const source = self.chat_template_source orelse chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };
        var tmpl = try chat_template.resolve(self.allocator, source, self.arch, model_name, !self.no_jinja);
        defer tmpl.deinit(self.allocator);
        const system = if (self.system_prompt.len > 0) self.system_prompt else null;
        return tmpl.apply(self.allocator, chat_history, system, true);
    }

    // ========================================================================
    // Text prefill
    // ========================================================================

    fn textPrefill(self: *InferenceEngine, input_tokens: []const u32) !PrefillResult {
        const n_prompt_tokens: i32 = @intCast(input_tokens.len);
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_prompt_tokens);
        self.ctx_graph.setNoAlloc(true);

        var graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var builder = graph_builder.GraphBuilder.init(self.ctx_graph, graph, &self.params, self.allocator);
        const logits = try self.model.buildGraph(&builder, input_tensor, n_prompt_tokens, @ptrCast(&self.kv_cache_mgr), 0);

        // 使用引擎持有的 gallocr（所有权上移）
        if (!self.gallocr.allocGraph(graph)) return error.GraphAllocFailed;

        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_prompt_tokens))];
            for (input_tokens, 0..) |token, j| slice[j] = @as(i32, @intCast(token));
        }

        const t_pp_start = engine_common.currentTimeMs();
        try graph.compute(self.n_threads);
        const t_pp_end = engine_common.currentTimeMs();
        const pp_time_s = @as(f64, @floatFromInt(t_pp_end - t_pp_start)) / 1000.0;

        const n_vocab = @as(usize, @intCast(self.params.n_vocab));
        const logits_heap = try self.allocator.alloc(f32, n_vocab);
        {
            const logits_data = try logits.dataGet(f32, self.allocator);
            defer self.allocator.free(logits_data);
            @memcpy(logits_heap, logits_data[@as(usize, @intCast(n_prompt_tokens - 1)) * n_vocab ..][0..n_vocab]);
        }

        return PrefillResult{ .logits = logits_heap, .pos = n_prompt_tokens, .pp_time_s = pp_time_s };
    }

    // ========================================================================
    // Stream prompt tokens
    // ========================================================================

    fn streamPromptTokens(self: *InferenceEngine, io: std.Io, tokens: []const u32) !void {
        if (self.benchmark) return;
        const stdout_file = std.Io.File.stdout();
        for (tokens) |token_id| {
            var buf: [128]u8 = undefined;
            const n = try self.tok.decodeSingle(token_id, &buf);
            if (n > 0) {
                try stdout_file.writeStreamingAll(io, buf[0..n]);
            }
        }
        try stdout_file.writeStreamingAll(io, "\n");
    }

    // ========================================================================
    // Text generation (public API)
    // ========================================================================

    pub fn generate(self: *InferenceEngine, io: std.Io, prompt: []const u8, max_tokens: u32) !void {
        const formatted_prompt = try self.applyChatTemplate(prompt);
        defer self.allocator.free(formatted_prompt);

        var input_tokens = try self.tok.encode(formatted_prompt, true, true);
        defer input_tokens.deinit(self.allocator);

        const n_prompt_tokens: i32 = @intCast(input_tokens.items.len);
        self.model.resetSSMStates();

        const prefill = try self.textPrefill(input_tokens.items);
        defer self.allocator.free(prefill.logits);

        if (self.verbose_prompt) {
            try verbose_mod.printVerbosePrompt(io, self.allocator, self.tok, formatted_prompt, input_tokens.items, prefill.logits, null);
        }

        const first_token = sampler.Sampler.sampleGreedyFromLogits(prefill.logits);
        try self.streamPromptTokens(io, input_tokens.items);

        try decode_mod.reserveDecodeGallocr(self.allocator, &self.kv_cache_mgr, &self.inc_ctx, self.model, &self.params);

        const dr = try decode_mod.runDecodeLoop(
            self.allocator,
            io,
            self.model,
            &self.params,
            self.tok,
            &self.kv_cache_mgr,
            &self.inc_ctx,
            self.n_threads,
            first_token,
            prefill.pos,
            max_tokens,
            .{
                .sample = struct {
                    fn f(_: *anyopaque, l: *ggml.Tensor) i32 {
                        return sampler.Sampler.sampleGreedy(l);
                    }
                }.f,
                .skipToken = struct {
                    fn f(c: *anyopaque, t: i32) bool {
                        return (@as(*InferenceEngine, @ptrCast(@alignCast(c)))).tok.isSkipToken(@intCast(t));
                    }
                }.f,
                .onComplete = null,
            },
            @ptrCast(self),
            self.benchmark,
        );

        const stdout_file = std.Io.File.stdout();
        try stdout_file.writeStreamingAll(io, "\n");
        decode_mod.printStats(self.arch, self.params.model_name, self.n_threads, n_prompt_tokens, dr.gen_count, prefill.pp_time_s, dr.tg_time_s, self.benchmark);
        if (!self.benchmark) {
            try stdout_file.writeStreamingAll(io, "\n");
        }
    }

    // ========================================================================
    // Embedding generation (public API)
    // ========================================================================

    pub fn generateEmbedding(self: *InferenceEngine, io: std.Io, prompt: []const u8) ![]f32 {
        return embedding_mod.generateEmbedding(self.allocator, io, self.model, &self.params, self.tok, self.ctx_graph, self.n_threads, prompt);
    }

    // ========================================================================
    // Chat loop (public API)
    // ========================================================================

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
                try history.append(self.allocator, try self.allocator.dupe(u8, line));
            }

            if (line[0] == '/') {
                if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) {
                    try stdout.writeStreamingAll(io, "Bye.\n");
                    break;
                } else if (std.mem.eql(u8, line, "/help")) {
                    try stdout.writeStreamingAll(io, "Available commands:\n  /help  /clear  /exit  /reset  /new\n");
                } else if (std.mem.eql(u8, line, "/clear")) {
                    try stdout.writeStreamingAll(io, "\x1b[2J\x1b[H");
                } else if (std.mem.eql(u8, line, "/reset")) {
                    self.kv_cache_mgr.reset();
                    self.model.resetSSMStates();
                    try stdout.writeStreamingAll(io, "KV cache and SSM states reset.\n");
                } else if (std.mem.eql(u8, line, "/new")) {
                    chat_history.clearAndFree(self.allocator);
                    self.kv_cache_mgr.reset();
                    self.model.resetSSMStates();
                    try stdout.writeStreamingAll(io, "New conversation started.\n");
                } else {
                    try stdout.writeStreamingAll(io, "Unknown command. Try /help.\n");
                }
                continue;
            }

            try chat_history.append(self.allocator, chat_template.ChatMessage.init("user", line));
            const formatted_prompt = try self.applyChatTemplateMultiTurn(chat_history.items);
            defer self.allocator.free(formatted_prompt);

            self.kv_cache_mgr.reset();
            self.model.resetSSMStates();
            var input_tokens = try self.tok.encode(formatted_prompt, true, true);
            defer input_tokens.deinit(self.allocator);

            const prefill = try self.textPrefill(input_tokens.items);
            defer self.allocator.free(prefill.logits);

            try decode_mod.reserveDecodeGallocr(self.allocator, &self.kv_cache_mgr, &self.inc_ctx, self.model, &self.params);

            // Collect output tokens via afterToken callback for building assistant message.
            var output_tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
            defer output_tokens.deinit(self.allocator);
            var cb_ctx = ChatTokenCollector{ .tokens = &output_tokens, .allocator = self.allocator };

            const dr = try decode_mod.runDecodeLoop(
                self.allocator,
                io,
                self.model,
                &self.params,
                self.tok,
                &self.kv_cache_mgr,
                &self.inc_ctx,
                self.n_threads,
                sampler.Sampler.sampleGreedyFromLogits(prefill.logits),
                prefill.pos,
                512,
                .{
                    .sample = struct {
                        fn f(_: *anyopaque, l: *ggml.Tensor) i32 {
                            return sampler.Sampler.sampleGreedy(l);
                        }
                    }.f,
                    .skipToken = struct {
                        fn f(c: *anyopaque, t: i32) bool {
                            return (@as(*InferenceEngine, @ptrCast(@alignCast(c)))).tok.isSkipToken(@intCast(t));
                        }
                    }.f,
                    .afterToken = struct {
                        fn f(c: *anyopaque, token: i32, decoded: []const u8) anyerror!bool {
                            const col = @as(*ChatTokenCollector, @ptrCast(@alignCast(c)));
                            try col.tokens.append(col.allocator, @intCast(token));
                            _ = decoded;
                            return true;
                        }
                    }.f,
                },
                @ptrCast(&cb_ctx),
                false,
            );

            try stdout.writeStreamingAll(io, "\n");

            // Reconstruct assistant message from collected tokens.
            if (output_tokens.items.len > 0) {
                var response_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
                defer response_buf.deinit(self.allocator);
                for (output_tokens.items) |tid| {
                    var buf: [128]u8 = undefined;
                    const n = try self.tok.decodeSingle(tid, &buf);
                    if (n > 0) try response_buf.appendSlice(self.allocator, buf[0..n]);
                }
                try chat_history.append(self.allocator, chat_template.ChatMessage.init("assistant", response_buf.items));
            }
            _ = dr;
        }
    }

    // ========================================================================
    // Multimodal generation (public API) — delegates to multimodal.zig
    // ========================================================================

    pub fn generateWithImage(self: *InferenceEngine, io: std.Io, prompt: []const u8, image_path: [:0]const u8, max_tokens: u32) !void {
        var ectx = self.toMultimodalContext();
        return multimodal_mod.generateWithImage(&ectx, io, prompt, image_path, max_tokens);
    }

    pub fn generateWithAudio(self: *InferenceEngine, io: std.Io, prompt: []const u8, audio_path: [:0]const u8, max_tokens: u32) !void {
        var ectx = self.toMultimodalContext();
        return multimodal_mod.generateWithAudio(&ectx, io, prompt, audio_path, max_tokens);
    }

    fn toMultimodalContext(self: *InferenceEngine) multimodal_mod.EngineContext {
        return .{
            .allocator = self.allocator,
            .ctx_graph = self.ctx_graph,
            .arch = self.arch,
            .model = self.model,
            .params = self.params,
            .tok = self.tok.*,
            .kv_cache_mgr = &self.kv_cache_mgr,
            .n_threads = self.n_threads,
            .verbose_prompt = self.verbose_prompt,
            .benchmark = self.benchmark,
            .inc_ctx = &self.inc_ctx,
            .gallocr = self.gallocr,
            .mm_manager = if (self.mm_manager) |m| m else null,
            .mtmd_context = self.mtmd_context,
            .capabilities = self.capabilities,
            .chat_template_source = self.chat_template_source,
            .system_prompt = self.system_prompt,
            .no_chat_template = self.no_chat_template,
            .no_jinja = self.no_jinja,
            .image_max_pixels = self.image_max_pixels,
        };
    }
};
