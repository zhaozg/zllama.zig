//! Model context — GGUF loading, model initialization, and resource setup.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.
//! Owns the model weights, KV cache, graph context, and multimodal resources.
//!
//! Reference: llama.cpp llama_model / llama_context

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_context = @import("graph_context");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const kv_cache = @import("kv_cache");
const mtmd = @import("mtmd");
const engine_common = @import("engine_common");
const chat_template = @import("chat_template");
const multimodal_mod = @import("multimodal");

const CliArgs = @import("../cli_args.zig").CliArgs;
const loadMMProj = @import("loader.zig").loadMMProj;

const logger = std.log.scoped(.core_context);

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
// ModelContext — owns all resources for a loaded model
// ============================================================================

/// Owns the model weights, KV cache, graph context, and multimodal resources.
/// Created by `init()` and consumed by `GraphPlanner` + `GraphExecutor`.
///
/// The context is the single source of truth for all model-related resources.
/// It does NOT own the decode loop or generation logic — those are in
/// `GraphPlanner` (planning) and `GraphExecutor` (execution).
pub const ModelContext = struct {
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

    /// Initialize the model context from a GGUF file path.
    /// This is the single entry point for loading a model.
    pub fn init(io: std.Io, allocator: std.mem.Allocator, model_path: [:0]const u8, cli_args: *const CliArgs) !ModelContext {
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

        return ModelContext{
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

    pub fn deinit(self: *ModelContext) void {
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

    pub fn applyChatTemplate(self: *ModelContext, user_prompt: []const u8) ![]const u8 {
        return self.applyChatTemplateWithMedia(user_prompt, null);
    }

    pub fn applyChatTemplateWithMedia(self: *ModelContext, user_prompt: []const u8, media: ?chat_template.Media) ![]const u8 {
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

    pub fn applyChatTemplateMultiTurn(self: *ModelContext, chat_history: []const chat_template.ChatMessage) ![]const u8 {
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
    // Multimodal context conversion
    // ========================================================================

    /// Convert to the EngineContext used by multimodal generation paths.
    pub fn toMultimodalContext(self: *ModelContext) multimodal_mod.EngineContext {
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

const testing = std.testing;

test "estimateKVCacheSize basic" {
    var params = model_if.ModelParams{
        .n_layer = 32,
        .n_kv_head = 8,
        .n_head_dim = 128,
        .max_seq_len = 4096,
    };
    const size = estimateKVCacheSize(&params);
    try testing.expect(size > 0);
    // 2 * 32 * 8 * 128 * 4 * 4096 = 1,073,741,824 bytes
    // +50% + overhead + 64MB = ~1.7 GB
    try testing.expect(size > 1_000_000_000);
}

test "estimateGraphSize basic" {
    var params = model_if.ModelParams{};
    const size = estimateGraphSize(&params);
    try testing.expectEqual(@as(usize, 2 * 1024 * 1024 * 1024), size);
}
