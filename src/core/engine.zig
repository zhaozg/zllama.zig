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
const mm = @import("mm");
const preprocess = @import("preprocess");
const engine_common = @import("engine_common");
const prefill_mod = @import("prefill");

const chat_template = @import("chat_template");
const CliArgs = @import("../cli_args.zig").CliArgs;
const loadMMProj = @import("loader.zig").loadMMProj;

const logger = std.log.scoped(.engine);

// ============================================================================
// VTable types for strategy dispatch
// ============================================================================

/// Callbacks for the shared decode loop.
const DecodeCallbacks = struct {
    buildStep: *const fn (ctx: *anyopaque, token: i32, pos: i32) anyerror!*ggml.Tensor,
    sample: *const fn (ctx: *anyopaque, logits: *ggml.Tensor) i32,
    skipToken: *const fn (ctx: *anyopaque, token: i32) bool,
    afterToken: ?*const fn (ctx: *anyopaque, token: i32, decoded: []const u8) anyerror!bool = null,
    onComplete: ?*const fn (ctx: *anyopaque, n_decoded: i32, tg_time_s: f64) void = null,
};

const PrefillResult = struct {
    logits: []f32,
    pos: i32,
    pp_time_s: f64,
};

// ============================================================================
// InferenceEngine
// ============================================================================

pub const InferenceEngine = struct {
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

    inc_ctx: graph_context.IncContext,

    mm_manager: ?mm.MultiModalManager = null,
    capabilities: model_if.ModelCapabilities = .{},

    chat_template_source: ?chat_template.TemplateSource = null,
    system_prompt: []const u8 = "",
    no_chat_template: bool = false,
    no_jinja: bool = false,

    /// Result of the decode loop.
    const DecodeResult = struct { gen_count: i32, tg_time_s: f64 };

    // ========================================================================
    // init / deinit
    // ========================================================================

    pub fn init(io: std.Io, allocator: std.mem.Allocator, model_path: [:0]const u8, cli_args: *const CliArgs) !InferenceEngine {
        const gguf_data = try engine_common.readFileToMemory(io, allocator, model_path);
        errdefer allocator.free(gguf_data);

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
        var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
        errdefer tok.deinit();
        logger.info("Tokenizer: {d} tokens", .{tok.vocabSize()});

        const n_threads = if (cli_args.n_threads > 0) cli_args.n_threads else ggml.recommendedThreads();
        const mem_size_estimate = 2 * 1024 * 1024 * 1024;
        logger.info("Estimated memory: {d} MB", .{mem_size_estimate / (1024 * 1024)});
        const ctx_weights = try ggml.Context.initNoAlloc(mem_size_estimate);
        errdefer ctx_weights.deinit();
        const ctx_kv_cache = try ggml.Context.init(mem_size_estimate);
        errdefer ctx_kv_cache.deinit();
        const max_seq_len = @min(params.max_seq_len, 2048);
        const hdim_kv = params.n_head_dim;
        const hdim_k = @max(params.n_head_dim, params.n_head_dim_k);
        const hdim_v = if (params.n_head_dim_v > 0) @max(params.n_head_dim, params.n_head_dim_v) else hdim_kv;
        var kv_cache_mgr = try kv_cache.KVCache.initWithKVDim(ctx_kv_cache, params.n_layer, params.n_kv_head, hdim_k, hdim_v, max_seq_len, allocator);
        errdefer kv_cache_mgr.deinit(allocator);
        model.setKVCacheContext(ctx_kv_cache);
        const ctx_graph = try ggml.Context.initNoAlloc(mem_size_estimate);
        const inc_ctx_size = 512 * 1024 * 1024;
        const inc_ctx = try graph_context.IncContext.init(allocator, &params, inc_ctx_size);

        var mm_manager: ?mm.MultiModalManager = null;
        if (cli_args.mmproj_path.len > 0) {
            mm_manager = try loadMMProj(io, allocator, cli_args.mmproj_path, &capabilities);
            logger.info("Multimodal encoder loaded from: {s}", .{cli_args.mmproj_path});
            if (capabilities.has_vision or capabilities.has_audio) {
                logger.info("Multi-modal: yes", .{});
                if (capabilities.has_vision) logger.info("  Vision: yes ({s})", .{capabilities.vision_encoder_type});
                if (capabilities.has_audio) logger.info("  Audio : yes ({s}, {d} Hz)", .{ capabilities.audio_encoder_type, capabilities.audio_sample_rate });
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
                logger.warn("Unknown chat template '{s}', falling back to arch default", .{cli_args.chat_template_name});
            }
        } else if (gguf_file.getString("tokenizer.chat_template")) |tmpl_str| {
            const owned = try allocator.dupe(u8, tmpl_str);
            chat_template_source = chat_template.TemplateSource{ .gguf_builtin = owned };
            logger.info("Chat template: from GGUF metadata", .{});
        }
        if (cli_args.verbose or cli_args.debug) {
            const source = chat_template_source orelse chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(arch, null) };
            if (chat_template.debugPrintTemplate(allocator, source, arch, null)) |debug_info| {
                defer allocator.free(debug_info);
                logger.info("{s}", .{debug_info});
            } else |_| {}
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
            .chat_template_source = chat_template_source,
            .system_prompt = system_prompt,
            .no_chat_template = cli_args.no_chat_template,
            .no_jinja = cli_args.no_jinja,
        };
    }

    pub fn deinit(self: *InferenceEngine) void {
        if (self.mm_manager) |*m| m.deinit();
        self.inc_ctx.deinit();
        self.ctx_graph.deinit();
        self.ctx_kv_cache.deinit();
        self.model.deinit(self.allocator);
        if (self.params.model_name.len > 0) self.allocator.free(self.params.model_name);
        self.allocator.free(self.gguf_data);
        self.tok.deinit();
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
    // Public chat template API
    // ========================================================================

    /// Apply chat template for a single-turn user prompt.
    /// Returns the formatted prompt (caller owns memory).
    pub fn applyChatTemplate(self: *InferenceEngine, user_prompt: []const u8) ![]const u8 {
        return self.applyChatTemplateWithMedia(user_prompt, null);
    }

    /// Apply chat template for a single-turn user prompt with optional media attachment.
    /// Returns the formatted prompt (caller owns memory).
    pub fn applyChatTemplateWithMedia(self: *InferenceEngine, user_prompt: []const u8, media: ?chat_template.Media) ![]const u8 {
        if (self.no_chat_template) return self.allocator.dupe(u8, user_prompt);
        const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
        const source = self.chat_template_source orelse chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };
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

    /// Apply chat template for multi-turn conversation (chat history).
    /// The last message in chat_history should be the new user message.
    /// Returns the formatted prompt (caller owns memory).
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
    // Shared: gallocr reservation
    // ========================================================================

    fn reserveDecodeGallocr(self: *InferenceEngine) !void {
        const saved_lens = try self.kv_cache_mgr.getAllLengths(self.allocator);
        defer self.allocator.free(saved_lens);
        const max_pos: u32 = self.kv_cache_mgr.max_seq_len -| 1;
        self.kv_cache_mgr.setAllLengths(max_pos);
        const reserve_step = try self.inc_ctx.beginStep();
        reserve_step.setToken(0);
        var reserve_builder = graph_builder.GraphBuilder.init(reserve_step.ctx, reserve_step.graph, &self.params, self.allocator);
        _ = try self.model.buildGraph(&reserve_builder, reserve_step.input_token, 1, @ptrCast(&self.kv_cache_mgr), @intCast(max_pos));
        try self.inc_ctx.reserveGallocr(reserve_step.graph);
        for (self.kv_cache_mgr.layers, 0..) |*layer, i| layer.current_len = saved_lens[i];
    }

    // ========================================================================
    // Shared: text prefill
    // ========================================================================

    fn textPrefill(self: *InferenceEngine, input_tokens: []const u32) !PrefillResult {
        const n_prompt_tokens: i32 = @intCast(input_tokens.len);
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_prompt_tokens);
        self.ctx_graph.setNoAlloc(true);

        var graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var builder = graph_builder.GraphBuilder.init(self.ctx_graph, graph, &self.params, self.allocator);
        const logits = try self.model.buildGraph(&builder, input_tensor, n_prompt_tokens, @ptrCast(&self.kv_cache_mgr), 0);

        const buft = ggml.backendCpuBufferType();
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();
        if (!galloc.allocGraph(graph)) return error.GraphAllocFailed;

        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_prompt_tokens))];
            for (input_tokens, 0..) |token, j| slice[j] = @as(i32, @intCast(token));
        }

        const t_pp_start = engine_common.currentTimeMs();
        try graph.compute(self.n_threads);
        const t_pp_end = engine_common.currentTimeMs();
        const pp_time_s = @as(f64, @floatFromInt(t_pp_end - t_pp_start)) / 1000.0;

        const logits_data = logits.dataF32();
        const n_vocab = @as(usize, @intCast(self.params.n_vocab));
        const logits_heap = try self.allocator.alloc(f32, n_vocab);
        @memcpy(logits_heap, logits_data[@as(usize, @intCast(n_prompt_tokens - 1)) * n_vocab ..][0..n_vocab]);

        return PrefillResult{ .logits = logits_heap, .pos = n_prompt_tokens, .pp_time_s = pp_time_s };
    }

    // ========================================================================
    // Shared: decode loop
    // ========================================================================

    fn runDecodeLoop(self: *InferenceEngine, io: std.Io, first_token: i32, start_pos: i32, max_tokens: u32, callbacks: DecodeCallbacks, ctx: *anyopaque) !DecodeResult {
        var current_token: i32 = first_token;
        var pos: i32 = start_pos;
        var gen_count: u32 = 0;
        var eog_detect_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
        defer eog_detect_buf.deinit(self.allocator);
        const t_tg_start = engine_common.currentTimeMs();

        while (gen_count < max_tokens) {
            if (self.tok.isEog(@intCast(current_token))) break;

            if (callbacks.skipToken(ctx, current_token)) {
                const step = try self.inc_ctx.beginStep();
                step.setToken(current_token);
                var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
                const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);
                if (!step.galloc.allocGraph(step.graph)) return error.GraphAllocFailed;
                try step.graph.compute(self.n_threads);
                current_token = callbacks.sample(ctx, inc_logits);
                pos += 1;
                gen_count += 1;
                continue;
            }

            var buf: [128]u8 = undefined;
            const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
            const decoded = buf[0..n];

            if (n > 0) {
                try eog_detect_buf.appendSlice(self.allocator, decoded);
                if (self.tok.isEogText(eog_detect_buf.items)) {
                    if (!self.benchmark) {
                        const stdout_file = std.Io.File.stdout();
                        try stdout_file.writeStreamingAll(io, decoded);
                    }
                    break;
                }
            }

            if (!self.benchmark and n > 0) {
                const stdout_file = std.Io.File.stdout();
                try stdout_file.writeStreamingAll(io, decoded);
            }

            if (callbacks.afterToken) |cb| {
                if (!try cb(ctx, current_token, decoded)) break;
            }

            const step = try self.inc_ctx.beginStep();
            step.setToken(current_token);
            var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
            const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);
            if (!step.galloc.allocGraph(step.graph)) return error.GraphAllocFailed;
            try step.graph.compute(self.n_threads);
            current_token = callbacks.sample(ctx, inc_logits);
            pos += 1;
            gen_count += 1;
        }

        const t_tg_end = engine_common.currentTimeMs();
        const tg_time_s = @as(f64, @floatFromInt(t_tg_end - t_tg_start)) / 1000.0;
        if (callbacks.onComplete) |cb| cb(ctx, @intCast(gen_count), tg_time_s);
        return .{ .gen_count = @intCast(gen_count), .tg_time_s = tg_time_s };
    }

    // ========================================================================
    // Shared: print stats
    // ========================================================================

    fn printStats(self: *InferenceEngine, n_prompt_tokens: i32, gen_count: i32, pp_time_s: f64, tg_time_s: f64) void {
        if (self.benchmark) {
            engine_common.printBenchmark(.{
                .model_name = if (self.params.model_name.len > 0) self.params.model_name else @tagName(self.arch),
                .arch_name = @tagName(self.arch),
                .n_threads = self.n_threads,
                .n_prompt_tokens = n_prompt_tokens,
                .n_decode = gen_count,
                .pp_time_s = pp_time_s,
                .tg_time_s = tg_time_s,
            });
        } else if (gen_count > 0) {
            engine_common.printSummary(gen_count, pp_time_s + tg_time_s);
        }
    }

    // ========================================================================
    // Shared: stream prompt tokens
    // ========================================================================

    fn streamPromptTokens(self: *InferenceEngine, io: std.Io, tokens: []const u32) !void {
        if (self.benchmark) return;
        const stdout_file = std.Io.File.stdout();
        for (tokens) |token_id| {
            var buf: [128]u8 = undefined;
            const n = try self.tok.decodeSingle(token_id, &buf);
            if (n > 0) try stdout_file.writeStreamingAll(io, buf[0..n]);
        }
    }

    // ========================================================================
    // Public API: text-only generation
    // ========================================================================

    pub fn generate(self: *InferenceEngine, io: std.Io, prompt: []const u8, max_tokens: u32) !void {
        const formatted_prompt = try self.applyChatTemplate(prompt);
        defer self.allocator.free(formatted_prompt);

        logger.info("formatted_prompt: {s}", .{formatted_prompt});

        var input_tokens = try self.tok.encode(formatted_prompt, true);
        defer input_tokens.deinit(self.allocator);

        const n_prompt_tokens: i32 = @intCast(input_tokens.items.len);
        self.model.resetSSMStates();

        const prefill = try self.textPrefill(input_tokens.items);
        defer self.allocator.free(prefill.logits);

        const first_token = sampler.Sampler.sampleGreedyFromLogits(prefill.logits);
        try self.streamPromptTokens(io, input_tokens.items);
        try self.reserveDecodeGallocr();

        const dr = try self.runDecodeLoop(io, first_token, prefill.pos, max_tokens, .{
            .buildStep = undefined,
            .sample = struct { fn f(_: *anyopaque, l: *ggml.Tensor) i32 { return sampler.Sampler.sampleGreedy(l); } }.f,
            .skipToken = struct { fn f(c: *anyopaque, t: i32) bool { return (@as(*InferenceEngine, @ptrCast(@alignCast(c)))).tok.isSkipToken(@intCast(t)); } }.f,
            .onComplete = null,
        }, @ptrCast(self));

        self.printStats(n_prompt_tokens, dr.gen_count, prefill.pp_time_s, dr.tg_time_s);

        if (!self.benchmark) {
            const stdout_file = std.Io.File.stdout();
            try stdout_file.writeStreamingAll(io, "\n");
        }
    }

    // ========================================================================
    // Public API: embedding generation
    // ========================================================================

    pub fn generateEmbedding(self: *InferenceEngine, io: std.Io, prompt: []const u8) ![]f32 {
        var input_tokens = try self.tok.encode(prompt, true);
        defer input_tokens.deinit(self.allocator);
        const n_tokens: i32 = @intCast(input_tokens.items.len);
        if (n_tokens == 0) return error.EmptyInput;

        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_tokens);
        self.ctx_graph.setNoAlloc(true);
        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
            for (input_tokens.items, 0..) |token, j| slice[j] = @as(i32, @intCast(token));
        }

        var graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var builder = graph_builder.GraphBuilder.init(self.ctx_graph, graph, &self.params, self.allocator);
        const embedding_vector = try self.model.buildGraph(&builder, input_tensor, n_tokens, null, 0);

        const buft = ggml.backendCpuBufferType();
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();
        if (!galloc.allocGraph(graph)) return error.GraphAllocFailed;
        try graph.compute(self.n_threads);

        const result_data = embedding_vector.dataF32();
        const n_embd = @as(usize, @intCast(self.params.n_embd));
        const result = try self.allocator.alloc(f32, n_embd);
        @memcpy(result, result_data[0..n_embd]);

        const stdout_file = std.Io.File.stdout();
        var buf: [128]u8 = undefined;
        for (result) |v| {
            const line = try std.fmt.bufPrint(&buf, "{d:.6}\n", .{v});
            try stdout_file.writeStreamingAll(io, line);
        }
        return result;
    }

    // ========================================================================
    // Public API: chat loop
    // ========================================================================

    pub fn chatLoop(self: *InferenceEngine, io: std.Io) !void {
        const stdin = std.Io.File.stdin();
        const stdout = std.Io.File.stdout();
        try stdout.writeStreamingAll(io, "zllama chat - Interactive AI\n");
        try stdout.writeStreamingAll(io, "Type /help for commands.\n\n");

        var line_buf: [4096]u8 = undefined;
        var history = std.ArrayListUnmanaged([]const u8){ .items = &.{}, .capacity = 0 };
        defer { for (history.items) |item| self.allocator.free(item); history.deinit(self.allocator); }
        var chat_history = std.ArrayListUnmanaged(chat_template.ChatMessage){ .items = &.{}, .capacity = 0 };
        defer chat_history.deinit(self.allocator);

        while (true) {
            try stdout.writeStreamingAll(io, ">>> ");
            var reader = stdin.reader(io, &line_buf);
            const line_slice = reader.interface.takeDelimiterExclusive('\n') catch |err| {
                if (err == error.EndOfStream) { try stdout.writeStreamingAll(io, "\n"); break; }
                return err;
            };
            const line = std.mem.trimEnd(u8, line_slice, "\r");
            if (line.len == 0) continue;

            if (history.items.len == 0 or !std.mem.eql(u8, history.getLast(), line)) {
                try history.append(self.allocator, try self.allocator.dupe(u8, line));
            }

            if (line[0] == '/') {
                if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) {
                    try stdout.writeStreamingAll(io, "Bye.\n"); break;
                } else if (std.mem.eql(u8, line, "/help")) {
                    try stdout.writeStreamingAll(io, "Available commands:\n  /help  /clear  /exit  /reset  /new  /image  /audio\n");
                } else if (std.mem.eql(u8, line, "/clear")) {
                    try stdout.writeStreamingAll(io, "\x1b[2J\x1b[H");
                } else if (std.mem.eql(u8, line, "/reset")) {
                    self.kv_cache_mgr.reset(); self.model.resetSSMStates();
                    try stdout.writeStreamingAll(io, "KV cache and SSM states reset.\n");
                } else if (std.mem.eql(u8, line, "/new")) {
                    chat_history.clearAndFree(self.allocator); self.kv_cache_mgr.reset(); self.model.resetSSMStates();
                    try stdout.writeStreamingAll(io, "New conversation started.\n");
                } else if (std.mem.startsWith(u8, line, "/image ")) {
                    const rest = line[7..]; const sp = std.mem.indexOf(u8, rest, " ") orelse { try stdout.writeStreamingAll(io, "Usage: /image <path> <prompt>\n"); continue; };
                    try stdout.writeStreamingAll(io, "Processing image...\n");
                    try chat_history.append(self.allocator, chat_template.ChatMessage.withMedia("user", rest[sp+1..], .{ .type = .image, .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } } }));
                    self.generateWithImage(io, rest[sp+1..], @ptrCast(@constCast(rest[0..sp])), 256) catch { try stdout.writeStreamingAll(io, "Image processing failed.\n"); };
                    try stdout.writeStreamingAll(io, "\n"); continue;
                } else if (std.mem.startsWith(u8, line, "/audio ")) {
                    const rest = line[7..]; const sp = std.mem.indexOf(u8, rest, " ") orelse { try stdout.writeStreamingAll(io, "Usage: /audio <path> <prompt>\n"); continue; };
                    try stdout.writeStreamingAll(io, "Processing audio...\n");
                    try chat_history.append(self.allocator, chat_template.ChatMessage.withMedia("user", rest[sp+1..], .{ .type = .audio, .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 0 } } }));
                    self.generateWithAudio(io, rest[sp+1..], @ptrCast(@constCast(rest[0..sp])), 256) catch { try stdout.writeStreamingAll(io, "Audio processing failed.\n"); };
                    try stdout.writeStreamingAll(io, "\n"); continue;
                } else {
                    try stdout.writeStreamingAll(io, "Unknown command. Try /help.\n");
                }
                continue;
            }

            // Normal message handling
            try chat_history.append(self.allocator, chat_template.ChatMessage.init("user", line));
            const formatted_prompt = try self.applyChatTemplateMultiTurn(chat_history.items);
            defer self.allocator.free(formatted_prompt);

            self.kv_cache_mgr.reset(); self.model.resetSSMStates();
            var input_tokens = try self.tok.encode(formatted_prompt, true);
            defer input_tokens.deinit(self.allocator);

            const prefill = try self.textPrefill(input_tokens.items);
            defer self.allocator.free(prefill.logits);

            var current_token = sampler.Sampler.sampleGreedyFromLogits(prefill.logits);
            var pos: i32 = prefill.pos;
            var gen_count: u32 = 0;
            var output_tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
            defer output_tokens.deinit(self.allocator);

            try self.reserveDecodeGallocr();

            while (gen_count < 512) {
                if (self.tok.isEog(@intCast(current_token))) break;
                var buf: [128]u8 = undefined;
                const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
                if (n > 0) try stdout.writeStreamingAll(io, buf[0..n]);
                try output_tokens.append(self.allocator, @intCast(current_token));

                const step = try self.inc_ctx.beginStep();
                step.setToken(current_token);
                var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
                const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);
                if (!step.galloc.allocGraph(step.graph)) return error.GraphAllocFailed;
                try step.graph.compute(self.n_threads);
                current_token = sampler.Sampler.sampleGreedy(inc_logits);
                pos += 1;
                gen_count += 1;
            }
            try stdout.writeStreamingAll(io, "\n");

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
        }
    }

    // ========================================================================
    // Public API: vision generation
    // ========================================================================

    pub fn generateWithImage(self: *InferenceEngine, io: std.Io, prompt: []const u8, image_path: [:0]const u8, max_tokens: u32) !void {
        var mm_mgr = self.mm_manager orelse return error.MMProjNotLoaded;
        if (!self.capabilities.has_vision) return error.VisionNotSupported;
        if (self.arch != .gemma4) { logger.warn("Vision only supported for Gemma4.", .{}); return self.generate(io, prompt, max_tokens); }
        const gemma4_model: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(self.model.ptr));

        const target_size: u32 = if (mm_mgr.vision_encoder) |enc| enc.params.image_size else 896;
        var img = try preprocess.loadImage(self.allocator, io, image_path, target_size, .auto);
        defer img.deinit();
        if (img.width == 0 or img.height == 0) return error.EmptyImage;

        self.ctx_graph.setNoAlloc(false);
        var vision_graph = try ggml.CGraph.initReserved(self.ctx_graph, 32768);
        const vision_embeddings = try mm_mgr.encodeMedia(self.ctx_graph, vision_graph, .{
            .media_type = .image, .image_data = img.data, .image_width = img.width, .image_height = img.height,
        });
        self.ctx_graph.setNoAlloc(true);
        const buft = ggml.backendCpuBufferType();
        var v_galloc = try ggml.Gallocr.init(buft);
        defer v_galloc.free();
        if (!v_galloc.allocGraph(vision_graph)) return error.GraphAllocFailed;
        try vision_graph.compute(self.n_threads);

        const n_vision_tokens: i32 = @intCast(vision_embeddings.ne()[1]);
        const n_embd_val: usize = @intCast(vision_embeddings.ne()[0]);
        if (n_embd_val != @as(usize, @intCast(self.params.n_embd))) return error.EmbeddingDimensionMismatch;

        const image_token_id: u32 = blk: {
            if (self.tok.textToToken("<|image|>")) |id| break :blk @as(u32, @intCast(id));
            if (self.tok.textToToken("<image>")) |id| break :blk @as(u32, @intCast(id));
            return error.NoImagePlaceholderToken;
        };

        const content_with_placeholder = try chat_template.ensurePlaceholderInContent(prompt, .image, self.allocator);
        defer if (content_with_placeholder.ptr != prompt.ptr) self.allocator.free(content_with_placeholder);
        const formatted_prompt = if (self.no_chat_template) try self.allocator.dupe(u8, content_with_placeholder)
        else try self.applyChatTemplateWithMedia(content_with_placeholder, chat_template.Media{ .type = .image, .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } } });
        defer self.allocator.free(formatted_prompt);

        var expanded = try chat_template.tokenizeWithPlaceholders(self.allocator, formatted_prompt, @ptrCast(&self.tok), tokenizeTextSegment, image_token_id, 0, @intCast(n_vision_tokens), 0);
        defer expanded.deinit();
        try self.multimodalPrefill(io, gemma4_model, &expanded, image_token_id, @intCast(n_vision_tokens), vision_embeddings, max_tokens);
    }

    // ========================================================================
    // Public API: audio generation
    // ========================================================================

    pub fn generateWithAudio(self: *InferenceEngine, io: std.Io, prompt: []const u8, audio_path: [:0]const u8, max_tokens: u32) !void {
        var mm_mgr = self.mm_manager orelse return error.MMProjNotLoaded;
        if (!self.capabilities.has_audio) return error.AudioNotSupported;
        if (self.arch != .gemma4) { logger.warn("Audio only supported for Gemma4.", .{}); return self.generate(io, prompt, max_tokens); }
        const gemma4_model: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(self.model.ptr));

        const wav_result = try preprocess.loadWav(self.allocator, io, audio_path);
        defer self.allocator.free(wav_result.samples);
        if (wav_result.samples.len == 0) return error.EmptyAudio;

        const preprocess_params = preprocess.AudioPreprocessParams.fromAudioEncoder(if (mm_mgr.audio_encoder) |enc| enc.params.n_mel_bins else preprocess.AUDIO_N_MEL_BINS);
        var mel = try preprocess.computeMelSpectrogram(self.allocator, wav_result.samples, wav_result.info.sample_rate, preprocess_params);
        defer mel.deinit();

        self.ctx_graph.setNoAlloc(false);
        var audio_graph = try ggml.CGraph.initReserved(self.ctx_graph, 32768);
        const audio_embeddings = try mm_mgr.encodeMedia(self.ctx_graph, audio_graph, .{
            .media_type = .audio, .mel_data = mel.data, .mel_bins = mel.n_mel_bins, .mel_frames = mel.n_frames,
            .audio_length_sec = @as(f32, @floatFromInt(wav_result.info.num_samples)) / @as(f32, @floatFromInt(wav_result.info.sample_rate)),
        });
        self.ctx_graph.setNoAlloc(true);
        const buft = ggml.backendCpuBufferType();
        var a_galloc = try ggml.Gallocr.init(buft);
        defer a_galloc.free();
        if (!a_galloc.allocGraph(audio_graph)) return error.GraphAllocFailed;
        try audio_graph.compute(self.n_threads);

        const n_audio_tokens: i32 = @intCast(audio_embeddings.ne()[1]);
        const n_embd_val: usize = @intCast(audio_embeddings.ne()[0]);
        if (n_embd_val != @as(usize, @intCast(self.params.n_embd))) return error.EmbeddingDimensionMismatch;

        const audio_token_id: u32 = blk: {
            if (self.tok.textToToken("<|audio|>")) |id| break :blk @as(u32, @intCast(id));
            if (self.tok.textToToken("<audio>")) |id| break :blk @as(u32, @intCast(id));
            return error.NoAudioPlaceholderToken;
        };

        const content_with_placeholder = try chat_template.ensurePlaceholderInContent(prompt, .audio, self.allocator);
        defer if (content_with_placeholder.ptr != prompt.ptr) self.allocator.free(content_with_placeholder);
        const formatted_prompt = if (self.no_chat_template) try self.allocator.dupe(u8, content_with_placeholder)
        else try self.applyChatTemplateWithMedia(content_with_placeholder, chat_template.Media{ .type = .audio, .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 0 } } });
        defer self.allocator.free(formatted_prompt);

        var expanded = try chat_template.tokenizeWithPlaceholders(self.allocator, formatted_prompt, @ptrCast(&self.tok), tokenizeTextSegment, 0, audio_token_id, 0, @intCast(n_audio_tokens));
        defer expanded.deinit();
        try self.multimodalPrefill(io, gemma4_model, &expanded, audio_token_id, @intCast(n_audio_tokens), audio_embeddings, max_tokens);
    }

    // ========================================================================
    // Shared: three-stage multimodal prefill + decode
    // ========================================================================

    fn multimodalPrefill(self: *InferenceEngine, io: std.Io, gemma4_model: *model_if.gemma4.Gemma4Model, expanded: *chat_template.TokenizedSegments, media_token_id: u32, n_media_tokens: i32, media_embeddings: *ggml.Tensor, max_tokens: u32) !void {
        const n_total_tokens: i32 = @intCast(expanded.tokens.items.len);
        const media_offset: u32 = if (expanded.offsets.len > 0) expanded.offsets[0].token_offset else 0;
        const media_count: u32 = if (expanded.offsets.len > 0) expanded.offsets[0].token_count else @intCast(n_media_tokens);
        const prefix_tokens = if (media_offset > 0) expanded.tokens.items[0..media_offset] else &[_]u32{};
        const suffix_start: u32 = media_offset + media_count;
        const suffix_tokens = if (suffix_start < n_total_tokens) expanded.tokens.items[suffix_start..@as(usize, @intCast(n_total_tokens))] else &[_]u32{};

        const mediaForwardFn = struct {
            fn f(mp: *anyopaque, c: *ggml.Context, g: *ggml.CGraph, it: *ggml.Tensor, nt: i32, kvc: ?*kv_cache.KVCache, sp: i32, eo: *ggml.Tensor, eoff: i32, causal: bool) anyerror!*ggml.Tensor {
                return (@as(*model_if.gemma4.Gemma4Model, @ptrCast(@alignCast(mp)))).mediaForward(c, g, it, nt, kvc, sp, eo, eoff, causal);
            }
        }.f;

        const embd_raw = media_embeddings.dataF32();
        const embd_dim: u32 = @intCast(media_embeddings.ne()[0]);
        const embd_heap = try self.allocator.dupe(f32, embd_raw);
        defer self.allocator.free(embd_heap);

        self.model.resetSSMStates();
        const pr = try prefill_mod.threeStagePrefill(self.ctx_graph, self.model, @ptrCast(@alignCast(gemma4_model)), &mediaForwardFn, &self.kv_cache_mgr, prefix_tokens, media_token_id, @intCast(media_count), embd_heap, embd_dim, suffix_tokens, &self.params, self.n_threads, self.allocator);

        var best_idx: i32 = 0;
        var best_val: f32 = pr.logits[0];
        for (pr.logits, 0..) |v, j| { if (v > best_val) { best_val = v; best_idx = @intCast(j); } }
        self.allocator.free(pr.logits);

        try self.reserveDecodeGallocr();
        var eog_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
        defer eog_buf.deinit(self.allocator);
        const t_tg_start = engine_common.currentTimeMs();

        var current_token: i32 = best_idx;
        var pos: i32 = pr.pos;
        var gen_count: u32 = 0;

        while (gen_count < max_tokens) {
            if (self.tok.isEog(@intCast(current_token))) break;
            if (self.tok.isSkipToken(@intCast(current_token))) {
                const step = try self.inc_ctx.beginStep(); step.setToken(current_token);
                var ib = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
                const il = try self.model.buildGraph(&ib, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);
                if (!step.galloc.allocGraph(step.graph)) return error.GraphAllocFailed;
                try step.graph.compute(self.n_threads);
                current_token = sampler.Sampler.sampleGreedy(il); pos += 1; gen_count += 1; continue;
            }
            var buf: [128]u8 = undefined;
            const n = try self.tok.decodeSingle(@intCast(current_token), &buf);
            const decoded = buf[0..n];
            if (n > 0) {
                try eog_buf.appendSlice(self.allocator, decoded);
                if (self.tok.isEogText(eog_buf.items)) { if (!self.benchmark) { const sf = std.Io.File.stdout(); try sf.writeStreamingAll(io, decoded); } break; }
            }
            if (!self.benchmark and n > 0) { const sf = std.Io.File.stdout(); try sf.writeStreamingAll(io, decoded); }

            const step = try self.inc_ctx.beginStep(); step.setToken(current_token);
            var ib = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
            const il = try self.model.buildGraph(&ib, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);
            if (!step.galloc.allocGraph(step.graph)) return error.GraphAllocFailed;
            try step.graph.compute(self.n_threads);
            current_token = sampler.Sampler.sampleGreedy(il); pos += 1; gen_count += 1;
        }

        const tg_time_s = @as(f64, @floatFromInt(engine_common.currentTimeMs() - t_tg_start)) / 1000.0;
        if (!self.benchmark) { const sf = std.Io.File.stdout(); try sf.writeStreamingAll(io, "\n"); }
        if (gen_count > 0) logger.info("Multimodal: {d} tokens in {d:.2}s ({d:.1} t/s)", .{ gen_count, pr.pp_time_s + tg_time_s, @as(f64, @floatFromInt(gen_count)) / (pr.pp_time_s + tg_time_s) });
    }
};

fn tokenizeTextSegment(ctx: ?*anyopaque, text: []const u8, alloc: std.mem.Allocator) ![]u32 {
    const tok: *tokenizer.Tokenizer = @ptrCast(@alignCast(ctx orelse return error.NullCtx));
    var result = try tok.encode(text, false);
    defer result.deinit(alloc);
    return try result.toOwnedSlice(alloc);
}
