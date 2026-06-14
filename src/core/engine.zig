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

/// Tokenize a text segment for multimodal placeholder expansion.
/// Used as callback for tokenizeWithPlaceholders.
fn tokenizeTextSegment(ctx: ?*anyopaque, text: []const u8, alloc: std.mem.Allocator) ![]u32 {
    const tok: *tokenizer.Tokenizer = @ptrCast(@alignCast(ctx orelse return error.NullCtx));
    var result = try tok.encode(text, false);
    defer result.deinit(alloc);
    return try result.toOwnedSlice(alloc);
}

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
        // Debug: print template detection info when verbose or debug,
        // regardless of source (GGUF built-in, preset, or arch default).
        if (cli_args.verbose or cli_args.debug) {
            const source = chat_template_source orelse
                chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(arch, null) };
            if (chat_template.debugPrintTemplate(allocator, source, arch, null)) |debug_info| {
                defer allocator.free(debug_info);
                logger.info("{s}", .{debug_info});
            } else |_| {}
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

        // Reset SSM states before prefill
        self.model.resetSSMStates();

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

        // --- Pre-reserve inc_ctx gallocr with worst-case graph ---
        // Build a graph with position = max_seq_len-1 to cover the widest possible
        // KV cache mask. This eliminates ggml_gallocr_needs_realloc during decode.
        //
        // IMPORTANT: Do NOT call inc_ctx.resetFull() after reservation!
        // ggml_gallocr uses tensor pointers as hash keys internally.
        // resetFull() destroys all tensors, invalidating the gallocr hash table
        // and causing infinite realloc loops (see deps/guide.md).
        {
            // Save current KV cache lengths
            const saved_lens = try self.kv_cache_mgr.getAllLengths(self.allocator);
            defer self.allocator.free(saved_lens);

            // Set all layer lengths to kv_cache.max_seq_len-1 (the actual allocated cache size)
            const max_pos: u32 = self.kv_cache_mgr.max_seq_len -| 1;
            self.kv_cache_mgr.setAllLengths(max_pos);

            // Build worst-case graph through inc_ctx
            const reserve_step = try self.inc_ctx.beginStep();
            reserve_step.setToken(0); // dummy token
            var reserve_builder = graph_builder.GraphBuilder.init(
                reserve_step.ctx,
                reserve_step.graph,
                &self.params,
                self.allocator,
            );
            _ = try self.model.buildGraph(
                &reserve_builder,
                reserve_step.input_token,
                1,
                @ptrCast(&self.kv_cache_mgr),
                @intCast(max_pos),
            );

            // Reserve gallocr buffers with this max graph
            try self.inc_ctx.reserveGallocr(reserve_step.graph);

            // Restore original KV cache lengths (setKv in buildGraph would have incremented them)
            for (self.kv_cache_mgr.layers, 0..) |*layer, i| {
                layer.current_len = saved_lens[i];
            }

            // DO NOT call inc_ctx.resetFull() here!
            // The gallocr reservation is tied to tensor pointers in the context.
            // resetFull() would destroy these pointers, causing infinite realloc.
            // The cached_input tensor from beginStep() above survives and will be
            // reused by subsequent beginStep() calls (cache_valid == true).
        }

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
                pp_time_s,
                pp_speed,
                tg_time_s,
                tg_speed,
                total_time_s,
                avg_speed,
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
                    self.model.resetSSMStates();
                    try stdout.writeStreamingAll(io, "KV cache and SSM states reset.\n");
                } else if (std.mem.eql(u8, line, "/new")) {
                    chat_history.clearAndFree(self.allocator);
                    self.kv_cache_mgr.reset();
                    self.model.resetSSMStates();
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
                    // Add user message with image media to chat history
                    const img_media = chat_template.Media{
                        .type = .image,
                        .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } },
                    };
                    try chat_history.append(self.allocator, chat_template.ChatMessage.withMedia("user", img_prompt, img_media));
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
                    // Add user message with audio media to chat history
                    const aud_media = chat_template.Media{
                        .type = .audio,
                        .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 0 } },
                    };
                    try chat_history.append(self.allocator, chat_template.ChatMessage.withMedia("user", aud_prompt, aud_media));
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
            self.model.resetSSMStates();

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

        const target_size: u32 = if (mm_mgr.vision_encoder) |enc| enc.params.image_size else 896;
        logger.info("Vision encoder target size: {d}x{d}", .{ target_size, target_size });
        var img = try preprocess.loadImage(self.allocator, io, image_path, target_size, .auto);
        defer img.deinit();
        // Validate image has content
        if (img.width == 0 or img.height == 0) {
            logger.err("Loaded image has zero dimensions", .{});
            return error.EmptyImage;
        }
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
        const vision_patch_size: u32 = if (mm_mgr.vision_encoder) |enc| enc.params.patch_size else 14;
        const vision_n_merge: u32 = if (mm_mgr.vision_encoder) |enc| enc.params.n_merge else 2;
        const expected_tokens: i32 = @intCast(@divTrunc(
            (@as(u32, @intCast(@divTrunc(target_size, vision_patch_size))) * @as(u32, @intCast(@divTrunc(target_size, vision_patch_size)))),
            vision_n_merge * vision_n_merge,
        ));
        if (n_vision_tokens != expected_tokens) {
            logger.info("Vision tokens={d} (expected ~{d} for {d}x{d}, patch={d}, merge={d}); mmproj may use different config", .{ n_vision_tokens, expected_tokens, target_size, target_size, vision_patch_size, vision_n_merge });
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
            // Use applyChatTemplateWithMedia which internally uses resolve()
            // (detectKind → preset for known templates like gemma4, never dead loops)
            const media = chat_template.Media{
                .type = .image,
                .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } },
            };
            break :blk try self.applyChatTemplateWithMedia(content_with_placeholder, media);
        };
        defer self.allocator.free(formatted_prompt);

        logger.info("Formatted prompt ({d} chars):\n{s}", .{ formatted_prompt.len, formatted_prompt });
        logger.debug("Template preview: {s}", .{formatted_prompt[0..@min(formatted_prompt.len, 256)]});

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

        const n_total_tokens: i32 = @intCast(expanded.tokens.items.len);
        logger.info("Total tokens for vision prefill: {d} (including {d} vision tokens)", .{ n_total_tokens, n_vision_tokens });
        // ===============================================================
        // Three-stage multimodal prefill (matches llama.cpp behavior):
        //   Pass 1: text prefix tokens (causal attention)
        //   Pass 2: media tokens with vision embeddings (non-causal, per-layer embd)
        //   Pass 3: text suffix tokens (causal, sampled for first token)
        // ===============================================================
        self.model.resetSSMStates();

        // Split token sequence into prefix / media / suffix using placeholder offset info
        const vision_offset: u32 = if (expanded.offsets.len > 0)
            expanded.offsets[0].token_offset
        else
            0;
        const vision_count: u32 = if (expanded.offsets.len > 0)
            expanded.offsets[0].token_count
        else
            @intCast(n_vision_tokens);
        logger.info("Three-stage prefill: prefix={d}, media={d} (offset={d}), suffix={d}", .{
            vision_offset,
            vision_count,
            vision_offset,
            @as(u32, @intCast(n_total_tokens)) - vision_offset - vision_count,
        });

        const prefix_tokens = if (vision_offset > 0)
            expanded.tokens.items[0..vision_offset]
        else
            &[_]u32{};
        const suffix_start: u32 = vision_offset + vision_count;
        const suffix_tokens = if (suffix_start < n_total_tokens)
            expanded.tokens.items[suffix_start..@as(usize, @intCast(n_total_tokens))]
        else
            &[_]u32{};

        // Media forward adapter: converts Gemma4Model.mediaForward to MediaForwardFn
        const mediaForwardFn = struct {
            fn forward(
                model_ptr: *anyopaque,
                fwd_ctx: *ggml.Context,
                fwd_graph: *ggml.CGraph,
                fwd_input_tokens: *ggml.Tensor,
                fwd_n_tokens: i32,
                fwd_cache: ?*kv_cache.KVCache,
                fwd_start_pos: i32,
                fwd_embd_override: *ggml.Tensor,
                fwd_embd_offset: i32,
                fwd_causal: bool,
            ) anyerror!*ggml.Tensor {
                const m: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(model_ptr));
                return m.mediaForward(fwd_ctx, fwd_graph, fwd_input_tokens, fwd_n_tokens, fwd_cache, fwd_start_pos, fwd_embd_override, fwd_embd_offset, fwd_causal);
            }
        }.forward;

        // Copy vision embeddings data to heap so it survives context reset
        const vision_embd_raw = vision_embeddings.dataF32();
        const vision_embd_dim: u32 = @intCast(vision_embeddings.ne()[0]);
        const vision_embd_heap = try self.allocator.dupe(f32, vision_embd_raw);
        defer self.allocator.free(vision_embd_heap);

        const prefill_result = try prefill_mod.threeStagePrefill(
            self.ctx_graph,
            self.model,
            @ptrCast(@alignCast(gemma4_model)),
            &mediaForwardFn,
            &self.kv_cache_mgr,
            prefix_tokens,
            image_token_id,
            @intCast(vision_count),
            vision_embd_heap,
            vision_embd_dim,
            suffix_tokens,
            &self.params,
            self.n_threads,
            self.allocator,
        );

        const pp_time_s = prefill_result.pp_time_s;
        // Greedy sample from heap logits
        var best_idx: i32 = 0;
        var best_val: f32 = prefill_result.logits[0];
        for (prefill_result.logits, 0..) |val, j| {
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(j);
            }
        }
        self.allocator.free(prefill_result.logits);
        var current_token: i32 = best_idx;
        var pos: i32 = prefill_result.pos;
        var gen_count: u32 = 0;

        // Buffer for EOG text detection (accumulates decoded output)
        var eog_detect_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
        defer eog_detect_buf.deinit(self.allocator);

        logger.info("First generated token (vision): id={d}, is_eog={}, pp_time={d:.3}s", .{
            current_token,
            self.tok.isEog(@intCast(current_token)),
            pp_time_s,
        });

        const t_tg_start = engine_common.currentTimeMs();

        // --- Pre-reserve inc_ctx gallocr with worst-case graph (vision) ---
        {
            const saved_lens = try self.kv_cache_mgr.getAllLengths(self.allocator);
            defer self.allocator.free(saved_lens);

            const max_pos: u32 = self.kv_cache_mgr.max_seq_len -| 1;
            self.kv_cache_mgr.setAllLengths(max_pos);

            const reserve_step = try self.inc_ctx.beginStep();
            reserve_step.setToken(0);
            var reserve_builder = graph_builder.GraphBuilder.init(
                reserve_step.ctx, reserve_step.graph, &self.params, self.allocator,
            );
            _ = try self.model.buildGraph(
                &reserve_builder, reserve_step.input_token, 1,
                @ptrCast(&self.kv_cache_mgr), @intCast(max_pos),
            );
            try self.inc_ctx.reserveGallocr(reserve_step.graph);

            for (self.kv_cache_mgr.layers, 0..) |*layer, i| {
                layer.current_len = saved_lens[i];
            }
        }

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
        // Validate audio has content
        if (wav_samples.len == 0) {
            logger.err("Loaded audio file is empty (no samples)", .{});
            return error.EmptyAudio;
        }
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
            // Use applyChatTemplateWithMedia which internally uses resolve()
            // (detectKind → preset for known templates like gemma4, never dead loops)
            const media = chat_template.Media{
                .type = .audio,
                .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 0 } },
            };
            break :blk try self.applyChatTemplateWithMedia(content_with_placeholder, media);
        };
        defer self.allocator.free(formatted_prompt);

        logger.info("Formatted prompt ({d} chars):\n{s}", .{ formatted_prompt.len, formatted_prompt });
        logger.debug("Template preview: {s}", .{formatted_prompt[0..@min(formatted_prompt.len, 256)]});
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

        const n_total_tokens: i32 = @intCast(expanded.tokens.items.len);
        logger.info("Total tokens for audio prefill: {d} (including {d} audio tokens)", .{ n_total_tokens, n_audio_tokens });

        // ===============================================================
        // Three-stage multimodal prefill (matches llama.cpp behavior):
        //   Pass 1: text prefix tokens (causal attention)
        //   Pass 2: media tokens with audio embeddings (non-causal, per-layer embd)
        //   Pass 3: text suffix tokens (causal, sampled for first token)
        // ===============================================================
        self.model.resetSSMStates();

        // Split token sequence into prefix / media / suffix using placeholder offset info
        const audio_offset: u32 = if (expanded.offsets.len > 0)
            expanded.offsets[0].token_offset
        else
            0;
        const audio_count: u32 = if (expanded.offsets.len > 0)
            expanded.offsets[0].token_count
        else
            @intCast(n_audio_tokens);
        logger.info("Three-stage prefill: prefix={d}, media={d} (offset={d}), suffix={d}", .{
            audio_offset,
            audio_count,
            audio_offset,
            @as(u32, @intCast(n_total_tokens)) - audio_offset - audio_count,
        });

        const prefix_tokens = if (audio_offset > 0)
            expanded.tokens.items[0..audio_offset]
        else
            &[_]u32{};
        const suffix_start: u32 = audio_offset + audio_count;
        const suffix_tokens = if (suffix_start < n_total_tokens)
            expanded.tokens.items[suffix_start..@as(usize, @intCast(n_total_tokens))]
        else
            &[_]u32{};

        // Media forward adapter: converts Gemma4Model.mediaForward to MediaForwardFn
        const mediaForwardFn = struct {
            fn forward(
                model_ptr: *anyopaque,
                fwd_ctx: *ggml.Context,
                fwd_graph: *ggml.CGraph,
                fwd_input_tokens: *ggml.Tensor,
                fwd_n_tokens: i32,
                fwd_cache: ?*kv_cache.KVCache,
                fwd_start_pos: i32,
                fwd_embd_override: *ggml.Tensor,
                fwd_embd_offset: i32,
                fwd_causal: bool,
            ) anyerror!*ggml.Tensor {
                const m: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(model_ptr));
                return m.mediaForward(fwd_ctx, fwd_graph, fwd_input_tokens, fwd_n_tokens, fwd_cache, fwd_start_pos, fwd_embd_override, fwd_embd_offset, fwd_causal);
            }
        }.forward;

        // Copy audio embeddings data to heap so it survives context reset
        const audio_embd_raw = audio_embeddings.dataF32();
        const audio_embd_dim: u32 = @intCast(audio_embeddings.ne()[0]);
        const audio_embd_heap = try self.allocator.dupe(f32, audio_embd_raw);
        defer self.allocator.free(audio_embd_heap);

        const prefill_result = try prefill_mod.threeStagePrefill(
            self.ctx_graph,
            self.model,
            @ptrCast(@alignCast(gemma4_model)),
            &mediaForwardFn,
            &self.kv_cache_mgr,
            prefix_tokens,
            audio_token_id,
            @intCast(audio_count),
            audio_embd_heap,
            audio_embd_dim,
            suffix_tokens,
            &self.params,
            self.n_threads,
            self.allocator,
        );

        const pp_time_s = prefill_result.pp_time_s;
        // Greedy sample from heap logits
        var best_idx: i32 = 0;
        var best_val: f32 = prefill_result.logits[0];
        for (prefill_result.logits, 0..) |val, j| {
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(j);
            }
        }
        self.allocator.free(prefill_result.logits);
        var current_token: i32 = best_idx;
        var pos: i32 = prefill_result.pos;
        var gen_count: u32 = 0;

        // Buffer for EOG text detection (accumulates decoded output)
        var eog_detect_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
        defer eog_detect_buf.deinit(self.allocator);

        logger.info("First generated token (audio): id={d}, is_eog={}, pp_time={d:.3}s", .{
            current_token,
            self.tok.isEog(@intCast(current_token)),
            pp_time_s,
        });
        const t_tg_start = engine_common.currentTimeMs();

        // --- Pre-reserve inc_ctx gallocr with worst-case graph (audio) ---
        {
            const saved_lens = try self.kv_cache_mgr.getAllLengths(self.allocator);
            defer self.allocator.free(saved_lens);

            const max_pos: u32 = self.kv_cache_mgr.max_seq_len -| 1;
            self.kv_cache_mgr.setAllLengths(max_pos);

            const reserve_step = try self.inc_ctx.beginStep();
            reserve_step.setToken(0);
            var reserve_builder = graph_builder.GraphBuilder.init(
                reserve_step.ctx, reserve_step.graph, &self.params, self.allocator,
            );
            _ = try self.model.buildGraph(
                &reserve_builder, reserve_step.input_token, 1,
                @ptrCast(&self.kv_cache_mgr), @intCast(max_pos),
            );
            try self.inc_ctx.reserveGallocr(reserve_step.graph);

            for (self.kv_cache_mgr.layers, 0..) |*layer, i| {
                layer.current_len = saved_lens[i];
            }
        }

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

