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
const engine_common = @import("engine_common");
const chat_template = @import("chat_template");

const logger = std.log.scoped(.simple);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = engine_common.logFilter,
};

fn printUsage(argv0: []const u8) void {
    std.debug.print(
        \\Usage: {s} [options] [prompt]
        \\
        \\Options:
        \\  -h, --help             Show this help
        \\  -m, --model <path>     Model file path (GGUF)
        \\  -n, --max-tokens <N>   Max tokens to generate (default: 32)
        \\  -t, --temperature <T>  Temperature (default: 0.7)
        \\  -k, --top-k <K>        Top-K sampling (default: 40)
        \\  -tp, --top-p <P>       Top-P sampling (default: 0.9)
        \\  -th, --threads <N>     Number of threads (default: auto)
        \\  -v, --verbose          Verbose output
        \\  -d, --debug            Debug output
        \\  --benchmark            Benchmark mode
        \\  --info                 Print model info and exit
        \\
    , .{argv0});
}

const CliArgs = struct {
    model_path: []const u8 = "",
    prompt: []const u8 = "",
    max_tokens: u32 = 32,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    n_threads: i32 = 0,
    verbose: bool = false,
    debug: bool = false,
    help: bool = false,
    benchmark: bool = false,
    info: bool = false,
    // Chat template
    chat_template_name: []const u8 = "",
    system_prompt: []const u8 = "",
    no_chat_template: bool = false,
    no_jinja: bool = false,

    pub fn parse(args_it: *std.process.Args.Iterator) !CliArgs {
        var result = CliArgs{};
        const argv0 = args_it.next() orelse "zllama-simple";
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                result.help = true;
            } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
                result.model_path = args_it.next() orelse return error.InvalidArgs;
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
            } else if (std.mem.eql(u8, arg, "--benchmark")) {
                result.benchmark = true;
            } else if (std.mem.startsWith(u8, arg, "-")) {} else if (std.mem.eql(u8, arg, "--info")) {
                result.info = true;
                logger.warn("unknown argument '{s}'", .{arg});
            } else if (std.mem.eql(u8, arg, "--chat-template")) {
                result.chat_template_name = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--system-prompt")) {
                result.system_prompt = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--no-chat-template")) {
                result.no_chat_template = true;
            } else if (std.mem.eql(u8, arg, "--no-jinja")) {
                result.no_jinja = true;
                result.prompt = arg;
            }
        }
        if (result.help) printUsage(argv0);
        return result;
    }
};

const SimpleEngine = struct {
    allocator: std.mem.Allocator,
    ctx_weights: *ggml.Context,
    ctx_graph: *ggml.Context,
    ctx_kv_cache: *ggml.Context,
    arch: model_if.Architecture,
    model: model_if.ModelInstance,
    params: model_if.ModelParams,
    tok: tokenizer.Tokenizer,
    kv_cache_mgr: kv_cache.KVCache,
    n_threads: i32,
    gguf_data: []u8,

    inc_ctx: graph_context.IncContext,
    benchmark: bool,

    // Chat template
    chat_template_source: ?chat_template.TemplateSource = null,
    system_prompt: []const u8 = "",
    no_chat_template: bool = false,
    no_jinja: bool = false,

    pub fn init(io: std.Io, allocator: std.mem.Allocator, model_path: []const u8, cli_args: *const CliArgs) !SimpleEngine {
        logger.info("Loading model: {s}", .{model_path});
        const cwd = std.Io.Dir.cwd();
        logger.info("Opening file...", .{});
        const file = try cwd.openFile(io, model_path, .{ .mode = .read_only });
        defer file.close(io);
        logger.info("File opened, getting stat...", .{});
        const stat = try file.stat(io);
        const file_size = @as(usize, @intCast(stat.size));
        logger.info("File size: {d} bytes", .{file_size});
        const gguf_data = try allocator.alloc(u8, file_size);
        errdefer allocator.free(gguf_data);
        logger.info("Reading file... ({d} MB)", .{@divFloor(file_size, 1024 * 1024)});
        {
            var offset: u64 = 0;
            const chunk_size: usize = 64 * 1024 * 1024; // 64MB chunks
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
        logger.info("Parsing GGUF...", .{});
        var gguf_file = try gguf.parse(gguf_data, allocator);
        defer gguf_file.deinit();
        logger.info("GGUF parsed, detecting architecture...", .{});
        const arch = registry.detectArchitecture(&gguf_file) orelse return error.UnsupportedArchitecture;
        logger.info("Detected architecture: {s}", .{@tagName(arch)});
        logger.info("Creating model...", .{});
        var model = try registry.createModel(allocator, &gguf_file, arch, io);
        logger.info("Model created, getting params...", .{});
        var params = model.getParams().*;

        // Copy model_name string from GGUF arena to heap (arena will be freed by gguf_file.deinit)
        if (params.model_name.len > 0) {
            params.model_name = try allocator.dupe(u8, params.model_name);
        }

        // Detect and log model capabilities
        const capabilities = registry.detectCapabilities(&gguf_file, arch);
        if (capabilities.has_vision or capabilities.has_audio) {
            logger.info("Multi-modal: yes", .{});
        } else {
            logger.info("Multi-modal: no (text-only)", .{});
        }
        logger.info("  Text  : yes", .{});
        if (capabilities.has_vision) {
            logger.info("  Vision: yes ({s})", .{capabilities.vision_encoder_type});
        } else {
            logger.info("  Vision: no", .{});
        }
        if (capabilities.has_audio) {
            logger.info("  Audio : yes ({s}, {d} Hz)", .{ capabilities.audio_encoder_type, capabilities.audio_sample_rate });
        } else {
            logger.info("  Audio : no", .{});
        }
        var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
        errdefer tok.deinit();
        logger.info("Tokenizer initialized.", .{});
        const n_threads = if (cli_args.n_threads > 0) cli_args.n_threads else ggml.recommendedThreads();
        // ctx_weights: NOT used for model weight loading (each model allocates its own).
        // Keep a small placeholder context for potential future use.
        const ctx_weights = try ggml.Context.initNoAlloc(16 * 1024 * 1024);
        errdefer ctx_weights.deinit();
        // ctx_kv_cache: sized for KV cache tensors per layer
        const max_seq_len = @min(params.max_seq_len, 2048);
        const hdim_kv = params.n_head_dim;
        const hdim_k = @max(params.n_head_dim, params.n_head_dim_k);
        const hdim_v = if (params.n_head_dim_v > 0) @max(params.n_head_dim, params.n_head_dim_v) else hdim_kv;
        // KV cache: n_layer * 2 * max_seq_len * n_kv_head * head_dim * sizeof(f32), plus 25% buffer
        const kv_cache_bytes = @as(usize, @intCast(params.n_layer)) * 2 * @as(usize, @intCast(max_seq_len)) * @as(usize, @intCast(params.n_kv_head)) * @as(usize, @intCast(hdim_kv)) * 4;
        const kv_cache_mem = @max(256 * 1024 * 1024, kv_cache_bytes + kv_cache_bytes / 4);
        const ctx_kv_cache = try ggml.Context.initNoAlloc(kv_cache_mem);
        errdefer ctx_kv_cache.deinit();
        var kv_cache_mgr = try kv_cache.KVCache.initWithKVDim(ctx_kv_cache, params.n_layer, params.n_kv_head, hdim_k, hdim_v, max_seq_len, allocator);
        errdefer kv_cache_mgr.deinit(allocator);
        // Set Qwen35's ctx_kv_cache (for persistent SSM states)
        model.setKVCacheContext(ctx_kv_cache);
        // ctx_graph: for compute graph building during prompt processing
        const ctx_graph = try ggml.Context.initNoAlloc(512 * 1024 * 1024);
        errdefer ctx_graph.deinit();
        {
            const b = ggml.backendCpuBufferType();
            try ggml.backendAllocCtxTensorsFromBuft(ctx_kv_cache, b);
        }

        // Create incremental decoding context with graph structure reuse
        const inc_ctx_size = 512 * 1024 * 1024; // 512MB for incremental
        const inc_ctx = try graph_context.IncContext.init(allocator, &params, inc_ctx_size);

        // Resolve chat template source
        var chat_template_source: ?chat_template.TemplateSource = null;
        var system_prompt: []const u8 = "";
        if (cli_args.system_prompt.len > 0) {
            system_prompt = try allocator.dupe(u8, cli_args.system_prompt);
        }
        if (cli_args.no_chat_template) {
            // Disabled - no template
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

        return SimpleEngine{
            .allocator = allocator,
            .ctx_weights = ctx_weights,
            .ctx_graph = ctx_graph,
            .ctx_kv_cache = ctx_kv_cache,
            .arch = arch,
            .model = model,
            .params = params,
            .tok = tok,
            .kv_cache_mgr = kv_cache_mgr,
            .n_threads = n_threads,
            .gguf_data = gguf_data,
            .inc_ctx = inc_ctx,
            .benchmark = cli_args.benchmark,
            .chat_template_source = chat_template_source,
            .system_prompt = system_prompt,
            .no_chat_template = cli_args.no_chat_template,
            .no_jinja = cli_args.no_jinja,
        };
    }

    pub fn deinit(self: *SimpleEngine) void {
        self.inc_ctx.deinit();
        self.kv_cache_mgr.deinit(self.allocator);
        self.tok.deinit();
        self.ctx_graph.deinit();
        self.ctx_kv_cache.deinit();
        // ctx_weights freed by model.deinit()
        self.model.deinit(self.allocator);
        if (self.params.model_name.len > 0) {
            self.allocator.free(self.params.model_name);
        }
        self.allocator.free(self.gguf_data);
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

    // ========================================================================
    // Public chat template API
    // ========================================================================

    /// Apply chat template for a single-turn user prompt.
    /// Returns the formatted prompt (caller owns memory).
    pub fn applyChatTemplate(self: *SimpleEngine, user_prompt: []const u8) ![]const u8 {
        // If --no-chat-template is set, pass through raw prompt
        if (self.no_chat_template) {
            return self.allocator.dupe(u8, user_prompt);
        }

        // Resolve template source: use GGUF built-in, --chat-template preset,
        // or fall back to architecture default (e.g. ChatML for Qwen)
        const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
        const source = self.chat_template_source orelse
            chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };

        var tmpl = try chat_template.resolve(self.allocator, source, self.arch, model_name, !self.no_jinja);
        defer tmpl.deinit(self.allocator);

        const messages = [_]chat_template.ChatMessage{
            chat_template.ChatMessage.init("user", user_prompt),
        };

        const system = if (self.system_prompt.len > 0) self.system_prompt else null;
        return tmpl.apply(self.allocator, &messages, system, true);
    }

    /// Apply chat template for multi-turn conversation (chat history).
    /// The last message in chat_history should be the new user message.
    /// Returns the formatted prompt (caller owns memory).
    pub fn applyChatTemplateMultiTurn(self: *SimpleEngine, chat_history: []const chat_template.ChatMessage) ![]const u8 {
        if (self.no_chat_template) {
            if (chat_history.len == 0) return self.allocator.dupe(u8, "");
            return self.allocator.dupe(u8, chat_history[chat_history.len - 1].content);
        }
        const model_name: ?[]const u8 = if (self.params.model_name.len > 0) self.params.model_name else null;
        const source = self.chat_template_source orelse
            chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(self.arch, model_name) };
        var tmpl = try chat_template.resolve(self.allocator, source, self.arch, model_name, !self.no_jinja);
        defer tmpl.deinit(self.allocator);
        const system = if (self.system_prompt.len > 0) self.system_prompt else null;
        return tmpl.apply(self.allocator, chat_history, system, true);
    }

    /// Decode a single token and write to stdout
    fn decodeAndPrintToken(self: *SimpleEngine, io: std.Io, token_id: u32) !void {
        var buf: [128]u8 = undefined;
        const n = try self.tok.decodeSingle(token_id, &buf);
        if (n > 0) {
            const stdout_file = std.Io.File.stdout();
            try stdout_file.writeStreamingAll(io, buf[0..n]);
        }
    }
    pub fn generate(self: *SimpleEngine, io: std.Io, prompt: []const u8, max_tokens: u32) !void {
        // Apply chat template
        const formatted_prompt = try self.applyChatTemplate(prompt);
        defer self.allocator.free(formatted_prompt);

        var input_tokens = try self.tok.encode(formatted_prompt, true, true);
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
            std.debug.print("Error: graph alloc failed\n", .{});
            return error.GraphAllocFailed;
        }

        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_prompt_tokens))];
            for (input_tokens.items, 0..) |token, j| slice[j] = @as(i32, @intCast(token));
        }

        // Prompt evaluation timing
        const t_pp_start = engine_common.currentTimeUs();
        try graph.compute(self.n_threads);
        const first_token = sampler.Sampler.sampleGreedy(logits);
        const t_pp_end = engine_common.currentTimeUs();
        const pp_time_s = @as(f64, @floatFromInt(t_pp_end - t_pp_start)) / 1000000.0;
        logger.debug("first_token (greedy) = {d}", .{first_token});

        // Print prompt (skip in benchmark mode)
        if (!self.benchmark) {
            for (input_tokens.items) |token_id| {
                try self.decodeAndPrintToken(io, @intCast(token_id));
            }
        }

        var n_decode: i32 = 0;
        var new_token_id: i32 = first_token;
        var pos: i32 = n_prompt_tokens;

        // --- Pre-reserve inc_ctx gallocr with worst-case graph ---
        // Build a graph with position = max_seq_len-1 to cover the widest possible
        // KV cache mask. This eliminates ggml_gallocr_needs_realloc during decode.
        {
            // Save current KV cache lengths
            const saved_lens = try self.kv_cache_mgr.getAllLengths(self.allocator);
            defer self.allocator.free(saved_lens);

            // Set all layer lengths to kv_cache.max_seq_len-1 (the actual allocated cache size,
            // capped at 2048 in init()). After setKv writes one token, current_len becomes
            // max_seq_len (full cache). This ensures the reservation covers the widest mask.
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
            //
            // Context memory grows each decode step but with the standard 512MB
            // allocation, ~5000 steps fit before overflow. If overflow occurs,
            // the caller must re-reserve gallocr after resetFull().
        }
        // Text generation timing
        const t_tg_start = engine_common.currentTimeUs();

        // Buffer for EOG text detection (accumulates decoded output)
        var eog_detect_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
        defer eog_detect_buf.deinit(self.allocator);

        // --- Incremental decoding (uses inc_ctx with graph structure reuse) ---
        while (n_decode < max_tokens) {
            // Check for EOG token
            if (self.tok.isEog(@intCast(new_token_id))) {
                logger.debug("EOG token {d} stopping", .{new_token_id});
                break;
            }

            // Skip control tokens (e.g. <|channel|>, <|channel>) that should
            // be filtered from output but don't stop generation.
            if (self.tok.isSkipToken(@intCast(new_token_id))) {
                // Prepare next batch: IncContext reuses input tensor + graph + gallocr
                const step = try self.inc_ctx.beginStep();
                step.setToken(new_token_id);

                var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
                const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), @intCast(pos));

                if (!step.galloc.allocGraph(step.graph)) {
                    logger.err("Failed to allocate incremental graph memory", .{});
                    return error.GraphAllocFailed;
                }

                try step.graph.compute(self.n_threads);

                new_token_id = sampler.Sampler.sampleGreedy(inc_logits);
                pos += 1;
                n_decode += 1;
                continue;
            }

            // Decode token
            var buf: [128]u8 = undefined;
            const n = try self.tok.decodeSingle(@intCast(new_token_id), &buf);
            const decoded = buf[0..n];

            // Check if decoded text contains EOG token string
            // (handles models that generate EOG token as sub-token sequence)
            if (n > 0) {
                try eog_detect_buf.appendSlice(self.allocator, decoded);
                if (self.tok.isEogText(eog_detect_buf.items)) {
                    logger.debug("EOG text detected, stopping", .{});
                    // Stream the current token before stopping
                    if (!self.benchmark) {
                        const stdout_file = std.Io.File.stdout();
                        try stdout_file.writeStreamingAll(io, decoded);
                    }
                    break;
                }
            }

            // Print output (skip in benchmark mode)
            logger.debug("decoding token {d}", .{new_token_id});
            if (!self.benchmark and n > 0) {
                const stdout_file = std.Io.File.stdout();
                try stdout_file.writeStreamingAll(io, decoded);
            }

            // Prepare next batch: IncContext reuses input tensor + graph + gallocr
            const step = try self.inc_ctx.beginStep();

            // Set input token on pre-allocated tensor
            step.setToken(new_token_id);

            // Build graph using cached context and graph object
            var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, &self.params, self.allocator);
            const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(&self.kv_cache_mgr), pos);

            // Reuse gallocr (avoids expensive graph analysis per token)
            if (!step.galloc.allocGraph(step.graph)) {
                std.debug.print("Error: inc graph alloc failed\n", .{});
                return error.GraphAllocFailed;
            }

            try step.graph.compute(self.n_threads);
            new_token_id = sampler.Sampler.sampleGreedy(inc_logits);
            logger.debug("next token (greedy) = {d}", .{new_token_id});
            pos += 1;
            n_decode += 1;
        }

        const t_tg_end = engine_common.currentTimeUs();
        const tg_time_s = @as(f64, @floatFromInt(t_tg_end - t_tg_start)) / 1000000.0;

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
            const tg_speed = if (tg_time_s > 0.0 and n_decode > 0)
                @as(f64, @floatFromInt(n_decode)) / tg_time_s
            else
                0.0;
            const avg_speed = if (total_time_s > 0.0 and n_decode > 0)
                @as(f64, @floatFromInt(n_decode)) / total_time_s
            else
                0.0;

            std.debug.print(
                \\
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
                n_decode,
                pp_time_s,
                pp_speed,
                tg_time_s,
                tg_speed,
                total_time_s,
                avg_speed,
            });
        } else if (n_decode > 0) {
            const total_time_s = pp_time_s + tg_time_s;
            const avg_speed = @as(f64, @floatFromInt(n_decode)) / total_time_s;
            std.debug.print("main: decoded {d} tokens in {d:.2} s, speed: {d:.2} t/s\n", .{ n_decode, total_time_s, avg_speed });
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();
    const args = CliArgs.parse(&args_iter) catch |err| {
        if (err == error.InvalidArgs) return;
        return err;
    };
    if (args.help) return;
    if (args.debug) {
        engine_common.setLogLevel(.debug);
    }
    if (args.verbose) {
        engine_common.setLogLevel(.info);
    }
    if (args.model_path.len == 0) {
        std.debug.print("Error: no model specified. Use --model <path>\n", .{});
        return;
    }
    logger.info("Loading model: {s}", .{args.model_path});
    var engine = SimpleEngine.init(io, allocator, args.model_path, &args) catch |err| {
        std.debug.print("Error: failed to initialize engine: {}\n", .{err});
        return;
    };
    defer engine.deinit();
    logger.info("Model loaded successfully.", .{});
    const prompt = if (args.prompt.len > 0) args.prompt else "Hello my name is";
    if (!args.benchmark) {
        logger.info("Prompt: \"{s}\"", .{prompt});
        logger.info("Max tokens: {d}", .{args.max_tokens});
    }
    engine.generate(io, prompt, args.max_tokens) catch |err| {
        std.debug.print("Error: generation failed: {}\n", .{err});
        return;
    };
}
