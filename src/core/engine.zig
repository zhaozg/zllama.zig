//! Inference engine — orchestrates model loading, prefill, and incremental decode.
//!
//! Composes three sub-modules:
//!   - `context.zig` (ModelContext): GGUF loading, model init, resource ownership
//!   - `planner.zig` (GraphPlanner): graph building, gallocr reservation
//!   - `executor.zig` (GraphExecutor): graph compute, result extraction
//!
//! This file is the public API surface. Internal complexity is delegated to
//! the three sub-modules, keeping each file ≤600 lines.
//!
//! Reference: llama.cpp llama_context / llama_decode

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_context = @import("graph_context");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const kv_cache = @import("kv_cache");
const engine_common = @import("engine_common");
const chat_template = @import("chat_template");
const decode_mod = @import("decode");
const verbose_mod = @import("verbose");
const embedding_mod = @import("embedding_gen");
const multimodal_mod = @import("multimodal");

const memory_pool = @import("memory_pool");
const memory_monitor = @import("memory_monitor");
const CliArgs = @import("../cli_args.zig").CliArgs;

// Import the three new sub-modules
const ModelContext = @import("context.zig").ModelContext;
const GraphPlanner = @import("planner.zig").GraphPlanner;
const GraphExecutor = @import("executor.zig").GraphExecutor;
const GraphPlan = @import("planner.zig").GraphPlan;
const PrefillResult = @import("planner.zig").PrefillResult;

const logger = std.log.scoped(.core_engine);

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
    /// Owns all model resources (weights, KV cache, graph context, etc.)
    ctx: ModelContext,

    /// Builds compute graphs
    planner: GraphPlanner,

    /// Executes compute graphs
    executor: GraphExecutor,

    /// 内存监控器（监控 KV Cache、Graph、Inc 等 context 的内存使用）
    mem_monitor: memory_monitor.MemoryMonitor,

    // ========================================================================
    // init / deinit
    // ========================================================================

    pub fn init(io: std.Io, allocator: std.mem.Allocator, model_path: [:0]const u8, cli_args: *const CliArgs) !InferenceEngine {
        var ctx = try ModelContext.init(io, allocator, model_path, cli_args);
        errdefer ctx.deinit();

        const planner_inst = GraphPlanner.init();

        const executor_inst = GraphExecutor.init(
            ctx.n_threads,
            @as(usize, @intCast(ctx.params.n_vocab)),
        );

        // 初始化内存监控器
        // NOTE: We only register ctx_graph and ctx_kv_cache here because they are
        // *ggml.Context pointers (heap-allocated) that survive the struct move.
        // The IncContext adapter (&ctx.inc_ctx) points to a field within the local
        // variable `ctx`, which becomes dangling after the return. We register it
        // in postInit() after the engine is in its final memory location.
        var mem_monitor_inst = memory_monitor.MemoryMonitor.init(allocator);
        try mem_monitor_inst.addContext(memory_monitor.adaptGgmlContext(ctx.ctx_graph, "graph"));
        try mem_monitor_inst.addContext(memory_monitor.adaptGgmlContext(ctx.ctx_kv_cache, "kv_cache"));

        return InferenceEngine{
            .ctx = ctx,
            .planner = planner_inst,
            .executor = executor_inst,
            .mem_monitor = mem_monitor_inst,
        };
    }

    /// Post-initialization: must be called once after init(), before any
    /// generate/chat calls. Registers the IncContext adapter with the memory
    /// monitor. This is a separate step because &self.ctx.inc_ctx is only
    /// valid after the engine struct is in its final memory location.
    pub fn postInit(self: *InferenceEngine) !void {
        try self.mem_monitor.addContext(memory_monitor.adaptIncContext(&self.ctx.inc_ctx));
    }

    pub fn deinit(self: *InferenceEngine) void {
        self.mem_monitor.deinit();
        self.planner.deinit();
        self.ctx.deinit();
    }

    // ========================================================================
    // Delegated properties (thin wrappers over ModelContext)
    // ========================================================================

    pub fn getArch(self: *const InferenceEngine) model_if.Architecture {
        return self.ctx.arch;
    }

    pub fn getParams(self: *const InferenceEngine) *const model_if.ModelParams {
        return &self.ctx.params;
    }

    pub fn getModel(self: *const InferenceEngine) model_if.ModelInstance {
        return self.ctx.model;
    }

    pub fn getTokenizer(self: *const InferenceEngine) *tokenizer.Tokenizer {
        return self.ctx.tok;
    }

    pub fn getKVCache(self: *const InferenceEngine) *kv_cache.KVCache {
        return &self.ctx.kv_cache_mgr;
    }

    pub fn getIncContext(self: *const InferenceEngine) *graph_context.IncContext {
        return &self.ctx.inc_ctx;
    }

    pub fn getGallocr(self: *const InferenceEngine) *ggml.Gallocr {
        return self.ctx.gallocr;
    }

    // ========================================================================
    // Chat template API (delegated to ModelContext)
    // ========================================================================

    pub fn applyChatTemplate(self: *InferenceEngine, user_prompt: []const u8) ![]const u8 {
        return self.ctx.applyChatTemplate(user_prompt);
    }

    pub fn applyChatTemplateWithMedia(self: *InferenceEngine, user_prompt: []const u8, media: ?chat_template.Media) ![]const u8 {
        return self.ctx.applyChatTemplateWithMedia(user_prompt, media);
    }

    pub fn applyChatTemplateMultiTurn(self: *InferenceEngine, chat_history: []const chat_template.ChatMessage) ![]const u8 {
        return self.ctx.applyChatTemplateMultiTurn(chat_history);
    }

    // ========================================================================
    // Text prefill
    // ========================================================================

    fn textPrefill(self: *InferenceEngine, input_tokens: []const u32) !PrefillResult {
        const plan = try self.planner.planPrefill(
            self.ctx.ctx_graph,
            self.ctx.model,
            &self.ctx.params,
            &self.ctx.kv_cache_mgr,
            self.ctx.allocator,
            input_tokens,
            self.ctx.gallocr,
        );
        return try self.executor.executePrefill(&plan, self.ctx.allocator);
    }

    // ========================================================================
    // Stream prompt tokens (only when verbose_prompt is enabled)
    // ========================================================================

    fn streamPromptTokens(self: *InferenceEngine, io: std.Io, tokens: []const u32) !void {
        if (!self.ctx.verbose_prompt or self.ctx.benchmark) return;
        const stdout_file = std.Io.File.stdout();
        for (tokens) |token_id| {
            var buf: [128]u8 = undefined;
            const n = try self.ctx.tok.decodeSingle(token_id, &buf);
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
        const formatted_prompt = try self.ctx.applyChatTemplate(prompt);
        defer self.ctx.allocator.free(formatted_prompt);

        var input_tokens = try self.ctx.tok.encode(formatted_prompt, true, true);
        defer input_tokens.deinit(self.ctx.allocator);

        const n_prompt_tokens: i32 = @intCast(input_tokens.items.len);
        self.ctx.model.resetSSMStates();

        const prefill = try self.textPrefill(input_tokens.items);
        defer self.ctx.allocator.free(prefill.logits);

        if (self.ctx.verbose_prompt) {
            try verbose_mod.printVerbosePrompt(io, self.ctx.allocator, self.ctx.tok, formatted_prompt, input_tokens.items, prefill.logits, null);
        }

        const first_token = sampler.Sampler.sampleGreedyFromLogits(prefill.logits);
        try self.streamPromptTokens(io, input_tokens.items);

        try decode_mod.reserveDecodeGallocr(
            self.ctx.allocator,
            &self.ctx.kv_cache_mgr,
            &self.ctx.inc_ctx,
            self.ctx.model,
            &self.ctx.params,
        );

        // 内存监控：prefill 后检查内存使用
        const prefill_report = try self.mem_monitor.check();
        defer self.ctx.allocator.free(prefill_report.contexts);
        if (prefill_report.max_alert != .normal) {
            logger.warn("Prefill memory: {d:.1}% used ({s})", .{
                prefill_report.total_ratio * 100,
                @tagName(prefill_report.max_alert),
            });
        }

        const dr = try decode_mod.runDecodeLoop(
            self.ctx.allocator,
            io,
            self.ctx.model,
            &self.ctx.params,
            self.ctx.tok,
            &self.ctx.kv_cache_mgr,
            &self.ctx.inc_ctx,
            self.ctx.n_threads,
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
                        return (@as(*InferenceEngine, @ptrCast(@alignCast(c)))).ctx.tok.isSkipToken(@intCast(t));
                    }
                }.f,
                .onComplete = null,
            },
            @ptrCast(self),
            self.ctx.benchmark,
        );

        const stdout_file = std.Io.File.stdout();
        try stdout_file.writeStreamingAll(io, "\n");
        printStats(self.ctx.arch, self.ctx.params.model_name, self.ctx.n_threads, n_prompt_tokens, dr.gen_count, prefill.pp_time_s, dr.tg_time_s, self.ctx.benchmark);

        // 内存监控：decode 后检查内存使用
        const decode_report = try self.mem_monitor.check();
        defer self.ctx.allocator.free(decode_report.contexts);
        if (decode_report.max_alert != .normal) {
            logger.warn("Decode memory: {d:.1}% used ({s})", .{
                decode_report.total_ratio * 100,
                @tagName(decode_report.max_alert),
            });
        }

        if (!self.ctx.benchmark) {
            try stdout_file.writeStreamingAll(io, "\n");
        }
    }

    // ========================================================================
    // Embedding generation (public API)
    // ========================================================================

    pub fn generateEmbedding(self: *InferenceEngine, io: std.Io, prompt: []const u8) ![]f32 {
        return embedding_mod.generateEmbedding(
            self.ctx.allocator,
            io,
            self.ctx.model,
            &self.ctx.params,
            self.ctx.tok,
            self.ctx.ctx_graph,
            self.ctx.n_threads,
            prompt,
        );
    }

    // ========================================================================
    // Chat loop (public API) — multi-turn conversation with KV cache accumulation
    // ========================================================================

    pub fn chatLoop(self: *InferenceEngine, io: std.Io) !void {
        const stdin = std.Io.File.stdin();
        const stdout = std.Io.File.stdout();
        try stdout.writeStreamingAll(io, "zllama chat - Interactive AI\n");
        try stdout.writeStreamingAll(io, "Type /help for commands.\n\n");

        var line_buf: [4096]u8 = undefined;
        var chat_history = std.ArrayListUnmanaged(chat_template.ChatMessage){ .items = &.{}, .capacity = 0 };
        defer chat_history.deinit(self.ctx.allocator);

        // Track whether this is the first turn (need full prefill) or subsequent (incremental decode)
        var is_first_turn = true;

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

            if (line[0] == '/') {
                if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) {
                    try stdout.writeStreamingAll(io, "Bye.\n");
                    break;
                } else if (std.mem.eql(u8, line, "/help")) {
                    try stdout.writeStreamingAll(io, "Available commands:\n  /help  /clear  /exit  /reset  /new\n");
                } else if (std.mem.eql(u8, line, "/clear")) {
                    try stdout.writeStreamingAll(io, "\x1b[2J\x1b[H");
                } else if (std.mem.eql(u8, line, "/reset")) {
                    self.ctx.kv_cache_mgr.reset();
                    self.ctx.model.resetSSMStates();
                    is_first_turn = true;
                    try stdout.writeStreamingAll(io, "KV cache and SSM states reset.\n");
                } else if (std.mem.eql(u8, line, "/new")) {
                    chat_history.clearAndFree(self.ctx.allocator);
                    self.ctx.kv_cache_mgr.reset();
                    self.ctx.model.resetSSMStates();
                    is_first_turn = true;
                    try stdout.writeStreamingAll(io, "New conversation started.\n");
                } else {
                    try stdout.writeStreamingAll(io, "Unknown command. Try /help.\n");
                }
                continue;
            }

            try chat_history.append(self.ctx.allocator, chat_template.ChatMessage.init("user", line));
            const formatted_prompt = try self.ctx.applyChatTemplateMultiTurn(chat_history.items);
            defer self.ctx.allocator.free(formatted_prompt);

            if (is_first_turn) {
                // First turn: full prefill of the entire prompt
                self.ctx.kv_cache_mgr.reset();
                self.ctx.model.resetSSMStates();
                is_first_turn = false;
            } else {
                // Subsequent turns: only prefill the new user input (the chat template
                // formats the full history, but we only encode the delta).
                // NOTE: For simplicity, we still prefill the full formatted prompt.
                // A future optimization would encode only the new tokens and append
                // them to the existing KV cache via incremental decode.
                // For now, reset KV cache to avoid context length explosion from
                // repeated prefill of the full history.
                self.ctx.kv_cache_mgr.reset();
                self.ctx.model.resetSSMStates();
            }

            var input_tokens = try self.ctx.tok.encode(formatted_prompt, true, true);
            defer input_tokens.deinit(self.ctx.allocator);

            const prefill = try self.textPrefill(input_tokens.items);
            defer self.ctx.allocator.free(prefill.logits);

            // 内存监控：chat prefill 后检查
            const prefill_report = try self.mem_monitor.check();
            defer self.ctx.allocator.free(prefill_report.contexts);
            if (prefill_report.max_alert != .normal) {
                logger.warn("Chat prefill memory: {d:.1}% used ({s})", .{
                    prefill_report.total_ratio * 100,
                    @tagName(prefill_report.max_alert),
                });
            }

            try decode_mod.reserveDecodeGallocr(
                self.ctx.allocator,
                &self.ctx.kv_cache_mgr,
                &self.ctx.inc_ctx,
                self.ctx.model,
                &self.ctx.params,
            );

            // Collect output tokens via afterToken callback for building assistant message.
            var output_tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
            defer output_tokens.deinit(self.ctx.allocator);
            var cb_ctx = ChatTokenCollector{ .tokens = &output_tokens, .allocator = self.ctx.allocator };

            const dr = try decode_mod.runDecodeLoop(
                self.ctx.allocator,
                io,
                self.ctx.model,
                &self.ctx.params,
                self.ctx.tok,
                &self.ctx.kv_cache_mgr,
                &self.ctx.inc_ctx,
                self.ctx.n_threads,
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
                            return (@as(*InferenceEngine, @ptrCast(@alignCast(c)))).ctx.tok.isSkipToken(@intCast(t));
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
                defer response_buf.deinit(self.ctx.allocator);
                for (output_tokens.items) |tid| {
                    var buf: [128]u8 = undefined;
                    const n = try self.ctx.tok.decodeSingle(tid, &buf);
                    if (n > 0) try response_buf.appendSlice(self.ctx.allocator, buf[0..n]);
                }
                try chat_history.append(self.ctx.allocator, chat_template.ChatMessage.init("assistant", response_buf.items));
            }
            _ = dr;
        }
    }

    // ========================================================================
    // Multimodal generation (public API) — delegates to multimodal.zig
    // ========================================================================

    pub fn generateWithImage(self: *InferenceEngine, io: std.Io, prompt: []const u8, image_path: [:0]const u8, max_tokens: u32) !void {
        var ectx = self.ctx.toMultimodalContext();
        return multimodal_mod.generateWithImage(&ectx, io, prompt, image_path, max_tokens);
    }

    pub fn generateWithAudio(self: *InferenceEngine, io: std.Io, prompt: []const u8, audio_path: [:0]const u8, max_tokens: u32) !void {
        var ectx = self.ctx.toMultimodalContext();
        return multimodal_mod.generateWithAudio(&ectx, io, prompt, audio_path, max_tokens);
    }
};

// ============================================================================
// Stats printing (re-exported from executor for convenience)
// ============================================================================

pub const printStats = @import("executor.zig").printStats;

const testing = std.testing;

test "InferenceEngine struct size" {
    try testing.expect(@sizeOf(InferenceEngine) > 0);
}
