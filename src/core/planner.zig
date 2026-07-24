//! Graph planner — builds and plans compute graphs for prefill.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.
//! The planner receives a ModelContext reference and produces GraphPlan
//! objects that the executor can run without accessing raw context pointers.
//!
//! Note: Decode graph construction is handled inline in decode.zig's
//! runDecodeLoop, and gallocr reservation is in decode.zig's
//! reserveDecodeGallocr. These were previously duplicated here.
//!
//! Reference: llama.cpp llm_graph_context / llama_build_graph

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const kv_cache = @import("kv_cache");

const logger = std.log.scoped(.core_planner);

// ============================================================================
// GraphPlan — the output of planning, consumed by executor
// ============================================================================

/// A complete plan for executing a compute graph.
/// The executor only needs this struct — it never accesses raw context pointers.
///
/// This is the key abstraction for dependency inversion:
/// - `planner` builds the graph and produces a `GraphPlan`
/// - `executor` consumes the `GraphPlan` without knowing about the model internals
pub const GraphPlan = struct {
    /// The compute graph to execute
    graph: *ggml.CGraph,
    /// The logits output tensor (for extracting results)
    logits: *ggml.Tensor,
    /// Number of tokens in this batch
    n_tokens: i32,
    /// Starting position in the KV cache
    start_pos: i32,
    /// Whether this is a prefill (multi-token) or decode (single-token) plan
    is_prefill: bool,
    /// Gallocr for memory allocation (owned by ModelContext, borrowed here)
    gallocr: *ggml.Gallocr,
};

// ============================================================================
// PrefillResult — output of prefill execution
// ============================================================================

pub const PrefillResult = struct {
    logits: []f32,
    pos: i32,
    pp_time_s: f64,
};

// ============================================================================
// PrefillGraphCache — caches prefill graph structure to eliminate realloc
// ============================================================================

/// Caches the prefill graph structure so that repeated prefill calls with
/// the same token count do not trigger ggml_gallocr_needs_realloc.
///
/// The cache stores the graph metadata (tensor descriptors, node count) and
/// the gallocr reservation. When a new prefill request arrives with the same
/// token count, the cached graph is reused directly.
///
/// Key insight: prefill graphs are deterministic for a given token count.
/// The graph structure (nodes, edges, tensor shapes) depends only on n_tokens,
/// not on the actual token values. So we can cache by n_tokens.
///
/// Reference: llama.cpp's approach of reusing the same graph for all prefill
/// calls with the same batch size.
pub const PrefillGraphCache = struct {
    /// Cached entry for a specific token count
    const Entry = struct {
        n_tokens: i32,
        /// The cached graph (reused across calls)
        graph: *ggml.CGraph,
        /// The logits tensor pointer (stable across calls)
        logits: *ggml.Tensor,
        /// The input tensor pointer (stable across calls)
        input_tensor: *ggml.Tensor,
    };

    entries: std.ArrayListUnmanaged(Entry),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PrefillGraphCache {
        return .{
            .entries = .{ .items = &.{}, .capacity = 0 },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PrefillGraphCache) void {
        self.entries.deinit(self.allocator);
    }

    /// Look up a cached prefill plan for the given token count.
    /// Returns null if no cached entry exists.
    pub fn lookup(self: *const PrefillGraphCache, n_tokens: i32) ?Entry {
        for (self.entries.items) |entry| {
            if (entry.n_tokens == n_tokens) return entry;
        }
        return null;
    }

    /// Insert a new entry into the cache.
    pub fn insert(self: *PrefillGraphCache, entry: Entry) !void {
        try self.entries.append(self.allocator, entry);
    }

    /// Invalidate all cached entries (e.g., after model reload).
    pub fn clear(self: *PrefillGraphCache) void {
        self.entries.clearAndFree(self.allocator);
    }
};

// ============================================================================
// GraphPlanner — builds and plans compute graphs
// ============================================================================

/// Builds compute graphs and produces GraphPlan objects.
/// Does NOT execute graphs — that's the executor's job.
///
/// The planner holds a reference to the ModelContext for accessing
/// the model's buildGraph vtable, KV cache, and graph context.
pub const GraphPlanner = struct {
    /// STATELESS DESIGN: Does NOT store pointers to ModelContext fields.
    /// All methods accept required references as parameters to avoid
    /// dangling pointer issues when InferenceEngine is moved.
    pub fn init() GraphPlanner {
        return .{};
    }

    pub fn deinit(self: *GraphPlanner) void {
        _ = self;
    }

    // ========================================================================
    // Prefill graph planning
    // ========================================================================

    /// Build a prefill graph for the given input tokens.
    /// Returns a GraphPlan ready for execution.
    pub fn planPrefill(
        self: *GraphPlanner,
        ctx_graph: *ggml.Context,
        model: model_if.ModelInstance,
        params: *const model_if.ModelParams,
        kv_cache_mgr: *kv_cache.KVCache,
        allocator: std.mem.Allocator,
        input_tokens: []const u32,
        gallocr: *ggml.Gallocr,
    ) !GraphPlan {
        _ = self;
        const n_prompt_tokens: i32 = @intCast(input_tokens.len);

        ctx_graph.setNoAlloc(false);
        const input_tensor = try ctx_graph.newTensor1d(.i32, n_prompt_tokens);
        ctx_graph.setNoAlloc(true);

        const graph = try ggml.CGraph.initReserved(ctx_graph, 16384);
        var builder = graph_builder.GraphBuilder.init(ctx_graph, graph, params, allocator);
        const logits = try model.buildGraph(&builder, input_tensor, n_prompt_tokens, @ptrCast(kv_cache_mgr), 0);

        // Copy input token data into the tensor
        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_prompt_tokens))];
            for (input_tokens, 0..) |token, j| slice[j] = @as(i32, @intCast(token));
        }

        return GraphPlan{
            .graph = graph,
            .logits = logits,
            .n_tokens = n_prompt_tokens,
            .start_pos = 0,
            .is_prefill = true,
            .gallocr = gallocr,
        };
    }
};

const testing = std.testing;

// ============================================================================
// PrefillGraphCache tests
// ============================================================================

test "PrefillGraphCache init and deinit" {
    var cache = PrefillGraphCache.init(testing.allocator);
    defer cache.deinit();
    try testing.expectEqual(@as(usize, 0), cache.entries.items.len);
}

test "PrefillGraphCache insert and lookup hit" {
    var cache = PrefillGraphCache.init(testing.allocator);
    defer cache.deinit();

    // Use fake pointers — lookup only compares n_tokens, never dereferences
    const entry = PrefillGraphCache.Entry{
        .n_tokens = 42,
        .graph = @ptrFromInt(0x1000),
        .logits = @ptrFromInt(0x2000),
        .input_tensor = @ptrFromInt(0x3000),
    };
    try cache.insert(entry);

    const found = cache.lookup(42);
    try testing.expect(found != null);
    try testing.expectEqual(@as(i32, 42), found.?.n_tokens);
    try testing.expectEqual(@intFromPtr(entry.graph), @intFromPtr(found.?.graph));
}

test "PrefillGraphCache lookup miss on empty" {
    var cache = PrefillGraphCache.init(testing.allocator);
    defer cache.deinit();

    try testing.expect(cache.lookup(10) == null);
    try testing.expect(cache.lookup(0) == null);
}

test "PrefillGraphCache lookup miss with entries" {
    var cache = PrefillGraphCache.init(testing.allocator);
    defer cache.deinit();

    const entry = PrefillGraphCache.Entry{
        .n_tokens = 10,
        .graph = @ptrFromInt(0x1000),
        .logits = @ptrFromInt(0x2000),
        .input_tensor = @ptrFromInt(0x3000),
    };
    try cache.insert(entry);

    try testing.expect(cache.lookup(20) == null);
    try testing.expect(cache.lookup(5) == null);
    try testing.expect(cache.lookup(10) != null); // still finds the real one
}

test "PrefillGraphCache multiple entries" {
    var cache = PrefillGraphCache.init(testing.allocator);
    defer cache.deinit();

    const token_counts = [_]i32{ 4, 8, 16, 32, 64 };
    for (token_counts) |n| {
        const entry = PrefillGraphCache.Entry{
            .n_tokens = n,
            .graph = @ptrFromInt(@as(usize, @intCast(n * 1000))),
            .logits = @ptrFromInt(@as(usize, @intCast(n * 2000))),
            .input_tensor = @ptrFromInt(@as(usize, @intCast(n * 3000))),
        };
        try cache.insert(entry);
    }

    // All inserted entries should be findable
    for (token_counts) |n| {
        try testing.expect(cache.lookup(n) != null);
    }
    // Non-inserted values should miss
    try testing.expect(cache.lookup(128) == null);
    try testing.expect(cache.lookup(2) == null);
}

test "PrefillGraphCache clear" {
    var cache = PrefillGraphCache.init(testing.allocator);
    defer cache.deinit();

    const entry = PrefillGraphCache.Entry{
        .n_tokens = 7,
        .graph = @ptrFromInt(0x7000),
        .logits = @ptrFromInt(0x8000),
        .input_tensor = @ptrFromInt(0x9000),
    };
    try cache.insert(entry);
    try testing.expect(cache.lookup(7) != null);

    cache.clear();
    try testing.expectEqual(@as(usize, 0), cache.entries.items.len);
    try testing.expect(cache.lookup(7) == null);
}

// ============================================================================
// Mock model for planPrefill test
// ============================================================================

const MockModelData = struct {
    params: model_if.ModelParams,

    fn getParams(ptr: *anyopaque) *const model_if.ModelParams {
        const self: *MockModelData = @ptrCast(@alignCast(ptr));
        return &self.params;
    }

    fn buildGraph(
        ptr: *anyopaque,
        builder: *graph_builder.GraphBuilder,
        input: *ggml.Tensor,
        n_tokens: i32,
        cache: ?*anyopaque,
        pos: i32,
    ) anyerror!*ggml.Tensor {
        _ = ptr;
        _ = cache;
        _ = pos;
        _ = n_tokens;
        // Minimal graph: forwardExpand + setOutput the input tensor.
        // In a real model this would do a full transformer forward pass.
        builder.forwardExpand(input);
        builder.setOutput(input);
        return input;
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        _ = ptr;
        _ = allocator;
        // Stack-allocated mock — nothing to free
    }
};

const mock_vtable = model_if.ModelVTable{
    .getParams = MockModelData.getParams,
    .buildGraph = MockModelData.buildGraph,
    .deinit = MockModelData.deinit,
};

test "GraphPlanner planPrefill with mock model" {
    // Setup: ggml context for the graph (needs ~2MB for 16384 graph nodes)
    const ctx_graph = try ggml.Context.init(2 * 1024 * 1024);
    defer ctx_graph.deinit();

    // Setup: ggml context for KV cache (separate, small)
    const ctx_kv = try ggml.Context.init(64 * 1024);
    defer ctx_kv.deinit();

    // Setup: backend and gallocr
    const backend = try ggml.backendCpuInit();
    defer ggml.backendFree(backend);

    const buft = ggml.backendCpuBufferType();
    var gallocr = try ggml.Gallocr.init(buft);
    defer gallocr.free();

    // Setup: KVCache — minimal: 1 layer, 1 KV head, 16 dim, max 8 tokens
    var kv = try kv_cache.KVCache.init(ctx_kv, 1, 1, 16, 8, testing.allocator);
    defer kv.deinit(testing.allocator);

    // Setup: mock model with trivial params
    var mock_data = MockModelData{
        .params = model_if.ModelParams{
            .n_vocab = 100,
            .n_embd = 32,
            .n_head = 2,
            .n_head_dim = 16,
            .n_kv_head = 1,
            .n_layer = 1,
            .n_ff = 128,
            .max_seq_len = 256,
        },
    };
    const mock_model = model_if.ModelInstance{
        .ptr = @ptrCast(&mock_data),
        .vtable = &mock_vtable,
    };

    // Execute: planPrefill with 5 input tokens
    var planner = GraphPlanner.init();
    const input_tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const plan = try planner.planPrefill(
        ctx_graph,
        mock_model,
        &mock_data.params,
        &kv,
        testing.allocator,
        &input_tokens,
        gallocr,
    );

    // Verify: GraphPlan metadata
    try testing.expectEqual(@as(i32, 5), plan.n_tokens);
    try testing.expectEqual(@as(i32, 0), plan.start_pos);
    try testing.expectEqual(true, plan.is_prefill);

    // Verify: graph and logits pointers are non-null
    try testing.expect(@intFromPtr(plan.graph) != 0);
    try testing.expect(@intFromPtr(plan.logits) != 0);

    // Verify: graph has at least one node (the input leaf from forwardExpand)
    try testing.expect(plan.graph.nNodes() > 0);

    // Verify: input token data was correctly copied into the tensor
    {
        const data = plan.logits.dataBytes();
        // data should be at least n_tokens * sizeof(i32) bytes
        try testing.expect(data.len >= @as(usize, @intCast(plan.n_tokens)) * @sizeOf(i32));
        const slice = @as([*]const i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(plan.n_tokens))];
        try testing.expectEqual(@as(i32, 1), slice[0]);
        try testing.expectEqual(@as(i32, 3), slice[2]);
        try testing.expectEqual(@as(i32, 5), slice[4]);
    }

    // Verify: gallocr is passed through correctly
    try testing.expectEqual(@intFromPtr(gallocr), @intFromPtr(plan.gallocr));
}

test "GraphPlan struct size" {
    try testing.expect(@sizeOf(GraphPlan) > 0);
}

test "GraphPlanner init" {
    const planner = GraphPlanner.init();
    _ = planner;
}
