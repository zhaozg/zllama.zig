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

test "GraphPlan struct size" {
    try testing.expect(@sizeOf(GraphPlan) > 0);
}

test "GraphPlanner init" {
    const planner = GraphPlanner.init();
    _ = planner;
}
