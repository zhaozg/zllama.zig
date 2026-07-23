//! Graph planner — builds and plans compute graphs for prefill and decode.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.
//! The planner receives a ModelContext reference and produces GraphPlan
//! objects that the executor can run without accessing raw context pointers.
//!
//! Reference: llama.cpp llm_graph_context / llama_build_graph

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const graph_context = @import("graph_context");
const kv_cache = @import("kv_cache");
const engine_common = @import("engine_common");

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
// GraphPlanner — builds and plans compute graphs
// ============================================================================

/// Builds compute graphs and produces GraphPlan objects.
/// Does NOT execute graphs — that's the executor's job.
///
/// The planner holds a reference to the ModelContext for accessing
/// the model's buildGraph vtable, KV cache, and graph context.
pub const GraphPlanner = struct {
    ctx_graph: *ggml.Context,
    model: model_if.ModelInstance,
    params: *const model_if.ModelParams,
    kv_cache_mgr: *kv_cache.KVCache,
    allocator: std.mem.Allocator,

    pub fn init(
        ctx_graph: *ggml.Context,
        model: model_if.ModelInstance,
        params: *const model_if.ModelParams,
        kv_cache_mgr: *kv_cache.KVCache,
        allocator: std.mem.Allocator,
    ) GraphPlanner {
        return .{
            .ctx_graph = ctx_graph,
            .model = model,
            .params = params,
            .kv_cache_mgr = kv_cache_mgr,
            .allocator = allocator,
        };
    }

    // ========================================================================
    // Prefill graph planning
    // ========================================================================

    /// Build a prefill graph for the given input tokens.
    /// Returns a GraphPlan ready for execution.
    pub fn planPrefill(
        self: *GraphPlanner,
        input_tokens: []const u32,
        gallocr: *ggml.Gallocr,
    ) !GraphPlan {
        const n_prompt_tokens: i32 = @intCast(input_tokens.len);
        self.ctx_graph.setNoAlloc(false);
        const input_tensor = try self.ctx_graph.newTensor1d(.i32, n_prompt_tokens);
        self.ctx_graph.setNoAlloc(true);

        const graph = try ggml.CGraph.initReserved(self.ctx_graph, 16384);
        var builder = graph_builder.GraphBuilder.init(self.ctx_graph, graph, self.params, self.allocator);
        const logits = try self.model.buildGraph(&builder, input_tensor, n_prompt_tokens, @ptrCast(self.kv_cache_mgr), 0);

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

    // ========================================================================
    // Decode graph planning
    // ========================================================================

    /// Build a single-token decode graph.
    /// Returns a GraphPlan ready for execution.
    pub fn planDecode(
        self: *GraphPlanner,
        token_id: i32,
        pos: i32,
        inc_ctx: *graph_context.IncContext,
        gallocr: *ggml.Gallocr,
    ) !GraphPlan {
        const step = try inc_ctx.beginStep();
        step.setToken(token_id);
        var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, self.params, self.allocator);
        const inc_logits = try self.model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(self.kv_cache_mgr), pos);

        return GraphPlan{
            .graph = step.graph,
            .logits = inc_logits,
            .n_tokens = 1,
            .start_pos = pos,
            .is_prefill = false,
            .gallocr = gallocr,
        };
    }

    // ========================================================================
    // Gallocr reservation
    // ========================================================================

    /// Reserve gallocr space for the worst-case incremental decode graph.
    /// Temporarily sets KV cache lengths to near-max to force worst-case allocation.
    pub fn reserveDecodeGallocr(
        self: *GraphPlanner,
        inc_ctx: *graph_context.IncContext,
        gallocr: *ggml.Gallocr,
    ) !void {
        _ = gallocr;
        const saved_lens = try self.kv_cache_mgr.getAllLengths(self.allocator);
        defer self.allocator.free(saved_lens);
        const max_pos: u32 = self.kv_cache_mgr.max_seq_len -| 1;
        self.kv_cache_mgr.setAllLengths(max_pos);
        const reserve_step = try inc_ctx.beginStep();
        reserve_step.setToken(0);
        var reserve_builder = graph_builder.GraphBuilder.init(reserve_step.ctx, reserve_step.graph, self.params, self.allocator);
        _ = try self.model.buildGraph(&reserve_builder, reserve_step.input_token, 1, @ptrCast(self.kv_cache_mgr), @intCast(max_pos));
        try inc_ctx.reserveGallocr(reserve_step.graph);
        for (self.kv_cache_mgr.layers, 0..) |*layer, i| layer.current_len = saved_lens[i];
    }
};

const testing = std.testing;

test "GraphPlan struct size" {
    try testing.expect(@sizeOf(GraphPlan) > 0);
}

test "GraphPlanner init" {
    // Just verify the struct can be created (no runtime test without model)
    const planner = GraphPlanner{
        .ctx_graph = undefined,
        .model = undefined,
        .params = undefined,
        .kv_cache_mgr = undefined,
        .allocator = undefined,
    };
    _ = planner;
}
