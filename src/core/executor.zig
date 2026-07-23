//! Graph executor — runs compute graphs and extracts results.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.
//! The executor only depends on GraphPlan — it never accesses raw context pointers.
//! This enforces dependency inversion: executor → planner → context.
//!
//! Reference: llama.cpp llama_graph_compute / llama_synchronize

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const engine_common = @import("engine_common");
const planner = @import("planner.zig");

const logger = std.log.scoped(.core_executor);

// ============================================================================
// GraphExecutor — runs compute graphs and extracts results
// ============================================================================

/// Executes compute graphs produced by GraphPlanner.
/// Does NOT build graphs — that's the planner's job.
///
/// The executor is stateless: it takes a GraphPlan, runs it, and returns results.
/// It does not hold any reference to the model or context.
pub const GraphExecutor = struct {
    n_threads: i32,
    n_vocab: usize,

    pub fn init(n_threads: i32, n_vocab: usize) GraphExecutor {
        return .{
            .n_threads = n_threads,
            .n_vocab = n_vocab,
        };
    }

    // ========================================================================
    // Graph execution
    // ========================================================================

    /// Execute a prefill graph and extract the last-token logits.
    /// Returns the logits for the last token (n_vocab floats).
    pub fn executePrefill(
        self: *GraphExecutor,
        plan: *const planner.GraphPlan,
        allocator: std.mem.Allocator,
    ) !planner.PrefillResult {
        const t_start = engine_common.currentTimeMs();

        // Allocate and compute
        if (!plan.gallocr.allocGraph(plan.graph)) return error.GraphAllocFailed;
        try plan.graph.compute(self.n_threads);

        const t_end = engine_common.currentTimeMs();
        const pp_time_s = @as(f64, @floatFromInt(t_end - t_start)) / 1000.0;

        // Extract logits for the last token
        const logits_heap = try allocator.alloc(f32, self.n_vocab);
        {
            const logits_data = try plan.logits.dataGet(f32, allocator);
            defer allocator.free(logits_data);
            const last_idx = @as(usize, @intCast(plan.n_tokens - 1));
            @memcpy(logits_heap, logits_data[last_idx * self.n_vocab ..][0..self.n_vocab]);
        }

        return planner.PrefillResult{
            .logits = logits_heap,
            .pos = plan.n_tokens,
            .pp_time_s = pp_time_s,
        };
    }

    /// Execute a single-token decode graph and return the logits tensor.
    /// The caller is responsible for sampling from the logits.
    pub fn executeDecode(
        self: *GraphExecutor,
        plan: *const planner.GraphPlan,
    ) !*ggml.Tensor {
        if (!plan.gallocr.allocGraph(plan.graph)) return error.GraphAllocFailed;
        try plan.graph.compute(self.n_threads);
        return plan.logits;
    }

    // ========================================================================
    // Convenience: execute a prefill plan and return the sampled first token
    // ========================================================================

    /// Execute prefill and sample the first token.
    /// Returns the sampled token ID and the prefill result.
    pub fn executePrefillAndSample(
        self: *GraphExecutor,
        plan: *const planner.GraphPlan,
        allocator: std.mem.Allocator,
        sampler_fn: *const fn (logits: *ggml.Tensor) i32,
    ) !struct { token: i32, result: planner.PrefillResult } {
        const result = try self.executePrefill(plan, allocator);
        // We need to sample from the raw logits tensor, not the heap copy.
        // The heap copy is for the caller; sampling uses the tensor directly.
        const token = sampler_fn(plan.logits);
        return .{ .token = token, .result = result };
    }
};

// ============================================================================
// Stats printing (moved from engine.zig)
// ============================================================================

pub fn printStats(
    arch: model_if.Architecture,
    model_name: []const u8,
    n_threads: i32,
    n_prompt_tokens: i32,
    gen_count: i32,
    pp_time_s: f64,
    tg_time_s: f64,
    benchmark: bool,
) void {
    if (benchmark) {
        engine_common.printBenchmark(.{
            .model_name = if (model_name.len > 0) model_name else @tagName(arch),
            .arch_name = @tagName(arch),
            .n_threads = n_threads,
            .n_prompt_tokens = n_prompt_tokens,
            .n_decode = gen_count,
            .pp_time_s = pp_time_s,
            .tg_time_s = tg_time_s,
        });
    } else if (gen_count > 0) {
        engine_common.printSummary(gen_count, pp_time_s + tg_time_s);
    }
}

const testing = std.testing;

test "GraphExecutor init" {
    const ex = GraphExecutor.init(4, 32000);
    try testing.expectEqual(@as(i32, 4), ex.n_threads);
    try testing.expectEqual(@as(usize, 32000), ex.n_vocab);
}
