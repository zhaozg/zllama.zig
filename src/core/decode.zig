//! Shared decode loop and gallocr reservation.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.
//!
//! Reference: llama.cpp llama_decode_internal / llama_synchronize

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const graph_context = @import("graph_context");
const kv_cache = @import("kv_cache");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const engine_common = @import("engine_common");

const logger = std.log.scoped(.core_decode);

// ============================================================================
// Callbacks for the shared decode loop
// ============================================================================

pub const DecodeCallbacks = struct {
    sample: *const fn (ctx: *anyopaque, logits: *ggml.Tensor) i32,
    skipToken: *const fn (ctx: *anyopaque, token: i32) bool,
    afterToken: ?*const fn (ctx: *anyopaque, token: i32, decoded: []const u8) anyerror!bool = null,
    onComplete: ?*const fn (ctx: *anyopaque, n_decoded: i32, tg_time_s: f64) void = null,
};

pub const DecodeResult = struct { gen_count: i32, tg_time_s: f64 };

// ============================================================================
// Gallocr reservation — pre-allocates graph memory for incremental decode
// ============================================================================

/// Reserve gallocr space for the worst-case incremental decode graph.
/// Temporarily sets KV cache lengths to near-max to force worst-case allocation.
pub fn reserveDecodeGallocr(
    allocator: std.mem.Allocator,
    kv_cache_mgr: *kv_cache.KVCache,
    inc_ctx: *graph_context.IncContext,
    model: model_if.ModelInstance,
    params: *const model_if.ModelParams,
) !void {
    const saved_lens = try kv_cache_mgr.getAllLengths(allocator);
    defer allocator.free(saved_lens);
    const max_pos: u32 = kv_cache_mgr.max_seq_len -| 1;
    kv_cache_mgr.setAllLengths(max_pos);
    const reserve_step = try inc_ctx.beginStep();
    reserve_step.setToken(0);
    var reserve_builder = graph_builder.GraphBuilder.init(reserve_step.ctx, reserve_step.graph, params, allocator);
    _ = try model.buildGraph(&reserve_builder, reserve_step.input_token, 1, @ptrCast(kv_cache_mgr), @intCast(max_pos));
    try inc_ctx.reserveGallocr(reserve_step.graph);
    for (kv_cache_mgr.layers, 0..) |*layer, i| layer.current_len = saved_lens[i];
}

// ============================================================================
// Shared decode loop
// ============================================================================

/// Run the incremental decode loop, producing up to `max_tokens` tokens.
/// Uses DecodeCallbacks for sampling decisions so text-only and multimodal
/// generation paths share the same loop body.
///
/// Reference: llama.cpp llama_decode_internal / llama_sample_token
pub fn runDecodeLoop(
    allocator: std.mem.Allocator,
    io: std.Io,
    model: model_if.ModelInstance,
    params: *const model_if.ModelParams,
    tok: *tokenizer.Tokenizer,
    kv_cache_mgr: *kv_cache.KVCache,
    inc_ctx: *graph_context.IncContext,
    n_threads: i32,
    first_token: i32,
    start_pos: i32,
    max_tokens: u32,
    callbacks: DecodeCallbacks,
    ctx: *anyopaque,
    benchmark: bool,
) !DecodeResult {
    var current_token: i32 = first_token;
    var pos: i32 = start_pos;
    var gen_count: u32 = 0;
    var eog_detect_buf = std.ArrayListUnmanaged(u8){ .items = &.{}, .capacity = 0 };
    defer eog_detect_buf.deinit(allocator);
    const t_tg_start = engine_common.currentTimeMs();

    // 内存压力检查：当上下文使用率超过 80% 时触发回收
    // 注意：如果 gallocr 已预规划（galloc_reserved），resetFull 会破坏哈希表，
    // 因此仅在非预规划状态下执行回收。
    if (!inc_ctx.galloc_reserved and inc_ctx.memRatio() >= 0.80) {
        logger.warn("Decode loop: context {d:.0}% full, triggering reset", .{inc_ctx.memRatio() * 100});
        inc_ctx.resetFull();
    }
    while (gen_count < max_tokens) {
        if (tok.isEog(@intCast(current_token))) break;

        if (callbacks.skipToken(ctx, current_token)) {
            const step = try inc_ctx.beginStep();
            step.setToken(current_token);
            var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, params, allocator);
            const inc_logits = try model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(kv_cache_mgr), pos);
            if (!step.galloc.allocGraph(step.graph)) return error.GraphAllocFailed;
            try step.graph.compute(n_threads);
            current_token = callbacks.sample(ctx, inc_logits);
            pos += 1;
            gen_count += 1;
            continue;
        }

        var buf: [128]u8 = undefined;
        const n = try tok.decodeSingle(@intCast(current_token), &buf);
        const decoded = buf[0..n];

        if (n > 0) {
            try eog_detect_buf.appendSlice(allocator, decoded);
            if (tok.isEogText(eog_detect_buf.items)) {
                if (!benchmark) {
                    const stdout_file = std.Io.File.stdout();
                    try stdout_file.writeStreamingAll(io, decoded);
                }
                break;
            }
        }

        if (!benchmark and n > 0) {
            const stdout_file = std.Io.File.stdout();
            try stdout_file.writeStreamingAll(io, decoded);
        }

        // 内存压力检查：当上下文使用率超过 80% 时触发回收
        if (!inc_ctx.galloc_reserved and inc_ctx.memRatio() >= 0.80) {
            logger.warn("Decode loop: context {d:.0}% full, triggering reset", .{inc_ctx.memRatio() * 100});
            inc_ctx.resetFull();
        }
        if (callbacks.afterToken) |cb| {
            if (!try cb(ctx, current_token, decoded)) break;
        }

        const step = try inc_ctx.beginStep();
        step.setToken(current_token);
        var inc_builder = graph_builder.GraphBuilder.init(step.ctx, step.graph, params, allocator);
        const inc_logits = try model.buildGraph(&inc_builder, step.input_token, 1, @ptrCast(kv_cache_mgr), pos);
        if (!step.galloc.allocGraph(step.graph)) return error.GraphAllocFailed;
        try step.graph.compute(n_threads);
        current_token = callbacks.sample(ctx, inc_logits);
        pos += 1;
        gen_count += 1;
    }

    const t_tg_end = engine_common.currentTimeMs();
    const tg_time_s = @as(f64, @floatFromInt(t_tg_end - t_tg_start)) / 1000.0;
    if (callbacks.onComplete) |cb| cb(ctx, @intCast(gen_count), tg_time_s);
    return .{ .gen_count = @intCast(gen_count), .tg_time_s = tg_time_s };
}

// ============================================================================
// Print stats
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
