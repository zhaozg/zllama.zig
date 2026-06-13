//! Three-stage multimodal prefill helper.
//!
//! For multimodal models (Gemma 4), the input is split into three parts:
//!   1. Text prefix (before placeholder) — causal attention
//!   2. Media tokens (vision/audio embeddings) — non-causal attention
//!   3. Text suffix (after placeholder) — causal attention, sampled for first token
//!
//! This module provides a reusable helper that encapsulates the three-pass
//! graph building, galloc management, and compute orchestration.

const std = @import("std");
const ggml = @import("ggml");
const graph_builder = @import("graph_builder");
const kv_cache = @import("kv_cache");
const model = @import("model");
const engine_common = @import("engine_common");

const log = std.log.scoped(.prefill);

/// Result of the three-stage prefill
pub const PrefillResult = struct {
    /// Logits from Pass 3 (for sampling first generated token)
    logits: *ggml.Tensor,
    /// Position after all three passes (for incremental decode)
    pos: i32,
    /// Prefill time in seconds
    pp_time_s: f64,
};

/// Function pointer type for the media embedding forward pass (Pass 2).
/// The callee must build a graph with embedding override for non-causal attention.
pub const MediaForwardFn = *const fn (
    /// Concrete model pointer (e.g., *Gemma4Model)
    model_ptr: *anyopaque,
    ctx: *ggml.Context,
    graph: *ggml.CGraph,
    input_tokens: *ggml.Tensor,
    n_tokens: i32,
    kv_cache_mgr: ?*kv_cache.KVCache,
    start_pos: i32,
    embd_override: *ggml.Tensor,
    embd_offset: i32,
    causal: bool,
) anyerror!*ggml.Tensor;

/// Execute a three-stage multimodal prefill.
///
/// Parameters:
/// - graph_ctx: ggml context for graph building (caller owns, must have enough free space)
/// - model_instance: the model instance (for Pass 1 and Pass 3 via buildGraph)
/// - model_ptr: opaque pointer to the concrete model (for Pass 2 via mediaForwardFn)
/// - mediaForwardFn: function to build the media pass graph
/// - kv_cache_ptr: casted KV cache pointer
/// - prefix_tokens: token IDs for text before the placeholder
/// - media_token_id: the placeholder token ID (e.g., <|image|> or <|audio|>)
/// - media_token_count: number of media tokens (vision/audio embedding count)
/// - media_embeddings: pre-computed media embeddings tensor
/// - suffix_tokens: token IDs for text after the placeholder
/// - n_threads: number of CPU threads
/// - allocator: memory allocator
pub fn threeStagePrefill(
    graph_ctx: *ggml.Context,
    model_instance: model.ModelInstance,
    model_ptr: *anyopaque,
    mediaForwardFn: MediaForwardFn,
    kv_cache_mgr: *kv_cache.KVCache,
    prefix_tokens: []const u32,
    media_token_id: u32,
    media_token_count: i32,
    media_embeddings: *ggml.Tensor,
    suffix_tokens: []const u32,
    params: *const model.ModelParams,
    n_threads: i32,
    allocator: std.mem.Allocator,
) !PrefillResult {
    const prefix_len: i32 = @intCast(prefix_tokens.len);
    const suffix_len: i32 = @intCast(suffix_tokens.len);
    const n_media: i32 = media_token_count;

    const kv_cache_ptr: ?*kv_cache.KVCache = kv_cache_mgr;
    var pp_time_s: f64 = 0.0;

    const buft = ggml.backendCpuBufferType();

    // ===================================================================
    // Pass 1: Text prefix only (causal attention)
    // ===================================================================
    if (prefix_len > 0) {
        // Keep setNoAlloc(false) so ggml ops inside buildGraph can allocate
        graph_ctx.setNoAlloc(false);
        const p1_input = try graph_ctx.newTensor1d(.i32, prefix_len);
        {
            const data = p1_input.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(prefix_len))];
            for (prefix_tokens, 0..) |t, j| {
                slice[j] = @as(i32, @intCast(t));
            }
        }

        var p1_graph = try ggml.CGraph.initReserved(graph_ctx, 16384);
        var p1_builder = graph_builder.GraphBuilder.init(graph_ctx, p1_graph, params, allocator);
        _ = try model_instance.buildGraph(&p1_builder, p1_input, prefix_len, @ptrCast(kv_cache_ptr), 0);
        graph_ctx.setNoAlloc(true);

        var p1_galloc = try ggml.Gallocr.init(buft);
        defer p1_galloc.free();
        if (!p1_galloc.allocGraph(p1_graph)) {
            log.err("Graph alloc failed: text-prefix pass", .{});
            return error.GraphAllocFailed;
        }
        try p1_graph.compute(n_threads);
        log.debug("Pass 1 (text prefix): {d} tokens ✓", .{prefix_len});
    }

    // ===================================================================
    // Pass 2: Media tokens only (non-causal attention)
    // ===================================================================
    {
        // Keep setNoAlloc(false) so ggml ops inside mediaForwardFn can allocate
        graph_ctx.setNoAlloc(false);
        const p2_input = try graph_ctx.newTensor1d(.i32, n_media);
        {
            const data = p2_input.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_media))];
            @memset(slice, @as(i32, @intCast(media_token_id)));
        }

        var p2_graph = try ggml.CGraph.initReserved(graph_ctx, 16384);
        _ = try mediaForwardFn(
            model_ptr,
            graph_ctx,
            p2_graph,
            p2_input,
            n_media,
            kv_cache_ptr,
            prefix_len,
            media_embeddings,
            0,
            false, // non-causal
        );
        graph_ctx.setNoAlloc(true);

        var p2_galloc = try ggml.Gallocr.init(buft);
        defer p2_galloc.free();
        if (!p2_galloc.allocGraph(p2_graph)) {
            log.err("Graph alloc failed: media pass", .{});
            return error.GraphAllocFailed;
        }

        try p2_graph.compute(n_threads);
    }

    // ===================================================================
    // Pass 3: Text suffix only (causal attention) — sample logits from here
    // ===================================================================
    const sfx_n: i32 = if (suffix_len > 0) suffix_len else 1;
    // Keep setNoAlloc(false) so ggml ops inside buildGraph can allocate
    graph_ctx.setNoAlloc(false);
    const p3_input = try graph_ctx.newTensor1d(.i32, sfx_n);
    {
        const data = p3_input.dataBytes();
        const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(sfx_n))];
        if (suffix_len > 0) {
            for (suffix_tokens, 0..) |t, j| {
                slice[j] = @as(i32, @intCast(t));
            }
        } else {
            // No real suffix: use a single dummy token (its KV entry is harmless
            // since we immediately sample and enter incremental decode).
            slice[0] = @as(i32, @intCast(media_token_id));
        }
    }

    const suffix_start_pos: i32 = prefix_len + n_media;
    var p3_graph = try ggml.CGraph.initReserved(graph_ctx, 16384);
    var p3_builder = graph_builder.GraphBuilder.init(graph_ctx, p3_graph, params, allocator);
    const logits = try model_instance.buildGraph(&p3_builder, p3_input, sfx_n, @ptrCast(kv_cache_ptr), suffix_start_pos);
    graph_ctx.setNoAlloc(true);

    var p3_galloc = try ggml.Gallocr.init(buft);
    defer p3_galloc.free();
    if (!p3_galloc.allocGraph(p3_graph)) {
        log.err("Graph alloc failed: text-suffix pass", .{});
        return error.GraphAllocFailed;
    }

    const t3_start = engine_common.currentTimeMs();
    try p3_graph.compute(n_threads);
    const t3_end = engine_common.currentTimeMs();
    pp_time_s += @as(f64, @floatFromInt(t3_end - t3_start)) / 1000.0;
    log.debug("Pass 3 (text suffix): {d} tokens ✓", .{sfx_n});

    const pos: i32 = suffix_start_pos + (if (suffix_len > 0) suffix_len else 1);

    return PrefillResult{
        .logits = logits,
        .pos = pos,
        .pp_time_s = pp_time_s,
    };
}

// ===================================================================
// Tests
// ===================================================================

const testing = std.testing;

test "PrefillResult default" {
    try testing.expectEqual(@as(usize, @sizeOf(PrefillResult)), @sizeOf(PrefillResult));
}
