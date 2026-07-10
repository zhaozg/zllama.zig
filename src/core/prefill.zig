//! Three-stage multimodal prefill helper.
//!
//! For multimodal models (Gemma 4), the input is split into three parts:
//!   1. Text prefix (before placeholder) — causal attention
//!   2. Media tokens (vision/audio embeddings) — non-causal attention
//!   3. Text suffix (after placeholder) — causal attention, sampled for first token
//!
//! This module provides a reusable helper that encapsulates the three-pass
//! graph building, galloc management, and compute orchestration.
//!
//! The ggml context is RESET between passes to free stale tensor descriptors
//! and graph nodes, avoiding ggml backend / Gallocr corruption from
//! cross-pass tensor aliasing (especially `ggml_cpy` ops into KV cache views).
//! Media embeddings are passed as raw f32 data so they survive context reset.
//!
//! Each pass:
//!   1. Context reset() — free all previous tensors/nodes
//!   2. setNoAlloc(false) — allow tensor allocation
//!   3. Build graph (tensors, graph nodes)
//!   4. setNoAlloc(true) — lock context
//!   5. Gallocr alloc + compute

const std = @import("std");
const ggml = @import("ggml");
const graph_builder = @import("graph_builder");
const kv_cache = @import("kv_cache");
const model = @import("model");
const engine_common = @import("engine_common");

const log = std.log.scoped(.prefill);

/// Result of the three-stage prefill
pub const PrefillResult = struct {
    /// Logits from Pass 3 (for sampling first generated token).
    /// Heap-allocated f32 slice, caller must free.
    logits: []f32,
    /// Position after all three passes (for incremental decode)
    pos: i32,
    /// Prefill time in seconds
    pp_time_s: f64,

    pub fn deinit(self: *PrefillResult, allocator: std.mem.Allocator) void {
        allocator.free(self.logits);
    }
};

/// Execute a three-stage multimodal prefill.
///
/// Parameters:
/// - graph_ctx: ggml context for graph building (caller owns, RESET between passes)
/// - model_instance: the model instance (for Pass 1/2/3 via buildGraph and buildMM)
/// - kv_cache_mgr: KV cache manager (shared across all three passes)
/// - prefix_tokens: token IDs for text before the placeholder
/// - media_token_id: the placeholder token ID (e.g., <|image|> or <|audio|>)
/// - media_token_count: number of media tokens (vision/audio embedding count)
/// - media_embeddings_data: pre-computed media embeddings as raw f32 slice, shape [n_embd, n_media]
/// - media_embd_dim: dimension of each embedding (n_embd, must match model)
/// - suffix_tokens: token IDs for text after the placeholder
/// - params: model parameters
/// - n_threads: number of CPU threads
/// - allocator: memory allocator
pub fn threeStagePrefill(
    graph_ctx: *ggml.Context,
    model_instance: model.ModelInstance,
    kv_cache_mgr: *kv_cache.KVCache,
    prefix_tokens: []const u32,
    media_token_id: u32,
    media_token_count: i32,
    media_embeddings_data: []const f32,
    media_embd_dim: u32,
    suffix_tokens: []const u32,
    params: *const model.ModelParams,
    n_threads: i32,
    allocator: std.mem.Allocator,
) !PrefillResult {
    const prefix_len: i32 = @intCast(prefix_tokens.len);
    const suffix_len: i32 = @intCast(suffix_tokens.len);
    const n_media: i32 = media_token_count;
    const n_embd_val: i64 = @intCast(media_embd_dim);

    const kv_cache_ptr: ?*kv_cache.KVCache = kv_cache_mgr;
    var pp_time_s: f64 = 0.0;

    const buft = ggml.backendCpuBufferType();

    // —— 三阶段预填充参数日志 ——
    log.info("=== Three-stage multimodal prefill ===", .{});
    log.info("  Prefix  (text, causal) : {d} tokens  [pos 0..{d})", .{ prefix_len, prefix_len });
    log.info("  Media   (embed,non-causal): {d} tokens  [pos {d}..{d})  embd_dim={d}  placeholder_token_id={d}", .{
        n_media, prefix_len, prefix_len + n_media, n_embd_val, media_token_id,
    });
    log.info("  Suffix  (text, causal) : {d} tokens  [pos {d}..{d})", .{
        suffix_len, prefix_len + n_media, prefix_len + n_media + suffix_len,
    });
    log.info("  Total tokens (3 passes): {d}", .{prefix_len + n_media + suffix_len});

    // Helper to create a media embeddings tensor in the fresh context
    const createMediaTensor = struct {
        fn create(ctx: *ggml.Context, data: []const f32, n_embd: i64, n_tokens: i32) !*ggml.Tensor {
            const t = try ctx.newTensor2d(.f32, n_embd, n_tokens);
            const t_data = t.dataBytes();
            const src_bytes = data.len * @sizeOf(f32);
            @memcpy(t_data[0..src_bytes], @as([*]const u8, @ptrCast(data.ptr))[0..src_bytes]);
            t.setName("mp_embd");
            return t;
        }
    }.create;

    // ===================================================================
    // Pass 1: Text prefix only (causal attention)
    // Positions: 0 .. prefix_len-1
    // ===================================================================
    if (prefix_len > 0) {
        graph_ctx.reset();
        // Use no_alloc = false — tensor data is allocated by ggml context.
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

        const t1_start = engine_common.currentTimeMs();
        try p1_graph.compute(n_threads);
        const t1_end = engine_common.currentTimeMs();
        pp_time_s += @as(f64, @floatFromInt(t1_end - t1_start)) / 1000.0;
        log.debug("Pass 1 (text prefix): {d} tokens in {d:.3}s ✓", .{ prefix_len, @as(f64, @floatFromInt(t1_end - t1_start)) / 1000.0 });
        log.debug("  -> KV cache current_len after Pass 1: {d}", .{kv_cache_mgr.currentLen()});
    }

    // ===================================================================
    // Pass 2: Media tokens in chunks (non-causal attention)
    // Positions: prefix_len .. prefix_len + n_media - 1
    // Chunked to avoid OOM from O(n_tokens²) attention for large vision sequences.
    // ===================================================================
    {
        const CHUNK_SIZE: i32 = 256;
        var chunk_start: i32 = 0;
        while (chunk_start < n_media) {
            const chunk_size: i32 = @min(CHUNK_SIZE, n_media - chunk_start);
            const chunk_pos: i32 = prefix_len + chunk_start;
            const embd_offset: usize = @as(usize, @intCast(chunk_start)) * @as(usize, @intCast(n_embd_val));
            const embd_len: usize = @as(usize, @intCast(chunk_size)) * @as(usize, @intCast(n_embd_val));
            const chunk_embd_data = media_embeddings_data[embd_offset..][0..embd_len];

            graph_ctx.reset();
            // Use no_alloc = false mode — tensor data is allocated by ggml context.
            graph_ctx.setNoAlloc(false);

            const p2_embd = try createMediaTensor(graph_ctx, chunk_embd_data, n_embd_val, chunk_size);

            const p2_input = try graph_ctx.newTensor1d(.i32, chunk_size);
            {
                const data = p2_input.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(chunk_size))];
                @memset(slice, @as(i32, @intCast(media_token_id)));
            }

            var p2_graph = try ggml.CGraph.initReserved(graph_ctx, 16384);
            _ = try model_instance.buildMM(
                graph_ctx,
                p2_graph,
                p2_input,
                chunk_size,
                @ptrCast(kv_cache_ptr),
                chunk_pos,
                p2_embd,
                0,
                false, // non-causal
            );
            graph_ctx.setNoAlloc(true);

            var p2_galloc = try ggml.Gallocr.init(buft);
            defer p2_galloc.free();
            if (!p2_galloc.allocGraph(p2_graph)) {
                log.err("Graph alloc failed: media pass chunk {d}", .{chunk_start});
                return error.GraphAllocFailed;
            }

            const t2_start = engine_common.currentTimeMs();
            try p2_graph.compute(n_threads);
            const t2_end = engine_common.currentTimeMs();
            pp_time_s += @as(f64, @floatFromInt(t2_end - t2_start)) / 1000.0;

            chunk_start += chunk_size;
        }

        log.debug("Pass 2 (media): {d} tokens in {d} chunks, non-causal", .{ n_media, @divTrunc(n_media + CHUNK_SIZE - 1, CHUNK_SIZE) });
        log.debug("  -> KV cache current_len after Pass 2: {d}", .{kv_cache_mgr.currentLen()});
    }

    // ===================================================================
    // Pass 3: Text suffix only (causal attention) — sample logits from here
    // Positions: prefix_len + n_media .. prefix_len + n_media + suffix_len - 1
    // ===================================================================
    const sfx_n: i32 = if (suffix_len > 0) suffix_len else 1;
    graph_ctx.reset();
    // Use no_alloc = false — tensor data is allocated by ggml context.
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
    log.debug("Pass 3 (text suffix): {d} tokens in {d:.3}s ✓", .{ sfx_n, @as(f64, @floatFromInt(t3_end - t3_start)) / 1000.0 });
    log.debug("  -> suffix pass start_pos={d}, causal=true (default text forward)", .{suffix_start_pos});
    log.debug("  -> KV cache current_len after Pass 3: {d}", .{kv_cache_mgr.currentLen()});

    // Copy last token's logits to heap before galloc is freed
    // logits shape: [n_vocab, n_tokens] — we want the last token
    const n_vocab = @as(usize, @intCast(params.n_vocab));
    const n_tok = @as(usize, @intCast(sfx_n));
    const last_offset = (n_tok - 1) * n_vocab;
    const logits_heap = try allocator.alloc(f32, n_vocab);
    {
        const logits_data = try logits.dataGet(f32, allocator);
        defer allocator.free(logits_data);
        @memcpy(logits_heap, logits_data[last_offset .. last_offset + n_vocab]);
    }

    const pos: i32 = suffix_start_pos + (if (suffix_len > 0) suffix_len else 1);

    return PrefillResult{
        .logits = logits_heap,
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
