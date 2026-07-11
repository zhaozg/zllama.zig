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
//! 内存策略（docs/MEMMGT.md §4.2.2）：
//! - 每个 Pass 使用独立的临时 ggml_context，Pass 完成后立即释放。
//! - 使用 measureGraph 精确测量所需内存大小，避免浪费。
//! - Gallocr 由调用者传入（所有权上移），跨 Pass 复用。
//!
//! 每个 Pass:
//!   1. 创建临时 context（大小由 measureGraph 精确确定）
//!   2. setNoAlloc(false) — 允许张量分配
//!   3. 构建图（张量、图节点）
//!   4. setNoAlloc(true) — 锁定 context
//!   5. Gallocr alloc + compute
//!   6. 释放临时 context

const std = @import("std");
const ggml = @import("ggml");
const graph_builder = @import("graph_builder");
const kv_cache = @import("kv_cache");
const model = @import("model");
const engine_common = @import("engine_common");

const log = std.log.scoped(.prefill);

/// 临时 context 的最小大小（512 MB），用于首次图构建。
/// Gemma4 35层模型在 no_alloc=false 模式下，每个中间张量的数据
/// 都分配在 context 中。35层 × ~8MB/层 ≈ 280MB 数据 + 元数据。
/// 512MB 确保首次构建不会因空间不足而崩溃。
const MIN_TEMP_CTX_SIZE: usize = 512 * 1024 * 1024;

/// 临时 context 的最大大小（4 GB），防止过度分配
const MAX_TEMP_CTX_SIZE: usize = 4 * 1024 * 1024 * 1024;

/// 元数据开销估计：每个张量约需 512 字节（GGML_OBJECT_SIZE + GGML_TENSOR_SIZE + 对齐）
/// 对于大型图（~2000 张量），元数据约 1MB。加 2MB 安全余量。
const METADATA_OVERHEAD: usize = 2 * 1024 * 1024;

/// 测量余量比例：在测量值基础上额外分配 20% 以防止溢出
const MEASURE_MARGIN_RATIO: f64 = 0.20;

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

/// 测量计算图所需内存，并返回带余量的 context 大小。
/// ggml.measureGraph 返回张量数据所需大小（通过 gallocr），
/// 但 context 还需要存储元数据（张量描述符、图节点等）。
/// 此函数在测量值基础上加上元数据开销和余量。
fn measureAndSize(graph: *ggml.CGraph, buft: *ggml.BackendBufferType, label: []const u8) usize {
    const measured = ggml.measureGraph(graph, buft) catch |err| {
        log.warn("measureGraph failed for {s} ({}), using default size", .{ label, err });
        return MIN_TEMP_CTX_SIZE;
    };
    // 总大小 = 数据大小 + 元数据开销 + 余量
    const total = measured + METADATA_OVERHEAD;
    const margin: usize = @intFromFloat(@as(f64, @floatFromInt(total)) * MEASURE_MARGIN_RATIO);
    const ctx_size = @min(total + margin, MAX_TEMP_CTX_SIZE);
    log.debug("  {s}: measured={d:.1} MB, allocated={d:.1} MB", .{
        label,
        @as(f64, @floatFromInt(measured)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(ctx_size)) / (1024.0 * 1024.0),
    });
    return ctx_size;
}

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
/// - gallocr: Gallocr 实例（所有权上移自 engine_common.computeGraph）
pub fn threeStagePrefill(
    graph_ctx: *ggml.Context, // 保留参数以保持 API 兼容性，每个 Pass 使用独立临时 context

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
    gallocr: *ggml.Gallocr,
) !PrefillResult {
    const prefix_len: i32 = @intCast(prefix_tokens.len);
    const suffix_len: i32 = @intCast(suffix_tokens.len);
    const n_media: i32 = media_token_count;
    const n_embd_val: i64 = @intCast(media_embd_dim);

    const kv_cache_ptr: ?*kv_cache.KVCache = kv_cache_mgr;
    var pp_time_s: f64 = 0.0;

    // 获取 CPU buffer type 用于测量
    const buft = ggml.backendCpuBufferType();

    _ = graph_ctx; // 未使用，每个 Pass 使用独立临时 context

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

    // ===================================================================
    // Pass 1: Text prefix only (causal attention)
    // Positions: 0 .. prefix_len-1
    // ===================================================================
    if (prefix_len > 0) {
        // 使用独立的临时 context（P0: docs/MEMMGT.md §4.2.2）
        // 先构建图来测量大小
        var p1_ctx = try ggml.Context.initNoAlloc(MIN_TEMP_CTX_SIZE);
        defer p1_ctx.deinit();

        p1_ctx.setNoAlloc(false);
        const p1_input = try p1_ctx.newTensor1d(.i32, prefix_len);
        {
            const data = p1_input.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(prefix_len))];
            for (prefix_tokens, 0..) |t, j| {
                slice[j] = @as(i32, @intCast(t));
            }
        }

        var p1_graph = try ggml.CGraph.initReserved(p1_ctx, 16384);
        var p1_builder = graph_builder.GraphBuilder.init(p1_ctx, p1_graph, params, allocator);
        _ = try model_instance.buildGraph(&p1_builder, p1_input, prefix_len, @ptrCast(kv_cache_ptr), 0);
        p1_ctx.setNoAlloc(true);

        // 测量并创建合适大小的 context
        const p1_ctx_size = measureAndSize(p1_graph, buft, "Pass 1");

        // 如果当前 context 太小，重新创建
        if (p1_ctx_size > MIN_TEMP_CTX_SIZE) {
            p1_ctx.deinit();
            p1_ctx = try ggml.Context.initNoAlloc(p1_ctx_size);
            p1_ctx.setNoAlloc(false);
            const p1_input2 = try p1_ctx.newTensor1d(.i32, prefix_len);
            {
                const data = p1_input2.dataBytes();
                const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(prefix_len))];
                for (prefix_tokens, 0..) |t, j| {
                    slice[j] = @as(i32, @intCast(t));
                }
            }
            p1_graph = try ggml.CGraph.initReserved(p1_ctx, 16384);
            p1_builder = graph_builder.GraphBuilder.init(p1_ctx, p1_graph, params, allocator);
            _ = try model_instance.buildGraph(&p1_builder, p1_input2, prefix_len, @ptrCast(kv_cache_ptr), 0);
            p1_ctx.setNoAlloc(true);
        }

        // 使用传入的 gallocr（所有权上移）
        if (!gallocr.allocGraph(p1_graph)) {
            log.err("Graph alloc failed: text-prefix pass", .{});
            return error.GraphAllocFailed;
        }

        const t1_start = engine_common.currentTimeMs();
        try p1_graph.compute(n_threads);
        const t1_end = engine_common.currentTimeMs();
        pp_time_s += @as(f64, @floatFromInt(t1_end - t1_start)) / 1000.0;
        log.debug("Pass 1 (text prefix): {d} tokens in {d:.3}s ✓", .{ prefix_len, @as(f64, @floatFromInt(t1_end - t1_start)) / 1000.0 });
        log.debug("  -> KV cache current_len after Pass 1: {d}", .{kv_cache_mgr.currentLen()});

        // Pass 1 完成，临时 context 由 defer 释放
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

            // 每个 chunk 使用独立的临时 context（P0: docs/MEMMGT.md §4.2.2）
            // 使用 no_alloc=true 模式：张量数据由 gallocr 分配，context 仅存储元数据。
            // 输入张量的数据需要手动分配（通过 setDataPtr）。
            var p2_ctx = try ggml.Context.initNoAlloc(MIN_TEMP_CTX_SIZE);
            defer p2_ctx.deinit();

            // 创建嵌入张量（no_alloc 模式下需要手动分配数据）
            const p2_embd = try p2_ctx.newTensor2d(.f32, n_embd_val, chunk_size);
            {
                const src_bytes = chunk_embd_data.len * @sizeOf(f32);
                const buf = @as([*]u8, @ptrCast(std.c.malloc(src_bytes) orelse return error.OutOfMemory))[0..src_bytes];
                @memcpy(buf, @as([*]const u8, @ptrCast(chunk_embd_data.ptr))[0..src_bytes]);
                p2_embd.setDataPtr(buf);
            }
            p2_embd.setName("mp_embd");

            // 创建输入 token 张量（no_alloc 模式下需要手动分配数据）
            const p2_input = try p2_ctx.newTensor1d(.i32, chunk_size);
            {
                const buf_size = @as(usize, @intCast(chunk_size)) * @sizeOf(i32);
                const buf = @as([*]u8, @ptrCast(std.c.malloc(buf_size) orelse return error.OutOfMemory))[0..buf_size];
                const slice = @as([*]i32, @ptrCast(@alignCast(buf.ptr)))[0..@as(usize, @intCast(chunk_size))];
                @memset(slice, @as(i32, @intCast(media_token_id)));
                p2_input.setDataPtr(buf);
            }

            var p2_graph = try ggml.CGraph.initReserved(p2_ctx, 16384);
            _ = try model_instance.buildMM(
                p2_ctx,
                p2_graph,
                p2_input,
                chunk_size,
                @ptrCast(kv_cache_ptr),
                chunk_pos,
                p2_embd,
                0,
                false, // non-causal
            );

            // 测量并调整 context 大小
            const p2_ctx_size = measureAndSize(p2_graph, buft, "Pass 2 chunk");

            if (p2_ctx_size > MIN_TEMP_CTX_SIZE) {
                p2_ctx.deinit();
                p2_ctx = try ggml.Context.initNoAlloc(p2_ctx_size);

                // 重新创建嵌入张量
                const p2_embd2 = try p2_ctx.newTensor2d(.f32, n_embd_val, chunk_size);
                {
                    const src_bytes = chunk_embd_data.len * @sizeOf(f32);
                    const buf = @as([*]u8, @ptrCast(std.c.malloc(src_bytes) orelse return error.OutOfMemory))[0..src_bytes];
                    @memcpy(buf, @as([*]const u8, @ptrCast(chunk_embd_data.ptr))[0..src_bytes]);
                    p2_embd2.setDataPtr(buf);
                }
                p2_embd2.setName("mp_embd");

                // 重新创建输入 token 张量
                const p2_input2 = try p2_ctx.newTensor1d(.i32, chunk_size);
                {
                    const buf_size = @as(usize, @intCast(chunk_size)) * @sizeOf(i32);
                    const buf = @as([*]u8, @ptrCast(std.c.malloc(buf_size) orelse return error.OutOfMemory))[0..buf_size];
                    const slice = @as([*]i32, @ptrCast(@alignCast(buf.ptr)))[0..@as(usize, @intCast(chunk_size))];
                    @memset(slice, @as(i32, @intCast(media_token_id)));
                    p2_input2.setDataPtr(buf);
                }

                p2_graph = try ggml.CGraph.initReserved(p2_ctx, 16384);
                _ = try model_instance.buildMM(
                    p2_ctx,
                    p2_graph,
                    p2_input2,
                    chunk_size,
                    @ptrCast(kv_cache_ptr),
                    chunk_pos,
                    p2_embd2,
                    0,
                    false,
                );
            }

            // 使用传入的 gallocr（所有权上移）
            if (!gallocr.allocGraph(p2_graph)) {
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

    // 使用独立的临时 context（P0: docs/MEMMGT.md §4.2.2）
    var p3_ctx = try ggml.Context.initNoAlloc(MIN_TEMP_CTX_SIZE);
    defer p3_ctx.deinit();

    p3_ctx.setNoAlloc(false);
    const p3_input = try p3_ctx.newTensor1d(.i32, sfx_n);
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
    var p3_graph = try ggml.CGraph.initReserved(p3_ctx, 16384);
    var p3_builder = graph_builder.GraphBuilder.init(p3_ctx, p3_graph, params, allocator);
    const logits = try model_instance.buildGraph(&p3_builder, p3_input, sfx_n, @ptrCast(kv_cache_ptr), suffix_start_pos);
    p3_ctx.setNoAlloc(true);

    // 测量并调整 context 大小
    const p3_ctx_size = measureAndSize(p3_graph, buft, "Pass 3");

    if (p3_ctx_size > MIN_TEMP_CTX_SIZE) {
        p3_ctx.deinit();
        p3_ctx = try ggml.Context.initNoAlloc(p3_ctx_size);
        p3_ctx.setNoAlloc(false);
        const p3_input2 = try p3_ctx.newTensor1d(.i32, sfx_n);
        {
            const data = p3_input2.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(sfx_n))];
            if (suffix_len > 0) {
                for (suffix_tokens, 0..) |t, j| {
                    slice[j] = @as(i32, @intCast(t));
                }
            } else {
                slice[0] = @as(i32, @intCast(media_token_id));
            }
        }
        p3_graph = try ggml.CGraph.initReserved(p3_ctx, 16384);
        p3_builder = graph_builder.GraphBuilder.init(p3_ctx, p3_graph, params, allocator);
        _ = try model_instance.buildGraph(&p3_builder, p3_input2, sfx_n, @ptrCast(kv_cache_ptr), suffix_start_pos);
        p3_ctx.setNoAlloc(true);
    }

    // 使用传入的 gallocr（所有权上移）
    if (!gallocr.allocGraph(p3_graph)) {
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
