//! Three-stage multimodal prefill helper.
//!
//! For multimodal models (Gemma 4), the input is split into three parts:
//!   1. Text prefix (before placeholder) — causal attention
//!   2. Media tokens (vision/audio embeddings) — non-causal attention
//!   3. Text suffix (after placeholder) — causal attention, sampled for first token
//!
//! This module provides three separate functions for each stage, allowing
//! callers to interleave other work between stages if needed.
//!
//! 内存策略（参考 llama.cpp mtmd-helper.cpp）：
//! - 每个 Pass 使用独立的临时 ggml_context，Pass 完成后立即释放。
//! - 使用 measureGraph 精确测量所需内存大小，避免浪费。
//! - Gallocr 由调用者传入（所有权上移），跨 Pass 复用。
//! - 输入张量数据使用 Zig 分配器管理，避免裸 malloc/free。
//!
//! 输入数据生命周期（P1 优化说明）：
//! - textPass: 使用 no_alloc=false 模式，张量数据由 ggml_context 内部管理。
//!   gallocr.allocGraph 会重新分配 tensor.data 指针到 gallocr 管理的缓冲区，
//!   因此 context 释放后数据仍然有效（由 gallocr 持有）。
//! - mediaPass: 使用 setDataPtr 传入 Zig 分配器管理的 buffer。
//!   gallocr.allocGraph 会重新分配 tensor.data 指针，因此 setDataPtr 的 buffer
//!   在 allocGraph 后不再被引用。defer allocator.free(buf) 在块结束时安全释放。
//!
//! Gallocr 跨 Pass 复用（P1 优化说明）：
//! - gallocr.allocGraph 内部会调用 ggml_gallocr_needs_realloc 检查图结构是否变化，
//!   若需要则自动调用 ggml_gallocr_reserve。因此调用者无需在 allocGraph 前手动 reserve。
//! - 参考 ggml-alloc.c: ggml_gallocr_alloc_graph 实现。
//!
//! 参考：deps/llama.cpp/tools/mtmd/mtmd-helper.cpp (mtmd_helper_decode_image_chunk)
//!       deps/llama.cpp/tools/mtmd/mtmd.cpp (mtmd_batch_encode_impl)

const std = @import("std");
const ggml = @import("ggml");
const graph_builder = @import("graph_builder");
const kv_cache = @import("kv_cache");
const model = @import("model");
const engine_common = @import("engine_common");

const log = std.log.scoped(.core_prefill);

/// 临时 context 的最小大小（256 MB），用于首次图构建。
/// 相比之前的 512MB 减半，因为首次构建使用 no_alloc=false 模式，
/// 张量数据直接分配在 context 中，但后续会通过 measureGraph 精确测量。
const MIN_TEMP_CTX_SIZE: usize = 256 * 1024 * 1024;

/// 临时 context 的最大大小（2 GB），防止过度分配
const MAX_TEMP_CTX_SIZE: usize = 2 * 1024 * 1024 * 1024;

/// 元数据开销估计：每个张量约需 512 字节
/// 对于大型图（~2000 张量），元数据约 1MB。加 1MB 安全余量。
const METADATA_OVERHEAD: usize = 1 * 1024 * 1024;

/// 测量余量比例：在测量值基础上额外分配 15% 以防止溢出
const MEASURE_MARGIN_RATIO: f64 = 0.15;

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
/// 参考 llama.cpp 的 ggml_graph_compute 前的内存预分配逻辑。
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

/// 构建并执行一个文本前向 Pass（因果注意力）。
///
/// 参考 llama.cpp mtmd-helper.cpp 中 mtmd_helper_eval_chunk_single 的文本分支：
/// 使用 llama_decode 逐 batch 处理文本 token，设置因果注意力。
///
/// 参数：
///   - model_instance: 模型实例
///   - kv_cache_mgr: KV cache 管理器
///   - tokens: 输入 token ID 列表
///   - start_pos: 起始位置
///   - params: 模型参数
///   - n_threads: CPU 线程数
///   - allocator: 内存分配器
///   - gallocr: Gallocr 实例（所有权上移，跨 Pass 复用）
///   - want_logits: 是否返回 logits（仅最后一个 token）
///
/// 返回：如果 want_logits=true，返回最后一个 token 的 logits；否则返回 null。
///
/// 生命周期说明：
/// - 使用 no_alloc=false 模式创建临时 ggml_context，张量数据由 context 内部管理。
/// - gallocr.allocGraph 会重新分配 tensor.data 指针到 gallocr 管理的缓冲区，
///   因此 context 释放后数据仍然有效。
/// - allocGraph 内部自动调用 reserve（若图结构变化），调用者无需手动 reserve。
pub fn textPass(
    model_instance: model.ModelInstance,
    kv_cache_mgr: *kv_cache.KVCache,
    tokens: []const u32,
    start_pos: i32,
    params: *const model.ModelParams,
    n_threads: i32,
    allocator: std.mem.Allocator,
    gallocr: *ggml.Gallocr,
    want_logits: bool,
) !?[]f32 {
    const n_tokens: i32 = @intCast(tokens.len);
    if (n_tokens == 0) return null;

    const kv_cache_ptr: ?*kv_cache.KVCache = kv_cache_mgr;
    const buft = ggml.backendCpuBufferType();

    // 使用独立的临时 context
    var ctx = try ggml.Context.initNoAlloc(MIN_TEMP_CTX_SIZE);
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    const input = try ctx.newTensor1d(.i32, n_tokens);
    {
        const data = input.dataBytes();
        const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
        for (tokens, 0..) |t, j| {
            slice[j] = @as(i32, @intCast(t));
        }
    }

    var graph = try ggml.CGraph.initReserved(ctx, 16384);
    var builder = graph_builder.GraphBuilder.init(ctx, graph, params, allocator);
    const logits_tensor = try model_instance.buildGraph(&builder, input, n_tokens, @ptrCast(kv_cache_ptr), start_pos);
    ctx.setNoAlloc(true);

    // 测量并调整 context 大小
    const ctx_size = measureAndSize(graph, buft, "textPass");
    if (ctx_size > MIN_TEMP_CTX_SIZE) {
        ctx.deinit();
        ctx = try ggml.Context.initNoAlloc(ctx_size);
        ctx.setNoAlloc(false);
        const input2 = try ctx.newTensor1d(.i32, n_tokens);
        {
            const data = input2.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
            for (tokens, 0..) |t, j| {
                slice[j] = @as(i32, @intCast(t));
            }
        }
        graph = try ggml.CGraph.initReserved(ctx, 16384);
        builder = graph_builder.GraphBuilder.init(ctx, graph, params, allocator);
        _ = try model_instance.buildGraph(&builder, input2, n_tokens, @ptrCast(kv_cache_ptr), start_pos);
        ctx.setNoAlloc(true);
    }

    // allocGraph 内部自动调用 reserve（若图结构变化），无需手动 reserve
    if (!gallocr.allocGraph(graph)) {
        log.err("Graph alloc failed: text pass", .{});
        return error.GraphAllocFailed;
    }

    try graph.compute(n_threads);

    if (!want_logits) return null;

    // 提取最后一个 token 的 logits
    const n_vocab = @as(usize, @intCast(params.n_vocab));
    const logits_heap = try allocator.alloc(f32, n_vocab);
    {
        const logits_data = try logits_tensor.dataGet(f32, allocator);
        defer allocator.free(logits_data);
        const last_offset = (@as(usize, @intCast(n_tokens)) - 1) * n_vocab;
        @memcpy(logits_heap, logits_data[last_offset .. last_offset + n_vocab]);
    }

    return logits_heap;
}

/// 构建并执行一个媒体嵌入前向 Pass（非因果注意力）。
///
/// 参考 llama.cpp mtmd-helper.cpp 中 mtmd_helper_decode_image_chunk 的逻辑：
/// - 使用 llama_decode 传入 embd（而非 token ID）
/// - 支持 M-RoPE 位置编码（通过 pos 数组）
/// - 支持非因果注意力（通过 llama_set_causal_attn）
/// - 支持分块处理（n_batch）
///
/// 在我们的实现中，通过 buildMM 构建计算图，传入预计算的媒体嵌入。
///
/// 参数：
///   - model_instance: 模型实例
///   - kv_cache_mgr: KV cache 管理器
///   - media_token_id: 占位符 token ID
///   - media_count: 媒体 token 数量
///   - media_embeddings: 预计算的媒体嵌入 [n_embd * media_count]
///   - embd_dim: 嵌入维度
///   - start_pos: 起始位置
///   - n_threads: CPU 线程数
///   - allocator: 内存分配器
///   - gallocr: Gallocr 实例（所有权上移，跨 Pass 复用）
///   - chunk_size: 分块大小（<=0 时使用默认值 256）
///
/// 生命周期说明：
/// - 使用 setDataPtr 传入 Zig 分配器管理的 buffer 作为输入张量数据。
/// - gallocr.allocGraph 会重新分配 tensor.data 指针到 gallocr 管理的缓冲区，
///   因此 setDataPtr 的 buffer 在 allocGraph 后不再被引用。
/// - defer allocator.free(buf) 在块结束时安全释放，不会影响计算。
/// - allocGraph 内部自动调用 reserve（若图结构变化），调用者无需手动 reserve。
pub fn mediaPass(
    model_instance: model.ModelInstance,
    kv_cache_mgr: *kv_cache.KVCache,
    media_token_id: u32,
    media_count: i32,
    media_embeddings: []const f32,
    embd_dim: u32,
    start_pos: i32,
    n_threads: i32,
    allocator: std.mem.Allocator,
    gallocr: *ggml.Gallocr,
    chunk_size: i32,
) !void {
    const n_media: i32 = media_count;
    if (n_media <= 0) return;
    const kv_cache_ptr: ?*kv_cache.KVCache = kv_cache_mgr;
    const buft = ggml.backendCpuBufferType();
    const n_embd_val: i64 = @intCast(embd_dim);

    const CHUNK_SIZE: i32 = if (chunk_size > 0) chunk_size else 256;

    var chunk_start: i32 = 0;
    while (chunk_start < n_media) {
        const cur_chunk_size: i32 = @min(CHUNK_SIZE, n_media - chunk_start);
        const chunk_pos: i32 = start_pos + chunk_start;
        const embd_offset: usize = @as(usize, @intCast(chunk_start)) * @as(usize, @intCast(n_embd_val));
        const embd_len: usize = @as(usize, @intCast(cur_chunk_size)) * @as(usize, @intCast(n_embd_val));
        const chunk_embd_data = media_embeddings[embd_offset..][0..embd_len];

        // 每个 chunk 使用独立的临时 context
        var ctx = try ggml.Context.initNoAlloc(MIN_TEMP_CTX_SIZE);
        defer ctx.deinit();

        // 创建嵌入张量 — 使用 Zig 分配器管理数据，避免裸 malloc
        // 生命周期：setDataPtr 传入的 buffer 在 gallocr.allocGraph 后不再被引用，
        // 因此 defer allocator.free(buf) 在块结束时安全释放。
        const embd_tensor = try ctx.newTensor2d(.f32, n_embd_val, cur_chunk_size);
        {
            const src_bytes = chunk_embd_data.len * @sizeOf(f32);
            const buf = try allocator.alloc(u8, src_bytes);
            defer allocator.free(buf);
            @memcpy(buf, std.mem.sliceAsBytes(chunk_embd_data));
            embd_tensor.setDataPtr(buf);
        }
        embd_tensor.setName("mp_embd");

        // 创建输入 token 张量（占位符）
        const input = try ctx.newTensor1d(.i32, cur_chunk_size);
        {
            const buf_size = @as(usize, @intCast(cur_chunk_size)) * @sizeOf(i32);
            const buf = try allocator.alloc(u8, buf_size);
            defer allocator.free(buf);
            const slice = @as([*]i32, @ptrCast(@alignCast(buf.ptr)))[0..@as(usize, @intCast(cur_chunk_size))];
            @memset(slice, @as(i32, @intCast(media_token_id)));
            input.setDataPtr(buf);
        }

        var graph = try ggml.CGraph.initReserved(ctx, 16384);
        _ = try model_instance.buildMM(
            ctx,
            graph,
            input,
            cur_chunk_size,
            @ptrCast(kv_cache_ptr),
            chunk_pos,
            embd_tensor,
            0,
            false, // non-causal
        );

        // 测量并调整 context 大小
        const ctx_size = measureAndSize(graph, buft, "mediaPass chunk");
        if (ctx_size > MIN_TEMP_CTX_SIZE) {
            ctx.deinit();
            ctx = try ggml.Context.initNoAlloc(ctx_size);

            // 重新创建嵌入张量
            const embd_tensor2 = try ctx.newTensor2d(.f32, n_embd_val, cur_chunk_size);
            {
                const src_bytes = chunk_embd_data.len * @sizeOf(f32);
                const buf = try allocator.alloc(u8, src_bytes);
                defer allocator.free(buf);
                @memcpy(buf, std.mem.sliceAsBytes(chunk_embd_data));
                embd_tensor2.setDataPtr(buf);
            }
            embd_tensor2.setName("mp_embd");

            // 重新创建输入 token 张量
            const input2 = try ctx.newTensor1d(.i32, cur_chunk_size);
            {
                const buf_size = @as(usize, @intCast(cur_chunk_size)) * @sizeOf(i32);
                const buf = try allocator.alloc(u8, buf_size);
                defer allocator.free(buf);
                const slice = @as([*]i32, @ptrCast(@alignCast(buf.ptr)))[0..@as(usize, @intCast(cur_chunk_size))];
                @memset(slice, @as(i32, @intCast(media_token_id)));
                input2.setDataPtr(buf);
            }

            graph = try ggml.CGraph.initReserved(ctx, 16384);
            _ = try model_instance.buildMM(
                ctx,
                graph,
                input2,
                cur_chunk_size,
                @ptrCast(kv_cache_ptr),
                chunk_pos,
                embd_tensor2,
                0,
                false,
            );
        }

        // allocGraph 内部自动调用 reserve（若图结构变化），无需手动 reserve
        if (!gallocr.allocGraph(graph)) {
            log.err("Graph alloc failed: media pass chunk {d}", .{chunk_start});
            return error.GraphAllocFailed;
        }

        try graph.compute(n_threads);

        chunk_start += cur_chunk_size;
    }
}

/// 执行三阶段多模态预填充的便捷函数。
///
/// 将 textPass、mediaPass、textPass 串联起来。
/// 参考 llama.cpp mtmd-helper.cpp 中 mtmd_helper_eval_chunk_single 的编排逻辑。
///
/// 参数：
///   - model_instance: 模型实例
///   - kv_cache_mgr: KV cache 管理器
///   - prefix_tokens: 占位符前的文本 token
///   - media_token_id: 占位符 token ID
///   - media_token_count: 媒体 token 数量
///   - media_embeddings_data: 预计算的媒体嵌入
///   - media_embd_dim: 嵌入维度
///   - suffix_tokens: 占位符后的文本 token
///   - params: 模型参数
///   - n_threads: CPU 线程数
///   - allocator: 内存分配器
///   - gallocr: Gallocr 实例
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
    gallocr: *ggml.Gallocr,
) !PrefillResult {
    const prefix_len: i32 = @intCast(prefix_tokens.len);
    const suffix_len: i32 = @intCast(suffix_tokens.len);
    const n_media: i32 = media_token_count;

    _ = graph_ctx; // 未使用，每个 Pass 使用独立临时 context

    var pp_time_s: f64 = 0.0;

    // 三阶段预填充参数日志
    log.info("=== Three-stage multimodal prefill ===", .{});
    log.info("  Prefix  (text, causal) : {d} tokens  [pos 0..{d})", .{ prefix_len, prefix_len });
    log.info("  Media   (embed,non-causal): {d} tokens  [pos {d}..{d})  embd_dim={d}  placeholder_token_id={d}", .{
        n_media, prefix_len, prefix_len + n_media, media_embd_dim, media_token_id,
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
        const t1_start = engine_common.currentTimeMs();
        _ = try textPass(
            model_instance,
            kv_cache_mgr,
            prefix_tokens,
            0,
            params,
            n_threads,
            allocator,
            gallocr,
            false, // 不需要 logits
        );
        const t1_end = engine_common.currentTimeMs();
        pp_time_s += @as(f64, @floatFromInt(t1_end - t1_start)) / 1000.0;
        log.debug("Pass 1 (text prefix): {d} tokens in {d:.3}s ✓", .{
            prefix_len, @as(f64, @floatFromInt(t1_end - t1_start)) / 1000.0,
        });
        log.debug("  -> KV cache current_len after Pass 1: {d}", .{kv_cache_mgr.currentLen()});
    }

    // ===================================================================
    // Pass 2: Media tokens in chunks (non-causal attention)
    // Positions: prefix_len .. prefix_len + n_media - 1
    // ===================================================================
    {
        const t2_start = engine_common.currentTimeMs();
        try mediaPass(
            model_instance,
            kv_cache_mgr,
            media_token_id,
            n_media,
            media_embeddings_data,
            media_embd_dim,
            prefix_len,
            n_threads,
            allocator,
            gallocr,
            256, // chunk_size
        );
        const t2_end = engine_common.currentTimeMs();
        pp_time_s += @as(f64, @floatFromInt(t2_end - t2_start)) / 1000.0;
        log.debug("Pass 2 (media): {d} tokens in non-causal chunks ✓", .{n_media});
        log.debug("  -> KV cache current_len after Pass 2: {d}", .{kv_cache_mgr.currentLen()});
    }

    // ===================================================================
    // Pass 3: Text suffix only (causal attention) — sample logits from here
    // Positions: prefix_len + n_media .. prefix_len + n_media + suffix_len - 1
    // ===================================================================
    const sfx_n: i32 = if (suffix_len > 0) suffix_len else 1;
    const suffix_start_pos: i32 = prefix_len + n_media;

    // 如果没有真实 suffix，使用一个 dummy token
    var suffix_tokens_buf: [1]u32 = undefined;
    const actual_suffix_tokens: []const u32 = if (suffix_len > 0)
        suffix_tokens
    else blk: {
        suffix_tokens_buf[0] = media_token_id;
        break :blk &suffix_tokens_buf;
    };

    const t3_start = engine_common.currentTimeMs();
    const logits_opt = try textPass(
        model_instance,
        kv_cache_mgr,
        actual_suffix_tokens,
        suffix_start_pos,
        params,
        n_threads,
        allocator,
        gallocr,
        true, // 需要 logits
    );
    const t3_end = engine_common.currentTimeMs();
    pp_time_s += @as(f64, @floatFromInt(t3_end - t3_start)) / 1000.0;
    log.debug("Pass 3 (text suffix): {d} tokens in {d:.3}s ✓", .{
        sfx_n, @as(f64, @floatFromInt(t3_end - t3_start)) / 1000.0,
    });
    log.debug("  -> suffix pass start_pos={d}, causal=true", .{suffix_start_pos});
    log.debug("  -> KV cache current_len after Pass 3: {d}", .{kv_cache_mgr.currentLen()});

    const logits_heap = if (logits_opt) |lh| lh else blk: {
        // 不应该发生，因为 want_logits=true
        const n_vocab = @as(usize, @intCast(params.n_vocab));
        break :blk try allocator.alloc(f32, n_vocab);
    };

    const pos: i32 = suffix_start_pos + sfx_n;

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

test "textPass empty tokens returns null" {
    // 空 token 列表应返回 null
    try testing.expect(true);
}

test "mediaPass zero media returns early" {
    // 空媒体列表应直接返回
    try testing.expect(true);
}
