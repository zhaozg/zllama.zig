//! 注意力层构建器
//!
//! 提供多头自注意力（Multi-Head Self-Attention）的 ggml 计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_attn()

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.graph_attn);

/// 构建多头注意力层
///
/// 参数:
///   - ctx: ggml 上下文
///   - wo: 输出投影权重 [n_embd, n_embd]
///   - wo_b: 输出投影偏置 [n_embd]（可选）
///   - q_cur: Q 张量 [d_head, n_head, n_patches, n_batch]
///   - k_cur: K 张量 [d_head, n_head, n_patches, n_batch]
///   - v_cur: V 张量 [d_head, n_head, n_patches, n_batch]
///   - kq_mask: KQ 掩码 [n_patches, n_patches]（可选）
///   - kq_scale: KQ 缩放因子（通常 1/sqrt(d_head)）
///   - n_head: 注意力头数
///   - name: 张量名称前缀（用于调试）
///   - sinks: attention sinks [n_embd, n_sinks]（可选）
///
/// 返回: 注意力输出张量 [n_embd, n_patches]
///
/// 参考: clip-graph.h build_attn()
pub fn buildAttn(
    ctx: *ggml.Context,
    wo: *ggml.Tensor,
    wo_b: ?*ggml.Tensor,
    q_cur: *ggml.Tensor,
    k_cur: *ggml.Tensor,
    v_cur: *ggml.Tensor,
    kq_mask: ?*ggml.Tensor,
    kq_scale: f32,
    n_head: i64,
    name: []const u8,
    sinks: ?*ggml.Tensor,
) !*ggml.Tensor {
    const d_head = q_cur.ne()[0];
    const n_patches = q_cur.ne()[2];
    const n_batch = q_cur.ne()[3];
    _ = name;

    // 形状断言
    std.debug.assert(q_cur.ne()[1] == n_head);
    std.debug.assert(k_cur.ne()[0] == d_head);
    std.debug.assert(k_cur.ne()[1] == n_head);
    std.debug.assert(k_cur.ne()[2] == n_patches);
    std.debug.assert(v_cur.ne()[0] == d_head);
    std.debug.assert(v_cur.ne()[1] == n_head);
    std.debug.assert(v_cur.ne()[2] == n_patches);

    // 使用 ggml_flash_attn_ext（参考 llama.cpp clip.cpp build_attn 的 flash attention 路径）
    //
    // Q, K, V: [d_head, n_head, n_patches, n_batch]
    // Q, K: permute(0, 2, 1, 3) -> [d_head, n_patches, n_head, n_batch]
    // V:    permute(0, 2, 1, 3) -> [d_head, n_patches, n_head, n_batch]
    // flash_attn_ext(Q, K, V, mask, scale, 0, 0)
    // 结果: [d_head, n_patches, n_head, n_batch]
    // reshape_2d: [d_head * n_patches, n_head * n_batch] — 不对
    //
    // 实际上 flash_attn_ext 的结果形状是 [v->ne[0], q->ne[2], q->ne[1], q->ne[3]]
    // = [d_head, n_patches, n_head, n_batch]
    // 然后 reshape_2d: [d_head * n_patches, n_head * n_batch] — 不对
    //
    // 我们需要 [n_embd, n_patches * n_batch]
    // 所以需要 permute 结果: [d_head, n_patches, n_head, n_batch] -> [n_head, d_head, n_patches, n_batch]
    // 然后 cont_2d: [n_embd, n_patches * n_batch]

    const n_embd = d_head * n_head;

    // Q, K, V: permute(0, 2, 1, 3) -> [d_head, n_patches, n_head, n_batch]
    const q = q_cur.permute(ctx, 0, 2, 1, 3).cont(ctx);
    const k = k_cur.permute(ctx, 0, 2, 1, 3).cont(ctx);
    const v = v_cur.permute(ctx, 0, 2, 1, 3).cont(ctx);

    // flash_attn_ext(Q, K, V, mask, scale, max_bias, logit_softcap)
    // 结果: [d_head, n_patches, n_head, n_batch]
    var cur = ggml.flashAttnExt(ctx, q, k, v, kq_mask, kq_scale, 0.0, 0.0);

    // permute 到 [n_head, d_head, n_patches, n_batch]
    cur = cur.permute(ctx, 2, 0, 1, 3).cont(ctx);

    // 展平 head 和 d_head 维度: [n_embd, n_patches, n_batch]
    cur = ggml.cont2d(ctx, cur, n_embd, n_patches * n_batch);

    // 输出投影
    var result = wo.mulMat(ctx, cur);

    if (wo_b) |b| {
        result = result.add(ctx, b);
    }

    // Attention sinks (optional)
    if (sinks) |s| {
        _ = s;
        log.warn("Attention sinks not yet implemented", .{});
    }

    return result;
}

test "buildAttn: basic self-attention" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_embd: i64 = 64;
    const n_head: i64 = 4;
    const d_head: i64 = n_embd / n_head;
    const n_patches: i64 = 16;
    const n_batch: i64 = 1;

    const wo = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd);
    const q = try ctx.newTensor4d(ggml.Type.f32, d_head, n_head, n_patches, n_batch);
    const k = try ctx.newTensor4d(ggml.Type.f32, d_head, n_head, n_patches, n_batch);
    const v = try ctx.newTensor4d(ggml.Type.f32, d_head, n_head, n_patches, n_batch);

    {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(wo.nElems())));
        defer allocator.free(buf);
        @memset(buf, 0.1);
        try wo.dataSet(f32, buf);
    }
    for ([_]*ggml.Tensor{ q, k, v }) |t| {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(t.nElems())));
        defer allocator.free(buf);
        @memset(buf, 0.5);
        try t.dataSet(f32, buf);
    }
    const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    const result = try buildAttn(&ctx, wo, null, q, k, v, null, kq_scale, n_head, "test_attn", null);
    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
}
