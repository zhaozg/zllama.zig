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
    _ = n_batch;
    _ = name;

    // 形状断言
    std.debug.assert(q_cur.ne()[1] == n_head);
    std.debug.assert(k_cur.ne()[0] == d_head);
    std.debug.assert(k_cur.ne()[1] == n_head);
    std.debug.assert(k_cur.ne()[2] == n_patches);
    std.debug.assert(v_cur.ne()[0] == d_head);
    std.debug.assert(v_cur.ne()[1] == n_head);
    std.debug.assert(v_cur.ne()[2] == n_patches);

    // 展平 head 和 batch 维度以便计算
    // Q: [d_head, n_head, n_patches, n_batch] → [d_head, n_head * n_batch, n_patches]
    // 但 ggml 的 mul_mat 在 ne[0] 维度收缩，所以我们需要 permute
    // 标准做法: permute Q/K 到 [n_patches, d_head, n_head, n_batch] 然后 mul_mat

    // QK^T: [n_patches, d_head] @ [d_head, n_patches] → [n_patches, n_patches]
    // 对每个 head 和 batch 分别计算

    // 方法: 使用 ggml_flash_attn_ext 或手动计算
    // 这里使用手动计算以保持兼容性

    // 1. 将 Q, K, V 重塑为 [d_head, n_patches, n_head * n_batch]
    //    通过 permute(2, 0, 1, 3) + cont + reshape
    const q_flat = q_cur.permute(ctx, 2, 0, 1, 3).cont(ctx);
    const k_flat = k_cur.permute(ctx, 2, 0, 1, 3).cont(ctx);
    const v_flat = v_cur.permute(ctx, 2, 0, 1, 3).cont(ctx);

    // 现在形状: [n_patches, d_head, n_head * n_batch]
    // 对每个 head*batch 计算注意力

    // 2. Q @ K^T: 对每个 head*batch
    //    Q: [n_patches, d_head], K: [n_patches, d_head]
    //    scores = Q @ K^T = mul_mat(K^T, Q^T)^T
    //    更简单: 使用 ggml_mul_mat

    // 将 Q 转置为 [d_head, n_patches] 用于 mul_mat
    const q_t = q_flat.permute(ctx, 1, 0, 2, 3).cont(ctx);
    const k_t = k_flat.permute(ctx, 1, 0, 2, 3).cont(ctx);

    // scores = Q^T @ K = [n_patches, d_head]^T @ [n_patches, d_head] → [d_head, d_head]? 不对
    // ggml_mul_mat(A, B) = A^T @ B
    // 所以 mul_mat(Q_t, k_t) = Q_t^T @ k_t = [n_patches, d_head] @ [d_head, n_patches] = [n_patches, n_patches]
    // 但 Q_t 是 [d_head, n_patches], k_t 是 [d_head, n_patches]
    // mul_mat(Q_t, k_t) = Q_t^T @ k_t = [n_patches, d_head] @ [d_head, n_patches] = [n_patches, n_patches]
    // 正确!

    var scores = q_t.mulMat(ctx, k_t);

    // 3. Scale
    scores = scores.scale(ctx, kq_scale);

    // 4. Mask (optional)
    if (kq_mask) |mask| {
        scores = scores.add(ctx, mask);
    }

    // 5. Softmax
    scores = scores.softMax(ctx);

    // 6. scores @ V
    // V: [d_head, n_patches] (per head*batch)
    // scores: [n_patches, n_patches]
    // result = V @ scores^T = (scores @ V^T)^T
    // mul_mat(V_t, scores) = V_t^T @ scores = [n_patches, d_head] @ [n_patches, n_patches] = [n_patches, d_head]? 不对
    //
    // 正确: scores @ V = [n_patches, n_patches] @ [n_patches, d_head] = [n_patches, d_head]
    // mul_mat(V^T, scores^T) = V @ scores^T = (scores @ V^T)^T
    // 所以: mul_mat(scores, V_t^T) = scores^T @ V_t = [n_patches, n_patches] @ [d_head, n_patches] = [n_patches, d_head]? 不对
    //
    // 标准做法: scores @ V
    // V 是 [d_head, n_patches], 需要转置为 [n_patches, d_head]
    const v_t = v_flat.permute(ctx, 1, 0, 2, 3).cont(ctx);

    // scores @ V: [n_patches, n_patches] @ [n_patches, d_head] = [n_patches, d_head]
    // mul_mat(V_t, scores) = V_t^T @ scores = [n_patches, d_head] @ [n_patches, n_patches] = [n_patches, d_head]? 不对
    // mul_mat(scores, V_t) = scores^T @ V_t = [n_patches, n_patches] @ [d_head, n_patches] = [n_patches, d_head]? 不对
    //
    // 使用 ggml_mul_mat: mul_mat(A, B) = A^T @ B
    // 我们需要 scores @ V = (V^T @ scores^T)^T
    // mul_mat(V_t, scores) = V_t^T @ scores = [n_patches, d_head] @ [n_patches, n_patches] = [n_patches, d_head]
    // 不对，维度不匹配
    //
    // 正确: scores [n_patches, n_patches], V [d_head, n_patches]
    // scores @ V^T = [n_patches, n_patches] @ [n_patches, d_head] = [n_patches, d_head]
    // 所以: mul_mat(V, scores^T) = V^T @ scores^T = [n_patches, d_head] @ [n_patches, n_patches] = [n_patches, d_head]? 不对
    //
    // 最简单: scores @ V
    // scores [n_patches, n_patches], V [d_head, n_patches]
    // 结果 [n_patches, d_head]
    // 使用 ggml_mul_mat: mul_mat(V, scores) = V^T @ scores = [n_patches, d_head] @ [n_patches, n_patches]
    // 维度: V^T [n_patches, d_head], scores [n_patches, n_patches] → 不匹配!
    //
    // 正确方式: scores @ V = (V^T @ scores^T)^T
    // scores^T [n_patches, n_patches], V^T [n_patches, d_head]
    // mul_mat(V, scores^T) = V^T @ scores^T = [n_patches, d_head] @ [n_patches, n_patches] = [n_patches, d_head]
    // 不对，[n_patches, d_head] @ [n_patches, n_patches] 维度不匹配
    //
    // 重新思考: scores [n_patches, n_patches], V [d_head, n_patches]
    // scores @ V = [n_patches, n_patches] @ [d_head, n_patches] → 维度不匹配!
    // V 需要是 [n_patches, d_head]
    //
    // 所以: V_t [n_patches, d_head], scores [n_patches, n_patches]
    // scores^T @ V_t = [n_patches, n_patches] @ [n_patches, d_head] = [n_patches, d_head]
    // mul_mat(V_t, scores) = V_t^T @ scores = [d_head, n_patches] @ [n_patches, n_patches] = [d_head, n_patches]
    // 然后转置回 [n_patches, d_head]

    var attn_out = v_t.mulMat(ctx, scores);

    // attn_out: [d_head, n_patches] (per head*batch)
    // 转置回 [n_patches, d_head]
    attn_out = attn_out.permute(ctx, 1, 0, 2, 3).cont(ctx);

    // 7. 重塑回 [d_head * n_head, n_patches] = [n_embd, n_patches]
    const n_embd = d_head * n_head;
    attn_out = attn_out.reshape2d(ctx, n_embd, n_patches);

    // 8. 输出投影
    var result = wo.mulMat(ctx, attn_out);

    if (wo_b) |b| {
        result = result.add(ctx, b);
    }

    // 9. Attention sinks (optional)
    if (sinks) |s| {
        _ = s;
        // TODO: implement attention sinks
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

    @memset(wo.dataF32(), 0.1);
    @memset(q.dataF32(), 0.5);
    @memset(k.dataF32(), 0.5);
    @memset(v.dataF32(), 0.5);

    const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    const result = try buildAttn(&ctx, wo, null, q, k, v, null, kq_scale, n_head, "test_attn", null);
    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
}
