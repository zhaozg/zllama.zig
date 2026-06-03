//! 注意力层（含 GQA 支持）
//!
//! 实现标准缩放点积注意力（Scaled Dot-Product Attention），
//! 支持 Grouped Query Attention (GQA)。

const std = @import("std");
const ggml = @import("../ggml.zig");

/// 注意力计算参数
pub const AttentionParams = struct {
    n_head: i64,
    n_kv_head: i64,
    head_dim: i64,
    n_tokens: i64,
    cache_len: i64,
    start_pos: i32,
    scale_factor: f32,
};

/// 执行缩放点积注意力计算
/// q: [head_dim, n_tokens, n_head] 查询张量（已应用 RoPE）
/// k: [head_dim, cache_len, n_kv_head] 键张量（已应用 RoPE）
/// v: [head_dim, cache_len, n_kv_head] 值张量
/// params: 注意力参数
/// 返回: [n_head * head_dim, n_tokens] 注意力输出
pub fn scaledDotProductAttention(
    ctx: *ggml.Context,
    q: *ggml.Tensor,
    k: *ggml.Tensor,
    v: *ggml.Tensor,
    params: AttentionParams,
) *ggml.Tensor {
    const n_head = params.n_head;
    const n_kv_head = params.n_kv_head;
    const head_dim = params.head_dim;
    const n_tokens = params.n_tokens;
    const cache_len = params.cache_len;
    const start_pos = params.start_pos;
    const scale_factor = params.scale_factor;

    // Q: [head_dim, n_tokens, n_head] -> permute(1,0,2) -> [n_tokens, head_dim, n_head]
    var q_perm = ggml.permute(ctx, q, 1, 0, 2, 3);
    q_perm = ggml.cont(ctx, q_perm);

    // K: [head_dim, cache_len, n_kv_head] -> permute(1,0,2) -> [cache_len, head_dim, n_kv_head]
    var k_perm = ggml.permute(ctx, k, 1, 0, 2, 3);
    k_perm = ggml.cont(ctx, k_perm);

    // V: [head_dim, cache_len, n_kv_head] -> permute(1,0,2) -> [cache_len, head_dim, n_kv_head]
    var v_perm = ggml.permute(ctx, v, 1, 0, 2, 3);
    v_perm = ggml.cont(ctx, v_perm);

    // GQA: 扩展 KV 头以匹配 Q 头数
    if (n_kv_head < n_head) {
        const n_rep = @divExact(n_head, n_kv_head);
        if (n_rep > 1) {
            // K: [cache_len, head_dim, n_kv_head] -> 展平为 [cache_len * head_dim, n_kv_head]
            const k_2d = ggml.reshape2d(ctx, k_perm, cache_len * head_dim, n_kv_head);
            const k_target = ctx.newTensor2d(.f32, cache_len * head_dim, n_head) catch unreachable;
            const k_rep = ggml.cont(ctx, ggml.repeat(ctx, k_2d, k_target));
            k_perm = ggml.reshape3d(ctx, k_rep, cache_len, head_dim, n_head);

            // V: [cache_len, head_dim, n_kv_head] -> 展平为 [cache_len * head_dim, n_kv_head]
            const v_2d = ggml.reshape2d(ctx, v_perm, cache_len * head_dim, n_kv_head);
            const v_target = ctx.newTensor2d(.f32, cache_len * head_dim, n_head) catch unreachable;
            const v_rep = ggml.cont(ctx, ggml.repeat(ctx, v_2d, v_target));
            v_perm = ggml.reshape3d(ctx, v_rep, cache_len, head_dim, n_head);
        }
    }

    // 展平 Q, K 为 2D 进行批量矩阵乘法
    var q_3d = ggml.permute(ctx, q_perm, 1, 0, 2, 3);
    q_3d = ggml.cont(ctx, q_3d);
    var k_3d = ggml.permute(ctx, k_perm, 1, 0, 2, 3);
    k_3d = ggml.cont(ctx, k_3d);

    // score = K^T * Q (batch over n_head)
    var kq = ggml.mulMat(ctx, k_3d, q_3d);

    // 缩放
    kq = ggml.scale(ctx, kq, scale_factor);

    // 因果 mask
    kq = ggml.reshape2d(ctx, kq, cache_len, n_tokens * n_head);
    kq = ggml.diagMaskInf(ctx, kq, start_pos);
    kq = ggml.softMax(ctx, kq);

    // 重塑回 3D: [n_tokens, cache_len, n_head]
    kq = ggml.reshape3d(ctx, kq, cache_len, n_tokens, n_head);

    // V: [cache_len, head_dim, n_head]
    const v_3d = ggml.cont(ctx, v_perm);

    // attn = softmax * V (batch over n_head)
    var attn = ggml.mulMat(ctx, kq, v_3d);

    // 置换为 [n_tokens, n_head, head_dim]
    attn = ggml.permute(ctx, attn, 0, 2, 1, 3);
    attn = ggml.cont(ctx, attn);

    // 展平为 [n_head * head_dim, n_tokens]
    attn = ggml.reshape2d(ctx, attn, n_head * head_dim, n_tokens);

    return attn;
}

const testing = std.testing;

test "AttentionParams basic" {
    const p = AttentionParams{
        .n_head = 32,
        .n_kv_head = 8,
        .head_dim = 128,
        .n_tokens = 1,
        .cache_len = 10,
        .start_pos = 0,
        .scale_factor = 0.08838834764831845, // 1/sqrt(128)
    };
    try testing.expectEqual(@as(i64, 32), p.n_head);
    try testing.expectEqual(@as(i64, 8), p.n_kv_head);
}
