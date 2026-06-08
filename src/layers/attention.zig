//! 注意力层（含 GQA 支持）
//!
//! 实现标准缩放点积注意力（Scaled Dot-Product Attention），
//! 支持 Grouped Query Attention (GQA)。
//!
//! 参考 llama.cpp build_attn_mha 实现。
//!
//! 输入布局（与 llama.cpp 一致）:
//!   q: [head_dim, n_head, n_tokens]  (已应用 RoPE)
//!   k: [head_dim, n_kv_head, cache_len] (已应用 RoPE)
//!   v: [head_dim, n_kv_head, cache_len]
//!
//! 流程（与 llama.cpp build_attn_mha 一致）:
//!   1. permute(0,2,1,3): [head_dim, n_tokens/cache_len, n_head/n_kv_head]
//!   2. kq = k @ q  (mul_mat, ggml 自动处理 GQA 广播)
//!   3. scale, soft_max
//!   4. v = v^T (transpose)
//!   5. kqv = v @ kq  (mul_mat, ggml 自动处理 GQA 广播)
//!   6. permute(0,2,1,3) + reshape 回 [n_head*head_dim, n_tokens]

const std = @import("std");
const ggml = @import("ggml");

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
/// q: [head_dim, n_head, n_tokens] 查询张量（已应用 RoPE）
/// k: [head_dim, n_kv_head, cache_len] 键张量（已应用 RoPE）
/// v: [head_dim, n_kv_head, cache_len] 值张量
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
    _ = params.n_kv_head;
    const head_dim = params.head_dim;
    const n_tokens = params.n_tokens;
    const start_pos = params.start_pos;
    const scale_factor = params.scale_factor;

    // Step 1: permute(0,2,1,3) 与 llama.cpp build_attn_mha 一致
    // Q: [head_dim, n_head, n_tokens] -> [head_dim, n_tokens, n_head]
    var q_perm = ggml.permute(ctx, q, 0, 2, 1, 3);
    q_perm = ggml.cont(ctx, q_perm);

    // K: [head_dim, n_kv_head, cache_len] -> [head_dim, cache_len, n_kv_head]
    var k_perm = ggml.permute(ctx, k, 0, 2, 1, 3);
    k_perm = ggml.cont(ctx, k_perm);

    // V: [head_dim, n_kv_head, cache_len] -> [head_dim, cache_len, n_kv_head]
    var v_perm = ggml.permute(ctx, v, 0, 2, 1, 3);
    v_perm = ggml.cont(ctx, v_perm);

    // Step 2: kq = k @ q (mul_mat)
    // ggml_mul_mat 自动处理 GQA: 当 k.ne[2] != q.ne[2] 时广播 k
    // k_perm: [head_dim, cache_len, n_kv_head] (ne[0]=head_dim, ne[1]=cache_len, ne[2]=n_kv_head)
    // q_perm: [head_dim, n_tokens, n_head]     (ne[0]=head_dim, ne[1]=n_tokens, ne[2]=n_head)
    // 输出: [cache_len, n_tokens, n_head]
    var kq = ggml.mulMat(ctx, k_perm, q_perm);

    // Step 3: 缩放
    kq = ggml.scale(ctx, kq, scale_factor);

    // Step 4: 因果 mask + softmax
    // kq: [cache_len, n_tokens, n_head] (3D)
    // diagMaskInf 对 3D 张量的 ne[0]xne[1] 切片应用 mask
    kq = ggml.diagMaskInf(ctx, kq, start_pos);
    kq = ggml.softMax(ctx, kq);

    // Step 5: v = v^T (transpose)
    // v_perm: [head_dim, cache_len, n_kv_head] -> transpose -> [cache_len, head_dim, n_kv_head]
    // Use explicit cont4d to guarantee contiguous strides (avoids ggml_is_transposed edge case)
    const v_t_transposed = ggml.transpose(ctx, v_perm);
    const v_ne = v_t_transposed.ne();
    const v_t = ggml.cont4d(ctx, v_t_transposed, v_ne[0], v_ne[1], v_ne[2], v_ne[3]);

    // Step 6: kqv = v @ kq (mul_mat)
    // v_t: [cache_len, head_dim, n_kv_head] (ne[0]=cache_len, ne[1]=head_dim, ne[2]=n_kv_head)
    // kq:  [cache_len, n_tokens, n_head]    (ne[0]=cache_len, ne[1]=n_tokens, ne[2]=n_head)
    // ggml_mul_mat 自动处理 GQA: 当 v_t.ne[2] != kq.ne[2] 时广播 v_t
    // 输出: [head_dim, n_tokens, n_head]
    var kqv = ggml.mulMat(ctx, v_t, kq);

    // Step 7: permute(0,2,1,3) 恢复布局
    // kqv: [head_dim, n_tokens, n_head] -> [head_dim, n_head, n_tokens]
    kqv = ggml.permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml.cont(ctx, kqv);

    // Step 8: 展平为 [n_head * head_dim, n_tokens]
    kqv = ggml.reshape2d(ctx, kqv, n_head * head_dim, n_tokens);

    return kqv;
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
