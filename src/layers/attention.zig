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
    /// Attention logit softcapping (0 = disabled). Used by Gemma 2/4.
    attn_logit_softcap: f32 = 0.0,
    /// Whether to use causal masking. Set false for multimodal embedding prefill.
    causal: bool = true,
};

/// 执行缩放点积注意力计算
/// q: [head_dim, n_head, n_tokens] 查询张量（已应用 RoPE）
/// k: [head_dim, n_kv_head, cache_len] 键张量（已应用 RoPE）
/// v: [head_dim, n_kv_head, cache_len] 值张量
/// params: 注意力参数
/// swa_window: 滑动窗口大小（0 或 null 表示全注意力，无 SWA 掩码）
/// 返回: [n_head * head_dim, n_tokens] 注意力输出
pub fn scaledDotProductAttention(
    ctx: *ggml.Context,
    q: *ggml.Tensor,
    k: *ggml.Tensor,
    v: *ggml.Tensor,
    params: AttentionParams,
    swa_window: ?i64,
) *ggml.Tensor {
    const n_head = params.n_head;
    _ = params.n_kv_head; // ggml_mul_mat auto-broadcast handles GQA
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
    var kq = ggml.mulMat(ctx, k_perm, q_perm);

    // Step 3: 缩放
    kq = ggml.scale(ctx, kq, scale_factor);

    // Step 3.5: attention logit softcapping (Gemma 2/4)
    if (params.attn_logit_softcap > 0.0) {
        kq = ggml.scale(ctx, kq, 1.0 / params.attn_logit_softcap);
        kq = ggml.tanh(ctx, kq);
        kq = ggml.scale(ctx, kq, params.attn_logit_softcap);
    }

    // Step 4: masking + softmax
    if (!params.causal) {
        // Non-causal: no mask, just softmax (for multimodal embedding prefill)
        kq = ggml.softMax(ctx, kq);
    } else if (swa_window) |window| {
        // SWA: build sliding-window + causal mask and use fused softmax_ext
        const mask = buildAttentionMask(ctx, params.cache_len, n_tokens, start_pos, window);
        kq = ggml.softMaxExt(ctx, kq, mask, 1.0, 0.0);
    } else {
        // Full attention: causal mask only
        kq = ggml.diagMaskInf(ctx, kq, start_pos);
        kq = ggml.softMax(ctx, kq);
    }

    // Step 5: v = v^T (transpose)
    const v_t_transposed = ggml.transpose(ctx, v_perm);
    const v_ne = v_t_transposed.ne();
    const v_t = ggml.cont4d(ctx, v_t_transposed, v_ne[0], v_ne[1], v_ne[2], v_ne[3]);

    // Step 6: kqv = v @ kq (mul_mat)
    var kqv = ggml.mulMat(ctx, v_t, kq);

    // Step 7: permute(0,2,1,3) 恢复布局
    kqv = ggml.permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml.cont(ctx, kqv);

    // Step 8: 展平为 [n_head * head_dim, n_tokens]
    kqv = ggml.reshape2d(ctx, kqv, n_head * head_dim, n_tokens);

    return kqv;
}

/// 构建 SWA（滑动窗口注意力）掩码张量
fn buildAttentionMask(
    ctx: *ggml.Context,
    cache_len: i64,
    n_tokens: i64,
    start_pos: i32,
    window_size: i64,
) *ggml.Tensor {
    ctx.setNoAlloc(false);
    const mask = ctx.newTensor2d(.f32, cache_len, n_tokens) catch {
        ctx.setNoAlloc(true);
        const fb = ctx.newTensor2d(.f32, 1, 1) catch unreachable;
        ctx.setNoAlloc(true);
        return fb;
    };
    ctx.setNoAlloc(true);

    const n_elems = @as(usize, @intCast(mask.nElems()));
    const data = std.heap.page_allocator.alloc(f32, n_elems) catch {
        return mask;
    };
    defer std.heap.page_allocator.free(data);
    const inf: f32 = -std.math.inf(f32);

    for (0..@as(usize, @intCast(cache_len))) |ci| {
        const cache_pos: i64 = @intCast(ci);
        for (0..@as(usize, @intCast(n_tokens))) |qi| {
            const query_pos: i64 = @as(i64, start_pos) + @as(i64, @intCast(qi));
            const dist: i64 = query_pos - cache_pos;
            if (dist >= 0 and dist < window_size) {
                data[qi * @as(usize, @intCast(cache_len)) + ci] = 0.0;
            } else {
                data[qi * @as(usize, @intCast(cache_len)) + ci] = inf;
            }
        }
    }
    mask.dataSet(f32, data) catch {};

    return mask;
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
        .scale_factor = 0.08838834764831845,
    };
    try testing.expectEqual(@as(i64, 32), p.n_head);
    try testing.expectEqual(@as(i64, 8), p.n_kv_head);
}
