//! SwiGLU 前馈网络层
//!
//! 实现 SwiGLU 激活函数的前馈网络：
//! FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down

const std = @import("std");
const ggml = @import("ggml");

/// SwiGLU FFN 前向计算
/// x: [n_embd, n_tokens] 输入张量
/// gate_weight: [n_ff, n_embd] 门控权重
/// up_weight: [n_ff, n_embd] 上投影权重
/// down_weight: [n_embd, n_ff] 下投影权重
/// 返回: [n_embd, n_tokens] 输出张量
pub fn swiGLU(
    ctx: *ggml.Context,
    x: *ggml.Tensor,
    gate_weight: *ggml.Tensor,
    up_weight: *ggml.Tensor,
    down_weight: *ggml.Tensor,
) *ggml.Tensor {
    // gate = silu(x @ W_gate)
    const gate = ggml.silu(ctx, ggml.mulMat(ctx, gate_weight, x));

    // up = x @ W_up
    const up = ggml.mulMat(ctx, up_weight, x);

    // hidden = gate * up
    const hidden = ggml.mul(ctx, gate, up);

    // out = hidden @ W_down
    return ggml.mulMat(ctx, down_weight, hidden);
}

const testing = std.testing;

test "swiGLU basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*ggml.Tensor)), @sizeOf(*ggml.Tensor));
}
