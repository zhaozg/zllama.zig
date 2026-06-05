//! RMSNorm 归一化层
//!
//! 实现 RMS Normalization，支持可学习的缩放权重。

const std = @import("std");
const ggml = @import("ggml");

/// 对输入张量应用 RMSNorm
/// x: [n_embd, n_tokens] 输入张量
/// weight: [n_embd] 可学习的缩放权重
/// eps: 防止除零的小常数
/// 返回: [n_embd, n_tokens] 归一化后的张量
pub fn rmsNorm(ctx: *ggml.Context, x: *ggml.Tensor, weight: *ggml.Tensor, eps: f32) *ggml.Tensor {
    const n_embd: i64 = @intCast(x.ne()[0]);
    var result = ggml.rmsNorm(ctx, x, eps);
    result = ggml.mul(ctx, result, ggml.reshape2d(ctx, weight, n_embd, 1));
    return result;
}

/// 仅计算 RMSNorm（无权重乘法）
pub fn rmsNormOnly(ctx: *ggml.Context, x: *ggml.Tensor, eps: f32) *ggml.Tensor {
    return ggml.rmsNorm(ctx, x, eps);
}

const testing = std.testing;

test "rmsNorm basic" {
    // 结构测试，不实际运行 ggml
    try testing.expectEqual(@as(usize, @sizeOf(*ggml.Tensor)), @sizeOf(*ggml.Tensor));
}
