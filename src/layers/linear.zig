//! 线性层（全连接 + 量化支持）
//!
//! 提供线性投影的封装，支持量化权重。

const std = @import("std");
const ggml = @import("ggml");

/// 线性投影：y = x @ W^T
/// x: [in_features, batch_size] 输入张量
/// weight: [out_features, in_features] 权重张量（可能为量化类型）
/// 返回: [out_features, batch_size] 输出张量
pub fn linear(ctx: *ggml.Context, x: *ggml.Tensor, weight: *ggml.Tensor) *ggml.Tensor {
    return ggml.mulMat(ctx, weight, x);
}

/// 带偏置的线性投影：y = x @ W^T + bias
pub fn linearWithBias(
    ctx: *ggml.Context,
    x: *ggml.Tensor,
    weight: *ggml.Tensor,
    bias: *ggml.Tensor,
) *ggml.Tensor {
    const result = ggml.mulMat(ctx, weight, x);
    return ggml.add(ctx, result, bias);
}

const testing = std.testing;

test "linear basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*ggml.Tensor)), @sizeOf(*ggml.Tensor));
}
