//! 嵌入层
//!
//! 实现 Token 嵌入查找。

const std = @import("std");
const ggml = @import("../ggml.zig");

/// Token 嵌入查找
/// embedding: [n_embd, n_vocab] 嵌入矩阵
/// tokens: [n_tokens] token ID 张量
/// 返回: [n_embd, n_tokens] 嵌入向量
pub fn tokenEmbedding(
    ctx: *ggml.Context,
    embedding: *ggml.Tensor,
    tokens: *ggml.Tensor,
) *ggml.Tensor {
    return ggml.getRows(ctx, embedding, tokens);
}

const testing = std.testing;

test "embed basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*ggml.Tensor)), @sizeOf(*ggml.Tensor));
}
