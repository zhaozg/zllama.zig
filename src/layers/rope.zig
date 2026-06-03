//! RoPE 位置编码层
//!
//! 实现旋转位置编码（Rotary Position Embedding）。
//! 支持 Qwen 3.5 的扩展 RoPE 参数。

const std = @import("std");
const ggml = @import("../ggml.zig");

/// RoPE 配置参数
pub const RopeParams = struct {
    rope_dim: i64 = 64,
    rope_theta: f32 = 10000000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_scaling_type: []const u8 = "",
    original_max_seq_len: u32 = 32768,
};

/// 对 Q 和 K 张量应用 RoPE 位置编码
/// q: [head_dim, n_tokens, n_head] 查询张量
/// k: [head_dim, n_tokens, n_kv_head] 键张量
/// pos_tensor: [n_tokens] 位置索引
/// params: RoPE 配置参数
/// 返回: { q_rope, k_rope }
pub fn applyRope(
    ctx: *ggml.Context,
    q: *ggml.Tensor,
    k: *ggml.Tensor,
    pos_tensor: *ggml.Tensor,
    params: RopeParams,
) struct { q: *ggml.Tensor, k: *ggml.Tensor } {
    const rope_dim = params.rope_dim;
    const rope_theta = params.rope_theta;
    const freq_scale = params.rope_scaling_factor;

    // Q: permute(0, 2, 1, 3) -> [n_tokens, head_dim, n_head]
    var q_rope = ggml.permute(ctx, q, 0, 2, 1, 3);
    q_rope = ggml.cont(ctx, q_rope);
    q_rope = ggml.ropeExt(ctx, q_rope, pos_tensor, 0, @intCast(rope_dim), 0, rope_theta, freq_scale, 0.0, 1.0, 0.0, 0.0);
    q_rope = ggml.permute(ctx, q_rope, 0, 2, 1, 3);
    q_rope = ggml.cont(ctx, q_rope);

    // K: permute(0, 2, 1, 3) -> [n_tokens, head_dim, n_kv_head]
    var k_rope = ggml.permute(ctx, k, 0, 2, 1, 3);
    k_rope = ggml.cont(ctx, k_rope);
    k_rope = ggml.ropeExt(ctx, k_rope, pos_tensor, 0, @intCast(rope_dim), 0, rope_theta, freq_scale, 0.0, 1.0, 0.0, 0.0);
    k_rope = ggml.permute(ctx, k_rope, 0, 2, 1, 3);
    k_rope = ggml.cont(ctx, k_rope);

    return .{ .q = q_rope, .k = k_rope };
}

/// 构建位置张量 [start_pos, start_pos+1, ..., start_pos+n_tokens-1]
pub fn buildPositionTensor(ctx: *ggml.Context, n_tokens: i32, start_pos: i32) *ggml.Tensor {
    ctx.setNoAlloc(false);
    const pos_tensor = ctx.newTensor1d(.i32, n_tokens) catch unreachable;
    ctx.setNoAlloc(true);
    const data = pos_tensor.dataBytes();
    const pos_slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
    for (0..@as(usize, @intCast(n_tokens))) |i| {
        pos_slice[i] = @as(i32, @intCast(i)) + start_pos;
    }
    return pos_tensor;
}

const testing = std.testing;

test "RopeParams defaults" {
    const p = RopeParams{};
    try testing.expectEqual(@as(i64, 64), p.rope_dim);
    try testing.expectEqual(@as(f32, 10000000.0), p.rope_theta);
}
