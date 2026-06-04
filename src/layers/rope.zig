//! RoPE 位置编码层
//!
//! 实现旋转位置编码（Rotary Position Embedding）。
//! 支持 Qwen 3.5 的扩展 RoPE 参数。
//!
//! 输入布局（与 llama.cpp 一致）:
//!   q: [head_dim, n_head, n_tokens]
//!   k: [head_dim, n_kv_head, n_tokens]
//!
//! ggml_rope_ext 直接在 [head_dim, n_head, n_tokens] 布局上操作，
//! 与 llama.cpp 的调用方式一致。

const std = @import("std");
const ggml = @import("..//ggml.zig");

/// RoPE 配置参数
pub const RopeParams = struct {
    rope_dim: i64 = 64,
    rope_theta: f32 = 10000000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_scaling_type: []const u8 = "",
    original_max_seq_len: u32 = 32768,
};

/// 对 Q 和 K 张量应用 RoPE 位置编码
/// q: [head_dim, n_head, n_tokens] 查询张量
/// k: [head_dim, n_kv_head, n_tokens] 键张量
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

    // 直接在 [head_dim, n_head, n_tokens] 布局上调用 ggml_rope_ext
    // 与 llama.cpp 的调用方式一致
    const q_rope = ggml.ropeExt(ctx, q, pos_tensor, 0, @intCast(rope_dim), 0, rope_theta, freq_scale, 0.0, 1.0, 0.0, 0.0);
    const k_rope = ggml.ropeExt(ctx, k, pos_tensor, 0, @intCast(rope_dim), 0, rope_theta, freq_scale, 0.0, 1.0, 0.0, 0.0);

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
/// 构建多位置张量用于 rope_multi (MRoPE/IMRoPE)
/// 返回 [n_tokens * 4] 形状，每 token 4 个位置值 [pos, pos, pos, pos]
pub fn buildMultiPositionTensor(ctx: *ggml.Context, n_tokens: i32, start_pos: i32) *ggml.Tensor {
    const n_total: i32 = n_tokens * 4;
    ctx.setNoAlloc(false);
    const pos_tensor = ctx.newTensor1d(.i32, n_total) catch unreachable;
    ctx.setNoAlloc(true);
    const data = pos_tensor.dataBytes();
    const pos_slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_total))];
    for (0..@as(usize, @intCast(n_tokens))) |i| {
        const pos: i32 = @as(i32, @intCast(i)) + start_pos;
        const base: usize = i * 4;
        pos_slice[base] = pos;
        pos_slice[base + 1] = pos;
        pos_slice[base + 2] = pos;
        pos_slice[base + 3] = pos;
    }
    return pos_tensor;
}

const testing = std.testing;

test "RopeParams defaults" {
    const p = RopeParams{};
    try testing.expectEqual(@as(i64, 64), p.rope_dim);
    try testing.expectEqual(@as(f32, 10000000.0), p.rope_theta);
}
