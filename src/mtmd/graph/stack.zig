//! Frame stacking 构建器
//!
//! 提供音频帧堆叠（frame stacking）的 ggml 计算图构建。
//! 用于 Ultravox 等音频编码器的帧堆叠操作。
//!
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_stack()

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.graph_stack);

/// 构建 frame stacking
///
/// 将相邻帧堆叠到嵌入维度中，减少序列长度。
/// 例如: [n_embed, n_frames] → [n_embed * stack_factor, n_frames / stack_factor]
///
/// 参数:
///   - ctx: ggml 上下文
///   - cur: 输入张量 [n_embed, n_frames]
///   - stack_factor: 堆叠因子
///   - n_embed: 嵌入维度
///
/// 返回: 堆叠后的张量 [n_embed * stack_factor, n_frames / stack_factor]
///
/// 参考: clip-graph.h build_stack()
pub fn buildStack(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    stack_factor: u32,
    n_embed: u32,
) !*ggml.Tensor {
    if (stack_factor <= 1) return cur;

    const n_frames = cur.ne()[1];
    const sf: i64 = @intCast(stack_factor);
    const ne: i64 = @intCast(n_embed);

    // 输入: [n_embed, n_frames]
    // 1. Reshape to [n_embed, n_frames / sf, sf]
    const out_frames = n_frames / sf;
    var result = cur.reshape3d(ctx, ne, out_frames, sf);
    result.setName("stack_reshaped");

    // 2. Permute to [sf, n_embed, out_frames]
    result = result.permute(ctx, 2, 0, 1, 3).cont(ctx);
    result.setName("stack_permuted");

    // 3. Reshape to [ne * sf, out_frames]
    result = result.reshape2d(ctx, ne * sf, out_frames);
    result.setName("stack_result");

    return result;
}

test "buildStack: basic stacking" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_embed: u32 = 64;
    const n_frames: i64 = 32;
    const stack_factor: u32 = 2;

    const cur = try ctx.newTensor2d(ggml.Type.f32, @as(i64, @intCast(n_embed)), n_frames);
    @memset(cur.dataF32(), 1.0);

    const result = try buildStack(&ctx, cur, stack_factor, n_embed);
    try testing.expectEqual(@as(i64, @intCast(n_embed * stack_factor)), result.ne()[0]);
    try testing.expectEqual(n_frames / @as(i64, @intCast(stack_factor)), result.ne()[1]);
}
