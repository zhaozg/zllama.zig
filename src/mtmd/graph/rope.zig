//! 2D RoPE 构建器
//!
//! 提供 2D 旋转位置编码（Rotary Position Embedding）的 ggml 计算图构建。
//! 使用 ggml_view + ggml_rope_ext 组合，避免 inplace 操作。
//!
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_rope_2d()

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.graph_rope);

/// 构建 2D RoPE
///
/// 将输入张量沿最后一个维度分成两半，分别应用不同的位置编码。
/// 第一半使用 pos_a，第二半使用 pos_b。
///
/// 参数:
///   - ctx: ggml 上下文
///   - cur: 输入张量 [d_head, n_head, n_patches, n_batch]
///   - pos_a: 第一半位置 [n_patches] (i32)
///   - pos_b: 第二半位置 [n_patches] (i32)
///   - freq_base: 频率基数
///   - interleave_freq: 是否交错频率
///
/// 返回: 应用 RoPE 后的张量 [d_head, n_head, n_patches, n_batch]
///
/// 参考: clip-graph.h build_rope_2d()
pub fn buildRope2D(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    pos_a: *ggml.Tensor,
    pos_b: *ggml.Tensor,
    freq_base: f32,
    interleave_freq: bool,

) !*ggml.Tensor {
    _ = interleave_freq;


    const d_head = cur.ne()[0];
    const n_head = cur.ne()[1];
    const n_patches = cur.ne()[2];
    const n_batch = cur.ne()[3];
    const d_head_half = d_head / 2;

    // 第一半: 使用 pos_a
    const first_half = cur.view4d(
        ctx,
        d_head_half,
        n_head,
        n_patches,
        n_batch,
        cur.nb()[1],
        cur.nb()[2],
        cur.nb()[3],
        0,
    );
    first_half.setName("rope_first_half");

    const rope_first = first_half.ropeExt(
        ctx,
        pos_a,
        null,
        @intCast(d_head_half),
        2,
        0,
        freq_base,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );
    rope_first.setName("rope_first");

    // 第二半: 使用 pos_b
    const offset: usize = @intCast(d_head_half * @sizeOf(f32));
    const second_half = cur.view4d(
        ctx,
        d_head_half,
        n_head,
        n_patches,
        n_batch,
        cur.nb()[1],
        cur.nb()[2],
        cur.nb()[3],
        offset,
    );
    second_half.setName("rope_second_half");

    const rope_second = second_half.ropeExt(
        ctx,
        pos_b,
        null,
        @intCast(d_head_half),
        2,
        0,
        freq_base,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );
    rope_second.setName("rope_second");

    // 合并两半
    const result = rope_first.concat(ctx, rope_second, 0);
    result.setName("rope_combined");

    return result;
}

/// 创建位置索引张量
///
/// 为 n_patches 个位置创建 X 和 Y 方向的位置索引。
///
/// 参数:
///   - ctx: ggml 上下文
///   - n_patches: 位置总数
///   - n_patches_x: X 方向位置数
///
/// 返回: (pos_x, pos_y) 两个 i32 张量，每个形状为 [n_patches]
pub fn createPositionIndices(
    ctx: *ggml.Context,
    n_patches: i64,
    n_patches_x: i64,
) !struct { pos_x: *ggml.Tensor, pos_y: *ggml.Tensor } {
    const pos_x = try ctx.newTensor1d(ggml.Type.i32, n_patches);
    pos_x.setName("pos_x");
    const pos_y = try ctx.newTensor1d(ggml.Type.i32, n_patches);
    pos_y.setName("pos_y");

    const px = pos_x.dataI32();
    const py = pos_y.dataI32();
    for (0..@as(usize, @intCast(n_patches))) |i| {
        px[i] = @mod(@as(i32, @intCast(i)), @as(i32, @intCast(n_patches_x)));
        py[i] = @divTrunc(@as(i32, @intCast(i)), @as(i32, @intCast(n_patches_x)));
    }

    return .{ .pos_x = pos_x, .pos_y = pos_y };
}

test "buildRope2D: basic 2D RoPE" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const d_head: i64 = 64;
    const n_head: i64 = 4;
    const n_patches: i64 = 16;
    const n_batch: i64 = 1;

    const cur = try ctx.newTensor4d(ggml.Type.f32, d_head, n_head, n_patches, n_batch);
    @memset(cur.dataF32(), 0.5);

    const indices = try createPositionIndices(&ctx, n_patches, 4);
    const freq_base: f32 = 10000.0;

    const result = try buildRope2D(&ctx, cur, indices.pos_x, indices.pos_y, freq_base, false);
    try testing.expectEqual(d_head, result.ne()[0]);
    try testing.expectEqual(n_head, result.ne()[1]);
    try testing.expectEqual(n_patches, result.ne()[2]);
    try testing.expectEqual(n_batch, result.ne()[3]);
}
