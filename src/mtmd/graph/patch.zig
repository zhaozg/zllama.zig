//! Patch embedding 构建器
//!
//! 提供 Conv2D patch embedding 和原始输入处理的 ggml 计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_inp(), build_inp_raw()

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.graph_patch);

/// 构建 Conv2D patch embedding
///
/// 输入: 归一化后的图像张量 [channels, height, width]
/// 输出: patch 张量 [n_embd, n_patches]
///
/// 处理流程:
///   1. Scale + bias (可选)
///   2. Conv2D patch embedding
///   3. Reshape + transpose → [n_embd, n_patches]
///
/// 参数:
///   - ctx: ggml 上下文
///   - inp_raw: 原始输入张量 [channels, height, width]
///   - patch_embd_w: Conv2D 权重 [kw, kh, channels, n_embd]
///   - patch_bias: Conv2D 偏置 [n_embd]（可选）
///   - img_width: 图像宽度
///   - img_height: 图像高度
///   - scale_val: 缩放值（可选，如 2.0）
///   - bias_val: 偏置值（可选，如 -1.0）
///
/// 返回: patch 张量 [n_embd, n_patches]
///
/// 参考: clip-graph.h build_inp()
pub fn buildInp(
    ctx: *ggml.Context,
    inp_raw: *ggml.Tensor,
    patch_embd_w: *ggml.Tensor,
    patch_bias: ?*ggml.Tensor,
    img_width: u32,
    img_height: u32,
    scale_val: ?f32,
    bias_val: ?f32,
) !*ggml.Tensor {
    var cur = inp_raw;

    // 1. Scale (optional) - applied directly on the graph
    if (scale_val) |s| {
        cur = cur.scale(ctx, s);
        cur.setName("inp_scaled");
    }
    if (bias_val) |b| {
        // Bias: use scaleBias with scale=1.0 to add bias
        // This works in no_alloc context since it's a graph operation
        cur = cur.scaleBias(ctx, 1.0, b);
        cur.setName("inp_biased");
    }

    // 2. Conv2D patch embedding
    const kw: i32 = @intCast(patch_embd_w.ne()[0]);
    const kh: i32 = @intCast(patch_embd_w.ne()[1]);
    const n_embd: i64 = patch_embd_w.ne()[3];
    const n_patches_x: i64 = @divTrunc(@as(i64, @intCast(img_width)), kw);
    const n_patches_y: i64 = @divTrunc(@as(i64, @intCast(img_height)), kh);
    const n_patches: i64 = n_patches_x * n_patches_y;

    cur = cur.conv2d(ctx, patch_embd_w, kw, kh, 0, 0, 1, 1);
    cur.setName("inp_conv");

    if (patch_bias) |pb| {
        cur = cur.add(ctx, pb);
        cur.setName("inp_conv_biased");
    }

    // 3. Reshape to [n_embd, n_patches]
    // Conv2D output: [n_embd, out_w, out_h] → reshape → transpose
    cur = cur.reshape3d(ctx, n_patches, n_embd, 1);
    cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
    cur.setName("inp_patches");

    return cur;
}

/// 构建原始输入处理
///
/// 输入: 原始图像数据 [channels, height, width]
/// 输出: 处理后的张量
///
/// 参数:
///   - ctx: ggml 上下文
///   - channels: 输入通道数（默认 3）
///
/// 参考: clip-graph.h build_inp_raw()
pub fn buildInpRaw(
    ctx: *ggml.Context,
    channels: u32,
) !*ggml.Tensor {
    _ = ctx;
    _ = channels;
    // 原始输入处理 - 由调用者负责创建输入张量
    // 此函数作为占位符，实际输入由外部提供
    log.warn("buildInpRaw: caller should provide input tensor directly", .{});
    return error.NotImplemented;
}

test "buildInp: basic patch embedding" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const channels: i64 = 3;
    const img_width: u32 = 224;
    const img_height: u32 = 224;
    const patch_size: i32 = 16;
    const n_embd: i64 = 768;

    const inp_raw = try ctx.newTensor3d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), channels);
    const patch_w = try ctx.newTensor4d(ggml.Type.f32, patch_size, patch_size, channels, n_embd);

    {
        const buf_inp = try std.testing.allocator.alloc(f32, @as(usize, @intCast(inp_raw.nElems())));
        defer std.testing.allocator.free(buf_inp);
        @memset(buf_inp, 0.5);
        try inp_raw.dataSet(f32, buf_inp);
    }
    {
        const buf_w = try std.testing.allocator.alloc(f32, @as(usize, @intCast(patch_w.nElems())));
        defer std.testing.allocator.free(buf_w);
        @memset(buf_w, 0.1);
        try patch_w.dataSet(f32, buf_w);
    }

    const result = try buildInp(&ctx, inp_raw, patch_w, null, img_width, img_height, 2.0, -1.0);
    try testing.expectEqual(n_embd, result.ne()[0]);

    const expected_patches = @as(i64, @intCast(img_width / @as(u32, @intCast(patch_size)))) *
        @as(i64, @intCast(img_height / @as(u32, @intCast(patch_size))));
    try testing.expectEqual(expected_patches, result.ne()[1]);
}
