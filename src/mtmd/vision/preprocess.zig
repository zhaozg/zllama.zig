//! 图像预处理
//!
//! 提供图像归一化、尺寸调整等预处理功能。
//! 参考: llama.cpp tools/mtmd/clip.cpp

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.vision_preprocess);

/// 归一化模式
pub const NormalizeMode = enum {
    /// (pixel/255.0 - mean) / std — 标准归一化
    standard,
    /// pixel/255.0 * 2 - 1 — SigLIP 风格
    siglip,
    /// 直接传递（无归一化）
    passthrough,
};

/// 将 RGB u8 图像数据归一化并填充到 ggml 张量
///
/// @param ctx ggml 上下文
/// @param image_data RGB 图像数据 [height][width][3]，值范围 [0, 255]
/// @param img_width 图像宽度
/// @param img_height 图像高度
/// @param mean 归一化均值（仅 standard 模式使用）
/// @param std 归一化标准差（仅 standard 模式使用）
/// @param mode 归一化模式
/// @returns 归一化后的张量 [width, height, 3] f32
pub fn normalizeToTensor(
    ctx: *ggml.Context,
    image_data: []const u8,
    img_width: u32,
    img_height: u32,
    mean: [3]f32,
    std_val: [3]f32,
    mode: NormalizeMode,
) !*ggml.Tensor {
    const W: usize = @intCast(img_width);
    const H: usize = @intCast(img_height);
    const wh: usize = W * H;

    var inp = try ctx.newTensor3d(ggml.Type.f32, @intCast(img_width), @intCast(img_height), 3);
    inp.setName("vision_input");

    // In no_alloc mode, the tensor data pointer is NULL.
    // We need to allocate the data manually so we can write to it.
    const no_alloc = ctx.getNoAlloc();
    if (no_alloc) {
        // Allocate data buffer for the input tensor using C malloc
        const data_size = @as(usize, @intCast(inp.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
        @memset(buf, 0);
        inp.setDataPtr(buf);
    }

    const n_elems = @as(usize, @intCast(inp.nElems()));
    const dst = try std.heap.page_allocator.alloc(f32, n_elems);
    defer std.heap.page_allocator.free(dst);
    log.err("normalizeToTensor: dst.ptr={*} len={d}", .{ dst.ptr, dst.len });

    switch (mode) {
        .standard => {
            const mean_r = mean[0];
            const mean_g = mean[1];
            const mean_b = mean[2];
            const std_r = std_val[0];
            const std_g = std_val[1];
            const std_b = std_val[2];

            for (0..H) |y| {
                for (0..W) |x| {
                    const src_idx = (y * W + x) * 3;
                    const dst_base = y * W + x;
                    dst[dst_base] = (@as(f32, @floatFromInt(image_data[src_idx + 0])) / 255.0 - mean_r) / std_r;
                    dst[dst_base + wh] = (@as(f32, @floatFromInt(image_data[src_idx + 1])) / 255.0 - mean_g) / std_g;
                    dst[dst_base + 2 * wh] = (@as(f32, @floatFromInt(image_data[src_idx + 2])) / 255.0 - mean_b) / std_b;
                }
            }
        },
        .siglip => {
            // SigLIP: pixel/255.0 * 2 - 1
            for (0..H) |y| {
                for (0..W) |x| {
                    const src_idx = (y * W + x) * 3;
                    const dst_base = y * W + x;
                    dst[dst_base] = @as(f32, @floatFromInt(image_data[src_idx + 0])) / 255.0 * 2.0 - 1.0;
                    dst[dst_base + wh] = @as(f32, @floatFromInt(image_data[src_idx + 1])) / 255.0 * 2.0 - 1.0;
                    dst[dst_base + 2 * wh] = @as(f32, @floatFromInt(image_data[src_idx + 2])) / 255.0 * 2.0 - 1.0;
                }
            }
        },
        .passthrough => {
            // 直接传递 u8 值作为 f32
            for (0..H) |y| {
                for (0..W) |x| {
                    const src_idx = (y * W + x) * 3;
                    const dst_base = y * W + x;
                    dst[dst_base] = @floatFromInt(image_data[src_idx + 0]);
                    dst[dst_base + wh] = @floatFromInt(image_data[src_idx + 1]);
                    dst[dst_base + 2 * wh] = @floatFromInt(image_data[src_idx + 2]);
                }
            }
        },
    }
    try inp.dataSet(f32, dst);

    return inp;
}
