//! 多模态预处理模块
//!
//! 提供图像数据的预处理功能：resize + 像素归一化 + 动态分辨率计算。
//! 所有图像解码操作通过 stb_image 绑定层（src/stb_image.zig）完成。

const std = @import("std");
const ggml = @import("ggml");
const stb_image = @import("stb_image");

const log = std.log.scoped(.mtmd_preprocess);

// ============================================================================
// 动态分辨率计算
// ============================================================================

pub const Size2D = struct { width: u32, height: u32 };

fn fnRoundByFactor(x: f64, factor: f64) f64 {
    return std.math.round(x / factor) * factor;
}
fn fnCeilByFactor(x: f64, factor: f64) f64 {
    return std.math.ceil(x / factor) * factor;
}
fn fnFloorByFactor(x: f64, factor: f64) f64 {
    return std.math.floor(x / factor) * factor;
}

/// 计算保持宽高比的缩放尺寸，使宽高均为 align_size 的倍数。
pub fn calcSizePreservedRatio(src_width: u32, src_height: u32, align_size: u32, min_pixels: u32, max_pixels: u32) Size2D {
    const width: f64 = @floatFromInt(src_width);
    const height: f64 = @floatFromInt(src_height);
    const align_f: f64 = @floatFromInt(align_size);
    var h_bar = @max(align_f, fnRoundByFactor(height, align_f));
    var w_bar = @max(align_f, fnRoundByFactor(width, align_f));
    const min_px: f64 = @floatFromInt(min_pixels);
    const max_px: f64 = @floatFromInt(max_pixels);
    if (h_bar * w_bar > max_px) {
        const beta = std.math.sqrt((height * width) / max_px);
        h_bar = @max(align_f, fnFloorByFactor(height / beta, align_f));
        w_bar = @max(align_f, fnFloorByFactor(width / beta, align_f));
    } else if (h_bar * w_bar < min_px) {
        const beta = std.math.sqrt(min_px / (height * width));
        h_bar = fnCeilByFactor(height * beta, align_f);
        w_bar = fnCeilByFactor(width * beta, align_f);
    }
    return .{ .width = @intFromFloat(w_bar), .height = @intFromFloat(h_bar) };
}
// ============================================================================
// 双线性插值
// ============================================================================

/// Resize an RGB image via bilinear interpolation.
pub fn resizeRGB(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    return bilinearResizeRGB(allocator, src, src_w, src_h, dst_w, dst_h);
}

fn bilinearResizeRGB(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    const dst = try allocator.alloc(u8, dst_w * dst_h * 3);
    const scale_x: f64 = @as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(dst_w));
    const scale_y: f64 = @as(f64, @floatFromInt(src_h)) / @as(f64, @floatFromInt(dst_h));
    for (0..dst_h) |dy| {
        const sy: f64 = (@as(f64, @floatFromInt(dy)) + 0.5) * scale_y - 0.5;
        const sy0_i: i32 = @intFromFloat(@floor(sy));
        const sy1_i: i32 = @min(sy0_i + 1, @as(i32, @intCast(src_h)) - 1);
        const sy0: u32 = @intCast(@max(sy0_i, 0));
        const sy1: u32 = @intCast(sy1_i);
        const fy: f64 = sy - @floor(sy);
        for (0..dst_w) |dx| {
            const sx: f64 = (@as(f64, @floatFromInt(dx)) + 0.5) * scale_x - 0.5;
            const sx0_i: i32 = @intFromFloat(@floor(sx));
            const sx1_i: i32 = @min(sx0_i + 1, @as(i32, @intCast(src_w)) - 1);
            const sx0: u32 = @intCast(@max(sx0_i, 0));
            const sx1: u32 = @intCast(sx1_i);
            const fx: f64 = sx - @floor(sx);
            for (0..3) |c| {
                const v00: f64 = @floatFromInt(src[(sy0 * src_w + sx0) * 3 + c]);
                const v01: f64 = @floatFromInt(src[(sy0 * src_w + sx1) * 3 + c]);
                const v10: f64 = @floatFromInt(src[(sy1 * src_w + sx0) * 3 + c]);
                const v11: f64 = @floatFromInt(src[(sy1 * src_w + sx1) * 3 + c]);
                const iy0: f64 = v00 * (1.0 - fx) + v01 * fx;
                const iy1: f64 = v10 * (1.0 - fx) + v11 * fx;
                const val: f64 = iy0 * (1.0 - fy) + iy1 * fy;
                dst[(dy * dst_w + dx) * 3 + c] = @intFromFloat(@round(val));
            }
        }
    }
    return dst;
}

// ============================================================================
// 综合预处理：resize（供 tokenize 阶段使用）
// ============================================================================

/// Resize an RGB u8 image (bilinear). Normalization to f32 happens inside the encoder.
pub fn resizeAndNormalize(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    mean: [3]f32,
    std_val: [3]f32,
) !struct { data: []u8, width: u32, height: u32 } {
    _ = mean;
    _ = std_val;
    const resized = try bilinearResizeRGB(allocator, src, src_w, src_h, dst_w, dst_h);
    return .{ .data = resized, .width = dst_w, .height = dst_h };
}

// ============================================================================
// 图像类型
// ============================================================================

pub const ProcessedImage = struct {
    data: []u8,
    width: u32,
    height: u32,
    allocator: std.mem.Allocator,
    pub fn deinit(self: *ProcessedImage) void {
        self.allocator.free(self.data);
    }
};

// ============================================================================
// 文件加载（通过 stb_image 绑定层）
// ============================================================================

/// 从文件加载图像并解码为原始 RGB 数据（不缩放），返回原始尺寸。
/// 所有图像解码操作均通过 src/stb_image.zig 封装层完成。
pub fn loadImageRaw(allocator: std.mem.Allocator, io: std.Io, filepath: []const u8) !ProcessedImage {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, filepath, .{ .mode = .read_only });
    defer file.close(io);
    const stat = try file.stat(io);
    const raw = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(raw);
    _ = try file.readPositionalAll(io, raw, 0);
    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const pixels = stb_image.loadFromMemory(raw.ptr, @intCast(raw.len), &w, &h, &comp, 3);
    if (pixels == null) return error.ImageDecodeFailed;
    defer stb_image.free(pixels);
    const sw: u32 = @intCast(w);
    const sh: u32 = @intCast(h);
    const pixel_bytes = pixels.?[0..@as(usize, @intCast(sw * sh * 3))];
    const owned = try allocator.alloc(u8, pixel_bytes.len);
    @memcpy(owned, pixel_bytes);
    return .{ .data = owned, .width = sw, .height = sh, .allocator = allocator };
}
/// 从文件加载图像，解码并缩放到 target_size × target_size。
/// 底层复用 loadImageRaw 完成解码，再通过 bilinearResizeRGB 缩放。
pub fn loadImage(allocator: std.mem.Allocator, io: std.Io, filepath: []const u8, target_size: u32, format_hint: ?[]const u8) !ProcessedImage {
    _ = format_hint;
    var raw_img = try loadImageRaw(allocator, io, filepath);
    defer raw_img.deinit();
    const resized = try bilinearResizeRGB(allocator, raw_img.data, raw_img.width, raw_img.height, target_size, target_size);
    return .{ .data = resized, .width = target_size, .height = target_size, .allocator = allocator };
}

/// Legacy: create ProcessedImage from raw RGB with resize.
pub fn fromRawRGB(allocator: std.mem.Allocator, rgb_data: []const u8, width: u32, height: u32, target_size: u32) !ProcessedImage {
    const resized = try bilinearResizeRGB(allocator, rgb_data[0..@as(usize, @intCast(width * height * 3))], width, height, target_size, target_size);
    return .{ .data = resized, .width = target_size, .height = target_size, .allocator = allocator };
}

/// Legacy: convert ProcessedImage to ggml f32 tensor with optional normalization mode.
/// Legacy image normalization mode (used by test_vision.zig)
pub const ImageNormalize = enum { div255, imagenet, siglip, none };

/// Legacy: convert ProcessedImage to ggml f32 tensor.
pub fn imageToTensor(ctx: *ggml.Context, image: *const ProcessedImage, normalize: ImageNormalize) !*ggml.Tensor {
    const W: usize = @intCast(image.width);
    const H: usize = @intCast(image.height);
    const tensor = try ctx.newTensor3d(ggml.Type.f32, @intCast(image.width), @intCast(image.height), 3);
    const n_elems = @as(usize, @intCast(tensor.nElems()));
    const data = try std.heap.page_allocator.alloc(f32, n_elems);
    defer std.heap.page_allocator.free(data);
    // Store in HWC layout (matching llama.cpp convention)
    // Memory: [R0,G0,B0, R1,G1,B1, ...]
    switch (normalize) {
        .siglip => for (0..W * H) |i| {
            const idx = i * 3;
            const r: f32 = @as(f32, @floatFromInt(image.data[idx])) / 255.0 * 2.0 - 1.0;
            const g: f32 = @as(f32, @floatFromInt(image.data[idx + 1])) / 255.0 * 2.0 - 1.0;
            const b: f32 = @as(f32, @floatFromInt(image.data[idx + 2])) / 255.0 * 2.0 - 1.0;
            data[idx + 0] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
        },
        .none => for (0..W * H) |i| {
            const idx = i * 3;
            data[idx + 0] = @floatFromInt(image.data[idx]);
            data[idx + 1] = @floatFromInt(image.data[idx + 1]);
            data[idx + 2] = @floatFromInt(image.data[idx + 2]);
        },
        .div255, .imagenet => for (0..W * H) |i| {
            const idx = i * 3;
            const r: f32 = @as(f32, @floatFromInt(image.data[idx])) / 255.0;
            const g: f32 = @as(f32, @floatFromInt(image.data[idx + 1])) / 255.0;
            const b: f32 = @as(f32, @floatFromInt(image.data[idx + 2])) / 255.0;
            data[idx + 0] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
        },
    }
    try tensor.dataSet(f32, data);
    return tensor;
}
