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
// 综合预处理
// ============================================================================

/// Simple resize to u8 RGB (no normalization). Returns resized u8 data.
/// Used by tokenize stage for determining patch counts before full encode.
pub fn resizeToU8(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) !struct { data: []u8, width: u32, height: u32 } {
    const resized = try bilinearResizeRGB(allocator, src, src_w, src_h, dst_w, dst_h);
    return .{ .data = resized, .width = dst_w, .height = dst_h };
}

/// Full resize + normalize pipeline (llama.cpp mtmd-image.cpp style):
/// compute optimal size → bilinear resize (llama.cpp formula) → normalize → create ggml tensor.
/// Returns the normalized tensor and the new dimensions.
pub fn resizeAndNormalize(
    ctx: *ggml.Context,
    allocator: std.mem.Allocator,
    image_data: []const u8,
    img_width: u32,
    img_height: u32,
    mean: [3]f32,
    std_val: [3]f32,
    align_size: u32,
    min_pixels: u32,
    max_pixels: u32,
) !struct { tensor: *ggml.Tensor, new_width: u32, new_height: u32 } {
    log.warn("resizeAndNormalize: image_data[0..5] = {d}, {d}, {d}, {d}, {d}", .{ image_data[0], image_data[1], image_data[2], image_data[3], image_data[4] });

    const size = calcSizePreservedRatio(img_width, img_height, align_size, min_pixels, max_pixels);
    log.warn("resize: {d}x{d} -> {d}x{d}", .{ img_width, img_height, size.width, size.height });

    const resized_u8 = try resizeBilinearU8(allocator, image_data, img_width, img_height, size.width, size.height);
    defer allocator.free(resized_u8);
    log.warn("resizeAndNormalize: resized_u8[0..5] = {d}, {d}, {d}, {d}, {d}", .{ resized_u8[0], resized_u8[1], resized_u8[2], resized_u8[3], resized_u8[4] });
    log.warn("resizeAndNormalize: mean=[{d:.6},{d:.6},{d:.6}] std=[{d:.6},{d:.6},{d:.6}]", .{ mean[0], mean[1], mean[2], std_val[0], std_val[1], std_val[2] });

    const wh: usize = @as(usize, @intCast(size.width)) * @as(usize, @intCast(size.height));
    const normalized = try allocator.alloc(f32, wh * 3);
    defer allocator.free(normalized);

    const mr: f64 = @floatCast(mean[0]);
    const mg: f64 = @floatCast(mean[1]);
    const mb: f64 = @floatCast(mean[2]);
    const sr: f64 = @floatCast(std_val[0]);
    const sg: f64 = @floatCast(std_val[1]);
    const sb: f64 = @floatCast(std_val[2]);

    // Normalize and store in HWC layout (matching llama.cpp convention)
    for (0..@as(usize, @intCast(size.height))) |y| {
        for (0..@as(usize, @intCast(size.width))) |x| {
            const hwc_idx = (y * @as(usize, @intCast(size.width)) + x) * 3;
            const r_u8: f64 = @floatFromInt(resized_u8[hwc_idx + 0]);
            const g_u8: f64 = @floatFromInt(resized_u8[hwc_idx + 1]);
            const b_u8: f64 = @floatFromInt(resized_u8[hwc_idx + 2]);
            normalized[hwc_idx + 0] = @as(f32, @floatCast((r_u8 / 255.0 - mr) / sr));
            normalized[hwc_idx + 1] = @as(f32, @floatCast((g_u8 / 255.0 - mg) / sg));
            normalized[hwc_idx + 2] = @as(f32, @floatCast((b_u8 / 255.0 - mb) / sb));
        }
    }
    log.warn("resizeAndNormalize: normalized first 9 elements (HWC): [{d:.6},{d:.6},{d:.6}, {d:.6},{d:.6},{d:.6}, {d:.6},{d:.6},{d:.6}]", .{
        normalized[0], normalized[1], normalized[2],
        normalized[3], normalized[4], normalized[5],
        normalized[6], normalized[7], normalized[8],
    });

    var inp = try ctx.newTensor3d(ggml.Type.f32, @intCast(size.width), @intCast(size.height), 3);

    // 如果 context 是 no_alloc 模式，需要先为 tensor 分配数据缓冲区。
    // 使用 allocator 分配（而非裸 std.c.malloc），确保内存可追踪。
    // 注意：gallocr.allocGraph 会重新分配 tensor.data 指针到 gallocr 管理的缓冲区，
    // 因此这里分配的缓冲区在 allocGraph 后不再被引用。
    // 遵循项目 "leak-to-exit" 策略：缓冲区由 page_allocator 分配，进程退出时自动回收。
    const no_alloc = ctx.getNoAlloc();
    if (no_alloc) {
        const ds = @as(usize, @intCast(inp.nBytes()));
        const buf = try allocator.alloc(u8, ds);
        @memset(buf, 0);
        inp.setDataPtr(buf);
        try inp.dataSet(f32, normalized);
    } else {
        try inp.dataSet(f32, normalized);
    }
    return .{ .tensor = inp, .new_width = size.width, .new_height = size.height };
}

/// Bilinear resize that outputs u8 (llama.cpp formula: dy * (sw-1)/(dw-1)).
/// This is the algorithm used by llama.cpp's mtmd-image.cpp resize_bilinear().
pub fn resizeBilinearU8(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) ![]u8 {
    const sw: i32 = @intCast(src_w);
    const sh: i32 = @intCast(src_h);
    var dw: i32 = @intCast(dst_w);
    var dh: i32 = @intCast(dst_h);
    if (dw <= 0) dw = 1;
    if (dh <= 0) dh = 1;
    const out: []u8 = try allocator.alloc(u8, @as(usize, @intCast(dw * dh * 3)));

    const x_ratio: f32 = if (dw > 1) @as(f32, @floatFromInt(sw - 1)) / @as(f32, @floatFromInt(dw - 1)) else 0.0;
    const y_ratio: f32 = if (dh > 1) @as(f32, @floatFromInt(sh - 1)) / @as(f32, @floatFromInt(dh - 1)) else 0.0;

    for (0..@as(usize, @intCast(dh))) |dy| {
        const py: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(dy)))) * y_ratio;
        const y0: i32 = @min(@as(i32, @intFromFloat(@floor(py))), sh - 1);
        const y1: i32 = @min(y0 + 1, sh - 1);
        const yf: f32 = py - @as(f32, @floatFromInt(y0));
        for (0..@as(usize, @intCast(dw))) |dx| {
            const px: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(dx)))) * x_ratio;
            const x0: i32 = @min(@as(i32, @intFromFloat(@floor(px))), sw - 1);
            const x1: i32 = @min(x0 + 1, sw - 1);
            const xf: f32 = px - @as(f32, @floatFromInt(x0));

            const idx00: usize = @as(usize, @intCast((y0 * sw + x0) * 3));
            const idx01: usize = @as(usize, @intCast((y0 * sw + x1) * 3));
            const idx10: usize = @as(usize, @intCast((y1 * sw + x0) * 3));
            const idx11: usize = @as(usize, @intCast((y1 * sw + x1) * 3));
            const db: usize = (dy * @as(usize, @intCast(dw)) + dx) * 3;

            inline for (0..3) |c| {
                const v00: f32 = @floatFromInt(src[idx00 + c]);
                const v01: f32 = @floatFromInt(src[idx01 + c]);
                const v10: f32 = @floatFromInt(src[idx10 + c]);
                const v11: f32 = @floatFromInt(src[idx11 + c]);
                // Use lerp formula: s + (e - s) * t  (matching C++ lerp)
                const top: f32 = v00 + (v01 - v00) * xf;
                const bot: f32 = v10 + (v11 - v10) * xf;
                // C++ static_cast<uint8_t> truncates (not rounds).
                // Zig @intFromFloat also truncates for float→int, matching C++ behavior.
                out[db + c] = @intFromFloat(top + (bot - top) * yf);
            }
        }
    }
    return out;
}

/// Legacy: resize to f32 in CHW layout (used by compare tools).
pub fn resizeBilinear(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) ![]f32 {
    const sw: usize = @intCast(src_w);
    const sh: usize = @intCast(src_h);
    const dw: usize = @intCast(dst_w);
    const dh: usize = @intCast(dst_h);
    const out: []f32 = try allocator.alloc(f32, dw * dh * 3);

    const rx: f64 = if (dw > 1) @as(f64, @floatFromInt(sw - 1)) / @as(f64, @floatFromInt(dw - 1)) else 0.0;
    const ry: f64 = if (dh > 1) @as(f64, @floatFromInt(sh - 1)) / @as(f64, @floatFromInt(dh - 1)) else 0.0;

    for (0..dh) |dy| {
        const syf: f64 = @as(f64, @floatFromInt(dy)) * ry;
        const sy0: usize = @intFromFloat(@floor(syf));
        const sy1: usize = @min(sy0 + 1, sh - 1);
        const fy: f64 = syf - @as(f64, @floatFromInt(sy0));
        for (0..dw) |dx| {
            const sxf: f64 = @as(f64, @floatFromInt(dx)) * rx;
            const sx0: usize = @intFromFloat(@floor(sxf));
            const sx1: usize = @min(sx0 + 1, sw - 1);
            const fx: f64 = sxf - @as(f64, @floatFromInt(sx0));

            const idx00: usize = (sy0 * sw + sx0) * 3;
            const idx01: usize = (sy0 * sw + sx1) * 3;
            const idx10: usize = (sy1 * sw + sx0) * 3;
            const idx11: usize = (sy1 * sw + sx1) * 3;
            const db: usize = dy * dw + dx;
            const wh2: usize = dw * dh;

            inline for (0..3) |c| {
                const v00: f64 = @floatFromInt(src[idx00 + c]);
                const v01: f64 = @floatFromInt(src[idx01 + c]);
                const v10: f64 = @floatFromInt(src[idx10 + c]);
                const v11: f64 = @floatFromInt(src[idx11 + c]);
                const top: f64 = v00 * (1.0 - fx) + v01 * fx;
                const bot: f64 = v10 * (1.0 - fx) + v11 * fx;
                out[c * wh2 + db] = @floatCast(top * (1.0 - fy) + bot * fy);
            }
        }
    }
    return out;
}

/// Normalization mode for image preprocessing.
pub const NormalizeMode = enum { standard, siglip, passthrough };

/// Convert u8 RGB image to ggml f32 tensor with normalization (HWC→CHW conversion).
/// Ref: llama.cpp clip.cpp set_input_f32() for images
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

    // 如果 context 是 no_alloc 模式，需要先为 tensor 分配数据缓冲区。
    // 使用 page_allocator 分配（而非裸 std.c.malloc），确保内存可追踪。
    const no_alloc = ctx.getNoAlloc();
    if (no_alloc) {
        const ds = @as(usize, @intCast(inp.nBytes()));
        const buf = try std.heap.page_allocator.alloc(u8, ds);
        defer std.heap.page_allocator.free(buf);
        @memset(buf, 0);
        inp.setDataPtr(buf);
    }

    const n_elems = @as(usize, @intCast(inp.nElems()));
    const dst = try std.heap.page_allocator.alloc(f32, n_elems);
    defer std.heap.page_allocator.free(dst);

    // Store in CHW layout (ggml convention for 3D tensor)
    switch (mode) {
        .standard => {
            const mr = mean[0];
            const mg = mean[1];
            const mb = mean[2];
            const sr = std_val[0];
            const sg = std_val[1];
            const sb = std_val[2];
            for (0..H) |y| {
                for (0..W) |x| {
                    const si = (y * W + x) * 3;
                    const db = y * W + x;
                    dst[db] = (@as(f32, @floatFromInt(image_data[si + 0])) / 255.0 - mr) / sr;
                    dst[db + wh] = (@as(f32, @floatFromInt(image_data[si + 1])) / 255.0 - mg) / sg;
                    dst[db + 2 * wh] = (@as(f32, @floatFromInt(image_data[si + 2])) / 255.0 - mb) / sb;
                }
            }
        },
        .siglip => {
            for (0..H) |y| {
                for (0..W) |x| {
                    const si = (y * W + x) * 3;
                    const db = y * W + x;
                    dst[db] = @as(f32, @floatFromInt(image_data[si + 0])) / 255.0 * 2.0 - 1.0;
                    dst[db + wh] = @as(f32, @floatFromInt(image_data[si + 1])) / 255.0 * 2.0 - 1.0;
                    dst[db + 2 * wh] = @as(f32, @floatFromInt(image_data[si + 2])) / 255.0 * 2.0 - 1.0;
                }
            }
        },
        .passthrough => {
            for (0..H) |y| {
                for (0..W) |x| {
                    const si = (y * W + x) * 3;
                    const db = y * W + x;
                    dst[db] = @floatFromInt(image_data[si + 0]);
                    dst[db + wh] = @floatFromInt(image_data[si + 1]);
                    dst[db + 2 * wh] = @floatFromInt(image_data[si + 2]);
                }
            }
        },
    }
    try inp.dataSet(f32, dst);
    return inp;
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

// ============================================================================
// 测试
// ============================================================================

test "calcSizePreservedRatio: no resize" {
    const r = calcSizePreservedRatio(224, 224, 48, 18432, 589824);
    try std.testing.expectEqual(@as(u32, 224), r.width);
}

test "calcSizePreservedRatio: downscale" {
    const r = calcSizePreservedRatio(1024, 1024, 48, 18432, 589824);
    try std.testing.expect(r.width < 1024);
    try std.testing.expect(r.width % 48 == 0);
}

test "resizeAndNormalize: basic" {
    const allocator = std.testing.allocator;
    const src_w: u32 = 4;
    const src_h: u32 = 4;
    var image_data: [48]u8 = [_]u8{128} ** 48;

    const ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();

    const mean: [3]f32 = .{ 0.5, 0.5, 0.5 };
    const std_val: [3]f32 = .{ 0.5, 0.5, 0.5 };

    const result = try resizeAndNormalize(ctx, allocator, &image_data, src_w, src_h, mean, std_val, 48, 0, 589824);
    try std.testing.expectEqual(@as(u32, 48), result.new_width);
    try std.testing.expectEqual(@as(u32, 48), result.new_height);

    const tensor = result.tensor;
    // resizeAndNormalize 在 no_alloc 模式下会为 tensor 分配数据缓冲区，
    // 该缓冲区由调用者负责释放。测试完成后需要手动释放。
    const tensor_data_buf = tensor.dataBytes();
    defer allocator.free(tensor_data_buf);

    const data = try tensor.dataGet(f32, allocator);
    defer allocator.free(data);

    const expected: f32 = (128.0 / 255.0 - 0.5) / 0.5;
    try std.testing.expectApproxEqAbs(expected, data[0], 1e-5);
    try std.testing.expect(data[0] < 0.5);
}
