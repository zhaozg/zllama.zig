//! 多模态预处理模块
//!
//! 提供图像数据的预处理功能：
//! - 图像：resize（双线性插值）+ 像素归一化
//! - 动态分辨率计算（calcSizePreservedRatio）
//!
//! 音频预处理已移至 src/mtmd/audio/ 子模块。
//! 通过 `@import("audio")` 或 `@import("mm").audio_mod` 访问。
//!
//! 图像格式支持：PPM (P6 binary), JPEG, PNG, BMP, GIF, TGA (via stb_image)
//!
//! 参考: llama.cpp tools/mtmd/clip-impl.h (clip_image_preprocess)
//!       llama.cpp tools/mtmd/mtmd-image.cpp (calc_size_preserved_ratio)

const std = @import("std");
const ggml = @import("ggml");
const stb_image = @import("stb_image");

const log = std.log.scoped(.mm_preprocess);

/// 2D 尺寸类型，用于 calcSizePreservedRatio 等函数的返回值
pub const Size2D = struct { width: u32, height: u32 };

// ============================================================================
// 动态分辨率计算
// ============================================================================

fn fnRoundByFactor(x: f64, factor: f64) f64 {
    return std.math.round(x / factor) * factor;
}

fn fnCeilByFactor(x: f64, factor: f64) f64 {
    return std.math.ceil(x / factor) * factor;
}

fn fnFloorByFactor(x: f64, factor: f64) f64 {
    return std.math.floor(x / factor) * factor;
}

/// 计算保持宽高比的缩放尺寸，使宽高均为 align_size 的倍数，
/// 且总像素数在 [min_pixels, max_pixels] 范围内。
///
/// 这是 Qwen3VL 等支持动态分辨率模型的核心预处理逻辑。
/// 参考: llama.cpp mtmd-image.cpp calc_size_preserved_ratio() (line 161)
///
/// @param src_width  原始图像宽度
/// @param src_height 原始图像高度
/// @param align_size 对齐单位（patch_size * n_merge，Qwen3VL: 16*2=32）
/// @param min_pixels 最小像素数（来自 GGUF clip.vision.image_min_pixels）
/// @param max_pixels 最大像素数（来自 GGUF clip.vision.image_max_pixels）
/// @returns 缩放后的目标宽高
pub fn calcSizePreservedRatio(
    src_width: u32,
    src_height: u32,
    align_size: u32,
    min_pixels: u32,
    max_pixels: u32,
) Size2D {
    const width: f64 = @floatFromInt(src_width);
    const height: f64 = @floatFromInt(src_height);
    const align_f: f64 = @floatFromInt(align_size);

    // 始终先向上对齐
    var h_bar = @max(align_f, fnRoundByFactor(height, align_f));
    var w_bar = @max(align_f, fnRoundByFactor(width, align_f));

    const min_px: f64 = @floatFromInt(min_pixels);
    const max_px: f64 = @floatFromInt(max_pixels);

    if (h_bar * w_bar > max_px) {
        // 超出最大像素限制，按比例缩小
        const beta = std.math.sqrt((height * width) / max_px);
        h_bar = @max(align_f, fnFloorByFactor(height / beta, align_f));
        w_bar = @max(align_f, fnFloorByFactor(width / beta, align_f));
    } else if (h_bar * w_bar < min_px) {
        // 低于最小像素限制，按比例放大
        const beta = std.math.sqrt(min_px / (height * width));
        h_bar = fnCeilByFactor(height * beta, align_f);
        w_bar = fnCeilByFactor(width * beta, align_f);
    }

    return .{
        .width = @intFromFloat(w_bar),
        .height = @intFromFloat(h_bar),
    };
}

// ============================================================================
// 图像预处理
// ============================================================================

/// 图像预处理结果：RGB 像素数据 + 尺寸信息
pub const ProcessedImage = struct {
    /// RGB 像素数据 [height * width * 3]，值范围 [0, 255]
    data: []u8,
    width: u32,
    height: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ProcessedImage) void {
        self.allocator.free(self.data);
    }
};

/// 从 PPM (P6 binary) 文件加载并预处理图像
/// PPM P6 格式:
///   P6\n<width> <height>\n<maxval>\n<binary RGB data>
pub fn loadPPM(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
    target_size: u32,
) !ProcessedImage {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, file_path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const raw = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(raw);
    const total_read = try file.readPositionalAll(io, raw, 0);
    if (total_read != raw.len) return error.FileReadError;

    // 解析 PPM header: "P6\n<width> <height>\n<maxval>\n"
    if (raw.len < 10 or !std.mem.eql(u8, raw[0..3], "P6\n")) {
        return error.InvalidPPMFormat;
    }

    var pos: usize = 3;
    // 跳过注释行
    while (pos < raw.len and raw[pos] == '#') {
        while (pos < raw.len and raw[pos] != '\n') pos += 1;
        pos += 1;
    }

    // 解析宽度
    const w_start = pos;
    while (pos < raw.len and raw[pos] != ' ') pos += 1;
    if (pos >= raw.len) return error.InvalidPPMFormat;
    const width = try std.fmt.parseInt(u32, raw[w_start..pos], 10);
    pos += 1;

    // 解析高度
    const h_start = pos;
    while (pos < raw.len and raw[pos] != '\n') pos += 1;
    if (pos >= raw.len) return error.InvalidPPMFormat;
    const height = try std.fmt.parseInt(u32, raw[h_start..pos], 10);
    pos += 1;

    // 解析最大颜色值
    const m_start = pos;
    while (pos < raw.len and raw[pos] != '\n') pos += 1;
    if (pos >= raw.len) return error.InvalidPPMFormat;
    const max_val = try std.fmt.parseInt(u32, raw[m_start..pos], 10);
    pos += 1;

    if (max_val != 255) return error.UnsupportedPPMDepth;

    const pixel_data = raw[pos..];
    const expected_len = @as(usize, width * height * 3);
    if (pixel_data.len < expected_len) return error.TruncatedPPMData;

    // 双线性插值缩放到目标尺寸
    const resized = try bilinearResizeRGB(allocator, pixel_data[0..expected_len], width, height, target_size, target_size);

    log.info("Loaded PPM image: {d}x{d} -> {d}x{d}", .{ width, height, target_size, target_size });

    return ProcessedImage{
        .data = resized,
        .width = target_size,
        .height = target_size,
        .allocator = allocator,
    };
}

/// 从文件加载图像（自动检测格式）
/// 支持 JPEG, PNG, BMP, GIF, TGA (via stb_image) 和 PPM
pub fn loadImage(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
    target_size: u32,
    format_hint: ?[]const u8,
) !ProcessedImage {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, file_path, .{ .mode = .read_only });
    defer file.close(io);

    // 读取文件头以检测格式
    var header_buf: [16]u8 = undefined;
    const n_read = try file.readPositionalAll(io, &header_buf, 0);
    if (n_read < 3) return error.InvalidImageFormat;

    const is_ppm = n_read >= 3 and std.mem.eql(u8, header_buf[0..3], "P6\n") or
        (n_read >= 3 and std.mem.eql(u8, header_buf[0..3], "P3\n"));
    const is_jpeg = n_read >= 3 and header_buf[0] == 0xFF and header_buf[1] == 0xD8 and header_buf[2] == 0xFF;
    const is_png = n_read >= 8 and std.mem.eql(u8, header_buf[0..8], &.{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });
    const is_gif = n_read >= 6 and (std.mem.eql(u8, header_buf[0..6], "GIF87a") or
        std.mem.eql(u8, header_buf[0..6], "GIF89a"));
    const is_bmp = n_read >= 2 and header_buf[0] == 'B' and header_buf[1] == 'M';

    const use_ppm = if (format_hint) |hint|
        std.mem.eql(u8, hint, "ppm")
    else
        is_ppm;

    if (use_ppm) {
        return loadPPM(allocator, io, file_path, target_size);
    }

    if (!is_jpeg and !is_png and !is_gif and !is_bmp) {
        return error.UnsupportedImageFormat;
    }

    // 使用 stb_image 加载
    const stat = try file.stat(io);
    const raw = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(raw);
    const total_read = try file.readPositionalAll(io, raw, 0);
    if (total_read != raw.len) return error.FileReadError;

    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const pixels = stb_image.loadFromMemory(raw.ptr, @intCast(raw.len), &w, &h, &comp, 3);
    if (pixels == null) {
        const reason = stb_image.failureReason();
        log.err("stb_image failed to load {s}: {s}", .{ file_path, reason });
        return error.ImageDecodeFailed;
    }
    defer stb_image.free(pixels);

    const src_w: u32 = @intCast(w);
    const src_h: u32 = @intCast(h);
    const pixel_bytes = pixels.?[0..@as(usize, @intCast(src_w * src_h * 3))];

    // 双线性插值缩放到目标尺寸
    const resized = try bilinearResizeRGB(allocator, pixel_bytes, src_w, src_h, target_size, target_size);

    log.info("Loaded image via stb_image: {d}x{d} -> {d}x{d}", .{ src_w, src_h, target_size, target_size });

    return ProcessedImage{
        .data = resized,
        .width = target_size,
        .height = target_size,
        .allocator = allocator,
    };
}

/// 从原始 RGB 数据创建 ProcessedImage（自动缩放）
pub fn fromRawRGB(
    allocator: std.mem.Allocator,
    rgb_data: []const u8,
    width: u32,
    height: u32,
    target_size: u32,
) !ProcessedImage {
    const n_channels: u32 = 3;
    const src_len = width * height * n_channels;
    if (rgb_data.len < src_len) return error.InvalidImageData;

    const resized = try bilinearResizeRGB(allocator, rgb_data[0..@intCast(src_len)], width, height, target_size, target_size);

    return ProcessedImage{
        .data = resized,
        .width = target_size,
        .height = target_size,
        .allocator = allocator,
    };
}

/// 缩放 RGB 图像到指定宽高（支持非正方形）
/// 这是 bilinearResizeRGB 的公共包装，供外部模块使用。
pub fn resizeRGB(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) ![]u8 {
    return bilinearResizeRGB(allocator, src, src_w, src_h, dst_w, dst_h);
}

// ============================================================================
// 双线性插值
// ============================================================================

/// 双线性插值缩放 RGB 图像
fn bilinearResizeRGB(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) ![]u8 {
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
                const idx00: usize = (sy0 * src_w + sx0) * 3 + c;
                const idx01: usize = (sy0 * src_w + sx1) * 3 + c;
                const idx10: usize = (sy1 * src_w + sx0) * 3 + c;
                const idx11: usize = (sy1 * src_w + sx1) * 3 + c;

                const v00: f64 = @floatFromInt(src[idx00]);
                const v01: f64 = @floatFromInt(src[idx01]);
                const v10: f64 = @floatFromInt(src[idx10]);
                const v11: f64 = @floatFromInt(src[idx11]);

                const iy0: f64 = v00 * (1.0 - fx) + v01 * fx;
                const iy1: f64 = v10 * (1.0 - fx) + v11 * fx;
                const val: f64 = iy0 * (1.0 - fy) + iy1 * fy;

                const idx: usize = (dy * dst_w + dx) * 3 + c;
                dst[idx] = @intFromFloat(@round(val));
            }
        }
    }

    return dst;
}

// ============================================================================
// PPM 解析辅助
// ============================================================================

fn skipWhitespace(data: []const u8, start: usize) usize {
    var pos = start;
    while (pos < data.len and (data[pos] == ' ' or data[pos] == '\t' or data[pos] == '\n' or data[pos] == '\r')) {
        pos += 1;
    }
    return pos;
}

// ============================================================================
// 图像转张量
// ============================================================================

/// 图像归一化模式
pub const ImageNormalize = enum {
    /// 除以 255.0，范围 [0, 1]
    div255,
    /// 使用 ImageNet 均值和标准差
    imagenet,
    /// SigLIP 风格：除以 255 再映射到 [-1, 1]：pixel = (x/255)*2 - 1
    siglip,
    /// 不归一化，保留原始 [0, 255] 范围
    none,
};

/// 将 ProcessedImage 转换为 ggml F32 张量 [3, height, width]
/// 像素值归一化到 [0, 1]（除以 255.0）
pub fn imageToTensor(
    ctx: *ggml.Context,
    image: *const ProcessedImage,
    normalize: ImageNormalize,
) !*ggml.Tensor {
    const h: i64 = @intCast(image.height);
    const w: i64 = @intCast(image.width);

    // 创建 3D 张量 [width, height, 3] (ggml 列优先)
    const tensor = try ctx.newTensor3d(ggml.Type.f32, w, h, 3);
    tensor.setName("image_input");

    const data = tensor.dataF32();
    const pixel_count = image.data.len;

    switch (normalize) {
        .div255 => {
            for (0..pixel_count) |i| {
                data[i] = @as(f32, @floatFromInt(image.data[i])) / 255.0;
            }
        },
        .siglip => {
            for (0..pixel_count) |i| {
                data[i] = (@as(f32, @floatFromInt(image.data[i])) / 255.0) * 2.0 - 1.0;
            }
        },
        .none => {
            for (0..pixel_count) |i| {
                data[i] = @as(f32, @floatFromInt(image.data[i]));
            }
        },
        .imagenet => {
            const mean: [3]f32 = .{ 0.485, 0.456, 0.406 };
            const std_dev: [3]f32 = .{ 0.229, 0.224, 0.225 };
            for (0..@as(usize, @intCast(h))) |y| {
                for (0..@as(usize, @intCast(w))) |x| {
                    for (0..3) |c| {
                        const src_idx = (y * @as(usize, @intCast(w)) + x) * 3 + c;
                        const dst_idx = c * @as(usize, @intCast(h * w)) + y * @as(usize, @intCast(w)) + x;
                        const pixel = @as(f32, @floatFromInt(image.data[src_idx])) / 255.0;
                        data[dst_idx] = (pixel - mean[c]) / std_dev[c];
                    }
                }
            }
        },
    }

    return tensor;
}
