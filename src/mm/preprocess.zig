//! 多模态预处理模块
//!
//! 提供图像和音频数据的预处理功能：
//! - 图像：resize（双线性插值）+ 像素归一化
//! - 音频：Mel 频谱提取（TODO：待实现 FFT/滤波器组）
//!
//! 图像格式支持：PPM (P6 binary) 和原始 RGB 字节数组
//!
//! 参考: llama.cpp tools/mtmd/clip-impl.h (clip_image_preprocess)

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.mm_preprocess);

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
    const total_read = try file.readPositionalAll(io, raw, 0);
    if (total_read != raw.len) return error.FileReadError;
    // 解析 PPM header
    if (raw.len < 10 or !std.mem.startsWith(u8, raw, "P6")) {
        return error.InvalidPPMFormat;
    }

    var pos: usize = 2; // skip "P6"
    var width: u32 = 0;
    var height: u32 = 0;
    var maxval: u32 = 0;

    // 跳过空白并解析
    pos = skipWhitespace(raw, pos);

    // 解析 width
    const w_start = pos;
    while (pos < raw.len and raw[pos] != ' ' and raw[pos] != '\n' and raw[pos] != '\r' and raw[pos] != '\t') : (pos += 1) {}
    width = std.fmt.parseUnsigned(u32, raw[w_start..pos], 10) catch return error.InvalidPPMFormat;

    pos = skipWhitespace(raw, pos);

    // 解析 height
    const h_start = pos;
    while (pos < raw.len and raw[pos] != ' ' and raw[pos] != '\n' and raw[pos] != '\r' and raw[pos] != '\t') : (pos += 1) {}
    height = std.fmt.parseUnsigned(u32, raw[h_start..pos], 10) catch return error.InvalidPPMFormat;

    pos = skipWhitespace(raw, pos);

    // 解析 maxval
    const m_start = pos;
    while (pos < raw.len and raw[pos] != ' ' and raw[pos] != '\n' and raw[pos] != '\r' and raw[pos] != '\t') : (pos += 1) {}
    maxval = std.fmt.parseUnsigned(u32, raw[m_start..pos], 10) catch return error.InvalidPPMFormat;

    pos = skipWhitespace(raw, pos);

    if (maxval > 255) return error.UnsupportedBitDepth;

    const pixel_data = raw[pos..];
    const expected_len = @as(usize, width * height * 3);
    if (pixel_data.len < expected_len) return error.TruncatedPPMData;

    // Resize 到目标尺寸
    const resized = try bilinearResizeRGB(allocator, pixel_data[0..expected_len], width, height, target_size, target_size);

    log.info("Loaded PPM image: {d}x{d} -> {d}x{d}", .{ width, height, target_size, target_size });

    return .{
        .data = resized,
        .width = target_size,
        .height = target_size,
        .allocator = allocator,
    };
}

/// 从原始 RGB 字节数组创建处理后的图像（不做 resize）
pub fn fromRawRGB(
    allocator: std.mem.Allocator,
    rgb_data: []const u8,
    width: u32,
    height: u32,
    target_size: u32,
) !ProcessedImage {
    const n_channels: u32 = 3;
    const src_len = width * height * n_channels;
    if (rgb_data.len < src_len) return error.InsufficientData;

    const resized = try bilinearResizeRGB(allocator, rgb_data[0..@intCast(src_len)], width, height, target_size, target_size);

    return .{
        .data = resized,
        .width = target_size,
        .height = target_size,
        .allocator = allocator,
    };
}

/// 双线性插值缩放 RGB 图像
/// @param src 源 RGB 数据 [h][w][3]
/// @param src_w 源宽度
/// @param src_h 源高度
/// @param dst_w 目标宽度
/// @param dst_h 目标高度
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
                dst[idx] = @intFromFloat(@round(@max(0.0, @min(255.0, val))));
            }
        }
    }

    return dst;
}

/// 跳过 PPM header 中的空白字符
fn skipWhitespace(data: []const u8, start: usize) usize {
    var pos = start;
    while (pos < data.len) {
        if (data[pos] == '#') {
            // 跳过注释行
            while (pos < data.len and data[pos] != '\n') : (pos += 1) {}
            if (pos < data.len) pos += 1;
            continue;
        }
        if (data[pos] != ' ' and data[pos] != '\n' and data[pos] != '\r' and data[pos] != '\t') {
            break;
        }
        pos += 1;
    }
    return pos;
}

/// 将 RGB 像素数据转换为 F32 张量并标准化
/// gemma4v: 每个像素 * 2.0 - 1.0 (值域 [-1, 1])
/// @param ctx ggml 计算上下文
/// @param image 处理后的图像
/// @param normalize 标准化策略 (.siglip 或 .none)
/// @returns F32 张量 [3, height, width]，值已标准化
pub fn imageToTensor(
    ctx: *ggml.Context,
    image: *const ProcessedImage,
    normalize: ImageNormalize,
) !*ggml.Tensor {
    const h: i64 = @intCast(image.height);
    const w: i64 = @intCast(image.width);

    // 创建 [3, height, width] = [C, H, W] 张量（ggml 行主序）
    // ggml 使用行主序: ne[0]=width, ne[1]=height, ne[2]=channels
    const tensor = try ctx.newTensor3d(ggml.Type.f32, w, h, 3);
    tensor.setName("image_input");

    const data = tensor.dataF32();
    const pixel_count = image.data.len;

    switch (normalize) {
        .siglip => {
            // SigLIP: pixel * 2.0 / 255.0 - 1.0 -> [-1, 1]
            for (0..pixel_count) |i| {
                data[i] = (@as(f32, @floatFromInt(image.data[i])) / 255.0) * 2.0 - 1.0;
            }
        },
        .none => {
            // 直接复制，不标准化
            for (0..pixel_count) |i| {
                data[i] = @floatFromInt(image.data[i]);
            }
        },
    }

    return tensor;
}

/// 图像标准化策略
pub const ImageNormalize = enum {
    /// SigLIP 标准化: (pixel/255.0) * 2.0 - 1.0 -> [-1, 1]
    siglip,
    /// 不标准化，保留 [0, 255]
    none,
};

// ============================================================================
// 音频预处理（Mel 频谱）
// ============================================================================

/// 音频预处理结果：Mel 频谱特征
pub const ProcessedAudio = struct {
    /// Mel 频谱 [n_mel_bins, n_frames]，值域已标准化
    data: []f32,
    n_mel_bins: u32,
    n_frames: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ProcessedAudio) void {
        self.allocator.free(self.data);
    }
};

/// 从 PCM F32 音频样本计算 Mel 频谱
///
/// TODO: 完整实现需要以下步骤：
/// 1. 预加重（pre-emphasis）系数 0.97
/// 2. 短时傅里叶变换（STFT）
///    - 帧长（frame_length）= 25ms @ 16kHz = 400 samples
///    - 帧移（hop_length）= 10ms @ 16kHz = 160 samples
///    - Hann 窗口
/// 3. Mel 滤波器组（128 bins, 80-7600 Hz）
/// 4. log10 压缩
///
/// 当前实现：创建占位张量（全零），标记为 TODO。
/// 原因：FFT/滤波器组实现需要约 300+ 行代码，
/// 且需要额外的测试验证，无法在本轮完成。
/// 暂时返回占位数据，使多模态流水线可编译。
pub fn computeMelSpectrogram(
    allocator: std.mem.Allocator,
    audio_data: []const f32,
    sample_rate: u32,
    n_mel_bins: u32,
) !ProcessedAudio {
    _ = audio_data;
    _ = sample_rate;

    // TODO: 实现完整的 Mel 频谱计算
    // 当前创建 1 帧占位数据（Conformer 需要至少 n_mel_bins * 1 的输入）
    const n_frames: u32 = 1;
    const data = try allocator.alloc(f32, n_mel_bins * n_frames);
    @memset(data, 0.0);

    log.warn("Mel spectrogram: using placeholder (FFT not yet implemented)", .{});

    return .{
        .data = data,
        .n_mel_bins = n_mel_bins,
        .n_frames = n_frames,
        .allocator = allocator,
    };
}

/// 将 Mel 频谱数据转换为 F32 张量 [n_frames, n_mel_bins]
pub fn melToTensor(
    ctx: *ggml.Context,
    mel: *const ProcessedAudio,
) !*ggml.Tensor {
    const tensor = try ctx.newTensor2d(
        ggml.Type.f32,
        @intCast(mel.n_frames),
        @intCast(mel.n_mel_bins),
    );
    tensor.setName("mel_input");

    const data = tensor.dataBytes();
    @memcpy(data[0 .. mel.data.len * @sizeOf(f32)], @as([*]const u8, @ptrCast(mel.data.ptr))[0 .. mel.data.len * @sizeOf(f32)]);

    return tensor;
}
