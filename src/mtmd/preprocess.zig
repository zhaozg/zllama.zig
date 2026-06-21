//! 多模态预处理模块
//!
//! 提供图像和音频数据的预处理功能：
//! - 图像：resize（双线性插值）+ 像素归一化
//! - 音频：WAV 读取 + Mel 频谱提取（STFT + Mel 滤波器组）
//!
//! 图像格式支持：PPM (P6 binary), JPEG, PNG, BMP, GIF, TGA (via stb_image)
//! 音频格式支持：WAV (16-bit PCM, mono/stereo)
//!
//! 参考: llama.cpp tools/mtmd/clip-impl.h (clip_image_preprocess)
//!       llama.cpp tools/mtmd/models/gemma4a.cpp (mel spectrogram)

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const fft_mod = @import("fft");
const math = std.math;
const stb_image = @import("stb_image");

const log = std.log.scoped(.mm_preprocess);

// ============================================================================
// 音频参数常量
// ============================================================================

/// 默认音频采样率 (Gemma4 E2B)
pub const AUDIO_SAMPLE_RATE: u32 = 16000;
/// STFT 帧长 (25ms @ 16kHz, matches llama.cpp gemma4a win_length=400)
pub const AUDIO_FRAME_LENGTH: u32 = 400;
/// STFT 帧移 (10ms @ 16kHz)
pub const AUDIO_HOP_LENGTH: u32 = 160;
/// FFT 点数（2 的幂）
pub const AUDIO_N_FFT: u32 = 512;
/// Mel 滤波器组数量
pub const AUDIO_N_MEL_BINS: u32 = 128;
/// Mel 最低频率 (gemma4a uses 0.0, full range)
pub const AUDIO_MEL_F_MIN: f32 = 0.0;
/// Mel 最高频率 (gemma4a uses sr/2)
pub const AUDIO_MEL_F_MAX: f32 = 8000.0;
/// 预加重系数 (gemma4a disables pre-emphasis)
pub const AUDIO_PRE_EMPHASIS: f32 = 0.0;
/// 对数偏移（防止 log(0)，matches gemma4a mel_floor=0.001）
pub const AUDIO_LOG_OFFSET: f32 = 0.001;

/// 音频预处理参数（可从 GGUF 元数据加载）
pub const AudioPreprocessParams = struct {
    sample_rate: u32 = AUDIO_SAMPLE_RATE,
    frame_length: u32 = AUDIO_FRAME_LENGTH,
    hop_length: u32 = AUDIO_HOP_LENGTH,
    n_fft: u32 = AUDIO_N_FFT,
    n_mel_bins: u32 = AUDIO_N_MEL_BINS,
    mel_f_min: f32 = AUDIO_MEL_F_MIN,
    mel_f_max: f32 = AUDIO_MEL_F_MAX,
    pre_emphasis: f32 = AUDIO_PRE_EMPHASIS,
    log_offset: f32 = AUDIO_LOG_OFFSET,

    /// 从音频编码器参数构建（部分参数从 GGUF 元数据加载）
    pub fn fromAudioEncoder(n_mel_bins: u32) AudioPreprocessParams {
        return .{ .n_mel_bins = n_mel_bins };
    }

    /// 从 GGUF 元数据加载音频预处理参数
    pub fn fromGGUF(gguf_file: *const gguf.GGUFFile) AudioPreprocessParams {
        var p = AudioPreprocessParams{};
        if (gguf_file.getU32("clip.audio.num_mel_bins")) |v| p.n_mel_bins = v;
        if (gguf_file.getU32("clip.audio.sample_rate")) |v| p.sample_rate = v;
        if (gguf_file.getU32("clip.audio.n_fft")) |v| p.n_fft = v;
        if (gguf_file.getU32("clip.audio.hop_length")) |v| p.hop_length = v;
        if (gguf_file.getU32("clip.audio.window_length")) |v| p.frame_length = v;
        return p;
    }
};

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
/// 通用图像加载器 — 自动检测格式（PPM / JPEG / PNG / BMP / GIF / TGA）
///
/// PPM (P6) 使用内置解析器，其他格式委托给 stb_image。
/// 所有图像都会被缩放到 target_size × target_size (双线性插值)。
pub fn loadImage(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
    target_size: u32,
    format_hint: enum { auto, ppm, stb },
) !ProcessedImage {
    // 读取文件头检测格式
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, file_path, .{ .mode = .read_only });
    defer file.close(io);

    var header_buf: [12]u8 = undefined;
    const n_read = try file.readPositionalAll(io, &header_buf, 0);

    // PPM P6 magic
    const is_ppm = n_read >= 3 and std.mem.eql(u8, header_buf[0..3], "P6\n") or
        (n_read >= 2 and std.mem.eql(u8, header_buf[0..2], "P6"));

    // JPEG magic: FF D8 FF
    const is_jpeg = n_read >= 3 and header_buf[0] == 0xFF and header_buf[1] == 0xD8 and header_buf[2] == 0xFF;

    // PNG magic: 89 50 4E 47 0D 0A 1A 0A
    const is_png = n_read >= 8 and std.mem.eql(u8, header_buf[0..8], &.{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });

    // GIF magic: GIF87a / GIF89a
    const is_gif = n_read >= 6 and (std.mem.eql(u8, header_buf[0..6], "GIF87a") or
        std.mem.eql(u8, header_buf[0..6], "GIF89a"));

    // BMP magic: BM
    const is_bmp = n_read >= 2 and header_buf[0] == 'B' and header_buf[1] == 'M';

    const use_ppm = switch (format_hint) {
        .ppm => true,
        .stb => false,
        .auto => is_ppm and !is_jpeg and !is_png and !is_gif and !is_bmp,
    };

    if (use_ppm) {
        return loadPPM(allocator, io, file_path, target_size);
    }

    // stb_image path — read full file into buffer and decode
    const stat = try file.stat(io);
    const raw = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(raw);
    const total_read = try file.readPositionalAll(io, raw, 0);
    if (total_read != raw.len) return error.FileReadError;

    var w: i32 = 0;
    var h: i32 = 0;
    var comp: i32 = 0;

    // Force RGB (3 channels)
    const pixels = stb_image.loadFromMemory(raw.ptr, @intCast(raw.len), &w, &h, &comp, 3);
    if (pixels == null) {
        const reason = stb_image.failureReason();
        log.err("stb_image failed: {s}", .{reason});
        return error.ImageDecodeFailed;
    }
    defer stb_image.free(pixels);

    const src_w: u32 = @intCast(w);
    const src_h: u32 = @intCast(h);
    const pixel_bytes = pixels.?[0..@as(usize, @intCast(src_w * src_h * 3))];

    log.info("Decoded image via stb_image: {d}x{d} ({d} channels)", .{ src_w, src_h, comp });

    // Resize to target size
    const resized = try bilinearResizeRGB(allocator, pixel_bytes, src_w, src_h, target_size, target_size);

    log.info("Processed image: {d}x{d} -> {d}x{d}", .{ src_w, src_h, target_size, target_size });

    return .{
        .data = resized,
        .width = target_size,
        .height = target_size,
        .allocator = allocator,
    };
}

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
// WAV 文件读取
// ============================================================================

/// WAV 文件信息
pub const WavInfo = struct {
    sample_rate: u32,
    num_channels: u16,
    bits_per_sample: u16,
    num_samples: u32,
};

/// 从 WAV 文件加载 PCM 数据
/// 支持 16-bit PCM, mono/stereo
/// 返回 F32 单声道样本（stereo 取左右平均）
pub fn loadWav(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
) !struct { samples: []f32, info: WavInfo } {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, file_path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const raw = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(raw);
    const total_read = try file.readPositionalAll(io, raw, 0);
    if (total_read != raw.len) return error.FileReadError;

    // 验证 RIFF header
    if (raw.len < 44 or !std.mem.eql(u8, raw[0..4], "RIFF") or !std.mem.eql(u8, raw[8..12], "WAVE")) {
        return error.InvalidWavFormat;
    }

    var pos: usize = 12;
    var fmt_found = false;
    var data_found = false;
    var audio_format: u16 = 0;
    var num_channels: u16 = 0;
    var sample_rate: u32 = 0;
    var bits_per_sample: u16 = 0;
    var data_offset: usize = 0;
    var data_size: u32 = 0;

    // 解析 chunks
    while (pos + 8 <= raw.len) {
        const chunk_id = raw[pos .. pos + 4];
        const chunk_size = std.mem.readInt(u32, @ptrCast(raw.ptr + pos + 4), .little);
        pos += 8;

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (pos + 16 > raw.len) return error.InvalidWavFormat;

            audio_format = std.mem.readInt(u16, @ptrCast(raw.ptr + pos), .little);
            num_channels = std.mem.readInt(u16, @ptrCast(raw.ptr + pos + 2), .little);
            sample_rate = std.mem.readInt(u32, @ptrCast(raw.ptr + pos + 4), .little);
            _ = std.mem.readInt(u32, @ptrCast(raw.ptr + pos + 8), .little); // byte_rate
            _ = std.mem.readInt(u16, @ptrCast(raw.ptr + pos + 12), .little); // block_align
            bits_per_sample = std.mem.readInt(u16, @ptrCast(raw.ptr + pos + 14), .little);
            fmt_found = true;
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data_offset = pos;
            data_size = chunk_size;
            data_found = true;
            // 我们可以跳过剩余 chunk
        }

        pos += chunk_size;
        // 对齐到偶数边界（WAV chunks 是 2 字节对齐）
        if (chunk_size % 2 != 0) {
            pos += 1;
        }
    }

    if (!fmt_found or !data_found) return error.InvalidWavFormat;
    if (audio_format != 1) return error.UnsupportedWavFormat; // 仅支持 PCM
    if (bits_per_sample != 16) return error.UnsupportedBitDepth;

    const bytes_per_sample: u32 = @divExact(@as(u32, bits_per_sample), 8);
    const num_samples: u32 = data_size / (bytes_per_sample * num_channels);

    if (data_offset + data_size > raw.len) return error.TruncatedWavData;

    // 转换为 F32 单声道
    const samples = try allocator.alloc(f32, num_samples);
    const raw_samples_ptr = raw.ptr + data_offset;

    if (num_channels == 1) {
        var i: u32 = 0;
        while (i < num_samples) : (i += 1) {
            const sample_i16 = std.mem.readInt(i16, @ptrCast(raw_samples_ptr + @as(usize, i * 2)), .little);
            samples[i] = @as(f32, @floatFromInt(sample_i16)) / 32768.0;
        }
    } else if (num_channels == 2) {
        var i: u32 = 0;
        while (i < num_samples) : (i += 1) {
            const off: usize = @as(usize, i) * 4;
            const left = std.mem.readInt(i16, @ptrCast(raw_samples_ptr + off), .little);
            const right = std.mem.readInt(i16, @ptrCast(raw_samples_ptr + off + 2), .little);
            samples[i] = @as(f32, @floatFromInt(left + right)) / 65536.0;
        }
    } else {
        return error.UnsupportedChannels;
    }

    log.info("Loaded WAV: {d} Hz, {d} ch, {d} bits, {d} samples ({d})", .{ sample_rate, num_channels, bits_per_sample, num_samples, @as(f64, @floatFromInt(num_samples)) / @as(f64, @floatFromInt(sample_rate)) });

    return .{
        .samples = samples,
        .info = .{
            .sample_rate = sample_rate,
            .num_channels = num_channels,
            .bits_per_sample = bits_per_sample,
            .num_samples = num_samples,
        },
    };
}
// ============================================================================
// FFT 实现：通过 @import("fft") 使用 Apple Accelerate vDSP
// ============================================================================

// ============================================================================
// Mel 滤波器组
// ============================================================================

/// 预计算 Mel 滤波器组权重矩阵 [n_mel_bins, n_fft/2+1]
/// 使用 HTK Mel 尺度（与 llama.cpp gemma4a 一致）
fn computeMelFilterbank(
    allocator: std.mem.Allocator,
    n_mel_bins: u32,
    n_fft: u32,
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
) ![]f32 {
    const n_freqs: usize = @intCast(n_fft / 2 + 1);
    const filterbank = try allocator.alloc(f32, @as(usize, n_mel_bins) * n_freqs);
    @memset(filterbank, 0.0);

    // HTK Mel scale: mel = 2595 * log10(1 + f/700)
    const hzToMel = struct {
        fn call(hz: f32) f32 {
            return 2595.0 * math.log10(1.0 + hz / 700.0);
        }
    }.call;

    const melToHz = struct {
        fn call(mel: f32) f32 {
            return 700.0 * (math.pow(f32, 10.0, mel / 2595.0) - 1.0);
        }
    }.call;

    const mel_min = hzToMel(f_min);
    const mel_max = hzToMel(f_max);

    // 在 Mel 尺度上均匀分布 n_mel_bins + 2 个点
    const mel_step = (mel_max - mel_min) / @as(f32, @floatFromInt(n_mel_bins + 1));

    // 每个滤波器的中心频率（Hz）
    var mel_centers = try allocator.alloc(f32, @as(usize, n_mel_bins) + 2);
    defer allocator.free(mel_centers);
    var center_hz = try allocator.alloc(f32, @as(usize, n_mel_bins) + 2);
    defer allocator.free(center_hz);

    for (0..@as(usize, n_mel_bins) + 2) |i| {
        mel_centers[i] = mel_min + @as(f32, @floatFromInt(i)) * mel_step;
        center_hz[i] = melToHz(mel_centers[i]);
    }

    // FFT bin 对应的频率
    var bin_freqs = try allocator.alloc(f32, n_freqs);
    defer allocator.free(bin_freqs);
    for (0..n_freqs) |i| {
        bin_freqs[i] = @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(sample_rate)) / @as(f32, @floatFromInt(n_fft));
    }

    // 对每个 mel bin 构建三角形滤波器
    for (0..@as(usize, n_mel_bins)) |m| {
        const left = center_hz[m];
        const center = center_hz[m + 1];
        const right = center_hz[m + 2];
        const row = filterbank[m * n_freqs .. (m + 1) * n_freqs];

        for (0..n_freqs) |k| {
            const freq = bin_freqs[k];
            if (freq <= left) {
                row[k] = 0.0;
            } else if (freq <= center) {
                row[k] = (freq - left) / (center - left);
            } else if (freq <= right) {
                row[k] = (right - freq) / (right - center);
            } else {
                row[k] = 0.0;
            }
        }
    }

    return filterbank;
}

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
/// 处理步骤（匹配 llama.cpp gemma4a 预处理器）：
/// 1. 半因果左填充（pad_left = window_len/2）
/// 2. 短时傅里叶变换（STFT）
///    - 帧长（frame_length）= 400 samples (25ms @ 16kHz)
///    - 帧移（hop_length）= 160 samples (10ms @ 16kHz)
///    - Hann 窗口（零填充到 FFT 大小）
///    - FFT 点数 = 512 (n_fft)
///    - 幅度谱（use_magnitude=true）
/// 3. Mel 滤波器组（HTK 尺度, 128 bins, 0-8000 Hz）
/// 4. 自然对数压缩（use_natural_log=true）
pub fn computeMelSpectrogram(
    allocator: std.mem.Allocator,
    audio_data: []const f32,
    sample_rate: u32,
    params: AudioPreprocessParams,
) !ProcessedAudio {
    if (audio_data.len == 0) return error.EmptyAudioData;

    const n_fft: u32 = params.n_fft;
    const f_min: f32 = params.mel_f_min;
    const f_max: f32 = params.mel_f_max;
    const frame_length: u32 = params.frame_length;
    const hop_length: u32 = params.hop_length;
    const n_freqs: u32 = n_fft / 2 + 1;
    const n_mel_bins: u32 = params.n_mel_bins;
    const log_offset: f32 = params.log_offset;

    // Step 1: 半因果左填充（匹配 llama.cpp gemma4a）
    // gemma4a 使用 no_padding=true + 手动左填充 pad_left = window_len/2
    const pad_left: usize = frame_length / 2;
    const n_padded = audio_data.len + pad_left;
    var padded = try allocator.alloc(f32, n_padded);
    defer allocator.free(padded);
    @memset(padded[0..pad_left], 0.0);
    @memcpy(padded[pad_left..], audio_data);

    // 初始化 Accelerate FFT 引擎（Hann 窗 + 重用缓冲区）
    var fft_engine = try fft_mod.AccelFFT.init(allocator, n_fft);
    defer fft_engine.deinit();

    // 预计算 Mel 滤波器组（HTK 尺度）
    const filterbank = try computeMelFilterbank(allocator, n_mel_bins, n_fft, sample_rate, f_min, f_max);
    defer allocator.free(filterbank);

    // Step 2: STFT - 计算帧数
    // 匹配 llama.cpp: (n_samples - frame_size) / frame_step + 1
    const n_frames: u32 = if (n_padded >= frame_length)
        @as(u32, @intCast((n_padded - frame_length) / hop_length)) + 1
    else
        0;

    if (n_frames == 0) return error.AudioTooShort;

    // 输出: [n_mel_bins, n_frames]（mel-major 布局，匹配 llama.cpp）
    const mel_out = try allocator.alloc(f32, @as(usize, n_mel_bins) * @as(usize, n_frames));

    const spectrum = try allocator.alloc(f32, @intCast(n_freqs));
    defer allocator.free(spectrum);

    // 逐帧处理
    var frame_buf: [512]f32 = undefined;
    for (0..n_frames) |fi| {
        const start: usize = fi * @as(usize, hop_length);

        // 提取帧（零填充到 FFT 大小）
        const frame_end = @min(start + frame_length, n_padded);
        @memset(frame_buf[0..n_fft], 0.0);
        @memcpy(frame_buf[0 .. frame_end - start], padded[start..frame_end]);

        // 使用 Accelerate vDSP 计算幅度谱
        fft_engine.powerSpectrum(frame_buf[0 .. frame_end - start], spectrum);

        // Step 3: Mel 滤波器组
        for (0..@as(usize, n_mel_bins)) |m| {
            const row = filterbank[m * @as(usize, n_freqs) .. (m + 1) * @as(usize, n_freqs)];
            var mel_val: f32 = 0.0;
            for (0..@as(usize, n_freqs)) |k| {
                mel_val += row[k] * spectrum[k];
            }
            // Step 4: 自然对数压缩（use_natural_log=true for gemma4a）
            mel_out[m * @as(usize, n_frames) + fi] = @log(@max(mel_val, log_offset));
        }
    }

    log.info("Mel spectrogram: {d} frames x {d} bins, sr={d}Hz", .{ n_frames, n_mel_bins, sample_rate });

    return .{
        .data = mel_out,
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
