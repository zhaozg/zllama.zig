//! 音频处理流水线编排器
//!
//! 串联各处理阶段，从 WAV 文件到 Mel 频谱特征。
//! 使用 Arena 分配器管理临时内存，避免碎片化。
//!
//! 参考: llama.cpp mtmd-audio.cpp

const std = @import("std");
const ggml = @import("ggml");
const fft_mod = @import("fft");

const config_mod = @import("config.zig");
const types = @import("types.zig");
const loader = @import("loader.zig");
const framing = @import("framing.zig");
const mel = @import("mel.zig");
const log_transform = @import("log_transform.zig");
const encoder = @import("encoder.zig");
const postprocess = @import("postprocess.zig");

const log = std.log.scoped(.audio_pipeline);

// ============================================================================
// 公开 API
// ============================================================================

/// 完整的音频处理流水线：从 WAV 文件到 Mel 频谱
///
/// 处理步骤（匹配 llama.cpp gemma4a 预处理器）：
/// 1. 加载 WAV 文件
/// 2. 重采样到 16kHz（如需要）
/// 3. 半因果左填充 + 分帧 + Hann 窗口
/// 4. FFT 功率谱
/// 5. Mel 滤波器组（HTK 尺度, 128 bins, 0-8000 Hz）
/// 6. 自然对数压缩
///
/// @param allocator 分配器（用于返回的 ProcessedAudio）
/// @param io I/O 实例
/// @param file_path WAV 文件路径
/// @param params 预处理参数
/// @returns ProcessedAudio（调用者负责 deinit）
pub fn processWavFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
    params: config_mod.AudioPreprocessParams,
) !types.ProcessedAudio {
    // 使用 Arena 管理临时内存
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp_alloc = arena.allocator();

    // 1. 加载 WAV 文件
    const wav_result = try loader.loadWav(tmp_alloc, io, file_path);
    if (wav_result.samples.len == 0) return error.EmptyAudio;

    // 2. 重采样到目标采样率（如需要）
    const samples = if (wav_result.info.sample_rate != params.sample_rate)
        try loader.resample(tmp_alloc, wav_result.samples, wav_result.info.sample_rate, params.sample_rate)
    else
        wav_result.samples;

    // 3. 分帧 + 加窗
    const framed = try framing.frameAudio(tmp_alloc, samples, .{
        .frame_length = params.frame_length,
        .hop_length = params.hop_length,
        .n_fft = params.n_fft,
    });

    // 4. FFT 功率谱
    var fft_engine = try fft_mod.AccelFFT.init(tmp_alloc, params.n_fft);
    defer fft_engine.deinit();

    const n_freqs: u32 = params.n_fft / 2 + 1;
    const spectrum = try tmp_alloc.alloc(f32, n_freqs);

    // 5. 预计算 Mel 滤波器组
    const filterbank = try mel.computeFilterbank(
        tmp_alloc,
        params.n_mel_bins,
        params.n_fft,
        params.sample_rate,
        params.mel_f_min,
        params.mel_f_max,
    );

    // 6. 逐帧处理：FFT -> Mel -> Log
    const mel_out = try allocator.alloc(f32, @as(usize, params.n_mel_bins) * @as(usize, framed.n_frames));

    for (0..framed.n_frames) |fi| {
        const frame_start = fi * @as(usize, framed.frame_size);

        // FFT 功率谱
        fft_engine.powerSpectrum(framed.frames[frame_start .. frame_start + framed.frame_size], spectrum);

        // Mel 滤波器组
        const mel_frame = mel_out[fi * @as(usize, params.n_mel_bins) .. (fi + 1) * @as(usize, params.n_mel_bins)];
        mel.applyFilterbank(spectrum, filterbank, params.n_mel_bins, n_freqs, mel_frame);

        // 对数变换
        log_transform.applyLogTransformInPlace(mel_frame, params.log_offset);
    }

    log.info("Audio pipeline: {d} frames x {d} mel bins, sr={d}Hz", .{
        framed.n_frames, params.n_mel_bins, params.sample_rate,
    });

    return .{
        .data = mel_out,
        .n_mel_bins = params.n_mel_bins,
        .n_frames = framed.n_frames,
        .allocator = allocator,
    };
}

/// 从 PCM F32 音频样本计算 Mel 频谱（不经过文件加载）
///
/// @param allocator 分配器（用于返回的 ProcessedAudio）
/// @param audio_data PCM F32 音频样本
/// @param sample_rate 音频采样率
/// @param params 预处理参数
/// @returns ProcessedAudio（调用者负责 deinit）
pub fn processPcmSamples(
    allocator: std.mem.Allocator,
    audio_data: []const f32,
    sample_rate: u32,
    params: config_mod.AudioPreprocessParams,
) !types.ProcessedAudio {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp_alloc = arena.allocator();

    // 重采样到目标采样率（如需要）
    const samples = if (sample_rate != params.sample_rate)
        try loader.resample(tmp_alloc, audio_data, sample_rate, params.sample_rate)
    else
        audio_data;

    // 分帧 + 加窗
    const framed = try framing.frameAudio(tmp_alloc, samples, .{
        .frame_length = params.frame_length,
        .hop_length = params.hop_length,
        .n_fft = params.n_fft,
    });

    // FFT 功率谱
    var fft_engine = try fft_mod.AccelFFT.init(tmp_alloc, params.n_fft);
    defer fft_engine.deinit();

    const n_freqs: u32 = params.n_fft / 2 + 1;
    const spectrum = try tmp_alloc.alloc(f32, n_freqs);

    // 预计算 Mel 滤波器组
    const filterbank = try mel.computeFilterbank(
        tmp_alloc,
        params.n_mel_bins,
        params.n_fft,
        params.sample_rate,
        params.mel_f_min,
        params.mel_f_max,
    );

    // 逐帧处理
    const mel_out = try allocator.alloc(f32, @as(usize, params.n_mel_bins) * @as(usize, framed.n_frames));

    for (0..framed.n_frames) |fi| {
        const frame_start = fi * @as(usize, framed.frame_size);

        fft_engine.powerSpectrum(framed.frames[frame_start .. frame_start + framed.frame_size], spectrum);

        const mel_frame = mel_out[fi * @as(usize, params.n_mel_bins) .. (fi + 1) * @as(usize, params.n_mel_bins)];
        mel.applyFilterbank(spectrum, filterbank, params.n_mel_bins, n_freqs, mel_frame);

        log_transform.applyLogTransformInPlace(mel_frame, params.log_offset);
    }

    return .{
        .data = mel_out,
        .n_mel_bins = params.n_mel_bins,
        .n_frames = framed.n_frames,
        .allocator = allocator,
    };
}

/// 将 Mel 频谱数据转换为 F32 张量 [n_frames, n_mel_bins]
pub fn melToTensor(
    ctx: *ggml.Context,
    mel_data: []const f32,
    n_frames: u32,
    n_mel_bins: u32,
) !*ggml.Tensor {
    return postprocess.melToTensor(ctx, mel_data, n_frames, n_mel_bins);
}
