//! 音频处理流水线编排器
//!
//! 串联各处理阶段，从 WAV 文件到 Mel 频谱特征。
//! 匹配 llama.cpp mtmd_audio_preprocessor_gemma4a 的精确逻辑。
//!
//! 参考: llama.cpp mtmd-audio.cpp (mtmd_audio_preprocessor_gemma4a)

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
const helper = @import("helper");

const log = std.log.scoped(.audio_pipeline);

/// 从 PCM F32 音频样本计算 Mel 频谱（不经过文件加载）
/// 匹配 llama.cpp mtmd_audio_preprocessor_gemma4a::preprocess() 的精确逻辑：
///   - 无预加重 (preemph=0.0)
///   - 无中心填充 (no_padding=true, 使用自定义 semicausal padding)
///   - 使用幅度谱 |X| (use_magnitude=true)
///   - 自然对数 (use_natural_log=true)
///   - mel_floor=0.001
///   - HTK mel scale, slaney_area_norm=false
///   - Hann window zero-padded to FFT size
///   - 30秒分块处理
///
/// @param allocator 分配器（用于返回的 ProcessedAudio）
/// @param audio_data PCM F32 音频样本
/// @param sample_rate 音频采样率
/// @param params 预处理参数
/// @returns ProcessedAudio（调用者负责 deinit）
pub fn processPcmSamples(
    io: std.Io,
    allocator: std.mem.Allocator,
    audio_data: []const f32,
    sample_rate: u32,
    params: config_mod.AudioPreprocessParams,
) !types.ProcessedAudio {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp_alloc = arena.allocator();

    // 重采样到目标采样率（如需要）
    const resampled = if (sample_rate != params.sample_rate)
        try loader.resample(tmp_alloc, audio_data, sample_rate, params.sample_rate)
    else
        audio_data;

    // gemma4a 不使用预加重 (preemph=0.0)
    const samples = resampled;

    helper.mtmdDebugSaveData(io, "zllama_audio_samples.json", "audio_samples", samples) catch |err| {
        log.debug("Failed to save audio samples debug data: {}", .{err});
    };

    // 预计算 Mel 滤波器组（HTK scale, slaney_area_norm=false）
    const filterbank = try mel.computeFilterbank(
        tmp_alloc,
        params.n_mel_bins,
        params.n_fft,
        params.sample_rate,
        params.mel_f_min,
        params.mel_f_max,
    );

    // 预计算 Hann 窗口（零填充到 FFT 大小）
    const hann_window = framing.computeHannWindow(params.frame_length);

    // 初始化 FFT 引擎
    var fft_engine = try fft_mod.AccelFFT.init(tmp_alloc, params.n_fft);
    defer fft_engine.deinit();

    const n_freqs: u32 = params.n_fft / 2 + 1;
    const frame_size: u32 = params.n_fft;
    const hop: u32 = params.hop_length;
    const pad_left: u32 = params.frame_length / 2;

    // 匹配 llama.cpp gemma4a: 30秒分块处理
    const chunk_samples: usize = 30 * params.sample_rate;

    // 先计算总帧数
    var total_frames: u32 = 0;

    var offset: usize = 0;
    while (offset < samples.len) {
        const chunk_len = @min(chunk_samples, samples.len - offset);

        // 匹配 llama.cpp gemma4a 的 semicausal padding 计算
        const n_with_left: u32 = @as(u32, @intCast(chunk_len)) + pad_left;
        const pt_frames: u32 = if (n_with_left >= params.frame_length + 1)
            @as(u32, @intCast((n_with_left - (params.frame_length + 1)) / hop)) + 1
        else
            0;
        const n_padded_needed: u32 = (pt_frames -| 1) * hop + frame_size;
        const total_pad: u32 = @max(n_padded_needed -| @as(u32, @intCast(chunk_len)), pad_left);
        const n_samples_padded: u32 = total_pad + @as(u32, @intCast(chunk_len));
        const n_frames: u32 = if (n_samples_padded >= frame_size)
            @as(u32, @intCast((n_samples_padded - frame_size) / hop)) + 1
        else
            0;

        // 匹配 llama.cpp: 裁剪到 PyTorch 帧数
        const actual_frames = @min(n_frames, pt_frames);
        total_frames += actual_frames;

        offset += chunk_samples;
    }

    // 分配 Mel 输出缓冲区 [n_mel_bins, n_frames] (mel-major 布局，匹配 llama.cpp)
    // llama.cpp: out.data[(size_t)j * out.n_len + i] = sum;  (j=mel_bin, i=frame_idx)
    const mel_out = try allocator.alloc(f32, @as(usize, params.n_mel_bins) * @as(usize, total_frames));
    @memset(mel_out, 0.0);

    // 逐块处理
    var mel_frame_offset: u32 = 0;
    offset = 0;

    while (offset < samples.len) {
        const chunk_len = @min(chunk_samples, samples.len - offset);
        const chunk_ptr = samples[offset..][0..chunk_len];

        // 匹配 llama.cpp gemma4a 的 semicausal padding
        const n_with_left: u32 = @as(u32, @intCast(chunk_len)) + pad_left;
        const pt_frames: u32 = if (n_with_left >= params.frame_length + 1)
            @as(u32, @intCast((n_with_left - (params.frame_length + 1)) / hop)) + 1
        else
            0;
        const n_padded_needed: u32 = (pt_frames -| 1) * hop + frame_size;
        const total_pad: u32 = @max(n_padded_needed -| @as(u32, @intCast(chunk_len)), pad_left);
        const n_samples_padded: u32 = total_pad + @as(u32, @intCast(chunk_len));
        const n_frames: u32 = if (n_samples_padded >= frame_size)
            @as(u32, @intCast((n_samples_padded - frame_size) / hop)) + 1
        else
            0;
        const actual_frames = @min(n_frames, pt_frames);

        // 构建填充缓冲区（匹配 llama.cpp: padded_samples = [pad_left zeros] + chunk_data + [right zeros]）
        const padded = try tmp_alloc.alloc(f32, n_samples_padded);
        @memset(padded, 0.0);
        @memcpy(padded[pad_left..][0..chunk_len], chunk_ptr);

        // 逐帧处理（匹配 llama.cpp log_mel_spectrogram_worker_thread）
        const fft_in = try tmp_alloc.alloc(f32, frame_size);
        const spectrum = try tmp_alloc.alloc(f32, n_freqs);

        for (0..actual_frames) |fi| {
            const start: usize = fi * @as(usize, hop);

            // 应用 Hann 窗口（匹配 llama.cpp: valid_len = min(frame_size, max(0, n_samples - offset))）
            const valid_len = @min(frame_size, if (n_samples_padded > start) n_samples_padded - start else 0);
            @memset(fft_in, 0.0);
            // 匹配 llama.cpp: for j < valid_len: fft_in[j] = hann[j] * samples[offset + j]
            // hann[j] for j >= frame_length is 0 (zero-padded to FFT size)
            for (0..valid_len) |j| {
                fft_in[j] = hann_window[j] * padded[start + j];
            }

            // FFT → 幅度谱 |X| (use_magnitude=true)
            fft_engine.powerSpectrum(fft_in[0..frame_size], spectrum);

            // Mel 滤波 + 自然对数
            // 匹配 llama.cpp mel-major 布局: out.data[mel_bin * n_len + frame_idx]
            for (0..@as(usize, params.n_mel_bins)) |m| {
                const mel_idx = m * @as(usize, total_frames) + @as(usize, mel_frame_offset) + fi;
                mel_out[mel_idx] = @log(@max(
                    mel.applyFilterbankSingle(spectrum, filterbank, @as(u32, @intCast(m)), n_freqs),
                    params.log_offset,
                ));
            }
        }

        mel_frame_offset += actual_frames;
        offset += chunk_samples;
    }

    log.info("Audio pipeline: {d} frames x {d} mel bins, sr={d}Hz (gemma4a exact match)", .{
        total_frames, params.n_mel_bins, params.sample_rate,
    });

    helper.mtmdDebugSaveData(io, "zllama_log_audio_mel.json", "log_audio_mel", mel_out) catch |err| {
        log.debug("Failed to save mel debug data: {}", .{err});
    };

    return .{
        .data = mel_out,
        .n_mel_bins = params.n_mel_bins,
        .n_frames = total_frames,
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
