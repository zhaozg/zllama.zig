//! 分帧与加窗
//!
//! 将 PCM 音频数据分割为重叠帧，并对每帧应用 Hann 窗口。
//! 匹配 llama.cpp gemma4a 的帧处理逻辑（semicausal padding + Hann window zero-padded to FFT size）。
//!
//! 参考: llama.cpp mtmd-audio.cpp (mtmd_audio_preprocessor_gemma4a::preprocess + log_mel_spectrogram_worker_thread)

const std = @import("std");
const math = std.math;
const config_mod = @import("config.zig");

const log = std.log.scoped(.audio_framing);

// ============================================================================
// 公开 API
// ============================================================================

/// 分帧参数
pub const FramingParams = struct {
    /// 窗口长度（帧大小）
    frame_length: u32 = config_mod.DEFAULT_FRAME_LENGTH,
    /// 帧移
    hop_length: u32 = config_mod.DEFAULT_HOP_LENGTH,
    /// FFT 点数（用于零填充）
    n_fft: u32 = config_mod.DEFAULT_N_FFT,
};

/// 预计算 Hann 窗口（零填充到 FFT 大小）
/// 匹配 llama.cpp gemma4a initialize():
///   cache.hann_window.assign(hparams.audio_n_fft, 0.0f);
///   for (i < audio_window_len) hann_window[i] = 0.5 - 0.5*cos(2*PI*i/window_len);
pub fn computeHannWindow(frame_length: u32) [512]f32 {
    var hann_window: [512]f32 = undefined;
    @memset(&hann_window, 0.0);
    const win_len: f32 = @floatFromInt(frame_length);
    for (0..frame_length) |i| {
        hann_window[i] = 0.5 - 0.5 * @cos(2.0 * math.pi * @as(f32, @floatFromInt(i)) / win_len);
    }
    return hann_window;
}

/// 计算帧数和填充量
/// 匹配 llama.cpp gemma4a preprocess():
///   pt_frames = (n_with_left - (window_len + 1)) / hop + 1
///   n_padded_needed = (pt_frames - 1) * hop + fft_size
///   total_pad = max(n_padded_needed - chunk_len, pad_left)
///   n_samples = total_pad + chunk_len
///   out.n_len = (n_samples - frame_size) / frame_step + 1
pub fn computeFrameCount(
    audio_len: usize,
    params: FramingParams,
) struct {
    n_frames: u32,
    total_pad: u32,
    n_samples_padded: u32,
} {
    const pad_left: u32 = params.frame_length / 2;
    const frame_size: u32 = params.n_fft;

    const n_with_left: u32 = @as(u32, @intCast(audio_len)) + pad_left;
    const pt_frames: u32 = if (n_with_left >= params.frame_length + 1)
        @as(u32, @intCast((n_with_left - (params.frame_length + 1)) / params.hop_length)) + 1
    else
        0;
    const n_padded_needed: u32 = (pt_frames -| 1) * params.hop_length + frame_size;
    const total_pad: u32 = @max(n_padded_needed -| @as(u32, @intCast(audio_len)), pad_left);
    const n_samples_padded: u32 = total_pad + @as(u32, @intCast(audio_len));
    const n_frames: u32 = if (n_samples_padded >= frame_size)
        @as(u32, @intCast((n_samples_padded - frame_size) / params.hop_length)) + 1
    else
        0;

    return .{ .n_frames = n_frames, .total_pad = total_pad, .n_samples_padded = n_samples_padded };
}

/// 逐帧处理回调接口
/// 对每帧应用 Hann 窗口，通过回调逐帧处理，避免一次性分配所有帧数据。
///
/// 处理步骤（匹配 llama.cpp gemma4a）：
/// 1. 半因果左填充（pad_left = frame_length/2），右填充到匹配 PyTorch 帧数
/// 2. 提取重叠帧
/// 3. 应用 Hann 窗口（零填充到 FFT 大小）
/// 4. 通过回调传递每帧数据
///
/// @param allocator 临时分配器
/// @param audio_data PCM F32 音频样本
/// @param params 分帧参数
/// @param hann_window 预计算的 Hann 窗口（零填充到 FFT 大小）
/// @param frame_callback 每帧回调：fn(frame_idx: u32, windowed_frame: []const f32, ctx: *ContextT) anyerror!void
/// @param ctx 传递给回调的上下文指针
pub fn frameAudioWithCallback(
    allocator: std.mem.Allocator,
    audio_data: []const f32,
    params: FramingParams,
    hann_window: []const f32,
    frame_callback: anytype,
    ctx: anytype,
) !u32 {
    if (audio_data.len == 0) return error.EmptyAudioData;

    const frame_count = computeFrameCount(audio_data.len, params);
    if (frame_count.n_frames == 0) return error.AudioTooShort;

    const frame_size: u32 = params.n_fft;
    const pad_left: u32 = params.frame_length / 2;
    const n_padded = audio_data.len + frame_count.total_pad;

    // 构建填充缓冲区（匹配 llama.cpp gemma4a 的 semicausal padding）
    // padded = [pad_left zeros] + audio_data + [right_padding zeros]
    var padded = try allocator.alloc(f32, n_padded);
    defer allocator.free(padded);
    @memset(padded, 0.0);
    @memcpy(padded[pad_left..][0..audio_data.len], audio_data);

    // 单帧工作缓冲区（复用避免重复分配）
    var frame_buf = try allocator.alloc(f32, frame_size);
    defer allocator.free(frame_buf);

    // 逐帧处理（匹配 llama.cpp log_mel_spectrogram_worker_thread）
    for (0..frame_count.n_frames) |fi| {
        const start: usize = fi * @as(usize, params.hop_length);

        @memset(frame_buf, 0.0);
        const valid_len = @min(frame_size, if (n_padded > start) n_padded - start else 0);
        const copy_actual = @min(valid_len, params.frame_length);
        for (0..copy_actual) |j| {
            frame_buf[j] = hann_window[j] * padded[start + j];
        }

        try @call(.auto, frame_callback, .{ @as(u32, @intCast(fi)), frame_buf[0..frame_size], ctx });
    }

    log.debug("Framing callback: {d} frames x {d} samples, pad={d}", .{
        frame_count.n_frames, frame_size, frame_count.total_pad,
    });

    return frame_count.n_frames;
}
