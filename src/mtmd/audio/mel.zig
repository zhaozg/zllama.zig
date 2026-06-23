//! Mel 滤波器组
//!
//! 预计算 Mel 滤波器组权重矩阵，将功率谱映射到 Mel 刻度。
//! 使用 HTK Mel 尺度（与 llama.cpp gemma4a 一致）。
//!
//! 参考: llama.cpp mtmd-audio.cpp

const std = @import("std");
const math = std.math;

const log = std.log.scoped(.audio_mel);

// ============================================================================
// 公开 API
// ============================================================================

/// 预计算 Mel 滤波器组权重矩阵 [n_mel_bins, n_fft/2+1]
/// 使用 HTK Mel 尺度（与 llama.cpp gemma4a 一致）
pub fn computeFilterbank(
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

    log.debug("Mel filterbank: {d} bins x {d} freqs, range {d:.0}-{d:.0} Hz", .{
        n_mel_bins, n_freqs, f_min, f_max,
    });

    return filterbank;
}

/// 将功率谱通过 Mel 滤波器组，得到 Mel 频谱
/// spectrum: [n_freqs] 功率谱
/// filterbank: [n_mel_bins * n_freqs] 滤波器组权重
/// 返回 [n_mel_bins] Mel 能量值
pub fn applyFilterbank(
    spectrum: []const f32,
    filterbank: []const f32,
    n_mel_bins: u32,
    n_freqs: u32,
    mel_out: []f32,
) void {
    for (0..@as(usize, n_mel_bins)) |m| {
        const row = filterbank[m * @as(usize, n_freqs) .. (m + 1) * @as(usize, n_freqs)];
        var mel_val: f32 = 0.0;
        for (0..@as(usize, n_freqs)) |k| {
            mel_val += row[k] * spectrum[k];
        }
        mel_out[m] = mel_val;
    }
}
