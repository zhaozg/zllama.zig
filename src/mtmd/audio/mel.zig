//! Mel 滤波器组
//!
//! 预计算 Mel 滤波器组权重矩阵，将频谱幅度映射到 Mel 刻度。
//! 使用 HTK Mel 尺度，slaney_area_norm=false（与 llama.cpp gemma4a 一致）。
//!
//! 参考: llama.cpp mtmd-audio.cpp (mtmd_audio_cache::fill_mel_filterbank_matrix)

const std = @import("std");
const math = std.math;

const log = std.log.scoped(.audio_mel);

// ============================================================================
// 公开 API
// ============================================================================

/// 预计算 Mel 滤波器组权重矩阵 [n_mel_bins, n_fft/2+1]
/// 使用 HTK Mel 尺度，slaney_area_norm=false（与 llama.cpp gemma4a 一致）
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
        fn call(hz: f64) f64 {
            return 2595.0 * @log10(1.0 + hz / 700.0);
        }
    }.call;

    const melToHz = struct {
        fn call(mel: f64) f64 {
            return 700.0 * (math.pow(f64, 10.0, mel / 2595.0) - 1.0);
        }
    }.call;

    const mel_min = hzToMel(@as(f64, f_min));
    const mel_max = hzToMel(@as(f64, f_max));

    // 在 Mel 尺度上均匀分布 n_mel_bins + 2 个点
    const mel_step = (mel_max - mel_min) / @as(f64, @floatFromInt(n_mel_bins + 1));

    // 每个滤波器的中心频率（Hz）
    var center_hz = try allocator.alloc(f64, @as(usize, n_mel_bins) + 2);
    defer allocator.free(center_hz);

    for (0..@as(usize, n_mel_bins) + 2) |i| {
        const mel_pt = mel_min + @as(f64, @floatFromInt(i)) * mel_step;
        center_hz[i] = melToHz(mel_pt);
    }

    // FFT bin 对应的频率
    const bin_hz_step = @as(f64, @floatFromInt(sample_rate)) / @as(f64, @floatFromInt(n_fft));

    // 对每个 mel bin 构建三角形滤波器
    // 匹配 llama.cpp: slaney_area_norm=false, scale=1.0
    for (0..@as(usize, n_mel_bins)) |m| {
        const f_left = center_hz[m];
        const f_center = center_hz[m + 1];
        const f_right = center_hz[m + 2];
        const denom_l = @max(1e-30, f_center - f_left);
        const denom_r = @max(1e-30, f_right - f_center);
        // slaney_area_norm=false → enorm=1.0
        const row = filterbank[m * n_freqs .. (m + 1) * n_freqs];

        for (0..n_freqs) |k| {
            const f = @as(f64, @floatFromInt(k)) * bin_hz_step;
            var w: f64 = 0.0;
            if (f >= f_left and f <= f_center) {
                w = (f - f_left) / denom_l;
            } else if (f > f_center and f <= f_right) {
                w = (f_right - f) / denom_r;
            }
            row[k] = @as(f32, @floatCast(w));
        }
    }

    log.debug("Mel filterbank: {d} bins x {d} freqs, range {d:.0}-{d:.0} Hz (HTK, no area norm)", .{
        n_mel_bins, n_freqs, f_min, f_max,
    });

    return filterbank;
}

/// 将频谱幅度通过 Mel 滤波器组，得到 Mel 能量值
/// spectrum: [n_freqs] 频谱幅度（|X|）
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
        mel_out[m] = applyFilterbankSingle(spectrum, filterbank, m, n_freqs);
    }
}

/// 计算单个 Mel bin 的能量值
/// 匹配 llama.cpp: sum += fft_out[k] * filters.data[(size_t)j * n_fft_bins + k]
pub fn applyFilterbankSingle(
    spectrum: []const f32,
    filterbank: []const f32,
    mel_bin: u32,
    n_freqs: u32,
) f32 {
    const row = filterbank[@as(usize, mel_bin) * @as(usize, n_freqs) ..][0..@as(usize, n_freqs)];
    var sum: f64 = 0.0;
    for (0..@as(usize, n_freqs)) |k| {
        sum += @as(f64, @floatCast(row[k])) * @as(f64, @floatCast(spectrum[k]));
    }
    return @as(f32, @floatCast(sum));
}
