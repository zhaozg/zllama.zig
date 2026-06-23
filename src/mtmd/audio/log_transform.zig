//! 对数变换
//!
//! 对 Mel 频谱进行自然对数压缩。
//! 匹配 llama.cpp gemma4a 的 use_natural_log=true 行为。
//!
//! 参考: llama.cpp mtmd-audio.cpp

const std = @import("std");
const math = std.math;

const log = std.log.scoped(.audio_log);

// ============================================================================
// 公开 API
// ============================================================================

/// 对 Mel 频谱进行自然对数压缩
/// mel_spectrum: [n_mel_bins] 输入 Mel 能量值
/// log_offset: 防止 log(0) 的小常数（默认 0.001）
/// 返回 [n_mel_bins] log-mel 值
pub fn applyLogTransform(
    mel_spectrum: []const f32,
    log_offset: f32,
    output: []f32,
) void {
    for (0..mel_spectrum.len) |i| {
        output[i] = @log(@max(mel_spectrum[i], log_offset));
    }
}

/// 对 Mel 频谱进行自然对数压缩（原地操作）
pub fn applyLogTransformInPlace(
    mel_spectrum: []f32,
    log_offset: f32,
) void {
    for (mel_spectrum) |*val| {
        val.* = @log(@max(val.*, log_offset));
    }
}

/// 计算 log-mel 频谱的统计信息（用于调试）
pub fn logMelStats(data: []const f32) struct { min: f32, max: f32, mean: f32 } {
    if (data.len == 0) return .{ .min = 0, .max = 0, .mean = 0 };

    var min: f32 = data[0];
    var max: f32 = data[0];
    var sum: f64 = 0;

    for (data) |v| {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += @as(f64, @floatCast(v));
    }

    return .{
        .min = min,
        .max = max,
        .mean = @as(f32, @floatCast(sum / @as(f64, @floatFromInt(data.len)))),
    };
}
