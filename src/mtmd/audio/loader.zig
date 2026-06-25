//! WAV 文件加载与重采样
//!
//! 提供 WAV 文件解析和 PCM 数据加载功能。
//! 支持 16-bit PCM, mono/stereo，自动转换为 F32 单声道。
//!
//! 参考: llama.cpp mtmd-audio.cpp

const std = @import("std");
const types = @import("types.zig");

const log = std.log.scoped(.audio_loader);

// ============================================================================
// 公开 API
// ============================================================================

/// 从 WAV 文件加载 PCM 数据
/// 支持 16-bit PCM, mono/stereo
/// 返回 F32 单声道样本（stereo 取左右平均）
pub fn loadWav(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
) !struct { samples: []f32, info: types.WavInfo } {
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

    log.info("Loaded WAV: {d} Hz, {d} ch, {d} bits, {d} samples ({d:.1}s)", .{
        sample_rate,                                                                 num_channels, bits_per_sample, num_samples,
        @as(f64, @floatFromInt(num_samples)) / @as(f64, @floatFromInt(sample_rate)),
    });

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
// 重采样
// ============================================================================

/// 将音频重采样到目标采样率（线性插值）
/// 如果当前采样率与目标相同，返回原始数据（不分配新内存）
pub fn resample(
    allocator: std.mem.Allocator,
    samples: []const f32,
    src_rate: u32,
    dst_rate: u32,
) ![]f32 {
    if (src_rate == dst_rate) return try allocator.dupe(f32, samples);

    const ratio = @as(f64, @floatFromInt(dst_rate)) / @as(f64, @floatFromInt(src_rate));
    const dst_len = @as(usize, @intFromFloat(@as(f64, @floatFromInt(samples.len)) * ratio));
    const result = try allocator.alloc(f32, dst_len);

    for (0..dst_len) |i| {
        const src_pos = @as(f64, @floatFromInt(i)) / ratio;
        const src_idx = @as(usize, @intFromFloat(@floor(src_pos)));
        const frac = src_pos - @floor(src_pos);

        if (src_idx + 1 < samples.len) {
            result[i] = @as(f32, @floatCast(
                @as(f64, @floatCast(samples[src_idx])) * (1.0 - frac) +
                    @as(f64, @floatCast(samples[src_idx + 1])) * frac,
            ));
        } else {
            result[i] = samples[@min(src_idx, samples.len - 1)];
        }
    }

    log.info("Resampled audio: {d} Hz -> {d} Hz, {d} -> {d} samples", .{ src_rate, dst_rate, samples.len, dst_len });
    return result;
}
