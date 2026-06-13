//! mtmd helper functions
//!
//! Utility functions for evaluating chunks, loading media from files,
//! getting decoder positions for M-RoPE, and video input.
//!
//! Reference: deps/llama.cpp/tools/mtmd/mtmd-helper.h

const std = @import("std");
const ggml = @import("ggml");
const mtmd = @import("mtmd");
const preprocess = @import("preprocess");
const stb_image = @import("stb_image");

const log = std.log.scoped(.mtmd_helper);

// ============================================================================
// Chunk evaluation helpers
// ============================================================================

/// Evaluate all chunks in sequence:
/// - Text chunks: call user-provided decode callback
/// - Image/audio chunks: encode with mtmd, then decode
///
/// Caller provides a `decodeFn` that takes tokens and returns new n_past.
/// Returns the updated n_past, or an error.
pub fn evalChunks(
    ctx: *mtmd.MtmdContext,
    ggctx: *ggml.Context,
    graph: *ggml.CGraph,
    chunks: mtmd.InputChunks,
    n_past: u32,
    decodeFn: anytype, // fn(tokens: []const i32, n_past: u32) !u32
) !u32 {
    _ = ggctx;
    _ = graph;
    var cur_n_past = n_past;

    for (chunks.entries.items) |*chunk| {
        switch (chunk.chunk_type) {
            .text => {
                if (chunk.tokens_text) |tokens| {
                    if (tokens.len > 0) {
                        cur_n_past = try decodeFn(tokens, cur_n_past);
                    }
                }
            },
            .image => {
                if (ctx.mm_manager.vision_encoder) |*enc| {
                    _ = enc;
                    log.warn("Image chunk evaluation not fully implemented", .{});
                }
            },
            .audio => {
                if (ctx.mm_manager.audio_encoder) |*enc| {
                    _ = enc;
                    log.warn("Audio chunk evaluation not fully implemented", .{});
                }
            },
        }
    }

    return cur_n_past;
}

/// Get decoder positions for all tokens in an image chunk.
/// Used by M-RoPE models to determine (t, x, y) positions.
pub fn imageGetDecoderPos(
    image: mtmd.ImageTokens,
    pos_0: u32,
    out_pos: []mtmd.DecoderPos,
) void {
    const n_tokens = image.nTokens();
    std.debug.assert(out_pos.len >= n_tokens);

    switch (image.pos) {
        .normal => {
            for (0..n_tokens) |i| {
                out_pos[i] = .{ .t = pos_0 + @as(u32, @intCast(i)), .x = 0, .y = 0 };
            }
        },
        .mrope => {
            for (0..@as(usize, @intCast(n_tokens))) |i| {
                const x = @as(u32, @intCast(i)) % image.nx;
                const y = @as(u32, @intCast(i)) / image.nx;
                out_pos[i] = .{ .t = pos_0, .x = x, .y = y };
            }
        },
        .hunyuanvl => {
            var idx: u32 = 0;
            out_pos[idx] = .{ .t = pos_0, .x = 0, .y = 0 };
            idx += 1;
            for (0..image.ny) |row| {
                for (0..image.nx) |col| {
                    out_pos[idx] = .{ .t = pos_0, .x = @as(u32, @intCast(col)), .y = @as(u32, @intCast(row)) };
                    idx += 1;
                }
                out_pos[idx] = .{ .t = pos_0, .x = image.nx, .y = @as(u32, @intCast(row)) };
                idx += 1;
            }
            out_pos[idx] = .{ .t = pos_0, .x = 0, .y = image.ny };
        },
    }
}

// ============================================================================
// File loading helpers
// ============================================================================

pub const BitmapWrapper = struct {
    bitmap: mtmd.Bitmap,
    allocator: std.mem.Allocator,
    video_ctx: ?*anyopaque = null,

    pub fn deinit(self: *BitmapWrapper) void {
        self.bitmap.deinit();
    }
};

/// Load a bitmap from a file (auto-detects image/audio).
pub fn bitmapInitFromFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    filepath: []const u8,
    placeholder: bool,
) !BitmapWrapper {
    const cwd = std.Io.Dir.cwd();

    const file = cwd.openFile(io, filepath, .{ .mode = .read_only }) catch |err| {
        log.err("Failed to open file '{s}': {}", .{ filepath, err });
        return err;
    };

    const stat = file.stat(io) catch |err| {
        log.err("Failed to stat file '{s}': {}", .{ filepath, err });
        return err;
    };

    const raw = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(raw);

    const total_read = try file.readPositionalAll(io, raw, 0);
    if (total_read != raw.len) {
        allocator.free(raw);
        return error.FileReadError;
    }
    file.close(io);

    return bitmapInitFromBuf(allocator, raw, placeholder);
}

/// Load a bitmap from a buffer (auto-detects image/audio).
pub fn bitmapInitFromBuf(
    allocator: std.mem.Allocator,
    buf: []const u8,
    placeholder: bool,
) !BitmapWrapper {
    if (placeholder) {
        return BitmapWrapper{
            .bitmap = mtmd.Bitmap.initPlaceholderImage(224, 224),
            .allocator = allocator,
        };
    }

    if (buf.len >= 4) {
        const magic = buf[0..4];

        if (magic[0] == 0xFF and magic[1] == 0xD8 and magic[2] == 0xFF) {
            return loadImageFromBuf(allocator, buf);
        }

        if (magic[0] == 0x89 and std.mem.eql(u8, magic[1..4], "PNG")) {
            return loadImageFromBuf(allocator, buf);
        }

        if (magic[0] == 'G' and magic[1] == 'I' and magic[2] == 'F') {
            return loadImageFromBuf(allocator, buf);
        }

        if (magic[0] == 'B' and magic[1] == 'M') {
            return loadImageFromBuf(allocator, buf);
        }

        if (magic[0] == 'R' and magic[1] == 'I' and magic[2] == 'F' and magic[3] == 'F') {
            return loadAudioFromBuf(allocator, buf);
        }
    }

    return loadImageFromBuf(allocator, buf);
}

/// Load image from buffer using stb_image
fn loadImageFromBuf(allocator: std.mem.Allocator, buf: []const u8) !BitmapWrapper {
    var width: c_int = 0;
    var height: c_int = 0;
    var channels: c_int = 0;

    const data = stb_image.stbi_load_from_memory(
        buf.ptr,
        @intCast(buf.len),
        &width,
        &height,
        &channels,
        3,
    );

    if (data == null) {
        log.err("stb_image failed to decode: {s}", .{stb_image.stbi_failure_reason()});
        return error.ImageDecodeFailed;
    }

    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    const size: usize = w * h * 3;

    const owned = try allocator.alloc(u8, size);
    @memcpy(owned, data[0..size]);

    stb_image.stbi_image_free(data);

    return BitmapWrapper{
        .bitmap = .{
            .nx = w,
            .ny = h,
            .data = owned,
            .allocator = allocator,
        },
        .allocator = allocator,
    };
}

/// Load audio from buffer (WAV 16-bit PCM)
fn loadAudioFromBuf(allocator: std.mem.Allocator, buf: []const u8) !BitmapWrapper {
    if (buf.len < 44) return error.InvalidWavFormat;

    if (!std.mem.eql(u8, buf[0..4], "RIFF")) return error.InvalidWavFormat;
    if (!std.mem.eql(u8, buf[8..12], "WAVE")) return error.InvalidWavFormat;
    if (!std.mem.eql(u8, buf[12..16], "fmt ")) return error.InvalidWavFormat;

    const audio_format = std.mem.readInt(u16, buf[20..22], .little);
    const num_channels = std.mem.readInt(u16, buf[22..24], .little);
    _ = std.mem.readInt(u32, buf[24..28], .little); // sample_rate
    const bits_per_sample = std.mem.readInt(u16, buf[34..36], .little);

    if (audio_format != 1) return error.UnsupportedWavFormat;
    if (bits_per_sample != 16) return error.UnsupportedBitDepth;

    var offset: usize = 36;
    while (offset + 8 <= buf.len) {
        if (std.mem.eql(u8, buf[offset..][0..4], "data")) {
            const data_size = std.mem.readInt(u32, buf[offset + 4 ..][0..4], .little);
            const data_start = offset + 8;
            const data_end = @min(data_start + @as(usize, @intCast(data_size)), buf.len);

            const sample_count = (data_end - data_start) / 2;
            if (num_channels > 1) {
                const mono_count = sample_count / @as(usize, @intCast(num_channels));
                const owned = try allocator.alloc(u8, mono_count * @sizeOf(f32));
                const dst = @as([*]f32, @ptrCast(@alignCast(owned.ptr)))[0..mono_count];
                const src = @as([*]const i16, @ptrCast(@alignCast(buf[data_start..].ptr)))[0..sample_count];

                for (0..mono_count) |i| {
                    var sum: f32 = 0;
                    for (0..@as(usize, @intCast(num_channels))) |c| {
                        sum += @as(f32, @floatFromInt(src[i * @as(usize, @intCast(num_channels)) + c]));
                    }
                    dst[i] = (sum / @as(f32, @floatFromInt(num_channels))) / 32768.0;
                }

                return BitmapWrapper{
                    .bitmap = mtmd.Bitmap{
                        .nx = @intCast(mono_count),
                        .ny = 1,
                        .is_audio = true,
                        .data = owned,
                        .allocator = allocator,
                    },
                    .allocator = allocator,
                };
            } else {
                const owned = try allocator.alloc(u8, sample_count * @sizeOf(f32));
                const dst = @as([*]f32, @ptrCast(@alignCast(owned.ptr)))[0..sample_count];
                const src = @as([*]const i16, @ptrCast(@alignCast(buf[data_start..].ptr)))[0..sample_count];

                for (0..sample_count) |i| {
                    dst[i] = @as(f32, @floatFromInt(src[i])) / 32768.0;
                }

                return BitmapWrapper{
                    .bitmap = mtmd.Bitmap{
                        .nx = @intCast(sample_count),
                        .ny = 1,
                        .is_audio = true,
                        .data = owned,
                        .allocator = allocator,
                    },
                    .allocator = allocator,
                };
            }
        }
        offset += 8 + std.mem.readInt(u32, buf[offset + 4 ..][0..4], .little);
    }

    return error.DataChunkNotFound;
}
