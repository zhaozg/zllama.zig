//! mtmd helper functions
//!
//! Utility functions for evaluating chunks, loading media from files,
//! getting decoder positions for M-RoPE.
//!
//! ## 设计说明
//!
//! 本模块位于 L5 层，不依赖 L6 层（引擎层）。
//! `evalChunks` 函数通过 `computeGraphFn` 回调参数接收 ggml 计算图执行能力，
//! 由调用者（L6/L7 层）注入 `computeGraph` 实现。
//! 这样设计消除了对 engine_common 的直接依赖，保持了 DAG 约束的完整性。
//!
//! Reference: deps/llama.cpp/tools/mtmd/mtmd-helper.h

const std = @import("std");
const ggml = @import("ggml");
const mtmd = @import("mm");
const preprocess = @import("preprocess");
const stb_image = @import("stb_image");

const log = std.log.scoped(.mtmd_helper);

// ============================================================================
// Chunk evaluation
// ============================================================================

/// Evaluate all chunks in sequence.
///
/// - Text: call decodeTextFn(tokens, n_past) → new n_past
/// - Image/audio: encode via encoder → extract embeddings →
///   call decodeMediaFn(embeddings, n_embd, n_tokens, n_past, non_causal) → new n_past
///
/// `computeGraphFn` 是用于执行 ggml 计算图的回调函数，由调用者注入。
/// 签名: fn (cgraph: *ggml.CGraph, n_threads: i32) anyerror!void
pub fn evalChunks(
    ctx: *mtmd.MtmdContext,
    io: std.Io,
    allocator: std.mem.Allocator,
    chunks: mtmd.InputChunks,
    n_past: u32,
    n_threads: i32,
    decodeTextFn: anytype,
    decodeMediaFn: anytype,
    computeGraphFn: anytype,
) !u32 {
    var cur_n_past = n_past;
    var idx: usize = 0;

    log.debug("evalChunks: starting with {d} chunks, n_past={d}", .{ chunks.entries.items.len, n_past });

    while (idx < chunks.entries.items.len) {
        const chunk = &chunks.entries.items[idx];
        log.debug("evalChunks: processing chunk {d}/{d}, type={s}", .{ idx, chunks.entries.items.len, @tagName(chunk.chunk_type) });
        switch (chunk.chunk_type) {
            .text => {
                if (chunk.tokens_text) |tokens| {
                    log.debug("evalChunks: text chunk with {d} tokens", .{tokens.len});
                    if (tokens.len > 0) cur_n_past = try decodeTextFn(tokens, cur_n_past);
                }
                idx += 1;
            },
            .image => {
                log.debug("evalChunks: image chunk at idx={d}", .{idx});
                if (ctx.mm_manager.vision_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.VisionEncoderNotAvailable;

                    // Count consecutive image chunks for batching
                    var batch_end = idx;
                    while (batch_end < chunks.entries.items.len and
                        chunks.entries.items[batch_end].chunk_type == .image) : (batch_end += 1)
                    {}
                    const batch_size: u32 = @intCast(batch_end - idx);
                    const can_batch = enc.supportBatch() and batch_size > 1;

                    log.debug("evalChunks: batch_size={d}, can_batch={}", .{ batch_size, can_batch });

                    if (can_batch) {
                        log.debug("Batching {d} consecutive image chunks", .{batch_size});
                    }

                    // Encode each image in the batch group
                    var bi: usize = idx;
                    while (bi < batch_end) : (bi += 1) {
                        log.debug("evalChunks: encoding image {d}/{d}", .{ bi - idx, batch_size });
                        const img = &(chunks.entries.items[bi].tokens_image orelse return error.MissingImageTokens);
                        const raw_pixels = img.getRawPixels() orelse return error.NoImageData;

                        log.debug("evalChunks: image size={d}x{d}, raw_pixels.len={d}", .{ img.nx, img.ny, raw_pixels.len });

                        const compute_ctx = try ggml.Context.initNoAlloc(4 * 1024 * 1024 * 1024);
                        defer compute_ctx.deinit();
                        const cgraph = try ggml.CGraph.initReserved(compute_ctx, 4096);

                        log.debug("evalChunks: created compute_ctx, calling enc.encode...", .{});
                        const out_tensor = try enc.encode(io, compute_ctx, cgraph, raw_pixels, img.nx, img.ny, n_threads);

                        log.debug("evalChunks: encode returned, out_tensor ne={any}", .{out_tensor.ne()});

                        // Compute the vision graph via injected callback
                        log.debug("evalChunks: computing vision graph...", .{});
                        compute_ctx.setNoAlloc(false);
                        try computeGraphFn(cgraph, n_threads);
                        log.debug("evalChunks: vision graph computed successfully", .{});

                        const n_embd_out: usize = @intCast(out_tensor.ne()[0]);
                        const n_tokens_out: usize = @intCast(out_tensor.ne()[1]);
                        log.debug("evalChunks: output embd={d}x{d}", .{ n_embd_out, n_tokens_out });
                        const embd_size = n_embd_out * n_tokens_out;
                        const embd_f32 = try allocator.alloc(f32, embd_size);
                        defer allocator.free(embd_f32);
                        {
                            const out_data = try out_tensor.dataGet(f32, allocator);
                            defer allocator.free(out_data);
                            @memcpy(embd_f32, out_data[0..embd_size]);
                        }

                        if (ctx.output_embd) |old| allocator.free(old);
                        ctx.output_embd = try allocator.dupe(f32, embd_f32);

                        const non_causal = ctx.decodeUseNonCausal(&chunks.entries.items[bi]);
                        log.debug("evalChunks: calling decodeMediaFn with n_past={d}", .{cur_n_past});
                        cur_n_past = try decodeMediaFn(embd_f32, @intCast(n_embd_out), @intCast(n_tokens_out), cur_n_past, non_causal);
                    }

                    idx = batch_end;
                    log.debug("evalChunks: batch done, idx now={d}", .{idx});
                } else return error.VisionEncoderNotAvailable;
            },
            .audio => {
                log.debug("evalChunks: audio chunk at idx={d}", .{idx});
                if (ctx.mm_manager.audio_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.AudioEncoderNotAvailable;

                    // Get or compute Mel spectrogram data.
                    var mel_f32: ?[]f32 = null;
                    var mel_bins: u32 = 0;
                    var mel_frames: u32 = 0;

                    if (chunk.mel_data) |mel| {
                        mel_f32 = mel;
                        mel_bins = chunk.mel_bins;
                        mel_frames = chunk.mel_frames;
                    } else if (chunk.audio_data) |raw_bytes| {
                        // Compute Mel from raw PCM f32 samples.
                        const f32_samples = @as([*]const f32, @ptrCast(@alignCast(raw_bytes.ptr)))[0 .. raw_bytes.len / @sizeOf(f32)];
                        const sr: u32 = if (ctx.caps.audio_sample_rate > 0) @intCast(ctx.caps.audio_sample_rate) else 16000;
                        const nm: u32 = if (ctx.mm_manager.audio_encoder) |*e| e.params.n_mel_bins else 128;
                        const audio_cfg = mtmd.audio_mod.config;
                        const pp = audio_cfg.AudioPreprocessParams.fromAudioEncoder(nm);
                        const processed = try mtmd.audio_mod.mel_spectrogram.processPcmSamples(io, allocator, f32_samples, sr, pp);
                        mel_f32 = processed.data;
                        mel_bins = processed.n_mel_bins;
                        mel_frames = processed.n_frames;
                        defer allocator.free(processed.data);
                    } else return error.NoAudioData;

                    const compute_ctx = try ggml.Context.initNoAlloc(4 * 1024 * 1024 * 1024);
                    defer compute_ctx.deinit();
                    const cgraph = try ggml.CGraph.initReserved(compute_ctx, 4096);

                    const out_tensor = try enc.encodeRaw(io, compute_ctx, cgraph, mel_f32.?, mel_bins, mel_frames, n_threads);

                    // Compute the audio graph via injected callback
                    compute_ctx.setNoAlloc(false);
                    try computeGraphFn(cgraph, n_threads);

                    const n_embd_out: usize = @intCast(out_tensor.ne()[0]);
                    const n_tokens_out: usize = @intCast(out_tensor.ne()[1]);
                    const embd_size = n_embd_out * n_tokens_out;
                    const embd_f32 = try allocator.alloc(f32, embd_size);
                    defer allocator.free(embd_f32);
                    {
                        const out_data = try out_tensor.dataGet(f32, allocator);
                        defer allocator.free(out_data);
                        @memcpy(embd_f32, out_data[0..embd_size]);
                    }

                    if (ctx.output_embd) |old| allocator.free(old);
                    ctx.output_embd = try allocator.dupe(f32, embd_f32);

                    const non_causal = ctx.decodeUseNonCausal(chunk);
                    cur_n_past = try decodeMediaFn(embd_f32, @intCast(n_embd_out), @intCast(n_tokens_out), cur_n_past, non_causal);
                } else return error.AudioEncoderNotAvailable;
                idx += 1;
            },
        }
    }
    return cur_n_past;
}

// ============================================================================
// Decoder positions for M-RoPE
// ============================================================================

pub fn imageGetDecoderPos(image: mtmd.ImageTokens, pos_0: u32, out_pos: []mtmd.DecoderPos) void {
    const n_tokens = image.nTokens();
    std.debug.assert(out_pos.len >= n_tokens);
    switch (image.pos) {
        .normal => {
            for (0..n_tokens) |i| out_pos[i] = .{ .t = pos_0 + @as(u32, @intCast(i)), .x = 0, .y = 0 };
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
// File loading
// ============================================================================

pub const BitmapWrapper = struct {
    bitmap: mtmd.Bitmap,
    allocator: std.mem.Allocator,
    pub fn deinit(self: *BitmapWrapper) void {
        self.bitmap.deinit();
    }
};

pub fn bitmapInitFromFile(allocator: std.mem.Allocator, io: std.Io, filepath: []const u8, placeholder: bool) !BitmapWrapper {
    if (placeholder) return BitmapWrapper{ .bitmap = mtmd.Bitmap.initPlaceholderImage(224, 224), .allocator = allocator };
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, filepath, .{ .mode = .read_only });
    defer file.close(io);
    const stat = try file.stat(io);
    const raw = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(raw);
    _ = try file.readPositionalAll(io, raw, 0);
    return bitmapInitFromBuf(allocator, raw, false);
}

pub fn bitmapInitFromBuf(allocator: std.mem.Allocator, buf: []const u8, placeholder: bool) !BitmapWrapper {
    if (placeholder) return BitmapWrapper{ .bitmap = mtmd.Bitmap.initPlaceholderImage(224, 224), .allocator = allocator };
    // JPEG
    if (buf.len >= 3 and buf[0] == 0xFF and buf[1] == 0xD8 and buf[2] == 0xFF) return loadImage(allocator, buf);
    // PNG
    if (buf.len >= 4 and std.mem.eql(u8, buf[0..4], &.{ 0x89, 0x50, 0x4E, 0x47 })) return loadImage(allocator, buf);
    // GIF
    if (buf.len >= 4 and buf[0] == 'G' and buf[1] == 'I' and buf[2] == 'F') return loadImage(allocator, buf);
    // BMP
    if (buf.len >= 2 and buf[0] == 'B' and buf[1] == 'M') return loadImage(allocator, buf);
    // WAV audio
    if (buf.len >= 4 and std.mem.eql(u8, buf[0..4], "RIFF")) return loadAudio(allocator, buf);
    return loadImage(allocator, buf);
}

fn loadImage(allocator: std.mem.Allocator, buf: []const u8) !BitmapWrapper {
    var width: c_int = 0;
    var height: c_int = 0;
    var channels: c_int = 0;
    const data = stb_image.loadFromMemory(buf.ptr, @intCast(buf.len), &width, &height, &channels, 3);
    if (data == null) return error.ImageDecodeFailed;
    defer stb_image.free(data);
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    const size: usize = w * h * 3;
    const owned = try allocator.alloc(u8, size);
    @memcpy(owned, data[0..size]);
    return BitmapWrapper{ .bitmap = .{ .nx = w, .ny = h, .data = owned, .allocator = allocator }, .allocator = allocator };
}

fn loadAudio(allocator: std.mem.Allocator, buf: []const u8) !BitmapWrapper {
    if (buf.len < 44) return error.InvalidWavFormat;
    const data_offset: usize = 44;
    const sample_count = (buf.len - data_offset) / 2;
    const owned = try allocator.alloc(u8, sample_count * @sizeOf(f32));
    const dst = @as([*]f32, @ptrCast(@alignCast(owned.ptr)))[0..sample_count];
    const src = @as([*]const i16, @ptrCast(@alignCast(buf[data_offset..].ptr)))[0..sample_count];
    for (0..sample_count) |i| dst[i] = @as(f32, @floatFromInt(src[i])) / 32768.0;
    return BitmapWrapper{ .bitmap = mtmd.Bitmap{ .nx = @intCast(sample_count), .ny = 1, .is_audio = true, .data = owned, .allocator = allocator }, .allocator = allocator };
}
