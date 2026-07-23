//! mtmd tokenizer — splits text by media markers, preprocesses bitmaps.
//!
//! Reference: deps/llama.cpp/tools/mtmd/mtmd.cpp (mtmd_tokenizer struct)
//!
//! ## 与 C++ 参考实现的关键差异
//!
//! C++ 的 add_media() 使用 image_preproc->preprocess() 预处理图像并返回
//! mtmd_image_preproc_out（包含 entries、grid、overview），然后将预处理后的
//! clip_image_f32 存储在 chunk 的 batch_f32 中。编码时直接使用这些预处理数据。
//!
//! 我们的实现将预处理延迟到 evalChunks 阶段：addImageChunk 只做 resize 到合适
//! 尺寸并存储 raw_pixels，实际的归一化和编码在 evalChunks 中完成。
//! 这是架构层面的差异，不影响功能正确性。
//!
//! 当前未实现的功能（未来可扩展）：
//! - llava-uhd tiling（grid/overview/slices 布局）
//! - slice template 特殊 token 插入（minicpmv、idefics3、step3vl 等）

const std = @import("std");
const tokenizer = @import("tokenizer");
const mtmd = @import("mm");
const preprocess = @import("preprocess");

const log = std.log.scoped(.mtmd_tokenize);

/// Split text by media marker, returning alternating text/marker parts.
/// Matches llama.cpp mtmd_tokenizer::split_text().
fn splitText(allocator: std.mem.Allocator, input: []const u8, delimiter: []const u8) std.ArrayList([]const u8) {
    var result = std.ArrayList([]const u8).initCapacity(allocator, 0) catch @panic("OOM");
    if (input.len == 0) return result;
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, input, start, delimiter)) |pos| {
        if (pos > start) {
            result.append(allocator, input[start..pos]) catch @panic("OOM");
        }
        result.append(allocator, delimiter) catch @panic("OOM");
        start = pos + delimiter.len;
    }
    if (start < input.len) {
        result.append(allocator, input[start..]) catch @panic("OOM");
    }
    return result;
}

pub fn tokenize(ctx: *mtmd.MtmdContext, io: std.Io, allocator: std.mem.Allocator, text: mtmd.InputText, bitmaps: []const mtmd.Bitmap) !mtmd.InputChunks {
    var chunks = mtmd.InputChunks.init(allocator);
    errdefer chunks.deinit();
    const marker = ctx.media_marker;

    // Step 1: Split text by media marker (matches C++ split_text)
    var parts = splitText(allocator, text.text, marker);
    defer parts.deinit(allocator);

    // Step 2: Validate marker/bitmap count
    var n_markers: usize = 0;
    for (parts.items) |part| {
        if (std.mem.eql(u8, part, marker)) n_markers += 1;
    }
    if (n_markers != bitmaps.len) {
        log.err("number of media markers in text ({d}) does not match number of bitmaps ({d})", .{ n_markers, bitmaps.len });
        return error.MarkerBitmapMismatch;
    }

    // Step 3: Handle frame merging (temporal merge) for Qwen-VL style models.
    // Matches llama.cpp mtmd_tokenizer::tokenize() lines 918-952.
    const n_merge_frames: u32 = if (ctx.mm_manager.vision_encoder) |*enc|
        if (enc.backend.n_temporal_merge) |ntm| ntm else 1
    else
        1;

    // Build merged_bitmaps: each entry is a group of 1 or 2 bitmaps.
    // For consecutive mergeable bitmap parts, merge them.
    var merged_groups = std.ArrayList(std.ArrayList(*const mtmd.Bitmap)).initCapacity(allocator, 0) catch @panic("OOM");
    defer {
        for (merged_groups.items) |*g| g.deinit(allocator);
        merged_groups.deinit(allocator);
    }

    if (n_merge_frames > 1) {
        var bm_idx: usize = 0;
        while (bm_idx < bitmaps.len) {
            var group = std.ArrayList(*const mtmd.Bitmap).initCapacity(allocator, 0) catch @panic("OOM");
            try group.append(allocator, &bitmaps[bm_idx]);
            // Try to merge with next bitmap if they can be merged
            if (bm_idx + 1 < bitmaps.len and bitmaps[bm_idx].canMergeWith(bitmaps[bm_idx + 1])) {
                log.debug("merging 2 frames at bitmap index {d} and {d}", .{ bm_idx, bm_idx + 1 });
                try group.append(allocator, &bitmaps[bm_idx + 1]);
                bm_idx += 2;
            } else {
                log.debug("no merging for bitmap index {d}", .{bm_idx});
                bm_idx += 1;
            }
            try merged_groups.append(allocator, group);
        }
    } else {
        for (bitmaps) |*bm| {
            var group = std.ArrayList(*const mtmd.Bitmap).initCapacity(allocator, 0) catch @panic("OOM");
            try group.append(allocator, bm);
            try merged_groups.append(allocator, group);
        }
    }

    // Step 4: Process parts, calling add_media for markers and add_text for text.
    var group_idx: usize = 0;
    var n_images_added: u32 = 0;

    for (parts.items) |part| {
        if (std.mem.eql(u8, part, marker)) {
            if (group_idx >= merged_groups.items.len) {
                return error.MarkerBitmapMismatch;
            }
            const group = &merged_groups.items[group_idx];
            group_idx += 1;

            // Determine if this is vision or audio from the first bitmap
            if (group.items.len > 0 and group.items[0].is_audio) {
                // Audio: process each bitmap individually (no batching for audio yet)
                for (group.items) |bm| {
                    try addAudioChunk(ctx, io, allocator, &chunks, bm.*);
                }
            } else {
                // Vision: pass the merged group to addImageChunk
                try addImageChunk(ctx, io, allocator, &chunks, group.items, &n_images_added, n_merge_frames);
            }
        } else {
            try addTextChunk(ctx, allocator, &chunks, part, text.parse_special);
        }
    }

    if (group_idx != merged_groups.items.len) {
        return error.MarkerBitmapMismatch;
    }

    // Step 5: Add BOS/EOS tokens if add_special is true.
    // Matches llama.cpp mtmd_tokenizer::tokenize() lines 972-995.
    if (text.add_special) {
        if (ctx.tok) |tok| {
            if (tok.vocab.getAddBos()) {
                const bos_token: i32 = @intCast(tok.vocab.tokenBos());
                if (chunks.entries.items.len > 0 and chunks.entries.items[0].chunk_type == .text) {
                    const first = &chunks.entries.items[0];
                    const old = first.tokens_text orelse &.{};
                    const m = try allocator.alloc(i32, old.len + 1);
                    m[0] = bos_token;
                    @memcpy(m[1..], old);
                    if (first.tokens_text) |t| allocator.free(t);
                    first.tokens_text = m;
                } else {
                    const o = try allocator.dupe(i32, &.{bos_token});
                    try chunks.entries.insert(allocator, 0, .{ .chunk_type = .text, .tokens_text = o });
                }
            }

            if (tok.vocab.getAddEos()) {
                const eos_token: i32 = @intCast(tok.vocab.tokenEos());
                try addTextTokens(allocator, &chunks, &.{eos_token});
            }
        }
    }

    return chunks;
}

fn addTextChunk(ctx: *mtmd.MtmdContext, al: std.mem.Allocator, chunks: *mtmd.InputChunks, txt: []const u8, ps: bool) !void {
    if (txt.len == 0) return;
    var toks = std.ArrayList(i32).initCapacity(al, 0) catch @panic("OOM");
    defer toks.deinit(al);
    if (ctx.tok) |tok| {
        var e = try tok.encode(txt, false, ps);
        defer e.deinit(al);
        for (e.items) |t| try toks.append(al, @intCast(t));
    } else for (txt) |c| try toks.append(al, @intCast(c));
    if (toks.items.len == 0) return;
    try addTextTokens(al, chunks, toks.items);
}

/// Add raw tokens to chunks, merging with last text chunk if possible.
/// Matches llama.cpp mtmd_tokenizer::add_text(tokens).
fn addTextTokens(al: std.mem.Allocator, chunks: *mtmd.InputChunks, tokens: []const i32) !void {
    if (tokens.len == 0) return;
    if (chunks.entries.items.len > 0 and chunks.entries.items[chunks.entries.items.len - 1].chunk_type == .text) {
        const last = &chunks.entries.items[chunks.entries.items.len - 1];
        const old = last.tokens_text orelse &.{};
        const m = try al.alloc(i32, old.len + tokens.len);
        @memcpy(m[0..old.len], old);
        @memcpy(m[old.len..], tokens);
        if (last.tokens_text) |t| al.free(t);
        last.tokens_text = m;
    } else {
        const o = try al.dupe(i32, tokens);
        try chunks.append(.{ .chunk_type = .text, .tokens_text = o });
    }
}

/// Add a media (image) chunk.
/// Matches llama.cpp mtmd_tokenizer::add_media() for vision.
///
/// `bitmaps` is a group of 1 or 2 bitmaps (for temporal merge).
/// For temporal merge (Qwen-VL), multiple bitmaps are merged into one embedding.
///
/// 与 C++ 参考实现的差异说明：
/// - C++ 使用 image_preproc->preprocess() 预处理图像并存储 clip_image_f32 在 batch_f32 中
/// - 我们存储 raw_pixels（resize 后的 u8 RGB 数据），预处理延迟到 evalChunks 阶段
/// - 因此我们不处理 llava-uhd tiling（grid/overview/slices），这需要完整的预处理管线
fn addImageChunk(
    ctx: *mtmd.MtmdContext,
    io: std.Io,
    al: std.mem.Allocator,
    chunks: *mtmd.InputChunks,
    bitmaps: []const *const mtmd.Bitmap,
    nia: *u32,
    n_merge_frames: u32,
) !void {
    if (!ctx.supportVision()) return error.VisionNotSupported;
    if (bitmaps.len == 0) return;

    // Add image begin token (matches C++: if (!ctx->img_beg.empty()) add_text(ctx->img_beg, true))
    if (ctx.img_beg.len > 0) try addTextChunk(ctx, al, chunks, ctx.img_beg, true);

    // Preprocess each bitmap in the merged group.
    // For temporal merge (Qwen-VL), multiple bitmaps are merged into one batch.
    var n_tokens: u32 = 0;
    var grid_nx: u32 = 0;
    var grid_ny: u32 = 0;
    var raw_pixels_list = std.ArrayList(?[]u8).initCapacity(al, 0) catch @panic("OOM");
    defer {
        for (raw_pixels_list.items) |rp| if (rp) |p| al.free(p);
        raw_pixels_list.deinit(al);
    }

    for (bitmaps) |bm| {
        var pw: u32 = bm.nx;
        var ph: u32 = bm.ny;
        if (!bm.isPlaceholder()) {
            const rd = bm.data orelse return error.NoImageData;
            const enc = ctx.mm_manager.vision_encoder orelse return error.VisionEncoderNotAvailable;
            const p = enc.params;
            if (p.image_min_pixels > 0 and p.image_max_pixels > 0) {
                const as = p.patch_size * @max(p.n_merge, 1);
                const mx = if (p.user_max_pixels > 0) p.user_max_pixels else p.image_max_pixels;
                const ns = preprocess.calcSizePreservedRatio(bm.nx, bm.ny, as, p.image_min_pixels, mx);
                pw = ns.width;
                ph = ns.height;
            }
            const res = try preprocess.resizeToU8(al, rd, bm.nx, bm.ny, pw, ph);
            try raw_pixels_list.append(al, res.data);
        } else {
            try raw_pixels_list.append(al, null);
        }

        // Estimate encoder output token count and token grid dimensions.
        // Matches llama.cpp mtmd_tokenizer::add_media() logic:
        //   - For M-RoPE models (qwen2vl/qwen3vl): nx = patches_x/2, ny = patches_y/2
        //   - For other models: nx = n_tokens, ny = 1
        if (ctx.mm_manager.vision_encoder) |*enc| {
            const this_n_tokens = enc.estimateOutputTokens(io, pw, ph);
            if (ctx.pos_type == .mrope) {
                const ps = enc.params.patch_size;
                const patches_x = (pw + ps - 1) / ps;
                const patches_y = (ph + ps - 1) / ps;
                grid_nx = patches_x / 2;
                grid_ny = patches_y / 2;
            }
            // For temporal merge (n_merge_frames > 1), only count tokens once
            // because paired inputs are merged to the same embedding.
            // Matches C++: if (clip_model_n_temporal_merge(ctx->ctx_v) == 2) { break; }
            if (n_merge_frames == 2) {
                n_tokens = this_n_tokens;
                break;
            }
            n_tokens += this_n_tokens;
        }
    }

    if (grid_nx == 0 and grid_ny == 0) {
        grid_nx = n_tokens;
        grid_ny = 1;
    }

    // Set n_temporal_merge from model capability.
    // Matches C++: image_tokens->n_temporal_merge = clip_model_n_temporal_merge(ctx->ctx_v);
    const temporal_merge = n_merge_frames;

    // For HunyuanVL, override position type and set image_idx.
    // Matches C++ lines 1208-1212.
    // Note: HunyuanVL detection would need projector_type info.
    // For now, we rely on ctx.pos_type which is set during init.

    // For temporal merge, store the first bitmap's pixels (the merge happens at encode time).
    const combined_pixels = if (raw_pixels_list.items.len > 0 and raw_pixels_list.items[0] != null)
        raw_pixels_list.items[0]
    else
        null;

    try chunks.append(.{
        .chunk_type = .image,
        .tokens_image = .{
            .nx = grid_nx,
            .ny = grid_ny,
            .n_tokens = n_tokens,
            .pos = ctx.pos_type,
            .image_idx = nia.*,
            .n_temporal_merge = temporal_merge,
            .id = if (bitmaps.len > 0) bitmaps[0].id else null,
            .raw_pixels = combined_pixels,
        },
        .id = if (bitmaps.len > 0) bitmaps[0].id else null,
    });

    // Add image end token (matches C++: if (!ctx->img_end.empty()) add_text(ctx->img_end, true))
    if (ctx.img_end.len > 0) try addTextChunk(ctx, al, chunks, ctx.img_end, true);
    nia.* += 1;
}

fn addAudioChunk(ctx: *mtmd.MtmdContext, io: std.Io, al: std.mem.Allocator, chunks: *mtmd.InputChunks, bm: mtmd.Bitmap) !void {
    if (!ctx.supportAudio()) return error.AudioNotSupported;
    if (ctx.aud_beg.len > 0) try addTextChunk(ctx, al, chunks, ctx.aud_beg, true);
    // Estimate encoder output token count (symmetric with addImageChunk).
    var n_tokens: u32 = 0;
    if (ctx.mm_manager.audio_encoder) |*enc| {
        n_tokens = enc.estimateOutputTokens(io, @as(f32, @floatFromInt(@max(bm.nx, 1))) / @as(f32, @floatFromInt(@max(@as(u32, @intCast(ctx.caps.audio_sample_rate)), 1))));
    }
    try chunks.append(.{ .chunk_type = .audio, .tokens_audio_n = n_tokens, .id = bm.id, .audio_data = bm.data });
    if (ctx.aud_end.len > 0) try addTextChunk(ctx, al, chunks, ctx.aud_end, true);
}
