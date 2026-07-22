//! mtmd tokenizer — splits text by media markers, preprocesses bitmaps.

const std = @import("std");
const tokenizer = @import("tokenizer");
const mtmd = @import("mm");
const preprocess = @import("preprocess");
const debug_save_chunks = mtmd.debug_save_chunks.saveInputChunks;

const log = std.log.scoped(.mtmd_tokenize);

pub fn tokenize(ctx: *mtmd.MtmdContext, io: std.Io, allocator: std.mem.Allocator, text: mtmd.InputText, bitmaps: []const mtmd.Bitmap) !mtmd.InputChunks {
    var chunks = mtmd.InputChunks.init(allocator);
    errdefer chunks.deinit();
    const marker = ctx.media_marker;
    var remaining = text.text;
    var i_bm: usize = 0;
    var n_images_added: u32 = 0;
    while (remaining.len > 0) {
        if (std.mem.indexOf(u8, remaining, marker)) |idx| {
            if (idx > 0) try addTextChunk(ctx, allocator, &chunks, remaining[0..idx], text.parse_special);
            if (i_bm >= bitmaps.len) return error.MarkerBitmapMismatch;
            const bm = bitmaps[i_bm];
            i_bm += 1;
            if (bm.is_audio) try addAudioChunk(ctx, io, allocator, &chunks, bm) else try addImageChunk(ctx, io, allocator, &chunks, bm, &n_images_added);
            remaining = remaining[idx + marker.len ..];
        } else {
            if (remaining.len > 0) try addTextChunk(ctx, allocator, &chunks, remaining, text.parse_special);
            break;
        }
    }
    if (i_bm != bitmaps.len) return error.MarkerBitmapMismatch;

    // Add BOS token if add_special is true and vocab has add_bos.
    // Matches llama.cpp behavior: mtmd_tokenizer::tokenize() adds BOS after all chunks are built.
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
        }
    }

    try debug_save_chunks(io, allocator, "debug_vision", "zllama_vision_mtmd_input_chunks.txt", text.text, &chunks);
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
    if (chunks.entries.items.len > 0 and chunks.entries.items[chunks.entries.items.len - 1].chunk_type == .text) {
        const last = &chunks.entries.items[chunks.entries.items.len - 1];
        const old = last.tokens_text orelse &.{};
        const m = try al.alloc(i32, old.len + toks.items.len);
        @memcpy(m[0..old.len], old);
        @memcpy(m[old.len..], toks.items);
        if (last.tokens_text) |t| al.free(t);
        last.tokens_text = m;
    } else {
        const o = try al.dupe(i32, toks.items);
        try chunks.append(.{ .chunk_type = .text, .tokens_text = o });
    }
}

fn addImageChunk(ctx: *mtmd.MtmdContext, io: std.Io, al: std.mem.Allocator, chunks: *mtmd.InputChunks, bm: mtmd.Bitmap, nia: *u32) !void {
    if (!ctx.supportVision()) return error.VisionNotSupported;
    if (ctx.img_beg.len > 0) try addTextChunk(ctx, al, chunks, ctx.img_beg, true);
    var pw: u32 = bm.nx;
    var ph: u32 = bm.ny;
    var pd: ?[]u8 = null;
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
        pd = res.data;
    }
    // Estimate encoder output token count and token grid dimensions.
    // Matches llama.cpp mtmd_tokenizer::add_media() logic:
    //   - For M-RoPE models (qwen2vl/qwen3vl): nx = patches_x/2, ny = patches_y/2
    //   - For other models: nx = n_tokens, ny = 1
    var n_tokens: u32 = 0;
    var grid_nx: u32 = 0;
    var grid_ny: u32 = 0;
    if (ctx.mm_manager.vision_encoder) |*enc| {
        n_tokens = enc.estimateOutputTokens(io, pw, ph);
        if (ctx.pos_type == .mrope) {
            const ps = enc.params.patch_size;
            const patches_x = (pw + ps - 1) / ps;
            const patches_y = (ph + ps - 1) / ps;
            grid_nx = patches_x / 2;
            grid_ny = patches_y / 2;
        } else {
            grid_nx = n_tokens;
            grid_ny = 1;
        }
    }
    try chunks.append(.{ .chunk_type = .image, .tokens_image = .{ .nx = grid_nx, .ny = grid_ny, .n_tokens = n_tokens, .pos = ctx.pos_type, .image_idx = nia.*, .id = bm.id, .raw_pixels = pd }, .id = bm.id });
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
