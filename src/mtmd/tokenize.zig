//! mtmd tokenizer
const std = @import("std");
const tokenizer = @import("tokenizer");
const mtmd = @import("mtmd");
const log = std.log.scoped(.mtmd_tokenize);

pub fn tokenize(ctx: *mtmd.MtmdContext, allocator: std.mem.Allocator, text: mtmd.InputText, bitmaps: []const mtmd.Bitmap) !mtmd.InputChunks {
    var chunks = mtmd.InputChunks.init(allocator);
    errdefer chunks.deinit();
    const marker = ctx.media_marker;
    var remaining = text.text;
    var i_bm: usize = 0;
    var n_images_added: u32 = 0;
    while (remaining.len > 0) {
        if (std.mem.indexOf(u8, remaining, marker)) |idx| {
            if (idx > 0) try addTextChunk(ctx, allocator, &chunks, remaining[0..idx], text.parse_special);
            if (i_bm >= bitmaps.len) { log.debug("More markers than bitmaps", .{}); return error.MarkerBitmapMismatch; }
            const bm = bitmaps[i_bm]; i_bm += 1;
            if (bm.is_audio) { try addAudioChunk(ctx, allocator, &chunks, bm); }
            else { try addImageChunk(ctx, allocator, &chunks, bm, &n_images_added); }
            remaining = remaining[idx + marker.len ..];
        } else {
            if (remaining.len > 0) try addTextChunk(ctx, allocator, &chunks, remaining, text.parse_special);
            break;
        }
    }
    if (i_bm != bitmaps.len) { log.debug("Mismatch: {d} vs {d}", .{ i_bm, bitmaps.len }); return error.MarkerBitmapMismatch; }
    return chunks;
}

fn addTextChunk(ctx: *mtmd.MtmdContext, allocator: std.mem.Allocator, chunks: *mtmd.InputChunks, txt: []const u8, parse_special: bool) !void {
    if (txt.len == 0) return;
    var tokens = std.ArrayList(i32).initCapacity(allocator, 0) catch @panic("OOM");
    defer tokens.deinit(allocator);
    if (ctx.tok) |tok| {
        // Text chunks within multimodal input: no BOS/EOS, but parse special tokens
        var encoded = try tok.encode(txt, false, parse_special);
        defer encoded.deinit(allocator);
        for (encoded.items) |token| try tokens.append(allocator, @intCast(token));
    } else {
        for (txt) |c| try tokens.append(allocator, @intCast(c));
    }
    if (tokens.items.len == 0) return;
    if (chunks.entries.items.len > 0 and chunks.entries.items[chunks.entries.items.len - 1].chunk_type == .text) {
        const last = &chunks.entries.items[chunks.entries.items.len - 1];
        const old = last.tokens_text orelse &.{};
        const merged = try allocator.alloc(i32, old.len + tokens.items.len);
        @memcpy(merged[0..old.len], old); @memcpy(merged[old.len..], tokens.items);
        if (last.tokens_text) |t| allocator.free(t);
        last.tokens_text = merged;
    } else {
        const owned = try allocator.dupe(i32, tokens.items);
        try chunks.append(.{ .chunk_type = .text, .tokens_text = owned });
    }
}

fn addImageChunk(ctx: *mtmd.MtmdContext, allocator: std.mem.Allocator, chunks: *mtmd.InputChunks, bm: mtmd.Bitmap, n_images_added: *u32) !void {
    if (!ctx.supportVision()) return error.VisionNotSupported;
    if (ctx.img_beg.len > 0) try addTextChunk(ctx, allocator, chunks, ctx.img_beg, true);
    var n_tokens: u32 = 0; var nx: u32 = 0; var ny: u32 = 0;
    if (ctx.mm_manager.vision_encoder) |*enc| { n_tokens = enc.estimateOutputTokens(bm.nx, bm.ny); nx = n_tokens; ny = 1; }
    try chunks.append(.{ .chunk_type = .image, .tokens_image = .{ .nx = nx, .ny = ny, .pos = ctx.pos_type, .image_idx = n_images_added.*, .id = bm.id, .patch_count = 1 }, .id = bm.id });
    if (ctx.img_end.len > 0) try addTextChunk(ctx, allocator, chunks, ctx.img_end, true);
    n_images_added.* += 1;
}

fn addAudioChunk(ctx: *mtmd.MtmdContext, allocator: std.mem.Allocator, chunks: *mtmd.InputChunks, bm: mtmd.Bitmap) !void {
    if (!ctx.supportAudio()) return error.AudioNotSupported;
    if (ctx.aud_beg.len > 0) try addTextChunk(ctx, allocator, chunks, ctx.aud_beg, true);
    var n_tokens: u32 = 0;
    if (ctx.mm_manager.audio_encoder) |*enc| {
        const dur: f32 = @as(f32, @floatFromInt(bm.nx)) / @as(f32, @floatFromInt(@max(ctx.getAudioSampleRate(), 1)));
        n_tokens = enc.estimateOutputTokens(dur);
    }
    try chunks.append(.{ .chunk_type = .audio, .tokens_audio_n = n_tokens, .id = bm.id });
    if (ctx.aud_end.len > 0) try addTextChunk(ctx, allocator, chunks, ctx.aud_end, true);
}
