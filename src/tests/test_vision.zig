//! Vision processing tests for the zllama.zig multimodal pipeline.
//!
//! Covers the main vision processing steps without requiring actual model files:
//!   - Placeholder token scanning and expansion
//!   - Image media type handling in chat template
//!   - Image bilinear resize
//!   - Image normalization

const std = @import("std");
const testing = std.testing;
const chat_template = @import("chat_template");
const preprocess = @import("preprocess");
const ggml = @import("ggml");

const MediaType = chat_template.MediaType;
const Media = chat_template.Media;
const ChatMessage = chat_template.ChatMessage;
const PlaceholderInfo = chat_template.PlaceholderInfo;

// ============================================================================
// Placeholder scanning — image
// ============================================================================

test "scanPlaceholders: single image" {
    const text = "Describe <|image|> this";
    const placeholders = try chat_template.scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 1), placeholders.len);
    try testing.expectEqual(MediaType.image, placeholders[0].media_type);
}

test "scanPlaceholders: image alt format" {
    const text = "Look at <image> here";
    const placeholders = try chat_template.scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 1), placeholders.len);
    try testing.expectEqual(MediaType.image, placeholders[0].media_type);
}

test "scanPlaceholders: multiple images" {
    const text = "<|image|>A<|image|>B<image>C";
    const placeholders = try chat_template.scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 3), placeholders.len);
    for (placeholders) |ph| {
        try testing.expectEqual(MediaType.image, ph.media_type);
    }
}

test "containsPlaceholder: image" {
    try testing.expect(chat_template.containsPlaceholder("<|image|>"));
    try testing.expect(chat_template.containsPlaceholder("<image>"));
    try testing.expect(chat_template.containsPlaceholder("Describe <|image|> this"));
    try testing.expect(!chat_template.containsPlaceholder("no image here"));
}

// ============================================================================
// Placeholder expansion — image tokens
// ============================================================================

test "expandPlaceholders: image with tokens" {
    // expandPlaceholders only expands placeholder tokens, not surrounding text
    const text = "<|image|>";
    var expanded = try chat_template.expandPlaceholders(
        testing.allocator, text, 200, 0, 1024, 0,
    );
    defer expanded.deinit();

    try testing.expectEqual(@as(usize, 1024), expanded.tokens.items.len);
    try testing.expectEqual(@as(usize, 1), expanded.offsets.len);
    try testing.expectEqual(MediaType.image, expanded.offsets[0].media_type);
    try testing.expectEqual(@as(u32, 1024), expanded.offsets[0].token_count);
    try testing.expectEqual(@as(u32, 200), expanded.tokens.items[0]);
    try testing.expectEqual(@as(u32, 200), expanded.tokens.items[1023]);
}

test "expandPlaceholders: two placeholders with tokens" {
    const text = "<|image|><|audio|>";
    var expanded = try chat_template.expandPlaceholders(
        testing.allocator, text, 200, 100, 1024, 20,
    );
    defer expanded.deinit();

    try testing.expectEqual(@as(usize, 2), expanded.offsets.len);

    try testing.expectEqual(MediaType.image, expanded.offsets[0].media_type);
    try testing.expectEqual(@as(u32, 1024), expanded.offsets[0].token_count);
    try testing.expectEqual(@as(u32, 200), expanded.tokens.items[0]);

    try testing.expectEqual(MediaType.audio, expanded.offsets[1].media_type);
    try testing.expectEqual(@as(u32, 20), expanded.offsets[1].token_count);
    try testing.expectEqual(@as(u32, 100), expanded.tokens.items[1024]);
}

// ============================================================================
// ensurePlaceholderInContent — image
// ============================================================================

test "ensurePlaceholderInContent: already has image" {
    const content = "Describe <|image|> this";
    const result = try chat_template.ensurePlaceholderInContent(content, .image, testing.allocator);
    defer if (result.ptr != content.ptr) testing.allocator.free(result);
    try testing.expectEqualStrings(content, result);
}

test "ensurePlaceholderInContent: add image" {
    const content = "Describe this";
    const result = try chat_template.ensurePlaceholderInContent(content, .image, testing.allocator);
    defer testing.allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "<|image|>") != null);
}

test "ensurePlaceholderInContent: add image with alt already present" {
    const content = "Describe <image> this";
    const result = try chat_template.ensurePlaceholderInContent(content, .image, testing.allocator);
    defer if (result.ptr != content.ptr) testing.allocator.free(result);
    try testing.expectEqualStrings(content, result);
}

// ============================================================================
// Media type — image in ChatMessage
// ============================================================================

test "ChatMessage withMedia: image" {
    var pixel_data = [_]u8{ 255, 0, 0 };
    const media = Media{
        .type = .image,
        .data = .{ .image = .{ .data = &pixel_data, .width = 1, .height = 1 } },
    };
    const msg = ChatMessage.withMedia("user", "Describe this", media);
    try testing.expect(msg.media != null);
    try testing.expect(msg.hasMediaType(.image));
    try testing.expect(!msg.hasMediaType(.audio));
    try testing.expectEqualStrings("user", msg.role);
    try testing.expectEqualStrings("Describe this", msg.content);
}

// ============================================================================
// Image bilinear resize
// ============================================================================

test "bilinearResizeRGB: 2x2 to 1x1" {
    const src_w: u32 = 2;
    const src_h: u32 = 2;
    const src = [_]u8{
        255, 0,   0,   // R
        0,   255, 0,   // G
        0,   0,   255, // B
        255, 255, 0,   // Y
    };
    const dst_w: u32 = 1;
    const dst_h: u32 = 1;

    var img = try preprocess.fromRawRGB(testing.allocator, &src, src_w, src_h, dst_w);
    defer img.deinit();

    try testing.expectEqual(dst_w, img.width);
    try testing.expectEqual(dst_h, img.height);
    try testing.expectEqual(@as(usize, 3), img.data.len); // 1x1x3

    // Bilinear average of all 4 corners should be ~128 for R and G channels
    try testing.expect(@abs(@as(i16, img.data[0]) - 128) <= 1);
    try testing.expect(@abs(@as(i16, img.data[1]) - 128) <= 1);
}

test "bilinearResizeRGB: 4x2 to 2x1" {
    const src_w: u32 = 4;
    const src_h: u32 = 2;
    const src = try testing.allocator.alloc(u8, src_w * src_h * 3);
    defer testing.allocator.free(src);
    @memset(src, 255);

    // fromRawRGB resizes to target_size × target_size, so 2 means 2×2
    const target_size: u32 = 2;

    var img = try preprocess.fromRawRGB(testing.allocator, src, src_w, src_h, target_size);
    defer img.deinit();

    try testing.expectEqual(target_size, img.width);
    try testing.expectEqual(target_size, img.height);
    try testing.expectEqual(@as(usize, target_size * target_size * 3), img.data.len);

    // All white should stay white
    for (img.data) |v| {
        try testing.expectEqual(@as(u8, 255), v);
    }
}

// ============================================================================
// Image normalization
// ============================================================================

test "imageToTensor: siglip normalization" {
    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();

    ctx.setNoAlloc(false);

    // 1x1 image: single red pixel
    const img_data = [_]u8{ 255, 128, 0 };
    var img = preprocess.ProcessedImage{
        .data = @constCast(&img_data),
        .width = 1,
        .height = 1,
        .allocator = testing.allocator,
    };

    const tensor = try preprocess.imageToTensor(ctx, &img, .siglip);
    const data = tensor.dataF32();

    // For 1x1 image, data is [R, G, B] regardless of layout
    // R=255 → (255/255)*2-1 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    // G=128 → (128/255)*2-1 ≈ 0.0039
    try testing.expectApproxEqAbs(@as(f32, 0.0039217), data[1], 1e-4);
    // B=0 → (0/255)*2-1 = -1.0
    try testing.expectApproxEqAbs(@as(f32, -1.0), data[2], 1e-5);

    ctx.setNoAlloc(true);
}

test "imageToTensor: none normalization (passthrough)" {
    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();

    ctx.setNoAlloc(false);

    const img_data = [_]u8{ 100, 150, 200 };
    var img = preprocess.ProcessedImage{
        .data = @constCast(&img_data),
        .width = 1,
        .height = 1,
        .allocator = testing.allocator,
    };

    const tensor = try preprocess.imageToTensor(ctx, &img, .none);
    const data = tensor.dataF32();

    try testing.expectApproxEqAbs(@as(f32, 100.0), data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 150.0), data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 200.0), data[2], 1e-5);

    ctx.setNoAlloc(true);
}

// ============================================================================
// Placeholder token offset computation
// ============================================================================

test "placeholderTokenOffset: image single" {
    const offsets = [_]PlaceholderInfo{
        .{ .start = 0, .length = 9, .media_type = .image, .token_count = 1024, .token_offset = 0 },
    };
    const offset = chat_template.placeholderTokenOffset(&offsets, 0);
    try testing.expectEqual(@as(u32, 0), offset);
}

test "placeholderTokenOffset: image second" {
    const offsets = [_]PlaceholderInfo{
        .{ .start = 0, .length = 9, .media_type = .image, .token_count = 1024, .token_offset = 0 },
        .{ .start = 9, .length = 9, .media_type = .image, .token_count = 1024, .token_offset = 1024 },
    };
    const offset = chat_template.placeholderTokenOffset(&offsets, 1);
    try testing.expectEqual(@as(u32, 1024), offset);
}
