//! Audio processing tests for the zllama.zig multimodal pipeline.
//!
//! Covers the main audio processing steps without requiring actual model files:
//!   - Placeholder token scanning and expansion
//!   - Audio media type handling in chat template
//!   - Mel filterbank computation
//!   - WAV loading basics (format detection)

const std = @import("std");
const testing = std.testing;
const chat_template = @import("chat_template");
const preprocess = @import("preprocess");

// Access types through chat_template module (re-exports)
const MediaType = chat_template.MediaType;
const Media = chat_template.Media;
const ChatMessage = chat_template.ChatMessage;
const PlaceholderInfo = chat_template.PlaceholderInfo;

// ============================================================================
// Placeholder scanning — audio
// ============================================================================

test "scanPlaceholders: single audio" {
    const text = "Transcribe <|audio|> please";
    const placeholders = try chat_template.scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 1), placeholders.len);
    try testing.expectEqual(MediaType.audio, placeholders[0].media_type);
    // "Transcribe " is 11 characters before <|audio|>
    try testing.expectEqual(@as(usize, 11), placeholders[0].start);
}

test "scanPlaceholders: audio alt format" {
    const text = "Listen to <audio> now";
    const placeholders = try chat_template.scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 1), placeholders.len);
    try testing.expectEqual(MediaType.audio, placeholders[0].media_type);
}

test "scanPlaceholders: mixed image and audio" {
    const text = "<|image|>First<|audio|>Second";
    const placeholders = try chat_template.scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 2), placeholders.len);
    try testing.expectEqual(MediaType.image, placeholders[0].media_type);
    try testing.expectEqual(MediaType.audio, placeholders[1].media_type);
}

test "containsPlaceholder: audio" {
    try testing.expect(chat_template.containsPlaceholder("<|audio|>"));
    try testing.expect(chat_template.containsPlaceholder("<audio>"));
    try testing.expect(chat_template.containsPlaceholder("Transcribe <|audio|> please"));
    try testing.expect(!chat_template.containsPlaceholder("plain text only"));
}

// ============================================================================
// Placeholder expansion — audio tokens
// ============================================================================

test "expandPlaceholders: audio with tokens" {
    // expandPlaceholders only expands placeholder tokens, not surrounding text
    const text = "<|audio|>";
    var expanded = try chat_template.expandPlaceholders(
        testing.allocator, text, 0, 100, 0, 20,
    );
    defer expanded.deinit();

    // Only audio placeholder tokens should be present
    try testing.expectEqual(@as(usize, 20), expanded.tokens.items.len);
    try testing.expectEqual(@as(usize, 1), expanded.offsets.len);
    try testing.expectEqual(MediaType.audio, expanded.offsets[0].media_type);
    try testing.expectEqual(@as(u32, 20), expanded.offsets[0].token_count);
    try testing.expectEqual(@as(u32, 100), expanded.tokens.items[0]);
    try testing.expectEqual(@as(u32, 100), expanded.tokens.items[19]);
}

// ============================================================================
// ensurePlaceholderInContent — audio
// ============================================================================

test "ensurePlaceholderInContent: already has audio" {
    const content = "Transcribe <|audio|> this";
    const result = try chat_template.ensurePlaceholderInContent(content, .audio, testing.allocator);
    defer if (result.ptr != content.ptr) testing.allocator.free(result);
    try testing.expectEqualStrings(content, result);
}

test "ensurePlaceholderInContent: add audio" {
    const content = "Transcribe this";
    const result = try chat_template.ensurePlaceholderInContent(content, .audio, testing.allocator);
    defer testing.allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "<|audio|>") != null);
}

test "ensurePlaceholderInContent: add audio with alt already present" {
    const content = "Transcribe <audio> this";
    const result = try chat_template.ensurePlaceholderInContent(content, .audio, testing.allocator);
    defer if (result.ptr != content.ptr) testing.allocator.free(result);
    try testing.expectEqualStrings(content, result);
}

// ============================================================================
// Media type — audio in ChatMessage
// ============================================================================

test "ChatMessage withMedia: audio" {
    const media = Media{
        .type = .audio,
        .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 16000 } },
    };
    const msg = ChatMessage.withMedia("user", "Transcribe this", media);
    try testing.expect(msg.media != null);
    try testing.expect(msg.hasMediaType(.audio));
    try testing.expect(!msg.hasMediaType(.image));
    try testing.expectEqualStrings("user", msg.role);
    try testing.expectEqualStrings("Transcribe this", msg.content);
}

// ============================================================================
// Mel filterbank computation
// ============================================================================

test "Mel filterbank: basic properties" {
    // Create a sine wave with enough samples for 1 frame.
    // With pad_left=frame_length/2=200 and frame_length=400:
    //   n_frames = (n_padded - 400) / 160 + 1
    //   For 1 frame: (n_padded - 400) / 160 + 1 = 1 → n_padded = 400
    //   So n_samples = 400 - pad_left = 200
    const n_samples = 200;
    var sine_samples = try testing.allocator.alloc(f32, n_samples);
    defer testing.allocator.free(sine_samples);

    const freq: f32 = 440.0;
    for (0..n_samples) |i| {
        sine_samples[i] = @sin(2.0 * std.math.pi * freq * @as(f32, @floatFromInt(i)) / 16000.0);
    }

    const params = preprocess.AudioPreprocessParams{
        .sample_rate = 16000,
        .frame_length = 400,
        .hop_length = 160,
        .n_fft = 512,
        .n_mel_bins = 16,
        .mel_f_min = 0.0,
        .mel_f_max = 8000.0,
        .pre_emphasis = 0.0,
        .log_offset = 0.001,
    };

    var mel = try preprocess.computeMelSpectrogram(testing.allocator, sine_samples, 16000, params);
    defer mel.deinit();

    // With 200 samples + 200 pad_left = 400 padded, exactly 1 frame
    try testing.expectEqual(@as(u32, 1), mel.n_frames);
    try testing.expectEqual(@as(u32, 16), mel.n_mel_bins);
    try testing.expect(mel.data.len == 16);

    // All Mel values should be finite (no NaN or inf)
    for (mel.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

// ============================================================================
// Placeholder token offset computation
// ============================================================================

test "placeholderTokenOffset: audio single" {
    const offsets = [_]PlaceholderInfo{
        .{ .start = 10, .length = 9, .media_type = .audio, .token_count = 20, .token_offset = 0 },
    };
    const offset = chat_template.placeholderTokenOffset(&offsets, 0);
    try testing.expectEqual(@as(u32, 0), offset);
}

test "placeholderTokenOffset: audio after image" {
    const offsets = [_]PlaceholderInfo{
        .{ .start = 0, .length = 9, .media_type = .image, .token_count = 1024, .token_offset = 0 },
        .{ .start = 9, .length = 9, .media_type = .audio, .token_count = 20, .token_offset = 1024 },
    };
    const offset = chat_template.placeholderTokenOffset(&offsets, 1);
    try testing.expectEqual(@as(u32, 1024), offset);
}
