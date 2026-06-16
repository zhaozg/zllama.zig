//! Gemma 4 template
//!
//! Format: <bos><|turn>system\n{system}<turn|>\n   ← system turn (when system present)
//!         <|turn>user\n<|media|>{user}<turn|>\n   ← media marker INLINE, per official Jinja
//!         <|turn>model\n{assistant}<turn|>\n
//!         <|turn>model\n<|channel>thought\n<channel|>   ← gen prompt (thinking disabled)
//!
//! Reference: google-gemma-4-31B-it-interleaved.jinja (deps/llama.cpp/models/templates/)
//!
//! Key behaviors from official template (verified 2025-06):
//!   - {{ bos_token }} at start
//!   - Media markers (<|image|>, <|audio|>) rendered INLINE via {{- '<|audio|>' -}}
//!     (the "-" modifier strips whitespace, so no newline between marker and text)
//!   - Empty thinking block <|channel>thought\n<channel|> when enable_thinking=false
//!   - System turn supports <|think|>\n prefix when enable_thinking=true
//!   - strip_thinking macro removes <|channel>thought\n...<channel|> from model output

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply Gemma 4 template.
///
/// Note: enable_thinking is currently hardcoded to false.
/// When true, the system turn would get a <|think|>\n prefix and
/// the generation prompt would NOT include the empty thinking block.
/// Full enable_thinking support requires plumbing through the vtable.
pub fn apply(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    // Default: thinking disabled
    const enable_thinking = false;

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    // --- BOS token ---
    // Official template: {{ bos_token }} at the very beginning.
    try buf.appendSlice(allocator, "<bos>");

    // Collect system content from the FIRST system message only
    // (only leading consecutive system messages, matching official template behavior)
    var system_content: ?[]const u8 = null;
    var has_system_from_messages = false;

    if (system_prompt) |sp| {
        if (sp.len > 0) {
            system_content = sp;
        }
    }

    // Check if the first message is a system message
    if (messages.len > 0 and std.mem.eql(u8, messages[0].role, "system")) {
        if (system_content) |sc| {
            // Merge: existing system_prompt + first message system content
            var merged = std.ArrayListUnmanaged(u8).empty;
            defer merged.deinit(allocator);
            try merged.appendSlice(allocator, sc);
            try merged.appendSlice(allocator, "\n\n");
            try merged.appendSlice(allocator, messages[0].content);
            system_content = try merged.toOwnedSlice(allocator);
        } else {
            system_content = messages[0].content;
        }
        has_system_from_messages = true;
    }

    const has_system = if (system_content) |sc| sc.len > 0 else false;

    // --- System turn (Gemma 4: dedicated turn, only from first message) ---
    if (has_system) {
        try buf.appendSlice(allocator, "<|turn>system\n");

        if (enable_thinking) {
            try buf.appendSlice(allocator, "<|think|>\n");
        }

        try buf.appendSlice(allocator, system_content.?);
        try buf.appendSlice(allocator, "<turn|>\n");
    }

    // --- Message turns ---
    // Skip the first message if it was a system message (already handled)
    const start_idx: usize = if (has_system_from_messages) 1 else 0;

    for (messages[start_idx..], 0..) |msg, i| {
        const role_tag = if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model"))
            "model"
        else
            msg.role;

        try buf.appendSlice(allocator, "<|turn>");
        try buf.appendSlice(allocator, role_tag);
        try buf.appendSlice(allocator, "\n");

        // User messages with media: render media marker INLINE before text content.
        // This matches the official template's {{- '<|audio|>' -}} behavior
        // where the "-" modifier strips surrounding whitespace.
        if (std.mem.eql(u8, role_tag, "user") and msg.media != null) {
            try appendMediaContent(&buf, allocator, msg.content, msg.media.?);
        } else if (std.mem.eql(u8, role_tag, "model")) {
            // For model messages, strip thinking blocks before rendering
            try appendStrippedThinking(&buf, allocator, msg.content);
        } else {
            try buf.appendSlice(allocator, msg.content);
        }

        try buf.appendSlice(allocator, "<turn|>\n");
        _ = i;
    }

    // --- Generation prompt ---
    // Official template:
    //   {%- if add_generation_prompt -%}
    //     {{- '<|turn>model\n' -}}
    //     {%- if not enable_thinking | default(false) -%}
    //       {{- '<|channel>thought\n<channel|>' -}}
    //     {%- endif -%}
    //   {%- endif -%}
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<|turn>model\n");
        // When thinking is disabled, add an empty thinking block to suppress
        // the model's internal reasoning. This matches the official template.
        if (enable_thinking) {
            try buf.appendSlice(allocator, "<|channel>thought\n<channel|>");
        }
    }

    return buf.toOwnedSlice(allocator);
}

/// Append user content with media marker placed INLINE before text.
/// Format: <media_marker><text_content>
///
/// This matches the official Jinja template behavior:
///   {{- '<|audio|>' -}}{{ item['text'] | trim }}
/// The "-" modifier in Jinja strips surrounding whitespace, so the marker
/// and text are directly adjacent (no newline between them).
fn appendMediaContent(
    buf: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    text_content: []const u8,
    media: types.Media,
) !void {
    const marker = switch (media.type) {
        .image => types.IMAGE_PLACEHOLDER,
        .audio => types.AUDIO_PLACEHOLDER,
        .none => "",
    };

    if (marker.len > 0) {
        // Inline marker: <|audio|>Transcribe the audio
        try buf.appendSlice(allocator, marker);
    }

    try buf.appendSlice(allocator, text_content);
}

/// Strip thinking blocks (<|channel>thought\n...<channel|>) from model content.
/// Only strips the "thought" channel, not other potential channels.
/// Reference: strip_thinking macro in google-gemma-4-31B-it-interleaved.jinja
fn appendStrippedThinking(
    buf: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    text: []const u8,
) !void {
    const thought_start_tag = "<|channel>thought\n";
    const channel_end_tag = "<channel|>";

    var remaining = text;
    while (remaining.len > 0) {
        if (std.mem.indexOf(u8, remaining, thought_start_tag)) |tag_pos| {
            // Output everything before the thought channel start tag
            try buf.appendSlice(allocator, remaining[0..tag_pos]);

            // Find the matching channel end tag
            const after_start = remaining[tag_pos + thought_start_tag.len ..];
            if (std.mem.indexOf(u8, after_start, channel_end_tag)) |end_pos| {
                // Skip the entire thought channel block
                remaining = after_start[end_pos + channel_end_tag.len ..];
            } else {
                // No closing tag found — output the rest as-is
                try buf.appendSlice(allocator, remaining[tag_pos..]);
                break;
            }
        } else {
            // No more thought channel tags
            try buf.appendSlice(allocator, remaining);
            break;
        }
    }
}

const testing = std.testing;

/// Helper: expected prefix for all Gemma 4 templates (bos_token)
const BOS = "<bos>";

test "gemma4: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    // Official format: <bos><|turn>user\n{content}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
    try testing.expectEqualStrings(
        BOS ++ "<|turn>user\n1+1=?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: with system" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are a helpful assistant.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        BOS ++ "<|turn>system\nYou are a helpful assistant.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: multi-turn" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "model", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        BOS ++ "<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: multi-turn with system" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "model", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        BOS ++ "<|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: strip thinking" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "1+1=?" },
        .{ .role = "model", .content = "<|channel>thought\nLet me think... 1+1=2\n<channel|>The answer is 2." },
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    // Thinking block should be stripped from model output
    try testing.expectEqualStrings(
        BOS ++ "<|turn>user\n1+1=?<turn|>\n<|turn>model\nThe answer is 2.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: system role message" {
    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "You are a math tutor." },
        .{ .role = "user", .content = "1+1=?" },
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        BOS ++ "<|turn>system\nYou are a math tutor.<turn|>\n<|turn>user\n1+1=?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: system role message with system_prompt" {
    // When both system_prompt and first message is system, they should be merged
    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "Be concise." },
        .{ .role = "user", .content = "Hello" },
    };
    const result = try apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "You are helpful.") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Be concise.") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|turn>system") != null);
}

test "gemma4: non-system first message" {
    // If the first message is NOT system, system_prompt is still used
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "system", .content = "This should be ignored" },
    };
    const result = try apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    // System prompt should be used, but the system message in the middle should be ignored
    // (only leading consecutive system messages are collected)
    try testing.expect(std.mem.indexOf(u8, result, "You are helpful.") != null);
    try testing.expect(std.mem.indexOf(u8, result, "This should be ignored") == null);
}

test "gemma4: no generation prompt" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, null, false);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        BOS ++ "<|turn>user\nHello<turn|>\n",
        result,
    );
}

test "gemma4: user message with image media (inline marker)" {
    // Official template: {{- '<|image|>' -}}{{ item['text'] | trim }}
    // The "-" modifier strips whitespace → marker is INLINE with text
    const media = types.Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 100, .height = 100 } },
    };
    const messages = [_]ChatMessage{
        ChatMessage.withMedia("user", "Describe this image", media),
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    // Media marker is INLINE: <|image|>Describe this image (no newline between)
    try testing.expectEqualStrings(
        BOS ++ "<|turn>user\n<|image|>Describe this image<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: user message with audio media (inline marker)" {
    // Official template: {{- '<|audio|>' -}}{{ item['text'] | trim }}
    const media = types.Media{
        .type = .audio,
        .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 16000 } },
    };
    const messages = [_]ChatMessage{
        ChatMessage.withMedia("user", "Transcribe the audio", media),
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    // Audio marker is INLINE: <|audio|>Transcribe the audio (no newline between)
    try testing.expectEqualStrings(
        BOS ++ "<|turn>user\n<|audio|>Transcribe the audio<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}
