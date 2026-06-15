//! Gemma 4 template
//!
//! Format: <|turn>system\n{system}<turn|>\n   ← system turn (when system present)
//!         <|turn>user\n{user}<turn|>\n
//!         <|turn>model\n{assistant}<turn|>\n
//!         <|turn>model\n   ← generation prompt
//!
//! Reference: google-gemma-4-31B-it-interleaved.jinja (GGUF built-in template)
//!
//! Key differences from Gemma 3:
//!   - Uses <|turn> / <turn|> delimiters (instead of <start_of_turn> / <end_of_turn>)
//!   - System messages get their own <|turn>system\n turn (not merged into user)
//!   - Model output may contain <|channel>thought\n...<channel|> thinking blocks
//!   - Generation prompt: just <|turn>model\n (no empty thinking block)
//!   - When enable_thinking=true: system turn gets <|think|>\n prefix

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

        // For model messages, strip thinking blocks before rendering
        if (std.mem.eql(u8, role_tag, "model")) {
            try appendStrippedThinking(&buf, allocator, msg.content);
        } else {
            try buf.appendSlice(allocator, msg.content);
        }

        try buf.appendSlice(allocator, "<turn|>\n");
        _ = i;
    }

    // --- Generation prompt ---
    // Official template: just <|turn>model\n (no empty thinking block)
    // The empty thinking block was a previous attempt to suppress reasoning
    // but it causes the model to output "thought" as the first token.
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<|turn>model\n");
    }

    return buf.toOwnedSlice(allocator);
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

test "gemma4: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    // Generation prompt is just <|turn>model\n (no empty thinking block)
    try testing.expectEqualStrings(
        "<|turn>user\n1+1=?<turn|>\n<|turn>model\n",
        result,
    );
}

test "gemma4: with system" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are a helpful assistant.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|turn>system\nYou are a helpful assistant.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n",
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
        "<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n",
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
        "<|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n",
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
        "<|turn>user\n1+1=?<turn|>\n<|turn>model\nThe answer is 2.<turn|>\n<|turn>model\n",
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
        "<|turn>system\nYou are a math tutor.<turn|>\n<|turn>user\n1+1=?<turn|>\n<|turn>model\n",
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
        "<|turn>user\nHello<turn|>\n",
        result,
    );
}
