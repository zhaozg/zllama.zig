//! Gemma 4 template
//!
//! Format: <|turn>system\n{system}<turn|>\n   ← system turn (when system present)
//!         <|turn>user\n{user}<turn|>\n
//!         <|turn>model\n{assistant}<turn|>\n
//!         <|turn>model\n<|channel>thought\n<channel|>   ← generation prompt
//!
//! Reference: google-gemma-4-31B-it-interleaved.jinja
//!
//! Key differences from Gemma 3:
//!   - Uses <|turn> / <turn|> delimiters (instead of <start_of_turn> / <end_of_turn>)
//!   - System messages get their own <|turn>system\n turn (not merged into user)
//!   - Model output may contain <|channel>thought\n...<channel|> thinking blocks
//!   - Generation prompt: <|turn>model\n + empty thinking block to suppress reasoning
//!   - When enable_thinking=true: system turn gets <|think|>\n, no empty thinking block

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply Gemma 4 template.
pub fn apply(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    // Default: thinking disabled (injects empty thinking block to suppress reasoning)
    const enable_thinking = false;

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    // Collect system content (either from explicit system_prompt or from messages)
    var system_content: std.ArrayListUnmanaged(u8) = .empty;
    defer system_content.deinit(allocator);

    if (system_prompt) |sp| {
        if (sp.len > 0) {
            try system_content.appendSlice(allocator, sp);
        }
    }

    // Collect system messages from the messages array
    var system_msg_indices: std.ArrayListUnmanaged(usize) = .empty;
    defer system_msg_indices.deinit(allocator);

    for (messages, 0..) |msg, i| {
        if (std.mem.eql(u8, msg.role, "system")) {
            try system_msg_indices.append(allocator, i);
            if (system_content.items.len > 0) {
                try system_content.appendSlice(allocator, "\n\n");
            }
            try system_content.appendSlice(allocator, msg.content);
        }
    }

    const has_system = system_content.items.len > 0;

    // --- System turn (Gemma 4: dedicated turn, not merged into user) ---
    if (has_system) {
        try buf.appendSlice(allocator, "<|turn>system\n");

        if (enable_thinking) {
            try buf.appendSlice(allocator, "<|think|>\n");
        }

        try buf.appendSlice(allocator, system_content.items);
        try buf.appendSlice(allocator, "<turn|>\n");
    }

    // --- Message turns ---
    // Track system message indices to skip them (they were handled in the system turn)
    const skip_indices = system_msg_indices.items;

    for (messages, 0..) |msg, i| {
        // Skip system messages (handled in system turn)
        var is_system = false;
        for (skip_indices) |si| {
            if (si == i) {
                is_system = true;
                break;
            }
        }
        if (is_system) continue;

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
    }

    // --- Generation prompt ---
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<|turn>model\n");
        if (!enable_thinking) {
            // Inject empty thinking block to suppress model reasoning
            try buf.appendSlice(allocator, "<|channel>thought\n<channel|>");
        }
    }

    return buf.toOwnedSlice(allocator);
}

/// Strip thinking blocks (<|channel>thought\n...<channel|>) from model content.
/// Reference: strip_thinking macro in google-gemma-4-31B-it-interleaved.jinja
fn appendStrippedThinking(
    buf: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    text: []const u8,
) !void {
    const channel_start_tag = "<|channel>";
    const channel_end_tag = "<channel|>";

    var remaining = text;
    while (remaining.len > 0) {
        if (std.mem.indexOf(u8, remaining, channel_start_tag)) |tag_pos| {
            // Output everything before the channel start tag
            try buf.appendSlice(allocator, remaining[0..tag_pos]);

            // Find the matching channel end tag
            const after_start = remaining[tag_pos + channel_start_tag.len ..];
            if (std.mem.indexOf(u8, after_start, channel_end_tag)) |end_pos| {
                // Skip the entire channel block
                remaining = after_start[end_pos + channel_end_tag.len ..];
            } else {
                // No closing tag found — output the rest as-is
                try buf.appendSlice(allocator, remaining[tag_pos..]);
                break;
            }
        } else {
            // No more channel tags
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
    try testing.expectEqualStrings(
        "<|turn>user\n1+1=?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}

test "gemma4: with system" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are a helpful assistant.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|turn>system\nYou are a helpful assistant.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
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
        "<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
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
        "<|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
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
    try testing.expectEqualStrings(
        "<|turn>user\n1+1=?<turn|>\n<|turn>model\nThe answer is 2.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
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
        "<|turn>system\nYou are a math tutor.<turn|>\n<|turn>user\n1+1=?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        result,
    );
}
