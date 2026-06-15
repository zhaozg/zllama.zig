//! DeepSeek V3 template
//!
//! Format: <｜User｜>{user}<｜Assistant｜>
//!         <｜User｜>{user}<｜Assistant｜>{assistant}<｜end▁of▁sentence｜>
//!         <｜User｜>{user}<｜Assistant｜>   ← generation prompt
//!
//! Note: DeepSeek V3 uses UTF-8 full-width angle brackets and special
//!       Unicode characters for its tags. The tags are:
//!       - <｜User｜>  (U+FF5C full-width vertical bar)
//!       - <｜Assistant｜>
//!       - <｜end▁of▁sentence｜>

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply DeepSeek V3 template.
pub fn apply(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    // System message: DeepSeek V3 doesn't have a separate system role.
    // If system prompt is provided, prepend it to the first user message.
    var system_content: std.ArrayListUnmanaged(u8) = .empty;
    defer system_content.deinit(allocator);

    if (system_prompt) |sp| {
        if (sp.len > 0) {
            try system_content.appendSlice(allocator, sp);
            try system_content.appendSlice(allocator, "\n\n");
        }
    }

    var system_merged = false;

    for (messages) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            if (system_content.items.len > 0) {
                try system_content.appendSlice(allocator, "\n\n");
            }
            try system_content.appendSlice(allocator, msg.content);
            continue;
        }

        if (std.mem.eql(u8, msg.role, "user")) {
            try buf.appendSlice(allocator, "<｜User｜>");

            // Merge system content into the first user message
            if (!system_merged and system_content.items.len > 0) {
                try buf.appendSlice(allocator, system_content.items);
                system_merged = true;
            }

            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "<｜Assistant｜>");
        } else if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model")) {
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "<｜end▁of▁sentence｜>");
        }
    }

    // Generation prompt: already ends with <｜Assistant｜> from the last user message
    // If there are no user messages, add the generation prompt marker
    if (add_generation_prompt) {
        // Check if we already ended with <｜Assistant｜>
        const ends_with_assistant = std.mem.endsWith(u8, buf.items, "<｜Assistant｜>");
        if (!ends_with_assistant) {
            try buf.appendSlice(allocator, "<｜Assistant｜>");
        }
    }

    return buf.toOwnedSlice(allocator);
}

const testing = std.testing;

test "deepseek3: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<｜User｜>Hello<｜Assistant｜>",
        result,
    );
}

test "deepseek3: with system" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<｜User｜>You are helpful.\n\nHello<｜Assistant｜>",
        result,
    );
}

test "deepseek3: multi-turn" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "assistant", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<｜User｜>Hi<｜Assistant｜>Hello!<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>",
        result,
    );
}
