//! Gemma (Gemma 3) template
//!
//! Format: <start_of_turn>user\n{user}<end_of_turn>\n
//!         <start_of_turn>model\n{assistant}<end_of_turn>\n
//!         <start_of_turn>model\n   ← generation prompt
//!
//! Note: Gemma uses "model" instead of "assistant".
//!       There is no explicit system role; system messages are merged
//!       into the first user turn (same as llama.cpp behavior).

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply Gemma template.
pub fn apply(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
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

    var system_merged = false;

    for (messages) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            if (system_content.items.len > 0) {
                try system_content.appendSlice(allocator, "\n\n");
            }
            try system_content.appendSlice(allocator, msg.content);
            continue;
        }

        const role_tag = if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model"))
            "model"
        else
            msg.role;

        try buf.appendSlice(allocator, "<start_of_turn>");
        try buf.appendSlice(allocator, role_tag);
        try buf.appendSlice(allocator, "\n");

        // Merge system content into the first non-system, non-model message
        if (!system_merged and
            !std.mem.eql(u8, msg.role, "assistant") and
            !std.mem.eql(u8, msg.role, "model") and
            system_content.items.len > 0)
        {
            try buf.appendSlice(allocator, system_content.items);
            try buf.appendSlice(allocator, "\n\n");
            system_merged = true;
        }

        try buf.appendSlice(allocator, msg.content);
        try buf.appendSlice(allocator, "<end_of_turn>\n");
    }

    // Generation prompt
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<start_of_turn>model\n");
    }

    return buf.toOwnedSlice(allocator);
}

const testing = std.testing;

test "gemma: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\n1+1=?<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}

test "gemma: with system merged" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are a helpful assistant.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nYou are a helpful assistant.\n\nHello<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}

test "gemma: multi-turn" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "model", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nHow are you?<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}
