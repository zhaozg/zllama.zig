//! Mistral v7 template
//!
//! Format: [INST] {user} [/INST]
//!         [INST] {system}\n\n{user} [/INST]
//!         {assistant}</s>
//!         [INST] {user} [/INST]
//!         ← generation prompt (no special marker, just empty)
//!
//! Note: Mistral v7 uses [INST] tags for user messages.
//!       Assistant responses are plain text followed by </s>.
//!       System prompt is prepended to the first user message.

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply Mistral v7 template.
pub fn apply(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    // Collect system content
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

        if (std.mem.eql(u8, msg.role, "user")) {
            try buf.appendSlice(allocator, "[INST] ");

            // Merge system content into the first user message
            if (!system_merged and system_content.items.len > 0) {
                try buf.appendSlice(allocator, system_content.items);
                try buf.appendSlice(allocator, "\n\n");
                system_merged = true;
            }

            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, " [/INST]");
        } else if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model")) {
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "</s>");
        }
    }

    // Generation prompt: Mistral doesn't have a special generation marker,
    // but we add a space to indicate the assistant should start speaking
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, " ");
    }

    return buf.toOwnedSlice(allocator);
}

const testing = std.testing;

test "mistral_v7: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "[INST] Hello [/INST] ",
        result,
    );
}

test "mistral_v7: with system" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "[INST] You are helpful.\n\nHello [/INST] ",
        result,
    );
}

test "mistral_v7: multi-turn" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "assistant", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "[INST] Hi [/INST]Hello!</s>[INST] How are you? [/INST] ",
        result,
    );
}
