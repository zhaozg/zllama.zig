//! TinyLlama template
//!
//! Format: <|system|>
//! {system}</s>
//! <|user|>
//! {user}</s>
//! <|assistant|>
//! {assistant}</s>
//! <|assistant|>

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply TinyLlama template.
pub fn apply(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    // System message
    if (system_prompt) |sp| {
        if (sp.len > 0) {
            try buf.appendSlice(allocator, "<|system|>\n");
            try buf.appendSlice(allocator, sp);
            try buf.appendSlice(allocator, "</s>\n");
        }
    }

    // Messages
    for (messages) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            try buf.appendSlice(allocator, "<|system|>\n");
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "</s>\n");
        } else if (std.mem.eql(u8, msg.role, "user")) {
            try buf.appendSlice(allocator, "<|user|>\n");
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "</s>\n");
        } else if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model")) {
            try buf.appendSlice(allocator, "<|assistant|>\n");
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "</s>\n");
        }
    }

    // Generation prompt
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<|assistant|>\n");
    }

    return buf.toOwnedSlice(allocator);
}

const testing = std.testing;

test "tinyllama: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|user|>\n1+1=?</s>\n<|assistant|>\n",
        result,
    );
}

test "tinyllama: with system" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|system|>\nYou are helpful.</s>\n<|user|>\nHello</s>\n<|assistant|>\n",
        result,
    );
}
