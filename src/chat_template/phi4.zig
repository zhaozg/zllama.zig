//! Phi-4 template
//!
//! Format: <|im_start|>system\n{system}<|im_end|>\n
//!         <|im_start|>user\n{user}<|im_sep|>\n
//!         <|im_start|>assistant\n{assistant}<|im_end|>\n
//!         <|im_start|>assistant\n   ← generation prompt
//!
//! Note: Phi-4 is similar to ChatML but uses <|im_sep|> instead of <|im_end|>
//!       for user messages. This allows the model to distinguish between
//!       user input boundaries and assistant output boundaries.

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply Phi-4 template.
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
            try buf.appendSlice(allocator, "<|im_start|>system\n");
            try buf.appendSlice(allocator, sp);
            try buf.appendSlice(allocator, "<|im_end|>\n");
        }
    }

    // Messages
    for (messages) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            try buf.appendSlice(allocator, "<|im_start|>system\n");
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "<|im_end|>\n");
        } else if (std.mem.eql(u8, msg.role, "user")) {
            try buf.appendSlice(allocator, "<|im_start|>user\n");
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "<|im_sep|>\n");
        } else if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model")) {
            try buf.appendSlice(allocator, "<|im_start|>assistant\n");
            try buf.appendSlice(allocator, msg.content);
            try buf.appendSlice(allocator, "<|im_end|>\n");
        }
    }

    // Generation prompt
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<|im_start|>assistant\n");
    }

    return buf.toOwnedSlice(allocator);
}

const testing = std.testing;

test "phi4: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_sep|>\n<|im_start|>assistant\n",
        result,
    );
}

test "phi4: with system" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_sep|>\n<|im_start|>assistant\n",
        result,
    );
}
