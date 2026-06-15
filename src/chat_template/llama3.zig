//! Llama 3 template
//!
//! Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
//!         <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
//!         <|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>
//!         <|start_header_id|>assistant<|end_header_id|>\n\n   ← generation prompt

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply Llama 3 template.
pub fn apply(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    // BOS token
    try buf.appendSlice(allocator, "<|begin_of_text|>");

    // System message
    if (system_prompt) |sp| {
        if (sp.len > 0) {
            try buf.appendSlice(allocator, "<|start_header_id|>system<|end_header_id|>\n\n");
            try buf.appendSlice(allocator, sp);
            try buf.appendSlice(allocator, "<|eot_id|>");
        }
    }

    // Messages
    for (messages) |msg| {
        const role_tag = if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model"))
            "assistant"
        else
            msg.role;

        try buf.appendSlice(allocator, "<|start_header_id|>");
        try buf.appendSlice(allocator, role_tag);
        try buf.appendSlice(allocator, "<|end_header_id|>\n\n");
        try buf.appendSlice(allocator, msg.content);
        try buf.appendSlice(allocator, "<|eot_id|>");
    }

    // Generation prompt
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    return buf.toOwnedSlice(allocator);
}

const testing = std.testing;

test "llama3: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        result,
    );
}
