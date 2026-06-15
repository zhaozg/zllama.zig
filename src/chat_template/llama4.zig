//! Llama 4 template
//!
//! Format: <|begin_of_text|><|header_start|>system<|header_end|>\n\n{system}<|eom_id|>
//!         <|header_start|>user<|header_end|>\n\n{user}<|eom_id|>
//!         <|header_start|>assistant<|header_end|>\n\n{assistant}<|eom_id|>
//!         <|header_start|>assistant<|header_end|>\n\n   ← generation prompt
//!
//! Note: Llama 4 uses <|header_start|>/<|header_end|> instead of
//!       <|start_header_id|>/<|end_header_id|>, and <|eom_id|> instead of <|eot_id|>.

const std = @import("std");
const types = @import("types");

const ChatMessage = types.ChatMessage;

/// Apply Llama 4 template.
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
            try buf.appendSlice(allocator, "<|header_start|>system<|header_end|>\n\n");
            try buf.appendSlice(allocator, sp);
            try buf.appendSlice(allocator, "<|eom_id|>");
        }
    }

    // Messages
    for (messages) |msg| {
        const role_tag = if (std.mem.eql(u8, msg.role, "assistant") or std.mem.eql(u8, msg.role, "model"))
            "assistant"
        else
            msg.role;

        try buf.appendSlice(allocator, "<|header_start|>");
        try buf.appendSlice(allocator, role_tag);
        try buf.appendSlice(allocator, "<|header_end|>\n\n");
        try buf.appendSlice(allocator, msg.content);
        try buf.appendSlice(allocator, "<|eom_id|>");
    }

    // Generation prompt
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, "<|header_start|>assistant<|header_end|>\n\n");
    }

    return buf.toOwnedSlice(allocator);
}

const testing = std.testing;

test "llama4: single turn" {
    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|begin_of_text|><|header_start|>user<|header_end|>\n\nHello<|eom_id|><|header_start|>assistant<|header_end|>\n\n",
        result,
    );
}
