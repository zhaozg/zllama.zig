//! Minja C++ bridge — Zig bindings
//!
//! Wraps the C ABI bridge (src/vendor/minja/bridge.cpp) around Google minja
//! for Jinja2 chat template rendering.
//!
//! The C++ bridge provides:
//! - `minja_chat_template_create/destroy` — parse and hold a template
//! - `minja_chat_template_apply` — render messages+options into a prompt
//! - `minja_render` — raw Jinja rendering with a JSON context
//!
//! Reference: deps/chat-template.cpp, docs/MINJA.md

const std = @import("std");

/// C bridge declarations — only used inside this module.
const c = @cImport({
    @cInclude("bridge.hpp");
});

/// Opaque handle to a parsed chat template.
pub const ChatTemplate = opaque {
    /// Create a chat_template from a Jinja source string.
    /// source: Jinja template string (e.g., from GGUF tokenizer.chat_template)
    /// bos_token: beginning-of-sequence token string
    /// eos_token: end-of-sequence token string
    /// Returns error if parsing fails.
    pub fn create(
        source: [:0]const u8,
        bos_token: [:0]const u8,
        eos_token: [:0]const u8,
    ) !*ChatTemplate {
        const ptr = c.minja_chat_template_create(source.ptr, bos_token.ptr, eos_token.ptr);
        if (ptr == null) {
            return error.MinjaParseError;
        }
        return @ptrCast(ptr);
    }

    /// Destroy a chat_template instance.
    pub fn destroy(self: *ChatTemplate) void {
        c.minja_chat_template_free(@ptrCast(self));
    }

    /// Apply the template to messages.
    /// messages_json: JSON array of message objects
    /// tools_json: JSON array of tool definitions, or null for none
    /// add_generation_prompt: whether to append generation prompt marker
    /// Returns the formatted prompt string. Caller owns the returned memory.
    pub fn apply(
        self: *const ChatTemplate,
        allocator: std.mem.Allocator,
        messages_json: [:0]const u8,
        tools_json: ?[:0]const u8,
        add_generation_prompt: bool,
    ) ![]u8 {
        const tools_ptr: [*c]const u8 = if (tools_json) |t| t.ptr else null;
        const result = c.minja_chat_template_apply(
            @ptrCast(self),
            messages_json.ptr,
            tools_ptr,
            add_generation_prompt,
        );
        if (result == null) {
            return error.MinjaRenderError;
        }
        const len = std.mem.len(result);
        // Copy to Zig-managed memory
        const copy = try allocator.alloc(u8, len);
        @memcpy(copy, result[0..len]);
        c.minja_free_string(result);
        return copy;
    }
};

/// Render a raw Jinja template with a JSON context.
/// No message normalization is performed — this is a direct Jinja render.
/// Returns the rendered string. Caller owns the returned memory.
pub fn render(
    allocator: std.mem.Allocator,
    template_str: [:0]const u8,
    context_json: [:0]const u8,
) ![]u8 {
    const result = c.minja_render(template_str.ptr, context_json.ptr);
    if (result == null) {
        return error.MinjaRenderError;
    }
    const len = std.mem.len(result);
    const copy = try allocator.alloc(u8, len);
    @memcpy(copy, result[0..len]);
    c.minja_free_string(result);
    return copy;
}

// ============================================================================
// Higher-level helpers for chat_template integration
// ============================================================================

/// ChatMessage type for use with applyTemplate.
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

/// Convert an array of ChatMessage into a JSON string suitable
/// for minja_chat_template_apply.
/// The output is a JSON array of {role, content} objects.
pub fn messagesToJson(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
) ![:0]u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.append(allocator, '[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try jsonAppendMessage(&buf, allocator, msg.role, msg.content);
    }
    try buf.append(allocator, ']');
    try buf.append(allocator, 0);
    const owned = try buf.toOwnedSlice(allocator);
    return owned[0 .. owned.len - 1 :0]; // exclude null from returned slice
}

/// Shorthand: apply a chat template directly from ChatMessage array.
/// If system_prompt is non-null, a system message is prepended.
/// Returns the formatted prompt string. Caller owns the returned memory.
pub fn applyTemplate(
    allocator: std.mem.Allocator,
    tmpl: *const ChatTemplate,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    tools_json: ?[:0]const u8,
    add_generation_prompt: bool,
) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.append(allocator, '[');

    var first = true;
    if (system_prompt) |sp| {
        try jsonAppendMessage(&buf, allocator, "system", sp);
        first = false;
    }
    for (messages) |msg| {
        if (!first) try buf.append(allocator, ',');
        try jsonAppendMessage(&buf, allocator, msg.role, msg.content);
        first = false;
    }
    try buf.append(allocator, ']');
    try buf.append(allocator, 0);
    const msgs_json: [:0]const u8 = buf.items[0 .. buf.items.len - 1 :0];

    return tmpl.apply(allocator, msgs_json, tools_json, add_generation_prompt);
}

// ============================================================================
// JSON helpers (minimal, no external dependency)
// ============================================================================

fn jsonAppendMessage(buf: *std.ArrayListAligned(u8, null), allocator: std.mem.Allocator, role: []const u8, content: []const u8) !void {
    try buf.append(allocator, '{');
    try jsonAppendString(buf, allocator, "role");
    try buf.append(allocator, ':');
    try jsonAppendString(buf, allocator, role);
    try buf.append(allocator, ',');
    try jsonAppendString(buf, allocator, "content");
    try buf.append(allocator, ':');
    try jsonAppendString(buf, allocator, content);
    try buf.append(allocator, '}');
}

fn jsonAppendString(buf: *std.ArrayListAligned(u8, null), allocator: std.mem.Allocator, s: []const u8) !void {
    try buf.append(allocator, '"');
    for (s) |ch| {
        switch (ch) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                try buf.appendSlice(allocator, "\\u00");
                const hex = "0123456789abcdef";
                try buf.append(allocator, hex[ch >> 4]);
                try buf.append(allocator, hex[ch & 0xF]);
            },
            else => try buf.append(allocator, ch),
        }
    }
    try buf.append(allocator, '"');
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "messagesToJson: single message" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
    };
    const json_str = try messagesToJson(testing.allocator, &messages);
    defer testing.allocator.free(json_str);
    try testing.expectEqualStrings(
        "[{\"role\":\"user\",\"content\":\"Hello\"}]",
        json_str,
    );
}

test "messagesToJson: multiple messages" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there!" },
    };
    const json_str = try messagesToJson(testing.allocator, &messages);
    defer testing.allocator.free(json_str);
    try testing.expectEqualStrings(
        "[{\"role\":\"user\",\"content\":\"Hello\"},{\"role\":\"assistant\",\"content\":\"Hi there!\"}]",
        json_str,
    );
}

test "messagesToJson: empty" {
    const messages: [0]ChatMessage = .{};
    const json_str = try messagesToJson(testing.allocator, &messages);
    defer testing.allocator.free(json_str);
    try testing.expectEqualStrings("[]", json_str);
}

test "messagesToJson: special characters" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello\n\"World\"" },
    };
    const json_str = try messagesToJson(testing.allocator, &messages);
    defer testing.allocator.free(json_str);
    try testing.expectEqualStrings(
        "[{\"role\":\"user\",\"content\":\"Hello\\n\\\"World\\\"\"}]",
        json_str,
    );
}

test "jsonAppendString: basic" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(testing.allocator);
    try jsonAppendString(&buf, testing.allocator, "hello");
    try testing.expectEqualStrings("\"hello\"", buf.items);
}

test "jsonAppendString: with escapes" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(testing.allocator);
    try jsonAppendString(&buf, testing.allocator, "a\"b\\c\nd");
    try testing.expectEqualStrings("\"a\\\"b\\\\c\\nd\"", buf.items);
}
