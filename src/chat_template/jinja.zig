//! Jinja template engine integration for chat templates.
//!
//! Wraps vibe_jinja (deps/zig-jinja) to render GGUF built-in Jinja chat templates.
//! Converts zllama ChatMessage types to vibe_jinja Value types (Dict/List),
//! and passes special variables (bos_token, eos_token, add_generation_prompt).
//!
//! Supports multimodal messages: when a message has media attached, the content
//! field is augmented with the appropriate placeholder marker (<|image|>/<|audio|>)
//! so that the Jinja template renders it correctly. After rendering, the
//! multimodal module's tokenizeWithPlaceholders() expands the markers into
//! actual placeholder tokens for embedding injection.
//!
//! Reference: llama.cpp common/jinja/ + deps/zig-jinja/

const std = @import("std");
const vibe_jinja = @import("vibe_jinja");
const types = @import("types");

const log = std.log.scoped(.chat_template_jinja);

/// Render a Jinja chat template with the given messages and special tokens.
///
/// Arguments:
/// - allocator: Memory allocator (caller owns returned string)
/// - template_str: Jinja template source string (from GGUF tokenizer.chat_template)
/// - messages: Array of chat messages to format
/// - bos_token: Beginning-of-sequence token string (e.g. "<|begin_of_text|>")
/// - eos_token: End-of-sequence token string (e.g. "<|eot_id|>")
/// - add_generation_prompt: If true, append the assistant prefix
///
/// Returns the rendered template string. Caller must free with allocator.free().
/// On error, returns null (caller should fall back to preset templates).
pub fn render(
    allocator: std.mem.Allocator,
    template_str: []const u8,
    messages: []const types.ChatMessage,
    bos_token: []const u8,
    eos_token: []const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    // Step 1: Convert messages to vibe_jinja Value types
    // For multimodal messages, ensure content has placeholder markers
    const messages_value = try messagesToList(allocator, messages);
    defer messages_value.deinit(allocator);

    // Step 2: Set up the jinja environment and runtime
    var env = vibe_jinja.Environment.init(allocator);
    defer env.deinit();

    var rt = vibe_jinja.runtime.Runtime.init(&env, allocator);
    defer rt.deinit();

    // Step 3: Build variables map
    var vars = std.StringHashMap(vibe_jinja.Value).init(allocator);
    defer {
        var iter = vars.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(allocator);
            allocator.free(entry.key_ptr.*);
        }
        vars.deinit();
    }

    // messages: list of dicts
    try vars.put(try allocator.dupe(u8, "messages"), vibe_jinja.Value{ .list = messages_value.list_ptr });

    // bos_token: optional string
    try vars.put(try allocator.dupe(u8, "bos_token"), vibe_jinja.Value{ .string = try allocator.dupe(u8, bos_token) });

    // eos_token: optional string
    try vars.put(try allocator.dupe(u8, "eos_token"), vibe_jinja.Value{ .string = try allocator.dupe(u8, eos_token) });

    // add_generation_prompt: boolean
    try vars.put(try allocator.dupe(u8, "add_generation_prompt"), vibe_jinja.Value{ .boolean = add_generation_prompt });

    // Step 4: Render
    const result = rt.renderString(template_str, vars, "chat_template") catch |err| {
        log.warn("Jinja render failed: {s}, falling back to preset", .{@errorName(err)});
        return error.JinjaRenderFailed;
    };

    return result;
}

/// Result of messagesToList - holds allocated list pointer.
const MessagesListResult = struct {
    list_ptr: *vibe_jinja.value.List,

    fn deinit(self: *const MessagesListResult, allocator: std.mem.Allocator) void {
        self.list_ptr.deinit(allocator);
    }
};

/// Convert zllama ChatMessage array to vibe_jinja value.List of value.Dict.
/// For multimodal messages, the content field is augmented with the appropriate
/// placeholder marker (<|image|>/<|audio|>) so that the Jinja template renders it.
/// The marker is prepended to the content text.
fn messagesToList(allocator: std.mem.Allocator, messages: []const types.ChatMessage) !MessagesListResult {
    const list_ptr = try allocator.create(vibe_jinja.value.List);
    errdefer allocator.destroy(list_ptr);
    list_ptr.* = vibe_jinja.value.List.init(allocator);

    for (messages) |msg| {
        const dict_ptr = try allocator.create(vibe_jinja.value.Dict);
        errdefer allocator.destroy(dict_ptr);
        dict_ptr.* = vibe_jinja.value.Dict.init(allocator);

        try dict_ptr.set("role", vibe_jinja.Value{ .string = try allocator.dupe(u8, msg.role) });

        // For multimodal messages, ensure the content includes the placeholder marker.
        // The marker is prepended to the text content so the Jinja template renders it
        // as part of the message content. After rendering, the multimodal module's
        // tokenizeWithPlaceholders() will scan for these markers and expand them.
        const content_to_use = if (msg.media) |media| blk: {
            // Only add placeholder if not already present in content
            if (types.containsPlaceholder(msg.content)) {
                break :blk msg.content;
            }
            const placeholder = switch (media.type) {
                .image => types.IMAGE_PLACEHOLDER,
                .audio => types.AUDIO_PLACEHOLDER,
                .none => msg.content,
            };
            if (placeholder.len == 0 or placeholder.ptr == msg.content.ptr) {
                break :blk msg.content;
            }
            // Allocate new string with placeholder prepended
            const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ placeholder, msg.content });
            break :blk combined;
        } else msg.content;

        try dict_ptr.set("content", vibe_jinja.Value{ .string = try allocator.dupe(u8, content_to_use) });

        // Free the temporary combined string if we allocated one
        if (content_to_use.ptr != msg.content.ptr) {
            allocator.free(content_to_use);
        }

        try list_ptr.append(vibe_jinja.Value{ .dict = dict_ptr });
    }

    return MessagesListResult{ .list_ptr = list_ptr };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "jinja: simple variable" {
    const allocator = testing.allocator;
    const template = "Hello, {{ name }}!";
    var env = vibe_jinja.Environment.init(allocator);
    defer env.deinit();

    var rt = vibe_jinja.runtime.Runtime.init(&env, allocator);
    defer rt.deinit();

    var vars = std.StringHashMap(vibe_jinja.Value).init(allocator);
    defer {
        var iter = vars.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(allocator);
            allocator.free(entry.key_ptr.*);
        }
        vars.deinit();
    }

    try vars.put(try allocator.dupe(u8, "name"), vibe_jinja.Value{ .string = try allocator.dupe(u8, "World") });

    const result = try rt.renderString(template, vars, "test");
    defer allocator.free(result);

    try testing.expectEqualStrings("Hello, World!", result);
}

test "jinja: if/else" {
    const allocator = testing.allocator;
    const template =
        \\{% if show %}
        \\visible
        \\{% else %}
        \\hidden
        \\{% endif %}
    ;

    var env = vibe_jinja.Environment.init(allocator);
    defer env.deinit();

    var rt = vibe_jinja.runtime.Runtime.init(&env, allocator);
    defer rt.deinit();

    // Test true
    {
        var vars = std.StringHashMap(vibe_jinja.Value).init(allocator);
        defer {
            var iter = vars.iterator();
            while (iter.next()) |entry| {
                entry.value_ptr.deinit(allocator);
                allocator.free(entry.key_ptr.*);
            }
            vars.deinit();
        }
        try vars.put(try allocator.dupe(u8, "show"), vibe_jinja.Value{ .boolean = true });
        const result = try rt.renderString(template, vars, "test");
        defer allocator.free(result);
        try testing.expect(std.mem.containsAtLeast(u8, result, 1, "visible"));
    }

    // Test false
    {
        var vars = std.StringHashMap(vibe_jinja.Value).init(allocator);
        defer {
            var iter = vars.iterator();
            while (iter.next()) |entry| {
                entry.value_ptr.deinit(allocator);
                allocator.free(entry.key_ptr.*);
            }
            vars.deinit();
        }
        try vars.put(try allocator.dupe(u8, "show"), vibe_jinja.Value{ .boolean = false });
        const result = try rt.renderString(template, vars, "test");
        defer allocator.free(result);
        try testing.expect(std.mem.containsAtLeast(u8, result, 1, "hidden"));
    }
}

test "jinja: for loop" {
    const allocator = testing.allocator;
    const template =
        \\{% for item in items %}
        \\{{ item }}
        \\{% endfor %}
    ;

    var env = vibe_jinja.Environment.init(allocator);
    defer env.deinit();

    var rt = vibe_jinja.runtime.Runtime.init(&env, allocator);
    defer rt.deinit();

    var vars = std.StringHashMap(vibe_jinja.Value).init(allocator);
    defer {
        var iter = vars.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(allocator);
            allocator.free(entry.key_ptr.*);
        }
        vars.deinit();
    }

    const list_ptr = try allocator.create(vibe_jinja.value.List);
    list_ptr.* = vibe_jinja.value.List.init(allocator);
    try list_ptr.append(vibe_jinja.Value{ .string = try allocator.dupe(u8, "a") });
    try list_ptr.append(vibe_jinja.Value{ .string = try allocator.dupe(u8, "b") });
    try list_ptr.append(vibe_jinja.Value{ .string = try allocator.dupe(u8, "c") });
    try vars.put(try allocator.dupe(u8, "items"), vibe_jinja.Value{ .list = list_ptr });

    const result = try rt.renderString(template, vars, "test");
    defer allocator.free(result);

    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "a"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "b"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "c"));
}

test "jinja: chatml render" {
    const allocator = testing.allocator;
    const template =
        \\{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
    ;

    const messages = [_]types.ChatMessage{
        types.ChatMessage.init("user", "Hello"),
    };

    const result = try render(allocator, template, &messages, "", "<|im_end|>", true);
    defer allocator.free(result);

    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|im_start|>user"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "Hello"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|im_start|>assistant"));
}

test "jinja: llama3 render" {
    const allocator = testing.allocator;
    const template =
        \\{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}
    ;

    const messages = [_]types.ChatMessage{
        types.ChatMessage.init("user", "Hello"),
    };

    const result = try render(allocator, template, &messages, "<|begin_of_text|>", "<|eot_id|>", true);
    defer allocator.free(result);

    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|begin_of_text|>"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|start_header_id|>user"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "Hello"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|start_header_id|>assistant"));
}

test "jinja: multimodal image placeholder in content" {
    const allocator = testing.allocator;
    const template =
        \\{% for message in messages %}{{ message['content'] }}{% endfor %}
    ;

    // Create a message with image media
    const media = types.Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 100, .height = 100 } },
    };
    const messages = [_]types.ChatMessage{
        types.ChatMessage.withMedia("user", "Describe this image", media),
    };

    const result = try render(allocator, template, &messages, "", "", false);
    defer allocator.free(result);

    // The content should have the image placeholder prepended
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|image|>"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "Describe this image"));
}

test "jinja: multimodal audio placeholder in content" {
    const allocator = testing.allocator;
    const template =
        \\{% for message in messages %}{{ message['content'] }}{% endfor %}
    ;

    // Create a message with audio media
    const media = types.Media{
        .type = .audio,
        .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 16000 } },
    };
    const messages = [_]types.ChatMessage{
        types.ChatMessage.withMedia("user", "Transcribe this", media),
    };

    const result = try render(allocator, template, &messages, "", "", false);
    defer allocator.free(result);

    // The content should have the audio placeholder prepended
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|audio|>"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "Transcribe this"));
}

test "jinja: multimodal with existing placeholder" {
    const allocator = testing.allocator;
    const template =
        \\{% for message in messages %}{{ message['content'] }}{% endfor %}
    ;

    // Content already has placeholder - should not duplicate
    const media = types.Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 100, .height = 100 } },
    };
    const messages = [_]types.ChatMessage{
        types.ChatMessage.withMedia("user", "<|image|>Describe this", media),
    };

    const result = try render(allocator, template, &messages, "", "", false);
    defer allocator.free(result);

    // Should only have one placeholder
    try testing.expectEqualStrings("<|image|>Describe this", result);
}

test "jinja: multimodal chatml render with image" {
    const allocator = testing.allocator;
    const template =
        \\{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
    ;

    const media = types.Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 100, .height = 100 } },
    };
    const messages = [_]types.ChatMessage{
        types.ChatMessage.withMedia("user", "What's in this image?", media),
    };

    const result = try render(allocator, template, &messages, "", "<|im_end|>", true);
    defer allocator.free(result);

    // The rendered template should contain the placeholder in the user message content
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|im_start|>user"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|image|>"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "What's in this image?"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|im_end|>"));
    try testing.expect(std.mem.containsAtLeast(u8, result, 1, "<|im_start|>assistant"));
}
