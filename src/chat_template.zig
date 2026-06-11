//! Chat template support for zllama.zig
//!
//! Formats user prompts with per-architecture conversation templates,
//! enabling correct behavior for instruction-tuned models.
//!
//! Architecture:
//!   Phase 1: Hardcoded presets + GGUF metadata reading + CLI flags
//!   Phase 2: Extended preset templates + auto-detection
//!   Phase 3: Jinja subset engine (for GGUF built-in templates)
//!
//! Reference: llama.cpp src/llama-chat.cpp, common/chat.cpp, docs/DIALOG_TEMPLATE.md

const std = @import("std");
const model = @import("model");

const log = std.log.scoped(.chat_template);

// ============================================================================
// Types
// ============================================================================

/// A single chat message with role and content.
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,

    pub fn init(role: []const u8, content: []const u8) ChatMessage {
        return .{ .role = role, .content = content };
    }
};

/// Known template format kinds.
pub const TemplateKind = enum {
    chatml,
    llama3,
    gemma,
    unknown,

    pub fn fromString(s: []const u8) ?TemplateKind {
        if (std.mem.eql(u8, s, "chatml")) return .chatml;
        if (std.mem.eql(u8, s, "llama3")) return .llama3;
        if (std.mem.eql(u8, s, "gemma")) return .gemma;
        return null;
    }
};

/// Source of a chat template.
pub const TemplateSource = union(enum) {
    /// GGUF built-in template string (from tokenizer.chat_template metadata)
    gguf_builtin: []const u8,
    /// Named preset template
    preset: TemplateKind,
    /// User-provided custom template string (--chat-template <jinja_str>)
    custom: []const u8,
};

/// A resolved chat template ready for formatting.
pub const Template = struct {
    kind: TemplateKind,
    source: TemplateSource,

    /// Apply this template to format messages.
    pub fn apply(
        self: *const Template,
        allocator: std.mem.Allocator,
        messages: []const ChatMessage,
        system_prompt: ?[]const u8,
        add_generation_prompt: bool,
    ) ![]const u8 {
        switch (self.kind) {
            .chatml => return applyChatml(allocator, messages, system_prompt, add_generation_prompt),
            .llama3 => return applyLlama3(allocator, messages, system_prompt, add_generation_prompt),
            .gemma => return applyGemma(allocator, messages, system_prompt, add_generation_prompt),
            .unknown => {
                // Fallback: treat as raw prompt (no template)
                if (messages.len == 1 and std.mem.eql(u8, messages[0].role, "user")) {
                    return allocator.dupe(u8, messages[0].content);
                }
                // For multi-turn with unknown template, use ChatML as safe default
                log.warn("unknown template kind, falling back to ChatML", .{});
                return applyChatml(allocator, messages, system_prompt, add_generation_prompt);
            },
        }
    }

    pub fn deinit(self: *Template, allocator: std.mem.Allocator) void {
        switch (self.source) {
            .gguf_builtin => |s| allocator.free(s),
            .preset => {},
            .custom => |s| allocator.free(s),
        }
    }
};

// ============================================================================
// Template resolution
// ============================================================================

/// Detect template kind from a Jinja template string (heuristic).
/// Reference: llama.cpp llm_chat_detect_template()
pub fn detectKind(tmpl_src: []const u8) TemplateKind {
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|im_start|>")) {
        return .chatml;
    }
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|start_header_id|>")) {
        return .llama3;
    }
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<start_of_turn>")) {
        return .gemma;
    }
    return .unknown;
}

/// Resolve template kind from architecture (default mapping).
pub fn kindForArchitecture(arch: model.Architecture) TemplateKind {
    return switch (arch) {
        .qwen2, .qwen35, .embedding_qwen2 => .chatml,
        .llama => .llama3,
        .gemma3, .gemma4 => .gemma,
    };
}

/// Resolve a template source to a concrete Template.
/// Priority: custom > gguf_builtin > preset_name > architecture default
pub fn resolve(
    allocator: std.mem.Allocator,
    source: TemplateSource,
    arch: model.Architecture,
) !Template {
    _ = allocator;
    switch (source) {
        .preset => |kind| {
            return Template{ .kind = kind, .source = source };
        },
        .gguf_builtin => |tmpl_str| {
            const kind = detectKind(tmpl_str);
            if (kind != .unknown) {
                return Template{ .kind = kind, .source = source };
            }
            // If GGUF template can't be detected, fall back to arch default
            log.warn("GGUF chat template not recognized, falling back to arch default", .{});
            const arch_kind = kindForArchitecture(arch);
            return Template{ .kind = arch_kind, .source = .{ .preset = arch_kind } };
        },
        .custom => |tmpl_str| {
            const kind = detectKind(tmpl_str);
            if (kind != .unknown) {
                return Template{ .kind = kind, .source = source };
            }
            log.warn("custom chat template not recognized, falling back to arch default", .{});
            const arch_kind = kindForArchitecture(arch);
            return Template{ .kind = arch_kind, .source = .{ .preset = arch_kind } };
        },
    }
}

// ============================================================================
// Convenience API (backward compatible)
// ============================================================================

/// Apply a chat template for a single-turn user prompt (convenience wrapper).
/// Uses architecture default template.
pub fn applySingleTurn(
    allocator: std.mem.Allocator,
    arch: model.Architecture,
    user_prompt: []const u8,
    system_prompt: ?[]const u8,
) ![]const u8 {
    const messages = [_]ChatMessage{
        ChatMessage.init("user", user_prompt),
    };
    return applyMultiTurn(allocator, arch, &messages, system_prompt, true);
}

/// Apply a chat template for multi-turn conversation (convenience wrapper).
/// Uses architecture default template.
pub fn applyMultiTurn(
    allocator: std.mem.Allocator,
    arch: model.Architecture,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    const kind = kindForArchitecture(arch);
    const source = TemplateSource{ .preset = kind };
    var tmpl = try resolve(allocator, source, arch);
    return tmpl.apply(allocator, messages, system_prompt, add_generation_prompt);
}

// ============================================================================
// ChatML (Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5)
// Format: <|im_start|>system\n{system}<|im_end|>\n
//         <|im_start|>user\n{user}<|im_end|>\n
//         <|im_start|>assistant\n{assistant}<|im_end|>\n
//         <|im_start|>assistant\n   ← generation prompt
// ============================================================================
fn applyChatml(
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
            try buf.appendSlice(allocator, "<|im_end|>\n");
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

// ============================================================================
// Llama 3
// Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
//         <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
//         <|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>
//         <|start_header_id|>assistant<|end_header_id|>\n\n   ← generation prompt
// ============================================================================
fn applyLlama3(
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

// ============================================================================
// Gemma (Gemma 3 / Gemma 4)
// Format: <start_of_turn>user\n{user}<end_of_turn>\n
//         <start_of_turn>model\n{assistant}<end_of_turn>\n
//         <start_of_turn>model\n   ← generation prompt
//
// Note: Gemma uses "model" instead of "assistant".
//       There is no explicit system role; system messages are merged
//       into the first user turn (same as llama.cpp behavior).
// ============================================================================
fn applyGemma(
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

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "detectKind: chatml" {
    try testing.expectEqual(TemplateKind.chatml, detectKind("<|im_start|>user\nHello<|im_end|>"));
}

test "detectKind: llama3" {
    try testing.expectEqual(TemplateKind.llama3, detectKind("<|start_header_id|>user<|end_header_id|>"));
}

test "detectKind: gemma" {
    try testing.expectEqual(TemplateKind.gemma, detectKind("<start_of_turn>user\nHello<end_of_turn>"));
}

test "detectKind: unknown" {
    try testing.expectEqual(TemplateKind.unknown, detectKind("{{ messages }}"));
}

test "kindForArchitecture" {
    try testing.expectEqual(TemplateKind.chatml, kindForArchitecture(.qwen2));
    try testing.expectEqual(TemplateKind.chatml, kindForArchitecture(.qwen35));
    try testing.expectEqual(TemplateKind.llama3, kindForArchitecture(.llama));
    try testing.expectEqual(TemplateKind.gemma, kindForArchitecture(.gemma3));
    try testing.expectEqual(TemplateKind.gemma, kindForArchitecture(.gemma4));
}

test "resolve: preset" {
    const source = TemplateSource{ .preset = .chatml };
    var tmpl = try resolve(testing.allocator, source, .qwen2);
    defer tmpl.deinit(testing.allocator);
    try testing.expectEqual(TemplateKind.chatml, tmpl.kind);
}

test "resolve: gguf_builtin" {
    const src = try testing.allocator.dupe(u8, "<|im_start|>user\nHello<|im_end|>");
    const source = TemplateSource{ .gguf_builtin = src };
    var tmpl = try resolve(testing.allocator, source, .qwen2);
    defer tmpl.deinit(testing.allocator);
    try testing.expectEqual(TemplateKind.chatml, tmpl.kind);
}

test "resolve: gguf_builtin unknown" {
    const src = try testing.allocator.dupe(u8, "{{ messages }}");
    const source = TemplateSource{ .gguf_builtin = src };
    var tmpl = try resolve(testing.allocator, source, .llama);
    defer tmpl.deinit(testing.allocator);
    // Falls back to arch default (llama -> llama3)
    try testing.expectEqual(TemplateKind.llama3, tmpl.kind);
}

test "Template.apply: chatml" {
    const source = TemplateSource{ .preset = .chatml };
    var tmpl = try resolve(testing.allocator, source, .qwen2);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        result,
    );
}

test "Template.apply: llama3" {
    const source = TemplateSource{ .preset = .llama3 };
    var tmpl = try resolve(testing.allocator, source, .llama);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        result,
    );
}

test "Template.apply: gemma" {
    const source = TemplateSource{ .preset = .gemma };
    var tmpl = try resolve(testing.allocator, source, .gemma3);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\n1+1=?<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}

test "Template.apply: with system prompt" {
    const source = TemplateSource{ .preset = .chatml };
    var tmpl = try resolve(testing.allocator, source, .qwen2);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        result,
    );
}

test "Template.apply: multi-turn chatml" {
    const source = TemplateSource{ .preset = .chatml };
    var tmpl = try resolve(testing.allocator, source, .qwen35);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "assistant", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n",
        result,
    );
}

test "Template.apply: multi-turn gemma" {
    const source = TemplateSource{ .preset = .gemma };
    var tmpl = try resolve(testing.allocator, source, .gemma3);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "model", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nHow are you?<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}

test "Template.apply: gemma with system merged" {
    const source = TemplateSource{ .preset = .gemma };
    var tmpl = try resolve(testing.allocator, source, .gemma3);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are a helpful assistant.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nYou are a helpful assistant.\n\nHello<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}

test "applySingleTurn: backward compat" {
    const result = try applySingleTurn(testing.allocator, .qwen2, "Hello", null);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        result,
    );
}

test "applyMultiTurn: backward compat" {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "assistant", .content = "Hello!" },
    };
    const result = try applyMultiTurn(testing.allocator, .qwen35, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>assistant\n",
        result,
    );
}
