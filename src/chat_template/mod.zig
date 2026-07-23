//! Chat template support for zllama.zig
//!
//! Formats user prompts with per-architecture conversation templates,
//! enabling correct behavior for instruction-tuned models.
//!
//! Architecture:
//!   Phase 1: Hardcoded presets + GGUF metadata reading + CLI flags
//!   Phase 2: Extended preset templates + auto-detection
//!   Phase 3: Jinja subset engine (for GGUF built-in templates)
//!   Phase 4: Multimodal placeholder support (expandPlaceholders, Media types)
//!
//! Reference: llama.cpp src/llama-chat.cpp, common/chat.cpp, docs/DIALOG_TEMPLATE.md
//!
//! Design: Each template kind is implemented in its own file under
//! src/chat_template/<kind>.zig, exposing a uniform `apply()` function.
//! The Template struct uses a vtable (TemplateVTable) to dispatch
//! to the correct implementation at runtime.

const std = @import("std");
const model = @import("model");
const minja = @import("minja");

// 导入子模块
const types = @import("types");
const multimodal = @import("multimodal");

// 导入各模板实现
const chatml = @import("chatml");
const llama3 = @import("llama3");
const llama4 = @import("llama4");
const gemma = @import("gemma");
const gemma4 = @import("gemma4");
const mistral_v7 = @import("mistral_v7");
const phi4 = @import("phi4");
const deepseek3 = @import("deepseek3");
const tinyllama = @import("tinyllama");

pub const ChatMessage = types.ChatMessage;
pub const Media = types.Media;
pub const MediaType = types.MediaType;
pub const PlaceholderInfo = types.PlaceholderInfo;
pub const ExpandedPlaceholders = types.ExpandedPlaceholders;
pub const ScanMarkers = types.ScanMarkers;
pub const ImagePosType = types.ImagePosType;

// 重新导出多模态辅助函数
pub const scanPlaceholders = multimodal.scanPlaceholders;
pub const expandPlaceholders = multimodal.expandPlaceholders;
pub const containsPlaceholder = multimodal.containsPlaceholder;
pub const containsPlaceholderEx = multimodal.containsPlaceholderEx;
pub const ensurePlaceholderInContent = multimodal.ensurePlaceholderInContent;

pub const tokenizeWithPlaceholders = multimodal.tokenizeWithPlaceholders;
pub const placeholderTokenOffset = multimodal.placeholderTokenOffset;
pub const TokenizedSegments = multimodal.TokenizedSegments;

const log = std.log.scoped(.chat_template);

// ============================================================================
// Template VTable
// ============================================================================

/// Function signature for all template apply functions.
const ApplyFn = *const fn (
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) anyerror![]const u8;

/// VTable for template dispatch.
pub const TemplateVTable = struct {
    apply: ApplyFn,
};

/// Get the vtable for a given template kind.
pub fn vtableForKind(kind: TemplateKind) TemplateVTable {
    return switch (kind) {
        .chatml => .{ .apply = chatml.apply },
        .llama3 => .{ .apply = llama3.apply },
        .llama4 => .{ .apply = llama4.apply },
        .gemma => .{ .apply = gemma.apply },
        .gemma4 => .{ .apply = gemma4.apply },
        .mistral_v7 => .{ .apply = mistral_v7.apply },
        .phi4 => .{ .apply = phi4.apply },
        .deepseek3 => .{ .apply = deepseek3.apply },
        .tinyllama => .{ .apply = tinyllama.apply },
        .unknown => .{ .apply = applyUnknown },
    };
}

// ============================================================================
// Template Format
// ============================================================================

/// Known template format kinds.
pub const TemplateKind = enum {
    chatml,
    llama3,
    llama4,
    gemma,
    gemma4,
    mistral_v7,
    phi4,
    deepseek3,
    tinyllama,
    unknown,

    pub fn fromString(s: []const u8) ?TemplateKind {
        if (std.mem.eql(u8, s, "chatml")) return .chatml;
        if (std.mem.eql(u8, s, "llama3")) return .llama3;
        if (std.mem.eql(u8, s, "llama4")) return .llama4;
        if (std.mem.eql(u8, s, "gemma")) return .gemma;
        if (std.mem.eql(u8, s, "gemma4")) return .gemma4;
        if (std.mem.eql(u8, s, "mistral-v7") or std.mem.eql(u8, s, "mistral")) return .mistral_v7;
        if (std.mem.eql(u8, s, "phi4") or std.mem.eql(u8, s, "phi-4")) return .phi4;
        if (std.mem.eql(u8, s, "deepseek3") or std.mem.eql(u8, s, "deepseek-v3")) return .deepseek3;
        if (std.mem.eql(u8, s, "tinyllama")) return .tinyllama;
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
    /// VTable for dispatch — initialized from kind.
    vtable: TemplateVTable,
    /// Whether Jinja rendering is enabled for unknown/custom templates.
    /// Set to false with --no-jinja flag.
    jinja_enabled: bool = true,

    /// Apply this template to format messages.
    /// When jinja_enabled and source is a template string (gguf_builtin or custom),
    /// Jinja rendering is tried first. On failure, falls back to the built-in preset.
    pub fn apply(
        self: *const Template,
        allocator: std.mem.Allocator,
        messages: []const ChatMessage,
        system_prompt: ?[]const u8,
        add_generation_prompt: bool,
    ) ![]const u8 {
        // Try Jinja rendering first when enabled and source is a template string
        if (self.jinja_enabled) {
            if (self.getTemplateString()) |tmpl_str| {
                if (tryJinjaApply(allocator, tmpl_str, messages, system_prompt, add_generation_prompt)) |result| {
                    return result;
                } else |_| {
                    log.warn("Jinja rendering failed ({d} byte template), falling back to built-in preset", .{tmpl_str.len});
                }
            }
        }
        // Fall back to vtable dispatch (built-in presets, ChatML, raw prompt, etc.)
        return self.vtable.apply(allocator, messages, system_prompt, add_generation_prompt);
    }

    /// Extract the Jinja template source string from the TemplateSource, if any.
    fn getTemplateString(self: *const Template) ?[]const u8 {
        return switch (self.source) {
            .gguf_builtin => |s| s,
            .custom => |s| s,
            .preset => null,
        };
    }

    pub fn deinit(self: *Template, allocator: std.mem.Allocator) void {
        _ = allocator;
        // Template does NOT own the source strings — they are borrowed from
        // the caller (InferenceEngine). The caller is responsible for freeing
        // them via chat_template_source / custom template storage.
        _ = self;
    }
};

// ============================================================================
// Jinja template rendering via minja C++ bridge
// ============================================================================

/// Try to render messages through the Jinja template engine (minja).
/// Returns the formatted prompt on success, or an error on failure.
fn tryJinjaApply(
    allocator: std.mem.Allocator,
    tmpl_str: []const u8,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    // minja requires null-terminated strings
    const tmpl_z = try allocator.dupeZ(u8, tmpl_str);
    defer allocator.free(tmpl_z);

    // Convert ChatMessage (with optional media) to minja ChatMessage (role+content only)
    // Media placeholders (e.g., <|image|>, <|audio|>) are expected to already be in content
    const minja_messages = try allocator.alloc(minja.ChatMessage, messages.len);
    defer allocator.free(minja_messages);
    for (messages, 0..) |msg, i| {
        minja_messages[i] = .{ .role = msg.role, .content = msg.content };
    }

    const tmpl = try minja.ChatTemplate.create(tmpl_z, "", "");
    defer tmpl.destroy();
    return try minja.applyTemplate(allocator, tmpl, minja_messages, system_prompt, null, add_generation_prompt);
}

// ============================================================================
// Unknown template fallback
// ============================================================================

/// Fallback for unknown templates.
/// Jinja rendering is attempted by Template.apply() before this vtable entry
/// is reached, so this function handles only the "Jinja failed or disabled" case.
fn applyUnknown(
    allocator: std.mem.Allocator,
    messages: []const ChatMessage,
    system_prompt: ?[]const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    // For multi-turn with unknown template, use ChatML as safe default
    if (messages.len != 1 or !std.mem.eql(u8, messages[0].role, "user")) {
        log.warn("unknown template kind, falling back to ChatML", .{});
        return chatml.apply(allocator, messages, system_prompt, add_generation_prompt);
    }
    // Single-message: return content directly as raw prompt
    return allocator.dupe(u8, messages[0].content);
}

// ============================================================================
// Template resolution
// ============================================================================

/// Detect template kind from a Jinja template string (heuristic).
/// Reference: llama.cpp llm_chat_detect_template()
pub fn detectKind(tmpl_src: []const u8) TemplateKind {
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|im_start|>")) {
        // Phi-4 uses <|im_start|> + <|im_sep|>
        if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|im_sep|>")) return .phi4;
        return .chatml;
    }
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|start_header_id|>")) return .llama3;
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|header_start|>")) return .llama4;
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<start_of_turn>")) return .gemma;
    // Gemma 4 uses <|turn> format (different from Gemma 3's <start_of_turn>)
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|turn>")) return .gemma4;
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "[INST]")) return .mistral_v7;
    // TinyLlama uses <|system|>, <|user|>, <|assistant|> with </s> separator
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<|system|>")) return .tinyllama;
    // DeepSeek V3 uses UTF-8 characters: <｜Assistant｜>
    if (std.mem.containsAtLeast(u8, tmpl_src, 1, "<｜Assistant｜>") or
        std.mem.containsAtLeast(u8, tmpl_src, 1, "<｜assistant｜>") or
        std.mem.containsAtLeast(u8, tmpl_src, 1, "<｜User｜>"))
    {
        return .deepseek3;
    }
    return .unknown;
}

/// Resolve template kind from architecture (default mapping).
/// If model_name is provided and contains "tinyllama", returns .tinyllama
/// for .llama architecture (since TinyLlama reports as "llama" arch).
pub fn kindForArchitecture(arch: model.Architecture, model_name: ?[]const u8) TemplateKind {
    // Check for TinyLlama by model name (TinyLlama reports as "llama" arch
    // but uses a different chat format)
    if (arch == .llama) {
        if (model_name) |name| {
            // Case-insensitive check for "tinyllama" in model name
            if (std.ascii.findIgnoreCase(name, "tinyllama") != null or
                std.ascii.findIgnoreCase(name, "tiny_llama") != null or
                std.ascii.findIgnoreCase(name, "tiny llama") != null)
            {
                return .tinyllama;
            }
        }
    }
    return switch (arch) {
        .qwen2 => .chatml,
        .qwen35 => .chatml,
        .qwen3vl => .chatml,
        .embedding_qwen2 => .chatml,
        .llama => .llama3,
        .minicpm => .llama3,
        .gemma3 => .gemma,
        .gemma4 => .gemma4,
    };
}

/// Resolve a template source to a concrete Template.
/// Priority chain: custom > GGUF built-in > preset > arch default > ChatML fallback
///
/// When source is .gguf_builtin or .custom and jinja_enabled:
///   - Known templates (detected by detectKind) use the built-in Zig preset
///     which has been carefully aligned with llama.cpp behavior.
///   - Unknown templates try Jinja rendering, falling back to vtable on failure.
///
/// When jinja_enabled is false:
///   1. detectKind → known preset (uses built-in Zig implementation)
///   2. If unknown → fall back to arch default preset
///   3. If arch default also unknown → ChatML fallback (inside apply())
///
/// model_name is optional and used to detect TinyLlama (which reports as "llama" arch).
/// jinja_enabled: when true, unknown GGUF/custom template strings are routed to Jinja rendering.
pub fn resolve(
    allocator: std.mem.Allocator,
    source: TemplateSource,
    arch: model.Architecture,
    model_name: ?[]const u8,
    jinja_enabled: bool,
) !Template {
    _ = allocator;
    switch (source) {
        .preset => |kind| {
            return Template{
                .kind = kind,
                .source = source,
                .vtable = vtableForKind(kind),
            };
        },
        .gguf_builtin => |tmpl_str| {
            // When Jinja is enabled, detect the template kind first.
            // Known templates (gemma4, chatml, llama3, etc.) use the built-in
            // Zig preset which has been precisely aligned with llama.cpp.
            // Unknown templates try Jinja rendering, falling back to vtable.
            if (jinja_enabled) {
                const hint = detectKind(tmpl_str);
                if (hint != .unknown) {
                    log.info("GGUF template ({d} bytes) detected as {s}, using built-in preset", .{ tmpl_str.len, @tagName(hint) });
                    return Template{
                        .kind = hint,
                        .source = source,
                        .vtable = vtableForKind(hint),
                        .jinja_enabled = false,
                    };
                }
                log.info("GGUF chat template ({d} bytes), will try Jinja rendering", .{tmpl_str.len});
                return Template{
                    .kind = .unknown,
                    .source = source,
                    .vtable = vtableForKind(.unknown),
                    .jinja_enabled = true,
                };
            }
            // Jinja disabled: detect kind and use built-in preset if known
            const kind = detectKind(tmpl_str);
            if (kind != .unknown) {
                log.info("GGUF template ({d} bytes) detected as {s}, using built-in preset", .{ tmpl_str.len, @tagName(kind) });
                return Template{
                    .kind = kind,
                    .source = source,
                    .vtable = vtableForKind(kind),
                };
            }
            // Jinja disabled + unknown template → fall back to arch default
            log.warn("GGUF chat template not recognized, falling back to arch default", .{});
            const arch_kind = kindForArchitecture(arch, model_name);
            return Template{
                .kind = arch_kind,
                .source = .{ .preset = arch_kind },
                .vtable = vtableForKind(arch_kind),
            };
        },
        .custom => |tmpl_str| {
            // For custom templates, always try Jinja rendering first when enabled
            if (jinja_enabled) {
                log.info("custom chat template ({d} bytes), will try Jinja rendering", .{tmpl_str.len});
                return Template{
                    .kind = .unknown,
                    .source = source,
                    .vtable = vtableForKind(.unknown),
                    .jinja_enabled = true,
                };
            }
            // Jinja disabled: detect kind and use built-in preset if known
            const kind = detectKind(tmpl_str);
            if (kind != .unknown) {
                return Template{
                    .kind = kind,
                    .source = source,
                    .vtable = vtableForKind(kind),
                };
            }
            // Jinja disabled + unknown → fall back to arch default
            log.warn("custom chat template not recognized and Jinja disabled, falling back to arch default", .{});
            const arch_kind = kindForArchitecture(arch, model_name);
            return Template{
                .kind = arch_kind,
                .source = .{ .preset = arch_kind },
                .vtable = vtableForKind(arch_kind),
            };
        },
    }
}

/// Debug helper: extract and format template diagnostics.
/// Prints the template string (truncated), detection result, and resolution path.
/// This is useful for debugging multimodal template issues.
///
/// Returns a formatted string describing the template state. Caller owns the memory.
pub fn debugPrintTemplate(
    allocator: std.mem.Allocator,
    source: TemplateSource,
    arch: model.Architecture,
    model_name: ?[]const u8,
) ![]const u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, "=== Template Debug ===\n");

    {
        const line = try std.fmt.allocPrint(allocator, "Architecture: {s}\n", .{@tagName(arch)});
        defer allocator.free(line);
        try buf.appendSlice(allocator, line);
    }
    if (model_name) |name| {
        const line = try std.fmt.allocPrint(allocator, "Model name: {s}\n", .{name});
        defer allocator.free(line);
        try buf.appendSlice(allocator, line);
    }

    switch (source) {
        .preset => |kind| {
            const line = try std.fmt.allocPrint(allocator, "Source: preset ({s})\n", .{@tagName(kind)});
            defer allocator.free(line);
            try buf.appendSlice(allocator, line);
        },
        .gguf_builtin => |tmpl_str| {
            const kind = detectKind(tmpl_str);
            {
                const line = try std.fmt.allocPrint(allocator, "Source: GGUF built-in ({d} bytes)\n", .{tmpl_str.len});
                defer allocator.free(line);
                try buf.appendSlice(allocator, line);
            }
            {
                const line = try std.fmt.allocPrint(allocator, "Detected kind: {s}\n", .{@tagName(kind)});
                defer allocator.free(line);
                try buf.appendSlice(allocator, line);
            }
            try buf.appendSlice(allocator, "Template preview:\n");
            try buf.appendSlice(allocator, tmpl_str);
            try buf.appendSlice(allocator, "\n");
            const line = try std.fmt.allocPrint(allocator, "... (total {d} bytes)\n", .{tmpl_str.len});
            defer allocator.free(line);
            try buf.appendSlice(allocator, line);
        },
        .custom => |tmpl_str| {
            const kind = detectKind(tmpl_str);
            {
                const line = try std.fmt.allocPrint(allocator, "Source: custom ({d} bytes)\n", .{tmpl_str.len});
                defer allocator.free(line);
                try buf.appendSlice(allocator, line);
            }
            {
                const line = try std.fmt.allocPrint(allocator, "Detected kind: {s}\n", .{@tagName(kind)});
                defer allocator.free(line);
                try buf.appendSlice(allocator, line);
            }
            try buf.appendSlice(allocator, "Template preview:\n");
            try buf.appendSlice(allocator, tmpl_str);
            try buf.appendSlice(allocator, "\n");
            const line = try std.fmt.allocPrint(allocator, "... (total {d} more bytes)\n", .{tmpl_str.len});
            defer allocator.free(line);
            try buf.appendSlice(allocator, line);
        },
    }

    try buf.appendSlice(allocator, "=== End Template Debug ===\n");
    return buf.toOwnedSlice(allocator);
}

// ============================================================================
// Convenience API (backward compatible)
// ============================================================================

/// Apply a chat template for a single-turn user prompt with optional media.
/// This is the shared implementation used by both engine.zig and multimodal.zig
/// to avoid code duplication.
pub fn applyWithMedia(
    allocator: std.mem.Allocator,
    arch: model.Architecture,
    model_name: ?[]const u8,
    chat_template_source: ?TemplateSource,
    no_chat_template: bool,
    no_jinja: bool,
    user_prompt: []const u8,
    media: ?Media,
    system_prompt: ?[]const u8,
) ![]const u8 {
    if (no_chat_template) return allocator.dupe(u8, user_prompt);

    const source = chat_template_source orelse TemplateSource{ .preset = kindForArchitecture(arch, model_name) };
    var tmpl = try resolve(allocator, source, arch, model_name, !no_jinja);
    defer tmpl.deinit(allocator);

    // Prepend the media placeholder to the content so it's available for
    // both Jinja template rendering (where media info is not passed separately)
    // and preset template rendering (where appendMediaContent will skip adding
    // the marker if it's already present, avoiding double placeholders).
    const effective_prompt = if (media) |m| blk: {
        break :blk try ensurePlaceholderInContent(user_prompt, m.type, allocator, null);
    } else user_prompt;
    const needs_free = media != null and effective_prompt.ptr != user_prompt.ptr;
    defer if (needs_free) allocator.free(effective_prompt);

    const messages = if (media) |m| [_]ChatMessage{
        ChatMessage.withMedia("user", effective_prompt, m),
    } else [_]ChatMessage{
        ChatMessage.init("user", effective_prompt),
    };
    return tmpl.apply(allocator, &messages, system_prompt, true);
}
/// Uses architecture default template.
pub fn applySingleTurn(
    allocator: std.mem.Allocator,
    arch: model.Architecture,
    user_prompt: []const u8,
    system_prompt: ?[]const u8,
) ![]const u8 {
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = user_prompt },
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
    const kind = kindForArchitecture(arch, null);
    const source = TemplateSource{ .preset = kind };
    var tmpl = try resolve(allocator, source, arch, null, false);
    return tmpl.apply(allocator, messages, system_prompt, add_generation_prompt);
}


