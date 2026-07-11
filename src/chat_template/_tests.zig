const std = @import("std");
const mod = @import("mod.zig");
const TemplateKind = mod.TemplateKind;
const detectKind = mod.detectKind;
const kindForArchitecture = mod.kindForArchitecture;
const resolve = mod.resolve;
const Template = mod.Template;
const TemplateSource = mod.TemplateSource;
const ChatMessage = mod.ChatMessage;
const Media = mod.Media;
const MediaType = mod.MediaType;
const debugPrintTemplate = mod.debugPrintTemplate;
const applySingleTurn = mod.applySingleTurn;
const applyMultiTurn = mod.applyMultiTurn;
const model = @import("model");
const minja = @import("minja");
const scanPlaceholders = mod.scanPlaceholders;
const expandPlaceholders = mod.expandPlaceholders;
const containsPlaceholder = mod.containsPlaceholder;
const ensurePlaceholderInContent = mod.ensurePlaceholderInContent;
const tokenizeWithPlaceholders = mod.tokenizeWithPlaceholders;
const placeholderTokenOffset = mod.placeholderTokenOffset;
const PlaceholderInfo = mod.PlaceholderInfo;
const ExpandedPlaceholders = mod.ExpandedPlaceholders;
const TokenizedSegments = mod.TokenizedSegments;

const testing = std.testing;

test "detectKind: chatml" {
    try testing.expectEqual(TemplateKind.chatml, detectKind("<|im_start|>user\nHello<|im_end|>"));
}

test "detectKind: llama3" {
    try testing.expectEqual(TemplateKind.llama3, detectKind("<|start_header_id|>user<|end_header_id|>"));
}

test "detectKind: llama4" {
    try testing.expectEqual(TemplateKind.llama4, detectKind("<|header_start|>user<|header_end|>"));
}

test "detectKind: gemma" {
    try testing.expectEqual(TemplateKind.gemma, detectKind("<start_of_turn>user\nHello<end_of_turn>"));
}

test "detectKind: mistral_v7" {
    try testing.expectEqual(TemplateKind.mistral_v7, detectKind("[INST] Hello [/INST]"));
}

test "detectKind: phi4" {
    try testing.expectEqual(TemplateKind.phi4, detectKind("<|im_start|>user\nHello<|im_sep|>\n"));
}

test "detectKind: deepseek3" {
    try testing.expectEqual(TemplateKind.deepseek3, detectKind("<｜User｜>Hello<｜Assistant｜>"));
}

test "detectKind: tinyllama" {
    try testing.expectEqual(TemplateKind.tinyllama, detectKind("<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n{assistant}</s>"));
}

test "detectKind: unknown" {
    try testing.expectEqual(TemplateKind.unknown, detectKind("{{ messages }}"));
}

test "kindForArchitecture" {
    try testing.expectEqual(TemplateKind.chatml, kindForArchitecture(.qwen2, null));
    try testing.expectEqual(TemplateKind.chatml, kindForArchitecture(.qwen35, null));
    try testing.expectEqual(TemplateKind.llama3, kindForArchitecture(.llama, null));
    try testing.expectEqual(TemplateKind.gemma, kindForArchitecture(.gemma3, null));
    try testing.expectEqual(TemplateKind.gemma4, kindForArchitecture(.gemma4, null));
    // TinyLlama detection via model name
    try testing.expectEqual(TemplateKind.tinyllama, kindForArchitecture(.llama, "TinyLlama-1.1B-Chat"));
    try testing.expectEqual(TemplateKind.tinyllama, kindForArchitecture(.llama, "tinyllama-1.1b"));
    // Regular Llama with model name should still be llama3
    try testing.expectEqual(TemplateKind.llama3, kindForArchitecture(.llama, "Llama-3.2-3B-Instruct"));
}

test "resolve: preset" {
    const source = TemplateSource{ .preset = .chatml };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
    defer tmpl.deinit(testing.allocator);
    try testing.expectEqual(TemplateKind.chatml, tmpl.kind);
}

test "resolve: gguf_builtin" {
    const src = try testing.allocator.dupe(u8, "<|im_start|>user\nHello<|im_end|>");
    const source = TemplateSource{ .gguf_builtin = src };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
    defer tmpl.deinit(testing.allocator);
    try testing.expectEqual(TemplateKind.chatml, tmpl.kind);
}

test "resolve: gguf_builtin unknown" {
    const src = try testing.allocator.dupe(u8, "{{ messages }}");
    const source = TemplateSource{ .gguf_builtin = src };
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);
    // Falls back to arch default (llama -> llama3)
    try testing.expectEqual(TemplateKind.llama3, tmpl.kind);
}

test "Template.apply: chatml" {
    const source = TemplateSource{ .preset = .chatml };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
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
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        result,
    );
}

test "Template.apply: llama4" {
    const source = TemplateSource{ .preset = .llama4 };
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|begin_of_text|><|header_start|>user<|header_end|>\n\nHello<|eom_id|><|header_start|>assistant<|header_end|>\n\n",
        result,
    );
}

test "Template.apply: gemma" {
    const source = TemplateSource{ .preset = .gemma };
    var tmpl = try resolve(testing.allocator, source, .gemma3, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\n1+1=?<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}

test "Template.apply: mistral_v7" {
    const source = TemplateSource{ .preset = .mistral_v7 };
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "[INST] Hello [/INST] ",
        result,
    );
}

test "Template.apply: phi4" {
    const source = TemplateSource{ .preset = .phi4 };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_sep|>\n<|im_start|>assistant\n",
        result,
    );
}

test "Template.apply: deepseek3" {
    const source = TemplateSource{ .preset = .deepseek3 };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<｜User｜>Hello<｜Assistant｜>",
        result,
    );
}

test "Template.apply: tinyllama" {
    const source = TemplateSource{ .preset = .tinyllama };
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|user|>\n1+1=?</s>\n<|assistant|>\n",
        result,
    );
}

test "Template.apply: with system prompt" {
    const source = TemplateSource{ .preset = .chatml };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
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
    var tmpl = try resolve(testing.allocator, source, .qwen35, null, false);
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
    var tmpl = try resolve(testing.allocator, source, .gemma3, null, false);
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
    var tmpl = try resolve(testing.allocator, source, .gemma3, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are a helpful assistant.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nYou are a helpful assistant.\n\nHello<end_of_turn>\n<start_of_turn>model\n",
        result,
    );
}

test "Template.apply: gemma4" {
    const source = TemplateSource{ .preset = .gemma4 };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "1+1=?" }};
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    // Gemma 4: no system → no system turn; generation prompt is just <|turn>model\n
    try testing.expectEqualStrings(
        "<|turn>user\n1+1=?<turn|>\n<|turn>model\n",
        result,
    );
}

test "Template.apply: gemma4 with system" {
    const source = TemplateSource{ .preset = .gemma4 };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are a helpful assistant.", true);
    defer testing.allocator.free(result);
    // Gemma 4: system gets its own <|turn>system\n turn
    try testing.expectEqualStrings(
        "<|turn>system\nYou are a helpful assistant.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n",
        result,
    );
}

test "Template.apply: gemma4 multi-turn" {
    const source = TemplateSource{ .preset = .gemma4 };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "model", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n",
        result,
    );
}

test "Template.apply: gemma4 multi-turn with system" {
    const source = TemplateSource{ .preset = .gemma4 };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "model", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try tmpl.apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n",
        result,
    );
}

test "Template.apply: gemma4 strip thinking" {
    const source = TemplateSource{ .preset = .gemma4 };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, false);
    defer tmpl.deinit(testing.allocator);

    // Model message with thinking block inside
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "1+1=?" },
        .{ .role = "model", .content = "<|channel>thought\nLet me think... 1+1=2\n<channel|>The answer is 2." },
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    // Thinking block should be stripped from model output
    try testing.expectEqualStrings(
        "<|turn>user\n1+1=?<turn|>\n<|turn>model\nThe answer is 2.<turn|>\n<|turn>model\n",
        result,
    );
}

test "Template.apply: gemma4 system role message" {
    const source = TemplateSource{ .preset = .gemma4 };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, false);
    defer tmpl.deinit(testing.allocator);

    // System role message in messages array
    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "You are a math tutor." },
        .{ .role = "user", .content = "1+1=?" },
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|turn>system\nYou are a math tutor.<turn|>\n<|turn>user\n1+1=?<turn|>\n<|turn>model\n",
        result,
    );
}

test "Template.apply: multi-turn mistral_v7" {
    const source = TemplateSource{ .preset = .mistral_v7 };
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "assistant", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "[INST] Hi [/INST]Hello!</s>[INST] How are you? [/INST] ",
        result,
    );
}

test "Template.apply: multi-turn deepseek3" {
    const source = TemplateSource{ .preset = .deepseek3 };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "assistant", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<｜User｜>Hi<｜Assistant｜>Hello!<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>",
        result,
    );
}

test "Template.apply: mistral_v7 with system" {
    const source = TemplateSource{ .preset = .mistral_v7 };
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "[INST] You are helpful.\n\nHello [/INST] ",
        result,
    );
}

test "Template.apply: phi4 with system" {
    const source = TemplateSource{ .preset = .phi4 };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_sep|>\n<|im_start|>assistant\n",
        result,
    );
}

test "Template.apply: deepseek3 with system" {
    const source = TemplateSource{ .preset = .deepseek3 };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<｜User｜>You are helpful.\n\nHello<｜Assistant｜>",
        result,
    );
}

test "Template.apply: tinyllama with system" {
    const source = TemplateSource{ .preset = .tinyllama };
    var tmpl = try resolve(testing.allocator, source, .llama, null, false);
    defer tmpl.deinit(testing.allocator);

    const messages = [_]ChatMessage{.{ .role = "user", .content = "Hello" }};
    const result = try tmpl.apply(testing.allocator, &messages, "You are helpful.", true);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings(
        "<|system|>\nYou are helpful.</s>\n<|user|>\nHello</s>\n<|assistant|>\n",
        result,
    );
}

test "fromString: valid names" {
    try testing.expectEqual(TemplateKind.chatml, TemplateKind.fromString("chatml").?);
    try testing.expectEqual(TemplateKind.llama3, TemplateKind.fromString("llama3").?);
    try testing.expectEqual(TemplateKind.llama4, TemplateKind.fromString("llama4").?);
    try testing.expectEqual(TemplateKind.gemma, TemplateKind.fromString("gemma").?);
    try testing.expectEqual(TemplateKind.mistral_v7, TemplateKind.fromString("mistral-v7").?);
    try testing.expectEqual(TemplateKind.mistral_v7, TemplateKind.fromString("mistral").?);
    try testing.expectEqual(TemplateKind.phi4, TemplateKind.fromString("phi4").?);
    try testing.expectEqual(TemplateKind.phi4, TemplateKind.fromString("phi-4").?);
    try testing.expectEqual(TemplateKind.deepseek3, TemplateKind.fromString("deepseek3").?);
    try testing.expectEqual(TemplateKind.deepseek3, TemplateKind.fromString("deepseek-v3").?);
    try testing.expectEqual(TemplateKind.tinyllama, TemplateKind.fromString("tinyllama").?);
}

test "fromString: invalid name" {
    try testing.expectEqual(@as(?TemplateKind, null), TemplateKind.fromString("nonexistent"));
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

// ============================================================================
// Multimodal Template Pipeline Tests
// ============================================================================

test "debugPrintTemplate: GGUF built-in" {
    const source = TemplateSource{ .gguf_builtin = "<|im_start|>user\nHello<|im_end|>" };
    const result = try debugPrintTemplate(testing.allocator, source, .qwen2, null);
    defer testing.allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "Source: GGUF built-in") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Detected kind: chatml") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>") != null);
}

test "debugPrintTemplate: preset" {
    const source = TemplateSource{ .preset = .gemma4 };
    const result = try debugPrintTemplate(testing.allocator, source, .gemma4, "gemma-4-2B-it");
    defer testing.allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "Source: preset (gemma4)") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Architecture: gemma4") != null);
}

test "detectKind: Gemma4 template with <|turn>" {
    // Simulate a Gemma4 Jinja template snippet (from google-gemma-4-31B-it-interleaved.jinja)
    const gemma4_template =
        \\{%- for message in messages -%}
        \\{{- '<|turn>' + role + '\n' }}
        \\{{- message['content'] | trim }}
        \\{{- '<turn|>\n' }}
        \\{%- endfor -%}
    ;
    const kind = detectKind(gemma4_template);
    try testing.expectEqual(TemplateKind.gemma4, kind);
}

test "detectKind: Gemma4 template NOT confused with Gemma3" {
    // Gemma4 uses <|turn>, Gemma3 uses <start_of_turn>
    // Ensure a Gemma4 template is NOT detected as Gemma3
    const gemma4_template = "{{- '<|turn>' + role + '\\\\n' }}{{- '<turn|>\\\\n' }}";
    const kind = detectKind(gemma4_template);
    try testing.expectEqual(TemplateKind.gemma4, kind);
}

test "detectKind: Gemma3 template with <start_of_turn>" {
    const gemma3_template = "{{ '<start_of_turn>' + role + '\\\\n' }}";
    const kind = detectKind(gemma3_template);
    try testing.expectEqual(TemplateKind.gemma, kind);
}

test "resolve: GGUF Gemma4 template routes to Jinja rendering" {
    // A Gemma4 Jinja template from GGUF — now routes to Jinja rendering
    // (the GGUF template is the canonical source; detectKind is only a hint)
    const gemma4_template =
        \\{%- for message in messages -%}
        \\{{- '<|turn>' + role + '\n' }}
        \\{{- message['content'] | trim }}
        \\{{- '<turn|>\n' }}
        \\{%- endfor -%}
        \\{%- if add_generation_prompt -%}
        \\{{- '<|turn>model\n' }}
        \\{%- endif -%}
    ;
    const source = TemplateSource{ .gguf_builtin = gemma4_template };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, true);
    defer tmpl.deinit(testing.allocator);

    // Should be .gemma4 with Jinja disabled (known templates use built-in preset)
    try testing.expectEqual(TemplateKind.gemma4, tmpl.kind);
    try testing.expect(!tmpl.jinja_enabled);

    // Apply the template and verify it renders correctly with image placeholder
    const media = Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } },
    };
    const content_with_placeholder = try ensurePlaceholderInContent("Describe this", .image, testing.allocator);
    defer if (content_with_placeholder.ptr != "Describe this".ptr) testing.allocator.free(content_with_placeholder);

    const messages = [_]ChatMessage{
        ChatMessage.withMedia("user", content_with_placeholder, media),
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);

    // The rendered output should contain the image placeholder
    try testing.expect(std.mem.indexOf(u8, result, "<|image|>") != null);
    // And the Gemma4 turn markers
    try testing.expect(std.mem.indexOf(u8, result, "<|turn>user") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<turn|>") != null);
    // Generation prompt should be present
    try testing.expect(std.mem.indexOf(u8, result, "<|turn>model") != null);
    // No system turn (no system prompt)
    try testing.expect(std.mem.indexOf(u8, result, "<|turn>system") == null);
}

test "resolve: unknown GGUF template uses Jinja fallback" {
    // A truly unknown template (no recognizable markers) should fall back to Jinja
    const unknown_template =
        \\{{ bos_token }}
        \\{% for message in messages %}
        \\{{ message['role'] }}: {{ message['content'] }}
        \\{% endfor %}
        \\{% if add_generation_prompt %}assistant: {% endif %}
    ;
    const source = TemplateSource{ .gguf_builtin = unknown_template };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, true);
    defer tmpl.deinit(testing.allocator);

    // Should stay as .unknown with Jinja enabled
    try testing.expectEqual(TemplateKind.unknown, tmpl.kind);
    try testing.expect(tmpl.jinja_enabled);

    // Apply and verify Jinja rendering works
    const messages = [_]ChatMessage{
        ChatMessage.init("user", "Hello"),
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "user: Hello") != null);
    try testing.expect(std.mem.indexOf(u8, result, "assistant:") != null);
}

test "resolve: unknown GGUF template with media message via Jinja" {
    // An unknown Jinja template that just echoes content - verify media placeholders work
    const unknown_template =
        \\{% for message in messages %}
        \\{{ message['role'] }}: {{ message['content'] }}
        \\{% endfor %}
    ;
    const source = TemplateSource{ .gguf_builtin = unknown_template };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, true);
    defer tmpl.deinit(testing.allocator);

    // Should use Jinja rendering
    try testing.expectEqual(TemplateKind.unknown, tmpl.kind);
    try testing.expect(tmpl.jinja_enabled);

    const media = Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } },
    };
    // Content WITHOUT placeholder - Jinja messagesToList should add it
    const messages = [_]ChatMessage{
        ChatMessage.withMedia("user", "Describe this", media),
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, false);
    defer testing.allocator.free(result);

    // Jinja messagesToList should prepend <|image|> to content
    try testing.expect(std.mem.indexOf(u8, result, "<|image|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Describe this") != null);
}

test "full multimodal pipeline: template render + placeholder scan" {
    // Test the complete pipeline:
    // 1. Template renders with placeholder in content
    // 2. scanPlaceholders finds the placeholder
    // 3. tokenizeWithPlaceholders expands it

    const template_str = "{{ messages[0]['content'] }}";
    const source = TemplateSource{ .gguf_builtin = template_str };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, true);
    defer tmpl.deinit(testing.allocator);

    const media = Media{
        .type = .audio,
        .data = .{ .audio = .{ .samples = &.{}, .sample_rate = 0 } },
    };
    const messages = [_]ChatMessage{
        ChatMessage.withMedia("user", "Transcribe", media),
    };
    const formatted = try tmpl.apply(testing.allocator, &messages, null, false);
    defer testing.allocator.free(formatted);

    // Verify Jinja rendered the content with audio placeholder
    try testing.expect(std.mem.indexOf(u8, formatted, "<|audio|>") != null);
    try testing.expect(std.mem.indexOf(u8, formatted, "Transcribe") != null);

    // Scan for placeholders
    const placeholders = try scanPlaceholders(formatted, testing.allocator);
    defer testing.allocator.free(placeholders);
    try testing.expectEqual(@as(usize, 1), placeholders.len);
    try testing.expectEqual(MediaType.audio, placeholders[0].media_type);

    // Tokenize with placeholder expansion
    const tokenizer_fn = struct {
        fn tokenize(_ctx: ?*anyopaque, text_seg: []const u8, alloc: std.mem.Allocator) ![]u32 {
            _ = _ctx;
            var tokens = try alloc.alloc(u32, text_seg.len);
            for (text_seg, 0..) |c, i| {
                tokens[i] = @intCast(c);
            }
            return tokens;
        }
    }.tokenize;

    var expanded = try tokenizeWithPlaceholders(
        testing.allocator,
        formatted,
        null,
        &tokenizer_fn,
        1,
        2, // image_token_id, audio_token_id
        0,
        3, // image_token_count, audio_token_count
    );
    defer expanded.deinit();
    try testing.expectEqual(@as(usize, 1), expanded.offsets.len);
    try testing.expectEqual(@as(u32, 3), expanded.offsets[0].token_count);
}

test "full multimodal pipeline: Gemma4 preset with image placeholder" {
    // Verify the Gemma4 preset correctly renders multimodal messages
    // This is the path currently used after the HEAD revert
    const source = TemplateSource{ .preset = .gemma4 };
    var tmpl = try resolve(testing.allocator, source, .gemma4, null, true);
    defer tmpl.deinit(testing.allocator);

    // Content with image placeholder already inserted (as done in main.zig)
    const content_with_placeholder = try ensurePlaceholderInContent("Describe this image", .image, testing.allocator);
    defer if (content_with_placeholder.ptr != "Describe this image".ptr) testing.allocator.free(content_with_placeholder);

    const media = Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } },
    };
    const messages = [_]ChatMessage{
        ChatMessage.withMedia("user", content_with_placeholder, media),
    };
    const result = try tmpl.apply(testing.allocator, &messages, null, true);
    defer testing.allocator.free(result);

    // Verify Gemma4 turn format with image placeholder
    try testing.expect(std.mem.indexOf(u8, result, "<|turn>user") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|image|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<turn|>") != null);
    // Generation prompt
    try testing.expect(std.mem.indexOf(u8, result, "<|turn>model") != null);
}

test "multimodal: ensurePlaceholderInContent idempotent" {
    // Adding placeholder twice should not duplicate it
    const content = "Describe this";
    const first = try ensurePlaceholderInContent(content, .image, testing.allocator);
    defer testing.allocator.free(first);
    try testing.expectEqualStrings("<|image|>Describe this", first);

    // Second call should return same content (placeholder already present)
    const second = try ensurePlaceholderInContent(first, .image, testing.allocator);
    try testing.expect(second.ptr == first.ptr); // Same pointer = no new allocation
}

test "multimodal: Jinja messagesToList does not double-add placeholder" {
    // When content already has a placeholder, messagesToList should NOT add another
    const content_with_placeholder = "<|image|>Describe this";
    const media = Media{
        .type = .image,
        .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } },
    };
    const messages = [_]ChatMessage{
        ChatMessage.withMedia("user", content_with_placeholder, media),
    };

    const template_str = "{{ messages[0]['content'] }}";
    const source = TemplateSource{ .gguf_builtin = template_str };
    var tmpl = try resolve(testing.allocator, source, .qwen2, null, true);
    defer tmpl.deinit(testing.allocator);

    const result = try tmpl.apply(testing.allocator, &messages, null, false);
    defer testing.allocator.free(result);

    // Should have exactly one <|image|>, not two
    const first = std.mem.indexOf(u8, result, "<|image|>");
    try testing.expect(first != null);
    const second = std.mem.indexOf(u8, result[first.? + 1 ..], "<|image|>");
    try testing.expectEqual(@as(?usize, null), second);
}
