//! 编码逻辑
//!
//! 实现文本到 token ID 列表的编码过程。
//! 参考 llama.cpp 的编码流程：
//! 1. 特殊 token 预分词（如果 parse_special=true）：扫描文本中的特殊 token
//! 2. 预分词（Pre-tokenization）：根据预分词器类型分割文本
//! 3. 添加空格前缀（如果配置需要）
//! 4. 逐词编码（Trie 贪婪匹配 + BPE 合并）
//! 5. 添加特殊 token（BOS/EOS）
//!
//! 注意：对于 SPM 模型（tokenizer.ggml.model = "llama"），空格前缀使用 ▁ (U+2581)
//! 而不是普通空格。Trie 中存储的是原始 token 字符串（包含 ▁），所以匹配时
//! 需要使用 ▁ 作为前缀。

const std = @import("std");
// NOTE: 通过 mod 模块的 pub const 导出访问子模块，确保与 mod.zig 使用同一个模块实例
// 避免因不同 @import 路径导致的类型不匹配（如 TrieNode 被创建为多个不兼容的类型）
const mod = @import("mod.zig");
const types = mod.types;
const trie = mod.trie;
const bpe = mod.bpe;
const unicode = mod.unicode;
const preTokenizeFalcon = @import("models/falcon.zig").preTokenizeFalcon;
const preTokenizeMpt = @import("models/mpt.zig").preTokenizeMpt;
const preTokenizeStarcoderStyle = @import("models/starcoder.zig").preTokenizeStarcoderStyle;
const preTokenizeDeepseekLlm = @import("models/deepseek_llm.zig").preTokenizeDeepseekLlm;
const preTokenizeDeepseekCoder = @import("models/deepseek_coder.zig").preTokenizeDeepseekCoder;
const preTokenizeDeepseek3Style = @import("models/deepseek3.zig").preTokenizeDeepseek3Style;
const preTokenizeBloomStyle = @import("models/bloom.zig").preTokenizeBloomStyle;
const preTokenizeGpt2Style = @import("models/gpt2_style.zig").preTokenizeGpt2Style;
const preTokenizeGpt2StyleNoSpace = @import("models/gpt2_style_nospace.zig").preTokenizeGpt2StyleNoSpace;
const preTokenizeLlama3 = @import("models/llama3.zig").preTokenizeLlama3;
const preTokenizeQwen2Style = @import("models/qwen2.zig").preTokenizeQwen2Style;
const preTokenizeGPT2 = @import("models/gpt2.zig").preTokenizeGPT2;
const preTokenizeQwen = @import("models/qwen.zig").preTokenizeQwen;
const preTokenizeNewlineOnly = @import("models/newline_only.zig").preTokenizeNewlineOnly;
const tryMatchContractionOrWord = @import("models/tryMatchContractionOrWord.zig").tryMatchContractionOrWord;
const encode_config = @import("encode_config.zig");
const encode_spm = @import("encode_spm.zig");
const encode_word = @import("encode_word.zig");

const log = std.log.scoped(.tokenizer);

/// SPM 空格标记：▁ (U+2581)，UTF-8 编码为 0xE2 0x96 0x81
const SPM_SPACE = "\xE2\x96\x81";

// ============================================================================
// 预分词器
// ============================================================================

/// 预分词结果：分割后的单词列表
pub const PreTokenized = struct {
    words: std.ArrayListUnmanaged([]const u8),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PreTokenized) void {
        for (self.words.items) |w| {
            self.allocator.free(w);
        }
        self.words.deinit(self.allocator);
    }
};

/// 根据预分词器类型分割文本
pub fn preTokenize(text: []const u8, pre_type: types.PreTokenizerType, allocator: std.mem.Allocator) !PreTokenized {
    var result = PreTokenized{
        .words = .empty,
        .allocator = allocator,
    };

    switch (pre_type) {
        .default => {
            try preTokenizeGPT2(text, &result);
        },
        .llama3 => {
            // llama-bpe, llama-v3, falcon3 等使用 llama3 pre_type 的模型
            // 使用 llama3 正则：\p{N}{1,3} 分组数字，不保留数字前导空格
            try preTokenizeLlama3(text, &result);
        },
        .gpt2 => {
            try preTokenizeGpt2Style(text, &result);
        },
        .gemma4 => {
            try preTokenizeNewlineOnly(text, &result);
        },
        .qwen2, .qwen35 => {
            // Qwen2/Qwen35 使用单数字 \p{N} 模式
            try preTokenizeQwen2Style(text, &result);
        },
        .falcon => {
            try preTokenizeFalcon(text, &result);
        },
        .mpt => {
            try preTokenizeMpt(text, &result);
        },
        .starcoder, .refact, .command_r, .smollm, .codeshell, .exaone, .minerva => {
            try preTokenizeStarcoderStyle(text, &result);
        },
        .deepseek_llm => {
            try preTokenizeDeepseekLlm(text, &result);
        },
        .deepseek_coder => {
            try preTokenizeDeepseekCoder(text, &result);
        },
        .deepseek3_llm, .hunyuan_dense, .joyai_llm => {
            try preTokenizeDeepseek3Style(text, &result);
        },
        .bloom, .poro, .gpt3_finnish => {
            try preTokenizeBloomStyle(text, &result);
        },
        else => {
            const word = try allocator.dupe(u8, text);
            try result.words.append(allocator, word);
        },
    }

    return result;
}

/// SPM 风格预分词：基于 Unicode 脚本分割，不分割 < 和 > 等特殊字符
/// 用于 SPM 模型（tokenizer.ggml.model = "llama"），避免特殊 token 被拆解
pub fn preTokenizeSPM(text: []const u8, allocator: std.mem.Allocator) !PreTokenized {
    var result = PreTokenized{
        .words = .empty,
        .allocator = allocator,
    };

    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) {
        // 检测特殊 token 边界：<...> 形式的 token 保持完整
        if (text[i] == '<') {
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            const close = std.mem.indexOfScalarPos(u8, text, i + 1, '>') orelse {
                i += 1;
                continue;
            };
            const word = try allocator.dupe(u8, text[i .. close + 1]);
            try result.words.append(allocator, word);
            i = close + 1;
            start = i;
        } else if (unicode.isAsciiWhitespace(text[i])) {
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            start = i;
            while (i < text.len and unicode.isAsciiWhitespace(text[i])) {
                i += 1;
            }
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            start = i;
        } else {
            i += 1;
        }
    }

    if (i > start) {
        const word = try allocator.dupe(u8, text[start..i]);
        try result.words.append(allocator, word);
    }

    return result;
}

/// 用于 gemma-4 等 SPM 风格 BPE 模型，在预分词前应用
/// 参考 llama.cpp 的 llama_escape_whitespace()
fn escapeWhitespace(text: []const u8, allocator: std.mem.Allocator) ![]u8 {
    var space_count: usize = 0;
    for (text) |c| {
        if (c == ' ') space_count += 1;
    }
    if (space_count == 0) return allocator.dupe(u8, text);

    const escaped_len = text.len + space_count * 2; // space (1 byte) → ▁ (3 bytes)
    var buf = try allocator.alloc(u8, escaped_len);
    var j: usize = 0;
    for (text) |c| {
        if (c == ' ') {
            buf[j] = 0xE2;
            j += 1;
            buf[j] = 0x96;
            j += 1;
            buf[j] = 0x81;
            j += 1;
        } else {
            buf[j] = c;
            j += 1;
        }
    }
    return buf[0..j];
}

/// 特殊 token 替换结果：文本中的特殊 token 已被替换为 token ID
/// 用于 parse_special 模式下的预处理
const SpecialTokenSegment = struct {
    token_id: u32,
};

const TextOrSpecial = union(enum) {
    text: []const u8,
    special: SpecialTokenSegment,
};

/// 对文本进行特殊 token 扫描，将其中的特殊 token 匹配并分割
/// 返回 text 和 special token 的交替列表
/// 与 llama.cpp 的 tokenizer_st_partition 对应
fn partitionSpecialTokens(
    text: []const u8,
    cache: []const mod.CacheSpecialToken,
    allocator: std.mem.Allocator,
) !std.ArrayListUnmanaged(TextOrSpecial) {
    var result: std.ArrayListUnmanaged(TextOrSpecial) = .empty;
    errdefer {
        for (result.items) |item| {
            if (item == .text) allocator.free(item.text);
        }
        result.deinit(allocator);
    }

    // 使用一个布尔数组标记哪些位置已被特殊 token 覆盖
    var covered = try allocator.alloc(bool, text.len);
    defer allocator.free(covered);
    @memset(covered, false);

    // 按长度降序遍历所有特殊 token（最长优先匹配）
    for (cache) |st| {
        if (st.text.len == 0) continue;
        var pos: usize = 0;
        while (pos + st.text.len <= text.len) {
            if (!covered[pos] and std.mem.eql(u8, text[pos .. pos + st.text.len], st.text)) {
                // 检查覆盖范围是否与其他特殊 token 重叠（只有未覆盖的区域才匹配）
                var all_free = true;
                for (pos..pos + st.text.len) |j| {
                    if (covered[j]) {
                        all_free = false;
                        break;
                    }
                }
                if (all_free) {
                    // 检查属性：control 和 unknown token 仅在 parse_special 时匹配
                    // user_defined token 总是匹配
                    for (pos..pos + st.text.len) |j| {
                        covered[j] = true;
                    }
                }
            }
            pos += 1;
        }
    }

    // 根据 covered 数组构建 text/special 交替列表
    var i: usize = 0;
    while (i < text.len) {
        if (covered[i]) {
            // 找到覆盖位置对应的特殊 token
            // 由于特殊 token 已按长度降序匹配，找到第一个匹配的即可
            const matched = findMatchingSpecial(text, i, cache) orelse {
                // 理论上不会发生，但作为回退
                i += 1;
                continue;
            };
            try result.append(allocator, .{ .special = .{ .token_id = matched.id } });
            i += matched.text.len;
        } else {
            // 收集连续的未覆盖文本
            const start = i;
            while (i < text.len and !covered[i]) {
                i += 1;
            }
            const segment = try allocator.dupe(u8, text[start..i]);
            try result.append(allocator, .{ .text = segment });
        }
    }

    return result;
}

/// 在给定位置查找匹配的特殊 token（假设该位置已被覆盖）
fn findMatchingSpecial(text: []const u8, pos: usize, cache: []const mod.CacheSpecialToken) ?mod.CacheSpecialToken {
    for (cache) |st| {
        if (pos + st.text.len <= text.len and std.mem.eql(u8, text[pos .. pos + st.text.len], st.text)) {
            return st;
        }
    }
    return null;
}

// ============================================================================
// 编码主函数
// ============================================================================

/// 编码：将文本转换为 token ID 列表
/// 完整的编码流程：
/// 1. 特殊 token 预分词（如果 parse_special=true）
/// 2. 预分词（SPM 模型使用 SPM 风格预分词）
/// 3. 添加空格前缀（如果配置需要）
/// 4. 逐词编码（Trie 贪婪匹配 + BPE 合并）
/// 5. 添加特殊 token（BOS/EOS）
pub fn encode(
    text: []const u8,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    ignore_merges: bool,
    parse_special: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (add_bos) {
        try tokens.append(config.allocator, config.special.bos);
    }

    const is_spm_model = config.model == .llama or config.model == .spm;

    // 1a. 特殊 token 预分词（如果 parse_special=true 且有缓存）
    if (parse_special and config.cache_special_tokens != null and config.cache_special_tokens.?.len > 0) {
        var segments = try partitionSpecialTokens(text, config.cache_special_tokens.?, config.allocator);
        defer {
            for (segments.items) |item| {
                if (item == .text) config.allocator.free(item.text);
            }
            segments.deinit(config.allocator);
        }

        for (segments.items) |seg| {
            switch (seg) {
                .text => |txt| {
                    // 对普通文本段进行常规编码
                    var word_tokens = try encodeSegment(txt, add_space_prefix, ignore_merges, is_spm_model, config);
                    defer word_tokens.deinit(config.allocator);
                    try tokens.appendSlice(config.allocator, word_tokens.items);
                },
                .special => |sp| {
                    // 直接添加特殊 token ID
                    try tokens.append(config.allocator, sp.token_id);
                },
            }
        }

        if (add_eos) {
            try tokens.append(config.allocator, config.special.eos);
        }
        return tokens;
    }

    // 1b. 常规编码
    if (is_spm_model) {
        // SPM 模型：使用基于字符的 bigram 合并
        if (text.len == 0) {
            // Empty text: no tokens (unless add_bos/add_eos already handled)
        } else {
            // 1. 添加空格前缀（如果需要）
            // 2. 转义空格（空格 → ▁）
            // 3. 对整个文本进行 SPM 编码
            var spm_text: []const u8 = text;
            var spm_needs_free = false;
            defer {
                if (spm_needs_free) config.allocator.free(@constCast(spm_text));
            }

            // 添加空格前缀
            if (add_space_prefix) {
                spm_text = try std.fmt.allocPrint(config.allocator, " {s}", .{text});
                spm_needs_free = true;
            }

            // 转义空格（空格 → ▁）
            const escaped = try escapeWhitespace(spm_text, config.allocator);
            if (spm_needs_free) config.allocator.free(@constCast(spm_text));
            spm_text = escaped;
            spm_needs_free = true;

            // SPM 编码
            var spm_tokens = try encode_spm.encodeSPM(spm_text, config, config.allocator);
            defer spm_tokens.deinit(config.allocator);
            try tokens.appendSlice(config.allocator, spm_tokens.items);
        }
    } else {
        // BPE 模型：使用预分词 + 逐词编码
        const needs_global_escape = config.escape_whitespaces;
        const escaped_text: ?[]const u8 = if (needs_global_escape)
            try escapeWhitespace(text, config.allocator)
        else
            null;
        defer if (escaped_text) |et| config.allocator.free(et);

        const effective_text = if (escaped_text) |et| et else text;

        var pre_tok = try preTokenize(effective_text, config.pre_type, config.allocator);
        defer pre_tok.deinit();

        var is_first_word = true;
        for (pre_tok.words.items) |word| {
            var word_tokens = try encode_word.encodeWord(word, add_space_prefix, ignore_merges, is_first_word, config);
            defer word_tokens.deinit(config.allocator);
            try tokens.appendSlice(config.allocator, word_tokens.items);
            is_first_word = false;
        }
    }

    if (add_eos) {
        try tokens.append(config.allocator, config.special.eos);
    }

    return tokens;
}

/// 编码单个文本段（用于 parse_special 模式下的普通文本部分）
fn encodeSegment(
    text: []const u8,
    add_space_prefix: bool,
    ignore_merges: bool,
    is_spm_model: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (text.len == 0) return tokens;

    // 对于 SPM 风格 BPE 模型（gemma-4 等），先全局转义空格再预分词
    const needs_global_escape = config.escape_whitespaces and !is_spm_model;
    const escaped_text: ?[]const u8 = if (needs_global_escape)
        try escapeWhitespace(text, config.allocator)
    else
        null;
    defer if (escaped_text) |et| config.allocator.free(et);

    const effective_text = if (escaped_text) |et| et else text;

    var pre_tok = if (is_spm_model)
        try preTokenizeSPM(effective_text, config.allocator)
    else
        try preTokenize(effective_text, config.pre_type, config.allocator);
    defer pre_tok.deinit();

    var is_first_word = true;
    for (pre_tok.words.items) |word| {
        if (is_spm_model and unicode.isAllWhitespace(word)) {
            is_first_word = false;
            continue;
        }

        var word_tokens = try encode_word.encodeWord(word, add_space_prefix, ignore_merges, is_first_word, config);
        defer word_tokens.deinit(config.allocator);
        try tokens.appendSlice(config.allocator, word_tokens.items);
        is_first_word = false;
    }

    return tokens;
}

pub const EncodeConfig = encode_config.EncodeConfig;
