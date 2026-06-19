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
        .default, .llama3 => {
            try preTokenizeGPT2(text, &result);
        },
        .gpt2 => {
            try preTokenizeGpt2Style(text, &result);
        },
        .gemma4 => {
            try preTokenizeNewlineOnly(text, &result);
        },
        .qwen2, .qwen35 => {
            try preTokenizeQwen(text, &result);
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
        } else if (isWhitespace(text[i])) {
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            start = i;
            while (i < text.len and isWhitespace(text[i])) {
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

/// Falcon 风格预分词
/// regex: [\p{P}\$\+<=\>\^~\|`]+ | 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S) | [0-9][0-9][0-9]
fn preTokenizeFalcon(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Punctuation and symbols: [\p{P}\$\+<=\>\^~\|`]+
        if (isPunctuationOrSymbol(text[i])) {
            const start = i;
            i += 1;
            while (i < text.len and isPunctuationOrSymbol(text[i])) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 2. Optional space + Unicode digits:  ?\p{N}+ (for digits like ½ ² ³)
        if (text[i] == ' ' and i + 1 < text.len and isUnicodeDigit(text, i + 1)) {
            const start = i;
            i += 1;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeDigit(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        if (isUnicodeDigit(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeDigit(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3. Contractions and word patterns (ASCII only): 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
        if (try tryMatchContractionOrWord(text, &i, result)) continue;

        // 4. Three-digit numbers: [0-9][0-9][0-9]
        if (i + 2 < text.len and isUnicodeDigit(text, i) and isUnicodeDigit(text, i+1) and isUnicodeDigit(text, i+2)) {
            const start = i;
            i += 3;
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // Fallback: single character
        const ch_len = utf8CharLen(text, i);
        const word = try result.allocator.dupe(u8, text[i..i+ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}

/// MPT 风格预分词
/// MPT 使用 GPT-2 风格 regex，与 GPT-2 共享相同的预分词逻辑。
/// 参考 llama.cpp: LLAMA_VOCAB_PRE_TYPE_MPT uses the same regex as GPT-2
fn preTokenizeMpt(text: []const u8, result: *PreTokenized) !void {
    try preTokenizeGpt2Style(text, result);
}

/// Starcoder/Refact/Command-R 风格预分词
/// regex: \p{N} | 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
///
/// 注意：\p{N} 匹配单个 Unicode 数字字符（包括 ¼ ½ ¾ ² ³ ¹ 等），
/// 在 tryMatchContractionOrWord 的 ?\p{N}+ 模式之前检查。
fn preTokenizeStarcoderStyle(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Single digit: \p{N} (ASCII and Unicode digits like ¼ ½ ¾ ² ³ ¹)
        if (isDigit(text[i]) or isUnicodeNumberChar(text, i)) {
            const ch_len = if (isDigit(text[i])) @as(usize, 1) else utf8CharLen(text, i);
            const word = try result.allocator.dupe(u8, text[i..i+ch_len]);
            try result.words.append(result.allocator, word);
            i += ch_len;
            continue;
        }

        // 2. Contractions and word patterns
        if (try tryMatchContractionOrWord(text, &i, result)) continue;

        // Fallback
        const ch_len = utf8CharLen(text, i);
        const word = try result.allocator.dupe(u8, text[i..i+ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}

/// 检测给定位置的 UTF-8 字符是否为 Unicode 数字字符 (\p{N})
/// 保守实现：只匹配已知的 Unicode 数字字符，避免将 CJK/emoji 等 3 字节字符误判为数字
/// 匹配的字符包括：
///   - 2-byte: ² (U+00B2), ³ (U+00B3), ¹ (U+00B9), ¼ (U+00BC), ½ (U+00BD), ¾ (U+00BE)
///   - 其他 Unicode 数字字符暂不扩展（避免误判）
fn isUnicodeNumberChar(text: []const u8, pos: usize) bool {
    if (pos >= text.len) return false;
    const b = text[pos];

    // ASCII digits are handled by isDigit in GPT-2 style patterns
    if (b < 0x80) return false;

    // Continuation bytes are not valid start of a character
    if (b < 0xC0) return false;

    // 2-byte UTF-8 (U+0080 ~ U+07FF)
    if (b >= 0xC2 and b < 0xE0 and pos + 1 < text.len) {
        const b2 = text[pos + 1];
        // Common number characters: ² ³ ¹ ¼ ½ ¾ (U+00B2, B3, B9, BC, BD, BE)
        if (b == 0xC2) {
            return switch (b2) {
                0xB2, 0xB3, 0xB9, 0xBC, 0xBD, 0xBE => true,
                else => false,
            };
        }
        return false;
    }

    // 3-byte and 4-byte: conservatively return false to avoid misclassifying
    // CJK characters, emojis, etc. as digits
    return false;
}

/// DeepSeek-LLM 风格预分词
fn preTokenizeDeepseekLlm(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Newlines: [\r\n]
        if (text[i] == '\r' or text[i] == '\n') {
            const start = i;
            i += 1;
            while (i < text.len and (text[i] == '\r' or text[i] == '\n')) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 2. Latin letters with optional space: \s?[A-Za-zµÀ-ÖØ-öø-ƺ...]+
        if (isLatinLetter(text[i])) {
            const has_space = (i > 0 and text[i-1] == ' ');
            const start = if (has_space) i - 1 else i;
            if (has_space) i += 0; // already at correct position
            i += utf8CharLen(text, i);
            while (i < text.len and isLatinLetter(text[i])) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3. Punctuation with optional space: \s?[!-\/:-~！-／：-～‘-‟　-。]+
        if (isPunctuationOrSymbol(text[i])) {
            const has_space = (i > 0 and text[i-1] == ' ');
            const start = if (has_space) i - 1 else i;
            if (has_space) i += 0;
            i += utf8CharLen(text, i);
            while (i < text.len and isPunctuationOrSymbol(text[i])) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 4. Whitespace: capture standalone whitespace
        if (isWhitespace(text[i])) {
            const start = i;
            while (i < text.len and isWhitespace(text[i])) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. CJK characters: [一-龥...]+
        if (isCJK(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isCJK(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 6. Numbers: \p{N}+
        if (isUnicodeDigit(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeDigit(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // Fallback
        i += 1;
    }
}

/// DeepSeek-Coder 风格预分词
/// regex: [\r\n] | \s?\p{L}+ | \s?\p{P}+ | [一-龥ࠀ-一가-퟿]+ | \p{N}
fn preTokenizeDeepseekCoder(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Newlines: [\r\n]
        if (text[i] == '\r' or text[i] == '\n') {
            const start = i;
            i += 1;
            while (i < text.len and (text[i] == '\r' or text[i] == '\n')) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 2. Letters with optional space: \s?\p{L}+
        if (text[i] == ' ' and i + 1 < text.len and isUnicodeLetter(text, i + 1)) {
            const start = i;
            i += 1;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeLetter(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        if (isUnicodeLetter(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeLetter(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3. Punctuation with optional space: \s?\p{P}+
        if (text[i] == ' ' and i + 1 < text.len and isPunctuationOrSymbol(text[i + 1])) {
            const start = i;
            i += 1;
            i += utf8CharLen(text, i);
            while (i < text.len and isPunctuationOrSymbol(text[i])) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        if (isPunctuationOrSymbol(text[i])) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isPunctuationOrSymbol(text[i])) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 4. CJK: [一-龥ࠀ-一가-퟿]+
        if (isCJK(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isCJK(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. Single digit: \p{N}
        if (isUnicodeDigit(text, i)) {
            const ch_len = utf8CharLen(text, i);
            const word = try result.allocator.dupe(u8, text[i..i+ch_len]);
            try result.words.append(result.allocator, word);
            i += ch_len;
            continue;
        }

        // 6. Whitespace: capture standalone whitespace (after letters/punct/digits)
        if (isWhitespace(text[i])) {
            const start = i;
            while (i < text.len and isWhitespace(text[i])) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // Fallback: single character
        i += 1;
    }
}

/// DeepSeek3 / Hunyuan-Dense / JoyAI-LLM 风格预分词
fn preTokenizeDeepseek3Style(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Numbers in groups of 1-3: \p{N}{1,3}
        if (isUnicodeDigit(text, i)) {
            const start = i;
            var count: usize = 0;
            var pos = i;
            while (pos < text.len and isUnicodeDigit(text, pos) and count < 3) {
                const ch_len = utf8CharLen(text, pos);
                pos += ch_len;
                count += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..pos]);
            try result.words.append(result.allocator, word);
            i = pos;
            continue;
        }

        // 2. CJK and Asian scripts
        if (isCJK(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isCJK(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3. Main BPE pattern: punctuation+letters, letters, punctuation, whitespace
        if (try tryMatchContractionOrWord(text, &i, result)) continue;

        // Fallback
        const ch_len = utf8CharLen(text, i);
        const word = try result.allocator.dupe(u8, text[i..i+ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}

/// Bloom/Poro/GPT3-Finnish 风格预分词
/// regex: ?[^(\s|.,!?…。，、।۔،)]+
fn preTokenizeBloomStyle(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // Skip whitespace and punctuation separators
        if (isWhitespace(text[i]) or isBloomSeparator(text[i])) {
            const start = i;
            i += 1;
            while (i < text.len and (isWhitespace(text[i]) or isBloomSeparator(text[i]))) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // Collect non-separator characters
        const start = i;
        i += utf8CharLen(text, i);
        while (i < text.len and !isWhitespace(text[i]) and !isBloomSeparator(text[i])) {
            i += utf8CharLen(text, i);
        }
        const word = try result.allocator.dupe(u8, text[start..i]);
        try result.words.append(result.allocator, word);
    }
}

fn isBloomSeparator(c: u8) bool {
    return switch (c) {
        '.', ',', '!', '?' => true,
        else => false,
    };
}

/// Try to match contraction or word pattern (GPT-2 style)
/// Returns true if a match was found and appended
fn tryMatchContractionOrWord(text: []const u8, i: *usize, result: *PreTokenized) !bool {
    // Contractions: 's, 't, 're, 've, 'm, 'll, 'd
    // Note: only match the contraction suffix itself, not the preceding word.
    // The preceding word is matched by the ?\p{L}+ pattern.
    if (i.* + 1 < text.len and text[i.*] == '\'') {
        const suffix = text[i.*+1..];
        if (suffix.len >= 1 and (suffix[0] == 's' or suffix[0] == 'S' or
            suffix[0] == 't' or suffix[0] == 'T' or
            suffix[0] == 'm' or suffix[0] == 'M' or
            suffix[0] == 'd' or suffix[0] == 'D')) {
            const word = try result.allocator.dupe(u8, text[i.*..i.*+2]);
            try result.words.append(result.allocator, word);
            i.* = i.* + 2;
            return true;
        }
        if (suffix.len >= 2 and ((suffix[0] == 'r' and suffix[1] == 'e') or
            (suffix[0] == 'R' and suffix[1] == 'E') or
            (suffix[0] == 'v' and suffix[1] == 'e') or
            (suffix[0] == 'V' and suffix[1] == 'E') or
            (suffix[0] == 'l' and suffix[1] == 'l') or
            (suffix[0] == 'L' and suffix[1] == 'L'))) {
            const word = try result.allocator.dupe(u8, text[i.*..i.*+3]);
            try result.words.append(result.allocator, word);
            i.* = i.* + 3;
            return true;
        }
    }

    // Optional space + letters:  ?\p{L}+
    if (text[i.*] == ' ' and i.* + 1 < text.len and isUnicodeLetter(text, i.* + 1)) {
        const start = i.*;
        i.* += 1;
        i.* += utf8CharLen(text, i.*);
        while (i.* < text.len and isUnicodeLetter(text, i.*)) {
            i.* += utf8CharLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }
    if (isUnicodeLetter(text, i.*)) {
        const start = i.*;
        i.* += utf8CharLen(text, i.*);
        while (i.* < text.len and isUnicodeLetter(text, i.*)) {
            i.* += utf8CharLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }

    // Optional space + digits:  ?\p{N}+
    if (text[i.*] == ' ' and i.* + 1 < text.len and isDigit(text[i.* + 1])) {
        const start = i.*;
        i.* += 1;
        i.* += 1;
        while (i.* < text.len and isDigit(text[i.*])) {
            i.* += 1;
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }
    if (isDigit(text[i.*])) {
        const start = i.*;
        i.* += 1;
        while (i.* < text.len and isDigit(text[i.*])) {
            i.* += 1;
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }

    // Optional space + symbols:  ?[^\s\p{L}\p{N}]+
    if (text[i.*] == ' ' and i.* + 1 < text.len and
        !isWhitespace(text[i.* + 1]) and
        !isUnicodeLetter(text, i.* + 1) and
        !isDigit(text[i.* + 1]) and
        !isUnicodeNumberChar(text, i.* + 1))
    {
        const start = i.*;
        i.* += 1;
        i.* += utf8CharLen(text, i.*);
        while (i.* < text.len and
            !isWhitespace(text[i.*]) and
            !isUnicodeLetter(text, i.*) and
            !isDigit(text[i.*]) and
            !isUnicodeNumberChar(text, i.*))
        {
            i.* += utf8CharLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }
    if (!isWhitespace(text[i.*]) and !isUnicodeLetter(text, i.*) and !isDigit(text[i.*]) and !isUnicodeNumberChar(text, i.*)) {
        const start = i.*;
        i.* += utf8CharLen(text, i.*);
        while (i.* < text.len and
            !isWhitespace(text[i.*]) and
            !isUnicodeLetter(text, i.*) and
            !isDigit(text[i.*]) and
            !isUnicodeNumberChar(text, i.*))
        {
            i.* += utf8CharLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }

    // Whitespace: \s+(?!\S) or \s+
    if (isWhitespace(text[i.*])) {
        var ws_count: usize = 0;
        while (i.* + ws_count < text.len and isWhitespace(text[i.* + ws_count])) {
            ws_count += 1;
        }

        // \s+(?!\S): if whitespace is followed by non-whitespace and count > 1, take n-1
        if (ws_count > 1 and i.* + ws_count < text.len) {
            const word = try result.allocator.dupe(u8, text[i.* .. i.* + ws_count - 1]);
            try result.words.append(result.allocator, word);
            i.* += ws_count - 1;
            return true;
        }

        // \s+: regular whitespace
        const word = try result.allocator.dupe(u8, text[i.* .. i.* + ws_count]);
        try result.words.append(result.allocator, word);
        i.* += ws_count;
        return true;
    }

    return false;
}

/// GPT-2 风格预分词（用于 gpt2/gemma4 pre_type）
/// 使用 GPT-2 原始正则模式：
/// 1. 收缩形式（'s, 't, 're, 've, 'm, 'll, 'd）
/// 2. ?\p{L}+ : 可选空格 + 字母序列
/// 3. ?\p{N}+ : 可选空格 + 数字序列
/// 4. ?[^\s\p{L}\p{N}]+ : 可选空格 + 符号序列
/// 5. \s+(?!\S) : 尾随空白
/// 参考 llama.cpp GPT2 pre_type regex
fn preTokenizeGpt2Style(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. 检查收缩形式（仅匹配收缩后缀本身，如 's, 't, 're, 've, 'm, 'll, 'd）
        // 对应 regex: 's|'t|'re|'ve|'m|'ll|'d
        // 注意：不包含前面的单词，前面的单词由 ?\p{L}+ 模式匹配
        if (i + 1 < text.len and text[i] == '\'') {
            const suffix = text[i+1..];
            if (suffix.len >= 1 and (suffix[0] == 's' or suffix[0] == 'S' or
                suffix[0] == 't' or suffix[0] == 'T' or
                suffix[0] == 'm' or suffix[0] == 'M' or
                suffix[0] == 'd' or suffix[0] == 'D'))
            {
                const word = try result.allocator.dupe(u8, text[i..i+2]);
                try result.words.append(result.allocator, word);
                i = i + 2;
                continue;
            }
            if (suffix.len >= 2 and ((suffix[0] == 'r' and suffix[1] == 'e') or
                (suffix[0] == 'R' and suffix[1] == 'E') or
                (suffix[0] == 'v' and suffix[1] == 'e') or
                (suffix[0] == 'V' and suffix[1] == 'E') or
                (suffix[0] == 'l' and suffix[1] == 'l') or
                (suffix[0] == 'L' and suffix[1] == 'L')))
            {
                const word = try result.allocator.dupe(u8, text[i..i+3]);
                try result.words.append(result.allocator, word);
                i = i + 3;
                continue;
            }
        }

        // 2. 可选空格 + 字母序列:  ?\p{L}+
        const has_space = (text[i] == ' ');
        const check_pos = if (has_space) i + 1 else i;
        if (check_pos < text.len and isUnicodeLetter(text, check_pos)) {
            const start = i;
            if (has_space) i += 1;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeLetter(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3. 可选空格 + 数字序列:  ?\p{N}+
        if (check_pos < text.len and isUnicodeDigit(text, check_pos)) {
            const start = i;
            if (has_space) i += 1;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeDigit(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 4. 可选空格 + 符号序列:  ?[^\s\p{L}\p{N}]+
        if (check_pos < text.len and !isWhitespace(text[check_pos]) and
            !isUnicodeLetter(text, check_pos) and !isUnicodeDigit(text, check_pos)) {
            const start = i;
            if (has_space) i += 1;
            i += utf8CharLen(text, i);
            while (i < text.len and !isWhitespace(text[i]) and
                !isUnicodeLetter(text, i) and !isUnicodeDigit(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. 空白序列：先尝试 \s+(?!\S)（后面有非空白时只取前 n-1 个），再尝试 \s+
        if (isWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and isWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // \s+(?!\S)：如果空白后面有非空白字符，且空白数 > 1，只取前 n-1 个
            if (ws_count > 1 and i + ws_count < text.len) {
                const word = try result.allocator.dupe(u8, text[i .. i + ws_count - 1]);
                try result.words.append(result.allocator, word);
                i += ws_count - 1;
                continue;
            }

            // \s+：普通空白序列
            const word = try result.allocator.dupe(u8, text[i .. i + ws_count]);
            try result.words.append(result.allocator, word);
            i += ws_count;
            continue;
        }

        // Skip any remaining unrecognized character
        i += 1;
    }
}

/// GPT-2 风格预分词（用于 llama3/default pre_type）
/// 使用 llama3 regex 模式：
/// 1. 收缩形式（'s, 't, 're, 've, 'm, 'll, 'd）
/// 2. [^\s\p{L}\p{N}]?\p{L}+ : 可选前导符号 + 字母序列
/// 3. \p{N}{1,3} : 数字序列（1-3位）
/// 4.  ?[^\s\p{L}\p{N}]+ : 空格 + 符号序列
/// 5. 空白序列：\s+
/// 参考 llama.cpp 的 GPT-2 regex
fn preTokenizeGPT2(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. 检查收缩形式：'s, 't, 're, 've, 'm, 'll, 'd
        if (i + 1 < text.len and text[i] == '\'') {
            const suffix = text[i+1..];
            if (suffix.len >= 1 and (suffix[0] == 's' or suffix[0] == 'S' or
                suffix[0] == 't' or suffix[0] == 'T' or
                suffix[0] == 'm' or suffix[0] == 'M' or
                suffix[0] == 'd' or suffix[0] == 'D')) {
                var word_start = i;
                while (word_start > 0 and !isWhitespace(text[word_start - 1])) {
                    word_start -= 1;
                }
                const word = try result.allocator.dupe(u8, text[word_start..i+2]);
                try result.words.append(result.allocator, word);
                i = i + 2;
                continue;
            }
            if (suffix.len >= 2 and ((suffix[0] == 'r' and suffix[1] == 'e') or
                (suffix[0] == 'R' and suffix[1] == 'E') or
                (suffix[0] == 'v' and suffix[1] == 'e') or
                (suffix[0] == 'V' and suffix[1] == 'E') or
                (suffix[0] == 'l' and suffix[1] == 'l') or
                (suffix[0] == 'L' and suffix[1] == 'L'))) {
                var word_start = i;
                while (word_start > 0 and !isWhitespace(text[word_start - 1])) {
                    word_start -= 1;
                }
                const word = try result.allocator.dupe(u8, text[word_start..i+3]);
                try result.words.append(result.allocator, word);
                i = i + 3;
                continue;
            }
        }

        // 2. [^\s\p{L}\p{N}]?\p{L}+ : 可选前导符号 + 字母序列
        // Check for optional leading punctuation/symbol before letters
        if (!isWhitespace(text[i]) and !isUnicodeLetter(text, i) and !isUnicodeDigit(text, i)) {
            // Possible leading symbol
            if (i + 1 < text.len and isUnicodeLetter(text, i + 1)) {
                const start = i;
                i += utf8CharLen(text, i);
                i += utf8CharLen(text, i);
                while (i < text.len and isUnicodeLetter(text, i)) {
                    i += utf8CharLen(text, i);
                }
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
                continue;
            }
        }
        if (isUnicodeLetter(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and isUnicodeLetter(text, i)) {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3a. 可选空格 + 数字序列:  ?\p{N}{1,3}
        if (text[i] == ' ' and i + 1 < text.len and isUnicodeDigit(text, i + 1)) {
            const start = i;
            i += 1;
            var count: usize = 0;
            var pos = i;
            while (pos < text.len and isUnicodeDigit(text, pos) and count < 3) {
                const ch_len = utf8CharLen(text, pos);
                pos += ch_len;
                count += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..pos]);
            try result.words.append(result.allocator, word);
            i = pos;
            continue;
        }
        // 3b. \p{N}{1,3} : 数字序列（1-3位）
        if (isUnicodeDigit(text, i)) {
            const start = i;
            var count: usize = 0;
            var pos = i;
            while (pos < text.len and isUnicodeDigit(text, pos) and count < 3) {
                const ch_len = utf8CharLen(text, pos);
                pos += ch_len;
                count += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..pos]);
            try result.words.append(result.allocator, word);
            i = pos;
            continue;
        }

        // 4.  ?[^\s\p{L}\p{N}]+ : 空格 + 符号序列
        if (text[i] == ' ' and i + 1 < text.len and
            !isWhitespace(text[i + 1]) and
            !isUnicodeLetter(text, i + 1) and
            !isUnicodeDigit(text, i + 1))
        {
            const start = i;
            i += 1;
            i += utf8CharLen(text, i);
            while (i < text.len and
                !isWhitespace(text[i]) and
                !isUnicodeLetter(text, i) and
                !isUnicodeDigit(text, i))
            {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        if (!isWhitespace(text[i]) and !isUnicodeLetter(text, i) and !isUnicodeDigit(text, i)) {
            const start = i;
            i += utf8CharLen(text, i);
            while (i < text.len and
                !isWhitespace(text[i]) and
                !isUnicodeLetter(text, i) and
                !isUnicodeDigit(text, i))
            {
                i += utf8CharLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. 空白序列：放在所有"可选空格"模式之后
        //    先尝试 \s+(?!\S)（尾随空白，后面有非空白时只取前 n-1 个），再尝试 \s+
        if (isWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and isWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // \s+(?!\S)：如果空白后面有非空白字符，且空白数 > 1，只取前 n-1 个
            // 这样剩下的一个空格可以被后面的 ?\p{L}+ 等模式匹配
            if (ws_count > 1 and i + ws_count < text.len) {
                const word = try result.allocator.dupe(u8, text[i .. i + ws_count - 1]);
                try result.words.append(result.allocator, word);
                i += ws_count - 1;
                continue;
            }

            // \s+：普通空白序列
            const word = try result.allocator.dupe(u8, text[i .. i + ws_count]);
            try result.words.append(result.allocator, word);
            i += ws_count;
            continue;
        }

        // Fallback: skip any unrecognized character
        i += 1;
    }
}

/// Qwen 风格预分词
fn preTokenizeQwen(text: []const u8, result: *PreTokenized) !void {
    try preTokenizeGPT2(text, result);
}

/// Gemma-4 风格预分词：仅按换行符分割
/// 空格由 escape_whitespaces 在编码前全局替换为 ▁ (U+2581)
/// 参考 llama.cpp LLAMA_VOCAB_PRE_TYPE_GEMMA4: regex "[^\n]+|[\n]+"
fn preTokenizeNewlineOnly(text: []const u8, result: *PreTokenized) !void {
    var start: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\n') {
            if (i > start) {
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
            }
            start = i;
            while (i < text.len and text[i] == '\n') {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            start = i;
        } else {
            i += 1;
        }
    }
    if (i > start) {
        const word = try result.allocator.dupe(u8, text[start..i]);
        try result.words.append(result.allocator, word);
    }
}

/// 全局空格转义：将文本中的所有空格 ' ' (0x20) 替换为 ▁ (U+2581, UTF-8: E2 96 81)
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
            buf[j] = 0xE2; j += 1;
            buf[j] = 0x96; j += 1;
            buf[j] = 0x81; j += 1;
        } else {
            buf[j] = c; j += 1;
        }
    }
    return buf[0..j];
}

fn isWhitespace(c: u8) bool {
    return switch (c) {
        ' ', '\t', '\n', '\r', 0x0B, 0x0C => true,
        else => false,
    };
}

fn isLetter(c: u8) bool {
    return switch (c) {
        'a'...'z', 'A'...'Z' => true,
        else => false,
    };
}

/// 检测给定位置的 UTF-8 字符是否为 Unicode 字母 (\p{L})
/// 支持多字节 UTF-8 序列，用于 GPT-2 预分词器
fn isUnicodeLetter(text: []const u8, pos: usize) bool {
    if (pos >= text.len) return false;
    const b = text[pos];
    if (b < 0x80) return isLetter(b);
    // Continuation bytes (0x80-0xBF) are not valid start of a character
    if (b < 0xC0) return false;
    // 2-byte sequence (0xC0-0xDF)
    if (b < 0xE0 and pos + 1 < text.len) {
        // 0xC3+ covers À-ÿ (Latin letters with diacritics) and beyond
        // 0xC2 covers control chars, symbols, fractions (not letters)
        if (b >= 0xC3) return true;
        return false;
    }
    // 3-byte (0xE0-0xEF) and 4-byte (0xF0-0xF7): likely letters/CJK
    return true;
}

/// 检测给定位置的 UTF-8 字符是否为 Unicode 数字 (\p{N})
fn isUnicodeDigit(text: []const u8, pos: usize) bool {
    if (pos >= text.len) return false;
    const b = text[pos];
    if (b < 0x80) return b >= '0' and b <= '9';
    // Continuation bytes: not valid character starts
    if (b < 0xC0) return false;
    // 2-byte sequence
    if (b < 0xE0 and pos + 1 < text.len) {
        if (b == 0xC2) {
            const b2 = text[pos + 1];
            return (b2 == 0xB2 or b2 == 0xB3 or b2 == 0xB9 or
                b2 == 0xBC or b2 == 0xBD or b2 == 0xBE);
        }
        return false;
    }
    return false;
}

fn isDigit(c: u8) bool {
    return switch (c) {
        '0'...'9' => true,
        else => false,
    };
}

fn isPunctuationOrSymbol(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

fn isLatinLetter(c: u8) bool {
    return isLetter(c) or c >= 0xC0; // rough approximation for Latin-1 supplement
}

fn isCJK(text: []const u8, pos: usize) bool {
    if (pos >= text.len) return false;
    const b = text[pos];
    // CJK characters are in 3-byte UTF-8 range (0xE0-0xEF)
    if (b >= 0xE0 and b < 0xF0) return true;
    return false;
}

fn isAllWhitespace(s: []const u8) bool {
    for (s) |c| {
        if (!isWhitespace(c)) return false;
    }
    return true;
}

/// 获取给定位置 UTF-8 字符的字节长度
fn utf8CharLen(text: []const u8, pos: usize) usize {
    if (pos >= text.len) return 0;
    const b = text[pos];
    if (b < 0x80) return 1;
    if (b < 0xC0) return 1; // continuation byte, treat as single byte
    if (b < 0xE0) return @min(2, text.len - pos);
    if (b < 0xF0) return @min(3, text.len - pos);
    return @min(4, text.len - pos);
}

// ============================================================================
// GPT-2 字节编码转换
// ============================================================================

/// 将文本转换为 GPT-2 字节编码
pub fn toGpt2ByteEncoding(text: []const u8, bytesToUnicodeFn: *const fn (byte: u8, ctx: ?*anyopaque) []const u8, ctx: ?*anyopaque, allocator: std.mem.Allocator) ![]u8 {
    var total_len: usize = 0;
    for (text) |byte| {
        total_len += bytesToUnicodeFn(byte, ctx).len;
    }

    var result = try std.ArrayListUnmanaged(u8).initCapacity(allocator, total_len);
    errdefer result.deinit(allocator);

    for (text) |byte| {
        const mapped = bytesToUnicodeFn(byte, ctx);
        try result.appendSlice(allocator, mapped);
    }

    return result.items;
}

// ============================================================================
// 特殊 Token 预分词（parse_special 模式）
// ============================================================================

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
                    if (covered[j]) { all_free = false; break; }
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
// SPM 编码（基于字符的 bigram 合并）
// ============================================================================

/// SPM bigram 用于优先级队列
const SpmBigram = struct {
    left: i32,
    right: i32,
    score: f32,
    size: usize,

    fn lessThan(context: void, a: @This(), b: @This()) std.math.Order {
        _ = context;
        // SPM: higher score = higher priority (less negative = more likely to merge)
        // Matches llama.cpp: l.score < r.score means l has lower priority
        if (a.score > b.score) return .lt;
        if (a.score < b.score) return .gt;
        if (a.left > b.left) return .lt;
        if (a.left < b.left) return .gt;
        return .eq;
    }
};

/// SPM 符号
const SpmSymbol = struct {
    text: []const u8, // pointer into the original text
    n: usize,
    prev: i32,
    next: i32,
};

/// SPM 编码：将文本编码为 token ID 列表
/// 参考 llama.cpp 的 llm_tokenizer_spm_session::tokenize
/// 流程：
/// 1. 将文本拆分为 UTF-8 字符
/// 2. 使用优先级队列反复合并 score 最高的相邻 token 对
/// 3. 合并后的 token 字符串必须在词表中存在
fn encodeSPM(
    text: []const u8,
    config: *const EncodeConfig,
    allocator: std.mem.Allocator,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (text.len == 0) return tokens;

    // 1. 拆分为 UTF-8 字符
    var symbols = std.ArrayListUnmanaged(SpmSymbol){ .items = &.{}, .capacity = 0 };
    defer symbols.deinit(allocator);

    var offs: usize = 0;
    var index: i32 = 0;
    while (offs < text.len) {
        const ch_len = utf8CharLen(text, offs);
        try symbols.append(allocator, SpmSymbol{
            .text = text[offs..],
            .n = ch_len,
            .prev = index - 1,
            .next = if (offs + ch_len >= text.len) -1 else index + 1,
        });
        offs += ch_len;
        index += 1;
    }

    if (symbols.items.len == 0) return tokens;

    // 2. 初始化优先级队列
    var work_queue = std.PriorityQueue(SpmBigram, void, SpmBigram.lessThan).initContext({});
    defer work_queue.deinit(allocator);

    for (1..symbols.items.len) |i| {
        tryAddSpmBigram(&symbols, &work_queue, @intCast(i - 1), @intCast(i), config);
    }

    // 3. 反复合并 score 最高的 pair
    while (work_queue.count() > 0) {
        const bigram = work_queue.pop().?;

        const left_idx = @as(usize, @intCast(bigram.left));
        const right_idx = @as(usize, @intCast(bigram.right));

        // 检查符号是否已被合并
        if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) continue;
        if (symbols.items[left_idx].next != bigram.right) continue;
        if (symbols.items[right_idx].prev != bigram.left) continue;

        // 检查 size 是否匹配
        if (symbols.items[left_idx].n + symbols.items[right_idx].n != bigram.size) continue;

        // 合并右符号到左符号
        symbols.items[left_idx].n += symbols.items[right_idx].n;
        symbols.items[right_idx].n = 0;

        // 从链表中移除右符号
        symbols.items[left_idx].next = symbols.items[right_idx].next;
        if (symbols.items[right_idx].next >= 0) {
            symbols.items[@as(usize, @intCast(symbols.items[right_idx].next))].prev = bigram.left;
        }

        // 添加新的 bigram
        tryAddSpmBigram(&symbols, &work_queue, symbols.items[left_idx].prev, bigram.left, config);
        tryAddSpmBigram(&symbols, &work_queue, bigram.left, symbols.items[left_idx].next, config);
    }

    // 4. 从符号链表重建 token 列表
    {
        var i: i32 = 0;
        while (i >= 0) {
            const idx = @as(usize, @intCast(i));
            const sym = &symbols.items[idx];
            if (sym.n > 0) {
                const token_text = sym.text[0..sym.n];
                // 查找 token
                if (config.textToTokenFn(token_text, config.ctx)) |tid| {
                    try tokens.append(allocator, tid);
                } else {
                    // 回退到字节 token
                    for (token_text) |byte| {
                        const tid = config.byteToTokenIdFn(byte, config.ctx);
                        try tokens.append(allocator, tid);
                    }
                }
            }
            i = sym.next;
        }
    }

    return tokens;
}

/// 尝试添加 SPM bigram 到优先级队列
fn tryAddSpmBigram(
    symbols: *std.ArrayListUnmanaged(SpmSymbol),
    queue: *std.PriorityQueue(SpmBigram, void, SpmBigram.lessThan),
    left: i32,
    right: i32,
    config: *const EncodeConfig,
) void {
    if (left < 0 or right < 0) return;
    const left_idx = @as(usize, @intCast(left));
    const right_idx = @as(usize, @intCast(right));
    if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) return;

    const left_sym = &symbols.items[left_idx];
    const right_sym = &symbols.items[right_idx];

    // 构建合并后的文本
    const merged_text = left_sym.text[0 .. left_sym.n + right_sym.n];

    // 查找合并后的 token
    const token_id = config.textToTokenFn(merged_text, config.ctx) orelse return;

    const score = config.tokenScoreFn(token_id, config.ctx);

    queue.push(config.allocator, SpmBigram{
        .left = left,
        .right = right,
        .score = score,
        .size = merged_text.len,
    }) catch {};
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
            var spm_tokens = try encodeSPM(spm_text, config, config.allocator);
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
            var word_tokens = try encodeWord(word, add_space_prefix, ignore_merges, is_first_word, config);
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
        if (is_spm_model and isAllWhitespace(word)) {
            is_first_word = false;
            continue;
        }

        var word_tokens = try encodeWord(word, add_space_prefix, ignore_merges, is_first_word, config);
        defer word_tokens.deinit(config.allocator);
        try tokens.appendSlice(config.allocator, word_tokens.items);
        is_first_word = false;
    }

    return tokens;
}

/// 编码单个词
fn encodeWord(
    word: []const u8,
    add_space_prefix: bool,
    ignore_merges: bool,
    is_first: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    const is_spm_model = config.model == .llama or config.model == .spm;

    // 步骤 1：确定基础文本（可能添加空格前缀）
    // 对于 escape_whitespaces 的模型（gemma-4 等），将空格转为 ▁ (U+2581)
    // 对于 SPM 模型，首词不添加空格前缀（除非词本身以空格开头）
    const BaseText = struct {
        text: []const u8,
        needs_free: bool,
    };
    const base = if (add_space_prefix and (!is_spm_model or !is_first)) blk: {
        if (is_spm_model) {
            if (word.len > 0 and isWhitespace(word[0])) {
                var ws_end: usize = 1;
                while (ws_end < word.len and isWhitespace(word[ws_end])) ws_end += 1;
                if (ws_end < word.len) {
                    break :blk BaseText{
                        .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word[ws_end..] }),
                        .needs_free = true,
                    };
                }
            }
            break :blk BaseText{
                .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word }),
                .needs_free = true,
            };
        } else if (config.escape_whitespaces) {
            if (word.len > 0 and isWhitespace(word[0])) {
                var ws_end: usize = 1;
                while (ws_end < word.len and isWhitespace(word[ws_end])) ws_end += 1;
                if (ws_end < word.len) {
                    break :blk BaseText{
                        .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word[ws_end..] }),
                        .needs_free = true,
                    };
                }
                // Word is all whitespace — keep as-is for token lookup
            }
            break :blk BaseText{ .text = word, .needs_free = false };
        } else {
            // If word already starts with whitespace (captured by ?\p{L}+ etc.),
            // don't add another space — it's already there from pre-tokenization.
            if (word.len > 0 and isWhitespace(word[0])) {
                break :blk BaseText{ .text = word, .needs_free = false };
            }
            break :blk BaseText{
                .text = try std.fmt.allocPrint(config.allocator, " {s}", .{word}),
                .needs_free = true,
            };
        }
    } else if (config.escape_whitespaces and word.len > 0 and isWhitespace(word[0])) blk: {
        var ws_end: usize = 1;
        while (ws_end < word.len and isWhitespace(word[ws_end])) ws_end += 1;
        if (ws_end < word.len) {
            break :blk BaseText{
                .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word[ws_end..] }),
                .needs_free = true,
            };
        }
        // Word is all whitespace — keep as-is for token lookup
        break :blk BaseText{ .text = word, .needs_free = false };
    } else BaseText{ .text = word, .needs_free = false };

    const base_text = base.text;
    const base_needs_free = base.needs_free;
    errdefer {
        if (base_needs_free) config.allocator.free(base_text);
    }

    // 步骤 2：对基础文本进行 GPT-2 字节编码（如果需要）
    const use_gpt2_encoding = config.bytesToUnicodeFn != null and config.merges.count() > 0;

    const final_text: []const u8 = if (use_gpt2_encoding) blk: {
        const encoded = try toGpt2ByteEncoding(base_text, config.bytesToUnicodeFn.?, config.ctx, config.allocator);
        if (base_needs_free) config.allocator.free(base_text);
        break :blk encoded;
    } else base_text;

    const final_needs_free = if (use_gpt2_encoding) true else base_needs_free;

    defer {
        if (final_needs_free) config.allocator.free(@constCast(final_text));
    }

    // 阶段 1：如果 ignore_merges=true 且整个词在词表中，直接使用
    // 与 llama.cpp 的 llm_tokenizer_bpe_session::tokenize 逻辑一致
    if (ignore_merges) {
        if (config.textToTokenFn(final_text, config.ctx)) |token_id| {
            try tokens.append(config.allocator, token_id);
            return tokens;
        }
    }

    // 阶段 2：Tokenization
    // BPE 模型：先拆分为最小单元，再 BPE 合并。
    // - GPT-2 byte-encoded BPE (llama-bpe, qwen2 等): 通过 unicodeToByte
    //   映射回原始 byte，用 byteToTokenIdFn 查找字节 token。
    // - Non-byte-encoded BPE (gemma-4, bert): 直接按 UTF-8 字符查找 token。
    if (config.merges.count() > 0) {
        var pos: usize = 0;
        while (pos < final_text.len) {
            if (config.unicodeToByte) |utb| {
                // GPT-2 byte-encoded BPE: decode each UTF-8 code point back to raw byte
                const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                const ch = final_text[pos .. pos + @as(usize, ch_len)];
                if (utb.get(ch)) |byte| {
                    const tid = config.byteToTokenIdFn(byte, config.ctx);
                    try tokens.append(config.allocator, tid);
                    pos += ch.len;
                } else {
                    const tid = config.byteToTokenIdFn(final_text[pos], config.ctx);
                    try tokens.append(config.allocator, tid);
                    pos += 1;
                }
            } else {
                // Non-byte-encoded BPE: character-level lookup
                const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                const ch = final_text[pos .. pos + @as(usize, ch_len)];
                if (config.textToTokenFn(ch, config.ctx)) |tid| {
                    try tokens.append(config.allocator, tid);
                } else if (config.escape_whitespaces and ch.len == 1 and isWhitespace(ch[0])) {
                    if (config.textToTokenFn(SPM_SPACE, config.ctx)) |tid| {
                        try tokens.append(config.allocator, tid);
                    } else {
                        const tid = config.byteToTokenIdFn(ch[0], config.ctx);
                        try tokens.append(config.allocator, tid);
                    }
                } else {
                    for (ch) |byte| {
                        const tid = config.byteToTokenIdFn(byte, config.ctx);
                        try tokens.append(config.allocator, tid);
                    }
                }
                pos += ch.len;
            }
        }
    } else {
        // 非 BPE 模型（SPM 等）：Trie 贪婪最长匹配
        var pos: usize = 0;
        while (pos < final_text.len) {
            const match = trie.longestMatch(config.trie_root, final_text, pos);
            if (match) |m| {
                try tokens.append(config.allocator, m.token_id);
                pos += m.len;
            } else {
                if (is_spm_model) {
                    try tokens.append(config.allocator, config.special.unk);
                    const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                    pos += ch_len;
                } else if (config.unicodeToByte) |utb| {
                    const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                    const ch = final_text[pos .. pos + @as(usize, ch_len)];
                    if (utb.get(ch)) |byte| {
                        try tokens.append(config.allocator, config.byteToTokenIdFn(byte, config.ctx));
                        pos += ch.len;
                    } else {
                        try tokens.append(config.allocator, config.byteToTokenIdFn(final_text[pos], config.ctx));
                        pos += 1;
                    }
                } else {
                    try tokens.append(config.allocator, config.byteToTokenIdFn(final_text[pos], config.ctx));
                    pos += 1;
                }
            }
        }
    }

    // 阶段 3：BPE 合并（如果有合并规则）
    // 注意：即使 ignore_merges=true，如果整个词不在词表中，仍然需要 BPE 合并
    // 这与 llama.cpp 的行为一致
    if (config.merges.count() > 0) {
        try bpe.applyBpeMerges(&tokens, config.merges, config.tokenToStringFn, config.textToTokenFn, config.ctx, config.allocator);
    }

    return tokens;
}

// ============================================================================
// 编码配置
// ============================================================================

/// 编码所需的配置和回调函数
pub const EncodeConfig = struct {
    allocator: std.mem.Allocator,
    special: types.SpecialTokens,
    pre_type: types.PreTokenizerType,
    model: types.TokenizerModel,
    vocab: std.ArrayListUnmanaged(types.VocabEntry),
    merges: std.StringHashMap(u32),
    trie_root: *const trie.TrieNode,
    tokenToStringFn: *const fn (token_id: u32, ctx: ?*anyopaque) ?[]const u8,
    textToTokenFn: *const fn (text: []const u8, ctx: ?*anyopaque) ?u32,
    byteToTokenIdFn: *const fn (byte: u8, ctx: ?*anyopaque) u32,
    bytesToUnicodeFn: ?*const fn (byte: u8, ctx: ?*anyopaque) []const u8 = null,
    unicodeToByte: ?*const std.StringHashMap(u8) = null,
    tokenScoreFn: *const fn (token_id: u32, ctx: ?*anyopaque) f32 = undefined,
    escape_whitespaces: bool = false,
    ctx: ?*anyopaque,
    /// 缓存的特殊 token 列表（按 text 长度降序排列），用于 parse_special 模式
    cache_special_tokens: ?[]const mod.CacheSpecialToken = null,
};
