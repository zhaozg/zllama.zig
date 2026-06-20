// Auto-generated from encode.zig - deepseek_llm pre-tokenizer
// Original lines 416-516

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeDeepseekLlm(text: []const u8, result: *PreTokenized) !void {
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

        // 2. Optional space + Latin letters: \s?[A-Za-zµÀ-ÖØ-öø-ƺ...]+
        // Must decode codepoint to properly check Latin letter range
        const has_space = (text[i] == ' ');
        const check_pos = if (has_space) i + 1 else i;
        if (check_pos < text.len) {
            const decoded = unicode.decodeCodepoint(text, check_pos);
            if (decoded.len > 0 and unicode.isLatinLetter(decoded.cp)) {
                const start = i;
                if (has_space) i += 1;
                i += unicode.charLen(text, i);
                while (i < text.len) {
                    const d = unicode.decodeCodepoint(text, i);
                    if (d.len == 0 or !unicode.isLatinLetter(d.cp)) break;
                    i += d.len;
                }
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
                continue;
            }
        }

        // 3. Optional space + Punctuation: \s?[!-\/:~！-／：-～‘-‟　-。]+
        if (check_pos < text.len and unicode.isAsciiPunctuationOrSymbol(text[check_pos])) {
            const start = i;
            if (has_space) i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isAsciiPunctuationOrSymbol(text[i])) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 4. CJK characters: [一-龥...]+
        if (unicode.isCJKAt(text, i)) {
            const start = i;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isCJKAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. Numbers (no leading space): \p{N}+
        if (unicode.isDigitAt(text, i)) {
            const start = i;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isDigitAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 6. Whitespace: capture standalone whitespace
        // 放在所有"可选空格"模式之后，确保前面的模式有机会匹配
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // \s+(?!\S)：如果空白后面有非空白字符，且空白数 > 1，只取前 n-1 个
            // 这样剩下的一个空格可以被前面的 ?\s+ 模式匹配
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

        // Fallback: single UTF-8 character
        i += unicode.charLen(text, i);
    }
}
