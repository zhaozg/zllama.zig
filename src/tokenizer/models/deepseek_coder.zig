// Auto-generated from encode.zig - deepseek_coder pre-tokenizer
// Original lines 518-627

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeDeepseekCoder(text: []const u8, result: *PreTokenized) !void {
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

        // 2. Emoji with optional space: \s?\p{Emoji}+
        // Emoji must be checked before letters to prevent emoji from being
        // treated as individual characters or merged with adjacent punctuation.
        if (text[i] == ' ' and i + 1 < text.len) {
            const next_cp = unicode.decodeCodepoint(text, i + 1);
            if (next_cp.len > 0 and unicode.isEmoji(next_cp.cp)) {
                const start = i;
                i += 1;
                i += unicode.charLen(text, i);
                while (i < text.len) {
                    const d = unicode.decodeCodepoint(text, i);
                    if (d.len == 0 or !unicode.isEmoji(d.cp)) break;
                    i += d.len;
                }
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
                continue;
            }
        }
        if (i < text.len) {
            const cp = unicode.decodeCodepoint(text, i);
            if (cp.len > 0 and unicode.isEmoji(cp.cp)) {
                const start = i;
                i += cp.len;
                while (i < text.len) {
                    const d = unicode.decodeCodepoint(text, i);
                    if (d.len == 0 or !unicode.isEmoji(d.cp)) break;
                    i += d.len;
                }
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
                continue;
            }
        }

        // 3. Letters with optional space: \s?\p{L}+
        // Note: use isLetterStrict (L* category only) to avoid matching
        // mark characters (Mc/Mn) like Khmer vowel signs.
        if (text[i] == ' ' and i + 1 < text.len) {
            const next_cp = unicode.decodeCodepoint(text, i + 1);
            if (next_cp.len > 0 and unicode.isLetterStrict(next_cp.cp) and !unicode.isEmoji(next_cp.cp)) {
                const start = i;
                i += 1;
                i += unicode.charLen(text, i);
                while (i < text.len) {
                    const d = unicode.decodeCodepoint(text, i);
                    if (d.len == 0 or !unicode.isLetterStrict(d.cp) or unicode.isEmoji(d.cp)) break;
                    i += d.len;
                }
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
                continue;
            }
        }
        if (i < text.len) {
            const cp = unicode.decodeCodepoint(text, i);
            if (cp.len > 0 and unicode.isLetterStrict(cp.cp) and !unicode.isEmoji(cp.cp)) {
                const start = i;
                i += cp.len;
                while (i < text.len) {
                    const d = unicode.decodeCodepoint(text, i);
                    if (d.len == 0 or !unicode.isLetterStrict(d.cp) or unicode.isEmoji(d.cp)) break;
                    i += d.len;
                }
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
                continue;
            }
        }

        // 4. Punctuation with optional space: \s?\p{P}+
        // Note: use isPunctuation (not isAsciiPunctuationOrSymbol) to avoid
        // merging emoji (symbols) with punctuation like '.'
        if (text[i] == ' ' and i + 1 < text.len and unicode.isPunctuationAt(text, i + 1)) {
            const start = i;
            i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isPunctuationAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        if (unicode.isPunctuationAt(text, i)) {
            const start = i;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isPunctuationAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. CJK: [一-龥ࠀ-一가-퟿]+
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

        // 6. Single digit: \p{N}
        if (unicode.isDigitAt(text, i)) {
            const ch_len = unicode.charLen(text, i);
            const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
            try result.words.append(result.allocator, word);
            i += ch_len;
            continue;
        }

        // 7. Whitespace: capture standalone whitespace (after letters/punct/digits)
        if (unicode.isAsciiWhitespace(text[i])) {
            const start = i;
            while (i < text.len and unicode.isAsciiWhitespace(text[i])) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // Fallback: single UTF-8 character as its own word
        const ch_len = unicode.charLen(text, i);
        const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}
