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

        // 3. Optional space + letters: \s?\p{L}+
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

        // 4. Optional space + punctuation: \s?\p{P}+
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

        // 5. Whitespace with backtracking: \s+(?!\S) or \s+
        // For '  Hello' -> ' ' + ' Hello' (backtrack 1 space for ?\p{L}+)
        // For '   Hello' -> '  ' + ' Hello' (backtrack 1 space for ?\p{L}+)
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // Backtrack: if there are >1 whitespace chars and the next non-whitespace
            // can be matched by ?\p{L}+ or ?\p{P}+, leave 1 space for the optional-space pattern
            if (ws_count > 1 and i + ws_count < text.len) {
                const can_match_optional = unicode.isLetterAt(text, i + ws_count) or
                    unicode.isPunctuationAt(text, i + ws_count) or
                    unicode.isDigitAt(text, i + ws_count);
                if (can_match_optional) {
                    const word = try result.allocator.dupe(u8, text[i .. i + ws_count - 1]);
                    try result.words.append(result.allocator, word);
                    i += ws_count - 1;
                    continue;
                }
            }

            const word = try result.allocator.dupe(u8, text[i .. i + ws_count]);
            try result.words.append(result.allocator, word);
            i += ws_count;
            continue;
        }

        // 6. Letters: \p{L}+
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

        // 7. Punctuation: \p{P}+
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

        // 8. CJK: [一-龥ࠀ-一가-퟿]+
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

        // 9. Single digit: \p{N}
        if (unicode.isDigitAt(text, i)) {
            const ch_len = unicode.charLen(text, i);
            const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
            try result.words.append(result.allocator, word);
            i += ch_len;
            continue;
        }

        // Fallback: single UTF-8 character as its own word
        const ch_len = unicode.charLen(text, i);
        const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}
