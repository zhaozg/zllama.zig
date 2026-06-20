// Auto-generated from encode.zig - falcon pre-tokenizer
// Original lines 157-280

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;
const tryMatchContractionOrWord = @import("tryMatchContractionOrWord.zig").tryMatchContractionOrWord;

pub fn preTokenizeFalcon(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Punctuation and symbols: [\p{P}\$\+<=\>\^~\|`]+
        if (unicode.isAsciiPunctuationOrSymbol(text[i])) {
            const start = i;
            i += 1;
            while (i < text.len and unicode.isAsciiPunctuationOrSymbol(text[i])) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 2. Optional space + Unicode digits:  ?\p{N}+ (for digits like ½ ² ³)
        // Must come before standalone whitespace to match ' ½' as one token
        if (text[i] == ' ' and i + 1 < text.len and unicode.isDigitAt(text, i + 1)) {
            const start = i;
            i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isDigitAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
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

        // 3. Optional space + letters:  ?\p{L}+ (before whitespace to match ' months' as one token)
        if (text[i] == ' ' and i + 1 < text.len and unicode.isLetterAt(text, i + 1)) {
            const start = i;
            i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isLetterAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 4. Optional space + symbols:  ?[^\s\p{L}\p{N}]+ (before whitespace to match ' 🦙' as one token)
        if (text[i] == ' ' and i + 1 < text.len and
            !unicode.isAsciiWhitespace(text[i + 1]) and
            !unicode.isLetterAt(text, i + 1) and
            !unicode.isDigitAt(text, i + 1))
        {
            const start = i;
            i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and
                !unicode.isAsciiWhitespace(text[i]) and
                !unicode.isLetterAt(text, i) and
                !unicode.isDigitAt(text, i))
            {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. Whitespace: \s+(?!\S) or \s+ (with backtracking for cases like '  Hello')
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // Only apply \s+(?!\S) backtracking when the next char can be matched
            // by ?\p{L}+, ?\p{N}+, or ?[^\s\p{L}\p{N}]+ patterns
            if (ws_count > 1 and i + ws_count < text.len) {
                const next = text[i + ws_count];
                const can_match_optional = unicode.isLetterAt(text, i + ws_count) or
                    unicode.isDigitAt(text, i + ws_count) or
                    (!unicode.isAsciiWhitespace(next) and
                     !unicode.isLetterAt(text, i + ws_count) and
                     !unicode.isDigitAt(text, i + ws_count));
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

        // 6. Contractions and word patterns (ASCII only): 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
        if (try tryMatchContractionOrWord(text, &i, result)) continue;

        // 5. Three-digit numbers: [0-9][0-9][0-9]
        if (i + 2 < text.len and unicode.isDigitAt(text, i) and unicode.isDigitAt(text, i + 1) and unicode.isDigitAt(text, i + 2)) {
            const start = i;
            i += 3;
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }



        // Fallback: single character
        const ch_len = unicode.charLen(text, i);
        const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}
