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
        // 1. Punctuation and symbols: [\p{P}\$+<=\>^~\|`]+
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

        // 3. Optional space + ASCII letters:  ?[A-Za-z]+
        // Note: only matches ASCII letters, not CJK or other Unicode letters.
        // CJK characters are handled separately (step 8).
        if (text[i] == ' ' and i + 1 < text.len and unicode.isAsciiLetter(text[i + 1])) {
            const start = i;
            i += 1;
            i += 1;
            while (i < text.len and unicode.isAsciiLetter(text[i])) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        if (unicode.isAsciiLetter(text[i])) {
            const start = i;
            i += 1;
            while (i < text.len and unicode.isAsciiLetter(text[i])) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 4. Optional space + non-ASCII letters (CJK, Greek, Cyrillic, etc.):  ?\p{L}+
        // Must come after ASCII letters to avoid merging CJK with ASCII.
        if (text[i] == ' ' and i + 1 < text.len and
            !unicode.isAsciiLetter(text[i + 1]) and
            unicode.isLetterAt(text, i + 1))
        {
            const start = i;
            i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and !unicode.isAsciiLetter(text[i]) and unicode.isLetterAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        if (!unicode.isAsciiLetter(text[i]) and unicode.isLetterAt(text, i)) {
            const start = i;
            i += unicode.charLen(text, i);
            while (i < text.len and !unicode.isAsciiLetter(text[i]) and unicode.isLetterAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. Optional space + non-ASCII symbols:  ?[^\s\p{L}\p{N}]+
        // Matches Unicode symbols (emoji, etc.) but NOT ASCII punctuation/symbols
        // (ASCII punctuation is handled by step 1)
        if (text[i] == ' ' and i + 1 < text.len and
            !unicode.isAsciiWhitespace(text[i + 1]) and
            !unicode.isLetterAt(text, i + 1) and
            !unicode.isDigitAt(text, i + 1) and
            !unicode.isAsciiPunctuationOrSymbol(text[i + 1]))
        {
            const start = i;
            i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and
                !unicode.isAsciiWhitespace(text[i]) and
                !unicode.isLetterAt(text, i) and
                !unicode.isDigitAt(text, i) and
                !unicode.isAsciiPunctuationOrSymbol(text[i]))
            {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 6. Whitespace: \s+(?!\S) or \s+ (with backtracking for cases like '  Hello')
        // Backtrack only when followed by a letter or digit (not punctuation/symbols).
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // Apply \s+(?!\S) backtracking only when the next char is a letter or digit
            if (ws_count > 1 and i + ws_count < text.len) {
                const can_match_optional = unicode.isLetterAt(text, i + ws_count) or
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

        // 7. Contractions and word patterns (ASCII only): 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
        if (try tryMatchContractionOrWord(text, &i, result)) continue;

        // 8. Three-digit numbers: [0-9][0-9][0-9]
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
