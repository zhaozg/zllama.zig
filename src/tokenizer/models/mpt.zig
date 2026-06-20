// Auto-generated from encode.zig - mpt pre-tokenizer
// Original lines 283-385

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeMpt(text: []const u8, result: *PreTokenized) !void {
    // MPT uses the same regex as GPT-2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
    // Reference: llama.cpp LLAMA_VOCAB_PRE_TYPE_MPT (same as GPT2)
    var i: usize = 0;
    while (i < text.len) {
        // 1. Check contractions (match only the suffix itself)
        if (i + 1 < text.len and text[i] == '\'') {
            const suffix = text[i + 1 ..];
            if (suffix.len >= 1 and (suffix[0] == 's' or suffix[0] == 'S' or
                suffix[0] == 't' or suffix[0] == 'T' or
                suffix[0] == 'm' or suffix[0] == 'M' or
                suffix[0] == 'd' or suffix[0] == 'D'))
            {
                const word = try result.allocator.dupe(u8, text[i .. i + 2]);
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
                const word = try result.allocator.dupe(u8, text[i .. i + 3]);
                try result.words.append(result.allocator, word);
                i = i + 3;
                continue;
            }
        }

        // 2. Optional space + letters:  ?\p{L}+
        const has_space = (text[i] == ' ');
        const check_pos = if (has_space) i + 1 else i;
        if (check_pos < text.len and unicode.isLetterAt(text, check_pos)) {
            const start = i;
            if (has_space) i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isLetterAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3. Optional space + digits:  ?\p{N}+
        if (check_pos < text.len and unicode.isDigitAt(text, check_pos)) {
            const start = i;
            if (has_space) i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isDigitAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 4. Optional space + symbols:  ?[^\s\p{L}\p{N}]+
        if (check_pos < text.len and !unicode.isAsciiWhitespace(text[check_pos]) and
            !unicode.isLetterAt(text, check_pos) and !unicode.isDigitAt(text, check_pos))
        {
            const start = i;
            if (has_space) i += 1;
            i += unicode.charLen(text, i);
            while (i < text.len and !unicode.isAsciiWhitespace(text[i]) and
                !unicode.isLetterAt(text, i) and !unicode.isDigitAt(text, i))
            {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 5. Whitespace: \s+(?!\S) or \s+
        // \s+(?!\S): whitespace not followed by non-whitespace (e.g. trailing spaces)
        // \s+: all other whitespace (e.g. '  ' before 'Hello')
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // \s+(?!\S): if whitespace is followed by non-whitespace and count > 1,
            // take all whitespace (no backtracking for MPT)
            // For '  Hello' -> '  ' + 'Hello' (not ' ' + ' Hello')
            if (ws_count > 1 and i + ws_count < text.len) {
                const word = try result.allocator.dupe(u8, text[i .. i + ws_count]);
                try result.words.append(result.allocator, word);
                i += ws_count;
                continue;
            }

            // \s+(?!\S): trailing whitespace (no non-whitespace after)
            const word = try result.allocator.dupe(u8, text[i .. i + ws_count]);
            try result.words.append(result.allocator, word);
            i += ws_count;
            continue;
        }

        // Skip any remaining unrecognized character
        i += 1;
    }
}
