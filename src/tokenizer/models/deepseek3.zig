// Auto-generated from encode.zig - deepseek3 pre-tokenizer
// Original lines 631-671

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;
const tryMatchContractionOrWord = @import("tryMatchContractionOrWord.zig").tryMatchContractionOrWord;

pub fn preTokenizeDeepseek3Style(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Numbers in groups of 1-3: \p{N}{1,3}
        if (unicode.isDigitAt(text, i)) {
            const start = i;
            var count: usize = 0;
            var pos = i;
            while (pos < text.len and unicode.isDigitAt(text, pos) and count < 3) {
                const ch_len = unicode.charLen(text, pos);
                pos += ch_len;
                count += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..pos]);
            try result.words.append(result.allocator, word);
            i = pos;
            continue;
        }

        // 2. CJK and Asian scripts
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

        // 3. Main BPE pattern: punctuation+letters, letters, punctuation, whitespace
        if (try tryMatchContractionOrWord(text, &i, result)) continue;

        // Fallback
        const ch_len = unicode.charLen(text, i);
        const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}
