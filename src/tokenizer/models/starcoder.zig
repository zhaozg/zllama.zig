// Auto-generated from encode.zig - starcoder pre-tokenizer
// Original lines 392-413

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;
const tryMatchContractionOrWord = @import("tryMatchContractionOrWord.zig").tryMatchContractionOrWord;

pub fn preTokenizeStarcoderStyle(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. Single digit: \p{N} (ASCII and Unicode digits like ¼ ½ ¾ ² ³ ¹)
        if (unicode.isAsciiDigit(text[i]) or unicode.isDigitAt(text, i)) {
            const ch_len = if (unicode.isAsciiDigit(text[i])) @as(usize, 1) else unicode.charLen(text, i);
            const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
            try result.words.append(result.allocator, word);
            i += ch_len;
            continue;
        }

        // 2. Contractions and word patterns
        if (try tryMatchContractionOrWord(text, &i, result)) continue;

        // Fallback
        const ch_len = unicode.charLen(text, i);
        const word = try result.allocator.dupe(u8, text[i .. i + ch_len]);
        try result.words.append(result.allocator, word);
        i += ch_len;
    }
}
