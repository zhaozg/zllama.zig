// Auto-generated from encode.zig - bloom pre-tokenizer
// Original lines 675-699

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeBloomStyle(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // Skip whitespace and punctuation separators
        if (unicode.isAsciiWhitespace(text[i]) or unicode.isBloomSeparator(text[i])) {
            const start = i;
            i += 1;
            while (i < text.len and (unicode.isAsciiWhitespace(text[i]) or unicode.isBloomSeparator(text[i]))) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // Collect non-separator characters
        const start = i;
        i += unicode.charLen(text, i);
        while (i < text.len and !unicode.isAsciiWhitespace(text[i]) and !unicode.isBloomSeparator(text[i])) {
            i += unicode.charLen(text, i);
        }
        const word = try result.allocator.dupe(u8, text[start..i]);
        try result.words.append(result.allocator, word);
    }
}
