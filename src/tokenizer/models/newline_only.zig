// Auto-generated from encode.zig - newline_only pre-tokenizer
// Original lines 1518-1542

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeNewlineOnly(text: []const u8, result: *PreTokenized) !void {
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
