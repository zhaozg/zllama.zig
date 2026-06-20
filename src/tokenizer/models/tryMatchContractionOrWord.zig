// Auto-generated from encode.zig - tryMatchContractionOrWord pre-tokenizer
// Original lines 703-840

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn tryMatchContractionOrWord(text: []const u8, i: *usize, result: *PreTokenized) !bool {
    // Contractions: 's, 't, 're, 've, 'm, 'll, 'd
    // Note: only match the contraction suffix itself, not the preceding word.
    // The preceding word is matched by the ?\p{L}+ pattern.
    if (i.* + 1 < text.len and text[i.*] == '\'') {
        const suffix = text[i.* + 1 ..];
        if (suffix.len >= 1 and (suffix[0] == 's' or suffix[0] == 'S' or
            suffix[0] == 't' or suffix[0] == 'T' or
            suffix[0] == 'm' or suffix[0] == 'M' or
            suffix[0] == 'd' or suffix[0] == 'D'))
        {
            const word = try result.allocator.dupe(u8, text[i.* .. i.* + 2]);
            try result.words.append(result.allocator, word);
            i.* = i.* + 2;
            return true;
        }
        if (suffix.len >= 2 and ((suffix[0] == 'r' and suffix[1] == 'e') or
            (suffix[0] == 'R' and suffix[1] == 'E') or
            (suffix[0] == 'v' and suffix[1] == 'e') or
            (suffix[0] == 'V' and suffix[1] == 'E') or
            (suffix[0] == 'l' and suffix[1] == 'l') or
            (suffix[0] == 'L' and suffix[1] == 'L')))
        {
            const word = try result.allocator.dupe(u8, text[i.* .. i.* + 3]);
            try result.words.append(result.allocator, word);
            i.* = i.* + 3;
            return true;
        }
    }

    // Optional space + letters:  ?\p{L}+
    if (text[i.*] == ' ' and i.* + 1 < text.len and unicode.isLetterAt(text, i.* + 1)) {
        const start = i.*;
        i.* += 1;
        i.* += unicode.charLen(text, i.*);
        while (i.* < text.len and unicode.isLetterAt(text, i.*)) {
            i.* += unicode.charLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }
    if (unicode.isLetterAt(text, i.*)) {
        const start = i.*;
        i.* += unicode.charLen(text, i.*);
        while (i.* < text.len and unicode.isLetterAt(text, i.*)) {
            i.* += unicode.charLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }

    // Optional space + digits:  ?\p{N}+
    if (text[i.*] == ' ' and i.* + 1 < text.len and unicode.isAsciiDigit(text[i.* + 1])) {
        const start = i.*;
        i.* += 1;
        i.* += 1;
        while (i.* < text.len and unicode.isAsciiDigit(text[i.*])) {
            i.* += 1;
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }
    if (unicode.isAsciiDigit(text[i.*])) {
        const start = i.*;
        i.* += 1;
        while (i.* < text.len and unicode.isAsciiDigit(text[i.*])) {
            i.* += 1;
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }

    // Optional space + symbols:  ?[^\s\p{L}\p{N}]+
    if (text[i.*] == ' ' and i.* + 1 < text.len and
        !unicode.isAsciiWhitespace(text[i.* + 1]) and
        !unicode.isLetterAt(text, i.* + 1) and
        !unicode.isAsciiDigit(text[i.* + 1]) and
        !unicode.isDigitAt(text, i.* + 1))
    {
        const start = i.*;
        i.* += 1;
        i.* += unicode.charLen(text, i.*);
        while (i.* < text.len and
            !unicode.isAsciiWhitespace(text[i.*]) and
            !unicode.isLetterAt(text, i.*) and
            !unicode.isAsciiDigit(text[i.*]) and
            !unicode.isDigitAt(text, i.*))
        {
            i.* += unicode.charLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }
    if (!unicode.isAsciiWhitespace(text[i.*]) and !unicode.isLetterAt(text, i.*) and !unicode.isAsciiDigit(text[i.*]) and !unicode.isDigitAt(text, i.*)) {
        const start = i.*;
        i.* += unicode.charLen(text, i.*);
        while (i.* < text.len and
            !unicode.isAsciiWhitespace(text[i.*]) and
            !unicode.isLetterAt(text, i.*) and
            !unicode.isAsciiDigit(text[i.*]) and
            !unicode.isDigitAt(text, i.*))
        {
            i.* += unicode.charLen(text, i.*);
        }
        const word = try result.allocator.dupe(u8, text[start..i.*]);
        try result.words.append(result.allocator, word);
        return true;
    }

    // Whitespace: \s+(?!\S) or \s+
    if (unicode.isAsciiWhitespace(text[i.*])) {
        var ws_count: usize = 0;
        while (i.* + ws_count < text.len and unicode.isAsciiWhitespace(text[i.* + ws_count])) {
            ws_count += 1;
        }

        // \s+(?!\S): if whitespace is followed by non-whitespace and count > 1, take n-1
        if (ws_count > 1 and i.* + ws_count < text.len) {
            const word = try result.allocator.dupe(u8, text[i.* .. i.* + ws_count - 1]);
            try result.words.append(result.allocator, word);
            i.* += ws_count - 1;
            return true;
        }

        // \s+: regular whitespace
        const word = try result.allocator.dupe(u8, text[i.* .. i.* + ws_count]);
        try result.words.append(result.allocator, word);
        i.* += ws_count;
        return true;
    }

    return false;
}
