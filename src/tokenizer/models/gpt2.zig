// Auto-generated from encode.zig - gpt2 pre-tokenizer
// Original lines 1349-1508

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeGPT2(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. 检查收缩形式：'s, 't, 're, 've, 'm, 'll, 'd
        if (i + 1 < text.len and text[i] == '\'') {
            const suffix = text[i + 1 ..];
            if (suffix.len >= 1 and (suffix[0] == 's' or suffix[0] == 'S' or
                suffix[0] == 't' or suffix[0] == 'T' or
                suffix[0] == 'm' or suffix[0] == 'M' or
                suffix[0] == 'd' or suffix[0] == 'D'))
            {
                var word_start = i;
                while (word_start > 0 and !unicode.isAsciiWhitespace(text[word_start - 1])) {
                    word_start -= 1;
                }
                const word = try result.allocator.dupe(u8, text[word_start .. i + 2]);
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
                var word_start = i;
                while (word_start > 0 and !unicode.isAsciiWhitespace(text[word_start - 1])) {
                    word_start -= 1;
                }
                const word = try result.allocator.dupe(u8, text[word_start .. i + 3]);
                try result.words.append(result.allocator, word);
                i = i + 3;
                continue;
            }
        }

        // 2. [^\s\p{L}\p{N}]?\p{L}+ : 可选前导符号 + 字母序列
        // Check for optional leading punctuation/symbol before letters
        if (!unicode.isAsciiWhitespace(text[i]) and !unicode.isLetterAt(text, i) and !unicode.isDigitAt(text, i)) {
            // Possible leading symbol
            if (i + 1 < text.len and unicode.isLetterAt(text, i + 1)) {
                const start = i;
                i += unicode.charLen(text, i);
                i += unicode.charLen(text, i);
                while (i < text.len and unicode.isLetterAt(text, i)) {
                    i += unicode.charLen(text, i);
                }
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
                continue;
            }
        }
        if (unicode.isLetterAt(text, i)) {
            const start = i;
            i += unicode.charLen(text, i);
            while (i < text.len and unicode.isLetterAt(text, i)) {
                i += unicode.charLen(text, i);
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 3a. 可选空格 + 数字序列:  ?\p{N}{1,3}
        if (text[i] == ' ' and i + 1 < text.len and unicode.isDigitAt(text, i + 1)) {
            const start = i;
            i += 1;
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
        // 3b. \p{N}{1,3} : 数字序列（1-3位）
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

        // 4.  ?[^\s\p{L}\p{N}]+ : 空格 + 符号序列
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
        if (!unicode.isAsciiWhitespace(text[i]) and !unicode.isLetterAt(text, i) and !unicode.isDigitAt(text, i)) {
            const start = i;
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

        // 5. 空白序列：放在所有"可选空格"模式之后
        //    先尝试 \s+(?!\S)（尾随空白，后面有非空白时只取前 n-1 个），再尝试 \s+
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // \s+(?!\S)：如果空白后面有非空白字符，且空白数 > 1，只取前 n-1 个
            // 这样剩下的一个空格可以被后面的 ?\p{L}+ 等模式匹配
            if (ws_count > 1 and i + ws_count < text.len) {
                const word = try result.allocator.dupe(u8, text[i .. i + ws_count - 1]);
                try result.words.append(result.allocator, word);
                i += ws_count - 1;
                continue;
            }

            // \s+：普通空白序列
            const word = try result.allocator.dupe(u8, text[i .. i + ws_count]);
            try result.words.append(result.allocator, word);
            i += ws_count;
            continue;
        }

        // Fallback: skip any unrecognized character
        i += 1;
    }
}
