// Auto-generated from encode.zig - gpt2_style_nospace pre-tokenizer
// Original lines 958-1077

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeGpt2StyleNoSpace(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. 检查收缩形式（仅匹配收缩后缀本身，如 's, 't, 're, 've, 'm, 'll, 'd）
        // 对应 regex: 's|'t|'re|'ve|'m|'ll|'d
        // 注意：不包含前面的单词，前面的单词由 ?\p{L}+ 模式匹配
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

        // 2. 可选前导符号 + 字母序列: [^\s\p{L}\p{N}]?\p{L}+
        // 匹配如 'all, .Hello 等（符号后紧跟字母）
        if (!unicode.isAsciiWhitespace(text[i]) and !unicode.isLetterAt(text, i) and !unicode.isDigitAt(text, i)) {
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

        // 3. 可选空格 + 字母序列:  ?\p{L}+
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

        // 4. 数字序列（不包含前导空格）: \p{N}+
        // 与 preTokenizeGpt2Style 不同，这里不匹配前导空格。
        // 空格会被单独作为空白序列处理。
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

        // 5. 可选空格 + 符号序列:  ?[^\s\p{L}\p{N}]+
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

        // 6. 空白序列：先尝试 \s+(?!\S)（后面有非空白时只取前 n-1 个），再尝试 \s+
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // \s+(?!\S)：如果空白后面有非空白字符，且空白数 > 1，只取前 n-1 个
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

        // Skip any remaining unrecognized character
        i += 1;
    }
}
