// Auto-generated from encode.zig - llama3 pre-tokenizer
// Original lines 1086-1214

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;

pub fn preTokenizeLlama3(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. 收缩形式（仅匹配收缩后缀本身）
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

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+ : 可选前导非换行/非字母/非数字 + 字母序列
        // 注意：空格不是\r\n，所以 [^\r\n\p{L}\p{N}]? 可以匹配空格
        // 先检查是否有前导字符（非换行、非字母、非数字）后跟字母
        // 注意：空格不是\r\n，所以 [^\r\n\p{L}\p{N}]? 可以匹配空格
        // 先检查是否有前导字符（非换行、非字母、非数字）后跟字母
        if (i + 1 < text.len and
            text[i] != '\r' and text[i] != '\n' and
            !unicode.isLetterAt(text, i) and !unicode.isDigitAt(text, i) and
            unicode.isLetterAt(text, i + 1))
        {
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

        // 3. 字母序列: \p{L}+
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

        // 4. 数字序列（1-3位，不包含前导空格）: \p{N}{1,3}
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

        // 5. 可选空格 + 符号序列:  ?[^\s\p{L}\p{N}]+[\r\n]*
        const has_space = (text[i] == ' ');
        const check_pos = if (has_space) i + 1 else i;
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
            // 可选换行符: [\r\n]*
            while (i < text.len and (text[i] == '\r' or text[i] == '\n')) {
                i += 1;
            }
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }

        // 6. 空白序列：\s*[\r\n]+ | \s+(?!\S) | \s+
        if (unicode.isAsciiWhitespace(text[i])) {
            var ws_count: usize = 0;
            while (i + ws_count < text.len and unicode.isAsciiWhitespace(text[i + ws_count])) {
                ws_count += 1;
            }

            // \s*[\r\n]+：优先匹配包含换行符的空白序列
            // 找到最后一个 \r 或 \n 的位置，匹配到该位置+1
            var last_newline: ?usize = null;
            {
                var j: usize = 0;
                while (j < ws_count) {
                    if (text[i + j] == '\r' or text[i + j] == '\n') {
                        last_newline = j + 1;
                    }
                    j += 1;
                }
            }
            if (last_newline) |nl_end| {
                // 匹配 \s*[\r\n]+：从当前位置到最后一个换行符（含）
                const word = try result.allocator.dupe(u8, text[i .. i + nl_end]);
                try result.words.append(result.allocator, word);
                i += nl_end;
                continue;
            }

            // \s+(?!\S)：如果空白后面有非空白字符，且空白数 > 1，只取前 n-1 个
            if (ws_count > 1 and i + ws_count < text.len) {
                const word = try result.allocator.dupe(u8, text[i .. i + ws_count - 1]);
                try result.words.append(result.allocator, word);
                i += ws_count - 1;
                continue;
            }

            // \s+：普通空白序列（纯空格/制表符，不含换行符）
            const word = try result.allocator.dupe(u8, text[i .. i + ws_count]);
            try result.words.append(result.allocator, word);
            i += ws_count;
            continue;
        }

        // Skip any remaining unrecognized character
        i += 1;
    }
}
