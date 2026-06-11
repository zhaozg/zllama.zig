//! 解码逻辑
//!
//! 实现 token ID 列表到 UTF-8 文本的解码过程。
//! 参考 llama.cpp 的 token_to_piece 和 detokenize 函数。
//!
//! 解码流程：
//! 1. 遍历 token ID 列表
//! 2. 跳过特殊 token（BOS, EOS, UNK, PAD, CONTROL 等）
//! 3. 根据 token 类型选择解码路径：
//!    - NORMAL: 替换空格标记（▁/Ġ）为普通空格，或解码字节编码（<0xXX>）
//!    - BYTE: 输出单个字节
//! 4. 清理空格（clean_spaces 后处理）

const std = @import("std");
const types = @import("types.zig");

const log = std.log.scoped(.tokenizer);

// ============================================================================
// 解码配置
// ============================================================================

pub const DecodeConfig = struct {
    model: types.TokenizerModel,
    special: types.SpecialTokens,
    token_types: std.ArrayListUnmanaged(types.TokenType),
    vocab: std.ArrayListUnmanaged(types.VocabEntry),
    clean_spaces: bool,
    escape_whitespaces: bool,
    /// GPT-2 字节解码映射：unicode codepoint (UTF-8) -> byte
    unicode_to_byte: ?*const std.StringHashMap(u8) = null,
};

// ============================================================================
// 解码主函数
// ============================================================================

/// 解码：将 token ID 列表转换为 UTF-8 文本
pub fn decode(
    token_ids: []const u32,
    config: *const DecodeConfig,
    allocator: std.mem.Allocator,
) ![]u8 {
    var result = std.ArrayList(u8).empty;

    for (token_ids) |token_id| {
        if (isSpecialToken(token_id, config)) continue;
        if (token_id >= config.vocab.items.len) continue;

        switch (config.vocab.items[token_id]) {
            .byte => |bv| try result.append(allocator, bv),
            .normal => |ts| {
                if (config.model == .tiktoken) {
                    try decodeTiktokenToken(ts, &result, allocator);
                } else {
                    // LLaMA/GPT2: 替换空格标记为实际空格，同时处理 <0xXX> 字节模式
                    try decodeNormalToken(ts, config, &result, allocator);
                }
            },
        }
    }

    // clean_spaces 后处理
    if (config.clean_spaces) {
        return cleanSpaces(result.items, allocator);
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// 特殊 token 检测
// ============================================================================

fn isSpecialToken(token_id: u32, config: *const DecodeConfig) bool {
    if (token_id == config.special.bos or
        token_id == config.special.eos or
        token_id == config.special.pad or
        token_id == config.special.unk or
        token_id == config.special.sep or
        token_id == config.special.cls or
        token_id == config.special.mask) return true;
    if (token_id < config.token_types.items.len and
        config.token_types.items[token_id] == .control) return true;
    return false;
}

// ============================================================================
// 普通 token 解码
// ============================================================================

/// 解码普通 token，将 LLaMA/GPT2 的空格标记替换为实际空格（U+0020）
/// LLaMA (SentencePiece): ▁ (U+2581, UTF-8: E2 96 81)
/// GPT-2: Ġ (U+0120, UTF-8: C4 A0)
///
/// 对于所有模型类型，同时处理 <0xXX> 字节模式：
/// 某些 SPM 模型的词表中，部分 token 使用 <0xXX> 格式表示原始字节。
/// 这些 token 虽然标记为 .normal 类型，但其内容需要解码为原始字节。
fn decodeNormalToken(
    token_str: []const u8,
    config: *const DecodeConfig,
    result: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    if (config.model == .gpt2 and config.unicode_to_byte != null) {
        // GPT-2 风格：字节编码，需要解码
        try decodeGPT2Token(token_str, config.unicode_to_byte.?, result, allocator);
        return;
    }

    // SPM / Llama 风格：替换 ▁ 为空格，同时处理 <0xXX> 字节模式
    var i: usize = 0;
    while (i < token_str.len) {
        // 检查是否是 <0xXX> 字节模式（tiktoken 格式的字节 token）
        if (token_str[i] == '<' and i + 3 < token_str.len and
            token_str[i + 1] == '0' and token_str[i + 2] == 'x')
        {
            const end = std.mem.indexOfScalar(u8, token_str[i + 1 ..], '>') orelse {
                try result.append(allocator, token_str[i]);
                i += 1;
                continue;
            };
            const hex_start = i + 3; // skip '<0x'
            const hex_end = i + 1 + end;
            const hex_str = token_str[hex_start..hex_end];
            if (hex_str.len == 2) {
                if (std.fmt.parseInt(u8, hex_str, 16)) |byte| {
                    try result.append(allocator, byte);
                    i = i + 1 + end + 1; // skip past '>'
                    continue;
                } else |_| {}
            }
            try result.append(allocator, token_str[i]);
            i += 1;
            continue;
        }

        // 检查是否是 U+2581 (▁) 的 UTF-8 序列 (LLaMA/SentencePiece)
        if (i + 2 < token_str.len and
            token_str[i] == 0xE2 and
            token_str[i + 1] == 0x96 and
            token_str[i + 2] == 0x81)
        {
            try result.append(allocator, ' '); // 替换为普通空格
            i += 3;
        }
        // 检查是否是 U+0120 (Ġ) 的 UTF-8 序列 (GPT-2)
        else if (i + 1 < token_str.len and
            token_str[i] == 0xC4 and
            token_str[i + 1] == 0xA0)
        {
            try result.append(allocator, ' '); // 替换为普通空格
            i += 2;
        } else {
            try result.append(allocator, token_str[i]);
            i += 1;
        }
    }
}

// ============================================================================
// GPT-2 字节解码
// ============================================================================

/// GPT-2 风格的字节解码
/// GPT-2 使用字节级编码，将字节映射到 Unicode 码点
/// 解码时，将 token 文本中的每个 Unicode 码点通过逆映射转回原始字节
fn decodeGPT2Token(
    token_str: []const u8,
    unicode_to_byte: *const std.StringHashMap(u8),
    result: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    var i: usize = 0;
    while (i < token_str.len) {
        // 解码一个 UTF-8 码点
        const byte = token_str[i];
        const cp_len = std.unicode.utf8ByteSequenceLength(byte) catch {
            // 无效的 UTF-8 起始字节，跳过
            i += 1;
            continue;
        };

        if (i + cp_len > token_str.len) {
            // 不完整的 UTF-8 序列，跳过
            i += 1;
            continue;
        }

        const cp_slice = token_str[i..i+cp_len];

        // 检查是否是 ASCII（直接输出）
        if (cp_len == 1 and byte < 0x80) {
            try result.append(allocator, byte);
            i += 1;
            continue;
        }

        // 通过 unicode_to_byte 映射查找
        if (unicode_to_byte.get(cp_slice)) |b| {
            try result.append(allocator, b);
            i += cp_len;
            continue;
        }

        // 如果映射中找不到，直接输出 UTF-8 字节
        // 这通常不会发生，但作为回退
        try result.appendSlice(allocator, cp_slice);
        i += cp_len;
    }
}

// ============================================================================
// tiktoken 解码
// ============================================================================

/// tiktoken 风格的 token 解码
/// tiktoken 使用 <0xXX> 格式表示字节 token
fn decodeTiktokenToken(
    token_str: []const u8,
    result: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    var rem = token_str;
    while (rem.len > 0) {
        // 查找 <0xXX> 格式的字节 token
        if (rem.len >= 4 and rem[0] == '<' and rem[1] == '0' and rem[2] == 'x') {
            const end = std.mem.indexOfScalar(u8, rem[1..], '>') orelse {
                try result.append(allocator, rem[0]);
                rem = rem[1..];
                continue;
            };
            const hex_str = rem[2 .. 2 + end - 1];
            if (hex_str.len == 2) {
                const byte = std.fmt.parseInt(u8, hex_str, 16) catch {
                    try result.appendSlice(allocator, rem[0 .. end + 1]);
                    rem = rem[end + 1 ..];
                    continue;
                };
                try result.append(allocator, byte);
                rem = rem[end + 1 ..];
                continue;
            }
        }
        try result.append(allocator, rem[0]);
        rem = rem[1..];
    }
}

// ============================================================================
// 空格清理（clean_spaces）
// ============================================================================

/// 清理解码后的文本中的多余空格
/// 参考 llama.cpp 的 detokenize 中的 clean_spaces 逻辑
fn cleanSpaces(text: []const u8, allocator: std.mem.Allocator) ![]u8 {
    if (text.len == 0) return allocator.dupe(u8, text);

    var result = try allocator.alloc(u8, text.len);
    var j: usize = 0;

    // 第一遍：移除标点符号前的空格
    // " ?", " !", " .", " ," -> "?", "!", ".", ","
    var i: usize = 0;
    while (i < text.len) {
        if (i + 1 < text.len and text[i] == ' ') {
            const next = text[i + 1];
            if (next == '?' or next == '!' or next == '.' or next == ',') {
                // 跳过空格
                i += 1;
                continue;
            }
        }
        result[j] = text[i];
        j += 1;
        i += 1;
    }

    // 第二遍：处理缩写（简化版）
    // 只处理常见的 's, 't, 'd, 'm, 're, 've, 'll
    // 这些在 GPT-2 预分词中已经处理，这里不做额外处理

    return allocator.realloc(result, j);
}
