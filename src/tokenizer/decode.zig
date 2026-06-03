//! 解码逻辑
//!
//! 实现 token ID 列表到 UTF-8 文本的解码过程。
//! 参考 llama.cpp 的 token_to_piece 和 detokenize 函数。
//!
//! 解码流程：
//! 1. 遍历 token ID 列表
//! 2. 跳过特殊 token（BOS, EOS, UNK, PAD, CONTROL 等）
//! 3. 根据 token 类型选择解码路径：
//!    - NORMAL: 替换空格标记（▁/Ġ）为普通空格，或解码字节编码
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
                    // LLaMA/GPT2: 将空格标记替换为实际空格
                    try decodeNormalToken(ts, config.escape_whitespaces, &result, allocator);
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
fn decodeNormalToken(
    token_str: []const u8,
    escape_whitespaces: bool,
    result: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    if (!escape_whitespaces) {
        // GPT-2 风格：字节编码，需要解码
        try decodeGPT2Token(token_str, result, allocator);
        return;
    }

    // SPM 风格：替换 ▁ 为空格
    var i: usize = 0;
    while (i < token_str.len) {
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
fn decodeGPT2Token(
    token_str: []const u8,
    result: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    var i: usize = 0;
    while (i < token_str.len) {
        const byte = token_str[i];
        // 检查是否是 UTF-8 编码的字节（码点 0x100-0x1FF）
        // UTF-8 编码：0xC4 0x80-0xBF (0x100-0x13F)
        //            0xC5 0x80-0xBF (0x140-0x17F)
        //            0xC6 0x80-0xBF (0x180-0x1BF)
        //            0xC7 0x80-0xBF (0x1C0-0x1FF)
        if (byte >= 0xC4 and byte <= 0xC7 and i + 1 < token_str.len) {
            const cp: u32 = (@as(u32, byte - 0xC4) << 6) | (@as(u32, token_str[i + 1]) & 0x3F) | 0x100;
            if (cp <= 0x1FF) {
                try result.append(allocator, @as(u8, @intCast(cp & 0xFF)));
                i += 2;
                continue;
            }
        }
        // 检查是否是 Latin-1 范围的字符（0x80-0xFF），这些直接映射到字节值
        // 在 GPT-2 字节编码中，0xC2 0x80-0xBF 表示 0x80-0xBF
        // 0xC3 0x80-0xBF 表示 0xC0-0xFF
        if (byte == 0xC2 and i + 1 < token_str.len) {
            try result.append(allocator, token_str[i + 1]);
            i += 2;
            continue;
        }
        if (byte == 0xC3 and i + 1 < token_str.len) {
            try result.append(allocator, 0xC0 + (token_str[i + 1] & 0x3F));
            i += 2;
            continue;
        }
        // 对于 ASCII 字符（0x00-0x7F），直接输出
        if (byte < 0x80) {
            try result.append(allocator, byte);
            i += 1;
            continue;
        }
        // 其他情况，跳过
        i += 1;
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
