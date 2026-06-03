//! 分词器工具函数
//!
//! 提供字节 token 解析、Unicode 映射等辅助功能。

const std = @import("std");
const types = @import("types.zig");

// ============================================================================
// 字节到 Unicode 映射（GPT-2 风格）
// ============================================================================

/// 生成 GPT-2 风格的 bytes_to_unicode 映射
/// 将 0-255 字节映射到 Unicode 码点，用于 GPT-2 的字节级编码
pub fn generateBytesToUnicode(allocator: std.mem.Allocator) ![]u32 {
    var mapping = try allocator.alloc(u32, 256);

    // 可打印 ASCII 和 Latin-1 范围直接映射
    var n: u32 = 0;
    var ch: u32 = 0;

    // 0x21-0x7E: ! 到 ~
    ch = 0x21;
    while (ch <= 0x7E) {
        mapping[ch] = ch;
        ch += 1;
    }

    // 0xA1-0xAC: ¡ 到 ¬
    ch = 0xA1;
    while (ch <= 0xAC) {
        mapping[ch] = ch;
        ch += 1;
    }

    // 0xAE-0xFF: ® 到 ÿ
    ch = 0xAE;
    while (ch <= 0xFF) {
        mapping[ch] = ch;
        ch += 1;
    }

    // 剩余字节映射到 0x100+ 范围
    n = 0;
    ch = 0;
    while (ch < 256) {
        if (ch < 0x21 or (ch > 0x7E and ch < 0xA1) or ch == 0xAD) {
            mapping[ch] = 256 + n;
            n += 1;
        }
        ch += 1;
    }

    return mapping;
}

// ============================================================================
// 字节 token 解析
// ============================================================================

/// 从 token 字符串中提取字节值
/// 根据不同的模型格式解析：
/// - llama/gpt2: 原始单字节
/// - tiktoken: <0xE4> 格式
/// - replit: b'<0xE4>' 格式
pub fn extractByteFromToken(token: []const u8, model: types.TokenizerModel) !u8 {
    switch (model) {
        .llama, .gpt2 => {
            if (token.len == 1) return token[0];
            return error.InvalidByteToken;
        },
        .tiktoken => {
            // 格式: <0xE4> 或 b'<0xE4>'
            var s = token;
            if (s.len >= 2 and s[0] == 'b' and s[1] == '\'') {
                s = s[2..];
            }
            if (s.len >= 4 and s[0] == '<' and s[1] == '0' and s[2] == 'x') {
                const end = std.mem.indexOfScalar(u8, s, '>') orelse return error.InvalidByteToken;
                const hex_str = s[3..end];
                if (hex_str.len == 2) {
                    return std.fmt.parseInt(u8, hex_str, 16);
                }
            }
            return error.InvalidByteToken;
        },
        .replit => {
            // 格式: b'<0xE4>'
            if (token.len >= 2 and token[0] == 'b' and token[1] == '\'') {
                const inner = token[2..];
                if (inner.len >= 4 and inner[0] == '<' and inner[1] == '0' and inner[2] == 'x') {
                    const end = std.mem.indexOfScalar(u8, inner, '>') orelse return error.InvalidByteToken;
                    const hex_str = inner[3..end];
                    if (hex_str.len == 2) {
                        return std.fmt.parseInt(u8, hex_str, 16);
                    }
                }
            }
            return error.InvalidByteToken;
        },
        .unknown => {
            // 尝试多种格式
            if (token.len == 1) return token[0];
            if (token.len >= 4 and token[0] == '<' and token[1] == '0' and token[2] == 'x') {
                const end = std.mem.indexOfScalar(u8, token, '>') orelse return error.InvalidByteToken;
                const hex_str = token[3..end];
                if (hex_str.len == 2) {
                    return std.fmt.parseInt(u8, hex_str, 16);
                }
            }
            return error.InvalidByteToken;
        },
    }
}

/// 判断 token 字符串是否为字节 token 格式
pub fn isByteTokenFormat(token: []const u8) bool {
    return (token.len == 1) or
        (token.len >= 4 and token[0] == '<' and token[1] == '0' and token[2] == 'x') or
        (token.len >= 2 and token[0] == 'b' and token[1] == '\'');
}

/// 解析字节 token（自动检测格式）
pub fn parseByteToken(token: []const u8) !u8 {
    return extractByteFromToken(token, .unknown);
}

/// 推断 token 是否为字节 token
pub fn inferIsByteToken(token: []const u8) bool {
    if (token.len == 1) return true;
    if (token.len >= 4 and token[0] == '<' and token[1] == '0' and token[2] == 'x') return true;
    if (token.len >= 2 and token[0] == 'b' and token[1] == '\'') return true;
    return false;
}

// ============================================================================
// 调试工具
// ============================================================================
pub fn hexDump(data: []const u8) void {
    _ = data;
    // Zig 0.16.0: std.io 被移除，hexDump 暂时禁用
}
