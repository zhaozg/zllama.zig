//! Unicode 属性查询模块
//!
//! 基于 uucode 库提供准确的 Unicode 属性查询，替代手写的启发式判断函数。
//! 支持查询：字母、数字、标点、空白、CJK 等 Unicode 类别。
//!
//! 使用方式：
//! ```zig
//! const unicode = @import("unicode");
//! if (unicode.isLetter(codepoint)) { ... }
//! if (unicode.isDigit(codepoint)) { ... }
//! ```

const std = @import("std");
const uucode = @import("uucode");

/// Unicode General Category 枚举
pub const GeneralCategory = uucode.types.GeneralCategory;

/// 判断 codepoint 是否为 Unicode 字母 (L* 类别)
pub inline fn isLetter(cp: u21) bool {
    return uucode.get(.is_alphabetic, cp);
}

/// 判断 codepoint 是否为 Unicode 大写字母 (Lu 类别)
pub inline fn isUppercase(cp: u21) bool {
    return uucode.get(.is_uppercase, cp);
}

/// 判断 codepoint 是否为 Unicode 小写字母 (Ll 类别)
pub inline fn isLowercase(cp: u21) bool {
    return uucode.get(.is_lowercase, cp);
}

/// 判断 codepoint 是否为 Unicode 数字 (Nd, Nl, No 类别)
/// 包括：ASCII 数字 (0-9)、上标/下标数字、分数数字、罗马数字等
/// 判断 codepoint 是否为 Unicode 数字 (Nd, Nl, No 类别)
/// 包括：ASCII 数字 (0-9)、上标/下标数字、分数数字、罗马数字等
pub inline fn isDigit(cp: u21) bool {
    const cat = uucode.get(.general_category, cp);
    return cat == .number_decimal_digit or
        cat == .number_letter or
        cat == .number_other;
}

/// 判断 codepoint 是否为 Unicode 十进制数字 (Nd 类别)
/// 仅包括：0-9 以及各种文字中的十进制数字
pub inline fn isDecimalDigit(cp: u21) bool {
    return uucode.get(.general_category, cp) == .number_decimal_digit;
}
/// 判断 codepoint 是否为 Unicode 标点符号 (P* 类别)
pub inline fn isPunctuation(cp: u21) bool {
    const cat = uucode.get(.general_category, cp);
    return switch (cat) {
        .punctuation_connector,
        .punctuation_dash,
        .punctuation_open,
        .punctuation_close,
        .punctuation_initial_quote,
        .punctuation_final_quote,
        .punctuation_other,
        => true,
        else => false,
    };
}

/// 判断 codepoint 是否为 Unicode 符号 (S* 类别)
/// 包括：数学符号、货币符号、修饰符符号、其他符号
pub inline fn isSymbol(cp: u21) bool {
    const cat = uucode.get(.general_category, cp);
    return switch (cat) {
        .symbol_math,
        .symbol_currency,
        .symbol_modifier,
        .symbol_other,
        => true,
        else => false,
    };
}

/// 判断 codepoint 是否为 Unicode 标点或符号 (P* 或 S* 类别)
pub inline fn isPunctuationOrSymbol(cp: u21) bool {
    return isPunctuation(cp) or isSymbol(cp);
}

/// 判断 codepoint 是否为 Unicode 空白字符 (Zs 类别)
/// 注意：不包括控制字符中的空白（如 tab, newline, carriage return）
pub inline fn isSpaceSeparator(cp: u21) bool {
    return uucode.get(.general_category, cp) == .separator_space;
}

/// 判断 codepoint 是否为 Unicode 分隔符 (Z* 类别)
/// 包括：空格分隔符 (Zs)、行分隔符 (Zl)、段落分隔符 (Zp)
pub inline fn isSeparator(cp: u21) bool {
    const cat = uucode.get(.general_category, cp);
    return switch (cat) {
        .separator_space,
        .separator_line,
        .separator_paragraph,
        => true,
        else => false,
    };
}

/// 判断 codepoint 是否为 CJK 统一表意文字
/// 包括：CJK 统一表意文字、CJK 扩展 A/B/C/D/E/F/G、CJK 兼容表意文字
pub inline fn isCJK(cp: u21) bool {
    return switch (cp) {
        // CJK Radicals Supplement
        0x2E80...0x2EFF,
        // Kangxi Radicals
        0x2F00...0x2FDF,
        // Ideographic Description Characters
        0x2FF0...0x2FFF,
        // CJK Symbols and Punctuation
        0x3000...0x303F,
        // CJK Strokes
        0x31C0...0x31EF,
        // Enclosed CJK Letters and Months
        0x3200...0x32FF,
        // CJK Compatibility
        0x3300...0x33FF,
        // CJK Unified Ideographs
        0x4E00...0x9FFF,
        // CJK Unified Ideographs Extension A
        0x3400...0x4DBF,
        // Yijing Hexagram Symbols
        0x4DC0...0x4DFF,
        // CJK Compatibility Ideographs
        0xF900...0xFAFF,
        // CJK Unified Ideographs Extension B
        0x20000...0x2A6DF,
        // CJK Unified Ideographs Extension C
        0x2A700...0x2B73F,
        // CJK Unified Ideographs Extension D
        0x2B740...0x2B81F,
        // CJK Unified Ideographs Extension E
        0x2B820...0x2CEAF,
        // CJK Unified Ideographs Extension F
        0x2CEB0...0x2EBEF,
        // CJK Unified Ideographs Extension G
        0x30000...0x3134F,
        // CJK Compatibility Ideographs Supplement
        0x2F800...0x2FA1F,
        => true,
        else => false,
    };
}

/// 判断 codepoint 是否为 emoji
pub inline fn isEmoji(cp: u21) bool {
    return uucode.get(.is_emoji, cp);
}

/// 获取 codepoint 的 Unicode 名称
pub inline fn name(cp: u21) []const u8 {
    return uucode.get(.name, cp);
}

/// 获取 codepoint 的 General Category
pub inline fn generalCategory(cp: u21) GeneralCategory {
    return uucode.get(.general_category, cp);
}

/// 从 UTF-8 文本中解码 codepoint
/// 返回 codepoint 和该字符的字节长度
pub fn decodeCodepoint(text: []const u8, pos: usize) struct { cp: u21, len: usize } {
    if (pos >= text.len) return .{ .cp = 0, .len = 0 };
    const b = text[pos];
    if (b < 0x80) return .{ .cp = b, .len = 1 };
    if (b < 0xC0) return .{ .cp = b, .len = 1 }; // continuation byte, treat as single byte
    if (b < 0xE0 and pos + 1 < text.len) {
        return .{ .cp = @as(u21, b & 0x1F) << 6 | @as(u21, text[pos + 1] & 0x3F), .len = 2 };
    }
    if (b < 0xF0 and pos + 2 < text.len) {
        return .{
            .cp = @as(u21, b & 0x0F) << 12 | @as(u21, text[pos + 1] & 0x3F) << 6 | @as(u21, text[pos + 2] & 0x3F),
            .len = 3,
        };
    }
    if (pos + 3 < text.len) {
        return .{
            .cp = @as(u21, b & 0x07) << 18 | @as(u21, text[pos + 1] & 0x3F) << 12 | @as(u21, text[pos + 2] & 0x3F) << 6 | @as(u21, text[pos + 3] & 0x3F),
            .len = 4,
        };
    }
    return .{ .cp = b, .len = 1 };
}

/// 判断 UTF-8 文本中指定位置的字符是否为 Unicode 字母
pub fn isLetterAt(text: []const u8, pos: usize) bool {
    const decoded = decodeCodepoint(text, pos);
    if (decoded.len == 0) return false;
    return isLetter(decoded.cp);
}

/// 判断 codepoint 是否为 Latin 字母（用于 DeepSeek-LLM 等模型的预分词）
/// 对应 llama.cpp 中 DeepSeek-LLM 的 Latin 字母范围：
/// [A-Za-zµÀ-ÖØ-öø-ƺ...] 包括 Latin、Greek、Cyrillic 等字母
/// 但不包括 CJK 表意文字和数字
pub inline fn isLatinLetter(cp: u21) bool {
    // Must be a letter (alphabetic) but not CJK, not a digit, and not emoji
    if (isDigit(cp)) return false;
    if (isCJK(cp)) return false;
    if (isEmoji(cp)) return false;
    return isLetter(cp);
}


/// 判断 UTF-8 文本中指定位置的字符是否为 Unicode 数字
pub fn isDigitAt(text: []const u8, pos: usize) bool {
    const decoded = decodeCodepoint(text, pos);
    if (decoded.len == 0) return false;
    return isDigit(decoded.cp);
}

/// 判断 UTF-8 文本中指定位置的字符是否为 Unicode 标点
pub fn isPunctuationAt(text: []const u8, pos: usize) bool {
    const decoded = decodeCodepoint(text, pos);
    if (decoded.len == 0) return false;
    return isPunctuation(decoded.cp);
}



/// 判断 UTF-8 文本中指定位置的字符是否为 Unicode 标点或符号
pub fn isPunctuationOrSymbolAt(text: []const u8, pos: usize) bool {
    const decoded = decodeCodepoint(text, pos);
    if (decoded.len == 0) return false;
    return isPunctuationOrSymbol(decoded.cp);
}

/// 判断 UTF-8 文本中指定位置的字符是否为 CJK
pub fn isCJKAt(text: []const u8, pos: usize) bool {
    const decoded = decodeCodepoint(text, pos);
    if (decoded.len == 0) return false;
    return isCJK(decoded.cp);
}

/// 判断 UTF-8 文本中指定位置的字符是否为 emoji
pub fn isEmojiAt(text: []const u8, pos: usize) bool {
    const decoded = decodeCodepoint(text, pos);
    if (decoded.len == 0) return false;
    return isEmoji(decoded.cp);
}

/// 获取 UTF-8 文本中指定位置字符的字节长度
pub fn charLen(text: []const u8, pos: usize) usize {
    return decodeCodepoint(text, pos).len;
}

// ============================================================================
// ASCII 快速路径函数
// 用于预分词器中快速扫描单字节 ASCII 字符，避免对每个字符都调用 uucode。
// 这些函数仅处理 ASCII 范围（0x00-0x7F），非 ASCII 字符应使用对应的
// Unicode codepoint 函数（如 isLetter(cp)、isDigit(cp) 等）。
// ============================================================================

/// 判断 ASCII 字节是否为空白字符
/// 包括：空格、制表符、换行符、回车符、垂直制表符、换页符
pub inline fn isAsciiWhitespace(c: u8) bool {
    return switch (c) {
        ' ', '\t', '\n', '\r', 0x0B, 0x0C => true,
        else => false,
    };
}

/// 判断 ASCII 字节是否为英文字母 (a-z, A-Z)
pub inline fn isAsciiLetter(c: u8) bool {
    return switch (c) {
        'a'...'z', 'A'...'Z' => true,
        else => false,
    };
}

/// 判断 ASCII 字节是否为数字 (0-9)
pub inline fn isAsciiDigit(c: u8) bool {
    return switch (c) {
        '0'...'9' => true,
        else => false,
    };
}

/// 判断 ASCII 字节是否为标点或符号
/// 包括：! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
pub inline fn isAsciiPunctuationOrSymbol(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

/// 判断 ASCII 字节是否为 Latin 字母（含 Latin-1 Supplement 扩展）
/// 即 ASCII 字母或字节值 >= 0xC0（粗略近似 Latin-1 Supplement 中的字母）
pub inline fn isAsciiLatinLetter(c: u8) bool {
    return isAsciiLetter(c) or c >= 0xC0;
}

/// 判断 UTF-8 文本片段是否全部由 ASCII 空白字符组成
pub fn isAllWhitespace(s: []const u8) bool {
    for (s) |c| {
        if (!isAsciiWhitespace(c)) return false;
    }
    return true;
}

/// 判断 ASCII 字节是否为 Bloom 模型分隔符
/// 包括：. , ! ?
pub inline fn isBloomSeparator(c: u8) bool {
    return switch (c) {
        '.', ',', '!', '?' => true,
        else => false,
    };
}

test "isLetter" {
    // ASCII letters
    try std.testing.expect(isLetter('A'));
    try std.testing.expect(isLetter('z'));
    // Non-letters
    try std.testing.expect(!isLetter('0'));
    try std.testing.expect(!isLetter(' '));
    try std.testing.expect(!isLetter('.'));
    // Latin extended
    try std.testing.expect(isLetter(0x00C0)); // À
    try std.testing.expect(isLetter(0x00FF)); // ÿ
    // CJK
    try std.testing.expect(isLetter(0x4E00)); // 一
    try std.testing.expect(isLetter(0x3400));
    // Greek
    try std.testing.expect(isLetter(0x03B1)); // α
    // Arabic
    try std.testing.expect(isLetter(0x0627)); // ا
    // Emoji is NOT a letter
    try std.testing.expect(!isLetter(0x1F600)); // 😀
}

test "isDigit" {
    // ASCII digits
    try std.testing.expect(isDigit('0'));
    try std.testing.expect(isDigit('9'));
    // Non-digits
    try std.testing.expect(!isDigit('A'));
    try std.testing.expect(!isDigit(' '));
    // Unicode digits
    try std.testing.expect(isDigit(0x00B2)); // ² (No)
    try std.testing.expect(isDigit(0x00B3)); // ³ (No)
    try std.testing.expect(isDigit(0x00B9)); // ¹ (No)
    try std.testing.expect(isDigit(0x00BC)); // ¼ (No)
    try std.testing.expect(isDigit(0x00BD)); // ½ (No)
    try std.testing.expect(isDigit(0x00BE)); // ¾ (No)
    // CJK is not a digit
    try std.testing.expect(!isDigit(0x4E00));
}

test "isPunctuationOrSymbol" {
    try std.testing.expect(isPunctuationOrSymbol('.'));
    try std.testing.expect(isPunctuationOrSymbol(','));
    try std.testing.expect(isPunctuationOrSymbol('!'));
    try std.testing.expect(isPunctuationOrSymbol('?'));
    try std.testing.expect(isPunctuationOrSymbol('('));
    try std.testing.expect(isPunctuationOrSymbol(')'));
    try std.testing.expect(isPunctuationOrSymbol('+'));
    try std.testing.expect(isPunctuationOrSymbol('$'));
    try std.testing.expect(!isPunctuationOrSymbol('A'));
    try std.testing.expect(!isPunctuationOrSymbol('0'));
    try std.testing.expect(!isPunctuationOrSymbol(' '));
}

test "isCJK" {
    try std.testing.expect(isCJK(0x4E00)); // 一
    try std.testing.expect(isCJK(0x9FFF)); // 鿿
    try std.testing.expect(isCJK(0x3400)); // 㐀
    try std.testing.expect(isCJK(0x2F800)); // 丽
    try std.testing.expect(!isCJK('A'));
    try std.testing.expect(!isCJK('0'));
    try std.testing.expect(!isCJK(0x03B1)); // α
}

test "decodeCodepoint" {
    // ASCII
    const d1 = decodeCodepoint("A", 0);
    try std.testing.expectEqual(@as(u21, 'A'), d1.cp);
    try std.testing.expectEqual(@as(usize, 1), d1.len);

    // 2-byte: À (U+00C0)
    const d2 = decodeCodepoint("\xC3\x80", 0);
    try std.testing.expectEqual(@as(u21, 0x00C0), d2.cp);
    try std.testing.expectEqual(@as(usize, 2), d2.len);

    // 3-byte: 一 (U+4E00)
    const d3 = decodeCodepoint("\xE4\xB8\x80", 0);
    try std.testing.expectEqual(@as(u21, 0x4E00), d3.cp);
    try std.testing.expectEqual(@as(usize, 3), d3.len);

    // 4-byte: 😀 (U+1F600)
    const d4 = decodeCodepoint("\xF0\x9F\x98\x80", 0);
    try std.testing.expectEqual(@as(u21, 0x1F600), d4.cp);
    try std.testing.expectEqual(@as(usize, 4), d4.len);
}

test "isLetterAt" {
    try std.testing.expect(isLetterAt("Hello", 0));
    try std.testing.expect(!isLetterAt("123", 0));
    try std.testing.expect(isLetterAt("\xC3\x80", 0)); // À
    try std.testing.expect(isLetterAt("\xE4\xB8\x80", 0)); // 一
    try std.testing.expect(!isLetterAt("\xF0\x9F\x98\x80", 0)); // 😀
}

test "isDigitAt" {
    try std.testing.expect(isDigitAt("123", 0));
    try std.testing.expect(!isDigitAt("ABC", 0));
    try std.testing.expect(isDigitAt("\xC2\xB2", 0)); // ²
    try std.testing.expect(isDigitAt("\xC2\xBC", 0)); // ¼
}
