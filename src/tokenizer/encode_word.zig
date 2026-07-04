//! 单词编码（GPT-2 字节编码 + BPE 合并）
const std = @import("std");
const mod = @import("mod.zig");
const unicode = mod.unicode;
const trie = mod.trie;
const bpe = mod.bpe;
const encode_config = @import("encode_config.zig");
const EncodeConfig = encode_config.EncodeConfig;

const SPM_SPACE = "\xE2\x96\x81";

fn escapeWhitespace(text: []const u8, allocator: std.mem.Allocator) ![]u8 {
    var space_count: usize = 0;
    for (text) |c| {
        if (c == ' ') space_count += 1;
    }
    if (space_count == 0) return allocator.dupe(u8, text);

    const escaped_len = text.len + space_count * 2; // space (1 byte) → ▁ (3 bytes)
    var buf = try allocator.alloc(u8, escaped_len);
    var j: usize = 0;
    for (text) |c| {
        if (c == ' ') {
            buf[j] = 0xE2;
            j += 1;
            buf[j] = 0x96;
            j += 1;
            buf[j] = 0x81;
            j += 1;
        } else {
            buf[j] = c;
            j += 1;
        }
    }
    return buf[0..j];
}

/// 获取给定位置 UTF-8 字符的字节长度
// ============================================================================
// GPT-2 字节编码转换
// ============================================================================

/// 将文本转换为 GPT-2 字节编码
pub fn toGpt2ByteEncoding(text: []const u8, bytesToUnicodeFn: *const fn (byte: u8, ctx: ?*anyopaque) []const u8, ctx: ?*anyopaque, allocator: std.mem.Allocator) ![]u8 {
    var total_len: usize = 0;
    for (text) |byte| {
        total_len += bytesToUnicodeFn(byte, ctx).len;
    }

    var result = try std.ArrayListUnmanaged(u8).initCapacity(allocator, total_len);
    errdefer result.deinit(allocator);

    for (text) |byte| {
        const mapped = bytesToUnicodeFn(byte, ctx);
        try result.appendSlice(allocator, mapped);
    }

    return result.items;
}

// ============================================================================
// 特殊 Token 预分词（parse_special 模式）
// ============================================================================

/// 编码单个词
pub fn encodeWord(
    word: []const u8,
    add_space_prefix: bool,
    _: bool, // ignore_merges — single-char whole-word matching is always attempted
    is_first: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    // 阶段 0：优先匹配原始单词（不添加任何前缀，不进行字节编码）
    // 仅当单词以空白开头时执行此匹配，这意味着预分词器已经将空白包含在单词中
    // （如 MPT 的 "  " 匹配 token 50276）。对于不以空白开头的单词，
    // 后续的 add_space_prefix 会添加空格前缀，此时匹配原始单词可能得到错误结果
    // （如 "½" 匹配到 token 121 而非正确的 GPT-2 编码形式 "Â½"）。
    if (word.len > 0 and unicode.isAsciiWhitespace(word[0])) {
        if (config.textToTokenFn(word, config.ctx)) |token_id| {
            try tokens.append(config.allocator, token_id);
            return tokens;
        }
    }

    const is_spm_model = config.model == .llama or config.model == .spm;

    // 步骤 1：确定基础文本（可能添加空格前缀）
    // 对于 escape_whitespaces 的模型（gemma-4 等），将空格转为 ▁ (U+2581)
    // 对于 SPM 模型，首词不添加空格前缀（除非词本身以空格开头）
    const BaseText = struct {
        text: []const u8,
        needs_free: bool,
    };
    const base = if (add_space_prefix and (!is_spm_model or !is_first)) blk: {
        if (is_spm_model) {
            if (word.len > 0 and unicode.isAsciiWhitespace(word[0])) {
                var ws_end: usize = 1;
                while (ws_end < word.len and unicode.isAsciiWhitespace(word[ws_end])) ws_end += 1;
                if (ws_end < word.len) {
                    break :blk BaseText{
                        .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word[ws_end..] }),
                        .needs_free = true,
                    };
                }
            }
            break :blk BaseText{
                .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word }),
                .needs_free = true,
            };
        } else if (config.escape_whitespaces) {
            if (word.len > 0 and unicode.isAsciiWhitespace(word[0])) {
                var ws_end: usize = 1;
                while (ws_end < word.len and unicode.isAsciiWhitespace(word[ws_end])) ws_end += 1;
                if (ws_end < word.len) {
                    break :blk BaseText{
                        .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word[ws_end..] }),
                        .needs_free = true,
                    };
                }
                // Word is all whitespace — keep as-is for token lookup
            }
            break :blk BaseText{ .text = word, .needs_free = false };
        } else {
            // If word already starts with whitespace (captured by ?\p{L}+ etc.),
            // don't add another space — it's already there from pre-tokenization.
            if (word.len > 0 and unicode.isAsciiWhitespace(word[0])) {
                break :blk BaseText{ .text = word, .needs_free = false };
            }
            break :blk BaseText{
                .text = try std.fmt.allocPrint(config.allocator, " {s}", .{word}),
                .needs_free = true,
            };
        }
    } else if (config.escape_whitespaces and word.len > 0 and unicode.isAsciiWhitespace(word[0])) blk: {
        var ws_end: usize = 1;
        while (ws_end < word.len and unicode.isAsciiWhitespace(word[ws_end])) ws_end += 1;
        if (ws_end < word.len) {
            break :blk BaseText{
                .text = try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word[ws_end..] }),
                .needs_free = true,
            };
        }
        // Word is all whitespace — keep as-is for token lookup
        break :blk BaseText{ .text = word, .needs_free = false };
    } else BaseText{ .text = word, .needs_free = false };

    const base_text = base.text;
    const base_needs_free = base.needs_free;
    errdefer {
        if (base_needs_free) config.allocator.free(base_text);
    }

    // 步骤 2：对基础文本进行 GPT-2 字节编码（如果需要）
    const use_gpt2_encoding = config.bytesToUnicodeFn != null and config.merges.count() > 0;

    const final_text: []const u8 = if (use_gpt2_encoding) blk: {
        const encoded = try toGpt2ByteEncoding(base_text, config.bytesToUnicodeFn.?, config.ctx, config.allocator);
        if (base_needs_free) config.allocator.free(base_text);
        break :blk encoded;
    } else base_text;

    const final_needs_free = if (use_gpt2_encoding) true else base_needs_free;

    defer {
        if (final_needs_free) config.allocator.free(@constCast(final_text));
    }

    // 阶段 1：整词优先匹配
    // 无论 ignore_merges 值如何，预分词得到的每个单词都应先尝试直接查表。
    // 如果词表中存在该单词，直接使用其 token ID，避免不必要的拆分和 BPE 合并。
    // 这是确保与 llama-tokenize 行为一致的关键。
    if (config.textToTokenFn(final_text, config.ctx)) |token_id| {
        try tokens.append(config.allocator, token_id);
        return tokens;
    }

    // 阶段 2：Tokenization
    // BPE 模型：先拆分为最小单元，再 BPE 合并。
    // - GPT-2 byte-encoded BPE (llama-bpe, qwen2 等): 通过 unicodeToByte
    //   映射回原始 byte，用 byteToTokenIdFn 查找字节 token。
    // - Non-byte-encoded BPE (gemma-4, bert): 直接按 UTF-8 字符查找 token。
    if (config.merges.count() > 0) {
        var pos: usize = 0;
        while (pos < final_text.len) {
            if (config.unicodeToByte) |utb| {
                // GPT-2 byte-encoded BPE: decode each UTF-8 code point back to raw byte
                const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                const ch = final_text[pos .. pos + @as(usize, ch_len)];
                if (utb.get(ch)) |byte| {
                    const tid = config.byteToTokenIdFn(byte, config.ctx);
                    try tokens.append(config.allocator, tid);
                    pos += ch.len;
                } else {
                    const tid = config.byteToTokenIdFn(final_text[pos], config.ctx);
                    try tokens.append(config.allocator, tid);
                    pos += 1;
                }
            } else {
                // Non-byte-encoded BPE: character-level lookup
                const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                const ch = final_text[pos .. pos + @as(usize, ch_len)];
                if (config.textToTokenFn(ch, config.ctx)) |tid| {
                    try tokens.append(config.allocator, tid);
                } else if (config.escape_whitespaces and ch.len == 1 and unicode.isAsciiWhitespace(ch[0])) {
                    if (config.textToTokenFn(SPM_SPACE, config.ctx)) |tid| {
                        try tokens.append(config.allocator, tid);
                    } else {
                        const tid = config.byteToTokenIdFn(ch[0], config.ctx);
                        try tokens.append(config.allocator, tid);
                    }
                } else {
                    for (ch) |byte| {
                        const tid = config.byteToTokenIdFn(byte, config.ctx);
                        try tokens.append(config.allocator, tid);
                    }
                }
                pos += ch.len;
            }
        }
    } else {
        // 非 BPE 模型（SPM 等）：Trie 贪婪最长匹配
        var pos: usize = 0;
        while (pos < final_text.len) {
            const match = trie.longestMatch(config.trie_root, final_text, pos);
            if (match) |m| {
                try tokens.append(config.allocator, m.token_id);
                pos += m.len;
            } else {
                if (is_spm_model) {
                    try tokens.append(config.allocator, config.special.unk);
                    const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                    pos += ch_len;
                } else if (config.unicodeToByte) |utb| {
                    const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                    const ch = final_text[pos .. pos + @as(usize, ch_len)];
                    if (utb.get(ch)) |byte| {
                        try tokens.append(config.allocator, config.byteToTokenIdFn(byte, config.ctx));
                        pos += ch.len;
                    } else {
                        try tokens.append(config.allocator, config.byteToTokenIdFn(final_text[pos], config.ctx));
                        pos += 1;
                    }
                } else {
                    try tokens.append(config.allocator, config.byteToTokenIdFn(final_text[pos], config.ctx));
                    pos += 1;
                }
            }
        }
    }

    // 阶段 3：BPE 合并（如果有合并规则）
    // 注意：即使 ignore_merges=true，如果整个词不在词表中，仍然需要 BPE 合并
    // 这与 llama.cpp 的行为一致
    if (config.merges.count() > 0) {
        try bpe.applyBpeMerges(&tokens, config.merges, config.tokenToStringFn, config.textToTokenFn, config.ctx, config.allocator);
    }

    return tokens;
}
