//! 分词器模块
//!
//! 提供 BPE/SPM 分词器的完整实现，包括编码和解码。
//! 支持多种分词器模型类型（llama, gpt2, tiktoken, replit）。
//!
//! Vocab（src/vocab.zig）持有词表数据和配置。
//! Tokenizer 封装 Vocab，在此基础上提供 encode/decode 能力。
//!
//! 参考 llama.cpp 的 llama-vocab.cpp 实现。

const std = @import("std");
const gguf = @import("gguf");
const vocab = @import("vocab");

// 重新导出 vocab 类型，保持向后兼容
pub const VocabType = vocab.VocabType;
pub const TokenType = vocab.TokenType;
pub const TokenAttr = vocab.TokenAttr;
pub const PreType = vocab.PreType;
pub const SpecialTokenIds = vocab.SpecialTokenIds;
pub const Vocab = vocab.Vocab;
pub const TokenData = vocab.TokenData;

// 内部子模块
pub const types = @import("types.zig");
pub const trie = @import("trie.zig");
pub const encode_mod = @import("encode.zig");
pub const decode_mod = @import("decode.zig");
pub const bpe = @import("bpe.zig");
pub const utils = @import("utils.zig");

// 向后兼容的类型别名（逐步弃用）
pub const TokenizerModel = VocabType;
pub const PreTokenizerType = PreType;
pub const SpecialTokens = SpecialTokenIds;
pub const VocabEntry = types.VocabEntry;
pub const TokenizerConfig = types.TokenizerConfig;

// 重新导出常用函数
pub const generateBytesToUnicode = utils.generateBytesToUnicode;
pub const extractByteFromToken = utils.extractByteFromToken;
pub const isByteTokenFormat = utils.isByteTokenFormat;
pub const parseByteToken = utils.parseByteToken;
pub const inferIsByteToken = utils.inferIsByteToken;
pub const hexDump = utils.hexDump;

// Trie 类型
pub const TrieNode = trie.TrieNode;
pub const MatchResult = trie.MatchResult;

const log = std.log.scoped(.tokenizer);

// ============================================================================
// 特殊 Token 缓存条目
// ============================================================================

/// 缓存的特殊 token，用于 parse_special 预分词匹配
/// 与 llama.cpp 的 cache_special_tokens 对应
pub const CacheSpecialToken = struct {
    id: u32,
    text: []const u8, // 指向 vocab 中的 owned 字符串（生命周期与 vocab 一致）
    attr: TokenAttr,
};

// ============================================================================
// GPT-2 字节编码映射
// ============================================================================

/// GPT-2 bytes_to_unicode 映射：byte → UTF-8 编码的 unicode codepoint
/// 与 OpenAI 的 bytes_to_unicode() 保持一致
const BytesToUnicodeMap = struct {
    mapping: [256][]const u8,
    reverse: std.StringHashMap(u8),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !BytesToUnicodeMap {
        var bs: [256]bool = [_]bool{false} ** 256;
        var byte_to_cp: [256]u32 = undefined;

        // 第一组：可打印 ASCII（33-126）
        for (0x21..0x7F) |b| {
            bs[b] = true;
            byte_to_cp[b] = @intCast(b);
        }
        // 第二组：161-172（0xA1-0xAC）
        for (0xA1..0xAD) |b| {
            bs[b] = true;
            byte_to_cp[b] = @intCast(b);
        }
        // 第三组：174-255（0xAE-0xFF）
        for (0xAE..0x100) |b| {
            bs[b] = true;
            byte_to_cp[b] = @intCast(b);
        }
        // 剩余字节：0-32, 127-160, 173 → 256 + n
        var n: u32 = 0;
        for (0..256) |b| {
            if (!bs[b]) {
                byte_to_cp[b] = 256 + n;
                n += 1;
            }
        }

        var map = BytesToUnicodeMap{
            .mapping = undefined,
            .reverse = std.StringHashMap(u8).init(allocator),
            .allocator = allocator,
        };

        for (0..256) |b| {
            var buf: [4]u8 = undefined;
            const cp = byte_to_cp[b];
            const len = std.unicode.utf8Encode(@intCast(cp), &buf) catch unreachable;
            map.mapping[b] = try allocator.dupe(u8, buf[0..len]);
        }

        for (0..256) |b| {
            try map.reverse.put(map.mapping[b], @intCast(b));
        }

        return map;
    }

    fn deinit(self: *BytesToUnicodeMap) void {
        for (&self.mapping) |mapped| {
            self.allocator.free(mapped);
        }
        self.reverse.deinit();
    }

    fn bytesToUnicode(self: *const BytesToUnicodeMap, byte: u8) []const u8 {
        return self.mapping[byte];
    }
};

// ============================================================================
// Tokenizer 主结构体
// ============================================================================

/// 分词器主结构体
/// 封装 Vocab 并提供 encode/decode 能力
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    /// 词表数据（持有所有权）
    vocab: Vocab,
    /// Trie 前缀树（用于贪婪最长匹配编码）
    trie_root: TrieNode,
    /// GPT-2 字节编码映射
    bytes_to_unicode_map: BytesToUnicodeMap,
    /// 用于 tokenToString 返回字节 token 的临时缓冲区
    byte_token_buf: [1]u8 = undefined,
    /// 缓存的特殊 token 列表（按 text 长度降序排列）
    cache_special_tokens: std.ArrayListUnmanaged(CacheSpecialToken) = .empty,

    // ========================================================================
    // 初始化
    // ========================================================================

    /// 从 GGUF 文件初始化分词器
    pub fn init(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Tokenizer {
        var tok = Tokenizer{
            .allocator = allocator,
            .vocab = try Vocab.init(gguf_file, allocator),
            .trie_root = TrieNode.init(allocator),
            .bytes_to_unicode_map = try BytesToUnicodeMap.init(allocator),
        };

        // 修复 byte_to_token 映射：对于 GPT-2 风格的 BPE 模型，
        // 字节 token 的 text 是经过 GPT-2 字节编码的 Unicode 字符，
        // 需要使用 bytes_to_unicode_map 的反向映射来正确建立 byte→token 关系
        try tok.fixByteToTokenMapping();

        // 构建 Trie（用于贪婪最长匹配编码）
        try tok.buildTrie();

        // 构建特殊 token 缓存
        try tok.buildCacheSpecialTokens();

        return tok;
    }

    /// 修复 byte_to_token 映射，使用 bytes_to_unicode_map 的反向映射
    /// 同时处理两种字节 token 格式：
    /// 1. GPT-2 编码的多字节 token（如 "Ã" = [0xC3,0x83] 表示 byte 0xC3）
    /// 2. 原始单字节 token（如直接存储 byte 值）
    /// 只设置尚未映射的 byte 槽位（first-wins），防止 BPE 合并 token
    /// 覆盖 byte token 映射。
    fn fixByteToTokenMapping(self: *Tokenizer) !void {
        const v = &self.vocab;
        const reverse = &self.bytes_to_unicode_map.reverse;

        for (v.tokens, 0..) |td, id| {
            const uid = @as(u32, @intCast(id));
            // 尝试通过反向映射查找字节值（处理 GPT-2 编码的多字节 token）
            if (reverse.get(td.text)) |byte| {
                if (v.byte_to_token[byte] == v.special.unk) {
                    v.byte_to_token[byte] = uid;
                }
            }
            // 也处理直接的单字节 token（回退：对于 raw byte 存储的模型）
            if (td.type == .byte and td.text.len == 1) {
                if (v.byte_to_token[td.text[0]] == v.special.unk) {
                    v.byte_to_token[td.text[0]] = uid;
                }
            }
        }
    }

    fn buildTrie(self: *Tokenizer) !void {
        for (self.vocab.tokens, 0..) |td, id| {
            // NOTE: 添加所有类型的 token 到 Trie，包括 byte 类型
            // 对于 BPE 模型，字节 token（0-255）通常是 byte 类型，
            // 必须添加到 Trie 中才能实现正确的贪婪匹配
            if (td.type == .normal or td.type == .control or td.type == .byte) {
                try trie.addToTrie(&self.trie_root, td.text, @intCast(id), self.allocator);
            }
        }
    }

    fn buildCacheSpecialTokens(self: *Tokenizer) !void {
        var special_count: usize = 0;
        for (self.vocab.tokens, 0..) |td, id| {
            if (id < self.vocab.tokens.len) {
                const tt = td.type;
                if (tt == .control or tt == .user_defined or tt == .unknown) {
                    special_count += 1;
                }
            } else if (td.text.len > 2 and td.text[0] == '<' and td.text[td.text.len - 1] == '>') {
                special_count += 1;
            }
        }

        if (special_count > 0) {
            try self.cache_special_tokens.ensureTotalCapacity(self.allocator, special_count);
            for (self.vocab.tokens, 0..) |td, id| {
                const tt = td.type;
                if (id < self.vocab.tokens.len) {
                    if (tt == .control or tt == .user_defined or tt == .unknown) {
                        self.cache_special_tokens.appendAssumeCapacity(.{
                            .id = @intCast(id),
                            .text = td.text,
                            .attr = vocab.TokenAttr.fromType(tt),
                        });
                    }
                } else if (td.text.len > 2 and td.text[0] == '<' and td.text[td.text.len - 1] == '>') {
                    self.cache_special_tokens.appendAssumeCapacity(.{
                        .id = @intCast(id),
                        .text = td.text,
                        .attr = vocab.TokenAttr{ .control = true },
                    });
                }
            }

            // 按 text 长度降序排列（最长优先匹配）
            std.mem.sort(CacheSpecialToken, self.cache_special_tokens.items, {}, struct {
                fn lessThan(_: void, a: CacheSpecialToken, b: CacheSpecialToken) bool {
                    return a.text.len > b.text.len;
                }
            }.lessThan);

            log.info("Tokenizer: {d} special tokens cached for parse_special", .{self.cache_special_tokens.items.len});
        }
    }

    // ========================================================================
    // 资源管理
    // ========================================================================

    pub fn deinit(self: *Tokenizer) void {
        self.cache_special_tokens.deinit(self.allocator);
        self.trie_root.deinit(self.allocator);
        self.bytes_to_unicode_map.deinit();
        self.vocab.deinit();
        self.* = undefined;
    }

    // ========================================================================
    // 编码
    // ========================================================================

    /// 将文本编码为 token ID 列表
    /// - add_special: 是否添加 BOS/EOS 等特殊 token
    /// - parse_special: 是否解析文本中的特殊 token（如 <|turn|>、<|audio|>）
    pub fn encode(self: *Tokenizer, text: []const u8, add_special: bool, parse_special: bool) !std.ArrayListUnmanaged(u32) {
        const add_bos = add_special and self.vocab.getAddBos();
        const add_eos = add_special and self.vocab.getAddEos();

        const cache_special: ?[]const CacheSpecialToken = if (parse_special and self.cache_special_tokens.items.len > 0)
            self.cache_special_tokens.items
        else
            null;

        // 对于 gemma-4/bert 等使用 <0xXX> 格式字节 token 的模型，不使用 GPT-2 字节编码
        const model_type = self.vocab.getType();
        const needs_gpt2_encoding = model_type != .gemma4 and model_type != .bert;

        const enc_config = encode_mod.EncodeConfig{
            .allocator = self.allocator,
            .special = self.vocab.getSpecial(),
            .pre_type = self.vocab.getPreType(),
            .model = model_type,
            .vocab = .empty, // initialized empty; not used (all lookups via callbacks)
            .merges = self.vocab.merges,
            .trie_root = &self.trie_root,
            .tokenToStringFn = tokenToStringWrapper,
            .textToTokenFn = textToTokenWrapper,
            .unicodeToByte = if (needs_gpt2_encoding) &self.bytes_to_unicode_map.reverse else null,
            .byteToTokenIdFn = byteToTokenIdWrapper,
            .bytesToUnicodeFn = if (needs_gpt2_encoding) bytesToUnicodeWrapper else null,
            .escape_whitespaces = self.vocab.getEscapeWhitespaces(),
            .ctx = @ptrCast(self),
            .cache_special_tokens = cache_special,
        };

        return encode_mod.encode(text, add_bos, add_eos, self.vocab.getAddSpacePrefix(), self.vocab.getIgnoreMerges(), parse_special, &enc_config);
    }

    // ========================================================================
    // 解码
    // ========================================================================

    pub fn decode(self: *const Tokenizer, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
        const dec_config = decode_mod.DecodeConfig{
            .model = self.vocab.getType(),
            .special = self.vocab.getSpecial(),
            .token_types = self.tokenTypesList(allocator),
            .vocab = self.vocabEntryList(allocator),
            .clean_spaces = self.vocab.getCleanSpaces(),
            .escape_whitespaces = self.vocab.getEscapeWhitespaces(),
            .unicode_to_byte = &self.bytes_to_unicode_map.reverse,
        };
        return decode_mod.decode(token_ids, &dec_config, allocator);
    }

    /// 构建 token types 列表（decode 模块需要）
    fn tokenTypesList(self: *const Tokenizer, allocator: std.mem.Allocator) std.ArrayListUnmanaged(TokenType) {
        var list: std.ArrayListUnmanaged(TokenType) = .empty;
        for (self.vocab.tokens) |td| {
            list.append(allocator, td.type) catch break;
        }
        return list;
    }

    /// 构建 vocab entry 列表（decode 模块需要）
    fn vocabEntryList(self: *const Tokenizer, allocator: std.mem.Allocator) std.ArrayListUnmanaged(types.VocabEntry) {
        var list: std.ArrayListUnmanaged(types.VocabEntry) = .empty;
        for (self.vocab.tokens) |td| {
            const entry: types.VocabEntry = switch (td.type) {
                .byte => types.VocabEntry{ .byte = td.text[0] },
                else => types.VocabEntry{ .normal = td.text },
            };
            list.append(allocator, entry) catch break;
        }
        return list;
    }

    // ========================================================================
    // 辅助方法
    // ========================================================================

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.nTokens();
    }

    pub fn isSpecialToken(self: *const Tokenizer, token_id: u32) bool {
        return self.vocab.isSpecial(token_id);
    }

    /// 判断 token 是否为 EOG (End-of-Generation) token
    pub fn isEog(self: *const Tokenizer, token_id: u32) bool {
        return self.vocab.isEog(token_id);
    }

    const eog_token_names = [_][]const u8{
        "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>",
        "<|end|>", "<end_of_turn>", "</s>", "<｜end▁of▁sentence｜>",
    };

    pub fn isEogText(self: *const Tokenizer, text: []const u8) bool {
        _ = self;
        for (eog_token_names) |name| {
            if (std.mem.indexOf(u8, text, name) != null) return true;
        }
        return false;
    }

    const skip_token_names = [_][]const u8{
        "<|channel|>", "<|channel>", "<channel|>",
        "<start_of_turn>", "<end_of_turn>",
    };

    pub fn isSkipToken(self: *const Tokenizer, token_id: u32) bool {
        const td = self.vocab.getTokenData(token_id) orelse return false;
        for (skip_token_names) |name| {
            if (std.mem.eql(u8, td.text, name)) return true;
        }
        return false;
    }

    pub fn byteToTokenId(self: *const Tokenizer, byte: u8) u32 {
        return self.vocab.byteToToken(byte);
    }

    /// 解码单个 token 到缓冲区
    pub fn decodeSingle(self: *const Tokenizer, token_id: u32, buf: []u8) !usize {
        if (self.isSpecialToken(token_id)) return 0;

        const td = self.vocab.getTokenData(token_id) orelse return 0;
        var written: usize = 0;

        switch (td.type) {
            .byte => {
                if (td.text.len > 0 and written < buf.len) {
                    buf[written] = td.text[0];
                    written += 1;
                }
            },
            else => {
                const model = self.vocab.getType();
                if (model == .tiktoken) {
                    return decodeTiktokenSingle(td.text, buf);
                } else if (model == .gpt2) {
                    return decodeGpt2Single(td.text, &self.bytes_to_unicode_map.reverse, buf);
                } else {
                    return decodeSPMSingle(td.text, buf);
                }
            },
        }

        return written;
    }

    /// 通过 token 名称（字符串）查找对应的 token ID
    pub fn textToToken(self: *const Tokenizer, text: []const u8) ?u32 {
        const match = trie.longestMatch(&self.trie_root, text, 0);
        if (match) |m| {
            if (m.len == text.len) return m.token_id;
        }
        return self.vocab.textToToken(text);
    }

    /// 将 token ID 转换为字符串表示（用于 BPE 合并）
    fn tokenToString(self: *Tokenizer, token_id: u32) ?[]const u8 {
        const td = self.vocab.getTokenData(token_id) orelse return null;
        // NOTE: byte 类型的 token 文本就是单个字节，直接返回 td.text 即可
        // 之前使用 byte_token_buf 会导致连续调用时数据被覆盖（BPE 合并中的 left_str/right_str）
        return td.text;
    }

    fn bytesToUnicodeFn(self: *const Tokenizer, byte: u8) []const u8 {
        return self.bytes_to_unicode_map.bytesToUnicode(byte);
    }
};

// ============================================================================
// 解码辅助函数（从旧 mod.zig 迁移）
// ============================================================================

fn decodeTiktokenSingle(ts: []const u8, buf: []u8) usize {
    var written: usize = 0;
    var rem = ts;
    while (rem.len > 0 and written < buf.len) {
        if (rem.len >= 4 and rem[0] == '<' and rem[1] == '0' and rem[2] == 'x') {
            const end = std.mem.indexOfScalar(u8, rem[1..], '>') orelse {
                buf[written] = rem[0];
                written += 1;
                rem = rem[1..];
                continue;
            };
            const hex_str = rem[2 .. 2 + end - 1];
            if (hex_str.len == 2) {
                const byte = std.fmt.parseInt(u8, hex_str, 16) catch {
                    const copy_len = @min(end + 1, buf.len - written);
                    @memcpy(buf[written..written+copy_len], rem[0..copy_len]);
                    written += copy_len;
                    rem = rem[end + 1 ..];
                    continue;
                };
                buf[written] = byte;
                written += 1;
                rem = rem[end + 1 ..];
                continue;
            }
        }
        buf[written] = rem[0];
        written += 1;
        rem = rem[1..];
    }
    return written;
}

fn decodeGpt2Single(ts: []const u8, unicode_to_byte: *const std.StringHashMap(u8), buf: []u8) usize {
    var written: usize = 0;
    var i: usize = 0;
    while (i < ts.len and written < buf.len) {
        const byte = ts[i];
        const cp_len = std.unicode.utf8ByteSequenceLength(byte) catch {
            i += 1;
            continue;
        };
        if (i + cp_len > ts.len) {
            i += 1;
            continue;
        }
        const cp_slice = ts[i..i+cp_len];
        if (cp_len == 1 and byte < 0x80) {
            buf[written] = byte;
            written += 1;
            i += 1;
            continue;
        }
        if (unicode_to_byte.get(cp_slice)) |b| {
            buf[written] = b;
            written += 1;
            i += cp_len;
            continue;
        }
        const copy_len = @min(cp_len, buf.len - written);
        @memcpy(buf[written..written+copy_len], cp_slice[0..copy_len]);
        written += copy_len;
        i += cp_len;
    }
    return written;
}

fn decodeSPMSingle(ts: []const u8, buf: []u8) usize {
    var written: usize = 0;
    var i: usize = 0;
    while (i < ts.len and written < buf.len) {
        if (ts[i] == '<' and i + 3 < ts.len and ts[i + 1] == '0' and ts[i + 2] == 'x') {
            const end = std.mem.indexOfScalar(u8, ts[i + 1 ..], '>') orelse {
                buf[written] = ts[i];
                written += 1;
                i += 1;
                continue;
            };
            const hex_str = ts[i + 3 .. i + 1 + end];
            if (hex_str.len == 2) {
                if (std.fmt.parseInt(u8, hex_str, 16)) |byte| {
                    buf[written] = byte;
                    written += 1;
                    i = i + 1 + end + 1;
                    continue;
                } else |_| {}
            }
            buf[written] = ts[i];
            written += 1;
            i += 1;
            continue;
        }
        if (i + 2 < ts.len and ts[i] == 0xE2 and ts[i+1] == 0x96 and ts[i+2] == 0x81) {
            buf[written] = ' ';
            written += 1;
            i += 3;
            continue;
        }
        if (i + 1 < ts.len and ts[i] == 0xC4 and ts[i+1] == 0xA0) {
            buf[written] = ' ';
            written += 1;
            i += 2;
            continue;
        }
        buf[written] = ts[i];
        written += 1;
        i += 1;
    }
    return written;
}

// ============================================================================
// 包装函数（用于回调）
// ============================================================================

fn tokenToStringWrapper(token_id: u32, ctx: ?*anyopaque) ?[]const u8 {
    const self: *Tokenizer = @ptrCast(@alignCast(ctx.?));
    return self.tokenToString(token_id);
}

fn byteToTokenIdWrapper(byte: u8, ctx: ?*anyopaque) u32 {
    const self: *const Tokenizer = @ptrCast(@alignCast(ctx.?));
    return self.byteToTokenId(byte);
}

fn bytesToUnicodeWrapper(byte: u8, ctx: ?*anyopaque) []const u8 {
    const self: *const Tokenizer = @ptrCast(@alignCast(ctx.?));
    return self.bytesToUnicodeFn(byte);
}

fn textToTokenWrapper(text: []const u8, ctx: ?*anyopaque) ?u32 {
    const self: *const Tokenizer = @ptrCast(@alignCast(ctx.?));
    return self.textToToken(text);
}
