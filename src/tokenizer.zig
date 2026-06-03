//! 分词器模块
//!
//! 提供 BPE 分词器的完整实现，包括编码和解码。
//! 支持多种分词器模型类型（llama, gpt2, tiktoken, replit）。
//!
//! 参考 llama.cpp 的 llama-vocab.cpp 实现。

const std = @import("std");
const gguf = @import("gguf.zig");

pub const types = @import("tokenizer/types.zig");
pub const trie = @import("tokenizer/trie.zig");
pub const encode_mod = @import("tokenizer/encode.zig");
pub const decode_mod = @import("tokenizer/decode.zig");
pub const bpe = @import("tokenizer/bpe.zig");
pub const utils = @import("tokenizer/utils.zig");

const log = std.log.scoped(.tokenizer);

// 重新导出常用类型
pub const TokenType = types.TokenType;
pub const TokenizerModel = types.TokenizerModel;
pub const PreTokenizerType = types.PreTokenizerType;
pub const SpecialTokens = types.SpecialTokens;
pub const VocabEntry = types.VocabEntry;
pub const TokenizerConfig = types.TokenizerConfig;
pub const TrieNode = trie.TrieNode;
pub const MatchResult = trie.MatchResult;

// 重新导出常用函数
pub const generateBytesToUnicode = utils.generateBytesToUnicode;
pub const extractByteFromToken = utils.extractByteFromToken;
pub const isByteTokenFormat = utils.isByteTokenFormat;
pub const parseByteToken = utils.parseByteToken;
pub const inferIsByteToken = utils.inferIsByteToken;
pub const hexDump = utils.hexDump;

// ============================================================================
// Tokenizer 主结构体
// ============================================================================

/// 分词器主结构体
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    config: TokenizerConfig,
    special: SpecialTokens,
    vocab: std.ArrayListUnmanaged(VocabEntry),
    token_types: std.ArrayListUnmanaged(TokenType),
    merges: std.StringHashMap(u32),
    trie_root: TrieNode,
    byte_to_token_id: [256]?u32 = [_]?u32{null} ** 256,
    /// 用于 tokenToString 返回字节 token 的临时缓冲区
    byte_token_buf: [1]u8 = undefined,
    /// GPT-2 字节编码映射：byte -> unicode codepoint (作为 UTF-8 字符串)
    bytes_to_unicode: [256][]const u8 = undefined,
    /// GPT-2 反向映射：unicode codepoint (UTF-8) -> byte
    unicode_to_byte: std.StringHashMap(u8),

    // ========================================================================
    // 初始化
    // ========================================================================

    /// 从 GGUF 文件初始化分词器
    pub fn init(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Tokenizer {
        var tok = Tokenizer{
            .allocator = allocator,
            .config = TokenizerConfig.fromGGUF(gguf_file),
            .special = SpecialTokens.fromGGUF(gguf_file),
            .vocab = .empty,
            .token_types = .empty,
            .merges = std.StringHashMap(u32).init(allocator),
            .trie_root = TrieNode.init(allocator),
            .unicode_to_byte = std.StringHashMap(u8).init(allocator),
        };

        log.info("Tokenizer model: '{s}' -> {s}", .{
            gguf_file.getString("tokenizer.ggml.model") orelse "unknown",
            @tagName(tok.config.model),
        });

        // 初始化 GPT-2 字节编码映射
        try tok.initBytesToUnicode();

        // 读取 token_type
        if (gguf_file.metadata.get("tokenizer.ggml.token_type")) |val| {
            for (val.array_val) |v| {
                const tv = v.asU32() orelse 0;
                try tok.token_types.append(allocator, @as(TokenType, @enumFromInt(tv)));
            }
        }

        // 读取词表
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            for (val.array_val, 0..) |v, i| {
                const token_str = v.asString() orelse continue;
                const tt = if (i < tok.token_types.items.len) tok.token_types.items[i] else TokenType.normal;

                if (tt == .byte) {
                    const byte = utils.extractByteFromToken(token_str, tok.config.model) catch {
                        if (utils.inferIsByteToken(token_str)) {
                            const b = utils.parseByteToken(token_str) catch continue;
                            try tok.vocab.append(allocator, VocabEntry{ .byte = b });
                            tok.byte_to_token_id[b] = @intCast(i);
                        }
                        continue;
                    };
                    try tok.vocab.append(allocator, VocabEntry{ .byte = byte });
                    tok.byte_to_token_id[byte] = @intCast(i);
                } else {
                    const owned = try allocator.dupe(u8, token_str);
                    try tok.vocab.append(allocator, VocabEntry{ .normal = owned });
                    try trie.addToTrie(&tok.trie_root, token_str, @intCast(i), allocator);
                }
            }
        }

        log.info("Tokenizer: {d} entries (normal={d}, byte={d})", .{
            tok.vocab.items.len,
            countNormalEntries(tok.vocab.items),
            countByteEntries(tok.vocab.items),
        });

        // 读取 BPE 合并规则
        if (gguf_file.metadata.get("tokenizer.ggml.merges")) |val| {
            try tok.merges.ensureTotalCapacity(@intCast(val.array_val.len));
            for (val.array_val, 0..) |v, rank| {
                const merge_str = v.asString() orelse continue;
                try tok.merges.put(merge_str, @intCast(rank));
            }
            log.info("Tokenizer: {d} BPE merge rules", .{val.array_val.len});
        }

        return tok;
    }

    /// 初始化 GPT-2 字节编码映射
    fn initBytesToUnicode(self: *Tokenizer) !void {
        // GPT-2 的 bytes_to_unicode 映射
        // 将 0-255 字节映射到可打印的 Unicode 字符
        var bs: [256]bool = [_]bool{false} ** 256;
        var cs: [256]u32 = undefined;
        var n: u32 = 0;

        // 可打印 ASCII: 0x21-0x7E (! 到 ~)
        var ch: u32 = 0x21;
        while (ch <= 0x7E) {
            bs[ch] = true;
            cs[ch] = ch;
            ch += 1;
        }

        // Latin-1: 0xA1-0xAC (¡ 到 ¬)
        ch = 0xA1;
        while (ch <= 0xAC) {
            bs[ch] = true;
            cs[ch] = ch;
            ch += 1;
        }

        // Latin-1: 0xAE-0xFF (® 到 ÿ)
        ch = 0xAE;
        while (ch <= 0xFF) {
            bs[ch] = true;
            cs[ch] = ch;
            ch += 1;
        }

        // 剩余字节映射到 0x100+ 范围
        n = 0;
        ch = 0;
        while (ch < 256) {
            if (!bs[ch]) {
                cs[ch] = 256 + n;
                n += 1;
            }
            ch += 1;
        }

        // 将 codepoint 转换为 UTF-8 字符串
        var buf: [4]u8 = undefined;
        for (&self.bytes_to_unicode, 0..) |*mapped, byte| {
            const cp = cs[byte];
            const len = std.unicode.utf8Encode(@intCast(cp), &buf) catch unreachable;
            mapped.* = try self.allocator.dupe(u8, buf[0..len]);
            try self.unicode_to_byte.put(mapped.*, @intCast(byte));
        }
    }

    // ========================================================================
    // 资源管理
    // ========================================================================

    pub fn deinit(self: *Tokenizer) void {
        for (self.vocab.items) |e| {
            if (e == .normal) self.allocator.free(e.normal);
        }
        self.vocab.deinit(self.allocator);
        self.token_types.deinit(self.allocator);
        self.merges.deinit();
        self.trie_root.deinit(self.allocator);
        for (&self.bytes_to_unicode) |mapped| {
            self.allocator.free(mapped);
        }
        self.unicode_to_byte.deinit();
    }

    // ========================================================================
    // 编码
    // ========================================================================

    pub fn encode(self: *Tokenizer, text: []const u8, add_special: bool) !std.ArrayListUnmanaged(u32) {
        const add_bos = add_special and self.config.add_bos;
        const add_eos = add_special and self.config.add_eos;

        const enc_config = encode_mod.EncodeConfig{
            .allocator = self.allocator,
            .special = self.special,
            .pre_type = self.config.pre_type,
            .model = self.config.model,
            .vocab = self.vocab,
            .merges = self.merges,
            .trie_root = &self.trie_root,
            .tokenToStringFn = tokenToStringWrapper,
            .byteToTokenIdFn = byteToTokenIdWrapper,
            .bytesToUnicodeFn = bytesToUnicodeWrapper,
            .unicodeToByte = &self.unicode_to_byte,
            .ctx = @ptrCast(self),
        };

        return encode_mod.encode(text, add_bos, add_eos, self.config.add_space_prefix, self.config.ignore_merges, &enc_config);
    }

    // ========================================================================
    // 解码
    // ========================================================================

    pub fn decode(self: *const Tokenizer, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
        const dec_config = decode_mod.DecodeConfig{
            .model = self.config.model,
            .special = self.special,
            .token_types = self.token_types,
            .vocab = self.vocab,
            .clean_spaces = self.config.clean_spaces,
            .escape_whitespaces = self.config.escape_whitespaces,
        };

        return decode_mod.decode(token_ids, &dec_config, allocator);
    }

    // ========================================================================
    // 辅助方法
    // ========================================================================

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.items.len;
    }

    pub fn isSpecialToken(self: *const Tokenizer, token_id: u32) bool {
        if (token_id == self.special.bos or token_id == self.special.eos or
            token_id == self.special.pad or token_id == self.special.unk or
            token_id == self.special.sep or token_id == self.special.cls or
            token_id == self.special.mask) return true;
        if (token_id < self.token_types.items.len and self.token_types.items[token_id] == .control) return true;
        return false;
    }

    pub fn byteToTokenId(self: *const Tokenizer, byte: u8) u32 {
        if (self.byte_to_token_id[byte]) |tid| return tid;
        return self.special.unk;
    }

    /// 将 token ID 转换为字符串表示（用于 BPE 合并）
    fn tokenToString(self: *Tokenizer, token_id: u32) ?[]const u8 {
        if (token_id >= self.vocab.items.len) return null;
        switch (self.vocab.items[token_id]) {
            .normal => |s| return s,
            .byte => |bv| {
                self.byte_token_buf[0] = bv;
                return &self.byte_token_buf;
            },
        }
    }

    /// 将字节转换为 GPT-2 编码后的字符串
    fn bytesToUnicode(self: *const Tokenizer, byte: u8) []const u8 {
        return self.bytes_to_unicode[byte];
    }
};

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
    return self.bytesToUnicode(byte);
}

// ============================================================================
// 内部辅助函数
// ============================================================================

fn countNormalEntries(vocab: []const VocabEntry) usize {
    var count: usize = 0;
    for (vocab) |e| {
        if (e == .normal) count += 1;
    }
    return count;
}

fn countByteEntries(vocab: []const VocabEntry) usize {
    var count: usize = 0;
    for (vocab) |e| {
        if (e == .byte) count += 1;
    }
    return count;
}
