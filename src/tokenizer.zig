//! 分词器模块
//!
//! 提供 BPE 分词器的完整实现，包括编码和解码。
//! 支持多种分词器模型类型（llama, gpt2, tiktoken, replit）。
//!
//! 参考 llama.cpp 的 llama-vocab.cpp 实现。

const std = @import("std");
const gguf = @import("gguf");

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

    /// token score 表（用于 SPM 编码）
    token_scores: std.ArrayListUnmanaged(f32),

    /// EOG (End-of-Generation) token ID 集合
    /// 与 llama.cpp 的 special_eog_ids 对应
    eog_ids: std.AutoArrayHashMapUnmanaged(u32, void) = .{},

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
            .token_scores = .empty,
            .eog_ids = .empty,
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

        // 读取 token scores（用于 SPM 编码）
        if (gguf_file.metadata.get("tokenizer.ggml.scores")) |scores_val| {
            for (scores_val.array_val) |sv| {
                const score = sv.asF32() orelse 0.0;
                try tok.token_scores.append(allocator, score);
            }
        }

        // 读取词表
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
                // 必须复制字符串，因为 GGUF 文件在 Tokenizer.init 返回后会被释放
                const owned_key = try allocator.dupe(u8, merge_str);
                tok.merges.putAssumeCapacity(owned_key, @intCast(rank));
            }
            log.info("Tokenizer: {d} BPE merge rules", .{val.array_val.len});
        }

        // 构建 EOG (End-of-Generation) token ID 集合
        // 与 llama.cpp 的 special_eog_ids 逻辑保持一致
        {
            try tok.eog_ids.ensureTotalCapacity(allocator, 16);

            // 从 GGUF 元数据收集
            if (gguf_file.getU32("tokenizer.ggml.eos_token_id")) |eos_id| {
                tok.eog_ids.put(allocator, eos_id, {}) catch {};
            }
            if (gguf_file.getU32("tokenizer.ggml.eot_token_id")) |eot_id| {
                tok.eog_ids.put(allocator, eot_id, {}) catch {};
            }
            if (gguf_file.getU32("tokenizer.ggml.eom_token_id")) |eom_id| {
                tok.eog_ids.put(allocator, eom_id, {}) catch {};
            }
            if (gguf_file.getU32("tokenizer.ggml.fim_pad_token_id")) |fim_pad_id| {
                tok.eog_ids.put(allocator, fim_pad_id, {}) catch {};
            }
            if (gguf_file.getU32("tokenizer.ggml.fim_rep_token_id")) |fim_rep_id| {
                tok.eog_ids.put(allocator, fim_rep_id, {}) catch {};
            }
            if (gguf_file.getU32("tokenizer.ggml.fim_sep_token_id")) |fim_sep_id| {
                tok.eog_ids.put(allocator, fim_sep_id, {}) catch {};
            }

            // 通过名称匹配收集 EOG tokens（与 llama.cpp 保持一致）
            const eog_names = [_][]const u8{
                "<|endoftext|>",
                "<|im_end|>",
                "<|fim_pad|>",
                "<|repo_name|>",
                "<|file_sep|>",
                "<|eot_id|>",
                "<|end|>",
                "<|END|>",
                "<EOS>",
                "<EOT>",
                "<end_of_text>",
                "<|end_of_text|>",
                "<end_of_utterance>",
                "<eos>",
                "<|return|>",
                "<|call|>",
                "<|flush|>",
                "<|calls|>",
                "<end_of_turn>",
                "</s>",
                "<|eom_id|>",
                "[EOT]",
                "[EOS]",
                "<turn|>",
                "<|tool_response>",
                "<｜end▁of▁sentence｜>",
            };
            for (tok.vocab.items, 0..) |entry, id| {
                if (entry == .normal) {
                    for (eog_names) |name| {
                        if (std.mem.eql(u8, entry.normal, name)) {
                            const uid = @as(u32, @intCast(id));
                            if (!tok.eog_ids.contains(uid)) {
                                tok.eog_ids.put(allocator, uid, {}) catch {};
                            }
                            break;
                        }
                    }
                }
            }

            // 确保 special.eos 在集合中
            if (tok.special.eos != 0 and !tok.eog_ids.contains(tok.special.eos)) {
                tok.eog_ids.put(allocator, tok.special.eos, {}) catch {};
            }

            log.info("Tokenizer: {d} EOG tokens", .{tok.eog_ids.count()});
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
        self.token_scores.deinit(self.allocator);

        // 释放 merges 中复制的 key 字符串
        var key_iter = self.merges.keyIterator();
        while (key_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.eog_ids.deinit(self.allocator);

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
            .textToTokenFn = textToTokenWrapper,
            .unicodeToByte = &self.unicode_to_byte,
            .byteToTokenIdFn = byteToTokenIdWrapper,
            .bytesToUnicodeFn = bytesToUnicodeWrapper,
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
            .unicode_to_byte = &self.unicode_to_byte,
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

    /// 判断 token 是否为 EOG (End-of-Generation) token
    /// 与 llama.cpp 的 llama_vocab_is_eog() 对应
    pub fn isEog(self: *const Tokenizer, token_id: u32) bool {
        return self.eog_ids.contains(token_id);
    }
    pub fn byteToTokenId(self: *const Tokenizer, byte: u8) u32 {
        if (self.byte_to_token_id[byte]) |tid| return tid;
        return self.special.unk;
    }

    /// 解码单个 token 到缓冲区（对齐 llama-simple 的 token_to_piece 行为）
    /// 返回写入的字节数
    pub fn decodeSingle(self: *const Tokenizer, token_id: u32, buf: []u8) !usize {
        if (token_id >= self.vocab.items.len) return 0;

        // 跳过特殊 token
        if (self.isSpecialToken(token_id)) return 0;

        var written: usize = 0;

        switch (self.vocab.items[token_id]) {
            .byte => |bv| {
                if (written < buf.len) {
                    buf[written] = bv;
                    written += 1;
                }
            },
            .normal => |ts| {
                if (self.config.model == .tiktoken) {
                    // tiktoken 风格：<0xXX> 格式
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
                } else if (!self.config.escape_whitespaces and self.unicode_to_byte.count() > 0) {
                    // GPT-2 风格：字节编码
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
                        if (self.unicode_to_byte.get(cp_slice)) |b| {
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
                } else {
                    // SPM 风格：替换 ▁ 或 Ġ 为空格
                    var i: usize = 0;
                    while (i < ts.len and written < buf.len) {
                        // U+2581 (▁) UTF-8: E2 96 81
                        if (i + 2 < ts.len and ts[i] == 0xE2 and ts[i+1] == 0x96 and ts[i+2] == 0x81) {
                            buf[written] = ' ';
                            written += 1;
                            i += 3;
                            continue;
                        }
                        // U+0120 (Ġ) UTF-8: C4 A0
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
                }
            },
        }

        return written;
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

    /// 将 token 字符串转换为 token ID
    fn textToToken(self: *const Tokenizer, text: []const u8) ?u32 {
        // 先在 Trie 中查找
        const match = trie.longestMatch(&self.trie_root, text, 0);
        if (match) |m| {
            if (m.len == text.len) return m.token_id;
        }
        // 再在词表中线性查找（作为回退）
        for (self.vocab.items, 0..) |entry, i| {
            if (entry == .normal and std.mem.eql(u8, entry.normal, text)) {
                return @intCast(i);
            }
        }
        return null;
    }

    /// 查找两个 token 的 BPE 合并 rank
    fn findBpeRank(self: *const Tokenizer, left: []const u8, right: []const u8) ?u32 {
        // merges 中的 key 格式为 "left right"
        // 使用动态分配避免缓冲区溢出
        const key = std.fmt.allocPrint(self.allocator, "{s} {s}", .{ left, right }) catch return null;
        defer self.allocator.free(key);
        return self.merges.get(key);
    }

    /// 获取 token 的 score
    fn tokenScore(self: *const Tokenizer, token_id: u32) f32 {
        if (token_id < self.token_scores.items.len) {
            return self.token_scores.items[token_id];
        }
        return 0.0;
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

fn textToTokenWrapper(text: []const u8, ctx: ?*anyopaque) ?u32 {
    const self: *const Tokenizer = @ptrCast(@alignCast(ctx.?));
    return self.textToToken(text);
}

fn findBpeRankWrapper(left: []const u8, right: []const u8, ctx: ?*anyopaque) ?u32 {
    const self: *const Tokenizer = @ptrCast(@alignCast(ctx.?));
    return self.findBpeRank(left, right);
}

fn tokenScoreWrapper(token_id: u32, ctx: ?*anyopaque) f32 {
    const self: *const Tokenizer = @ptrCast(@alignCast(ctx.?));
    return self.tokenScore(token_id);
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
