//! 分词器模块
//!
//! 提供 BPE 分词器的完整实现，包括编码和解码。
//! 支持多种分词器模型类型（llama, gpt2, tiktoken, replit）。
//!
//! 参考 llama.cpp 的 llama-vocab.cpp 实现。

const std = @import("std");
const gguf = @import("gguf");

pub const types = @import("types.zig");
pub const trie = @import("trie.zig");
pub const encode_mod = @import("encode.zig");
pub const decode_mod = @import("decode.zig");
pub const bpe = @import("bpe.zig");
pub const utils = @import("utils.zig");

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
// 特殊 Token 缓存条目
// ============================================================================

/// 缓存的特殊 token，用于 parse_special 预分词匹配
/// 与 llama.cpp 的 cache_special_tokens 对应
pub const CacheSpecialToken = struct {
    id: u32,
    text: []const u8, // 指向 vocab 中的 owned 字符串（生命周期与 vocab 一致）
    attr: types.TokenAttr,
};

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

    /// 缓存的特殊 token 列表（按 text 长度降序排列）
    /// 包含类型为 .control、.user_defined、.unknown 的 token
    /// 用于 parse_special 模式下的预分词匹配
    cache_special_tokens: std.ArrayListUnmanaged(CacheSpecialToken),

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
            .cache_special_tokens = .empty,
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
            try tok.eog_ids.ensureTotalCapacity(allocator, 32);

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
                "<|im_start|>",
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

            if (tok.special.eos != 0 and !tok.eog_ids.contains(tok.special.eos)) {
                tok.eog_ids.put(allocator, tok.special.eos, {}) catch {};
            }
            if (tok.special.eot != 0 and !tok.eog_ids.contains(tok.special.eot)) {
                tok.eog_ids.put(allocator, tok.special.eot, {}) catch {};
            }
            if (tok.special.eom != 0 and !tok.eog_ids.contains(tok.special.eom)) {
                tok.eog_ids.put(allocator, tok.special.eom, {}) catch {};
            }

            log.info("Tokenizer: {d} EOG tokens", .{tok.eog_ids.count()});
        }

        // 构建特殊 token 缓存（用于 parse_special 模式）
        // 与 llama.cpp 的 cache_special_tokens 构建逻辑一致
        {
            var special_count: usize = 0;
            for (tok.vocab.items, 0..) |entry, id| {
                if (id < tok.token_types.items.len) {
                    const tt = tok.token_types.items[id];
                    if (tt == .control or tt == .user_defined or tt == .unknown) {
                        special_count += 1;
                    }
                } else if (entry == .normal) {
                    if (entry.normal.len > 2 and entry.normal[0] == '<' and entry.normal[entry.normal.len - 1] == '>') {
                        special_count += 1;
                    }
                }
            }

            if (special_count > 0) {
                try tok.cache_special_tokens.ensureTotalCapacity(allocator, special_count);
                for (tok.vocab.items, 0..) |entry, id| {
                    if (id < tok.token_types.items.len) {
                        const tt = tok.token_types.items[id];
                        if (tt == .control or tt == .user_defined or tt == .unknown) {
                            tok.cache_special_tokens.appendAssumeCapacity(.{
                                .id = @intCast(id),
                                .text = if (entry == .normal) entry.normal else "",
                                .attr = types.tokenTypeToAttr(tt),
                            });
                        }
                    } else if (entry == .normal) {
                        if (entry.normal.len > 2 and entry.normal[0] == '<' and entry.normal[entry.normal.len - 1] == '>') {
                            tok.cache_special_tokens.appendAssumeCapacity(.{
                                .id = @intCast(id),
                                .text = entry.normal,
                                .attr = types.TokenAttr{ .control = true },
                            });
                        }
                    }
                }

                // 按 text 长度降序排列（最长优先匹配）
                std.mem.sort(CacheSpecialToken, tok.cache_special_tokens.items, {}, struct {
                    fn lessThan(_: void, a: CacheSpecialToken, b: CacheSpecialToken) bool {
                        return a.text.len > b.text.len;
                    }
                }.lessThan);

                log.info("Tokenizer: {d} special tokens cached for parse_special", .{tok.cache_special_tokens.items.len});
            }
        }

        return tok;
    }

    /// 初始化 GPT-2 字节编码映射
    fn initBytesToUnicode(self: *Tokenizer) !void {
        var bs: [256]bool = [_]bool{false} ** 256;
        var cs: [256]u32 = undefined;
        var n: u32 = 0;

        for (0x21..0x7F) |b| {
            bs[b] = true;
            cs[n] = @intCast(b);
            n += 1;
        }
        for (0xA1..0xAD) |b| {
            bs[b] = true;
            cs[n] = @intCast(b);
            n += 1;
        }
        for (0xAE..0x100) |b| {
            bs[b] = true;
            cs[n] = @intCast(b);
            n += 1;
        }
        for (0..256) |b| {
            if (!bs[b]) {
                cs[n] = @intCast(256 + b);
                n += 1;
            }
        }

        for (0..256) |b| {
            var buf: [4]u8 = undefined;
            const cp = cs[b];
            const len = std.unicode.utf8Encode(@intCast(cp), &buf) catch unreachable;
            self.bytes_to_unicode[b] = try self.allocator.dupe(u8, buf[0..len]);
        }

        for (0..256) |b| {
            try self.unicode_to_byte.put(self.bytes_to_unicode[b], @intCast(b));
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

        var key_iter = self.merges.keyIterator();
        while (key_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.merges.deinit();
        self.eog_ids.deinit(self.allocator);
        self.cache_special_tokens.deinit(self.allocator);

        self.trie_root.deinit(self.allocator);
        for (&self.bytes_to_unicode) |mapped| {
            self.allocator.free(mapped);
        }
        self.unicode_to_byte.deinit();
    }

    // ========================================================================
    // 编码
    // ========================================================================

    /// 将文本编码为 token ID 列表
    /// - add_special: 是否添加 BOS/EOS 等特殊 token
    /// - parse_special: 是否解析文本中的特殊 token（如 <|turn|>、<|audio|>）
    pub fn encode(self: *Tokenizer, text: []const u8, add_special: bool, parse_special: bool) !std.ArrayListUnmanaged(u32) {
        const add_bos = add_special and self.config.add_bos;
        const add_eos = add_special and self.config.add_eos;

        const cache_special: ?[]const CacheSpecialToken = if (parse_special and self.cache_special_tokens.items.len > 0)
            self.cache_special_tokens.items
        else
            null;

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
            .cache_special_tokens = cache_special,
        };

        return encode_mod.encode(text, add_bos, add_eos, self.config.add_space_prefix, self.config.ignore_merges, parse_special, &enc_config);
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
    pub fn isEog(self: *const Tokenizer, token_id: u32) bool {
        return self.eog_ids.contains(token_id);
    }

    const eog_token_names = [_][]const u8{
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|end|>",
        "<end_of_turn>",
        "</s>",
        "<｜end▁of▁sentence｜>",
    };

    pub fn isEogText(self: *const Tokenizer, text: []const u8) bool {
        _ = self;
        for (eog_token_names) |name| {
            if (std.mem.indexOf(u8, text, name) != null) return true;
        }
        return false;
    }

    const skip_token_names = [_][]const u8{
        "<|channel|>",
        "<|channel>",
        "<channel|>",
        "<start_of_turn>",
        "<end_of_turn>",
    };

    pub fn isSkipToken(self: *const Tokenizer, token_id: u32) bool {
        if (token_id >= self.vocab.items.len) return false;
        const entry = self.vocab.items[token_id];
        if (entry != .normal) return false;
        for (skip_token_names) |name| {
            if (std.mem.eql(u8, entry.normal, name)) return true;
        }
        return false;
    }

    pub fn byteToTokenId(self: *const Tokenizer, byte: u8) u32 {
        if (self.byte_to_token_id[byte]) |tid| return tid;
        return self.special.unk;
    }

    /// 解码单个 token 到缓冲区
    pub fn decodeSingle(self: *const Tokenizer, token_id: u32, buf: []u8) !usize {
        if (token_id >= self.vocab.items.len) return 0;
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
                } else if (self.config.model == .gpt2) {
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

    fn bytesToUnicode(self: *const Tokenizer, byte: u8) []const u8 {
        return self.bytes_to_unicode[byte];
    }

    /// 通过 token 名称（字符串）查找对应的 token ID
    pub fn textToToken(self: *const Tokenizer, text: []const u8) ?u32 {
        const match = trie.longestMatch(&self.trie_root, text, 0);
        if (match) |m| {
            if (m.len == text.len) return m.token_id;
        }
        for (self.vocab.items, 0..) |entry, i| {
            if (entry == .normal and std.mem.eql(u8, entry.normal, text)) {
                return @intCast(i);
            }
        }
        return null;
    }

    fn findBpeRank(self: *const Tokenizer, left: []const u8, right: []const u8) ?u32 {
        const key = std.fmt.allocPrint(self.allocator, "{s} {s}", .{ left, right }) catch return null;
        defer self.allocator.free(key);
        return self.merges.get(key);
    }

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
