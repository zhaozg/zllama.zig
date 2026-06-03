//! BPE 分词器
//!
//! 从 GGUF 元数据中读取词表和合并规则，实现 BPE 编码/解码。
//! 支持特殊 token 处理（BOS, EOS, UNK, PAD）。
//!
//! 解码流程（参考 llama.cpp）：
//! 1. 从 GGUF 读取 tokenizer.ggml.tokens（词表字符串）和 tokenizer.ggml.token_type（类型数组）
//! 2. token_type == 6 (LLAMA_TOKEN_TYPE_BYTE) 表示字节 token
//! 3. 加载时，字节 token 提取实际字节值，普通 token 存储 UTF-8 字符串
//! 4. 解码时，字节 token 输出单个字节，普通 token 直接输出文本
//! 5. 所有 token 收集后一次性解码，确保 UTF-8 字节序列正确组合
//!
//! 字节 token 格式说明（取决于 tokenizer.ggml.model）：
//! - "llama" / "gpt2" (BPE): 原始单字节存储，token_bytes.len == 1
//! - "tiktoken" (Qwen): 表示为 "<0xE4>" 可打印形式
//! - "replit": 可能使用 b'<0xE4>' 格式
//!
//! 编码流程（BPE）：
//! 1. 将输入文本按 UTF-8 字节序列分解为初始 token 列表（每个字节一个 token）
//! 2. 反复查找并合并最优先的相邻 token 对（依据 merges 表）
//! 3. 合并直到无法继续
//! 4. 添加 BOS token（可选）
//!
//! 编码优化：使用 Trie（前缀树）实现贪婪最长匹配，替代 O(n*m) 线性扫描。

const std = @import("std");
const gguf = @import("gguf.zig");

const log = std.log.scoped(.tokenizer);

// ============================================================================
// Token 类型常量（与 llama.cpp 保持一致）
// ============================================================================

/// LLAMA_TOKEN_TYPE 枚举值
pub const TokenType = enum(u32) {
    undefined = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
    _,
};

// ============================================================================
// 分词器模型类型
// ============================================================================

/// 分词器模型类型，决定字节 token 的表示格式
pub const TokenizerModel = enum {
    llama,
    gpt2,
    tiktoken,
    replit,
    unknown,

    pub fn fromString(s: []const u8) TokenizerModel {
        if (std.ascii.eqlIgnoreCase(s, "llama")) return .llama;
        if (std.ascii.eqlIgnoreCase(s, "gpt2")) return .gpt2;
        if (std.ascii.eqlIgnoreCase(s, "tiktoken")) return .tiktoken;
        if (std.ascii.eqlIgnoreCase(s, "replit")) return .replit;
        return .unknown;
    }
};

// ============================================================================
// 特殊 Token ID
// ============================================================================

pub const SpecialTokens = struct {
    bos: u32 = 1,
    eos: u32 = 2,
    unk: u32 = 0,
    pad: u32 = 0,
    sep: u32 = 2,
    cls: u32 = 1,
    mask: u32 = 0,

    pub fn fromGGUF(gguf_file: *const gguf.GGUFFile) SpecialTokens {
        return SpecialTokens{
            .bos = gguf_file.getU32("tokenizer.ggml.bos_token_id") orelse 1,
            .eos = gguf_file.getU32("tokenizer.ggml.eos_token_id") orelse 2,
            .unk = gguf_file.getU32("tokenizer.ggml.unknown_token_id") orelse 0,
            .pad = gguf_file.getU32("tokenizer.ggml.padding_token_id") orelse 0,
            .sep = gguf_file.getU32("tokenizer.ggml.sep_token_id") orelse 2,
            .cls = gguf_file.getU32("tokenizer.ggml.cls_token_id") orelse 1,
            .mask = gguf_file.getU32("tokenizer.ggml.mask_token_id") orelse 0,
        };
    }
};

// ============================================================================
// 词表条目类型
// ============================================================================

pub const VocabEntry = union(enum) {
    normal: []const u8,
    byte: u8,
};

// ============================================================================
// Trie 节点
// ============================================================================

const TrieNode = struct {
    children: std.AutoHashMapUnmanaged(u8, *TrieNode) = .{},
    token_id: ?u32 = null,

    fn deinit(self: *TrieNode, allocator: std.mem.Allocator) void {
        var it = self.children.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit(allocator);
            allocator.destroy(entry.value_ptr.*);
        }
        self.children.deinit(allocator);
    }
};

const MatchResult = struct {
    token_id: u32,
    len: usize,
};

// ============================================================================
// BPE 分词器
// ============================================================================

pub const Tokenizer = struct {
    vocab: std.ArrayListUnmanaged(VocabEntry) = .empty,
    /// vocab_reverse 的 key 与 vocab 中 normal 字符串共享所有权，
    /// 释放时只通过 vocab 释放，vocab_reverse 不负责释放 key
    vocab_reverse: std.StringHashMapUnmanaged(u32) = .{},
    /// merges 的 key 通过 allocator.dupe 分配，deinit 中负责释放
    merges: std.StringHashMapUnmanaged(u32) = .{},
    special: SpecialTokens = .{},
    model: TokenizerModel = .unknown,
    byte_decoder: std.AutoHashMapUnmanaged(u32, u8) = .{},
    trie_root: TrieNode = .{},
    byte_to_token_id: [256]?u32 = .{null} ** 256,
    token_types: std.ArrayListUnmanaged(TokenType) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Tokenizer {
        var tok = Tokenizer{ .allocator = allocator };
        errdefer tok.deinit();

        tok.special = SpecialTokens.fromGGUF(gguf_file);

        const tokenizer_model_raw = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";
        const model_name = gguf_file.getString("general.name") orelse "";
        const model_arch = gguf_file.getString("general.architecture") orelse "";

        const is_qwen = if (model_name.len > 0)
            std.ascii.indexOfIgnoreCase(model_name, "qwen") != null
        else if (model_arch.len > 0)
            std.ascii.indexOfIgnoreCase(model_arch, "qwen") != null
        else
            false;

        if (is_qwen) {
            tok.model = .tiktoken;
            log.info("Tokenizer model: detected Qwen model '{s}', forcing tiktoken mode", .{model_name});
        } else {
            tok.model = TokenizerModel.fromString(tokenizer_model_raw);
            log.info("Tokenizer model: '{s}' -> {s}", .{ tokenizer_model_raw, @tagName(tok.model) });
        }

        // 构建 byte_decoder
        {
            const bytes_to_unicode = try generateBytesToUnicode(allocator);
            defer allocator.free(bytes_to_unicode);
            for (bytes_to_unicode, 0..) |codepoint, byte| {
                try tok.byte_decoder.put(allocator, codepoint, @intCast(byte));
            }
        }

        // 读取 token_type
        if (gguf_file.metadata.get("tokenizer.ggml.token_type")) |val| {
            if (val.value_type == .array) {
                for (val.array_val) |item| {
                    const v: u32 = switch (item.value_type) {
                        .int32 => @as(u32, @intCast(item.int32_val)),
                        .uint32 => item.uint32_val,
                        else => @intFromEnum(TokenType.normal),
                    };
                    try tok.token_types.append(allocator, @as(TokenType, @enumFromInt(v)));
                }
            }
        }

        // 读取词表
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            if (val.value_type == .array) {
                for (val.array_val, 0..) |item, i| {
                    const tt = if (i < tok.token_types.items.len) tok.token_types.items[i] else TokenType.normal;
                    if (item.value_type == .string) {
                        const s = item.string_val;
                        if (tt == .byte) {
                            const byte_val = extractByteFromToken(s, tok.model) catch blk: {
                                if (s.len == 1) break :blk s[0];
                                // 无法提取字节，作为 normal token 处理
                                const owned = try allocator.dupe(u8, s);
                                errdefer allocator.free(owned);
                                try tok.vocab.append(allocator, VocabEntry{ .normal = owned });
                                try tok.vocab_reverse.put(allocator, owned, @intCast(i));
                                try tok.addToTrie(owned, @intCast(i));
                                continue;
                            };
                            try tok.vocab.append(allocator, VocabEntry{ .byte = byte_val });
                            tok.byte_to_token_id[byte_val] = @intCast(i);
                        } else {
                            const owned = try allocator.dupe(u8, s);
                            errdefer allocator.free(owned);
                            try tok.vocab.append(allocator, VocabEntry{ .normal = owned });
                            try tok.vocab_reverse.put(allocator, owned, @intCast(i));
                            try tok.addToTrie(owned, @intCast(i));
                        }
                    } else {
                        const placeholder = try std.fmt.allocPrint(allocator, "[token_{d}]", .{i});
                        errdefer allocator.free(placeholder);
                        try tok.vocab.append(allocator, VocabEntry{ .normal = placeholder });
                    }
                }
            }
        }

        // 读取 BPE 合并规则
        if (gguf_file.metadata.get("tokenizer.ggml.merges")) |val| {
            if (val.value_type == .array) {
                log.info("Tokenizer: {d} BPE merge rules", .{val.array_val.len});
                for (val.array_val, 0..) |item, i| {
                    if (item.value_type == .string) {
                        const s = item.string_val;
                        const key = try allocator.dupe(u8, s);
                        errdefer allocator.free(key);
                        try tok.merges.put(allocator, key, @intCast(i));
                    }
                }
            }
        }

        // 打印统计信息
        {
            var nc: u32 = 0;
            var bc: u32 = 0;
            for (tok.vocab.items) |e| {
                switch (e) {
                    .normal => nc += 1,
                    .byte => bc += 1,
                }
            }
            log.info("Tokenizer: {d} entries (normal={d}, byte={d})", .{ tok.vocab.items.len, nc, bc });
        }

        return tok;
    }

    fn addToTrie(self: *Tokenizer, token_str: []const u8, token_id: u32) !void {
        var node = &self.trie_root;
        for (token_str) |byte| {
            const gop = try node.children.getOrPut(self.allocator, byte);
            if (!gop.found_existing) {
                gop.value_ptr.* = try self.allocator.create(TrieNode);
                gop.value_ptr.*.* = TrieNode{};
            }
            node = gop.value_ptr.*;
        }
        node.token_id = token_id;
    }

    pub fn deinit(self: *Tokenizer) void {
        // 注意：vocab_reverse 的 key 与 vocab 中 normal 字符串共享所有权，
        // 所以这里只释放 HashMap 本身，不释放 key（由 vocab 的 normal 字符串释放）
        self.vocab_reverse.deinit(self.allocator);
        self.byte_decoder.deinit(self.allocator);
        self.trie_root.deinit(self.allocator);
        self.token_types.deinit(self.allocator);
        // 释放 vocab 中的 normal 字符串（这是所有权的真正持有者）
        for (self.vocab.items) |entry| {
            if (entry == .normal) self.allocator.free(entry.normal);
        }
        self.vocab.deinit(self.allocator);
        // merges 的 key 是通过 allocator.dupe 分配的，需要释放
        {
            var it = self.merges.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
        }
        self.merges.deinit(self.allocator);
    }

    /// 编码：将文本转换为 token ID 列表
    /// 阶段 1：Trie 贪婪最长匹配
    /// 阶段 2：BPE 合并规则
    pub fn encode(self: *const Tokenizer, text: []const u8) !std.ArrayListUnmanaged(u32) {
        var tokens: std.ArrayListUnmanaged(u32) = .empty;
        try tokens.append(self.allocator, self.special.bos);

        if (self.vocab.items.len == 0) {
            for (text) |byte| try tokens.append(self.allocator, self.byteToTokenId(byte));
            return tokens;
        }

        // 阶段 1：Trie 贪婪最长匹配
        var pos: usize = 0;
        while (pos < text.len) {
            const match = self.trieLongestMatch(text, pos);
            if (match) |m| {
                try tokens.append(self.allocator, m.token_id);
                pos += m.len;
            } else {
                try tokens.append(self.allocator, self.byteToTokenId(text[pos]));
                pos += 1;
            }
        }

        // 阶段 2：BPE 合并
        if (self.merges.count() > 0) try self.applyBpeMerges(&tokens);

        return tokens;
    }

    fn trieLongestMatch(self: *const Tokenizer, text: []const u8, pos: usize) ?MatchResult {
        var node = &self.trie_root;
        var best: ?MatchResult = null;
        var cl: usize = 0;
        var i = pos;
        while (i < text.len) {
            if (node.children.get(text[i])) |child| {
                node = child;
                cl += 1;
                if (node.token_id) |tid| best = MatchResult{ .token_id = tid, .len = cl };
                i += 1;
            } else break;
        }
        return best;
    }

    fn applyBpeMerges(self: *const Tokenizer, tokens: *std.ArrayListUnmanaged(u32)) !void {
        if (tokens.items.len < 2) return;
        var changed = true;
        while (changed) {
            changed = false;
            var best_idx: ?usize = null;
            var best_rank: u32 = std.math.maxInt(u32);
            var i: usize = 0;
            while (i + 1 < tokens.items.len) {
                const ls = self.tokenToString(tokens.items[i]) orelse {
                    i += 1;
                    continue;
                };
                const rs = self.tokenToString(tokens.items[i + 1]) orelse {
                    i += 1;
                    continue;
                };
                // 使用空格分隔的格式查找合并规则（与 merges 中的 key 格式一致）
                var kb: [1024]u8 = undefined;
                const key = std.fmt.bufPrint(&kb, "{s} {s}", .{ ls, rs }) catch {
                    i += 1;
                    continue;
                };
                if (self.merges.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
                i += 1;
            }
            if (best_idx) |idx| {
                const ls = self.tokenToString(tokens.items[idx]) orelse {
                    changed = false;
                    continue;
                };
                const rs = self.tokenToString(tokens.items[idx + 1]) orelse {
                    changed = false;
                    continue;
                };
                var mb: [2048]u8 = undefined;
                const merged = std.fmt.bufPrint(&mb, "{s}{s}", .{ ls, rs }) catch {
                    changed = false;
                    continue;
                };
                if (self.vocab_reverse.get(merged)) |new_id| {
                    tokens.items[idx] = new_id;
                    _ = tokens.orderedRemove(idx + 1);
                    changed = true;
                }
            }
        }
    }

    fn tokenToString(self: *const Tokenizer, token_id: u32) ?[]const u8 {
        if (token_id >= self.vocab.items.len) return null;
        switch (self.vocab.items[token_id]) {
            .normal => |s| return s,
            .byte => return null,
        }
    }

    /// 解码：将 token ID 列表转换为 UTF-8 文本
    pub fn decode(self: *const Tokenizer, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
        var result = std.ArrayList(u8).empty;
        for (token_ids) |token_id| {
            if (self.isSpecialToken(token_id)) continue;
            if (token_id < self.vocab.items.len) {
                switch (self.vocab.items[token_id]) {
                    .byte => |bv| try result.append(allocator, bv),
                    .normal => |ts| {
                        if (self.model == .tiktoken) {
                            try self.decodeTiktokenToken(ts, &result, allocator);
                        } else {
                            try result.appendSlice(allocator, ts);
                        }
                    },
                }
            }
        }
        return result.toOwnedSlice(allocator);
    }

    fn isSpecialToken(self: *const Tokenizer, token_id: u32) bool {
        if (token_id == self.special.bos or token_id == self.special.eos or
            token_id == self.special.pad or token_id == self.special.unk or
            token_id == self.special.sep or token_id == self.special.cls or
            token_id == self.special.mask) return true;
        if (token_id < self.token_types.items.len and self.token_types.items[token_id] == .control) return true;
        return false;
    }

    fn decodeTiktokenToken(self: *const Tokenizer, token_str: []const u8, result: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        var rem = token_str;
        while (rem.len > 0) {
            const cl = std.unicode.utf8ByteSequenceLength(rem[0]) catch {
                try result.append(allocator, rem[0]);
                rem = rem[1..];
                continue;
            };
            if (rem.len < cl) {
                try result.append(allocator, rem[0]);
                rem = rem[1..];
                continue;
            }
            const cp = std.unicode.utf8Decode(rem[0..cl]) catch {
                try result.append(allocator, rem[0]);
                rem = rem[1..];
                continue;
            };
            if (self.byte_decoder.get(cp)) |bv| {
                try result.append(allocator, bv);
            } else {
                try result.appendSlice(allocator, rem[0..cl]);
            }
            rem = rem[cl..];
        }
    }

    fn byteToTokenId(self: *const Tokenizer, byte: u8) u32 {
        if (self.byte_to_token_id[byte]) |tid| return tid;
        return @as(u32, byte) + 3;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.items.len;
    }
};

// ============================================================================
// 模块级工具函数
// ============================================================================

pub fn generateBytesToUnicode(allocator: std.mem.Allocator) ![]u32 {
    var list = try std.ArrayList(u32).initCapacity(allocator, 256);
    defer list.deinit(allocator);
    var bs: [256]bool = .{false} ** 256;
    for (33..127) |b| bs[b] = true;
    for (161..173) |b| bs[b] = true;
    for (174..256) |b| bs[b] = true;
    var n: u32 = 0;
    for (0..256) |b| {
        try list.append(allocator, if (bs[b]) @as(u32, @intCast(b)) else blk: {
            n += 1;
            break :blk 256 + n - 1;
        });
    }
    return list.toOwnedSlice(allocator);
}

pub fn extractByteFromToken(token: []const u8, model: TokenizerModel) !u8 {
    const is_hex = token.len == 6 and token[0] == '<' and token[1] == '0' and
        (token[2] == 'x' or token[2] == 'X') and token[5] == '>';
    const is_bhex = token.len == 9 and std.mem.eql(u8, token[0..2], "b'") and
        token[2] == '<' and token[7] == '>' and token[8] == '\'';
    const is_bhex_space = token.len == 10 and std.mem.eql(u8, token[0..2], "b'") and
        token[2] == '<' and token[7] == '>' and token[8] == '\'' and token[9] == ' ';

    switch (model) {
        .llama, .gpt2 => {
            if (token.len == 1) return token[0];
            if (is_hex) return std.fmt.parseInt(u8, token[3..5], 16);
            return error.InvalidByteToken;
        },
        .tiktoken => {
            if (is_hex) return std.fmt.parseInt(u8, token[3..5], 16);
            if (is_bhex) return std.fmt.parseInt(u8, token[4..6], 16);
            if (token.len == 1) return token[0];
            return error.InvalidByteToken;
        },
        .replit => {
            if (is_bhex) return std.fmt.parseInt(u8, token[4..6], 16);
            if (is_hex) return std.fmt.parseInt(u8, token[3..5], 16);
            if (token.len == 1) return token[0];
            return error.InvalidByteToken;
        },
        .unknown => {
            if (token.len == 1) return token[0];
            if (is_hex) return std.fmt.parseInt(u8, token[3..5], 16);
            if (is_bhex) return std.fmt.parseInt(u8, token[4..6], 16);
            if (is_bhex_space) return std.fmt.parseInt(u8, token[4..6], 16);
            return error.InvalidByteToken;
        },
    }
}

pub fn isByteTokenFormat(token: []const u8) bool {
    _ = extractByteFromToken(token, .unknown) catch return false;
    return true;
}

pub fn parseByteToken(token: []const u8) !u8 {
    return extractByteFromToken(token, .unknown);
}

pub fn inferIsByteToken(token: []const u8) bool {
    if (token.len == 1) return true;
    if (token.len == 6 and token[0] == '<' and token[1] == '0' and
        (token[2] == 'x' or token[2] == 'X') and token[5] == '>') return true;
    if (token.len == 9 and std.mem.eql(u8, token[0..2], "b'") and
        token[2] == '<' and token[7] == '>' and token[8] == '\'') return true;
    return false;
}

pub fn hexDump(data: []const u8) void {
    const hex = "0123456789abcdef";
    var buf: [80]u8 = undefined;
    var pos: usize = 0;
    for (data, 0..) |byte, i| {
        if (i > 0 and i % 16 == 0) {
            std.debug.print("{s}\n", .{buf[0..pos]});
            pos = 0;
        }
        if (pos == 0) {
            _ = std.fmt.bufPrint(buf[pos..], "{x:0>8}: ", .{i}) catch {};
            pos += 10;
        }
        buf[pos] = hex[byte >> 4];
        buf[pos + 1] = hex[byte & 0x0F];
        buf[pos + 2] = ' ';
        pos += 3;
    }
    if (pos > 0) std.debug.print("{s}\n", .{buf[0..pos]});
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "SpecialTokens defaults" {
    const st = SpecialTokens{};
    try testing.expectEqual(@as(u32, 1), st.bos);
    try testing.expectEqual(@as(u32, 2), st.eos);
    try testing.expectEqual(@as(u32, 0), st.unk);
}

test "extractByteFromToken with model" {
    try testing.expectEqual(@as(u8, 0x41), try extractByteFromToken("A", .llama));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("\xE4", .llama));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("<0xE4>", .tiktoken));
    try testing.expectEqual(@as(u8, 0x0A), try extractByteFromToken("<0x0A>", .tiktoken));
    try testing.expectEqual(@as(u8, 0xFF), try extractByteFromToken("<0xFF>", .tiktoken));
    try testing.expectEqual(@as(u8, 0x00), try extractByteFromToken("<0x00>", .tiktoken));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("b'<0xE4>'", .tiktoken));
    try testing.expectEqual(@as(u8, 0x41), try extractByteFromToken("A", .unknown));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("<0xE4>", .unknown));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("b'<0xE4>'", .unknown));
    try testing.expectError(error.InvalidByteToken, extractByteFromToken("Hello", .unknown));
}

test "TokenizerModel fromString" {
    try testing.expectEqual(TokenizerModel.llama, TokenizerModel.fromString("llama"));
    try testing.expectEqual(TokenizerModel.gpt2, TokenizerModel.fromString("gpt2"));
    try testing.expectEqual(TokenizerModel.tiktoken, TokenizerModel.fromString("tiktoken"));
    try testing.expectEqual(TokenizerModel.replit, TokenizerModel.fromString("replit"));
    try testing.expectEqual(TokenizerModel.unknown, TokenizerModel.fromString("unknown_model"));
}

test "inferIsByteToken" {
    try testing.expect(inferIsByteToken("A"));
    try testing.expect(inferIsByteToken("<0xE4>"));
    try testing.expect(inferIsByteToken("b'<0xE4>'"));
    try testing.expect(!inferIsByteToken("Hello"));
}

test "decode handles byte tokens correctly" {
    var mv: std.ArrayListUnmanaged(VocabEntry) = .empty;
    defer mv.deinit(testing.allocator);
    try mv.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "Hello") });
    try mv.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, " ") });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0xE4 });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0xB8 });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0x96 });
    try mv.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "World") });
    var tok = Tokenizer{ .allocator = testing.allocator, .vocab = mv };
    defer {
        for (tok.vocab.items) |e| {
            if (e == .normal) testing.allocator.free(e.normal);
        }
        tok.vocab.deinit(testing.allocator);
    }
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const result = try tok.decode(&ids, testing.allocator);
    defer testing.allocator.free(result);
    try testing.expectEqualSlices(u8, "Hello 世界World", result);
}

test "tiktoken byte token decode flow" {
    var mv: std.ArrayListUnmanaged(VocabEntry) = .empty;
    defer mv.deinit(testing.allocator);
    try mv.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "Hello") });
    try mv.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, " ") });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0xE4 });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0xB8 });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0xAD });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0xE5 });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0x9B });
    try mv.append(testing.allocator, VocabEntry{ .byte = 0xBD });
    try mv.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "!") });
    var tok = Tokenizer{ .allocator = testing.allocator, .vocab = mv, .model = .tiktoken };
    defer {
        for (tok.vocab.items) |e| {
            if (e == .normal) testing.allocator.free(e.normal);
        }
        tok.vocab.deinit(testing.allocator);
    }
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = try tok.decode(&ids, testing.allocator);
    defer testing.allocator.free(result);
    try testing.expectEqualSlices(u8, "Hello 中国!", result);
}

// ============================================================================
// 辅助函数：创建模拟 Tokenizer
// ============================================================================

fn createMockTokenizer() !Tokenizer {
    var tok = Tokenizer{ .allocator = testing.allocator };
    const entries = [_][]const u8{ "Hello", " ", "World", "how", "are", "you", "?", "!", ",", "\n", "\t", "Special", "chars", "@", "#", "$", "%", "^", "&", "*", "(", ")", "Mix", "English" };
    for (entries, 0..) |e, i| {
        const owned = try testing.allocator.dupe(u8, e);
        try tok.vocab.append(testing.allocator, VocabEntry{ .normal = owned });
        try tok.vocab_reverse.put(testing.allocator, owned, @intCast(i));
        try tok.addToTrie(owned, @intCast(i));
    }
    for (0..256) |b| {
        const bv: u8 = @intCast(b);
        try tok.vocab.append(testing.allocator, VocabEntry{ .byte = bv });
        tok.byte_to_token_id[bv] = @intCast(entries.len + b);
    }
    return tok;
}

fn destroyMockTokenizer(tok: *Tokenizer) void {
    // 注意：vocab_reverse 的 key 与 vocab 中 normal 字符串共享所有权，
    // 所以这里只释放 HashMap 本身，不释放 key
    tok.vocab_reverse.deinit(testing.allocator);
    tok.trie_root.deinit(testing.allocator);
    tok.token_types.deinit(testing.allocator);
    tok.byte_decoder.deinit(testing.allocator);
    // 释放 vocab 中的 normal 字符串（这是所有权的真正持有者）
    for (tok.vocab.items) |e| {
        if (e == .normal) testing.allocator.free(e.normal);
    }
    tok.vocab.deinit(testing.allocator);
    // 释放 merges 的 key
    {
        var it = tok.merges.iterator();
        while (it.next()) |entry| {
            testing.allocator.free(entry.key_ptr.*);
        }
    }
    tok.merges.deinit(testing.allocator);
}

test "encode-decode roundtrip basic" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    const texts = &[_][]const u8{ "Hello", "World", "Hello World", "Hello, World!", "Special chars: @#$%^&*()" };
    for (texts) |text| {
        var ids = try tok.encode(text);
        defer ids.deinit(testing.allocator);
        const decoded = try tok.decode(ids.items, testing.allocator);
        defer testing.allocator.free(decoded);
        try testing.expectEqualStrings(text, decoded);
    }
}

test "encode-decode roundtrip byte-level" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    for (0..256) |b| {
        const input = [_]u8{@intCast(b)};
        var ids = try tok.encode(&input);
        defer ids.deinit(testing.allocator);
        const decoded = try tok.decode(ids.items, testing.allocator);
        defer testing.allocator.free(decoded);
        try testing.expectEqualSlices(u8, &input, decoded);
    }
}

test "special tokens are filtered in decode" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    tok.special.bos = 0;
    tok.special.eos = 1;
    tok.special.pad = 2;
    const ids = [_]u32{ 0, 3, 1 };
    const decoded = try tok.decode(&ids, testing.allocator);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings("Hello", decoded);
}

test "control tokens are filtered in decode" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    try tok.token_types.append(testing.allocator, .control);
    try tok.token_types.append(testing.allocator, .normal);
    try tok.token_types.append(testing.allocator, .control);
    const ids = [_]u32{ 0, 1, 2 };
    const decoded = try tok.decode(&ids, testing.allocator);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings("Hello", decoded);
}

test "Trie longest match" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    const m1 = tok.trieLongestMatch("Hello World", 0);
    try testing.expect(m1 != null);
    if (m1) |m| {
        try testing.expectEqual(@as(u32, 0), m.token_id);
        try testing.expectEqual(@as(usize, 5), m.len);
    }
    const m2 = tok.trieLongestMatch("Hello World", 5);
    try testing.expect(m2 != null);
    if (m2) |m| {
        try testing.expectEqual(@as(u32, 1), m.token_id);
        try testing.expectEqual(@as(usize, 1), m.len);
    }
    const m3 = tok.trieLongestMatch("Hello World", 6);
    try testing.expect(m3 != null);
    if (m3) |m| {
        try testing.expectEqual(@as(u32, 2), m.token_id);
        try testing.expectEqual(@as(usize, 5), m.len);
    }
}

test "encode with Trie" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    var ids = try tok.encode("Hello World");
    defer ids.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 4), ids.items.len);
    try testing.expectEqual(tok.special.bos, ids.items[0]);
    try testing.expectEqual(@as(u32, 0), ids.items[1]);
    try testing.expectEqual(@as(u32, 1), ids.items[2]);
    try testing.expectEqual(@as(u32, 2), ids.items[3]);
}

test "encode-decode roundtrip with BPE merges" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    // 使用空格分隔的格式
    const mk1 = try testing.allocator.dupe(u8, "Hello ");
    try tok.merges.put(testing.allocator, mk1, 0);
    const mk2 = try testing.allocator.dupe(u8, "Hello World");
    try tok.merges.put(testing.allocator, mk2, 1);
    var ids = try tok.encode("Hello World");
    defer ids.deinit(testing.allocator);
    const decoded = try tok.decode(ids.items, testing.allocator);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings("Hello World", decoded);
}

test "isSpecialToken detection" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    tok.special.bos = 100;
    tok.special.eos = 101;
    tok.special.pad = 102;
    tok.special.unk = 103;
    try testing.expect(tok.isSpecialToken(100));
    try testing.expect(tok.isSpecialToken(101));
    try testing.expect(tok.isSpecialToken(102));
    try testing.expect(tok.isSpecialToken(103));
    try testing.expect(!tok.isSpecialToken(0));
    try testing.expect(!tok.isSpecialToken(1));
}

test "generateBytesToUnicode roundtrip" {
    const mapping = try generateBytesToUnicode(testing.allocator);
    defer testing.allocator.free(mapping);
    try testing.expectEqual(@as(usize, 256), mapping.len);
    for (33..127) |b| try testing.expectEqual(@as(u32, @intCast(b)), mapping[b]);
    var seen = std.AutoHashMap(u32, void).init(testing.allocator);
    defer seen.deinit();
    for (mapping) |cp| {
        try testing.expect(!seen.contains(cp));
        try seen.put(cp, {});
    }
}

test "byte_to_token_id mapping" {
    var tok = try createMockTokenizer();
    defer destroyMockTokenizer(&tok);
    for (0..256) |b| {
        const tid = tok.byteToTokenId(@intCast(b));
        try testing.expect(tid >= 24);
    }
}
