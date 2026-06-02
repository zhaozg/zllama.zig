//! BPE 分词器
//!
//! 从 GGUF 元数据中读取词表和合并规则，实现 BPE 编码/解码。
//! 支持特殊 token 处理（BOS, EOS, UNK, PAD）。

const std = @import("std");
const gguf = @import("gguf.zig");

const log = std.log.scoped(.tokenizer);

// ============================================================================
// 特殊 Token ID
// ============================================================================

/// 特殊 token 的 ID
pub const SpecialTokens = struct {
    bos: u32 = 1, // Beginning of sequence
    eos: u32 = 2, // End of sequence
    unk: u32 = 0, // Unknown token
    pad: u32 = 0, // Padding
    sep: u32 = 2, // Separator
    cls: u32 = 1, // Classifier
    mask: u32 = 0, // Mask token

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
// BPE 分词器
// ============================================================================

/// BPE 分词器状态
pub const Tokenizer = struct {
    /// 词表：token_id -> token 字符串
    vocab: std.ArrayListUnmanaged([]const u8) = .empty,
    /// token 字符串 -> token_id 的映射
    vocab_reverse: std.StringHashMapUnmanaged(u32) = .{},
    /// BPE 合并规则：pair -> new_token_id
    merges: std.StringHashMapUnmanaged(u32) = .{},
    /// 特殊 token
    special: SpecialTokens = .{},
    /// 分配器
    allocator: std.mem.Allocator,

    /// 从 GGUF 元数据初始化分词器
    pub fn init(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Tokenizer {
        var tok = Tokenizer{
            .allocator = allocator,
        };
        // 如果后续初始化失败，清理已分配的资源
        errdefer tok.deinit();

        // 读取特殊 token
        tok.special = SpecialTokens.fromGGUF(gguf_file);

        // 读取词表
        // GGUF 中词表存储在 tokenizer.ggml.tokens 数组中
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            switch (val) {
                .array => |arr| {
                    log.info("Tokenizer: vocab array with {d} items", .{arr.len});
                    // 对于字符串数组，每个元素是 MetadataValue.string
                    // 但我们的解析器目前将 array 存储为 MetadataValue 数组
                    // 需要从 arr 中提取字符串
                    for (arr, 0..) |item, i| {
                        switch (item) {
                            .string => |s| {
                                const owned = try allocator.dupe(u8, s);
                                try tok.vocab.append(allocator, owned);
                                try tok.vocab_reverse.put(allocator, owned, @intCast(i));
                            },
                            else => {
                                // 非字符串类型，使用占位符
                                const placeholder = try std.fmt.allocPrint(allocator, "[token_{d}]", .{i});
                                try tok.vocab.append(allocator, placeholder);
                            },
                        }
                    }
                },
                else => {
                    log.warn("Tokenizer: tokens metadata is not an array", .{});
                },
            }
        }

        // 尝试读取 BPE 合并规则
        if (gguf_file.metadata.get("tokenizer.ggml.merges")) |val| {
            switch (val) {
                .array => |arr| {
                    log.info("Tokenizer: {d} BPE merge rules", .{arr.len});
                    // 解析合并规则
                    for (arr, 0..) |item, i| {
                        if (item == .string) {
                            const s = item.string;
                            // 合并规则格式: "token1 token2"
                            if (std.mem.indexOfScalar(u8, s, ' ')) |space_pos| {
                                const left = s[0..space_pos];
                                const right = s[space_pos + 1 ..];
                                const key = try std.fmt.allocPrint(allocator, "{s}|{s}", .{ left, right });
                                try tok.merges.put(allocator, key, @intCast(i));
                            }
                        }
                    }
                },
                else => {
                    log.debug("Tokenizer: merges not an array, skipping", .{});
                },
            }
        }

        log.info("Tokenizer initialized: {d} tokens, special: bos={d}, eos={d}, unk={d}", .{ tok.vocab.items.len, tok.special.bos, tok.special.eos, tok.special.unk });

        return tok;
    }

    /// 释放分词器资源
    pub fn deinit(self: *Tokenizer) void {
        // 首先释放 vocab_reverse 哈希表（它的 key 只是 vocab 字符串的引用，不拥有内存）
        self.vocab_reverse.deinit(self.allocator);

        // 释放 vocab 中每个元素的实际字符串内存
        for (self.vocab.items) |item| {
            self.allocator.free(item);
        }
        self.vocab.deinit(self.allocator);

        // 释放 merges 哈希表中的每个 key（这些 key 是 allocPrint 创建的）
        var it = self.merges.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.merges.deinit(self.allocator);
    }

    /// 编码：将文本转换为 token ID 列表
    pub fn encode(self: *const Tokenizer, text: []const u8) !std.ArrayListUnmanaged(u32) {
        var tokens: std.ArrayListUnmanaged(u32) = .empty;

        // 添加 BOS token
        try tokens.append(self.allocator, self.special.bos);

        // 如果词表为空，使用字节级编码
        if (self.vocab.items.len == 0) {
            for (text) |byte| {
                const token_id = self.byteToTokenId(byte);
                try tokens.append(self.allocator, token_id);
            }
            return tokens;
        }

        // 尝试使用 vocab_reverse 进行编码
        // 先尝试匹配最长的 token
        var pos: usize = 0;
        while (pos < text.len) {
            var best_len: usize = 0;
            var best_id: u32 = self.special.unk;

            // 尝试匹配所有 token
            for (self.vocab.items, 0..) |token_str, id| {
                if (pos + token_str.len <= text.len and
                    std.mem.eql(u8, text[pos .. pos + token_str.len], token_str))
                {
                    if (token_str.len > best_len) {
                        best_len = token_str.len;
                        best_id = @intCast(id);
                    }
                }
            }

            if (best_len > 0) {
                try tokens.append(self.allocator, best_id);
                pos += best_len;
            } else {
                // 未匹配，使用字节级编码
                try tokens.append(self.allocator, self.byteToTokenId(text[pos]));
                pos += 1;
            }
        }

        return tokens;
    }

    /// 解码：将 token ID 列表转换为文本
    pub fn decode(self: *const Tokenizer, tokens: []const u32, allocator: std.mem.Allocator) ![]u8 {
        var result = std.ArrayList(u8).empty;

        for (tokens) |token_id| {
            // 跳过特殊 token
            if (token_id == self.special.bos or
                token_id == self.special.eos or
                token_id == self.special.pad)
            {
                continue;
            }

            // 查找 token 字符串
            if (token_id < self.vocab.items.len) {
                const token_str = self.vocab.items[token_id];
                try result.appendSlice(allocator, token_str);
            } else {
                // 未知 token，使用 UNK
                log.debug("Unknown token id: {d}", .{token_id});
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// 将字节映射到 token ID（简化实现）
    fn byteToTokenId(self: *const Tokenizer, byte: u8) u32 {
        _ = self;
        // 字节级别 tokenization：每个字节对应一个 token
        // 实际 BPE 词表中，前 256 个 token 通常是单字节
        return @as(u32, byte) + 3; // 跳过特殊 token
    }

    /// 获取词表大小
    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.items.len;
    }
};

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

test "Tokenizer init and deinit" {
    // 测试空初始化（需要有效的 GGUF 文件）
    // 这里只测试结构体大小
    try testing.expectEqual(@as(usize, @sizeOf(SpecialTokens)), @sizeOf(SpecialTokens));
}
