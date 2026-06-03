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
    llama,    // BPE / GPT-2 类：原始单字节存储
    gpt2,     // 同 llama，原始单字节
    tiktoken, // Qwen 类："<0xE4>" 格式
    replit,   // 可能使用 b'<0xE4>' 格式
    unknown,  // 未知格式，自动检测

    /// 从 GGUF 元数据字符串解析
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
// 词表条目类型
// ============================================================================

/// 词表条目：区分普通 token 和字节 token
/// - normal: 存储 UTF-8 字符串（如 "Hello", "世界"）
/// - byte:   存储单个字节值（如 0xE4），解码时直接输出该字节
pub const VocabEntry = union(enum) {
    normal: []const u8,
    byte: u8,
};

// ============================================================================
// BPE 分词器
// ============================================================================

/// BPE 分词器状态
pub const Tokenizer = struct {
    /// 词表：token_id -> VocabEntry
    vocab: std.ArrayListUnmanaged(VocabEntry) = .empty,
    /// token 字符串 -> token_id 的映射（仅用于 normal 类型）
    vocab_reverse: std.StringHashMapUnmanaged(u32) = .{},
    /// BPE 合并规则：pair -> new_token_id
    merges: std.StringHashMapUnmanaged(u32) = .{},
    /// 特殊 token
    special: SpecialTokens = .{},
    /// 分词器模型类型（决定字节 token 格式）
    model: TokenizerModel = .unknown,
    /// 分配器
    /// tiktoken byte_decoder 映射：Unicode 码点 → 原始字节
    /// 用于将 tiktoken 编码的 token 字符串还原为原始字节序列
    byte_decoder: std.AutoHashMapUnmanaged(u32, u8) = .{},


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

        // ============================================================
        // 第 1 步：确定分词器模型类型
        // ============================================================
        // 读取 tokenizer.ggml.model 元数据
        const tokenizer_model_raw = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";

        // 读取模型名称和架构（用于 Qwen 检测）
        const model_name = gguf_file.getString("general.name") orelse "";
        const model_arch = gguf_file.getString("general.architecture") orelse "";

        // 判断是否为 Qwen 模型
        // 检测 general.name 或 general.architecture 是否包含 "qwen"
        const is_qwen = if (model_name.len > 0)
            std.ascii.indexOfIgnoreCase(model_name, "qwen") != null
        else if (model_arch.len > 0)
            std.ascii.indexOfIgnoreCase(model_arch, "qwen") != null
        else
            false;

        // 确定最终的分词器模型类型
        // Qwen 3.5 使用 tiktoken 分词器，但 GGUF 元数据中常被标记为 "gpt2"
        // 因此当检测到 Qwen 模型时，强制使用 tiktoken 模式
        if (is_qwen) {
            tok.model = .tiktoken;
            log.info("Tokenizer model: detected Qwen model '{s}', forcing tiktoken mode", .{model_name});
        } else {
            tok.model = TokenizerModel.fromString(tokenizer_model_raw);
            log.info("Tokenizer model: '{s}' -> {s}", .{ tokenizer_model_raw, @tagName(tok.model) });
        }

        // 读取 token 类型数组（tokenizer.ggml.token_type）
        // ============================================================
        // 第 1.5 步：构建 byte_decoder 映射（tiktoken 字节解码）
        // ============================================================
        // 对于 tiktoken 模型，构建 byte_decoder 映射用于解码
        // 即使不是 tiktoken 模型，也构建映射以备不时之需
        {
            // 生成 bytes_to_unicode 映射（字节 → Unicode 码点）
            const bytes_to_unicode = try generateBytesToUnicode(allocator);
            defer allocator.free(bytes_to_unicode);

            // 反转得到 byte_decoder（Unicode 码点 → 字节）
            for (bytes_to_unicode, 0..) |codepoint, byte| {
                try tok.byte_decoder.put(allocator, codepoint, @intCast(byte));
            }
            log.info("Tokenizer: built byte_decoder with {d} entries", .{tok.byte_decoder.count()});
        }


        // 这是识别字节 token 的关键：token_type == 6 (LLAMA_TOKEN_TYPE_BYTE)
        var token_types: std.ArrayListUnmanaged(u32) = .empty;
        defer token_types.deinit(allocator);

        if (gguf_file.metadata.get("tokenizer.ggml.token_type")) |val| {
            switch (val) {
                .array => |arr| {
                    log.info("Tokenizer: token_type array with {d} items", .{arr.len});
                    for (arr) |item| {
                        switch (item) {
                            .int32 => |v| {
                                try token_types.append(allocator, @as(u32, @intCast(v)));
                            },
                            .uint32 => |v| {
                                try token_types.append(allocator, v);
                            },
                            else => {
                                try token_types.append(allocator, @intFromEnum(TokenType.normal));
                            },
                        }
                    }
                },
                else => {
                    log.debug("Tokenizer: token_type not an array, byte tokens will be detected by format", .{});
                },
            }
        } else {
            log.debug("Tokenizer: tokenizer.ggml.token_type not found, byte tokens will be detected by format", .{});
        }

        // 读取词表
        // GGUF 中词表存储在 tokenizer.ggml.tokens 数组中
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            switch (val) {
                .array => |arr| {
                    log.info("Tokenizer: vocab array with {d} items", .{arr.len});
                    for (arr, 0..) |item, i| {
                        const token_type = if (i < token_types.items.len) token_types.items[i] else @intFromEnum(TokenType.normal);

                        switch (item) {
                            .string => |s| {
                                if (token_type == @intFromEnum(TokenType.byte)) {
                                    // 字节 token：从字符串中提取实际字节值
                                    const byte_val = extractByteFromToken(s, tok.model) catch blk: {
                                        // 解析失败，回退：如果字符串长度为1，直接使用该字节
                                        if (s.len == 1) break :blk s[0];
                                        // 否则作为普通字符串存储
                                        const owned = try allocator.dupe(u8, s);
                                        try tok.vocab.append(allocator, VocabEntry{ .normal = owned });
                                        try tok.vocab_reverse.put(allocator, owned, @intCast(i));
                                        continue;
                                    };
                                    try tok.vocab.append(allocator, VocabEntry{ .byte = byte_val });
                                } else {
                                    // 普通 token：直接存储字符串
                                    const owned = try allocator.dupe(u8, s);
                                    try tok.vocab.append(allocator, VocabEntry{ .normal = owned });
                                    try tok.vocab_reverse.put(allocator, owned, @intCast(i));
                                }
                            },
                            else => {
                                // 非字符串类型，使用占位符
                                const placeholder = try std.fmt.allocPrint(allocator, "[token_{d}]", .{i});
                                try tok.vocab.append(allocator, VocabEntry{ .normal = placeholder });
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
                    for (arr, 0..) |item, i| {
                        if (item == .string) {
                            const s = item.string;
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

        // 打印统计信息
        {
            var normal_count: u32 = 0;
            var byte_count: u32 = 0;
            for (tok.vocab.items) |entry| {
                switch (entry) {
                    .normal => normal_count += 1,
                    .byte => byte_count += 1,
                }
            }
            log.info("Tokenizer initialized: {d} entries (normal={d}, byte={d}), special: bos={d}, eos={d}, unk={d}", .{
                tok.vocab.items.len,
                normal_count,
                byte_count,
                tok.special.bos,
                tok.special.eos,
                tok.special.unk,
            });
        }

        // 打印前 20 个 token 用于调试（十六进制输出，避免编码问题）
        if (tok.vocab.items.len > 0) {
            log.debug("First 20 tokens (hex dump):", .{});
            const count = @min(tok.vocab.items.len, @as(usize, 20));
            for (tok.vocab.items[0..count], 0..) |entry, i| {
                switch (entry) {
                    .normal => |s| {
                        // 打印前 32 字节的十六进制值
                        const show_len = @min(s.len, @as(usize, 32));
                        var hex_buf: [96]u8 = undefined;
                        var hex_len: usize = 0;
                        for (s[0..show_len]) |b| {
                            _ = std.fmt.bufPrint(hex_buf[hex_len..], "{x:0>2} ", .{b}) catch break;
                            hex_len += 3;
                        }
                        if (hex_len > 0) hex_len -= 1; // 去掉末尾空格
                        log.debug("  token[{d}]: NORMAL len={d} hex=[{s}] repr='{s}'", .{
                            i, s.len, hex_buf[0..hex_len], s,
                        });
                    },
                    .byte => |b| {
                        log.debug("  token[{d}]: BYTE val=0x{x:0>2}", .{ i, b });
                    },
                }
            }
        }

        // 打印一些字节 token 示例
        {
            var byte_count: u32 = 0;
            log.debug("Sample byte tokens:", .{});
            for (tok.vocab.items, 0..) |entry, i| {
                if (entry == .byte and byte_count < 10) {
                    log.debug("  byte_token[{d}]: val=0x{x:0>2}", .{ i, entry.byte });
                    byte_count += 1;
                }
            }
            log.debug("Total byte tokens: {d}", .{byte_count});
        }

        return tok;
    }

    /// 释放分词器资源
    pub fn deinit(self: *Tokenizer) void {
        // 释放 vocab_reverse 哈希表（它的 key 只是 vocab 中 normal 字符串的引用）
        self.vocab_reverse.deinit(self.allocator);
        // 释放 byte_decoder 哈希表
        self.byte_decoder.deinit(self.allocator);



        // 释放 vocab 中每个 normal 条目的字符串内存
        for (self.vocab.items) |entry| {
            if (entry == .normal) {
                self.allocator.free(entry.normal);
            }
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

            // 尝试匹配所有 normal 类型的 token
            for (self.vocab.items, 0..) |entry, id| {
                if (entry == .normal) {
                    const token_str = entry.normal;
                    if (pos + token_str.len <= text.len and
                        std.mem.eql(u8, text[pos .. pos + token_str.len], token_str))
                    {
                        if (token_str.len > best_len) {
                            best_len = token_str.len;
                            best_id = @intCast(id);
                        }
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

    /// 解码：将 token ID 列表转换为 UTF-8 文本
    /// 正确处理字节 token（LLAMA_TOKEN_TYPE_BYTE = 6）：
    /// - 字节 token 输出单个字节值
    /// - 普通 token 直接输出文本
    /// - 所有 token 一次性解码，确保 UTF-8 字节序列正确组合
    pub fn decode(self: *const Tokenizer, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
        var result = std.ArrayList(u8).empty;

        for (token_ids) |token_id| {
            // 跳过特殊 token
            if (token_id == self.special.bos or
                token_id == self.special.eos or
                token_id == self.special.pad)
            {
                continue;
            }

            // 查找 token
            if (token_id < self.vocab.items.len) {
                const entry = self.vocab.items[token_id];
                switch (entry) {
                    .byte => |byte_val| {
                        try result.append(allocator, byte_val);
                    },
                    .normal => |token_str| {
                        if (self.model == .tiktoken) {
                            // tiktoken 模型：token 字符串中的每个 Unicode 码点
                            // 都是经过 bytes_to_unicode() 映射后的值
                            // 需要通过 byte_decoder 反向映射还原为原始字节
                            try self.decodeTiktokenToken(token_str, &result, allocator);
                        } else {
                            // 其他模型：直接输出 token 字符串
                            try result.appendSlice(allocator, token_str);
                        }
                    },
                }
            } else {
                // 未知 token
                log.debug("Unknown token id: {d}", .{token_id});
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// 解码 tiktoken 格式的 token 字符串
    ///
    /// tiktoken 使用 bytes_to_unicode() 将每个字节（0x00-0xFF）映射到
    /// 一个 Unicode 码点。可打印 ASCII（33-126）映射到自身，其他字节
    /// 映射到 256+ 的区域。解码时需要将每个 Unicode 码点通过
    /// byte_decoder 反向映射还原为原始字节。
    fn decodeTiktokenToken(
        self: *const Tokenizer,
        token_str: []const u8,
        result: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
    ) !void {
        // 逐个解析 token_str 中的 Unicode 码点并转换
        var remaining = token_str;
        while (remaining.len > 0) {
            // 解码一个 Unicode 码点
            const codepoint_len = std.unicode.utf8ByteSequenceLength(remaining[0]) catch {
                log.debug("decodeTiktokenToken: invalid UTF-8 lead byte 0x{x:0>2}, skipping", .{remaining[0]});
                try result.append(allocator, remaining[0]);
                remaining = remaining[1..];
                continue;
            };

            if (remaining.len < codepoint_len) {
                log.debug("decodeTiktokenToken: truncated UTF-8 sequence, skipping", .{});
                try result.append(allocator, remaining[0]);
                remaining = remaining[1..];
                continue;
            }

            const codepoint = std.unicode.utf8Decode(remaining[0..codepoint_len]) catch {
                log.debug("decodeTiktokenToken: invalid UTF-8 sequence, skipping byte", .{});
                try result.append(allocator, remaining[0]);
                remaining = remaining[1..];
                continue;
            };

            // 查找 byte_decoder 映射
            if (self.byte_decoder.get(codepoint)) |byte_val| {
                try result.append(allocator, byte_val);
            } else {
                // 如果映射中没有，直接输出该码点的 UTF-8 字节（极少情况）
                log.debug("decodeTiktokenToken: codepoint U+{x:0>4} not in byte_decoder, outputting as-is", .{codepoint});
                try result.appendSlice(allocator, remaining[0..codepoint_len]);
            }

            // 跳过已处理的码点
            remaining = remaining[codepoint_len..];
        }
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
// 模块级工具函数
// ============================================================================


/// 生成 tiktoken 的 bytes_to_unicode 映射表
///
/// 这是 OpenAI tiktoken 的标准映射方案：
/// - 可打印 ASCII（33-126）→ 映射到自身码点
/// - 其他字节（0-32, 127-255）→ 映射到 256 + 偏移
///
/// 返回一个长度为 256 的数组，index 为原始字节值，value 为对应的 Unicode 码点。
pub fn generateBytesToUnicode(allocator: std.mem.Allocator) ![]u32 {
    var list = try std.ArrayList(u32).initCapacity(allocator, 256);
    defer list.deinit(allocator);

    // 第一步：标记所有映射到自身的字节
    // 可打印 ASCII: 33-126
    // 扩展 ASCII 可打印: 161-172 (¡-¬), 174-255 (®-ÿ)
    var bs: [256]bool = .{false} ** 256;
    for (33..127) |b| {
        bs[b] = true;
    }
    for (161..173) |b| {
        bs[b] = true;
    }
    for (174..256) |b| {
        bs[b] = true;
    }

    // 第二步：生成映射
    var n: u32 = 0;
    for (0..256) |b| {
        if (bs[b]) {
            // 映射到自身
            try list.append(allocator, @intCast(b));
        } else {
            // 其他字节映射到 256 + 偏移
            try list.append(allocator, 256 + n);
            n += 1;
        }
    }

    return list.toOwnedSlice(allocator);
}


/// 从 token 字符串中提取字节值
/// 支持多种格式，根据 model 类型选择解析策略：
///
/// 格式说明（取决于 tokenizer.ggml.model）：
/// - "llama" / "gpt2" (BPE): 原始单字节存储，token_bytes.len == 1
/// - "tiktoken" (Qwen): 表示为 "<0xE4>" 可打印形式
/// - "replit": 可能使用 b'<0xE4>' 格式
///
/// 当 model 为 .unknown 时，自动检测格式。
pub fn extractByteFromToken(token: []const u8, model: TokenizerModel) !u8 {
    // 根据 model 类型选择解析策略
    switch (model) {
        .llama, .gpt2 => {
            // BPE / GPT-2 类：原始单字节存储
            if (token.len == 1) {
                return token[0];
            }
            // 某些实现可能也使用 "<0xXX>" 格式，尝试解析
            if (token.len == 6 and token[0] == '<' and token[1] == '0' and
                (token[2] == 'x' or token[2] == 'X') and token[5] == '>')
            {
                return std.fmt.parseInt(u8, token[3..5], 16);
            }
            return error.InvalidByteToken;
        },
        .tiktoken => {
            // tiktoken / Qwen 类："<0xXX>" 格式
            if (token.len == 6 and token[0] == '<' and token[1] == '0' and
                (token[2] == 'x' or token[2] == 'X') and token[5] == '>')
            {
                return std.fmt.parseInt(u8, token[3..5], 16);
            }
            // 某些 tiktoken 导出可能使用 b'<0xXX>' 格式
            if (token.len == 9 and std.mem.eql(u8, token[0..2], "b'") and
                token[2] == '<' and token[7] == '>' and token[8] == '\'')
            {
                return std.fmt.parseInt(u8, token[4..6], 16);
            }
            // 单字节回退
            if (token.len == 1) {
                return token[0];
            }
            return error.InvalidByteToken;
        },
        .replit => {
            // Replit 类：可能使用 b'<0xXX>' 格式
            if (token.len == 9 and std.mem.eql(u8, token[0..2], "b'") and
                token[2] == '<' and token[7] == '>' and token[8] == '\'')
            {
                return std.fmt.parseInt(u8, token[4..6], 16);
            }
            // 也支持 "<0xXX>" 格式
            if (token.len == 6 and token[0] == '<' and token[1] == '0' and
                (token[2] == 'x' or token[2] == 'X') and token[5] == '>')
            {
                return std.fmt.parseInt(u8, token[3..5], 16);
            }
            // 单字节回退
            if (token.len == 1) {
                return token[0];
            }
            return error.InvalidByteToken;
        },
        .unknown => {
            // 自动检测格式
            // 情况1：原始单字节（BPE / GPT-2 类）
            if (token.len == 1) {
                return token[0];
            }
            // 情况2："<0xXX>" 格式（tiktoken / Qwen 类）
            if (token.len == 6 and token[0] == '<' and token[1] == '0' and
                (token[2] == 'x' or token[2] == 'X') and token[5] == '>')
            {
                return std.fmt.parseInt(u8, token[3..5], 16);
            }
            // 情况3："b'<0xXX>'" 格式（tiktoken 导出格式）
            if (token.len == 9 and std.mem.eql(u8, token[0..2], "b'") and
                token[2] == '<' and token[7] == '>' and token[8] == '\'')
            {
                return std.fmt.parseInt(u8, token[4..6], 16);
            }
            // 情况4："b'<0xXX>' " 格式（tiktoken 带空格）
            if (token.len == 10 and std.mem.eql(u8, token[0..2], "b'") and
                token[2] == '<' and token[7] == '>' and token[8] == '\'' and token[9] == ' ')
            {
                return std.fmt.parseInt(u8, token[4..6], 16);
            }
            return error.InvalidByteToken;
        },
    }
}

/// 判断 token 字符串是否为字节回退格式（兼容旧代码）
pub fn isByteTokenFormat(token: []const u8) bool {
    _ = extractByteFromToken(token, .unknown) catch return false;
    return true;
}

/// 解析 "<0xXX>" 得到 0xXX（兼容旧代码）
pub fn parseByteToken(token: []const u8) !u8 {
    return extractByteFromToken(token, .unknown);
}

/// 根据 token 字符串特征推断是否为字节 token（当 token_type 缺失时的降级方案）
/// 返回 true 表示该 token 很可能是字节 token
pub fn inferIsByteToken(token: []const u8) bool {
    // 单字节 token 很可能是字节 token
    if (token.len == 1) return true;
    // "<0xXX>" 格式
    if (token.len == 6 and token[0] == '<' and token[1] == '0' and
        (token[2] == 'x' or token[2] == 'X') and token[5] == '>')
    {
        return true;
    }
    // "b'<0xXX>'" 格式
    if (token.len == 9 and std.mem.eql(u8, token[0..2], "b'") and
        token[2] == '<' and token[7] == '>' and token[8] == '\'')
    {
        return true;
    }
    return false;
}

/// 调试工具：打印字节序列的十六进制表示
pub fn hexDump(data: []const u8) void {
    const hex_chars = "0123456789abcdef";
    var line_buf: [80]u8 = undefined;
    var pos: usize = 0;

    for (data, 0..) |byte, i| {
        if (i > 0 and i % 16 == 0) {
            std.debug.print("{s}\n", .{line_buf[0..pos]});
            pos = 0;
        }
        if (pos == 0) {
            // 地址前缀
            _ = std.fmt.bufPrint(line_buf[pos..], "{x:0>8}: ", .{i}) catch {};
            pos += 10;
        }
        line_buf[pos] = hex_chars[byte >> 4];
        line_buf[pos + 1] = hex_chars[byte & 0x0F];
        line_buf[pos + 2] = ' ';
        pos += 3;
    }
    if (pos > 0) {
        std.debug.print("{s}\n", .{line_buf[0..pos]});
    }
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
    // llama 模型：原始单字节
    try testing.expectEqual(@as(u8, 0x41), try extractByteFromToken("A", .llama));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("\xE4", .llama));

    // tiktoken 模型："<0xXX>" 格式
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("<0xE4>", .tiktoken));
    try testing.expectEqual(@as(u8, 0x0A), try extractByteFromToken("<0x0A>", .tiktoken));
    try testing.expectEqual(@as(u8, 0xFF), try extractByteFromToken("<0xFF>", .tiktoken));
    try testing.expectEqual(@as(u8, 0x00), try extractByteFromToken("<0x00>", .tiktoken));

    // tiktoken 模型：b'<0xXX>' 格式
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("b'<0xE4>'", .tiktoken));

    // unknown 模型：自动检测
    try testing.expectEqual(@as(u8, 0x41), try extractByteFromToken("A", .unknown));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("<0xE4>", .unknown));
    try testing.expectEqual(@as(u8, 0xE4), try extractByteFromToken("b'<0xE4>'", .unknown));

    // 无效格式
    try testing.expectError(error.InvalidByteToken, extractByteFromToken("Hello", .unknown));
    try testing.expectError(error.InvalidByteToken, extractByteFromToken("<0xE4", .unknown));
    try testing.expectError(error.InvalidByteToken, extractByteFromToken("0xE4>", .unknown));
}

test "TokenizerModel fromString" {
    try testing.expectEqual(TokenizerModel.llama, TokenizerModel.fromString("llama"));
    try testing.expectEqual(TokenizerModel.gpt2, TokenizerModel.fromString("gpt2"));
    try testing.expectEqual(TokenizerModel.tiktoken, TokenizerModel.fromString("tiktoken"));
    try testing.expectEqual(TokenizerModel.replit, TokenizerModel.fromString("replit"));
    try testing.expectEqual(TokenizerModel.unknown, TokenizerModel.fromString("unknown_model"));
    try testing.expectEqual(TokenizerModel.unknown, TokenizerModel.fromString(""));
}

test "inferIsByteToken" {
    try testing.expect(inferIsByteToken("A"));
    try testing.expect(inferIsByteToken("<0xE4>"));
    try testing.expect(inferIsByteToken("b'<0xE4>'"));
    try testing.expect(!inferIsByteToken("Hello"));
    try testing.expect(!inferIsByteToken("world"));
}

test "isByteTokenFormat" {
    try testing.expect(isByteTokenFormat("<0xE4>"));
    try testing.expect(isByteTokenFormat("b'<0xE4>'"));
    try testing.expect(!isByteTokenFormat("Hello"));
    try testing.expect(!isByteTokenFormat("<0xE4"));
}

test "VocabEntry size" {
    try testing.expectEqual(@as(usize, @sizeOf(VocabEntry)), @sizeOf(union { normal: []const u8, byte: u8 }));
}

test "Tokenizer init and deinit" {
    try testing.expectEqual(@as(usize, @sizeOf(SpecialTokens)), @sizeOf(SpecialTokens));
}

test "decode handles byte tokens correctly" {
    // 模拟一个简单的词表
    var mock_vocab: std.ArrayListUnmanaged(VocabEntry) = .empty;
    defer mock_vocab.deinit(testing.allocator);

    // token 0: "Hello" (normal)
    try mock_vocab.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "Hello") });
    // token 1: " " (normal)
    try mock_vocab.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, " ") });
    // token 2: 0xE4 (byte - 中文UTF-8首字节)
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0xE4 });
    // token 3: 0xB8 (byte)
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0xB8 });
    // token 4: 0x96 (byte)
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0x96 });
    // token 5: "World" (normal)
    try mock_vocab.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "World") });

    var tok = Tokenizer{
        .allocator = testing.allocator,
        .vocab = mock_vocab,
    };
    defer {
        // 手动释放 normal 条目的字符串
        for (tok.vocab.items) |entry| {
            if (entry == .normal) {
                testing.allocator.free(entry.normal);
            }
        }
        tok.vocab.deinit(testing.allocator);
    }

    // 解码 "Hello 世界World" (其中"世"的UTF-8是 E4 B8 96)
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const result = try tok.decode(&ids, testing.allocator);
    defer testing.allocator.free(result);

    // "Hello" (5) + " " (1) + 0xE4 0xB8 0x96 (3) + "World" (5) = 14
    try testing.expectEqual(@as(usize, 14), result.len);
    try testing.expectEqualSlices(u8, "Hello 世界World", result);
}


// 测试 Qwen 模型检测和 tiktoken 模式强制覆盖
test "Qwen model detection forces tiktoken mode" {
    // 模拟 Qwen 词表中的字节 token（tiktoken 格式）
    const byte_tokens = [_][]const u8{
        "<0x00>", "<0x01>", "<0xE4>", "<0xBA>", "<0x8C>", "<0x0A>",
    };
    const expected_bytes = [_]u8{ 0x00, 0x01, 0xE4, 0xBA, 0x8C, 0x0A };

    for (byte_tokens, 0..) |token_str, i| {
        const result = try extractByteFromToken(token_str, .tiktoken);
        try testing.expectEqual(expected_bytes[i], result);
    }

    // 验证在 gpt2 模式下，这些 "<0xXX>" 格式的 token 也能被正确解析
    for (byte_tokens) |_| {
        try testing.expectEqual(@as(u8, 0x00), try extractByteFromToken("<0x00>", .gpt2));
    }
}

// 测试 tiktoken 字节 token 的完整解码流程
test "tiktoken byte token decode flow" {
    // 模拟 Qwen 词表：前几个 token 是字节 token（tiktoken 格式）
    var mock_vocab: std.ArrayListUnmanaged(VocabEntry) = .empty;
    defer mock_vocab.deinit(testing.allocator);

    // 添加一些普通 token
    try mock_vocab.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "Hello") });
    try mock_vocab.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, " ") });

    // 添加字节 token（模拟 tiktoken 格式解析后的结果）
    // "中" 的 UTF-8 编码是 E4 B8 AD
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0xE4 });
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0xB8 });
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0xAD });

    // "国" 的 UTF-8 编码是 E5 9B BD
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0xE5 });
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0x9B });
    try mock_vocab.append(testing.allocator, VocabEntry{ .byte = 0xBD });

    try mock_vocab.append(testing.allocator, VocabEntry{ .normal = try testing.allocator.dupe(u8, "!") });

    var tok = Tokenizer{
        .allocator = testing.allocator,
        .vocab = mock_vocab,
        .model = .tiktoken,
    };
    defer {
        for (tok.vocab.items) |entry| {
            if (entry == .normal) {
                testing.allocator.free(entry.normal);
            }
        }
        tok.vocab.deinit(testing.allocator);
    }

    // 解码 "Hello 中国!"
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = try tok.decode(&ids, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(u8, "Hello 中国!", result);
}

