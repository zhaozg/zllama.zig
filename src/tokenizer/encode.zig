//! 编码逻辑
//!
//! 实现文本到 token ID 列表的编码过程。
//! 参考 llama.cpp 的编码流程：
//! 1. 特殊 token 预分词（如果 parse_special=true）：扫描文本中的特殊 token
//! 2. 预分词（Pre-tokenization）：根据预分词器类型分割文本
//! 3. 添加空格前缀（如果配置需要）
//! 4. 逐词编码（Trie 贪婪匹配 + BPE 合并）
//! 5. 添加特殊 token（BOS/EOS）
//!
//! 注意：对于 SPM 模型（tokenizer.ggml.model = "llama"），空格前缀使用 ▁ (U+2581)
//! 而不是普通空格。Trie 中存储的是原始 token 字符串（包含 ▁），所以匹配时
//! 需要使用 ▁ 作为前缀。

const std = @import("std");
const types = @import("types.zig");
const trie = @import("trie.zig");
const bpe = @import("bpe.zig");
const mod = @import("mod.zig");

const log = std.log.scoped(.tokenizer);

/// SPM 空格标记：▁ (U+2581)，UTF-8 编码为 0xE2 0x96 0x81
const SPM_SPACE = "\xE2\x96\x81";

// ============================================================================
// 预分词器
// ============================================================================

/// 预分词结果：分割后的单词列表
pub const PreTokenized = struct {
    words: std.ArrayListUnmanaged([]const u8),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PreTokenized) void {
        for (self.words.items) |w| {
            self.allocator.free(w);
        }
        self.words.deinit(self.allocator);
    }
};

/// 根据预分词器类型分割文本
pub fn preTokenize(text: []const u8, pre_type: types.PreTokenizerType, allocator: std.mem.Allocator) !PreTokenized {
    var result = PreTokenized{
        .words = .empty,
        .allocator = allocator,
    };

    switch (pre_type) {
        .default, .llama3, .gpt2, .gemma4 => {
            try preTokenizeGPT2(text, &result);
        },
        .qwen2, .qwen35 => {
            try preTokenizeQwen(text, &result);
        },
        else => {
            const word = try allocator.dupe(u8, text);
            try result.words.append(allocator, word);
        },
    }

    return result;
}

/// SPM 风格预分词：基于 Unicode 脚本分割，不分割 < 和 > 等特殊字符
/// 用于 SPM 模型（tokenizer.ggml.model = "llama"），避免特殊 token 被拆解
pub fn preTokenizeSPM(text: []const u8, allocator: std.mem.Allocator) !PreTokenized {
    var result = PreTokenized{
        .words = .empty,
        .allocator = allocator,
    };

    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) {
        // 检测特殊 token 边界：<...> 形式的 token 保持完整
        if (text[i] == '<') {
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            const close = std.mem.indexOfScalarPos(u8, text, i + 1, '>') orelse {
                i += 1;
                continue;
            };
            const word = try allocator.dupe(u8, text[i .. close + 1]);
            try result.words.append(allocator, word);
            i = close + 1;
            start = i;
        } else if (isWhitespace(text[i])) {
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            start = i;
            while (i < text.len and isWhitespace(text[i])) {
                i += 1;
            }
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            start = i;
        } else {
            i += 1;
        }
    }

    if (i > start) {
        const word = try allocator.dupe(u8, text[start..i]);
        try result.words.append(allocator, word);
    }

    return result;
}

/// GPT-2 风格预分词
fn preTokenizeGPT2(text: []const u8, result: *PreTokenized) !void {
    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) {
        if (isWhitespace(text[i])) {
            if (i > start) {
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
            }
            start = i;
            while (i < text.len and isWhitespace(text[i])) {
                i += 1;
            }
            if (i > start) {
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
            }
            start = i;
        } else if (isPunctuation(text[i])) {
            if (i > start) {
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
            }
            const word = try result.allocator.dupe(u8, text[i .. i + 1]);
            try result.words.append(result.allocator, word);
            i += 1;
            start = i;
        } else {
            i += 1;
        }
    }

    if (i > start) {
        const word = try result.allocator.dupe(u8, text[start..i]);
        try result.words.append(result.allocator, word);
    }
}

/// Qwen 风格预分词
fn preTokenizeQwen(text: []const u8, result: *PreTokenized) !void {
    try preTokenizeGPT2(text, result);
}

fn isWhitespace(c: u8) bool {
    return switch (c) {
        ' ', '\t', '\n', '\r', 0x0B, 0x0C => true,
        else => false,
    };
}

fn isPunctuation(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

fn isAllWhitespace(s: []const u8) bool {
    for (s) |c| {
        if (!isWhitespace(c)) return false;
    }
    return true;
}

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

/// 特殊 token 替换结果：文本中的特殊 token 已被替换为 token ID
/// 用于 parse_special 模式下的预处理
const SpecialTokenSegment = struct {
    token_id: u32,
};

const TextOrSpecial = union(enum) {
    text: []const u8,
    special: SpecialTokenSegment,
};

/// 对文本进行特殊 token 扫描，将其中的特殊 token 匹配并分割
/// 返回 text 和 special token 的交替列表
/// 与 llama.cpp 的 tokenizer_st_partition 对应
fn partitionSpecialTokens(
    text: []const u8,
    cache: []const mod.CacheSpecialToken,
    allocator: std.mem.Allocator,
) !std.ArrayListUnmanaged(TextOrSpecial) {
    var result: std.ArrayListUnmanaged(TextOrSpecial) = .empty;
    errdefer {
        for (result.items) |item| {
            if (item == .text) allocator.free(item.text);
        }
        result.deinit(allocator);
    }

    // 使用一个布尔数组标记哪些位置已被特殊 token 覆盖
    var covered = try allocator.alloc(bool, text.len);
    defer allocator.free(covered);
    @memset(covered, false);

    // 按长度降序遍历所有特殊 token（最长优先匹配）
    for (cache) |st| {
        if (st.text.len == 0) continue;
        var pos: usize = 0;
        while (pos + st.text.len <= text.len) {
            if (!covered[pos] and std.mem.eql(u8, text[pos .. pos + st.text.len], st.text)) {
                // 检查覆盖范围是否与其他特殊 token 重叠（只有未覆盖的区域才匹配）
                var all_free = true;
                for (pos..pos + st.text.len) |j| {
                    if (covered[j]) { all_free = false; break; }
                }
                if (all_free) {
                    // 检查属性：control 和 unknown token 仅在 parse_special 时匹配
                    // user_defined token 总是匹配
                    for (pos..pos + st.text.len) |j| {
                        covered[j] = true;
                    }
                }
            }
            pos += 1;
        }
    }

    // 根据 covered 数组构建 text/special 交替列表
    var i: usize = 0;
    while (i < text.len) {
        if (covered[i]) {
            // 找到覆盖位置对应的特殊 token
            // 由于特殊 token 已按长度降序匹配，找到第一个匹配的即可
            const matched = findMatchingSpecial(text, i, cache) orelse {
                // 理论上不会发生，但作为回退
                i += 1;
                continue;
            };
            try result.append(allocator, .{ .special = .{ .token_id = matched.id } });
            i += matched.text.len;
        } else {
            // 收集连续的未覆盖文本
            const start = i;
            while (i < text.len and !covered[i]) {
                i += 1;
            }
            const segment = try allocator.dupe(u8, text[start..i]);
            try result.append(allocator, .{ .text = segment });
        }
    }

    return result;
}

/// 在给定位置查找匹配的特殊 token（假设该位置已被覆盖）
fn findMatchingSpecial(text: []const u8, pos: usize, cache: []const mod.CacheSpecialToken) ?mod.CacheSpecialToken {
    for (cache) |st| {
        if (pos + st.text.len <= text.len and std.mem.eql(u8, text[pos .. pos + st.text.len], st.text)) {
            return st;
        }
    }
    return null;
}

// ============================================================================
// 编码主函数
// ============================================================================

/// 编码：将文本转换为 token ID 列表
/// 完整的编码流程：
/// 1. 特殊 token 预分词（如果 parse_special=true）
/// 2. 预分词（SPM 模型使用 SPM 风格预分词）
/// 3. 添加空格前缀（如果配置需要）
/// 4. 逐词编码（Trie 贪婪匹配 + BPE 合并）
/// 5. 添加特殊 token（BOS/EOS）
pub fn encode(
    text: []const u8,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    ignore_merges: bool,
    parse_special: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (add_bos) {
        try tokens.append(config.allocator, config.special.bos);
    }

    if (config.vocab.items.len == 0) {
        for (text) |byte| try tokens.append(config.allocator, config.byteToTokenIdFn(byte, config.ctx));
        return tokens;
    }

    const is_spm_model = config.model == .llama or config.model == .spm;

    // 1a. 特殊 token 预分词（如果 parse_special=true 且有缓存）
    if (parse_special and config.cache_special_tokens != null and config.cache_special_tokens.?.len > 0) {
        var segments = try partitionSpecialTokens(text, config.cache_special_tokens.?, config.allocator);
        defer {
            for (segments.items) |item| {
                if (item == .text) config.allocator.free(item.text);
            }
            segments.deinit(config.allocator);
        }

        for (segments.items) |seg| {
            switch (seg) {
                .text => |txt| {
                    // 对普通文本段进行常规编码
                    var word_tokens = try encodeSegment(txt, add_space_prefix, ignore_merges, is_spm_model, config);
                    defer word_tokens.deinit(config.allocator);
                    try tokens.appendSlice(config.allocator, word_tokens.items);
                },
                .special => |sp| {
                    // 直接添加特殊 token ID
                    try tokens.append(config.allocator, sp.token_id);
                },
            }
        }

        if (add_eos) {
            try tokens.append(config.allocator, config.special.eos);
        }
        return tokens;
    }

    // 1b. 常规预分词（无特殊 token 扫描）
    var pre_tok = if (is_spm_model)
        try preTokenizeSPM(text, config.allocator)
    else
        try preTokenize(text, config.pre_type, config.allocator);
    defer pre_tok.deinit();

    // 2. 对每个词进行编码
    for (pre_tok.words.items) |word| {
        if (is_spm_model and add_space_prefix and isAllWhitespace(word)) {
            continue;
        }

        var word_tokens = try encodeWord(word, add_space_prefix, ignore_merges, config);
        defer word_tokens.deinit(config.allocator);
        try tokens.appendSlice(config.allocator, word_tokens.items);
    }

    if (add_eos) {
        try tokens.append(config.allocator, config.special.eos);
    }

    return tokens;
}

/// 编码单个文本段（用于 parse_special 模式下的普通文本部分）
fn encodeSegment(
    text: []const u8,
    add_space_prefix: bool,
    ignore_merges: bool,
    is_spm_model: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (text.len == 0) return tokens;

    var pre_tok = if (is_spm_model)
        try preTokenizeSPM(text, config.allocator)
    else
        try preTokenize(text, config.pre_type, config.allocator);
    defer pre_tok.deinit();

    for (pre_tok.words.items) |word| {
        if (is_spm_model and add_space_prefix and isAllWhitespace(word)) continue;

        var word_tokens = try encodeWord(word, add_space_prefix, ignore_merges, config);
        defer word_tokens.deinit(config.allocator);
        try tokens.appendSlice(config.allocator, word_tokens.items);
    }

    return tokens;
}

/// 编码单个词
fn encodeWord(
    word: []const u8,
    add_space_prefix: bool,
    ignore_merges: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    const is_spm_model = config.model == .llama or config.model == .spm;

    // 步骤 1：确定基础文本（可能添加空格前缀）
    const base_text = if (add_space_prefix) blk: {
        if (is_spm_model) {
            if (word.len > 0 and isWhitespace(word[0])) {
                var ws_end: usize = 1;
                while (ws_end < word.len and isWhitespace(word[ws_end])) ws_end += 1;
                if (ws_end < word.len) {
                    break :blk try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word[ws_end..] });
                }
            }
            break :blk try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word });
        } else {
            break :blk try std.fmt.allocPrint(config.allocator, " {s}", .{word});
        }
    } else word;

    const base_needs_free = add_space_prefix;
    errdefer {
        if (base_needs_free) config.allocator.free(base_text);
    }

    // 步骤 2：对基础文本进行 GPT-2 字节编码（如果需要）
    const use_gpt2_encoding = config.bytesToUnicodeFn != null and config.model == .gpt2;

    const final_text: []const u8 = if (use_gpt2_encoding) blk: {
        const encoded = try toGpt2ByteEncoding(base_text, config.bytesToUnicodeFn.?, config.ctx, config.allocator);
        if (base_needs_free) config.allocator.free(base_text);
        break :blk encoded;
    } else base_text;

    const final_needs_free = if (use_gpt2_encoding) true else base_needs_free;

    defer {
        if (final_needs_free) config.allocator.free(@constCast(final_text));
    }

    // 阶段 1：Trie 贪婪最长匹配
    var pos: usize = 0;
    while (pos < final_text.len) {
        const match = trie.longestMatch(config.trie_root, final_text, pos);
        if (match) |m| {
            try tokens.append(config.allocator, m.token_id);
            pos += m.len;
        } else {
            if (config.unicodeToByte) |utb| {
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

    // 阶段 2：BPE 合并（如果不忽略合并规则）
    if (!ignore_merges and config.merges.count() > 0) {
        try bpe.applyBpeMerges(&tokens, config.merges, config.tokenToStringFn, config.textToTokenFn, config.ctx, config.allocator);
    }

    return tokens;
}

// ============================================================================
// 编码配置
// ============================================================================

/// 编码所需的配置和回调函数
pub const EncodeConfig = struct {
    allocator: std.mem.Allocator,
    special: types.SpecialTokens,
    pre_type: types.PreTokenizerType,
    model: types.TokenizerModel,
    vocab: std.ArrayListUnmanaged(types.VocabEntry),
    merges: std.StringHashMap(u32),
    trie_root: *const trie.TrieNode,
    tokenToStringFn: *const fn (token_id: u32, ctx: ?*anyopaque) ?[]const u8,
    textToTokenFn: *const fn (text: []const u8, ctx: ?*anyopaque) ?u32,
    byteToTokenIdFn: *const fn (byte: u8, ctx: ?*anyopaque) u32,
    bytesToUnicodeFn: ?*const fn (byte: u8, ctx: ?*anyopaque) []const u8 = null,
    unicodeToByte: ?*const std.StringHashMap(u8) = null,
    ctx: ?*anyopaque,
    /// 缓存的特殊 token 列表（按 text 长度降序排列），用于 parse_special 模式
    cache_special_tokens: ?[]const mod.CacheSpecialToken = null,
};
