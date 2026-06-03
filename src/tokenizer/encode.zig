//! 编码逻辑
//!
//! 实现文本到 token ID 列表的编码过程。
//! 参考 llama.cpp 的编码流程：
//! 1. 预分词（Pre-tokenization）：根据预分词器类型分割文本
//! 2. 添加空格前缀（如果配置需要）
//! 3. 逐词编码（Trie 贪婪匹配 + BPE 合并）
//! 4. 添加特殊 token（BOS/EOS）
//!
//! 注意：对于 SPM 模型（tokenizer.ggml.model = "llama"），空格前缀使用 ▁ (U+2581)
//! 而不是普通空格。Trie 中存储的是原始 token 字符串（包含 ▁），所以匹配时
//! 需要使用 ▁ 作为前缀。

const std = @import("std");
const types = @import("types.zig");
const trie = @import("trie.zig");
const bpe = @import("bpe.zig");

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
/// 参考 llama.cpp 的 llm_tokenizer_bpe_session::tokenize 和 llm_tokenizer_spm_session::tokenize
pub fn preTokenize(text: []const u8, pre_type: types.PreTokenizerType, allocator: std.mem.Allocator) !PreTokenized {
    var result = PreTokenized{
        .words = .empty,
        .allocator = allocator,
    };

    switch (pre_type) {
        .default, .llama3, .gpt2 => {
            // GPT-2 / LLaMA 3 风格预分词
            try preTokenizeGPT2(text, &result);
        },
        .qwen2, .qwen35 => {
            // Qwen 风格预分词
            try preTokenizeQwen(text, &result);
        },
        else => {
            // 默认：将整个文本作为一个词
            const word = try allocator.dupe(u8, text);
            try result.words.append(allocator, word);
        },
    }

    return result;
}

/// GPT-2 风格预分词
/// 使用简化的正则表达式模式分割文本
fn preTokenizeGPT2(text: []const u8, result: *PreTokenized) !void {
    // GPT-2 的预分词正则表达式（简化版）：
    // '(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    //
    // 简化实现：按空白和标点分割，但保留空格前缀
    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) {
        // 查找下一个单词边界
        if (isWhitespace(text[i])) {
            // 输出前面的非空白部分
            if (i > start) {
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
            }
            // 收集连续的空白
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
            // 输出前面的非标点部分
            if (i > start) {
                const word = try result.allocator.dupe(u8, text[start..i]);
                try result.words.append(result.allocator, word);
            }
            // 单个标点作为一个词
            const word = try result.allocator.dupe(u8, text[i .. i + 1]);
            try result.words.append(result.allocator, word);
            i += 1;
            start = i;
        } else {
            i += 1;
        }
    }

    // 输出剩余部分
    if (i > start) {
        const word = try result.allocator.dupe(u8, text[start..i]);
        try result.words.append(result.allocator, word);
    }
}

/// Qwen 风格预分词
fn preTokenizeQwen(text: []const u8, result: *PreTokenized) !void {
    // Qwen 使用 tiktoken 分词器，预分词逻辑与 GPT-2 类似
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

// ============================================================================
// GPT-2 字节编码转换
// ============================================================================

/// 将文本转换为 GPT-2 字节编码
/// GPT-2 的字节编码将每个字节映射到一个可打印的 Unicode 字符
pub fn toGpt2ByteEncoding(text: []const u8, bytesToUnicodeFn: *const fn (byte: u8, ctx: ?*anyopaque) []const u8, ctx: ?*anyopaque, allocator: std.mem.Allocator) ![]u8 {
    // 计算总长度
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
// 编码主函数
// ============================================================================

/// 编码：将文本转换为 token ID 列表
/// 完整的编码流程：
/// 1. 预分词
/// 2. 添加空格前缀（如果配置需要）
/// 3. 逐词编码（Trie 贪婪匹配 + BPE 合并）
/// 4. 添加特殊 token（BOS/EOS）
pub fn encode(
    text: []const u8,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    ignore_merges: bool,
    config: *const EncodeConfig,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (add_bos) {
        try tokens.append(config.allocator, config.special.bos);
    }

    if (config.vocab.items.len == 0) {
        // 没有词表，使用字节级回退
        for (text) |byte| try tokens.append(config.allocator, config.byteToTokenIdFn(byte, config.ctx));
        return tokens;
    }

    // 1. 预分词
    var pre_tok = try preTokenize(text, config.pre_type, config.allocator);
    defer pre_tok.deinit();

    // 2. 对每个词进行编码
    for (pre_tok.words.items) |word| {
        var word_tokens = try encodeWord(word, add_space_prefix, ignore_merges, config);
        defer word_tokens.deinit(config.allocator);
        try tokens.appendSlice(config.allocator, word_tokens.items);
    }

    if (add_eos) {
        try tokens.append(config.allocator, config.special.eos);
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

    // 确定要编码的文本
    // 对于 SPM 模型（tokenizer.ggml.model = "llama"），空格前缀使用 ▁ (U+2581)
    // 因为 Trie 中存储的 token 字符串包含 ▁ 作为空格标记
    // 对于 BPE 模型（tokenizer.ggml.model = "gpt2"），空格前缀使用普通空格
    //
    // 注意：需要先处理 GPT-2 字节编码，再添加空格前缀（或反之），
    // 但必须确保每个分配只释放一次，避免 double free。

    // 步骤 1：确定基础文本（可能添加空格前缀）
    const base_text = if (add_space_prefix) blk: {
        // SPM 模型使用 ▁ 作为空格标记，BPE 模型使用普通空格
        if (config.model == .llama) {
            break :blk try std.fmt.allocPrint(config.allocator, "{s}{s}", .{ SPM_SPACE, word });
        } else {
            break :blk try std.fmt.allocPrint(config.allocator, " {s}", .{word});
        }
    } else word;

    // 只有 base_text 是分配的内存时才需要释放
    const base_needs_free = add_space_prefix;
    errdefer {
        if (base_needs_free) config.allocator.free(base_text);
    }

    // 步骤 2：对基础文本进行 GPT-2 字节编码（如果需要）
    // 注意：SPM 模型不使用 GPT-2 字节编码，只有 BPE 模型使用
    const use_gpt2_encoding = config.bytesToUnicodeFn != null and config.model != .llama;

    const final_text: []const u8 = if (use_gpt2_encoding) blk: {
        const encoded = try toGpt2ByteEncoding(base_text, config.bytesToUnicodeFn.?, config.ctx, config.allocator);
        // 如果 base_text 是分配的，现在可以释放了（因为 encoded 是独立副本）
        if (base_needs_free) config.allocator.free(base_text);
        break :blk encoded;
    } else base_text;

    // final_text 是否需要释放
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
            // 如果 Trie 匹配失败，尝试使用字节 token
            // 对于 GPT-2 编码的文本，需要先将字符解码回原始字节
            if (config.unicodeToByte) |utb| {
                // 尝试匹配 1 个字符的 GPT-2 编码
                const ch_len = std.unicode.utf8ByteSequenceLength(final_text[pos]) catch 1;
                const ch = final_text[pos .. pos + @as(usize, ch_len)];
                if (utb.get(ch)) |byte| {
                    try tokens.append(config.allocator, config.byteToTokenIdFn(byte, config.ctx));
                    pos += ch.len;
                } else {
                    // 回退到原始字节
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
        try bpe.applyBpeMerges(&tokens, config.merges, config.tokenToStringFn, config.ctx, config.allocator);
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
    byteToTokenIdFn: *const fn (byte: u8, ctx: ?*anyopaque) u32,
    bytesToUnicodeFn: ?*const fn (byte: u8, ctx: ?*anyopaque) []const u8 = null,
    unicodeToByte: ?*const std.StringHashMap(u8) = null,
    ctx: ?*anyopaque,
};
