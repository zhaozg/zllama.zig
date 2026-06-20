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
// NOTE: 通过 mod 模块的 pub const 导出访问子模块，确保与 mod.zig 使用同一个模块实例
// 避免因不同 @import 路径导致的类型不匹配（如 TrieNode 被创建为多个不兼容的类型）
const mod = @import("mod.zig");
const types = mod.types;
const trie = mod.trie;
const bpe = mod.bpe;
const unicode = mod.unicode;
const preTokenizeFalcon = @import("models/falcon.zig").preTokenizeFalcon;
const preTokenizeMpt = @import("models/mpt.zig").preTokenizeMpt;
const preTokenizeStarcoderStyle = @import("models/starcoder.zig").preTokenizeStarcoderStyle;
const preTokenizeDeepseekLlm = @import("models/deepseek_llm.zig").preTokenizeDeepseekLlm;
const preTokenizeDeepseekCoder = @import("models/deepseek_coder.zig").preTokenizeDeepseekCoder;
const preTokenizeDeepseek3Style = @import("models/deepseek3.zig").preTokenizeDeepseek3Style;
const preTokenizeBloomStyle = @import("models/bloom.zig").preTokenizeBloomStyle;
const preTokenizeGpt2Style = @import("models/gpt2_style.zig").preTokenizeGpt2Style;
const preTokenizeGpt2StyleNoSpace = @import("models/gpt2_style_nospace.zig").preTokenizeGpt2StyleNoSpace;
const preTokenizeLlama3 = @import("models/llama3.zig").preTokenizeLlama3;
const preTokenizeQwen2Style = @import("models/qwen2.zig").preTokenizeQwen2Style;
const preTokenizeGPT2 = @import("models/gpt2.zig").preTokenizeGPT2;
const preTokenizeQwen = @import("models/qwen.zig").preTokenizeQwen;
const preTokenizeNewlineOnly = @import("models/newline_only.zig").preTokenizeNewlineOnly;
const tryMatchContractionOrWord = @import("models/tryMatchContractionOrWord.zig").tryMatchContractionOrWord;


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
        .default => {
            try preTokenizeGPT2(text, &result);
        },
        .llama3 => {
            // llama-bpe, llama-v3, falcon3 等使用 llama3 pre_type 的模型
            // 使用 llama3 正则：\p{N}{1,3} 分组数字，不保留数字前导空格
            try preTokenizeLlama3(text, &result);
        },
        .gpt2 => {
            try preTokenizeGpt2Style(text, &result);
        },
        .gemma4 => {
            try preTokenizeNewlineOnly(text, &result);
        },
        .qwen2, .qwen35 => {
            // Qwen2/Qwen35 使用单数字 \p{N} 模式
            try preTokenizeQwen2Style(text, &result);
        },
        .falcon => {
            try preTokenizeFalcon(text, &result);
        },
        .mpt => {
            try preTokenizeMpt(text, &result);
        },
        .starcoder, .refact, .command_r, .smollm, .codeshell, .exaone, .minerva => {
            try preTokenizeStarcoderStyle(text, &result);
        },
        .deepseek_llm => {
            try preTokenizeDeepseekLlm(text, &result);
        },
        .deepseek_coder => {
            try preTokenizeDeepseekCoder(text, &result);
        },
        .deepseek3_llm, .hunyuan_dense, .joyai_llm => {
            try preTokenizeDeepseek3Style(text, &result);
        },
        .bloom, .poro, .gpt3_finnish => {
            try preTokenizeBloomStyle(text, &result);
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
        } else if (unicode.isAsciiWhitespace(text[i])) {
            if (i > start) {
                const word = try allocator.dupe(u8, text[start..i]);
                try result.words.append(allocator, word);
            }
            start = i;
            while (i < text.len and unicode.isAsciiWhitespace(text[i])) {
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

/// 用于 gemma-4 等 SPM 风格 BPE 模型，在预分词前应用
/// 参考 llama.cpp 的 llama_escape_whitespace()
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
                    if (covered[j]) {
                        all_free = false;
                        break;
                    }
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
// SPM 编码（基于字符的 bigram 合并）
// ============================================================================

/// SPM bigram 用于优先级队列
const SpmBigram = struct {
    left: i32,
    right: i32,
    score: f32,
    size: usize,

    fn lessThan(context: void, a: @This(), b: @This()) std.math.Order {
        _ = context;
        // SPM: higher score = higher priority (less negative = more likely to merge)
        // Matches llama.cpp: l.score < r.score means l has lower priority
        if (a.score > b.score) return .lt;
        if (a.score < b.score) return .gt;
        // When scores are equal, prefer leftmost pair (smaller left index)
        // This matches llama.cpp behavior: leftmost pair gets merged first
        // when scores are tied, ensuring correct tokenization of repeated chars
        if (a.left < b.left) return .lt;
        if (a.left > b.left) return .gt;
        return .eq;
    }
};

/// SPM 符号
const SpmSymbol = struct {
    text: []const u8, // pointer into the original text
    n: usize,
    prev: i32,
    next: i32,
};

/// SPM 编码：将文本编码为 token ID 列表
/// 参考 llama.cpp 的 llm_tokenizer_spm_session::tokenize
/// 流程：
/// 1. 将文本拆分为 UTF-8 字符
/// 2. 使用优先级队列反复合并 score 最高的相邻 token 对
/// 3. 合并后的 token 字符串必须在词表中存在
fn encodeSPM(
    text: []const u8,
    config: *const EncodeConfig,
    allocator: std.mem.Allocator,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (text.len == 0) return tokens;

    // 1. 拆分为 UTF-8 字符
    var symbols = std.ArrayListUnmanaged(SpmSymbol){ .items = &.{}, .capacity = 0 };
    defer symbols.deinit(allocator);

    var offs: usize = 0;
    var index: i32 = 0;
    while (offs < text.len) {
        const ch_len = unicode.charLen(text, offs);
        try symbols.append(allocator, SpmSymbol{
            .text = text[offs..],
            .n = ch_len,
            .prev = index - 1,
            .next = if (offs + ch_len >= text.len) -1 else index + 1,
        });
        offs += ch_len;
        index += 1;
    }

    if (symbols.items.len == 0) return tokens;

    // 2. 初始化优先级队列
    var work_queue = std.PriorityQueue(SpmBigram, void, SpmBigram.lessThan).initContext({});
    defer work_queue.deinit(allocator);

    for (1..symbols.items.len) |i| {
        tryAddSpmBigram(&symbols, &work_queue, @intCast(i - 1), @intCast(i), config);
    }

    // 3. 反复合并 score 最高的 pair
    while (work_queue.count() > 0) {
        const bigram = work_queue.pop().?;

        const left_idx = @as(usize, @intCast(bigram.left));
        const right_idx = @as(usize, @intCast(bigram.right));

        // 检查符号是否已被合并
        if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) continue;
        if (symbols.items[left_idx].next != bigram.right) continue;
        if (symbols.items[right_idx].prev != bigram.left) continue;

        // 检查 size 是否匹配
        if (symbols.items[left_idx].n + symbols.items[right_idx].n != bigram.size) continue;

        // 合并右符号到左符号
        symbols.items[left_idx].n += symbols.items[right_idx].n;
        symbols.items[right_idx].n = 0;

        // 从链表中移除右符号
        symbols.items[left_idx].next = symbols.items[right_idx].next;
        if (symbols.items[right_idx].next >= 0) {
            symbols.items[@as(usize, @intCast(symbols.items[right_idx].next))].prev = bigram.left;
        }

        // 添加新的 bigram
        tryAddSpmBigram(&symbols, &work_queue, symbols.items[left_idx].prev, bigram.left, config);
        tryAddSpmBigram(&symbols, &work_queue, bigram.left, symbols.items[left_idx].next, config);
    }

    // 4. 从符号链表重建 token 列表
    {
        var i: i32 = 0;
        while (i >= 0) {
            const idx = @as(usize, @intCast(i));
            const sym = &symbols.items[idx];
            if (sym.n > 0) {
                const token_text = sym.text[0..sym.n];
                // 查找 token
                if (config.textToTokenFn(token_text, config.ctx)) |tid| {
                    try tokens.append(allocator, tid);
                } else {
                    // 回退到字节 token
                    for (token_text) |byte| {
                        const tid = config.byteToTokenIdFn(byte, config.ctx);
                        try tokens.append(allocator, tid);
                    }
                }
            }
            i = sym.next;
        }
    }

    return tokens;
}

/// 尝试添加 SPM bigram 到优先级队列
fn tryAddSpmBigram(
    symbols: *std.ArrayListUnmanaged(SpmSymbol),
    queue: *std.PriorityQueue(SpmBigram, void, SpmBigram.lessThan),
    left: i32,
    right: i32,
    config: *const EncodeConfig,
) void {
    if (left < 0 or right < 0) return;
    const left_idx = @as(usize, @intCast(left));
    const right_idx = @as(usize, @intCast(right));
    if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) return;

    const left_sym = &symbols.items[left_idx];
    const right_sym = &symbols.items[right_idx];

    // 构建合并后的文本
    const merged_text = left_sym.text[0 .. left_sym.n + right_sym.n];

    // 查找合并后的 token
    const token_id = config.textToTokenFn(merged_text, config.ctx) orelse return;

    const score = config.tokenScoreFn(token_id, config.ctx);

    queue.push(config.allocator, SpmBigram{
        .left = left,
        .right = right,
        .score = score,
        .size = merged_text.len,
    }) catch {};
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

    // 1b. 常规编码
    if (is_spm_model) {
        // SPM 模型：使用基于字符的 bigram 合并
        if (text.len == 0) {
            // Empty text: no tokens (unless add_bos/add_eos already handled)
        } else {
            // 1. 添加空格前缀（如果需要）
            // 2. 转义空格（空格 → ▁）
            // 3. 对整个文本进行 SPM 编码
            var spm_text: []const u8 = text;
            var spm_needs_free = false;
            defer {
                if (spm_needs_free) config.allocator.free(@constCast(spm_text));
            }

            // 添加空格前缀
            if (add_space_prefix) {
                spm_text = try std.fmt.allocPrint(config.allocator, " {s}", .{text});
                spm_needs_free = true;
            }

            // 转义空格（空格 → ▁）
            const escaped = try escapeWhitespace(spm_text, config.allocator);
            if (spm_needs_free) config.allocator.free(@constCast(spm_text));
            spm_text = escaped;
            spm_needs_free = true;

            // SPM 编码
            var spm_tokens = try encodeSPM(spm_text, config, config.allocator);
            defer spm_tokens.deinit(config.allocator);
            try tokens.appendSlice(config.allocator, spm_tokens.items);
        }
    } else {
        // BPE 模型：使用预分词 + 逐词编码
        const needs_global_escape = config.escape_whitespaces;
        const escaped_text: ?[]const u8 = if (needs_global_escape)
            try escapeWhitespace(text, config.allocator)
        else
            null;
        defer if (escaped_text) |et| config.allocator.free(et);

        const effective_text = if (escaped_text) |et| et else text;

        var pre_tok = try preTokenize(effective_text, config.pre_type, config.allocator);
        defer pre_tok.deinit();

        var is_first_word = true;
        for (pre_tok.words.items) |word| {
            var word_tokens = try encodeWord(word, add_space_prefix, ignore_merges, is_first_word, config);
            defer word_tokens.deinit(config.allocator);
            try tokens.appendSlice(config.allocator, word_tokens.items);
            is_first_word = false;
        }
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

    // 对于 SPM 风格 BPE 模型（gemma-4 等），先全局转义空格再预分词
    const needs_global_escape = config.escape_whitespaces and !is_spm_model;
    const escaped_text: ?[]const u8 = if (needs_global_escape)
        try escapeWhitespace(text, config.allocator)
    else
        null;
    defer if (escaped_text) |et| config.allocator.free(et);

    const effective_text = if (escaped_text) |et| et else text;

    var pre_tok = if (is_spm_model)
        try preTokenizeSPM(effective_text, config.allocator)
    else
        try preTokenize(effective_text, config.pre_type, config.allocator);
    defer pre_tok.deinit();

    var is_first_word = true;
    for (pre_tok.words.items) |word| {
        if (is_spm_model and unicode.isAllWhitespace(word)) {
            is_first_word = false;
            continue;
        }

        var word_tokens = try encodeWord(word, add_space_prefix, ignore_merges, is_first_word, config);
        defer word_tokens.deinit(config.allocator);
        try tokens.appendSlice(config.allocator, word_tokens.items);
        is_first_word = false;
    }

    return tokens;
}

/// 编码单个词
fn encodeWord(
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
    tokenScoreFn: *const fn (token_id: u32, ctx: ?*anyopaque) f32 = undefined,
    escape_whitespaces: bool = false,
    ctx: ?*anyopaque,
    /// 缓存的特殊 token 列表（按 text 长度降序排列），用于 parse_special 模式
    cache_special_tokens: ?[]const mod.CacheSpecialToken = null,
};
