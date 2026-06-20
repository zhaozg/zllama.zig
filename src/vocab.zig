//! 词表（Vocab）模块
//!
//! 对齐 llama.cpp 的 llama_vocab 设计。
//! Vocab 是纯数据模块，持有从 GGUF 解析的所有词表信息，
//! 提供 token↔text 双向查找、token 类型判定、EOG 检测等基础能力。
//! 分词算法（BPE/SPM）不在此模块中，由 tokenizer 模块负责。
//!
//! GGUF 元数据键（tokenizer.ggml.*）：
//!   - tokenizer.ggml.model         → 分词器算法类型
//!   - tokenizer.ggml.pre           → 预分词器变体
//!   - tokenizer.ggml.tokens        → token 字符串数组
//!   - tokenizer.ggml.scores        → token 分数数组
//!   - tokenizer.ggml.token_type    → token 类型数组
//!   - tokenizer.ggml.merges        → BPE 合并规则
//!   - tokenizer.ggml.bos_token_id  → 句首 token ID
//!   - tokenizer.ggml.eos_token_id  → 句尾 token ID
//!   - ... 其他特殊 token ID

const std = @import("std");
const gguf = @import("gguf");

const log = std.log.scoped(.vocab);

// ============================================================================
// 枚举类型
// ============================================================================

/// 分词器算法类型，对应 tokenizer.ggml.model
pub const VocabType = enum {
    llama, // SentencePiece (SPM / Unigram)
    gpt2, // Byte-level BPE
    tiktoken, // tiktoken (OpenAI)
    replit, // Replit tokenizer
    spm, // SentencePiece (alias)
    gemma4, // Gemma-4 BPE
    bert, // BERT-style BPE (bert-bge etc.)
    unknown,

    pub fn fromString(s: []const u8) VocabType {
        if (std.ascii.eqlIgnoreCase(s, "llama")) return .llama;
        if (std.ascii.eqlIgnoreCase(s, "gpt2")) return .gpt2;
        if (std.ascii.eqlIgnoreCase(s, "tiktoken")) return .tiktoken;
        if (std.ascii.eqlIgnoreCase(s, "replit")) return .replit;
        if (std.ascii.eqlIgnoreCase(s, "spm")) return .spm;
        if (std.ascii.eqlIgnoreCase(s, "gemma4")) return .gemma4;
        if (std.ascii.eqlIgnoreCase(s, "bert")) return .bert;
        return .unknown;
    }

    pub fn toString(self: VocabType) []const u8 {
        return @tagName(self);
    }

    /// 是否使用 SentencePiece 风格的 tokenization
    pub fn isSPM(self: VocabType) bool {
        return self == .llama or self == .spm;
    }

    /// 是否使用 BPE 风格的 tokenization
    pub fn isBPE(self: VocabType) bool {
        return self == .gpt2 or self == .tiktoken or self == .gemma4 or self == .bert;
    }
};

/// Token 类型，对应 llama.cpp 的 LLAMA_TOKEN_TYPE
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

/// Token 属性位标志，对应 llama.cpp 的 llama_token_attr
pub const TokenAttr = packed struct(u32) {
    undefined: bool = false,
    normal: bool = false,
    unknown: bool = false,
    control: bool = false,
    user_defined: bool = false,
    unused: bool = false,
    byte: bool = false,
    _padding: u25 = 0,

    pub fn fromType(tt: TokenType) TokenAttr {
        return switch (tt) {
            .undefined => TokenAttr{ .undefined = true },
            .normal => TokenAttr{ .normal = true },
            .unknown => TokenAttr{ .unknown = true },
            .control => TokenAttr{ .control = true },
            .user_defined => TokenAttr{ .user_defined = true },
            .unused => TokenAttr{ .unused = true },
            .byte => TokenAttr{ .byte = true },
            _ => TokenAttr{},
        };
    }
};

/// 预分词器类型，对应 llama.cpp 的 llama_vocab_pre_type
pub const PreType = enum {
    default,
    llama3,
    deepseek_llm,
    deepseek_coder,
    falcon,
    mpt,
    starcoder,
    gpt2,
    refact,
    command_r,
    stablelm2,
    qwen2,
    olmo,
    dbrx,
    smaug,
    poro,
    chatglm3,
    chatglm4,
    viking,
    jais,
    tekken,
    smollm,
    codeshell,
    bloom,
    gpt3_finnish,
    exaone,
    chameleon,
    minerva,
    deepseek3_llm,
    gpt4o,
    super_bpe,
    trillion,
    bailingmoe,
    llama4,
    pixtral,
    seed_coder,
    hunyuan,
    kimi_k2,
    hunyuan_dense,
    grok_2,
    granite_docling,
    minimax_m2,
    afmoe,
    solar_open,
    youtu,
    exaone_moe,
    qwen35,
    tiny_aya,
    joyai_llm,
    jais2,
    gemma4,
    sarvam_moe,
    minicpm5,

    pub fn fromString(s: []const u8) PreType {
        if (std.mem.eql(u8, s, "default")) return .default;
        if (std.mem.eql(u8, s, "llama3") or
            std.mem.eql(u8, s, "llama-v3") or
            std.mem.eql(u8, s, "llama-bpe") or
            std.mem.eql(u8, s, "falcon3") or
            std.mem.eql(u8, s, "falcon-h1") or
            std.mem.eql(u8, s, "pixtral") or
            std.mem.eql(u8, s, "midm-2.0") or
            std.mem.eql(u8, s, "lfm2") or
            std.mem.eql(u8, s, "jina-v5-nano")) return .llama3;
        if (std.mem.eql(u8, s, "deepseek-llm")) return .deepseek_llm;
        if (std.mem.eql(u8, s, "deepseek-coder")) return .deepseek_coder;
        if (std.mem.eql(u8, s, "deepseek-v3")) return .deepseek3_llm;
        if (std.mem.eql(u8, s, "falcon")) return .falcon;
        if (std.mem.eql(u8, s, "mpt")) return .mpt;
        if (std.mem.eql(u8, s, "starcoder")) return .starcoder;
        if (std.mem.eql(u8, s, "gpt2") or std.mem.eql(u8, s, "gpt-2")) return .gpt2;
        if (std.mem.eql(u8, s, "refact")) return .refact;
        if (std.mem.eql(u8, s, "command-r")) return .command_r;
        if (std.mem.eql(u8, s, "stablelm2")) return .stablelm2;
        if (std.mem.eql(u8, s, "qwen2") or std.mem.eql(u8, s, "qwen2.5")) return .qwen2;
        if (std.mem.eql(u8, s, "olmo")) return .olmo;
        if (std.mem.eql(u8, s, "dbrx")) return .dbrx;
        if (std.mem.eql(u8, s, "smaug")) return .smaug;
        if (std.mem.eql(u8, s, "poro")) return .poro;
        if (std.mem.eql(u8, s, "chatglm3")) return .chatglm3;
        if (std.mem.eql(u8, s, "chatglm4")) return .chatglm4;
        if (std.mem.eql(u8, s, "viking")) return .viking;
        if (std.mem.eql(u8, s, "jais")) return .jais;
        if (std.mem.eql(u8, s, "tekken")) return .tekken;
        if (std.mem.eql(u8, s, "smollm")) return .smollm;
        if (std.mem.eql(u8, s, "codeshell")) return .codeshell;
        if (std.mem.eql(u8, s, "bloom")) return .bloom;
        if (std.mem.eql(u8, s, "gpt3-finnish")) return .gpt3_finnish;
        if (std.mem.eql(u8, s, "exaone")) return .exaone;
        if (std.mem.eql(u8, s, "chameleon")) return .chameleon;
        if (std.mem.eql(u8, s, "minerva")) return .minerva;
        if (std.mem.eql(u8, s, "gpt4o")) return .gpt4o;
        if (std.mem.eql(u8, s, "super-bpe")) return .super_bpe;
        if (std.mem.eql(u8, s, "trillion")) return .trillion;
        if (std.mem.eql(u8, s, "bailingmoe")) return .bailingmoe;
        if (std.mem.eql(u8, s, "llama4")) return .llama4;
        if (std.mem.eql(u8, s, "seed-coder")) return .seed_coder;
        if (std.mem.eql(u8, s, "hunyuan")) return .hunyuan;
        if (std.mem.eql(u8, s, "kimi-k2")) return .kimi_k2;
        if (std.mem.eql(u8, s, "hunyuan-dense")) return .hunyuan_dense;
        if (std.mem.eql(u8, s, "grok-2")) return .grok_2;
        if (std.mem.eql(u8, s, "granite-docling")) return .granite_docling;
        if (std.mem.eql(u8, s, "minimax-m2")) return .minimax_m2;
        if (std.mem.eql(u8, s, "afmoe")) return .afmoe;
        if (std.mem.eql(u8, s, "solar-open")) return .solar_open;
        if (std.mem.eql(u8, s, "youtu")) return .youtu;
        if (std.mem.eql(u8, s, "exaone-moe")) return .exaone_moe;
        if (std.mem.eql(u8, s, "qwen35")) return .qwen35;
        if (std.mem.eql(u8, s, "tiny-aya")) return .tiny_aya;
        if (std.mem.eql(u8, s, "joyai-llm")) return .joyai_llm;
        if (std.mem.eql(u8, s, "jais2")) return .jais2;
        if (std.mem.eql(u8, s, "gemma4")) return .gemma4;
        if (std.mem.eql(u8, s, "bert-bge")) return .gpt2;
        if (std.mem.eql(u8, s, "sarvam-moe")) return .sarvam_moe;
        if (std.mem.eql(u8, s, "minicpm5")) return .minicpm5;
        return .default;
    }

    pub fn toString(self: PreType) []const u8 {
        return @tagName(self);
    }
};

// ============================================================================
// Token 数据结构
// ============================================================================

/// 单个 token 的完整信息，对应 llama.cpp 的 llama_vocab::token_data
pub const TokenData = struct {
    /// token 文本（由 Vocab 拥有）
    text: []const u8,
    /// token 分数（SPM/Unigram 使用，BPE 为 0）
    score: f32,
    /// token 类型属性
    type: TokenType,
};

// ============================================================================
// 特殊 Token ID（从 GGUF metadata 解析）
// ============================================================================

pub const SpecialTokenIds = struct {
    bos: u32 = std.math.maxInt(u32),
    eos: u32 = std.math.maxInt(u32),
    unk: u32 = std.math.maxInt(u32),
    pad: u32 = std.math.maxInt(u32),
    sep: u32 = std.math.maxInt(u32),
    cls: u32 = std.math.maxInt(u32),
    mask: u32 = std.math.maxInt(u32),
    eot: u32 = std.math.maxInt(u32),
    eom: u32 = std.math.maxInt(u32),
    /// 换行 token ID（如果有）
    nl: u32 = std.math.maxInt(u32),

    pub fn fromGGUF(gguf_file: *const gguf.GGUFFile) SpecialTokenIds {
        return SpecialTokenIds{
            .bos = gguf_file.getU32("tokenizer.ggml.bos_token_id") orelse std.math.maxInt(u32),
            .eos = gguf_file.getU32("tokenizer.ggml.eos_token_id") orelse std.math.maxInt(u32),
            .unk = gguf_file.getU32("tokenizer.ggml.unknown_token_id") orelse std.math.maxInt(u32),
            .pad = gguf_file.getU32("tokenizer.ggml.padding_token_id") orelse std.math.maxInt(u32),
            .sep = gguf_file.getU32("tokenizer.ggml.sep_token_id") orelse std.math.maxInt(u32),
            .cls = gguf_file.getU32("tokenizer.ggml.cls_token_id") orelse std.math.maxInt(u32),
            .mask = gguf_file.getU32("tokenizer.ggml.mask_token_id") orelse std.math.maxInt(u32),
            .eot = gguf_file.getU32("tokenizer.ggml.eot_token_id") orelse std.math.maxInt(u32),
            .eom = gguf_file.getU32("tokenizer.ggml.eom_token_id") orelse std.math.maxInt(u32),
            .nl = gguf_file.getU32("tokenizer.ggml.nl_token_id") orelse std.math.maxInt(u32),
        };
    }

    /// 返回是否已设置（值不是 maxInt）
    pub fn has(self: SpecialTokenIds, field: enum { bos, eos, unk, pad, sep, cls, mask, eot, eom, nl }) bool {
        const val: u32 = switch (field) {
            .bos => self.bos,
            .eos => self.eos,
            .unk => self.unk,
            .pad => self.pad,
            .sep => self.sep,
            .cls => self.cls,
            .mask => self.mask,
            .eot => self.eot,
            .eom => self.eom,
            .nl => self.nl,
        };
        return val != std.math.maxInt(u32);
    }
};

// ============================================================================
// Vocab 结构体
// ============================================================================

/// 词表，对齐 llama.cpp 的 llama_vocab
///
/// 持有从 GGUF metadata 解析的所有词表信息：
///   - token 文本、分数、类型
///   - BPE 合并规则
///   - 特殊 token ID
///   - 分词器配置
///   - EOG token 集合
///   - 双向查找结构（token↔text, byte↔token）
pub const Vocab = struct {
    allocator: std.mem.Allocator,

    // ---- 分词器标识 ----
    /// 分词器算法类型（llama/gpt2/tiktoken/replit/spm）
    type: VocabType,
    /// 预分词器变体
    pre_type: PreType,

    // ---- Token 数据 ----
    /// token 列表，索引即 token ID
    tokens: []const TokenData,

    /// BPE 合并规则表：key="left right", value=rank
    /// 注意：key 由 Vocab 拥有，deinit 时释放
    merges: std.StringHashMap(u32),

    // ---- 特殊 token ----
    special: SpecialTokenIds = .{},

    // ---- 配置标志 ----
    add_space_prefix: bool = false,
    add_bos: bool = false,
    add_eos: bool = false,
    add_sep: bool = false,
    ignore_merges: bool = false,
    clean_spaces: bool = false,
    remove_extra_whitespaces: bool = false,
    escape_whitespaces: bool = true,
    treat_whitespace_as_suffix: bool = false,

    // ---- 双向查找结构 ----
    /// text → token_id 映射（用于 BPE 快速查找）
    text_to_token: std.StringHashMapUnmanaged(u32) = .{},
    /// byte → token_id 映射（用于字节回退）
    byte_to_token: [256]u32 = [_]u32{std.math.maxInt(u32)} ** 256,

    // ---- EOG token 集合 ----
    /// End-of-Generation token ID 集合
    eog_ids: std.AutoArrayHashMapUnmanaged(u32, void) = .{},

    // ========================================================================
    // 从 GGUF 初始化
    // ========================================================================

    /// 从 GGUF 文件解析并初始化 Vocab
    pub fn init(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Vocab {
        const vocab_type_str = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";
        const vocab_type = VocabType.fromString(vocab_type_str);
        const pre_type_str = gguf_file.getString("tokenizer.ggml.pre") orelse "";
        var pre_type = PreType.fromString(pre_type_str);

        // 如果 pre_type 是 default 但 vocab type 暗示了特定的 pre_type，使用它
        if (pre_type == .default) {
            if (vocab_type == .gemma4) pre_type = .gemma4;
        }

        log.info("Vocab type: '{s}' → {s}, pre: '{s}' → {s}", .{
            vocab_type_str, vocab_type.toString(),
            pre_type_str, pre_type.toString(),
        });

        // 读取 token 字符串数组
        const token_strings = blk: {
            const val = gguf_file.metadata.get("tokenizer.ggml.tokens") orelse return error.MissingVocabTokens;
            const arr = val.array_val;
            var list = try std.ArrayList([]const u8).initCapacity(allocator, arr.len);
            errdefer list.deinit(allocator);
            for (arr) |item| {
                const s = item.asString() orelse return error.InvalidVocabToken;
                const owned = try allocator.dupe(u8, s);
                try list.append(allocator, owned);
            }
            break :blk try list.toOwnedSlice(allocator);
        };
        const n_tokens = token_strings.len;

        // 读取 token 类型
        var token_types = try allocator.alloc(TokenType, n_tokens);
        @memset(token_types, .normal);
        if (gguf_file.metadata.get("tokenizer.ggml.token_type")) |val| {
            const arr = val.array_val;
            for (arr, 0..) |item, i| {
                if (i >= n_tokens) break;
                const tv = item.asU32() orelse 0;
                token_types[i] = @enumFromInt(@min(tv, @intFromEnum(TokenType.byte)));
            }
        }

        // 读取 token 分数
        var token_scores = try allocator.alloc(f32, n_tokens);
        @memset(token_scores, 0.0);
        if (gguf_file.metadata.get("tokenizer.ggml.scores")) |val| {
            const arr = val.array_val;
            for (arr, 0..) |item, i| {
                if (i >= n_tokens) break;
                token_scores[i] = item.asF32() orelse 0.0;
            }
        }

        // 构建 TokenData 数组
        var tokens = try allocator.alloc(TokenData, n_tokens);
        errdefer {
            for (tokens) |td| allocator.free(td.text);
            allocator.free(tokens);
        }
        for (token_strings, token_types, token_scores, 0..) |text, tt, score, i| {
            tokens[i] = TokenData{
                .text = text,
                .score = score,
                .type = tt,
            };
        }

        // 释放中间数组：数据已转移到 TokenData
        // - token_strings: 释放指针数组，字符串本身已由 tokens[].text 持有
        // - token_types: 释放类型数组，值已由 tokens[].type 持有
        // - token_scores: 释放分数数组，值已由 tokens[].score 持有
        allocator.free(token_strings);
        allocator.free(token_types);
        allocator.free(token_scores);

        // 特殊 token ID
        const special = SpecialTokenIds.fromGGUF(gguf_file);

        // 读取配置
        var vocab = Vocab{
            .allocator = allocator,
            .type = vocab_type,
            .pre_type = pre_type,
            .tokens = tokens,
            .merges = std.StringHashMap(u32).init(allocator),
            .special = special,
        };

        // 根据 vocab type 设置默认配置
        vocab.initDefaults();

        // 从 GGUF 覆盖配置
        vocab.overrideFromGGUF(gguf_file);

        // 构建双向查找结构
        try vocab.buildLookups(allocator);

        // 读取 BPE 合并规则
        try vocab.loadMerges(gguf_file, allocator);

        // 构建 EOG 集合
        try vocab.buildEogSet(allocator);

        log.info("Vocab: {d} tokens, type={s}, pre={s}, merges={d}", .{
            n_tokens, vocab_type.toString(), pre_type.toString(), vocab.merges.count(),
        });

        return vocab;
    }

    // ========================================================================
    // 资源管理
    // ========================================================================

    pub fn deinit(self: *Vocab) void {
        // 释放 token 文本
        for (self.tokens) |td| {
            self.allocator.free(td.text);
        }
        self.allocator.free(self.tokens);

        // 释放 text→token 映射的 key
        var tt_iter = self.text_to_token.keyIterator();
        while (tt_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.text_to_token.deinit(self.allocator);

        // 释放 BPE merge 规则的 key
        {
            var m_iter = self.merges.keyIterator();
            while (m_iter.next()) |key| {
                self.allocator.free(key.*);
            }
        }
        self.merges.deinit();
        self.eog_ids.deinit(self.allocator);

        self.* = undefined;
    }

    // ========================================================================
    // Token 类型判定
    // ========================================================================

    pub fn nTokens(self: *const Vocab) usize {
        return self.tokens.len;
    }

    pub fn isValidToken(self: *const Vocab, id: u32) bool {
        return id < self.tokens.len;
    }

    pub fn getTokenData(self: *const Vocab, id: u32) ?TokenData {
        if (id >= self.tokens.len) return null;
        return self.tokens[id];
    }

    pub fn tokenText(self: *const Vocab, id: u32) ?[]const u8 {
        if (id >= self.tokens.len) return null;
        return self.tokens[id].text;
    }

    pub fn tokenScore(self: *const Vocab, id: u32) f32 {
        if (id >= self.tokens.len) return 0.0;
        return self.tokens[id].score;
    }

    pub fn tokenType(self: *const Vocab, id: u32) TokenType {
        if (id >= self.tokens.len) return .undefined;
        return self.tokens[id].type;
    }

    pub fn tokenAttr(self: *const Vocab, id: u32) TokenAttr {
        return TokenAttr.fromType(self.tokenType(id));
    }

    pub fn isNormal(self: *const Vocab, id: u32) bool {
        return self.tokenType(id) == .normal;
    }

    pub fn isUnknown(self: *const Vocab, id: u32) bool {
        return self.tokenType(id) == .unknown;
    }

    pub fn isControl(self: *const Vocab, id: u32) bool {
        return self.tokenType(id) == .control;
    }

    pub fn isByte(self: *const Vocab, id: u32) bool {
        return self.tokenType(id) == .byte;
    }

    pub fn isUserDefined(self: *const Vocab, id: u32) bool {
        return self.tokenType(id) == .user_defined;
    }

    pub fn isUnused(self: *const Vocab, id: u32) bool {
        return self.tokenType(id) == .unused;
    }

    /// 是否为特殊 token（BOS/EOS/UNK/PAD/SEP/CLS/MASK/CONTROL）
    pub fn isSpecial(self: *const Vocab, id: u32) bool {
        if (id == self.special.bos or id == self.special.eos or
            id == self.special.pad or id == self.special.unk or
            id == self.special.sep or id == self.special.cls or
            id == self.special.mask) return true;
        return self.isControl(id);
    }

    /// 是否为 EOG (End-of-Generation) token
    pub fn isEog(self: *const Vocab, id: u32) bool {
        return self.eog_ids.contains(id);
    }

    // ========================================================================
    // 双向查找
    // ========================================================================

    /// 通过文本查找 token ID
    pub fn textToToken(self: *const Vocab, text: []const u8) ?u32 {
        return self.text_to_token.get(text);
    }

    /// 通过字节值查找 token ID
    pub fn byteToToken(self: *const Vocab, byte: u8) u32 {
        const id = self.byte_to_token[byte];
        if (id == std.math.maxInt(u32)) return self.special.unk;
        return id;
    }

    /// 通过 token ID 获取字节值（仅 byte 类型 token）
    pub fn tokenToByte(self: *const Vocab, id: u32) ?u8 {
        if (id >= self.tokens.len) return null;
        if (self.tokens[id].type != .byte) return null;
        return self.tokens[id].text[0];
    }

    // ========================================================================
    // BPE 合并规则查找
    // ========================================================================

    /// 查找 BPE 合并规则的 rank（left right → rank）
    pub fn findBpeRank(self: *const Vocab, left: []const u8, right: []const u8) ?u32 {
        // 在栈上构建临时 key（避免堆分配）
        var buf: [512]u8 = undefined;
        if (left.len + 1 + right.len <= buf.len) {
            @memcpy(buf[0..left.len], left);
            buf[left.len] = ' ';
            @memcpy(buf[left.len + 1 .. left.len + 1 + right.len], right);
            return self.merges.get(buf[0 .. left.len + 1 + right.len]);
        }
        // 长 token 回退到分配
        const key = std.fmt.allocPrint(self.allocator, "{s} {s}", .{ left, right }) catch return null;
        defer self.allocator.free(key);
        return self.merges.get(key);
    }

    // ========================================================================
    // 特殊 token ID 访问器
    // ========================================================================

    pub fn tokenBos(self: *const Vocab) u32 { return self.special.bos; }
    pub fn tokenEos(self: *const Vocab) u32 { return self.special.eos; }
    pub fn tokenUnk(self: *const Vocab) u32 { return self.special.unk; }
    pub fn tokenPad(self: *const Vocab) u32 { return self.special.pad; }
    pub fn tokenSep(self: *const Vocab) u32 { return self.special.sep; }
    pub fn tokenCls(self: *const Vocab) u32 { return self.special.cls; }
    pub fn tokenMask(self: *const Vocab) u32 { return self.special.mask; }
    pub fn tokenEot(self: *const Vocab) u32 { return self.special.eot; }
    pub fn tokenEom(self: *const Vocab) u32 { return self.special.eom; }
    pub fn tokenNl(self: *const Vocab) u32 { return self.special.nl; }

    // ========================================================================
    // 配置访问器
    // ========================================================================

    pub fn getAddSpacePrefix(self: *const Vocab) bool { return self.add_space_prefix; }
    pub fn getAddBos(self: *const Vocab) bool { return self.add_bos; }
    pub fn getAddEos(self: *const Vocab) bool { return self.add_eos; }
    pub fn getAddSep(self: *const Vocab) bool { return self.add_sep; }
    pub fn getIgnoreMerges(self: *const Vocab) bool { return self.ignore_merges; }
    pub fn getCleanSpaces(self: *const Vocab) bool { return self.clean_spaces; }
    pub fn getRemoveExtraWhitespaces(self: *const Vocab) bool { return self.remove_extra_whitespaces; }
    pub fn getEscapeWhitespaces(self: *const Vocab) bool { return self.escape_whitespaces; }
    pub fn getTreatWhitespaceAsSuffix(self: *const Vocab) bool { return self.treat_whitespace_as_suffix; }
    pub fn getType(self: *const Vocab) VocabType { return self.type; }
    pub fn getPreType(self: *const Vocab) PreType { return self.pre_type; }
    pub fn getSpecial(self: *const Vocab) SpecialTokenIds { return self.special; }

    // ========================================================================
    // 内部：初始化
    // ========================================================================

    /// 根据 vocab type 和 pre_type 设置默认配置
    fn initDefaults(self: *Vocab) void {
        switch (self.type) {
            .llama, .spm => {
                self.add_space_prefix = true;
                self.clean_spaces = false;
                self.add_bos = true;
                self.add_eos = false;
                self.escape_whitespaces = true;
                self.ignore_merges = false;
            },
            .gpt2 => {
                self.add_space_prefix = false;
                self.escape_whitespaces = false;
                self.ignore_merges = false;
                self.add_bos = false;
                switch (self.pre_type) {
                    .llama3 => {
                        self.ignore_merges = true;
                        self.add_bos = true;
                    },
                    .deepseek_llm, .deepseek_coder, .deepseek3_llm, .youtu => {
                        self.clean_spaces = false;
                    },
                    else => {},
                }
            },
            .gemma4 => {
                self.add_space_prefix = false;
                self.escape_whitespaces = true;  // gemma-4 使用 ▁ (U+2581) 代替空格
                self.ignore_merges = false;
                self.add_bos = false;
            },
            .bert => {
                self.add_space_prefix = false;
                self.escape_whitespaces = false;
                self.ignore_merges = false;
                self.add_bos = false;
            },
            .tiktoken => {
                self.add_space_prefix = false;
                self.escape_whitespaces = false;
                self.clean_spaces = true;
                self.ignore_merges = true;
            },
            .replit => {
                self.add_space_prefix = false;
                self.escape_whitespaces = false;
                self.clean_spaces = true;
            },
            .unknown => {},
        }
    }

    /// 从 GGUF 元数据覆盖配置值
    fn overrideFromGGUF(self: *Vocab, gguf_file: *const gguf.GGUFFile) void {
        if (gguf_file.getBool("tokenizer.ggml.add_bos")) |v| self.add_bos = v;
        if (gguf_file.getBool("tokenizer.ggml.add_eos")) |v| self.add_eos = v;
        if (gguf_file.getBool("tokenizer.ggml.add_sep")) |v| self.add_sep = v;
        if (gguf_file.getBool("tokenizer.ggml.add_space_prefix")) |v| {
            self.add_space_prefix = v;
        }
    }

    /// 构建 text→token 和 byte→token 双向查找
    fn buildLookups(self: *Vocab, allocator: std.mem.Allocator) !void {
        // 初始化 byte→token 为 unk
        @memset(&self.byte_to_token, self.special.unk);

        // 第一遍：处理所有 token 的 text→token 映射和普通 byte→token 映射
        // 跳过 <0xXX> 格式的字节 token（留给第二遍处理，确保不被单字节 normal token 覆盖）
        for (self.tokens, 0..) |td, id| {
            const uid = @as(u32, @intCast(id));

            // text→token 映射
            // 添加所有类型的 token，包括 byte、unknown、user_defined、unused 等。
            // 对于 BPE 模型，合并后的 token 可能是 unknown 或 unused 类型，
            // 必须添加到 HashMap 中才能被 BPE 合并过程找到。
            // 这与 llama.cpp 的行为一致，llama.cpp 的 vocab 查找不限制 token 类型。
            {
                const key = try allocator.dupe(u8, td.text);
                try self.text_to_token.put(allocator, key, uid);
            }

            // byte→token 映射（跳过 <0xXX> 格式，留给第二遍）
            // 注意：对于 GPT-2 风格的 BPE 模型（gpt2/tiktoken/replit），
            // 字节 token 的文本是 GPT-2 编码后的字符（多字节），而不是原始字节。
            // 因此不能通过 td.text.len == 1 来识别字节 token。
            // byte→token 映射由 fixByteToTokenMapping 通过 GPT-2 编码反向映射建立。
            const is_gpt2_bpe = self.type == .gpt2 or self.type == .tiktoken or self.type == .replit;
            const is_hex_byte = td.text.len == 6 and td.text[0] == '<' and td.text[1] == '0' and td.text[2] == 'x' and td.text[5] == '>';
            if (!is_hex_byte and !is_gpt2_bpe) {
                if (td.type == .byte or (td.type == .normal and td.text.len == 1)) {
                    if (td.text.len == 1) {
                        self.byte_to_token[td.text[0]] = uid;
                    }
                }
            }
        }

        // 第二遍：处理 <0xXX> 格式的字节 token（如 gemma-4）
        // 必须放在最后，确保不被单字节 normal token 覆盖
        for (self.tokens, 0..) |td, id| {
            if (td.text.len == 6 and td.text[0] == '<' and td.text[1] == '0' and td.text[2] == 'x' and td.text[5] == '>') {
                const hex_str = td.text[3..5];
                const byte_val = std.fmt.parseInt(u8, hex_str, 16) catch continue;
                self.byte_to_token[byte_val] = @as(u32, @intCast(id));
            }
        }
    }
    /// 加载 BPE 合并规则
    fn loadMerges(self: *Vocab, gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !void {
        const val = gguf_file.metadata.get("tokenizer.ggml.merges") orelse return;
        const arr = val.array_val;

        try self.merges.ensureTotalCapacity(@intCast(arr.len));
        for (arr, 0..) |item, rank| {
            const merge_str = item.asString() orelse continue;
            const owned_key = try allocator.dupe(u8, merge_str);
            self.merges.putAssumeCapacity(owned_key, @intCast(rank));
        }
        log.info("Vocab: {d} BPE merge rules loaded", .{arr.len});
    }

    /// 构建 EOG (End-of-Generation) token ID 集合
    fn buildEogSet(self: *Vocab, allocator: std.mem.Allocator) !void {
        try self.eog_ids.ensureTotalCapacity(allocator, 32);

        if (self.special.eos != std.math.maxInt(u32)) {
            self.eog_ids.putAssumeCapacity(self.special.eos, {});
        }
        if (self.special.eot != std.math.maxInt(u32)) {
            self.eog_ids.putAssumeCapacity(self.special.eot, {});
        }
        if (self.special.eom != std.math.maxInt(u32)) {
            self.eog_ids.putAssumeCapacity(self.special.eom, {});
        }

        // 通过名称匹配收集 EOG tokens
        const eog_names = [_][]const u8{
            "<|endoftext|>", "<|im_end|>", "<|im_start|>",
            "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
            "<|eot_id|>", "<|end|>", "<|END|>", "<EOS>", "<EOT>",
            "<end_of_text>", "<|end_of_text|>", "<end_of_utterance>",
            "<eos>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
            "<end_of_turn>", "</s>", "<|eom_id|>", "[EOT]", "[EOS]",
            "<|tool_response>", "<｜end▁of▁sentence｜>",
        };

        for (self.tokens, 0..) |td, id| {
            for (eog_names) |name| {
                if (std.mem.eql(u8, td.text, name)) {
                    const uid = @as(u32, @intCast(id));
                    if (!self.eog_ids.contains(uid)) {
                        self.eog_ids.putAssumeCapacity(uid, {});
                    }
                    break;
                }
            }
        }

        log.info("Vocab: {d} EOG tokens", .{self.eog_ids.count()});
    }
};

// ============================================================================
// 测试
// ============================================================================

test "VocabType fromString" {
    const testing = std.testing;
    try testing.expectEqual(VocabType.llama, VocabType.fromString("llama"));
    try testing.expectEqual(VocabType.gpt2, VocabType.fromString("gpt2"));
    try testing.expectEqual(VocabType.gemma4, VocabType.fromString("gemma4"));
    try testing.expectEqual(VocabType.unknown, VocabType.fromString("nonexistent"));
}

test "VocabType isSPM" {
    const testing = std.testing;
    try testing.expect(VocabType.llama.isSPM());
    try testing.expect(VocabType.spm.isSPM());
    try testing.expect(!VocabType.gpt2.isSPM());
}

test "PreType fromString" {
    const testing = std.testing;
    try testing.expectEqual(PreType.default, PreType.fromString("default"));
    try testing.expectEqual(PreType.llama3, PreType.fromString("llama3"));
    try testing.expectEqual(PreType.qwen2, PreType.fromString("qwen2"));
    try testing.expectEqual(PreType.llama3, PreType.fromString("llama-bpe"));
    try testing.expectEqual(PreType.qwen2, PreType.fromString("qwen2.5"));
}

test "SpecialTokenIds defaults" {
    const st = SpecialTokenIds{};
    try std.testing.expectEqual(std.math.maxInt(u32), st.bos);
    try std.testing.expectEqual(std.math.maxInt(u32), st.eos);
}
