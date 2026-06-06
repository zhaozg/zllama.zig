//! 分词器类型定义
//!
//! 定义分词器相关的枚举、结构体和常量。
//! 参考 llama.cpp 的 llama_vocab.h 和 llama.h。

const std = @import("std");
const gguf = @import("gguf");

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

/// LLAMA_TOKEN_ATTR 位标志（与 llama.cpp 保持一致）
pub const TokenAttr = packed struct(u32) {
    normal: bool = false,
    unknown: bool = false,
    control: bool = false,
    user_defined: bool = false,
    unused: bool = false,
    byte: bool = false,
    _padding: u26 = 0,
};

/// 从 TokenType 转换为 TokenAttr 位标志
pub fn tokenTypeToAttr(tt: TokenType) TokenAttr {
    return switch (tt) {
        .normal => TokenAttr{ .normal = true },
        .unknown => TokenAttr{ .unknown = true },
        .control => TokenAttr{ .control = true },
        .user_defined => TokenAttr{ .user_defined = true },
        .unused => TokenAttr{ .unused = true },
        .byte => TokenAttr{ .byte = true },
        else => TokenAttr{},
    };
}

// ============================================================================
// 分词器模型类型
// ============================================================================

/// 分词器模型类型，决定字节 token 的表示格式和编码/解码策略
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

    pub fn toString(self: TokenizerModel) []const u8 {
        return switch (self) {
            .llama => "llama",
            .gpt2 => "gpt2",
            .tiktoken => "tiktoken",
            .replit => "replit",
            .unknown => "unknown",
        };
    }
};

// ============================================================================
// 预分词器类型（与 llama.cpp 的 llama_vocab_pre_type 对应）
// ============================================================================

/// 预分词器类型，决定编码前的文本分割策略
pub const PreTokenizerType = enum {
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

    pub fn fromString(s: []const u8) PreTokenizerType {
        // Zig 0.16.0 没有 ComptimeStringMap，使用手动匹配
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
        if (std.mem.eql(u8, s, "gpt2")) return .gpt2;
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
        if (std.mem.eql(u8, s, "sarvam-moe")) return .sarvam_moe;
        if (std.mem.eql(u8, s, "minicpm5")) return .minicpm5;
        return .default;
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
    eot: u32 = 0,  // End-of-Turn token (e.g. <|eot_id|>)
    eom: u32 = 0,  // End-of-Message token (e.g. <|eom_id|>)

    pub fn fromGGUF(gguf_file: *const gguf.GGUFFile) SpecialTokens {
        return SpecialTokens{
            .bos = gguf_file.getU32("tokenizer.ggml.bos_token_id") orelse 1,
            .eos = gguf_file.getU32("tokenizer.ggml.eos_token_id") orelse 2,
            .unk = gguf_file.getU32("tokenizer.ggml.unknown_token_id") orelse 0,
            .pad = gguf_file.getU32("tokenizer.ggml.padding_token_id") orelse 0,
            .sep = gguf_file.getU32("tokenizer.ggml.sep_token_id") orelse 2,
            .cls = gguf_file.getU32("tokenizer.ggml.cls_token_id") orelse 1,
            .mask = gguf_file.getU32("tokenizer.ggml.mask_token_id") orelse 0,
            .eot = gguf_file.getU32("tokenizer.ggml.eot_token_id") orelse 0,
            .eom = gguf_file.getU32("tokenizer.ggml.eom_token_id") orelse 0,
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
// 分词器配置
// ============================================================================

/// 分词器配置，从 GGUF 元数据读取
pub const TokenizerConfig = struct {
    /// 分词器模型类型
    model: TokenizerModel = .unknown,
    /// 预分词器类型
    pre_type: PreTokenizerType = .default,
    /// 是否在编码前添加空格前缀（SPM 模型需要）
    add_space_prefix: bool = false,
    /// 是否在编码时添加 BOS token
    add_bos: bool = false,
    /// 是否在编码时添加 EOS token
    add_eos: bool = false,
    /// 是否添加 SEP token
    add_sep: bool = false,
    /// 是否忽略 BPE 合并规则（LLaMA 3 等使用）
    ignore_merges: bool = false,
    /// 是否在解码后清理空格（移除标点前的空格）
    clean_spaces: bool = false,
    /// 是否移除多余的空白
    remove_extra_whitespaces: bool = false,
    /// 是否转义空白（SPM-style BPE 使用 ▁ 表示空格）
    escape_whitespaces: bool = true,
    /// 是否将空白视为后缀
    treat_whitespace_as_suffix: bool = false,

    /// 从 GGUF 元数据读取分词器配置
    pub fn fromGGUF(gguf_file: *const gguf.GGUFFile) TokenizerConfig {
        const model_raw = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";
        const model = TokenizerModel.fromString(model_raw);
        const pre_raw = gguf_file.getString("tokenizer.ggml.pre") orelse "";
        const pre_type = PreTokenizerType.fromString(pre_raw);

        var config = TokenizerConfig{
            .model = model,
            .pre_type = pre_type,
        };

        // 根据模型类型设置默认值（与 llama.cpp 保持一致）
        switch (model) {
            .llama => {
                // SPM 模型
                config.add_space_prefix = true;
                config.clean_spaces = false;
                config.add_bos = true;
                config.add_eos = false;
                config.escape_whitespaces = true;
            },
            .gpt2 => {
                // BPE 模型
                config.add_space_prefix = false;
                config.escape_whitespaces = false;
                switch (pre_type) {
                    .llama3 => {
                        config.ignore_merges = true;
                        config.add_bos = true;
                    },
                    .qwen2, .qwen35 => {
                        // Qwen2/Qwen35 使用 GPT-2 BPE，需要 BPE 合并
                        // 在 llama.cpp 中，Qwen2/Qwen35 的 ignore_merges 为 false（BPE 默认值）
                        // 且 add_bos 为 false（BPE 默认值）
                        config.ignore_merges = false;
                        config.add_bos = false;
                    },
                    .deepseek_llm, .deepseek_coder, .deepseek3_llm, .youtu => {
                        config.clean_spaces = false;
                    },
                    else => {},
                }
            },
            .tiktoken => {
                config.add_space_prefix = false;
                config.escape_whitespaces = false;
                config.clean_spaces = true;
                config.ignore_merges = true;
            },
            .replit => {
                config.add_space_prefix = false;
                config.escape_whitespaces = false;
                config.clean_spaces = true;
            },
            .unknown => {},
        }

        // 从 GGUF 元数据覆盖默认值
        if (gguf_file.getBool("tokenizer.ggml.add_space_prefix")) |v| config.add_space_prefix = v;
        if (gguf_file.getBool("tokenizer.ggml.add_bos")) |v| config.add_bos = v;
        if (gguf_file.getBool("tokenizer.ggml.add_eos")) |v| config.add_eos = v;
        if (gguf_file.getBool("tokenizer.ggml.add_sep")) |v| config.add_sep = v;

        return config;
    }
};
