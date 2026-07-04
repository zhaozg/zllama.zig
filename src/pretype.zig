//! 分词器预处理类型枚举
//! 从 vocab.zig 拆分，减少主文件行数。
const std = @import("std");

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

