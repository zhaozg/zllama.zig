//! Vocab 内部辅助函数（从 vocab.zig 提取）
const std = @import("std");
const gguf = @import("gguf");

const Vocab = @import("./vocab.zig").Vocab;

const log = std.log.scoped(.tokenizer_vocab);

// ========================================================================
// 内部：初始化
// ========================================================================

/// 根据 vocab type 和 pre_type 设置默认配置
pub fn initDefaults(self: *Vocab) void {
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
            self.escape_whitespaces = true; // gemma-4 使用 ▁ (U+2581) 代替空格
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
pub fn overrideFromGGUF(self: *Vocab, gguf_file: *const gguf.GGUFFile) void {
    if (gguf_file.getBool("tokenizer.ggml.add_bos")) |v| self.add_bos = v;
    if (gguf_file.getBool("tokenizer.ggml.add_eos")) |v| self.add_eos = v;
    if (gguf_file.getBool("tokenizer.ggml.add_sep")) |v| self.add_sep = v;
    if (gguf_file.getBool("tokenizer.ggml.add_space_prefix")) |v| {
        self.add_space_prefix = v;
    }
}

/// 构建 text→token 和 byte→token 双向查找
pub fn buildLookups(self: *Vocab, allocator: std.mem.Allocator) !void {
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
pub fn loadMerges(self: *Vocab, gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !void {
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
pub fn buildEogSet(self: *Vocab, allocator: std.mem.Allocator) !void {
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
        "<|endoftext|>",    "<|im_end|>",         "<|im_start|>",
        "<|fim_pad|>",      "<|repo_name|>",      "<|file_sep|>",
        "<|eot_id|>",       "<|end|>",            "<|END|>",
        "<EOS>",            "<EOT>",              "<end_of_text>",
        "<|end_of_text|>",  "<end_of_utterance>", "<eos>",
        "<|return|>",       "<|call|>",           "<|flush|>",
        "<|calls|>",        "<end_of_turn>",      "</s>",
        "<|eom_id|>",       "[EOT]",              "[EOS]",
        "<|tool_response>",
        "<｜end▁of▁sentence｜>",
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
