//! 模型抽象接口
//!
//! 定义 ModelInterface 接口契约，支持多模型架构的零成本抽象。
//! 所有模型实现必须遵循此接口。
//!
//! 设计原则：
//! - 注册表使用虚表模式（ModelVTable）实现运行时多态
//! - 模型内部使用编译时多态（comptime）实现零成本抽象
//! - 接口最小化：只暴露必要的生命周期和前向方法

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const kv_cache = @import("kv_cache");

// 导入并重新导出模型实现
pub const qwen2 = @import("models/qwen2.zig");
pub const qwen35 = @import("models/qwen35.zig");
pub const qwen3vl = @import("models/qwen3vl.zig");
pub const llama = @import("models/llama.zig");
pub const gemma3 = @import("models/gemma3.zig");
pub const gemma4 = @import("models/gemma4.zig");
pub const embedding = @import("models/embedding.zig");

pub const gemma4_graph = @import("models/gemma4_graph.zig");

pub const RopeScaling = struct {
    rope_type: []const u8 = "",
    factor: f32 = 1.0,
    original_max_seq_len: u32 = 32768,
};

/// 支持的模型架构枚举
pub const Architecture = enum {
    qwen2,
    qwen35,
    qwen3vl,
    llama,
    gemma3,
    gemma4,
    embedding_qwen2,

    /// 从 GGUF 元数据中的 general.architecture 字段解析
    pub fn fromString(s: []const u8) ?Architecture {
        if (std.mem.eql(u8, s, "qwen2") or
            std.mem.eql(u8, s, "qwen2.5") or
            std.mem.eql(u8, s, "qwen3"))
        {
            return .qwen2;
        }
        if (std.mem.eql(u8, s, "qwen3.5") or
            std.mem.eql(u8, s, "qwen35"))
        {
            return .qwen35;
        }
        if (std.mem.eql(u8, s, "qwen3vl")) {
            return .qwen3vl;
        }
        if (std.mem.eql(u8, s, "llama") or
            std.mem.eql(u8, s, "llama2") or
            std.mem.eql(u8, s, "llama3"))
        {
            return .llama;
        }
        if (std.mem.eql(u8, s, "gemma3")) {
            return .gemma3;
        }
        if (std.mem.eql(u8, s, "gemma4")) {
            return .gemma4;
        }
        if (std.mem.eql(u8, s, "bert") or
            std.mem.eql(u8, s, "nomic-bert") or
            std.mem.eql(u8, s, "embedding"))
        {
            return .embedding_qwen2;
        }
        return null;
    }
};

/// 模型超参数
/// 所有模型共享的通用参数
pub const ModelParams = struct {
    // 基础维度
    n_vocab: u32 = 0,
    n_embd: u32 = 0,
    n_head: u32 = 0,
    n_head_dim: u32 = 0,
    /// K/V 专用 head dim（Gemma 3 等模型使用 attention.key_length/value_length）
    n_head_dim_k: u32 = 0,
    n_head_dim_v: u32 = 0,
    n_kv_head: u32 = 0,
    n_layer: u32 = 0,
    n_ff: u32 = 0,
    n_expert: u32 = 0,
    n_expert_used: u32 = 0,

    // 位置编码
    max_seq_len: u32 = 32768,
    rope_theta: f32 = 10000000.0,
    rope_dim: u32 = 64,

    // 归一化
    norm_eps: f32 = 1e-6,

    // 模型名称（来自 GGUF general.name 元数据）
    model_name: []const u8 = "",
    // 分词器类型名称
    tokenizer_name: []const u8 = "",

    pub fn deinit(self: *ModelParams) void {
        _ = self;
    }
};

/// 模型虚表
///
/// 每个模型实现提供此虚表，注册表通过虚表调用模型方法。
/// 模型权重基础结构
/// 包含所有模型共享的权重张量
pub const ModelWeights = struct {
    params: ModelParams,
    /// Token 嵌入权重 [n_embd, n_vocab]
    token_embd: *ggml.Tensor,
    /// 输出投影权重（可选，部分模型共享 token_embd）
    output_weight: ?*ggml.Tensor,
    /// 输出层归一化权重
    output_norm_weight: *ggml.Tensor,
};

/// 模型实例（运行时多态）
pub const ModelInstance = struct {
    ptr: *anyopaque,
    vtable: *const ModelVTable,

    pub fn getParams(self: ModelInstance) *const ModelParams {
        return self.vtable.getParams(self.ptr);
    }

    pub fn buildGraph(self: ModelInstance, builder: *graph_builder.GraphBuilder, input: *ggml.Tensor, n_tokens: i32, cache: ?*anyopaque, pos: i32) !*ggml.Tensor {
        return self.vtable.buildGraph(self.ptr, builder, input, n_tokens, cache, pos);
    }

    pub fn resetSSMStates(self: ModelInstance) void {
        if (self.vtable.resetSSMStates) |reset| {
            reset(self.ptr);
        }
    }

    pub fn setKVCacheContext(self: ModelInstance, ctx: *ggml.Context) void {
        if (self.vtable.setKVCacheContext) |set| {
            set(self.ptr, ctx);
        }
    }

    pub fn getPerLayerMaxSeqLen(self: ModelInstance, allocator: std.mem.Allocator) ?[]u32 {
        if (self.vtable.getPerLayerMaxSeqLen) |f| {
            return f(self.ptr, allocator);
        }
        return null;
    }

    pub fn deinit(self: ModelInstance, allocator: std.mem.Allocator) void {
        self.vtable.deinit(self.ptr, allocator);
    }

    /// Build a multimodal forward graph with embedding override.
    /// For models that support vision/audio (gemma4, qwen3vl), builds
    /// the LLM forward pass with pre-computed media embeddings substituted
    /// in place of placeholder tokens. Returns error.MMNotSupported for
    /// text-only models.
    pub fn buildMM(self: ModelInstance, ctx: *ggml.Context, graph: *ggml.CGraph, input_tokens: *ggml.Tensor, n_tokens: i32, cache: ?*anyopaque, pos: i32, embd_override: *ggml.Tensor, embd_offset: i32, causal: bool) !*ggml.Tensor {
        if (self.vtable.buildMM) |f| {
            return f(self.ptr, ctx, graph, input_tokens, n_tokens, cache, pos, embd_override, embd_offset, causal);
        }
        return error.MMNotSupported;
    }
};

/// 模型虚表定义
pub const ModelVTable = struct {
    getParams: *const fn (ptr: *anyopaque) *const ModelParams,
    buildGraph: *const fn (ptr: *anyopaque, builder: *graph_builder.GraphBuilder, input: *ggml.Tensor, n_tokens: i32, cache: ?*anyopaque, pos: i32) anyerror!*ggml.Tensor,
    resetSSMStates: ?*const fn (ptr: *anyopaque) void = null,
    setKVCacheContext: ?*const fn (ptr: *anyopaque, ctx: *ggml.Context) void = null,
    deinit: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator) void,
    /// 获取每层的最大序列长度（ISWA 模式）。
    /// 返回 null 表示所有层使用相同的 max_seq_len。
    /// 非 null 时，返回的切片长度必须等于 n_layer。
    /// 调用者负责释放返回的切片。
    getPerLayerMaxSeqLen: ?*const fn (ptr: *anyopaque, allocator: std.mem.Allocator) ?[]u32 = null,
    /// 构建多模态前向图（带嵌入覆盖）。
    /// 用于视觉/音频等媒体 token 的非因果注意力前向。
    /// 返回 null 或未设置表示该模型不支持多模态。
    buildMM: ?*const fn (ptr: *anyopaque, ctx: *ggml.Context, graph: *ggml.CGraph, input_tokens: *ggml.Tensor, n_tokens: i32, cache: ?*anyopaque, pos: i32, embd_override: *ggml.Tensor, embd_offset: i32, causal: bool) anyerror!*ggml.Tensor = null,
};

/// 模型架构相关的特殊 Token 标记
/// 动态配置，消除 chat_template 和 MtmdContext 中的硬编码。
pub const SpecialTokens = struct {
    /// 图像开始标记（如 `<|image>` `<|vision_start|>`）
    img_beg: []const u8 = "",
    /// 图像结束标记（如 `<image|>` `<|vision_end|>`）
    img_end: []const u8 = "",
    /// 音频开始标记（如 `<|audio>`）
    aud_beg: []const u8 = "",
    /// 音频结束标记（如 `<audio|>`）
    aud_end: []const u8 = "",
    /// 通用媒体占位符（默认 `<__media__>`）
    media_placeholder: []const u8 = "<__media__>",
};

/// 模型能力描述
/// 描述模型支持哪些输入模态以及对应的特殊 Token 标记
pub const ModelCapabilities = struct {
    /// 是否支持文本输入（总是 true）
    has_text: bool = true,
    /// 是否支持图像输入（视觉编码器）
    has_vision: bool = false,
    /// 是否支持音频输入（音频编码器，如 Conformer）
    has_audio: bool = false,
    /// 音频采样率（Hz），如 16000
    audio_sample_rate: i32 = 0,
    /// 视觉编码器类型名称
    vision_encoder_type: []const u8 = "",
    /// 音频编码器类型名称
    audio_encoder_type: []const u8 = "",
    /// 架构相关的特殊 Token 标记
    special_tokens: SpecialTokens = .{},

    /// 格式化输出能力描述
    pub fn format(self: ModelCapabilities, writer: anytype) !void {
        try writer.print("  Text  : {s}\n", .{if (self.has_text) "yes" else "no"});
        try writer.print("  Vision: {s}", .{if (self.has_vision) "yes" else "no"});
        if (self.has_vision and self.vision_encoder_type.len > 0) {
            try writer.print(" ({s})", .{self.vision_encoder_type});
        }
        try writer.print("\n", .{});
        try writer.print("  Audio : {s}", .{if (self.has_audio) "yes" else "no"});
        if (self.has_audio) {
            if (self.audio_encoder_type.len > 0) {
                try writer.print(" ({s})", .{self.audio_encoder_type});
            }
            if (self.audio_sample_rate > 0) {
                try writer.print(", sample_rate={d} Hz", .{self.audio_sample_rate});
            }
        }
        try writer.print("\n", .{});
    }
};
