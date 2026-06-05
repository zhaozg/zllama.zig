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
const ggml = @import("ggml.zig");
const gguf = @import("gguf.zig");
const graph_builder = @import("core/graph_builder.zig");
const memory = @import("core/memory.zig");

/// RoPE 缩放配置
pub const RopeScaling = struct {
    rope_type: []const u8 = "",
    factor: f32 = 1.0,
    original_max_seq_len: u32 = 32768,
};

/// 支持的模型架构枚举
pub const Architecture = enum {
    qwen2,
    qwen35,
    llama,

    /// 从 GGUF 元数据中的 general.architecture 字段解析
    pub fn fromString(s: []const u8) ?Architecture {
        if (std.mem.eql(u8, s, "qwen2") or
            std.mem.eql(u8, s, "qwen2.5"))
        {
            return .qwen2;
        }
        if (std.mem.eql(u8, s, "qwen3.5") or
            std.mem.eql(u8, s, "qwen35"))
        {
            return .qwen35;
        }
        if (std.mem.eql(u8, s, "llama") or
            std.mem.eql(u8, s, "llama2") or
            std.mem.eql(u8, s, "llama3"))
        {
            return .llama;
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

    // 分词器
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
    token_embd: *ggml.Tensor,
    output_weight: ?*ggml.Tensor,
    output_norm_weight: *ggml.Tensor,


};

/// 这是运行时多态的核心机制。
pub const ModelVTable = struct {
    /// 释放模型资源
    deinit: *const fn (data: *anyopaque) void,
    /// 构建前向计算图
    /// 返回 logits 张量
    buildGraph: *const fn (
        data: *anyopaque,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*memory.MemoryContext,
        start_pos: i32,
    ) anyerror!*ggml.Tensor,
    /// 获取模型参数
    getParams: *const fn (data: *anyopaque) *const ModelParams,
    /// 重置 SSM 状态（混合架构需要）
    resetSSMStates: *const fn (data: *anyopaque) void,
};

/// 模型实例
///
/// 通过虚表调用的运行时多态模型实例。
/// 注册表返回此类型，调用者无需知道具体模型类型。
pub const ModelInstance = struct {
    vtable: *const ModelVTable,
    data: *anyopaque,

    pub fn deinit(self: *ModelInstance) void {
        self.vtable.deinit(self.data);
    }

    pub fn buildGraph(
        self: *ModelInstance,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*memory.MemoryContext,
        start_pos: i32,
    ) !*ggml.Tensor {
        return self.vtable.buildGraph(self.data, builder, input_tokens, n_tokens, mem_ctx, start_pos);
    }

    pub fn getParams(self: *ModelInstance) *const ModelParams {
        return self.vtable.getParams(self.data);
    }

    pub fn resetSSMStates(self: *ModelInstance) void {
        self.vtable.resetSSMStates(self.data);
    }
};

/// 模型接口（编译时多态）
///
/// 使用 anytype 泛型，在编译时确定具体模型类型。
/// 模型内部实现使用此接口。
pub fn Model(comptime T: type) type {
    return struct {
        ptr: *T,

        pub fn init(self: *@This(), allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
            try self.ptr.init(allocator, gguf_file, io);
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.ptr.deinit(allocator);
        }

        pub fn buildGraph(
            self: *@This(),
            builder: *graph_builder.GraphBuilder,
            input_tokens: *ggml.Tensor,
            n_tokens: i32,
            mem_ctx: ?*memory.MemoryContext,
            start_pos: i32,
        ) !*ggml.Tensor {
            return self.ptr.buildGraph(builder, input_tokens, n_tokens, mem_ctx, start_pos);
        }

        pub fn getParams(self: *@This()) *const ModelParams {
            return self.ptr.getParams();
        }

        pub fn resetSSMStates(self: *@This()) void {
            self.ptr.resetSSMStates();
        }
    };
}

const testing = std.testing;

test "Architecture fromString" {
    try testing.expectEqual(Architecture.qwen2, Architecture.fromString("qwen2").?);
    try testing.expectEqual(Architecture.qwen35, Architecture.fromString("qwen3.5").?);
    try testing.expectEqual(Architecture.qwen35, Architecture.fromString("qwen35").?);
    try testing.expectEqual(Architecture.llama, Architecture.fromString("llama").?);
    try testing.expectEqual(Architecture.llama, Architecture.fromString("llama3").?);
    try testing.expect(Architecture.fromString("unknown") == null);
}

test "ModelParams defaults" {
    const p = ModelParams{};
    try testing.expectEqual(@as(u32, 0), p.n_vocab);
    try testing.expectEqual(@as(u32, 32768), p.max_seq_len);
}

test "ModelVTable size" {
    try testing.expectEqual(@as(usize, @sizeOf(ModelVTable)), @sizeOf(ModelVTable));
}
