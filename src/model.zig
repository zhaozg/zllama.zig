//! 模型抽象接口
//!
//! 定义 Model 接口，支持多模型架构的零成本抽象。
//! 使用 Zig 的 comptime 和 switch 实现编译时分发。

const std = @import("std");
const ggml = @import("ggml.zig");
const gguf = @import("gguf.zig");
const kv_cache = @import("kv_cache.zig");

/// RoPE 缩放配置
pub const RopeScaling = struct {
    rope_type: []const u8 = "",
    factor: f32 = 1.0,
    original_max_seq_len: u32 = 32768,
};

/// 支持的模型架构枚举
pub const Architecture = enum {
    qwen2,
    llama,
    // 未来可扩展：qwen3_moe, mixtral, ...

    /// 从 GGUF 元数据中的 general.architecture 字段解析
    pub fn fromString(s: []const u8) ?Architecture {
        if (std.mem.eql(u8, s, "qwen2") or
            std.mem.eql(u8, s, "qwen2.5") or
            std.mem.eql(u8, s, "qwen3.5") or
            std.mem.eql(u8, s, "qwen35"))
        {
            return .qwen2;
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

/// 模型超参数基类
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

/// 模型权重基类
/// 所有模型共享的通用权重结构
pub const ModelWeights = struct {
    params: ModelParams,

    // Token 嵌入
    token_embd: *ggml.Tensor,

    // 输出层
    output_weight: ?*ggml.Tensor = null,
    output_norm_weight: *ggml.Tensor,

    pub fn deinit(self: *ModelWeights, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

/// 模型接口（编译时多态）
/// 使用 anytype 泛型，在编译时确定具体模型类型
pub fn Model(comptime T: type) type {
    return struct {
        ptr: *T,

        pub fn init(self: *@This(), allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
            try self.ptr.init(allocator, gguf_file, io);
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.ptr.deinit(allocator);
        }

        pub fn forward(
            self: *@This(),
            ctx: *ggml.Context,
            graph: *ggml.CGraph,
            input_tokens: *ggml.Tensor,
            n_tokens: i32,
            kv_cache_mgr: ?*kv_cache.KVCache,
            start_pos: i32,
        ) !*ggml.Tensor {
            return self.ptr.forward(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos);
        }

        pub fn getParams(self: *@This()) *const ModelParams {
            return self.ptr.getParams();
        }

        pub fn getWeights(self: *@This()) *const ModelWeights {
            return self.ptr.getWeights();
        }
    };
}

const testing = std.testing;

test "Architecture fromString" {
    try testing.expectEqual(Architecture.qwen2, Architecture.fromString("qwen2").?);
    try testing.expectEqual(Architecture.qwen2, Architecture.fromString("qwen3.5").?);
    try testing.expectEqual(Architecture.llama, Architecture.fromString("llama").?);
    try testing.expectEqual(Architecture.llama, Architecture.fromString("llama3").?);
    try testing.expect(Architecture.fromString("unknown") == null);
}

test "ModelParams defaults" {
    const p = ModelParams{};
    try testing.expectEqual(@as(u32, 0), p.n_vocab);
    try testing.expectEqual(@as(u32, 32768), p.max_seq_len);
}
