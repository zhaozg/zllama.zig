//! 计算图构建辅助
//!
//! 提供 GraphBuilder 上下文，封装常见的图构建操作。
//! 参考 llama.cpp 的 llm_graph_context 设计。
//!
//! 每个模型通过 GraphBuilder 构建计算图，共享算子通过此上下文调用。
//!
//! GraphBuilder 的 build* 方法委托给 src/layers/ 模块，
//! 确保与模型直接调用的 layers/ 函数共享同一实现，消除重复代码。

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const memory = @import("memory");

// 委托给共享 layers 模块
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const attention = @import("attention");
const swiglu = @import("swiglu");

/// RoPE 配置
pub const RopeConfig = struct {
    rope_dim: i64 = 64,
    rope_theta: f32 = 10000000.0,
    freq_scale: f32 = 1.0,
    mode: i32 = 2, // 0=NORM, 2=NEOX
};

/// 归一化类型
pub const NormType = enum(u8) {
    rms = 0,
    layer = 1,
    group = 2,
};

/// FFN 激活类型
pub const FFNActType = enum(u8) {
    silu = 0,
    gelu = 1,
    relu = 2,
    swiglu = 3,
    geglu = 4,
    reglu = 5,
};

/// 图构建上下文
///
/// 封装 ggml context 和 cgraph，提供便捷的图构建方法。
/// 每个模型在 forward 时创建 GraphBuilder 实例。
pub const GraphBuilder = struct {
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    params: *const model_if.ModelParams,
    allocator: std.mem.Allocator,

    /// 初始化图构建器
    pub fn init(
        ctx: *ggml.Context,
        gf: *ggml.CGraph,
        params: *const model_if.ModelParams,
        allocator: std.mem.Allocator,
    ) GraphBuilder {
        return GraphBuilder{
            .ctx = ctx,
            .gf = gf,
            .params = params,
            .allocator = allocator,
        };
    }

    // ======================================================================
    // 归一化操作
    // ======================================================================

    /// 构建 RMS 归一化（委托给 layers/rms_norm 模块）
    pub fn buildRmsNorm(self: *GraphBuilder, x: *ggml.Tensor, weight: *ggml.Tensor, eps: f32) *ggml.Tensor {
        return rms_norm.rmsNorm(self.ctx, x, weight, eps);
    }

    // ======================================================================
    // RoPE 位置编码
    // ======================================================================

    /// 对 Q 和 K 应用 RoPE（委托给 layers/rope 模块）
    pub fn buildRope(
        self: *GraphBuilder,
        q: *ggml.Tensor,
        k: *ggml.Tensor,
        pos: *ggml.Tensor,
        config: RopeConfig,
    ) struct { q: *ggml.Tensor, k: *ggml.Tensor } {
        const result = rope.applyRope(self.ctx, q, k, pos, .{
            .rope_dim = config.rope_dim,
            .rope_theta = config.rope_theta,
            .rope_scaling_factor = config.freq_scale,
            .mode = config.mode,
        });
        return .{ .q = result.q, .k = result.k };
    }

    /// 构建位置张量（委托给 layers/rope 模块）
    pub fn buildPositionTensor(self: *GraphBuilder, n_tokens: i32, start_pos: i32) !*ggml.Tensor {
        return rope.buildPositionTensor(self.ctx, n_tokens, start_pos);
    }

    // ======================================================================
    // 注意力操作
    // ======================================================================

    /// 构建缩放点积注意力（委托给 layers/attention 模块）
    pub fn buildAttention(
        self: *GraphBuilder,
        q: *ggml.Tensor,
        k: *ggml.Tensor,
        v: *ggml.Tensor,
        n_head: i64,
        n_kv_head: i64,
        head_dim: i64,
        n_tokens: i64,
        cache_len: i64,
        start_pos: i32,
    ) *ggml.Tensor {
        return attention.scaledDotProductAttention(self.ctx, q, k, v, .{
            .n_head = n_head,
            .n_kv_head = n_kv_head,
            .head_dim = head_dim,
            .n_tokens = n_tokens,
            .cache_len = cache_len,
            .start_pos = start_pos,
            .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        }, null);
    }

    // ======================================================================
    // FFN 操作
    // ======================================================================

    /// 构建 SwiGLU FFN（委托给 layers/swiglu 模块）
    pub fn buildSwiGLU(
        self: *GraphBuilder,
        x: *ggml.Tensor,
        gate: *ggml.Tensor,
        up: *ggml.Tensor,
        down: *ggml.Tensor,
    ) *ggml.Tensor {
        return swiglu.swiGLU(self.ctx, x, gate, up, down);
    }

    // ======================================================================
    // 图操作
    // ======================================================================

    /// 展开前向计算
    pub fn forwardExpand(self: *GraphBuilder, tensor: *ggml.Tensor) void {
        self.gf.buildForwardExpand(tensor);
    }
    /// 设置输出张量
    pub fn setOutput(self: *GraphBuilder, tensor: *ggml.Tensor) void {
        _ = self;
        ggml.setOutput(tensor);
    }
};

const testing = std.testing;

test "RopeConfig defaults" {
    const c = RopeConfig{};
    try testing.expectEqual(@as(i64, 64), c.rope_dim);
    try testing.expectEqual(@as(f32, 10000000.0), c.rope_theta);
}

test "NormType enum" {
    try testing.expectEqual(@as(u8, 0), @intFromEnum(NormType.rms));
    try testing.expectEqual(@as(u8, 3), @intFromEnum(NormType.swiglu));
}
