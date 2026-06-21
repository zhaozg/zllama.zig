//! 计算图构建辅助
//!
//! 提供 GraphBuilder 上下文，封装常见的图构建操作。
//! 参考 llama.cpp 的 llm_graph_context 设计。
//!
//! 每个模型通过 GraphBuilder 构建计算图，共享算子通过此上下文调用。

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const memory = @import("memory");
const rope = @import("rope");

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

    /// 构建 RMS 归一化
    /// x: [n_embd, n_tokens]
    /// weight: [n_embd]
    pub fn buildRmsNorm(self: *GraphBuilder, x: *ggml.Tensor, weight: *ggml.Tensor, eps: f32) *ggml.Tensor {
        var result = ggml.rmsNorm(self.ctx, x, eps);
        result = ggml.mul(self.ctx, result, ggml.reshape2d(self.ctx, weight, @intCast(self.params.n_embd), 1));
        return result;
    }

    // ======================================================================
    // RoPE 位置编码
    // ======================================================================

    /// 对 Q 和 K 应用 RoPE
    /// q: [head_dim, n_head, n_tokens]
    /// k: [head_dim, n_kv_head, n_tokens]
    /// pos: [n_tokens] 位置索引
    pub fn buildRope(
        self: *GraphBuilder,
        q: *ggml.Tensor,
        k: *ggml.Tensor,
        pos: *ggml.Tensor,
        config: RopeConfig,
    ) struct { q: *ggml.Tensor, k: *ggml.Tensor } {
        const q_rope = ggml.ropeExt(
            self.ctx,
            q,
            pos,
            null,
            @intCast(config.rope_dim),
            config.mode,
            0,
            config.rope_theta,
            config.freq_scale,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        const k_rope = ggml.ropeExt(
            self.ctx,
            k,
            pos,
            null,
            @intCast(config.rope_dim),
            config.mode,
            0,
            config.rope_theta,
            config.freq_scale,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        return .{ .q = q_rope, .k = k_rope };
    }

    /// 构建位置张量 [start_pos, start_pos+1, ..., start_pos+n_tokens-1]
    /// 委托给 rope 模块的实现
    pub fn buildPositionTensor(self: *GraphBuilder, n_tokens: i32, start_pos: i32) !*ggml.Tensor {
        return rope.buildPositionTensor(self.ctx, n_tokens, start_pos);
    }

    // ======================================================================
    // 注意力操作
    // ======================================================================

    /// 构建缩放点积注意力
    /// q: [head_dim, n_head, n_tokens] (已 RoPE)
    /// k: [head_dim, n_kv_head, cache_len] (已 RoPE)
    /// v: [head_dim, n_kv_head, cache_len]
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
        const scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        _ = n_kv_head;

        // permute(0,2,1,3) 与 llama.cpp build_attn_mha 一致
        var q_perm = ggml.permute(self.ctx, q, 0, 2, 1, 3);
        q_perm = ggml.cont(self.ctx, q_perm);
        var k_perm = ggml.permute(self.ctx, k, 0, 2, 1, 3);
        k_perm = ggml.cont(self.ctx, k_perm);
        var v_perm = ggml.permute(self.ctx, v, 0, 2, 1, 3);
        v_perm = ggml.cont(self.ctx, v_perm);

        // kq = k @ q (mul_mat, ggml 自动处理 GQA 广播)
        var kq = ggml.mulMat(self.ctx, k_perm, q_perm);
        kq = ggml.scale(self.ctx, kq, scale_factor);

        // 因果 mask + softmax
        kq = ggml.reshape2d(self.ctx, kq, cache_len, n_tokens * n_head);
        kq = ggml.diagMaskInf(self.ctx, kq, start_pos);
        kq = ggml.softMax(self.ctx, kq);
        kq = ggml.reshape3d(self.ctx, kq, cache_len, n_tokens, n_head);

        // v^T @ kq
        const v_t = ggml.cont(self.ctx, ggml.transpose(self.ctx, v_perm));
        var kqv = ggml.mulMat(self.ctx, v_t, kq);

        // permute(0,2,1,3) 恢复布局
        kqv = ggml.permute(self.ctx, kqv, 0, 2, 1, 3);
        kqv = ggml.cont(self.ctx, kqv);

        // 展平为 [n_head * head_dim, n_tokens]
        kqv = ggml.reshape2d(self.ctx, kqv, n_head * head_dim, n_tokens);
        return kqv;
    }

    // ======================================================================
    // FFN 操作
    // ======================================================================

    /// 构建 SwiGLU FFN
    /// x: [n_embd, n_tokens]
    /// gate: [n_ff, n_embd]
    /// up: [n_ff, n_embd]
    /// down: [n_embd, n_ff]
    pub fn buildSwiGLU(
        self: *GraphBuilder,
        x: *ggml.Tensor,
        gate: *ggml.Tensor,
        up: *ggml.Tensor,
        down: *ggml.Tensor,
    ) *ggml.Tensor {
        const gate_out = ggml.mulMat(self.ctx, gate, x);
        const up_out = ggml.mulMat(self.ctx, up, x);
        const silu_out = ggml.silu(self.ctx, gate_out);
        const mul_out = ggml.mul(self.ctx, silu_out, up_out);
        return ggml.mulMat(self.ctx, down, mul_out);
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
