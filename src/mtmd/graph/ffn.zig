//! FFN 层构建器
//!
//! 提供 FFN（前馈网络）的 ggml 计算图构建，支持多种激活函数。
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_ffn()

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");

const FFNOpType = types.FFNOpType;

const log = std.log.scoped(.graph_ffn);

/// 构建 FFN 层
///
/// 支持以下激活函数:
///   - GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
///   - GELU_ERF: x * 0.5 * (1 + erf(x / sqrt(2)))
///   - SiLU: x * sigmoid(x)
///   - GELU_QUICK: x * sigmoid(1.702 * x)
///   - RELU_SQR: relu(x)^2
///
/// 参数:
///   - ctx: ggml 上下文
///   - cur: 输入张量 [n_embd, n_patches]
///   - up: up 投影权重 [n_ff, n_embd]
///   - up_b: up 投影偏置 [n_ff]（可选）
///   - gate: gate 投影权重 [n_ff, n_embd]（可选，GLU 变体需要）
///   - gate_b: gate 投影偏置 [n_ff]（可选）
///   - down: down 投影权重 [n_embd, n_ff]
///   - down_b: down 投影偏置 [n_embd]（可选）
///   - type_op: FFN 激活函数类型
///   - name: 张量名称前缀（用于调试）
///
/// 返回: FFN 输出张量 [n_embd, n_patches]
///
/// 参考: clip-graph.h build_ffn()
pub fn buildFFN(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    up: *ggml.Tensor,
    up_b: ?*ggml.Tensor,
    gate: ?*ggml.Tensor,
    gate_b: ?*ggml.Tensor,
    down: *ggml.Tensor,
    down_b: ?*ggml.Tensor,
    type_op: FFNOpType,
    name: []const u8,
) !*ggml.Tensor {
    // Up projection: [n_ff, n_embd] @ [n_embd, n_patches] → [n_ff, n_patches]
    var up_result = up.mulMat(ctx, cur);

    if (up_b) |b| {
        up_result = up_result.add(ctx, b);
    }

    // Gate projection (optional, for GLU variants)
    var gate_result: ?*ggml.Tensor = null;
    if (gate) |g| {
        gate_result = g.mulMat(ctx, cur);
        if (gate_b) |gb| {
            gate_result = gate_result.?.add(ctx, gb);
        }
    }

    var activated = blk: {
        break :blk switch (type_op) {
            .gelu => up_result.gelu(ctx),
            .gelu_erf => up_result.geluErf(ctx),
            .silu => up_result.silu(ctx),
            .gelu_quick => up_result.geluQuick(ctx),
            .relu_sqr => {
                const relu = up_result.relu(ctx);
                break :blk relu.mul(ctx, relu);
            },
        };
    };

    // Gate (element-wise multiply with gate projection)
    if (gate_result) |g| {
        activated = activated.mul(ctx, g);
    }

    // Down projection: [n_embd, n_ff] @ [n_ff, n_patches] → [n_embd, n_patches]
    var result = down.mulMat(ctx, activated);

    if (down_b) |b| {
        result = result.add(ctx, b);
    }

    _ = name; // 保留参数以保持 API 兼容性
    return result;
}

test "buildFFN: SiLU activation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_embd: i64 = 64;
    const n_ff: i64 = 256;
    const n_patches: i64 = 16;

    const cur = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_embd);
    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_ff);

    @memset(cur.dataF32(), 1.0);
    @memset(up_w.dataF32(), 0.1);
    @memset(down_w.dataF32(), 0.1);

    const result = try buildFFN(&ctx, cur, up_w, null, null, null, down_w, null, .silu, "test_ffn");
    try testing.expectEqual(@as(i64, n_embd), result.ne()[0]);
    try testing.expectEqual(@as(i64, n_patches), result.ne()[1]);
}

test "buildFFN: GELU with gate" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_embd: i64 = 64;
    const n_ff: i64 = 256;
    const n_patches: i64 = 16;

    const cur = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_embd);
    const gate_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_embd);
    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_ff);

    @memset(cur.dataF32(), 1.0);
    @memset(up_w.dataF32(), 0.1);
    @memset(gate_w.dataF32(), 0.1);
    @memset(down_w.dataF32(), 0.1);

    const result = try buildFFN(&ctx, cur, up_w, null, gate_w, null, down_w, null, .gelu, "test_ffn_gate");
    try testing.expectEqual(@as(i64, n_embd), result.ne()[0]);
    try testing.expectEqual(@as(i64, n_patches), result.ne()[1]);
}
