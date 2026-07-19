//! FFN 层构建器
//!
//! 提供 FFN（前馈网络）的 ggml 计算图构建，支持多种激活函数。
//! 参考: deps/llama.cpp/tools/mtmd/clip.cpp clip_graph::build_ffn()

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");

const FFNOpType = types.FFNOpType;
const BuildMMFn = types.BuildMMFn;
const defaultBuildMM = types.defaultBuildMM;
const CbFn = types.CbFn;
const defaultCb = types.defaultCb;

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
///   - up: up 投影权重 [n_ff, n_embd]（可选，为 null 时跳过 up 投影）
///   - up_b: up 投影偏置 [n_ff]（可选）
///   - gate: gate 投影权重 [n_ff, n_embd]（可选，GLU 变体需要）
///   - gate_b: gate 投影偏置 [n_ff]（可选）
///   - down: down 投影权重 [n_embd, n_ff]（可选，为 null 时跳过 down 投影）
///   - down_b: down 投影偏置 [n_embd]（可选）
///   - type_op: FFN 激活函数类型
///   - il: 层索引（用于调试回调，-1 表示不使用）
///   - build_mm: 矩阵乘法回调（对应 C++ clip_graph::build_mm 虚拟函数）
///   - data: 模型私有数据指针，传递给 build_mm 回调
///   - cb: 张量命名回调（可选），对应 C++ clip_graph::cb
///
/// 返回: FFN 输出张量 [n_embd, n_patches]
///
/// 参考: clip.cpp clip_graph::build_ffn()
pub fn buildFFN(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    up: ?*ggml.Tensor,
    up_b: ?*ggml.Tensor,
    gate: ?*ggml.Tensor,
    gate_b: ?*ggml.Tensor,
    down: ?*ggml.Tensor,
    down_b: ?*ggml.Tensor,
    type_op: FFNOpType,
    il: i32,
    build_mm: BuildMMFn,
    data: ?*anyopaque,
    cb: ?CbFn,
) !*ggml.Tensor {
    // C++: ggml_tensor * tmp = up ? build_mm(up, cur) : cur;
    // Up projection: [n_ff, n_embd] @ [n_embd, n_patches] → [n_ff, n_patches]
    var up_result = if (up) |u| blk: {
        const t = build_mm(ctx, u, cur, data);
        // C++: cb(tmp, "ffn_up", il);
        if (cb) |cbf| cbf(t, "ffn_up", il);
        break :blk t;
    } else cur;

    // C++: if (up_b) { tmp = ggml_add(ctx0, tmp, up_b); cb(tmp, "ffn_up_b", il); }
    if (up_b) |b| {
        up_result = up_result.add(ctx, b);
        if (cb) |cbf| cbf(up_result, "ffn_up_b", il);
    }

    // Gate projection (optional, for GLU variants)
    // C++: if (gate) { cur = build_mm(gate, cur); cb(cur, "ffn_gate", il); ... }
    var gate_result: ?*ggml.Tensor = null;
    if (gate) |g| {
        gate_result = build_mm(ctx, g, cur, data);
        // C++: cb(cur, "ffn_gate", il);
        if (cb) |cbf| cbf(gate_result.?, "ffn_gate", il);

        if (gate_b) |gb| {
            gate_result = gate_result.?.add(ctx, gb);
            // C++: cb(cur, "ffn_gate_b", il);
            if (cb) |cbf| cbf(gate_result.?, "ffn_gate_b", il);
        }
    }

    // Activation
    // C++: switch (type_op) { case FFN_SILU: ... cb(cur, "ffn_swiglu", il); ... }
    const activated = blk: {
        if (gate_result) |g| {
            // GLU variants: use split activation (tensor method)
            const t = switch (type_op) {
                .silu => g.swigluSplit(ctx, up_result),
                .gelu => g.gegluSplit(ctx, up_result),
                .gelu_erf => g.gegluErfSplit(ctx, up_result),
                .gelu_quick => g.gegluQuickSplit(ctx, up_result),
                .relu_sqr => {
                    const relu = g.relu(ctx);
                    break :blk relu.mul(ctx, relu);
                },
            };
            // C++: cb(cur, "ffn_swiglu"/"ffn_geglu"/etc, il);
            if (cb) |cbf| {
                const name = switch (type_op) {
                    .silu => "ffn_swiglu",
                    .gelu => "ffn_geglu",
                    .gelu_erf => "ffn_geglu_erf",
                    .gelu_quick => "ffn_geglu_quick",
                    .relu_sqr => "ffn_relu_sqr",
                };
                cbf(t, name, il);
            }
            break :blk t;
        } else {
            // Non-GLU variants
            const t = switch (type_op) {
                .silu => up_result.silu(ctx),
                .gelu => up_result.gelu(ctx),
                .gelu_erf => up_result.geluErf(ctx),
                .gelu_quick => up_result.geluQuick(ctx),
                .relu_sqr => {
                    const relu = up_result.relu(ctx);
                    break :blk relu.mul(ctx, relu);
                },
            };
            // C++: cb(cur, "ffn_silu"/"ffn_gelu"/etc, il);
            if (cb) |cbf| {
                const name = switch (type_op) {
                    .silu => "ffn_silu",
                    .gelu => "ffn_gelu",
                    .gelu_erf => "ffn_gelu_erf",
                    .gelu_quick => "ffn_gelu_quick",
                    .relu_sqr => "ffn_relu_sqr",
                };
                cbf(t, name, il);
            }
            break :blk t;
        }
    };

    // Down projection: [n_embd, n_ff] @ [n_ff, n_patches] → [n_embd, n_patches]
    // C++: if (down) { cur = build_mm(down, cur); }
    var result = if (down) |d| blk: {
        const t = build_mm(ctx, d, activated, data);
        break :blk t;
    } else activated;

    // C++: if (down_b) { cb(cur, "ffn_down", il); cur = ggml_add(ctx0, cur, down_b); }
    // Note: cb("ffn_down") is called ONLY when down_b exists, BEFORE the add.
    if (down_b) |b| {
        if (cb) |cbf| cbf(result, "ffn_down", il);
        result = result.add(ctx, b);
    }

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

    {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(cur.nElems())));
        defer allocator.free(buf);
        try cur.dataSet(f32, buf);
    }
    for ([_]*ggml.Tensor{ up_w, down_w }) |t| {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(t.nElems())));
        defer allocator.free(buf);
        @memset(buf, 0.1);
        try t.dataSet(f32, buf);
    }
    const result = try buildFFN(&ctx, cur, up_w, null, null, null, down_w, null, .silu, -1, defaultBuildMM, null, null);
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

    {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(cur.nElems())));
        defer allocator.free(buf);
        @memset(buf, 1.0);
        try cur.dataSet(f32, buf);
    }
    for ([_]*ggml.Tensor{ up_w, gate_w, down_w }) |t| {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(t.nElems())));
        defer allocator.free(buf);
        @memset(buf, 0.1);
        try t.dataSet(f32, buf);
    }
    const result = try buildFFN(&ctx, cur, up_w, null, gate_w, null, down_w, null, .gelu, -1, defaultBuildMM, null, null);
    try testing.expectEqual(@as(i64, n_embd), result.ne()[0]);
    try testing.expectEqual(@as(i64, n_patches), result.ne()[1]);
}
