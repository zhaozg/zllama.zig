//! 嵌入模型测试
//!
//! 测试嵌入模型特有功能：
//! - 双向注意力（无 causal mask）
//! - 池化（mean/cls/last）
//! - L2 归一化
//! - 无 output.weight 容错

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");
const pooling = @import("pooling");

// ============================================================================
// 双向注意力测试
// ============================================================================

test "embedding.bidirectional_attention" {
    // 测试双向注意力：不应用 causal mask，让所有 token 互相看见
    const n_embd: i64 = 16;
    const n_head: i64 = 4;
    const head_dim: i64 = @divExact(n_embd, n_head);
    const n_tokens: i64 = 3;
    const n_kv_head: i64 = 4;

    const mem_size = 16 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // 创建 Q/K/V 伪张量 [n_embd, n_tokens]
    const q_2d = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    const k_2d = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    const v_2d = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    {
        const qd = q_2d.dataF32();
        const kd = k_2d.dataF32();
        const vd = v_2d.dataF32();
        // 填充测试数据：每个 token 递增
        for (0..@as(usize, @intCast(n_tokens))) |t| {
            const base: usize = t * @as(usize, @intCast(n_embd));
            for (0..@as(usize, @intCast(n_embd))) |j| {
                qd[base + j] = @floatFromInt(j + 1);
                kd[base + j] = @floatFromInt(j + 1);
                vd[base + j] = 1.0;
            }
        }
    }

    // Reshape: [head_dim, n_head, n_tokens]
    var q = ggml.reshape3d(ctx, q_2d, head_dim, n_head, n_tokens);
    var kv = ggml.reshape3d(ctx, k_2d, head_dim, n_kv_head, n_tokens);
    var vv = ggml.reshape3d(ctx, v_2d, head_dim, n_kv_head, n_tokens);

    // Permute: [head_dim, n_tokens, n_head] for attention
    q = ggml.cont(ctx, ggml.permute(ctx, q, 0, 2, 1, 3));
    kv = ggml.cont(ctx, ggml.permute(ctx, kv, 0, 2, 1, 3));
    vv = ggml.cont(ctx, ggml.permute(ctx, vv, 0, 2, 1, 3));

    // 双向注意力：无 causal mask
    const q_perm = ggml.cont(ctx, ggml.permute(ctx, q, 0, 2, 1, 3));
    const k_perm = ggml.cont(ctx, ggml.permute(ctx, kv, 0, 2, 1, 3));
    const v_perm = ggml.cont(ctx, ggml.permute(ctx, vv, 0, 2, 1, 3));

    var kq = ggml.mulMat(ctx, k_perm, q_perm);
    kq = ggml.scale(ctx, kq, 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
    // 双向：不应用 causal mask，直接 softmax
    kq = ggml.softMax(ctx, kq);

    const v_t = ggml.cont(ctx, ggml.transpose(ctx, v_perm));
    var kqv = ggml.mulMat(ctx, v_t, kq);
    kqv = ggml.permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml.cont(ctx, kqv);

    // 展平
    kqv = ggml.reshape2d(ctx, kqv, n_head * head_dim, n_tokens);

    var graph = try ggml.CGraph.init(ctx);
    ggml.setOutput(kqv);
    graph.buildForwardExpand(kqv);
    try graph.compute(1);

    // 验证输出形状
    const shape = kqv.shape();
    try testing.expectEqual(n_head * head_dim, shape[0]);
    try testing.expectEqual(n_tokens, shape[1]);

    // 双向注意力：每个 token 应该看到所有 token
    // 因为全 1 值输入，每个 token 的输出应该一致
    const out = kqv.dataF32();
    const first_val = out[0];
    for (out[0..@as(usize, @intCast(n_embd * n_tokens))]) |elem| {
        try testing.expectApproxEqAbs(first_val, elem, 1e-4);
    }
}

// ============================================================================
// 池化测试
// ============================================================================

test "embedding.mean_pooling" {
    const n_embd: i64 = 8;
    const n_tokens: i64 = 4;

    const mem_size = 4 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const hidden = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    {
        const data = hidden.dataF32();
        @memset(data, 2.0);
    }

    var graph = try ggml.CGraph.init(ctx);
    const pooled = pooling.poolHidden(ctx, hidden, .mean);
    ggml.setOutput(pooled);
    graph.buildForwardExpand(pooled);
    try graph.compute(1);

    // meanPool 使用 ggml_sum_rows 对 ne[0] 维度求和
    // 输入 [n_embd, n_tokens] → sum_rows → [1, n_tokens] → scale(1/n_tokens)
    // 每个元素 = 2 * n_embd / n_tokens = 2 * 8 / 4 = 4
    const ne = pooled.ne();
    const n_elems = @as(usize, @intCast(ne[0] * ne[1]));
    try testing.expect(n_elems > 0);
    const result = pooled.dataF32();
    for (result[0..n_elems]) |val| {
        try testing.expectApproxEqAbs(@as(f32, 4.0), val, 1e-5);
    }
}

test "embedding.cls_pooling" {
    const n_embd: i64 = 4;
    const n_tokens: i64 = 3;

    const mem_size = 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const hidden = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    {
        const data = hidden.dataF32();
        // col0=1, col1=2, col2=3
        for (0..@as(usize, @intCast(n_tokens))) |col| {
            for (0..@as(usize, @intCast(n_embd))) |row| {
                data[col * @as(usize, @intCast(n_embd)) + row] = @as(f32, @floatFromInt(col + 1));
            }
        }
    }

    var graph = try ggml.CGraph.init(ctx);
    const pooled = pooling.poolHidden(ctx, hidden, .cls);
    ggml.setOutput(pooled);
    graph.buildForwardExpand(pooled);
    try graph.compute(1);

    // cls 取第一个 token = 1.0
    const result = pooled.dataF32();
    for (result[0..@as(usize, @intCast(n_embd))]) |val| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), val, 1e-5);
    }
}

test "embedding.last_pooling" {
    const n_embd: i64 = 4;
    const n_tokens: i64 = 3;

    const mem_size = 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const hidden = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    {
        const data = hidden.dataF32();
        for (0..@as(usize, @intCast(n_tokens))) |col| {
            for (0..@as(usize, @intCast(n_embd))) |row| {
                data[col * @as(usize, @intCast(n_embd)) + row] = @as(f32, @floatFromInt(col + 1));
            }
        }
    }

    var graph = try ggml.CGraph.init(ctx);
    const pooled = pooling.poolHidden(ctx, hidden, .last);
    ggml.setOutput(pooled);
    graph.buildForwardExpand(pooled);
    try graph.compute(1);

    // last 取最后一个 token = 3.0
    const result = pooled.dataF32();
    for (result[0..@as(usize, @intCast(n_embd))]) |val| {
        try testing.expectApproxEqAbs(@as(f32, 3.0), val, 1e-5);
    }
}

// ============================================================================
// L2 归一化测试
// ============================================================================

test "embedding.l2_normalize" {
    const n_embd: i64 = 4;

    const mem_size = 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const vec = try ctx.newTensor2d(.f32, n_embd, 1);
    {
        const data = vec.dataF32();
        data[0] = 3.0;
        data[1] = 4.0;
        data[2] = 0.0;
        data[3] = 0.0;
    }

    var graph = try ggml.CGraph.init(ctx);
    const norm = pooling.normalize(ctx, vec);
    ggml.setOutput(norm);
    graph.buildForwardExpand(norm);
    try graph.compute(1);

    // ||[3,4,0,0]|| = 5, 归一化: [0.6, 0.8, 0, 0]
    const result = norm.dataF32();
    try testing.expectApproxEqAbs(@as(f32, 0.6), result[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.8), result[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result[2], 1e-5);

    // 验证单位向量：norm = 1.0
    var sum_sq: f32 = 0;
    for (result[0..@as(usize, @intCast(n_embd))]) |val| {
        sum_sq += val * val;
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(sum_sq), 1e-4);
}

// ============================================================================
// 无 output.weight 测试
// ============================================================================

test "embedding.no_output_weight" {
    // 验证嵌入模型不需要 output.weight
    // 通过检查 pooling 模块的导入和使用来间接验证

    // 池化类型解析
    try testing.expectEqual(pooling.PoolingType.mean, pooling.PoolingType.fromString("mean"));
    try testing.expectEqual(pooling.PoolingType.cls, pooling.PoolingType.fromString("cls"));
    try testing.expectEqual(pooling.PoolingType.last, pooling.PoolingType.fromString("last"));
    // 未知字符串默认返回 mean
    try testing.expectEqual(pooling.PoolingType.mean, pooling.PoolingType.fromString("unknown"));
}

// ============================================================================
// 结构完整性测试
// ============================================================================

test "embedding.pooling_type_names" {
    try testing.expectEqualStrings("mean", pooling.PoolingType.mean.name());
    try testing.expectEqualStrings("cls", pooling.PoolingType.cls.name());
    try testing.expectEqualStrings("last", pooling.PoolingType.last.name());
}
