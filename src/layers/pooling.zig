//! 池化层 — 将 hidden states 压缩为单一向量
//!
//! 用于嵌入模型（Embedding Model）的最后一层，
//! 将 [n_embd, n_tokens] 的 hidden states 压缩为 [n_embd] 向量。
//!
//! 支持三种池化策略：
//! - mean: 所有 token 取均值（默认，BGE/通用嵌入）
//! - cls:  取第一个 token（BERT 风格）
//! - last: 取最后一个 token（GPT 风格）

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.pooling);

/// 池化类型
pub const PoolingType = enum {
    mean,
    cls,
    last,

    pub fn fromString(s: []const u8) PoolingType {
        if (std.mem.eql(u8, s, "mean")) return .mean;
        if (std.mem.eql(u8, s, "cls")) return .cls;
        if (std.mem.eql(u8, s, "last")) return .last;
        return .mean; // default
    }

    pub fn name(self: PoolingType) []const u8 {
        return switch (self) {
            .mean => "mean",
            .cls => "cls",
            .last => "last",
        };
    }
};

/// mean 池化: 对 tokens 维度取均值
///
/// hidden: [n_embd, n_tokens]
/// 返回: [n_embd, 1] (2D，需 reshape to 1D)
pub fn meanPool(ctx: *ggml.Context, hidden: *ggml.Tensor) *ggml.Tensor {
    const n_tokens = hidden.ne()[1];
    // ggml.sumRows: [n_embd, n_tokens] → [n_embd, 1]
    const summed = ggml.sumRows(ctx, hidden);
    // 缩放为均值
    const scale_val = 1.0 / @as(f32, @floatFromInt(n_tokens));
    const mean = ggml.scale(ctx, summed, scale_val);
    mean.setName("pooling.mean");
    return mean;
}

/// cls 池化: 取第一个 token 的 hidden state
///
/// hidden: [n_embd, n_tokens]
/// 返回: [n_embd, 1]
pub fn clsPool(ctx: *ggml.Context, hidden: *ggml.Tensor) *ggml.Tensor {
    const n_embd = hidden.ne()[0];
    _ = hidden.ne()[1];
    // 取第 0 列: [n_embd, 1]
    const result = ctx.view2d(hidden, n_embd, 1, hidden.nb()[1], // nb1: stride for column dim
        0); // offset 0 = first column
    result.setName("pooling.cls");
    return result;
}

/// last 池化: 取最后一个 token 的 hidden state
///
/// hidden: [n_embd, n_tokens]
/// 返回: [n_embd, 1]
pub fn lastPool(ctx: *ggml.Context, hidden: *ggml.Tensor) *ggml.Tensor {
    const n_embd = hidden.ne()[0];
    const n_tokens = hidden.ne()[1];
    // 偏移量为 (n_tokens - 1) * nb[1]
    const offset: usize = @intCast((n_tokens - 1) * @as(i64, @intCast(hidden.nb()[1])));
    const result = ctx.view2d(hidden, n_embd, 1, hidden.nb()[1], offset);
    result.setName("pooling.last");
    return result;
}

/// 池化调度
///
/// hidden: [n_embd, n_tokens]
/// pooling_type: 池化策略
/// 返回: [n_embd, 1] 池化后的向量
pub fn poolHidden(
    ctx: *ggml.Context,
    hidden: *ggml.Tensor,
    pooling_type: PoolingType,
) *ggml.Tensor {
    return switch (pooling_type) {
        .mean => meanPool(ctx, hidden),
        .cls => clsPool(ctx, hidden),
        .last => lastPool(ctx, hidden),
    };
}

/// L2 归一化
///
/// 将向量归一化为单位向量：vec / ||vec||₂
/// 归一化后余弦相似度 = 向量内积
///
/// vec: [n_embd, 1] 或 [n_embd]
/// 返回: [n_embd, 1] 归一化后的向量
pub fn normalize(ctx: *ggml.Context, vec: *ggml.Tensor) *ggml.Tensor {
    const result = ggml.l2Norm(ctx, vec, 1e-6);
    result.setName("embd.l2_norm");
    return result;
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "PoolingType fromString" {
    try testing.expectEqual(PoolingType.mean, PoolingType.fromString("mean"));
    try testing.expectEqual(PoolingType.cls, PoolingType.fromString("cls"));
    try testing.expectEqual(PoolingType.last, PoolingType.fromString("last"));
    try testing.expectEqual(PoolingType.mean, PoolingType.fromString("unknown"));
}

test "mean pool basic" {
    const n_embd: i64 = 8;
    const n_tokens: i64 = 4;

    const mem_size = 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // 创建 hidden [n_embd, n_tokens]，填 1.0
    const hidden = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    hidden.setName("test_hidden");
    {
        const buf = try std.testing.allocator.alloc(f32, @as(usize, @intCast(n_embd * n_tokens)));
        defer std.testing.allocator.free(buf);
        @memset(buf, 1.0);
        try hidden.dataSet(f32, buf);
    }

    var graph = try ggml.CGraph.init(ctx);
    const pooled = meanPool(ctx, hidden);
    ggml.setOutput(pooled);
    graph.buildForwardExpand(pooled);
    try graph.compute(1);

    // 均值应为 1.0
    {
        const result_data = try pooled.dataGet(f32, std.testing.allocator);
        defer std.testing.allocator.free(result_data);
        for (result_data[0..@as(usize, @intCast(n_embd))]) |v| {
            try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-5);
        }
    }

    const shape = pooled.shape();
    try testing.expectEqual(n_embd, shape[0]);
    try testing.expectEqual(@as(i64, 1), shape[1]);
}

test "cls pool basic" {
    const n_embd: i64 = 4;
    const n_tokens: i64 = 3;

    const mem_size = 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const hidden = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    hidden.setName("test_hidden");
    {
        const buf = try std.testing.allocator.alloc(f32, @as(usize, @intCast(n_embd * n_tokens)));
        defer std.testing.allocator.free(buf);
        // 填充可区分值: col0 = 1, col1 = 2, col2 = 3
        // 布局: [n_embd, n_tokens] 列优先
        for (0..@as(usize, @intCast(n_tokens))) |col| {
            for (0..@as(usize, @intCast(n_embd))) |row| {
                buf[col * @as(usize, @intCast(n_embd)) + row] = @as(f32, @floatFromInt(col + 1));
            }
        }
        try hidden.dataSet(f32, buf);
    }

    const pooled = clsPool(ctx, hidden);
    pooled.setName("test_pooled");
    ggml.setOutput(pooled);
    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(pooled);
    try graph.compute(1);

    // cls 取第一个 token，值应为 1.0
    {
        const result_data = try pooled.dataGet(f32, std.testing.allocator);
        defer std.testing.allocator.free(result_data);
        for (result_data[0..@as(usize, @intCast(n_embd))]) |v| {
            try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-5);
        }
    }
}

test "L2 normalize basic" {
    const n_embd: i64 = 4;
    const n_tokens: i64 = 3;

    const mem_size = 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const hidden = try ctx.newTensor2d(.f32, n_embd, n_tokens);
    hidden.setName("test_hidden");
    const vec = try ctx.newTensor2d(.f32, n_embd, 1);
    vec.setName("test_vec");
    {
        const buf = try std.testing.allocator.alloc(f32, @as(usize, @intCast(n_embd)));
        defer std.testing.allocator.free(buf);
        buf[0] = 3.0;
        buf[1] = 4.0;
        buf[2] = 0.0;
        buf[3] = 0.0;
        try vec.dataSet(f32, buf);
    }

    var graph = try ggml.CGraph.init(ctx);
    const norm = normalize(ctx, vec);
    ggml.setOutput(norm);
    graph.buildForwardExpand(norm);

    try graph.compute(1);

    // ||[3,4,0,0]|| = 5, 归一化: [0.6, 0.8, 0, 0]
    {
        const result_data = try norm.dataGet(f32, std.testing.allocator);
        defer std.testing.allocator.free(result_data);
        try testing.expectApproxEqAbs(@as(f32, 0.6), result_data[0], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.8), result_data[1], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.0), result_data[2], 1e-5);
    }
}

test "L2 normalize unit vector" {
    const n_embd: i64 = 4;

    const mem_size = 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const vec = try ctx.newTensor2d(.f32, n_embd, 1);
    vec.setName("test_vec");
    {
        const buf = try std.testing.allocator.alloc(f32, @as(usize, @intCast(vec.nElems())));
        defer std.testing.allocator.free(buf);
        buf[0] = 3.0;
        buf[1] = 4.0;
        buf[2] = 0.0;
        buf[3] = 0.0;
        try vec.dataSet(f32, buf);
    }

    var graph = try ggml.CGraph.init(ctx);
    const norm = normalize(ctx, vec);
    ggml.setOutput(norm);
    graph.buildForwardExpand(norm);

    try graph.compute(1);

    // ||[3,4,0,0]|| = 5, 归一化: [0.6, 0.8, 0, 0]
    {
        const result_data = try norm.dataGet(f32, std.testing.allocator);
        defer std.testing.allocator.free(result_data);
        try testing.expectApproxEqAbs(@as(f32, 0.6), result_data[0], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.8), result_data[1], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.0), result_data[2], 1e-5);
    }
}
