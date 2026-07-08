//! ggml_pool 测试
//!
//! 对应 deps/ggml/tests/test-pool.c
//! 测试 ggml_pool_1d 和 ggml_pool_2d 在 avg/max 模式下的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// 测试 avg pool 1d - f32
test "ggml_pool_1d avg f32" {
    const mem_size: usize = 2 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf_f32: [1024]f32 = undefined;
    for (&buf_f32, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }

    const t = try ctx.newTensor2d(.f32, 10, 2);
    @memcpy(t.dataF32(), buf_f32[0..20]);

    const t_pooled = ggml.pool1d(ctx, t, @intFromEnum(ggml.PoolOp.avg), 3, 3, 0);

    try testing.expectEqual(@as(i64, 3), t_pooled.ne()[0]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[1]);
    try testing.expectEqual(@as(i64, 1), t_pooled.ne()[2]);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t_pooled);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = t_pooled.dataF32();
    try testing.expectEqual(@as(f32, 2.0), output[0]);
    try testing.expectEqual(@as(f32, 5.0), output[1]);
    try testing.expectEqual(@as(f32, 8.0), output[2]);
    try testing.expectEqual(@as(f32, 12.0), output[3]);
    try testing.expectEqual(@as(f32, 15.0), output[4]);
    try testing.expectEqual(@as(f32, 18.0), output[5]);
}

// 测试 avg pool 1d - f16
test "ggml_pool_1d avg f16" {
    const mem_size: usize = 2 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf_f32: [1024]f32 = undefined;
    var buf_f16: [1024]u16 = undefined;
    for (&buf_f32, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
        buf_f16[i] = @as(u16, @bitCast(ggml.c.ggml_fp32_to_fp16(buf_f32[i])));
    }

    const t = try ctx.newTensor2d(.f16, 10, 2);
    @memcpy(t.dataBytes(), std.mem.sliceAsBytes(buf_f16[0..20]));

    const t_pooled = ggml.pool1d(ctx, t, @intFromEnum(ggml.PoolOp.avg), 3, 3, 0);

    try testing.expectEqual(@as(i64, 3), t_pooled.ne()[0]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[1]);
    try testing.expectEqual(@as(i64, 1), t_pooled.ne()[2]);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t_pooled);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = t_pooled.dataF32();
    try testing.expectEqual(@as(f32, 2.0), output[0]);
    try testing.expectEqual(@as(f32, 5.0), output[1]);
    try testing.expectEqual(@as(f32, 8.0), output[2]);
    try testing.expectEqual(@as(f32, 12.0), output[3]);
    try testing.expectEqual(@as(f32, 15.0), output[4]);
    try testing.expectEqual(@as(f32, 18.0), output[5]);
}

// 测试 max pool 1d - f32
test "ggml_pool_1d max f32" {
    const mem_size: usize = 2 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf_f32: [1024]f32 = undefined;
    for (&buf_f32, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }

    const t = try ctx.newTensor2d(.f32, 10, 2);
    @memcpy(t.dataF32(), buf_f32[0..20]);

    const t_pooled = ggml.pool1d(ctx, t, @intFromEnum(ggml.PoolOp.max), 3, 3, 0);

    try testing.expectEqual(@as(i64, 3), t_pooled.ne()[0]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[1]);
    try testing.expectEqual(@as(i64, 1), t_pooled.ne()[2]);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t_pooled);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = t_pooled.dataF32();
    try testing.expectEqual(@as(f32, 3.0), output[0]);
    try testing.expectEqual(@as(f32, 6.0), output[1]);
    try testing.expectEqual(@as(f32, 9.0), output[2]);
    try testing.expectEqual(@as(f32, 13.0), output[3]);
    try testing.expectEqual(@as(f32, 16.0), output[4]);
    try testing.expectEqual(@as(f32, 19.0), output[5]);
}

// 测试 max pool 1d - f16
test "ggml_pool_1d max f16" {
    const mem_size: usize = 2 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf_f32: [1024]f32 = undefined;
    var buf_f16: [1024]u16 = undefined;
    for (&buf_f32, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
        buf_f16[i] = @as(u16, @bitCast(ggml.c.ggml_fp32_to_fp16(buf_f32[i])));
    }

    const t = try ctx.newTensor2d(.f16, 10, 2);
    @memcpy(t.dataBytes(), std.mem.sliceAsBytes(buf_f16[0..20]));

    const t_pooled = ggml.pool1d(ctx, t, @intFromEnum(ggml.PoolOp.max), 3, 3, 0);

    try testing.expectEqual(@as(i64, 3), t_pooled.ne()[0]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[1]);
    try testing.expectEqual(@as(i64, 1), t_pooled.ne()[2]);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t_pooled);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = t_pooled.dataF32();
    try testing.expectEqual(@as(f32, 3.0), output[0]);
    try testing.expectEqual(@as(f32, 6.0), output[1]);
    try testing.expectEqual(@as(f32, 9.0), output[2]);
    try testing.expectEqual(@as(f32, 13.0), output[3]);
    try testing.expectEqual(@as(f32, 16.0), output[4]);
    try testing.expectEqual(@as(f32, 19.0), output[5]);
}

// 测试 avg pool 2d - f32
test "ggml_pool_2d avg f32" {
    const mem_size: usize = 2 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf_f32: [1024]f32 = undefined;
    for (&buf_f32, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }

    const t = try ctx.newTensor3d(.f32, 10, 10, 2);
    @memcpy(t.dataF32(), buf_f32[0..200]);

    const t_pooled = t.pool2d(ctx, @intFromEnum(ggml.PoolOp.avg), 3, 4, 3, 4, 0, 0);

    try testing.expectEqual(@as(i64, 3), t_pooled.ne()[0]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[1]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[2]);
    try testing.expectEqual(@as(i64, 1), t_pooled.ne()[3]);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t_pooled);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = t_pooled.dataF32();
    try testing.expectEqual(@as(f32, 17.0), output[0]);
    try testing.expectEqual(@as(f32, 20.0), output[1]);
    try testing.expectEqual(@as(f32, 23.0), output[2]);
    try testing.expectEqual(@as(f32, 57.0), output[3]);
    try testing.expectEqual(@as(f32, 60.0), output[4]);
    try testing.expectEqual(@as(f32, 63.0), output[5]);
    try testing.expectEqual(@as(f32, 117.0), output[6]);
    try testing.expectEqual(@as(f32, 120.0), output[7]);
    try testing.expectEqual(@as(f32, 123.0), output[8]);
    try testing.expectEqual(@as(f32, 157.0), output[9]);
    try testing.expectEqual(@as(f32, 160.0), output[10]);
    try testing.expectEqual(@as(f32, 163.0), output[11]);
}

// 测试 max pool 2d - f32
test "ggml_pool_2d max f32" {
    const mem_size: usize = 2 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf_f32: [1024]f32 = undefined;
    for (&buf_f32, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }

    const t = try ctx.newTensor3d(.f32, 10, 10, 2);
    @memcpy(t.dataF32(), buf_f32[0..200]);

    const t_pooled = t.pool2d(ctx, @intFromEnum(ggml.PoolOp.max), 3, 4, 3, 4, 0, 0);

    try testing.expectEqual(@as(i64, 3), t_pooled.ne()[0]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[1]);
    try testing.expectEqual(@as(i64, 2), t_pooled.ne()[2]);
    try testing.expectEqual(@as(i64, 1), t_pooled.ne()[3]);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t_pooled);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = t_pooled.dataF32();
    try testing.expectEqual(@as(f32, 33.0), output[0]);
    try testing.expectEqual(@as(f32, 36.0), output[1]);
    try testing.expectEqual(@as(f32, 39.0), output[2]);
    try testing.expectEqual(@as(f32, 73.0), output[3]);
    try testing.expectEqual(@as(f32, 76.0), output[4]);
    try testing.expectEqual(@as(f32, 79.0), output[5]);
    try testing.expectEqual(@as(f32, 133.0), output[6]);
    try testing.expectEqual(@as(f32, 136.0), output[7]);
    try testing.expectEqual(@as(f32, 139.0), output[8]);
    try testing.expectEqual(@as(f32, 173.0), output[9]);
    try testing.expectEqual(@as(f32, 176.0), output[10]);
    try testing.expectEqual(@as(f32, 179.0), output[11]);
}
