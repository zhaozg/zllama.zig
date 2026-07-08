//! ggml_dup 测试
//!
//! 对应 deps/ggml/tests/test-dup.c
//! 测试 ggml_cpy 在不同类型间复制张量数据的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// 测试 cpy 在 f32 类型上的基本复制
test "ggml_cpy f32 basic" {
    const mem_size: usize = 128 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // 创建源张量 10x11
    const src = try ctx.newTensor2d(.f32, 10, 11);
    const dst = try ctx.newTensor2d(.f32, 10, 11);

    // 填充源数据: arange
    const src_data = src.dataF32();
    for (src_data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i));
    }
    // 清零目标
    @memset(dst.dataF32(), 0);

    // 使用 ggml_cpy 进行复制
    const cpy_result = ggml.cpy(ctx, src, dst);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(cpy_result);

    // 计算
    _ = ggml.c.ggml_graph_compute_with_ctx(
        @ptrCast(ctx),
        @ptrCast(graph),
        1,
    );

    // 验证结果
    const dst_data = dst.dataF32();
    for (dst_data, 0..) |v, i| {
        try testing.expectEqual(@as(f32, @floatFromInt(i)), v);
    }
}

// 测试 cpy 在 i32 类型上的基本复制
test "ggml_cpy i32 basic" {
    const mem_size: usize = 128 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const src = try ctx.newTensor2d(.i32, 10, 11);
    const dst = try ctx.newTensor2d(.i32, 10, 11);

    const src_data = src.dataI32();
    for (src_data, 0..) |*v, i| {
        v.* = @as(i32, @intCast(i));
    }
    @memset(dst.dataI32(), 0);

    const cpy_result = ggml.cpy(ctx, src, dst);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(cpy_result);

    _ = ggml.c.ggml_graph_compute_with_ctx(
        @ptrCast(ctx),
        @ptrCast(graph),
        1,
    );

    const dst_data = dst.dataI32();
    for (dst_data, 0..) |v, i| {
        try testing.expectEqual(@as(i32, @intCast(i)), v);
    }
}
