//! ggml_arange 测试
//!
//! 对应 deps/ggml/tests/test-arange.cpp
//! 测试 ggml_arange 创建等差数列张量的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// 测试 arange 的基本功能：从 0 到 3，步长 1
test "ggml_arange basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // 创建 arange 张量: [0, 1, 2]
    const t = ggml.arange(ctx, 0.0, 3.0, 1.0);

    // 验证形状
    try testing.expectEqual(@as(i64, 3), t.ne()[0]);

    // 创建计算图并计算
    const backend = try ggml.backendCpuInit();
    defer ggml.backendFree(backend);

    const buft = ggml.backendCpuBufferType();
    var gallocr = try ggml.Gallocr.init(buft);
    defer gallocr.free();

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t);

    _ = gallocr.allocGraph(graph);
    ggml.backendCpuSetNThreads(backend, 4);
    _ = ggml.backendGraphCompute(backend, graph);

    // 读取结果
    const output = t.dataF32();

    try testing.expectEqual(@as(f32, 0.0), output[0]);
    try testing.expectEqual(@as(f32, 1.0), output[1]);
    try testing.expectEqual(@as(f32, 2.0), output[2]);
}

// 测试 arange 的步长参数（正序）
test "ggml_arange step" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // ggml_arange 要求 stop > start，步长必须为正
    const t = ggml.arange(ctx, 0.0, 5.0, 1.0);

    try testing.expectEqual(@as(i64, 5), t.ne()[0]);

    const backend = try ggml.backendCpuInit();
    defer ggml.backendFree(backend);

    const buft = ggml.backendCpuBufferType();
    var gallocr = try ggml.Gallocr.init(buft);
    defer gallocr.free();

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t);

    _ = gallocr.allocGraph(graph);
    ggml.backendCpuSetNThreads(backend, 4);
    _ = ggml.backendGraphCompute(backend, graph);

    const output = t.dataF32();

    try testing.expectEqual(@as(f32, 0.0), output[0]);
    try testing.expectEqual(@as(f32, 1.0), output[1]);
    try testing.expectEqual(@as(f32, 2.0), output[2]);
    try testing.expectEqual(@as(f32, 3.0), output[3]);
    try testing.expectEqual(@as(f32, 4.0), output[4]);
}
