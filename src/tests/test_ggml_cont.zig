//! ggml_cont 测试
//!
//! 对应 deps/ggml/tests/test-cont.c
//! 测试 ggml_cont 在转置后使张量连续的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// 测试 cont 在 f32 和 f16 类型上的正确性
test "ggml_cont f32 and f16" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.initNoAlloc(mem_size);
    defer ctx.deinit();

    // 创建输入张量
    const in_f32 = try ctx.newTensor1d(.f32, 2);
    const in_f16 = try ctx.newTensor1d(.f16, 2);

    // 初始化 backend 并分配张量内存
    const backend = try ggml.backendCpuInit();
    defer ggml.backendFree(backend);

    try ggml.backendAllocCtxTensors(ctx, backend);

    // 设置输入数据
    const f32_data = [_]f32{ 1.0, 2.0 };
    const f16_data = [_]u16{
        @as(u16, @bitCast(ggml.c.ggml_fp32_to_fp16(1.0))),
        @as(u16, @bitCast(ggml.c.ggml_fp32_to_fp16(2.0))),
    };
    ggml.backendTensorSet(in_f32, std.mem.sliceAsBytes(&f32_data), 0);
    ggml.backendTensorSet(in_f16, std.mem.sliceAsBytes(&f16_data), 0);

    // 构建图：transpose + cont
    const t_f32 = ggml.transpose(ctx, in_f32);
    const out_f32 = ggml.cont(ctx, t_f32);

    const t_f16 = ggml.transpose(ctx, in_f16);
    const out_f16 = ggml.cont(ctx, t_f16);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(out_f32);
    graph.buildForwardExpand(out_f16);

    const buft = ggml.backendCpuBufferType();
    var gallocr = try ggml.Gallocr.init(buft);
    defer gallocr.free();

    _ = gallocr.allocGraph(graph);
    ggml.backendCpuSetNThreads(backend, 4);
    _ = ggml.backendGraphCompute(backend, graph);

    // 验证 f32 输出
    {
        const expected = [_]f32{ 1.0, 2.0 };
        try testing.expectEqual(@as(i64, 1), out_f32.ne()[0]);
        try testing.expectEqual(@as(i64, 2), out_f32.ne()[1]);

        var buf: [2]f32 = undefined;
        ggml.backendTensorGet(out_f32, std.mem.sliceAsBytes(&buf), 0);
        try testing.expectEqual(expected[0], buf[0]);
        try testing.expectEqual(expected[1], buf[1]);
    }

    // 验证 f16 输出
    {
        try testing.expectEqual(@as(i64, 1), out_f16.ne()[0]);
        try testing.expectEqual(@as(i64, 2), out_f16.ne()[1]);

        var buf: [2]u16 = undefined;
        ggml.backendTensorGet(out_f16, std.mem.sliceAsBytes(&buf), 0);
        const v0 = ggml.c.ggml_fp16_to_fp32(@as(ggml.c.ggml_fp16_t, @bitCast(buf[0])));
        const v1 = ggml.c.ggml_fp16_to_fp32(@as(ggml.c.ggml_fp16_t, @bitCast(buf[1])));
        try testing.expectEqual(@as(f32, 1.0), v0);
        try testing.expectEqual(@as(f32, 2.0), v1);
    }
}
