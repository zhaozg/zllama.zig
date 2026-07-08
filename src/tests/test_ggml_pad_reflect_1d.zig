//! ggml_pad_reflect_1d 测试
//!
//! 对应 deps/ggml/tests/test-pad-reflect-1d.cpp
//! 测试 ggml_pad_reflect_1d 在 1D 和 2D 张量上的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// 检查张量值与预期值是否一致
fn checkTensor(t: *ggml.Tensor, expected: []const f32, ne0: i64, ne1: i64, ne2: i64) !void {
    try testing.expectEqual(ne0, t.ne()[0]);
    try testing.expectEqual(ne1, t.ne()[1]);
    try testing.expectEqual(ne2, t.ne()[2]);

    const output = t.dataF32();
    for (expected, 0..) |exp, i| {
        if (output[i] != exp) {
            std.debug.print("expected {d:.1}, got {d:.1} at [{d}]\n", .{ exp, output[i], i });
        }
        try testing.expectEqual(exp, output[i]);
    }
}

// 测试 1D 张量的反射填充
test "ggml_pad_reflect_1d 1D tensor" {
    const mem_size: usize = 16 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // 创建 1D 输入 [1, 2, 3, 4]
    const t = try ctx.newTensor1d(.f32, 4);
    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    @memcpy(t.dataF32(), &input_data);

    // Test case 1: pad left=1, right=1
    // Expected: [2, 1, 2, 3, 4, 3]
    const expected_1 = [_]f32{ 2.0, 1.0, 2.0, 3.0, 4.0, 3.0 };
    const out_1 = ggml.padReflect1d(ctx, t, 1, 1);

    // Test case 2: pad left=2, right=1
    // Expected: [3, 2, 1, 2, 3, 4, 3]
    const expected_2 = [_]f32{ 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0 };
    const out_2 = ggml.padReflect1d(ctx, t, 2, 1);

    // Test case 3: pad left=1, right=2
    // Expected: [2, 1, 2, 3, 4, 3, 2]
    const expected_3 = [_]f32{ 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0 };
    const out_3 = ggml.padReflect1d(ctx, t, 1, 2);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(out_1);
    graph.buildForwardExpand(out_2);
    graph.buildForwardExpand(out_3);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    try checkTensor(out_1, &expected_1, 6, 1, 1);
    try checkTensor(out_2, &expected_2, 7, 1, 1);
    try checkTensor(out_3, &expected_3, 7, 1, 1);
}

// 测试 2D 张量的反射填充（每行独立填充）
test "ggml_pad_reflect_1d 2D tensor" {
    const mem_size: usize = 16 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // 创建 2D 输入 (5 cols x 4 rows)
    const t = try ctx.newTensor2d(.f32, 5, 4);
    const input_data = [_]f32{
        1.0, 2.0, 3.0, 4.0, 5.0, // row 1
        6.0, 7.0, 8.0, 9.0, 10.0, // row 2
        11.0, 12.0, 13.0, 14.0, 15.0, // row 3
        16.0, 17.0, 18.0, 19.0, 20.0, // row 4
    };
    @memcpy(t.dataF32(), &input_data);

    // Test case 4: pad left=3, right=2 on a 2D tensor
    const expected_4 = [_]f32{
        4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, // row 1
        9.0, 8.0, 7.0, 6.0, 7.0, 8.0, 9.0, 10.0, 9.0, 8.0, // row 2
        14.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, // row 3
        19.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0, // row 4
    };
    const out_4 = ggml.padReflect1d(ctx, t, 3, 2);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(out_4);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    try checkTensor(out_4, &expected_4, 10, 4, 1);
}
