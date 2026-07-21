//! ggml_conv 测试
//!
//! 对应 deps/ggml/tests/test-conv1d.cpp, test-conv2d.cpp
//! 测试 ggml_conv_1d, ggml_conv_2d 等卷积算子的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// ============================================================================
// conv1d 测试
// ============================================================================

test "ggml_conv1d basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor3d(.f32, 4, 1, 1);
    const kernel = try ctx.newTensor3d(.f32, 3, 1, 1);

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const kernel_data = [_]f32{ 1.0, 0.0, -1.0 };
    @memcpy(input.dataF32(), &input_data);
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.conv1d(ctx, kernel, input, 1, 0, 1);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = result.dataF32();
    try testing.expectEqual(@as(f32, -2.0), output[0]);
    try testing.expectEqual(@as(f32, -2.0), output[1]);
}

test "ggml_conv1d stride2" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor3d(.f32, 6, 1, 1);
    const kernel = try ctx.newTensor3d(.f32, 3, 1, 1);

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const kernel_data = [_]f32{ 1.0, 0.0, -1.0 };
    @memcpy(input.dataF32(), &input_data);
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.conv1d(ctx, kernel, input, 2, 0, 1);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = result.dataF32();
    try testing.expectEqual(@as(f32, -2.0), output[0]);
    try testing.expectEqual(@as(f32, -2.0), output[1]);
}

test "ggml_conv1d padding" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor3d(.f32, 4, 1, 1);
    const kernel = try ctx.newTensor3d(.f32, 3, 1, 1);

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const kernel_data = [_]f32{ 1.0, 0.0, -1.0 };
    @memcpy(input.dataF32(), &input_data);
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.conv1d(ctx, kernel, input, 1, 1, 1);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = result.dataF32();
    try testing.expectEqual(@as(f32, -2.0), output[0]);
    try testing.expectEqual(@as(f32, -2.0), output[1]);
    try testing.expectEqual(@as(f32, -2.0), output[2]);
    try testing.expectEqual(@as(f32, 3.0), output[3]);
}

// ============================================================================
// conv2d 测试
// ============================================================================

test "ggml_conv2d basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor4d(.f32, 4, 4, 1, 1);
    const kernel = try ctx.newTensor4d(.f32, 3, 3, 1, 1);

    var input_buf: [16]f32 = undefined;
    for (&input_buf, 0..) |*v, i| { v.* = @as(f32, @floatFromInt(i + 1)); }
    @memcpy(input.dataF32(), &input_buf);

    const kernel_data = [_]f32{
        -1.0, -1.0, -1.0,
        -1.0,  8.0, -1.0,
        -1.0, -1.0, -1.0,
    };
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.conv2d(ctx, kernel, input, 1, 1, 0, 0, 1, 1);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    try testing.expectEqual(@as(i64, 2), result.ne()[0]);
    try testing.expectEqual(@as(i64, 2), result.ne()[1]);

    const output = result.dataF32();
    try testing.expectEqual(@as(f32, 0.0), output[0]);
    try testing.expectEqual(@as(f32, 0.0), output[1]);
    try testing.expectEqual(@as(f32, 0.0), output[2]);
    try testing.expectEqual(@as(f32, 0.0), output[3]);
}

// ============================================================================
// conv1d_dw (depthwise) 测试
// ============================================================================

test "ggml_conv1d_dw basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor3d(.f32, 4, 2, 1);
    const kernel = try ctx.newTensor3d(.f32, 3, 1, 2);

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const kernel_data = [_]f32{ 1.0, 0.0, -1.0, 0.5, 1.0, 0.5 };
    @memcpy(input.dataF32(), &input_data);
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.conv1dDw(ctx, kernel, input, 1, 0, 1);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = result.dataF32();
    try testing.expectEqual(@as(f32, -2.0), output[0]);
    try testing.expectEqual(@as(f32, -2.0), output[1]);
    try testing.expectEqual(@as(f32, 12.0), output[2]);
    try testing.expectEqual(@as(f32, 14.0), output[3]);
}

// ============================================================================
// conv_transpose_1d 测试
// ============================================================================

test "ggml_conv_transpose_1d basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor3d(.f32, 2, 1, 1);
    const kernel = try ctx.newTensor3d(.f32, 3, 1, 1);

    const input_data = [_]f32{ 1.0, 2.0 };
    const kernel_data = [_]f32{ 1.0, 0.0, -1.0 };
    @memcpy(input.dataF32(), &input_data);
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.convTranspose1d(ctx, kernel, input, 2, 0, 1);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    try testing.expectEqual(@as(i64, 5), result.ne()[0]);

    const output = result.dataF32();
    try testing.expectEqual(@as(f32, 1.0), output[0]);
    try testing.expectEqual(@as(f32, 0.0), output[1]);
    try testing.expectEqual(@as(f32, 1.0), output[2]);
    try testing.expectEqual(@as(f32, 0.0), output[3]);
    try testing.expectEqual(@as(f32, -2.0), output[4]);
}

// ============================================================================
// conv2d_sk_p0 测试
// ============================================================================

test "ggml_conv2d_sk_p0 basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor4d(.f32, 6, 6, 1, 1);
    const kernel = try ctx.newTensor4d(.f32, 3, 3, 1, 1);

    var input_buf: [36]f32 = undefined;
    for (&input_buf, 0..) |*v, i| { v.* = @as(f32, @floatFromInt(i + 1)); }
    @memcpy(input.dataF32(), &input_buf);

    const kernel_data = [_]f32{ 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0 };
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.conv2dSkP0(ctx, kernel, input);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    try testing.expectEqual(@as(i64, 2), result.ne()[0]);
    try testing.expectEqual(@as(i64, 2), result.ne()[1]);

    const output = result.dataF32();
    try testing.expectEqual(@as(f32, -6.0), output[0]);
}

// ============================================================================
// conv2d_s1_ph 测试
// ============================================================================

test "ggml_conv2d_s1_ph basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor4d(.f32, 4, 4, 1, 1);
    const kernel = try ctx.newTensor4d(.f32, 3, 3, 1, 1);

    var input_buf: [16]f32 = undefined;
    for (&input_buf, 0..) |*v, i| { v.* = @as(f32, @floatFromInt(i + 1)); }
    @memcpy(input.dataF32(), &input_buf);

    const kernel_data = [_]f32{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.conv2dS1Ph(ctx, kernel, input);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    try testing.expectEqual(@as(i64, 4), result.ne()[0]);
    try testing.expectEqual(@as(i64, 4), result.ne()[1]);

    const output = result.dataF32();
    for (input_buf, 0..) |exp, i| { try testing.expectEqual(exp, output[i]); }
}

// ============================================================================
// conv_transpose_2d_p0 测试
// ============================================================================

test "ggml_conv_transpose_2d_p0 basic" {
    const mem_size: usize = 64 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const input = try ctx.newTensor4d(.f32, 2, 2, 1, 1);
    const kernel = try ctx.newTensor4d(.f32, 3, 3, 1, 1);

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const kernel_data = [_]f32{ 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0 };
    @memcpy(input.dataF32(), &input_data);
    @memcpy(kernel.dataF32(), &kernel_data);

    const result = ggml.convTranspose2dP0(ctx, kernel, input, 2);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(result);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    try testing.expectEqual(@as(i64, 5), result.ne()[0]);
    try testing.expectEqual(@as(i64, 5), result.ne()[1]);

    const output = result.dataF32();
    var has_nonzero = false;
    for (output[0..25]) |v| { if (v != 0.0) { has_nonzero = true; break; } }
    try testing.expect(has_nonzero);
}
