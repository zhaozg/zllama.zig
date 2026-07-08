//! ggml_interpolate 测试
//!
//! 对应 deps/ggml/tests/test-interpolate.cpp
//! 测试 ggml_interpolate 在 nearest/bilinear 模式下的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

fn checkEqual(result: []const f32, expected: []const f32) bool {
    if (result.len != expected.len) return false;
    for (result, expected) |r, e| {
        if (@abs(r - e) > 1e-4) return false;
    }
    return true;
}

fn testInterpolate(name: []const u8, src_ne: [4]i64, src_data: []const f32, dst_ne: [4]i32, expected: []const f32, mode: u32) !bool {
    ggml.c.ggml_time_init();

    const mem_size: usize = 64 * @as(usize, @intCast(ggml.c.ggml_tensor_overhead())) + @as(usize, @intCast(ggml.c.ggml_graph_overhead()));
    var ctx = try ggml.Context.initNoAlloc(mem_size);
    defer ctx.deinit();

    // 创建源张量
    const src = ggml.c.ggml_new_tensor(
        @ptrCast(ctx),
        @intFromEnum(ggml.Type.f32),
        4,
        @ptrCast(@constCast(&src_ne)),
    );
    const src_tensor: *ggml.Tensor = @ptrCast(src);

    const res = ggml.interpolate(ctx, src_tensor, dst_ne[0], dst_ne[1], dst_ne[2], dst_ne[3], mode);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(res);

    const backend = try ggml.backendCpuInit();
    defer ggml.backendFree(backend);

    ggml.backendCpuSetNThreads(backend, 2);
    try ggml.backendAllocCtxTensors(ctx, backend);

    ggml.backendTensorSet(src_tensor, std.mem.sliceAsBytes(src_data), 0);
    _ = ggml.backendGraphCompute(backend, graph);

    const n_res = @as(usize, @intCast(ggml.c.ggml_nelements(@ptrCast(@alignCast(res)))));
    const res_values = try testing.allocator.alloc(f32, n_res);
    defer testing.allocator.free(res_values);
    ggml.backendTensorGet(res, std.mem.sliceAsBytes(res_values), 0);

    const passed = checkEqual(res_values, expected);
    if (!passed) {
        std.debug.print("FAIL: {s}\n", .{name});
    }
    return passed;
}

// 测试 upscale x2 nearest
test "ggml_interpolate upscale_x2_nearest" {
    const input = [_]f32{ 0.0, 1.0, 2.0, 4.0 };
    const expected = [_]f32{
        0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
        2.0, 2.0, 4.0, 4.0,
        2.0, 2.0, 4.0, 4.0,
    };
    try testing.expect(try testInterpolate(
        "upscale_x2_nearest",
        .{ 2, 2, 1, 1 },
        &input,
        .{ 4, 4, 1, 1 },
        &expected,
        @intFromEnum(ggml.ScaleMode.nearest),
    ));
}

// 测试 upscale x2 bilinear
test "ggml_interpolate upscale_x2_bilinear" {
    const input = [_]f32{ 0.0, 1.0, 2.0, 4.0 };
    const expected = [_]f32{
        0.0, 0.2500, 0.7500, 1.00,
        0.5, 0.8125, 1.4375, 1.75,
        1.5, 1.9375, 2.8125, 3.25,
        2.0, 2.5000, 3.5000, 4.00,
    };
    try testing.expect(try testInterpolate(
        "upscale_x2_bilinear",
        .{ 2, 2, 1, 1 },
        &input,
        .{ 4, 4, 1, 1 },
        &expected,
        @intFromEnum(ggml.ScaleMode.bilinear),
    ));
}

// 测试 upscale x2 bilinear align_corners
test "ggml_interpolate upscale_x2_bilinear_align_corners" {
    const input = [_]f32{ 0.0, 1.0, 2.0, 4.0 };
    const expected = [_]f32{
        0.0000, 0.3333, 0.6667, 1.0000,
        0.6667, 1.1111, 1.5556, 2.0000,
        1.3333, 1.8889, 2.4444, 3.0000,
        2.0000, 2.6667, 3.3333, 4.0000,
    };
    try testing.expect(try testInterpolate(
        "upscale_x2_bilinear_align_corners",
        .{ 2, 2, 1, 1 },
        &input,
        .{ 4, 4, 1, 1 },
        &expected,
        @intFromEnum(ggml.ScaleMode.bilinear) | @intFromEnum(ggml.ScaleFlag.align_corners),
    ));
}

// 测试 upscale x1.5 bilinear align_corners
test "ggml_interpolate upscale_x1_5_bilinear_align_corners" {
    const input = [_]f32{ 0.0, 1.0, 2.0, 4.0 };
    const expected = [_]f32{
        0.0, 1.0,
        1.0, 2.5,
        2.0, 4.0,
    };
    try testing.expect(try testInterpolate(
        "upscale_x1_5_bilinear_align_corners",
        .{ 2, 2, 1, 1 },
        &input,
        .{ 2, 3, 1, 1 },
        &expected,
        @intFromEnum(ggml.ScaleMode.bilinear) | @intFromEnum(ggml.ScaleFlag.align_corners),
    ));
}

// 测试 downscale nearest
test "ggml_interpolate downscale_nearest" {
    const input = [_]f32{
        0.0, -1.0, -2.0, 0.0,
        1.0, 2.0, 4.0, 4.0,
        2.0, 2.0, 1.0, 1.0,
        1.0, 2.0, 3.0, 4.0,
        2.0, 2.0, 2.0, 2.0,
        -2.0, 2.0, -4.0, 4.0,
    };
    const expected = [_]f32{
        0.0, -2.0,
        1.0, 3.0,
    };
    try testing.expect(try testInterpolate(
        "downscale_nearest",
        .{ 4, 3, 2, 1 },
        &input,
        .{ 2, 1, 2, 1 },
        &expected,
        @intFromEnum(ggml.ScaleMode.nearest),
    ));
}

// 测试 downscale bilinear
test "ggml_interpolate downscale_bilinear" {
    const input = [_]f32{
        0.0, -1.0, -2.0, 0.0,
        1.0, 2.0, 4.0, 4.0,
        2.0, 2.0, 1.0, 1.0,
        1.0, 2.0, 3.0, 4.0,
        2.0, 2.0, 2.0, 2.0,
        -2.0, 2.0, -4.0, 4.0,
    };
    const expected = [_]f32{
        0.1667, -0.3750, 0.7500,
        1.7917, 1.8750, 1.7500,
        1.3750, 2.3750, 3.3750,
        -0.5000, -0.2500, 2.5000,
    };
    try testing.expect(try testInterpolate(
        "downscale_bilinear",
        .{ 4, 3, 2, 1 },
        &input,
        .{ 3, 2, 2, 1 },
        &expected,
        @intFromEnum(ggml.ScaleMode.bilinear),
    ));
}

// 测试 downscale bilinear align_corners
test "ggml_interpolate downscale_bilinear_align_corners" {
    const input = [_]f32{
        0.0, -1.0, -2.0, 0.0,
        1.0, 2.0, 4.0, 4.0,
        2.0, 2.0, 1.0, 1.0,
        1.0, 2.0, 3.0, 4.0,
        2.0, 2.0, 2.0, 2.0,
        -2.0, 2.0, -4.0, 4.0,
    };
    const expected = [_]f32{
        0.0, -1.5, 0.0,
        2.0, 1.5, 1.0,
        1.0, 2.5, 4.0,
        -2.0, -1.0, 4.0,
    };
    try testing.expect(try testInterpolate(
        "downscale_bilinear_align_corners",
        .{ 4, 3, 2, 1 },
        &input,
        .{ 3, 2, 2, 1 },
        &expected,
        @intFromEnum(ggml.ScaleMode.bilinear) | @intFromEnum(ggml.ScaleFlag.align_corners),
    ));
}
