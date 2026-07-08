//! ggml_custom_op 测试
//!
//! 对应 deps/ggml/tests/test-customop.c
//! 测试 ggml_map_custom1/2/3 和 ggml_custom_4d 自定义算子的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

var g_custom1_count: i32 = 0;
var g_custom2_count: i32 = 0;
var g_custom3_count: i32 = 0;

const g_userdata = "ggml";

// 自定义一元算子：dst[i] = a[i] * 2
fn custom1(dst: [*c]ggml.c.struct_ggml_tensor, a: [*c]const ggml.c.struct_ggml_tensor, ith: c_int, nth: c_int, userdata: ?*anyopaque) callconv(.c) void {
    _ = userdata;
    _ = ggml.c.ggml_are_same_shape(dst, a);

    _ = @atomicRmw(c_int, &g_custom1_count, .Add, 1, .monotonic);

    const a_data = ggml.c.ggml_get_data_f32(a);
    const dst_data = ggml.c.ggml_get_data_f32(dst);

    const ne: c_int = @intCast(ggml.c.ggml_nelements(dst));
    const dr = @divTrunc(ne + nth - 1, nth);
    const ie0 = dr * ith;
    const ie1 = @min(ie0 + dr, ne);

    var i: c_int = ie0;
    while (i < ie1) : (i += 1) {
        dst_data[@as(usize, @intCast(i))] = a_data[@as(usize, @intCast(i))] * 2.0;
    }
}

// 自定义二元算子：dst[i] = a[i] + b[i]
fn custom2(dst: [*c]ggml.c.struct_ggml_tensor, a: [*c]const ggml.c.struct_ggml_tensor, b: [*c]const ggml.c.struct_ggml_tensor, ith: c_int, nth: c_int, userdata: ?*anyopaque) callconv(.c) void {
    _ = userdata;
    _ = ggml.c.ggml_are_same_shape(dst, a);
    _ = ggml.c.ggml_are_same_shape(dst, b);

    _ = @atomicRmw(c_int, &g_custom2_count, .Add, 1, .monotonic);

    const a_data = ggml.c.ggml_get_data_f32(a);
    const b_data = ggml.c.ggml_get_data_f32(b);
    const dst_data = ggml.c.ggml_get_data_f32(dst);

    const nr: c_int = @intCast(ggml.c.ggml_nrows(dst));
    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);
    const dst_ptr: *ggml.c.struct_ggml_tensor = @ptrCast(dst);
    const nc: c_int = @intCast(dst_ptr.ne[0]);

    var ir: c_int = ir0;
    while (ir < ir1) : (ir += 1) {
        var ic: c_int = 0;
        while (ic < nc) : (ic += 1) {
            const idx = @as(usize, @intCast(ir * nc + ic));
            dst_data[idx] = a_data[idx] + b_data[idx];
        }
    }
}

// 自定义三元算子：dst[i] = a[i] + b[i] + c[i]
fn custom3(dst: [*c]ggml.c.struct_ggml_tensor, a: [*c]const ggml.c.struct_ggml_tensor, b: [*c]const ggml.c.struct_ggml_tensor, c: [*c]const ggml.c.struct_ggml_tensor, ith: c_int, nth: c_int, userdata: ?*anyopaque) callconv(.c) void {
    _ = userdata;
    _ = ith;
    _ = nth;
    _ = ggml.c.ggml_are_same_shape(dst, a);
    _ = ggml.c.ggml_are_same_shape(dst, b);
    _ = ggml.c.ggml_are_same_shape(dst, c);

    _ = @atomicRmw(c_int, &g_custom3_count, .Add, 1, .monotonic);

    const a_data = ggml.c.ggml_get_data_f32(a);
    const b_data = ggml.c.ggml_get_data_f32(b);
    const c_data = ggml.c.ggml_get_data_f32(c);
    const dst_data = ggml.c.ggml_get_data_f32(dst);

    const ne = @as(usize, @intCast(ggml.c.ggml_nelements(dst)));
    var i: usize = 0;
    while (i < ne) : (i += 1) {
        dst_data[i] = a_data[i] + b_data[i] + c_data[i];
    }
}

// 自定义 5 输入算子：dst[i] = src0[i] + src1[i] * src2[i] - src3[i] * src4[i]
fn custom(dst: [*c]ggml.c.struct_ggml_tensor, ith: c_int, nth: c_int, userdata: ?*anyopaque) callconv(.c) void {
    _ = userdata;
    const dst_ptr: *ggml.c.struct_ggml_tensor = @ptrCast(dst);
    const src0 = dst_ptr.src[0];
    const src1 = dst_ptr.src[1];
    const src2 = dst_ptr.src[2];
    const src3 = dst_ptr.src[3];
    const src4 = dst_ptr.src[4];

    const dst_data: [*]i32 = @ptrCast(@alignCast(ggml.c.ggml_get_data(dst)));
    const src0_data = ggml.c.ggml_get_data_f32(src0);
    const src1_data = ggml.c.ggml_get_data_f32(src1);
    const src2_data = ggml.c.ggml_get_data_f32(src2);
    const src3_data = ggml.c.ggml_get_data_f32(src3);
    const src4_data = ggml.c.ggml_get_data_f32(src4);

    const ne = @as(usize, @intCast(ggml.c.ggml_nelements(dst)));
    var i: usize = @intCast(ith);
    while (i < ne) : (i += @as(usize, @intCast(nth))) {
        dst_data[i] = @as(i32, @intFromFloat(src0_data[i] + src1_data[i] * src2_data[i] - src3_data[i] * src4_data[i]));
    }
}

// 测试 map_custom1
test "ggml_map_custom1" {
    g_custom1_count = 0;
    const mem_size: usize = 1 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf1: [1024]f32 = undefined;
    for (&buf1, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }

    const t = try ctx.newTensor2d(.f32, 10, 2);
    @memcpy(t.dataF32(), buf1[0..20]);

    const m1 = ggml.mapCustom1(ctx, t, custom1, 2, null);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(m1);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = m1.dataF32();
    for (output, 0..) |v, i| {
        try testing.expectEqual(buf1[i] * 2.0, v);
    }
    try testing.expectEqual(@as(i32, 2), g_custom1_count);
}

// 测试 map_custom2
test "ggml_map_custom2" {
    g_custom2_count = 0;
    const mem_size: usize = 1 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf1: [1024]f32 = undefined;
    var buf2: [1024]f32 = undefined;
    for (&buf1, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }
    for (&buf2, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 2.0;
    }

    const t1 = try ctx.newTensor2d(.f32, 10, 2);
    const t2 = try ctx.newTensor2d(.f32, 10, 2);
    @memcpy(t1.dataF32(), buf1[0..20]);
    @memcpy(t2.dataF32(), buf2[0..20]);

    const m2 = ggml.mapCustom2(ctx, t1, t2, custom2, ggml.n_tasks_max, @ptrCast(@constCast(&g_userdata)));

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(m2);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = m2.dataF32();
    for (output, 0..) |v, i| {
        try testing.expectEqual(buf1[i] + buf2[i], v);
    }
    try testing.expect(g_custom2_count > 0);
}

// 测试 map_custom3
test "ggml_map_custom3" {
    g_custom3_count = 0;
    const mem_size: usize = 1 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf1: [1024]f32 = undefined;
    var buf2: [1024]f32 = undefined;
    var buf3: [1024]f32 = undefined;
    for (&buf1, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }
    for (&buf2, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 2.0;
    }
    for (&buf3, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 3.0;
    }

    const t1 = try ctx.newTensor2d(.f32, 10, 2);
    const t2 = try ctx.newTensor2d(.f32, 10, 2);
    const t3 = try ctx.newTensor2d(.f32, 10, 2);
    @memcpy(t1.dataF32(), buf1[0..20]);
    @memcpy(t2.dataF32(), buf2[0..20]);
    @memcpy(t3.dataF32(), buf3[0..20]);

    const m3 = ggml.mapCustom3(ctx, t1, t2, t3, custom3, 1, @ptrCast(@constCast(&g_userdata)));

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(m3);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = m3.dataF32();
    for (output, 0..) |v, i| {
        try testing.expectEqual(buf1[i] + buf2[i] + buf3[i], v);
    }
    try testing.expectEqual(@as(i32, 1), g_custom3_count);
}

// 测试 custom_4d（多输入自定义算子）
test "ggml_custom_4d" {
    const mem_size: usize = 1 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    var buf1: [1024]f32 = undefined;
    var buf2: [1024]f32 = undefined;
    var buf3: [1024]f32 = undefined;
    for (&buf1, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }
    for (&buf2, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 2.0;
    }
    for (&buf3, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 3.0;
    }

    const t1 = try ctx.newTensor2d(.f32, 10, 2);
    const t2 = try ctx.newTensor2d(.f32, 10, 2);
    const t3 = try ctx.newTensor2d(.f32, 10, 2);
    const t4 = try ctx.newTensor2d(.f32, 10, 2);
    const t5 = try ctx.newTensor2d(.f32, 10, 2);
    @memcpy(t1.dataF32(), buf1[0..20]);
    @memcpy(t2.dataF32(), buf2[0..20]);
    @memcpy(t3.dataF32(), buf3[0..20]);
    @memcpy(t4.dataF32(), buf1[0..20]);
    @memcpy(t5.dataF32(), buf2[0..20]);

    const args = [_]*ggml.Tensor{ t1, t2, t3, t4, t5 };
    const m4 = ggml.custom4d(ctx, .i32, 10, 2, 1, 1, &args, custom, ggml.n_tasks_max, @ptrCast(@constCast(&g_userdata)));

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(m4);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = m4.dataI32();
    for (output, 0..) |v, i| {
        const expected = buf1[i] + buf2[i] * buf3[i] - buf1[i] * buf2[i];
        try testing.expectEqual(@as(i32, @intFromFloat(expected)), v);
    }
}
