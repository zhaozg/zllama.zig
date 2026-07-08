//! ggml_roll 测试
//!
//! 对应 deps/ggml/tests/test-roll.cpp
//! 测试 ggml_roll 在不同维度上滚动张量元素的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

/// 参考实现：滚动张量
fn rollReference(src: []const f32, ne: [4]i64, shift: [4]i32) std.array_list.AlignedManaged(f32, null) {
    const ne0 = ne[0];
    const ne1 = ne[1];
    const ne2 = ne[2];
    const ne3 = ne[3];
    const total = @as(usize, @intCast(ne0 * ne1 * ne2 * ne3));

    var dst = std.array_list.AlignedManaged(f32, null).init(testing.allocator);
    dst.resize(total) catch @panic("OOM");
    const dst_slice = dst.items;

    var idx3: i64 = 0;
    while (idx3 < ne3) : (idx3 += 1) {
        var idx2: i64 = 0;
        while (idx2 < ne2) : (idx2 += 1) {
            var idx1: i64 = 0;
            while (idx1 < ne1) : (idx1 += 1) {
                var idx0: i64 = 0;
                while (idx0 < ne0) : (idx0 += 1) {
                    const src_i03 = wrap(idx3 - shift[3], ne3);
                    const src_i02 = wrap(idx2 - shift[2], ne2);
                    const src_i01 = wrap(idx1 - shift[1], ne1);
                    const src_i00 = wrap(idx0 - shift[0], ne0);

                    const dst_idx = @as(usize, @intCast(idx3 * (ne2 * ne1 * ne0) + idx2 * (ne1 * ne0) + idx1 * ne0 + idx0));
                    const src_idx = @as(usize, @intCast(src_i03 * (ne2 * ne1 * ne0) + src_i02 * (ne1 * ne0) + src_i01 * ne0 + src_i00));
                    dst_slice[dst_idx] = src[src_idx];
                }
            }
        }
    }
    return dst;
}

fn wrap(i: i64, ne: i64) i64 {
    if (i < 0) return i + ne;
    if (i >= ne) return i - ne;
    return i;
}

fn f32Range(n: usize) std.array_list.AlignedManaged(f32, null) {
    var values = std.array_list.AlignedManaged(f32, null).init(testing.allocator);
    values.resize(n) catch @panic("OOM");
    for (values.items, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i));
    }
    return values;
}

fn checkEqual(result: []const f32, expected: []const f32) bool {
    if (result.len != expected.len) return false;
    for (result, expected) |r, e| {
        if (@abs(r - e) > 1e-5) return false;
    }
    return true;
}

// 测试 roll 的基本功能
test "ggml_roll basic" {
    const ne = [_]i64{ 3, 7, 4, 2 };
    const shift = [_]i32{ 1, 0, -1, 0 };

    try testRoll(ne, shift, false);
}

// 测试 roll 的负偏移
test "ggml_roll negative shift" {
    const ne = [_]i64{ 37, 42, 59, 2 };
    const shift = [_]i32{ -4, 3, -7, 1 };

    try testRoll(ne, shift, false);
}

// 测试 roll 在 permute 后的张量上
test "ggml_roll permuted" {
    const ne = [_]i64{ 37, 42, 59, 2 };
    const shift = [_]i32{ -4, 3, -7, 1 };

    try testRoll(ne, shift, true);
}

fn testRoll(ne: [4]i64, shift: [4]i32, permute: bool) !void {
    ggml.c.ggml_time_init();

    const mem_size: usize = 64 * @as(usize, @intCast(ggml.c.ggml_tensor_overhead())) + @as(usize, @intCast(ggml.c.ggml_graph_overhead()));
    var ctx = try ggml.Context.initNoAlloc(mem_size);
    defer ctx.deinit();

    // 创建源张量
    const src = ggml.c.ggml_new_tensor(
        @ptrCast(ctx),
        @intFromEnum(ggml.Type.f32),
        4,
        @ptrCast(@constCast(&ne)),
    );
    const src_tensor: *ggml.Tensor = @ptrCast(src);

    var res: *ggml.Tensor = undefined;
    if (!permute) {
        res = ggml.roll(ctx, src_tensor, shift[0], shift[1], shift[2], shift[3]);
    } else {
        const p = ggml.permute(ctx, src_tensor, 0, 3, 1, 2);
        const r = ggml.roll(ctx, p, shift[0], shift[2], shift[3], shift[1]);
        const p2 = ggml.permute(ctx, r, 0, 2, 3, 1);
        res = ggml.cont(ctx, p2);
    }

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(res);

    // 创建 backend
    const backend = try ggml.backendCpuInit();
    defer ggml.backendFree(backend);

    ggml.backendCpuSetNThreads(backend, 2);
    try ggml.backendAllocCtxTensors(ctx, backend);

    // 填充源数据
    const src_values = f32Range(@as(usize, @intCast(ggml.c.ggml_nelements(src))));
    defer src_values.deinit();
    ggml.backendTensorSet(src_tensor, std.mem.sliceAsBytes(src_values.items), 0);

    // 计算
    _ = ggml.backendGraphCompute(backend, graph);

    // 读取结果
    const n_res = @as(usize, @intCast(ggml.c.ggml_nelements(@ptrCast(@alignCast(res)))));
    const res_values = try testing.allocator.alloc(f32, n_res);
    defer testing.allocator.free(res_values);
    ggml.backendTensorGet(res, std.mem.sliceAsBytes(res_values), 0);

    // 参考结果
    const expected = rollReference(src_values.items, ne, shift);
    defer expected.deinit();

    try testing.expect(checkEqual(res_values, expected.items));
}
