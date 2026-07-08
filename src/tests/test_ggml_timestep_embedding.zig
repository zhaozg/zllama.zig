//! ggml_timestep_embedding 测试
//!
//! 对应 deps/ggml/tests/test-timestep_embedding.cpp
//! 测试 ggml_timestep_embedding 的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// 测试 timestep_embedding 的基本功能
test "ggml_timestep_embedding basic" {
    const ts = [_]f32{ 12.0, 24.0 };
    const dim: i32 = 15;
    const max_period: i32 = 10000;

    const mem_size: usize = 16 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    const timesteps = try ctx.newTensor1d(.f32, 2);
    @memcpy(timesteps.dataF32(), &ts);

    const t = ggml.timestepEmbedding(ctx, timesteps, dim, max_period);

    try testing.expectEqual(@as(i64, dim), t.ne()[0]);
    try testing.expectEqual(@as(i64, 2), t.ne()[1]);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(t);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 4);

    const output = t.dataF32();
    const n_elems = @as(usize, @intCast(ggml.c.ggml_nelements(@ptrCast(@alignCast(t)))));

    // 验证输出不为全零
    var has_nonzero = false;
    for (output[0..n_elems]) |v| {
        if (v != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);

    // 验证输出不包含 NaN
    for (output[0..n_elems]) |v| {
        try testing.expect(!std.math.isNan(v));
    }

    // 验证形状正确
    try testing.expectEqual(@as(usize, @intCast(dim * 2)), n_elems);
}
