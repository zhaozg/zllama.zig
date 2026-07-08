//! ggml_rel_pos 测试
//!
//! 对应 deps/ggml/tests/test-rel-pos.c
//! 测试 ggml_get_rel_pos 和 ggml_add_rel_pos 的正确性。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// 测试相对位置编码的获取与添加
test "ggml_get_rel_pos and ggml_add_rel_pos" {
    const mem_size: usize = 2 * 1024 * 1024;
    var ctx = try ggml.Context.init(mem_size);
    defer ctx.deinit();

    // 准备 f16 数据
    var buf_f16: [1024]u16 = undefined;
    for (&buf_f16, 0..) |*v, i| {
        v.* = @as(u16, @bitCast(ggml.c.ggml_fp32_to_fp16(@as(f32, @floatFromInt(i)))));
    }

    // 创建 3x3 f16 张量
    const t = try ctx.newTensor2d(.f16, 3, 3);
    const t_2 = try ctx.newTensor2d(.f16, 3, 3);

    // 复制数据
    @memcpy(t.dataBytes(), std.mem.sliceAsBytes(buf_f16[0..9]));
    @memcpy(t_2.dataBytes(), std.mem.sliceAsBytes(buf_f16[1..10]));

    // 获取相对位置编码
    const rw = ggml.getRelPos(ctx, t, 2, 2);
    const rh = ggml.getRelPos(ctx, t_2, 2, 2);

    // 转换为 f32
    const rw_f32_dst = try ctx.newTensor3d(.f32, 3, 2, 2);
    const rh_f32_dst = try ctx.newTensor3d(.f32, 3, 2, 2);
    const rw_f32 = ggml.cpy(ctx, rw, rw_f32_dst);
    const rh_f32 = ggml.cpy(ctx, rh, rh_f32_dst);

    // 创建输入张量 (9, 4)
    const in_tensor = try ctx.newTensor2d(.f32, 9, 4);
    const out_inplace = try ctx.newTensor2d(.f32, 9, 4);

    // 全部填充为 1.0
    @memset(in_tensor.dataF32(), 1.0);
    @memset(out_inplace.dataF32(), 1.0);

    // 计算 add_rel_pos
    const out = ggml.addRelPos(ctx, in_tensor, rw_f32, rh_f32);

    var graph = try ggml.CGraph.init(ctx);
    graph.buildForwardExpand(out);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph), 1);

    // 计算 add_rel_pos_inplace
    const out_ip = ggml.addRelPosInplace(ctx, out_inplace, rw_f32, rh_f32);

    var graph2 = try ggml.CGraph.init(ctx);
    graph2.buildForwardExpand(out_ip);
    _ = ggml.c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(graph2), 1);

    // 预期输出
    const expected = [_]f32{
        8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0,
        2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0,
        14.0, 15.0, 16.0, 15.0, 16.0, 17.0, 16.0, 17.0, 18.0,
        8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0,
    };

    // 验证 out
    {
        const result = out.dataF32();
        for (expected, 0..) |exp, i| {
            try testing.expectEqual(exp, result[i]);
        }
    }

    // 验证 out_inplace
    {
        const result = out_ip.dataF32();
        for (expected, 0..) |exp, i| {
            try testing.expectEqual(exp, result[i]);
        }
    }
}
