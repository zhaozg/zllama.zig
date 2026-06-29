//! 归一化层构建器
//!
//! 提供 LayerNorm 和 RMSNorm 的 ggml 计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_norm()

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");

const NormType = types.NormType;

const log = std.log.scoped(.graph_norm);

/// 构建归一化层
///
/// 参数:
///   - ctx: ggml 上下文
///   - cur: 输入张量 [n_embd, n_patches]
///   - mw: 权重张量 [n_embd]
///   - mb: 偏置张量 [n_embd]（可选，RMSNorm 不需要）
///   - norm_type: 归一化类型
///   - norm_eps: epsilon
///   - name: 张量名称前缀（用于调试）
///
/// 返回: 归一化后的张量 [n_embd, n_patches]
///
/// 参考: clip-graph.h build_norm()
pub fn buildNorm(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    mw: *ggml.Tensor,
    mb: ?*ggml.Tensor,
    norm_type: NormType,
    norm_eps: f32,
    name: []const u8,
) !*ggml.Tensor {
    var result = cur;

    switch (norm_type) {
        .layer_norm => {
            result = result.norm(ctx, norm_eps);
            result.setName(name);
        },
        .rms_norm => {
            result = result.rmsNorm(ctx, norm_eps);
            result.setName(name);
        },
    }

    // 广播乘法: mw [n_embd] × result [n_embd, n_patches]
    result = result.mul(ctx, reshapeForBroadcast(ctx, mw));
    result.setName(name);

    if (mb) |b| {
        result = result.add(ctx, b);
        result.setName(name);
    }

    return result;
}

/// 将 1D 权重张量 [n] 重塑为 [n, 1] 以便与 [n_embd, n_patches] 张量广播。
/// ggml 广播规则: b=[n, 1] vs a=[n_embd, n_patches]
///   ne[0]: n == n_embd (ok)
///   ne[1]: 1 <= n_patches (ok)
pub fn reshapeForBroadcast(ctx: *ggml.Context, t: *ggml.Tensor) *ggml.Tensor {
    const n = t.ne()[0];
    return ctx.view2d(t, n, 1, ggml.Type.rowSize(t.dataType(), n), 0);
}

test "buildNorm: LayerNorm basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_embd: i64 = 64;
    const n_patches: i64 = 16;

    const cur = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
    const mw = try ctx.newTensor1d(ggml.Type.f32, n_embd);
    const mb = try ctx.newTensor1d(ggml.Type.f32, n_embd);

    // Fill with simple values
    @memset(cur.dataF32(), 1.0);
    @memset(mw.dataF32(), 1.0);
    @memset(mb.dataF32(), 0.0);

    const result = try buildNorm(&ctx, cur, mw, mb, .layer_norm, 1e-5, "test_norm");
    try testing.expectEqual(@as(i64, n_embd), result.ne()[0]);
    try testing.expectEqual(@as(i64, n_patches), result.ne()[1]);
}

test "buildNorm: RMSNorm basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_embd: i64 = 64;
    const n_patches: i64 = 16;

    const cur = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
    const mw = try ctx.newTensor1d(ggml.Type.f32, n_embd);

    @memset(cur.dataF32(), 1.0);
    @memset(mw.dataF32(), 1.0);

    const result = try buildNorm(&ctx, cur, mw, null, .rms_norm, 1e-5, "test_rms");
    try testing.expectEqual(@as(i64, n_embd), result.ne()[0]);
    try testing.expectEqual(@as(i64, n_patches), result.ne()[1]);
}
