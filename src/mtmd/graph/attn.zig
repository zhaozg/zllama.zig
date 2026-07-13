//! 注意力层构建器
//!
//! 提供多头自注意力（Multi-Head Self-Attention）的 ggml 计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/clip.cpp clip_graph::build_attn()

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.graph_attn);

/// 构建多头注意力层
///
/// 参数:
///   - ctx: ggml 上下文
///   - wo: 输出投影权重 [n_embd, n_embd]
///   - wo_b: 输出投影偏置 [n_embd]（可选）
///   - q_cur: Q 张量 [d_head, n_head, n_patches, n_batch]
///   - k_cur: K 张量 [d_head, n_head, n_patches, n_batch]
///   - v_cur: V 张量 [d_head, n_head, n_patches, n_batch]
///   - kq_mask: KQ 掩码 [n_patches, n_patches]（可选）
///   - kq_scale: KQ 缩放因子（通常 1/sqrt(d_head)）
///   - n_head: 注意力头数
///   - name: 张量名称前缀（用于调试）
///   - sinks: attention sinks [n_embd, n_sinks]（可选）
///
/// 返回: 注意力输出张量 [n_embd, n_patches * n_batch]
///
/// 参考: clip.cpp clip_graph::build_attn()
pub fn buildAttn(
    ctx: *ggml.Context,
    wo: *ggml.Tensor,
    wo_b: ?*ggml.Tensor,
    q_cur: *ggml.Tensor,
    k_cur: *ggml.Tensor,
    v_cur: *ggml.Tensor,
    kq_mask: ?*ggml.Tensor,
    kq_scale: f32,
    n_head: i64,
    name: []const u8,
    sinks: ?*ggml.Tensor,
) !*ggml.Tensor {
    const d_head = q_cur.ne()[0];
    const n_patches = q_cur.ne()[2];
    const n_batch = q_cur.ne()[3];
    _ = name;
    std.debug.assert(k_cur.ne()[0] == d_head);
    std.debug.assert(k_cur.ne()[2] == n_patches);
    std.debug.assert(v_cur.ne()[0] == d_head);
    std.debug.assert(v_cur.ne()[2] == n_patches);

    // 对应 C++: ggml_build_forward_expand(gf, q_cur); ggml_build_forward_expand(gf, k_cur); ggml_build_forward_expand(gf, v_cur);
    // 在 Zig 中，这些张量通过图构建自动加入

    // Q, K: permute(0, 2, 1, 3) -> [d_head, n_patches, n_head, n_batch]
    // C++: ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    const q = q_cur.permute(ctx, 0, 2, 1, 3);
    const k = k_cur.permute(ctx, 0, 2, 1, 3);

    // V: permute(0, 2, 1, 3) -> [d_head, n_patches, n_head, n_batch]
    // C++: ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);
    const v = v_cur.permute(ctx, 0, 2, 1, 3);

    // flash_attn_ext(Q, K, V, mask, scale, max_bias, logit_softcap)
    // C++: cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
    // 结果: [d_head, n_patches, n_head, n_batch]
    var cur = ggml.flashAttnExt(ctx, q, k, v, kq_mask, kq_scale, 0.0, 0.0);

    // Attention sinks (optional)
    // C++: if (sinks != nullptr) { ggml_flash_attn_ext_add_sinks(cur, sinks); }
    if (sinks) |s| {
        _ = s;
        log.warn("Attention sinks not yet implemented", .{});
    }

    // C++: cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    // C++: cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    // flash_attn_ext 输出: [d_head, n_patches, n_head, n_batch]
    // reshape_2d: [d_head * n_patches, n_head * n_batch]
    cur = cur.reshape2d(ctx, d_head * n_patches, n_head * n_batch);
    // 输出投影（对应 C++: if (wo) { cur = build_mm(wo, cur); }）
    // wo: [n_embd, n_embd], cur: [d_head * n_patches, n_head * n_batch]
    // ggml_mul_mat 计算: wo^T @ cur
    // wo 的 ne[0]=n_embd 是内积维度，cur 的 ne[0]=d_head*n_patches
    // 注意: ggml_mul_mat 使用第一个张量的 ne[0] 作为内积维度
    // 所以 wo: [n_embd, n_embd] 作为 A, cur: [d_head*n_patches, n_head*n_batch] 作为 B
    // 内积维度是 n_embd = n_head * d_head
    // 但 cur 的 ne[0] = d_head * n_patches
    // 所以需要先转置 cur: [n_head * n_batch, d_head * n_patches]
    // 然后 mul_mat(wo, cur^T) 得到 [n_embd, d_head * n_patches]
    // 然后再转置得到 [d_head * n_patches, n_embd]
    // 这不对... 让我重新思考

    // 实际上在 C++ 中，build_attn 返回 [n_embd, n_pos*B]
    // 所以 wo.mulMat 必须产生这个形状
    // 在 ggml 中，mul_mat(A, B) 计算 A @ B
    // A: [ne0_A, ne1_A] = [K, M]
    // B: [ne0_B, ne1_B] = [N, K]  (注意 B 的 ne0 是 N, ne1 是 K)
    // 结果: [M, N]
    // 等等，实际上 ggml_mul_mat 的约定是:
    // A: [ne0, ne1] = [K, M]  -> 矩阵是 M x K
    // B: [ne0, ne1] = [K, N]  -> 矩阵是 N x K
    // 结果: [M, N]
    // 所以 wo: [n_embd, n_embd] = [K, M] where K=n_embd, M=n_embd
    // cur: [d_head*n_patches, n_head*n_batch] = [K', N] where K'=d_head*n_patches, N=n_head*n_batch
    // 内积维度 K 必须等于 K'，所以 n_embd = d_head * n_patches
    // 这只在 n_patches = n_head 时成立

    // 实际上，我可能搞错了 ggml 的 mul_mat 约定。
    // 在 ggml 中，ggml_mul_mat(A, B) 计算 A @ B^T
    // A: [M, K], B: [N, K], 结果: [M, N]
    // 所以 wo: [n_embd, n_embd] = [M, K] where M=n_embd, K=n_embd
    // cur: [d_head*n_patches, n_head*n_batch] = [N, K'] where N=d_head*n_patches, K'=n_head*n_batch
    // 内积维度 K 必须等于 K'，所以 n_embd = n_head * n_batch
    // 这只在 n_batch = d_head 时成立

    // 看来我需要对 cur 进行 permute 来得到正确的形状
    // flash_attn_ext 输出: [d_head, n_patches, n_head, n_batch]
    // 我们需要 permute 到 [n_head, d_head, n_patches, n_batch]
    // 然后 cont_2d 到 [n_embd, n_patches * n_batch]
    // 这样 wo.mulMat 就是 [n_embd, n_embd] @ [n_embd, n_patches*n_batch] -> [n_embd, n_patches*n_batch]

    const n_embd = d_head * n_head;
    cur = cur.permute(ctx, 2, 0, 1, 3).cont(ctx);
    cur = ggml.cont2d(ctx, cur, n_embd, n_patches * n_batch);

    var result = wo.mulMat(ctx, cur);

    if (wo_b) |b| {
        result = result.add(ctx, b);
    }

    return result;
}

test "buildAttn: basic self-attention" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_embd: i64 = 64;
    const n_head: i64 = 4;
    const d_head: i64 = n_embd / n_head;
    const n_patches: i64 = 16;
    const n_batch: i64 = 1;

    const wo = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd);
    const q = try ctx.newTensor4d(ggml.Type.f32, d_head, n_head, n_patches, n_batch);
    const k = try ctx.newTensor4d(ggml.Type.f32, d_head, n_head, n_patches, n_batch);
    const v = try ctx.newTensor4d(ggml.Type.f32, d_head, n_head, n_patches, n_batch);

    {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(wo.nElems())));
        defer allocator.free(buf);
        @memset(buf, 0.1);
        try wo.dataSet(f32, buf);
    }
    for ([_]*ggml.Tensor{ q, k, v }) |t| {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(t.nElems())));
        defer allocator.free(buf);
        @memset(buf, 0.5);
        try t.dataSet(f32, buf);
    }
    const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    const result = try buildAttn(&ctx, wo, null, q, k, v, null, kq_scale, n_head, "test_attn", null);
    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
}
