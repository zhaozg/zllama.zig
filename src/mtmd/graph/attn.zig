//! 注意力层构建器
//!
//! 提供多头自注意力（Multi-Head Self-Attention）的 ggml 计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/clip.cpp clip_graph::build_attn()

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");

const BuildMMFn = types.BuildMMFn;
const defaultBuildMM = types.defaultBuildMM;
const FlashAttnType = types.FlashAttnType;
const CbFn = types.CbFn;
const defaultCb = types.defaultCb;

const log = std.log.scoped(.graph_attn);

/// 构建多头注意力层
///
/// 参数顺序对齐 C++: clip_graph::build_attn(wo, wo_b, q_cur, k_cur, v_cur, kq_mask, kq_scale, il, sinks)
/// Zig 额外参数: flash_attn_type, build_mm, data（替代 C++ 虚函数 build_mm 和成员 flash_attn_type）
///
/// 参数:
///   - ctx: ggml 上下文
///   - gf: ggml 计算图（用于 build_forward_expand）
///   - wo: 输出投影权重 [n_embd, n_embd]
///   - wo_b: 输出投影偏置 [n_embd]（可选）
///   - q_cur: Q 张量 [d_head, n_head, n_patches, n_batch]
///   - k_cur: K 张量 [d_head, n_head, n_patches, n_batch]
///   - v_cur: V 张量 [d_head, n_head, n_patches, n_batch]
///   - kq_mask: KQ 掩码 [n_patches, n_patches]（可选）
///   - kq_scale: KQ 缩放因子（通常 1/sqrt(d_head)）
///   - il: 层索引（用于调试命名，-1 表示不使用）
///   - sinks: attention sinks [n_embd, n_sinks]（可选）
///   - flash_attn_type: flash attention 类型（.enabled / .disabled / .auto）
///   - build_mm: 矩阵乘法回调（对应 C++ clip_graph::build_mm 虚拟函数）
///   - data: 模型私有数据指针，传递给 build_mm 回调
///   - cb: 张量命名回调（可选），对应 C++ clip_graph::cb
///
/// 返回: 注意力输出张量 [n_embd, n_patches * n_batch]
///
/// 参考: clip.cpp clip_graph::build_attn()
pub fn buildAttn(
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    wo: *ggml.Tensor,
    wo_b: ?*ggml.Tensor,
    q_cur: *ggml.Tensor,
    k_cur: *ggml.Tensor,
    v_cur: *ggml.Tensor,
    kq_mask: ?*ggml.Tensor,
    kq_scale: f32,
    il: i32,
    sinks: ?*ggml.Tensor,
    flash_attn_type: FlashAttnType,
    build_mm: BuildMMFn,
    data: ?*anyopaque,
    cb: ?CbFn,
) !*ggml.Tensor {
    const d_head = q_cur.ne()[0];
    const n_head = q_cur.ne()[1];
    const n_patches = q_cur.ne()[2];
    const n_batch = q_cur.ne()[3];
    std.debug.assert(k_cur.ne()[0] == d_head);
    std.debug.assert(k_cur.ne()[1] == n_head);
    std.debug.assert(k_cur.ne()[2] == n_patches);
    std.debug.assert(v_cur.ne()[0] == d_head);
    std.debug.assert(v_cur.ne()[1] == n_head);
    std.debug.assert(v_cur.ne()[2] == n_patches);

    // C++: ggml_build_forward_expand(gf, q_cur); ggml_build_forward_expand(gf, k_cur); ggml_build_forward_expand(gf, v_cur);
    // 这些节点被一起加入图，防止重排序导致图分裂数增加
    gf.buildForwardExpand(q_cur);
    gf.buildForwardExpand(k_cur);
    gf.buildForwardExpand(v_cur);

    // Q, K: permute(0, 2, 1, 3) -> [d_head, n_patches, n_head, n_batch]
    // C++: ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    const q = q_cur.permute(ctx, 0, 2, 1, 3);
    const k = k_cur.permute(ctx, 0, 2, 1, 3);

    var cur: *ggml.Tensor = undefined;

    // 解析 flash_attn_type: .auto 时默认启用 flash attention
    // C++ 中 CLIP_FLASH_ATTN_TYPE_AUTO 在 warmup 阶段通过 reserve_compute_meta
    // 检测后端是否支持 flash attention。如果支持则保持 ENABLED，否则回退到 DISABLED。
    // 在 Zig 端，我们默认启用 flash attention（CPU backend 总是支持），
    // 如果后端不支持，ggml 内部会回退到 CPU 计算。
    const resolved_type: FlashAttnType = if (flash_attn_type == .auto) .enabled else flash_attn_type;

    log.debug("flash_attn_type={} resolved={}", .{ flash_attn_type, resolved_type });

    // C++: if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
    if (resolved_type == .enabled) {
        // Flash attention 路径
        // C++: ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);
        const v = v_cur.permute(ctx, 0, 2, 1, 3);

        // C++: k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        //       v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        //       if (kq_mask) { kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16); }
        const k_f16 = ggml.cast(ctx, k, ggml.Type.f16);
        const v_f16 = ggml.cast(ctx, v, ggml.Type.f16);
        const mask_f16 = if (kq_mask) |m| ggml.cast(ctx, m, ggml.Type.f16) else null;

        // flash_attn_ext(Q, K, V, mask, scale, max_bias, logit_softcap)
        // C++: cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        // 结果: [d_head, n_head, n_patches, n_batch]
        cur = ggml.flashAttnExt(ctx, q, k_f16, v_f16, mask_f16, kq_scale, 0.0, 0.0);

        // C++: ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        ggml.flashAttnExtSetPrec(cur, .f32);

        // Attention sinks (optional)
        // C++: if (sinks != nullptr) { ggml_flash_attn_ext_add_sinks(cur, sinks); }
        if (sinks) |s| {
            ggml.flashAttnExtAddSinks(cur, s);
        }

        // C++: cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        // flash_attn_ext 输出: [d_head, n_head, n_patches, n_batch]
        // reshape_2d: [d_head * n_head, n_patches * n_batch] = [n_embd, n_patches * n_batch]
        cur = cur.reshape2d(ctx, d_head * n_head, n_patches * n_batch);
    } else {
        // 非 flash_attn 路径（CLIP_FLASH_ATTN_TYPE_DISABLED）
        // C++: ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        //       v = ggml_cont(ctx0, v);
        var v = v_cur.permute(ctx, 1, 2, 0, 3);
        v = ggml.cont(ctx, v);

        // C++: ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        //      // F32 may not needed for vision encoders?
        //      // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        var kq = ggml.mulMat(ctx, k, q);

        // C++: kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);
        kq = ggml.softMaxExt(ctx, kq, kq_mask, kq_scale, 0.0);

        // Attention sinks (optional) — non-flash path uses soft_max_add_sinks
        // C++: if (sinks != nullptr) { ggml_soft_max_add_sinks(kq, sinks); }
        if (sinks) |s| {
            ggml.softMaxAddSinks(kq, s);
        }

        // C++: ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        const kqv = ggml.mulMat(ctx, v, kq);

        // C++: cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        //      cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        cur = kqv.permute(ctx, 0, 2, 1, 3);
        cur = ggml.cont2d(ctx, cur, cur.ne()[0] * cur.ne()[1], cur.ne()[2] * cur.ne()[3]);
    }

    // C++: cb(cur, "kqv_out", il);
    if (cb) |cbf| cbf(cur, "kqv_out", il);

    // 输出投影（对应 C++: if (wo) { cur = build_mm(wo, cur); }）
    // 使用 build_mm 回调（支持 clamp 等模型特定操作）
    var result = build_mm(ctx, wo, cur, data);

    if (wo_b) |b| {
        result = result.add(ctx, b);
    }

    return result;
}

test "buildAttn: basic self-attention (flash)" {
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

    // Create a graph for testing
    var gf = try ctx.newGraph();
    defer gf.deinit();

    const result = try buildAttn(&ctx, &gf, wo, null, q, k, v, null, kq_scale, -1, null, .enabled, defaultBuildMM, null, null);
    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
}

test "buildAttn: basic self-attention (non-flash)" {
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

    // Create a graph for testing
    var gf = try ctx.newGraph();
    defer gf.deinit();

    const result = try buildAttn(&ctx, &gf, wo, null, q, k, v, null, kq_scale, -1, null, .disabled, defaultBuildMM, null, null);
    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
}

test "buildAttn: auto resolves to flash" {
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

    var gf = try ctx.newGraph();
    defer gf.deinit();

    // .auto should resolve to .enabled (flash path)
    const result = try buildAttn(&ctx, &gf, wo, null, q, k, v, null, kq_scale, -1, null, .auto, defaultBuildMM, null, null);
    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
}
