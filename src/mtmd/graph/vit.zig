//! 通用 ViT 图构建器
//!
//! 提供 Vision Transformer (ViT) 的通用计算图构建函数。
//! 覆盖大多数模型，特殊模型可复制此函数进行定制。
//!
//! 参考: deps/llama.cpp/tools/mtmd/clip.cpp clip_graph::build_vit()

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");
const norm_builder = @import("norm.zig");
const ffn_builder = @import("ffn.zig");
const attn_builder = @import("attn.zig");
const rope_builder = @import("rope.zig");

const NormType = types.NormType;
const FFNOpType = types.FFNOpType;
const BuildVitOpts = types.BuildVitOpts;
const ViTLayerWeights = types.ViTLayerWeights;
const VisionEncoderWeights = types.VisionEncoderWeights;
const VisionHParams = types.VisionHParams;

const log = std.log.scoped(.graph_vit);

/// 张量命名回调（对应 C++ clip_graph::cb()）
/// 当 il >= 0 时，格式化为 "{name}-{il}"；否则直接设为 name。
/// 注意: ggml 的 setName 接受 [:0]const u8，不支持格式化。
/// 这里使用简单的命名策略：当 il >= 0 时，使用 "{name}_{il}" 格式。
pub fn cb(cur: *ggml.Tensor, name: [:0]const u8, il: i32) void {
    if (il >= 0) {
        var buf: [128]u8 = undefined;
        if (std.fmt.bufPrintZ(&buf, "{s}-{d}", .{ name, il })) |formatted| {
            cur.setName(formatted);
        } else |_| {
            cur.setName(name);
        }
    } else {
        cur.setName(name);
    }
}

/// 构建 Vision Transformer 计算图
///
/// 参数:
///   - ctx: ggml 上下文
///   - gf: ggml 计算图（用于 build_forward_expand）
///   - inp: 输入张量 [n_embd, n_patches, n_batch]
///   - n_pos: 位置数量
///   - norm_t: 归一化类型
///   - ffn_t: FFN 激活函数类型
///   - learned_pos_embd: 可学习位置嵌入 [n_embd, n_pos]（可选）
///   - weights: 视觉编码器权重
///   - hparams: 视觉编码器超参数
///   - add_pos: 添加位置嵌入的回调函数（对 Q/K 应用 2D RoPE 等）
///             回调签名: fn (ctx, cur, layer) -> cur
///   - opts: 构建选项
///
/// 返回: 编码后的张量 [n_embd, n_patches, n_batch]
///
/// 参考: clip.cpp clip_graph::build_vit()
pub fn buildVit(
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    inp: *ggml.Tensor,
    n_pos: i64,
    norm_t: NormType,
    ffn_t: FFNOpType,
    learned_pos_embd: ?*ggml.Tensor,
    weights: *const VisionEncoderWeights,
    hparams: *const VisionHParams,
    add_pos: ?*const fn (*ggml.Context, *ggml.Tensor, *const ViTLayerWeights) *ggml.Tensor,
    opts: BuildVitOpts,
) !*ggml.Tensor {
    // batch dim: inp is [n_embd, n_pos, B]
    const B = inp.ne()[2];
    const n_embd = inp.ne()[0];
    const n_patches = inp.ne()[1];
    const n_head: i64 = @intCast(hparams.n_head);
    const n_head_kv: i64 = if (hparams.n_head_kv > 0) @intCast(hparams.n_head_kv) else n_head;
    const d_head = @divExact(n_embd, n_head);
    const eps = hparams.eps;

    // 1. 添加位置嵌入（对应 C++: if (learned_pos_embd) { inp = ggml_add(...); cb(inp, "pos_embed", -1); }）
    var cur = inp;
    if (learned_pos_embd) |pos_embd| {
        cur = cur.add(ctx, pos_embd);
        cb(cur, "pos_embed", -1);
    }

    // flatten batch; unflatten again in attention
    // 对应 C++: inp = ggml_reshape_2d(ctx0, inp, n_embd, n_pos * B);
    var inpL = cur.reshape2d(ctx, n_embd, n_patches * B);

    // 2. Pre-LN (optional)（对应 C++: if (model.pre_ln_w) { inpL = build_norm(...); cb(inpL, "pre_ln", -1); }）
    if (weights.pre_ln_w) |pln_w| {
        inpL = try norm_builder.buildNorm(ctx, inpL, pln_w, weights.pre_ln_b, norm_t, eps, "pre_ln");
        cb(inpL, "pre_ln", -1);
    }

    // 3. ViT blocks（对应 C++: for (int il = 0; il < n_layer; il++)）
    for (weights.layers, 0..) |*layer, il| {
        const il_i32: i32 = @intCast(il);

        // C++: ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states
        var hidden = inpL;

        // --- Pre-attention norm ---
        // C++: cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        //       cb(cur, "layer_inp_normed", il);
        hidden = try norm_builder.buildNorm(ctx, hidden, layer.ln_1_w orelse return error.MissingLN1Weight, layer.ln_1_b, norm_t, eps, "blk");
        cb(hidden, "layer_inp_normed", il_i32);

        // --- Self-attention ---
        {
            var Qcur: ?*ggml.Tensor = null;
            var Kcur: ?*ggml.Tensor = null;
            var Vcur: ?*ggml.Tensor = null;

            if (layer.qkv_w) |qkv_w| {
                // fused qkv（对应 C++: cur = build_mm(layer.qkv_w, cur);）
                // 使用 build_mm 回调（支持 clamp 等模型特定操作）
                var qkv = opts.build_mm(ctx, qkv_w, hidden);
                if (layer.qkv_b) |qkv_b| {
                    qkv = qkv.add(ctx, qkv_b);
                }

                // Q/K/V as [d_head, n_head, n_pos, B]
                // C++: Qcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B, ...)
                // C++: nb1 = ggml_row_size(cur->type, d_head), nb2 = cur->nb[1], nb3 = cur->nb[1] * n_pos
                const row_size = ggml.Type.rowSize(qkv.dataType(), d_head);
                const nb2: usize = @intCast(qkv.nb()[1]); // stride for n_embd dimension (bytes)
                const nb3: usize = nb2 * @as(usize, @intCast(n_pos)); // stride for n_pos dimension (bytes)
                Qcur = qkv.view4d(ctx, d_head, n_head, n_patches, B, row_size, nb2, nb3, 0);
                Kcur = qkv.view4d(ctx, d_head, n_head, n_patches, B, row_size, nb2, nb3, ggml.Type.rowSize(qkv.dataType(), n_embd));
                Vcur = qkv.view4d(ctx, d_head, n_head, n_patches, B, row_size, nb2, nb3, ggml.Type.rowSize(qkv.dataType(), 2 * n_embd));

                // Q/K norm after split (fused path)
                if (layer.q_norm) |qn| {
                    Qcur = try norm_builder.buildNorm(ctx, Qcur.?, qn, null, norm_t, eps, "blk");
                    cb(Qcur.?, "Qcur_norm", il_i32);
                }
                if (layer.k_norm) |kn| {
                    Kcur = try norm_builder.buildNorm(ctx, Kcur.?, kn, null, norm_t, eps, "blk");
                    cb(Kcur.?, "Kcur_norm", il_i32);
                }
            } else {
                // separate q, k, v（对应 C++: Qcur = build_mm(layer.q_w, cur);）
                // 使用 build_mm 回调（支持 clamp 等模型特定操作）
                Qcur = opts.build_mm(ctx, layer.q_w orelse return error.MissingQWeight, hidden);
                if (layer.q_b) |qb| {
                    Qcur = Qcur.?.add(ctx, qb);
                }

                Kcur = opts.build_mm(ctx, layer.k_w orelse return error.MissingKWeight, hidden);
                if (layer.k_b) |kb| {
                    Kcur = Kcur.?.add(ctx, kb);
                }

                Vcur = opts.build_mm(ctx, layer.v_w orelse return error.MissingVWeight, hidden);
                if (layer.v_b) |vb| {
                    Vcur = Vcur.?.add(ctx, vb);
                }

                // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
                // C++: bool norm_per_head = layer.q_norm && layer.q_norm->ne[0] == d_head;
                const norm_per_head = if (layer.q_norm) |qn| qn.ne()[0] == d_head else false;

                if (!norm_per_head) {
                    if (layer.q_norm) |qn| {
                        Qcur = try norm_builder.buildNorm(ctx, Qcur.?, qn, null, norm_t, eps, "blk");
                        cb(Qcur.?, "Qcur_norm", il_i32);
                    }
                    if (layer.k_norm) |kn| {
                        Kcur = try norm_builder.buildNorm(ctx, Kcur.?, kn, null, norm_t, eps, "blk");
                        cb(Kcur.?, "Kcur_norm", il_i32);
                    }
                }

                // Reshape to [d_head, n_head, n_patches, n_batch]
                // C++: Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, n_pos, B);
                Qcur = Qcur.?.reshape4d(ctx, d_head, n_head, n_patches, B);
                Kcur = Kcur.?.reshape4d(ctx, d_head, n_head_kv, n_patches, B);
                Vcur = Vcur.?.reshape4d(ctx, d_head, n_head_kv, n_patches, B);

                if (norm_per_head) {
                    if (layer.q_norm) |qn| {
                        Qcur = try norm_builder.buildNorm(ctx, Qcur.?, qn, null, norm_t, eps, "blk");
                        cb(Qcur.?, "Qcur_norm_per_head", il_i32);
                    }
                    if (layer.k_norm) |kn| {
                        Kcur = try norm_builder.buildNorm(ctx, Kcur.?, kn, null, norm_t, eps, "blk");
                        cb(Kcur.?, "Kcur_norm_per_head", il_i32);
                    }
                }
            }

            // C++: cb(Qcur, "Qcur", il); cb(Kcur, "Kcur", il); cb(Vcur, "Vcur", il);
            cb(Qcur.?, "Qcur", il_i32);
            cb(Kcur.?, "Kcur", il_i32);
            cb(Vcur.?, "Vcur", il_i32);

            // 2D RoPE via add_pos callback（对应 C++: if (add_pos) { Qcur = add_pos(Qcur, layer); ... }）
            if (add_pos) |cb_add_pos| {
                Qcur = cb_add_pos(ctx, Qcur.?, layer);
                Kcur = cb_add_pos(ctx, Kcur.?, layer);
                cb(Qcur.?, "Qcur_pos", il_i32);
                cb(Kcur.?, "Kcur_pos", il_i32);
            }

            // Vcur RMSNorm (gemma4v specific, controlled by opts)
            // C++: if (proj_type == PROJECTOR_TYPE_GEMMA4V) { Vcur = ggml_rms_norm(ctx0, Vcur, eps); cb(Vcur, "Vcur_normed", il); }
            if (opts.v_norm) {
                Vcur = Vcur.?.rmsNorm(ctx, opts.v_norm_eps);
                cb(Vcur.?, "Vcur_normed", il_i32);
            }

            // Attention（对应 C++: cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, opts.attn_mask, kq_scale, il);）
            // kq_scale: gemma4v uses 1.0, other models use 1/sqrt(d_head)
            const kq_scale = opts.kq_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(d_head))));
            hidden = try attn_builder.buildAttn(
                ctx,
                gf,
                layer.o_w orelse return error.MissingOutputWeight,
                layer.o_b,
                Qcur.?,
                Kcur.?,
                Vcur.?,
                opts.attn_mask,
                kq_scale,
                n_head,
                "blk",
                layer.attn_sinks,
                opts.build_mm, // 传递 build_mm 回调
            );
            // C++: cb(cur, "attn_out", il);
            cb(hidden, "attn_out", il_i32);
        }

        // Layer scale 1 (optional)（对应 C++: if (layer.ls_1_w) { cur = ggml_mul(ctx0, cur, layer.ls_1_w); cb(cur, "attn_out_scaled", il); }）
        if (layer.ls_1_w) |ls1| {
            hidden = hidden.mul(ctx, ls1);
            cb(hidden, "attn_out_scaled", il_i32);
        }

        // Post-attention norm (optional, e.g. gemma4)
        // C++: if (layer.attn_post_norm_w) { cur = build_norm(cur, layer.attn_post_norm_w, nullptr, norm_t, eps, il); cb(cur, "attn_post_normed", il); }
        if (layer.attn_post_norm_w) |apn| {
            hidden = try norm_builder.buildNorm(ctx, hidden, apn, null, norm_t, eps, "blk");
            cb(hidden, "attn_post_normed", il_i32);
        }

        // Residual 1（对应 C++: cur = ggml_add(ctx0, cur, inpL);）
        hidden = hidden.add(ctx, inpL);
        inpL = hidden; // inpL = residual, hidden = hidden_states

        // C++: cb(cur, "ffn_inp", il);
        cb(hidden, "ffn_inp", il_i32);

        // --- Pre-FFN norm ---
        // C++: cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il); cb(cur, "ffn_inp_normed", il);
        hidden = try norm_builder.buildNorm(ctx, hidden, layer.ln_2_w orelse return error.MissingLN2Weight, layer.ln_2_b, norm_t, eps, "blk");
        cb(hidden, "ffn_inp_normed", il_i32);

        // --- FFN ---
        // C++: cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, ffn_t, il);
        hidden = try ffn_builder.buildFFN(
            ctx,
            hidden,
            layer.ff_up_w orelse return error.MissingFFNUpWeight,
            layer.ff_up_b,
            layer.ff_gate_w,
            layer.ff_gate_b,
            layer.ff_down_w orelse return error.MissingFFNDownWeight,
            layer.ff_down_b,
            ffn_t,
            "blk",
            opts.build_mm, // 传递 build_mm 回调
        );
        // C++: cb(cur, "ffn_out", il);
        cb(hidden, "ffn_out", il_i32);

        // Post-FFN norm (optional)
        // C++: if (layer.ff_post_norm_w) { cur = build_norm(cur, layer.ff_post_norm_w, nullptr, norm_t, eps, il); cb(cur, "ffn_post_normed", il); }
        if (layer.ff_post_norm_w) |fpn| {
            hidden = try norm_builder.buildNorm(ctx, hidden, fpn, null, norm_t, eps, "blk");
            cb(hidden, "ffn_post_normed", il_i32);
        }

        // Layer scale 2 (optional)
        // C++: if (layer.ls_2_w) { cur = ggml_mul(ctx0, cur, layer.ls_2_w); cb(cur, "ffn_out_scaled", il); }
        if (layer.ls_2_w) |ls2| {
            hidden = hidden.mul(ctx, ls2);
            cb(hidden, "ffn_out_scaled", il_i32);
        }

        // Residual 2（对应 C++: cur = ggml_add(ctx0, inpL, cur); cb(cur, "layer_out", il);）
        hidden = inpL.add(ctx, hidden);
        cb(hidden, "layer_out", il_i32);

        // Layer scale out (optional)
        // C++: if (layer.ls_out_w) { cur = ggml_mul(ctx0, cur, layer.ls_out_w); cb(cur, "layer_out_scaled", il); }
        if (layer.ls_out_w) |ls_out| {
            hidden = hidden.mul(ctx, ls_out);
            cb(hidden, "layer_out_scaled", il_i32);
        }

        inpL = hidden;
    }

    // 4. Post-LN (optional)（对应 C++: if (model.post_ln_w) { inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1); }）
    if (weights.post_ln_w) |poln_w| {
        inpL = try norm_builder.buildNorm(ctx, inpL, poln_w, weights.post_ln_b, norm_t, eps, "post_ln");
    }

    // restore the batch dim（对应 C++: GGML_ASSERT(inpL->ne[1] % B == 0); inpL = ggml_reshape_3d(ctx0, inpL, n_embd, inpL->ne[1] / B, B);）
    std.debug.assert(@rem(inpL.ne()[1], B) == 0);
    inpL = inpL.reshape3d(ctx, n_embd, @divExact(inpL.ne()[1], B), B);

    return inpL;
}

/// 调整位置嵌入大小（通过双线性插值）
///
/// 使用双线性插值将位置嵌入调整到目标大小。
/// 注意：ggml_repeat 是沿维度重复（不是插值），所以这里使用手动插值。
///
/// 参数:
///   - ctx: ggml 上下文
///   - pos_embd: 原始位置嵌入 [n_embd, src_size]
///   - target_size: 目标大小
///   - interpolation_mode: 插值模式（0=双线性，其他保留）
///
/// 返回: 调整后的位置嵌入 [n_embd, target_size]
///
/// 参考: clip-graph.h resize_position_embeddings()
pub fn resizePositionEmbeddings(
    ctx: *ggml.Context,
    pos_embd: *ggml.Tensor,
    target_size: i64,
    interpolation_mode: u32,
) !*ggml.Tensor {
    _ = interpolation_mode;
    const n_embd = pos_embd.ne()[0];
    const src_size = pos_embd.ne()[1];

    log.info("resizePositionEmbeddings: n_embd={d}, src_size={d}, target_size={d}", .{ n_embd, src_size, target_size });

    if (src_size == target_size) {
        return pos_embd;
    }

    // 如果 target_size 是 src_size 的整数倍，使用 ggml_repeat（沿 dim 2 重复）
    // 否则使用双线性插值（通过 CPU 端手动计算）
    if (@rem(target_size, src_size) == 0) {
        // 整数倍缩放：使用 repeat
        var cur = pos_embd.reshape4d(ctx, n_embd, 1, src_size, 1);
        cur.setName("pos_embd_4d");

        const repeated = try ctx.newTensor4d(ggml.Type.f32, n_embd, 1, target_size, 1);
        cur = ggml.repeat(ctx, cur, repeated);
        cur.setName("pos_embd_upscaled");

        cur = cur.reshape2d(ctx, n_embd, target_size);
        cur.setName("pos_embd_resized");
        return cur;
    }
    // 创建目标张量
    const result = try ctx.newTensor2d(ggml.Type.f32, n_embd, target_size);
    result.setName("pos_embd_resized");

    // In no_alloc mode, the tensor data pointer is NULL.
    // We need to allocate the data manually so we can write to it.
    const no_alloc = ctx.getNoAlloc();
    if (no_alloc) {
        const data_size = @as(usize, @intCast(result.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
        @memset(buf, 0);
        result.setDataPtr(buf);
    }

    // 使用 dataGet 安全读取源张量数据
    const src_data = try pos_embd.dataGet(f32, std.heap.page_allocator);
    defer std.heap.page_allocator.free(src_data);

    // 分配目标缓冲区
    const dst_buf = try std.heap.page_allocator.alloc(f32, @as(usize, @intCast(n_embd * target_size)));
    defer std.heap.page_allocator.free(dst_buf);

    const scale = @as(f64, @floatFromInt(src_size)) / @as(f64, @floatFromInt(target_size));

    for (0..@as(usize, @intCast(target_size))) |dst_idx| {
        const src_f: f64 = (@as(f64, @floatFromInt(dst_idx)) + 0.5) * scale - 0.5;
        const src0_i: i64 = @intFromFloat(@floor(src_f));
        const src1_i: i64 = @min(src0_i + 1, src_size - 1);
        const src0: usize = @intCast(@max(src0_i, 0));
        const src1: usize = @intCast(src1_i);
        const frac: f64 = src_f - @floor(src_f);

        for (0..@as(usize, @intCast(n_embd))) |e| {
            const v0 = src_data[src0 * @as(usize, @intCast(n_embd)) + e];
            const v1 = src_data[src1 * @as(usize, @intCast(n_embd)) + e];
            dst_buf[dst_idx * @as(usize, @intCast(n_embd)) + e] = @as(f32, @floatCast(@as(f64, v0) * (1.0 - frac) + @as(f64, v1) * frac));
        }
    }

    // 通过 dataSet 安全写入结果张量
    try result.dataSet(f32, dst_buf);

    return result;
}

test "buildVit: basic ViT forward" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const n_embd: i64 = 64;
    const n_patches: i64 = 16;
    const n_layer: usize = 2;
    const n_ff: i64 = 256;
    const n_head: i64 = 4;
    const n_batch: i64 = 1;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 4 * 1024 * 1024 });
    defer ctx.deinit();

    // Create input [n_embd, n_patches, n_batch]
    const inp = try ctx.newTensor3d(ggml.Type.f32, n_embd, n_patches, n_batch);
    {
        const buf = try allocator.alloc(f32, @as(usize, @intCast(n_embd * n_patches * n_batch)));
        defer allocator.free(buf);
        @memset(buf, 0.5);
        try inp.dataSet(f32, buf);
    }

    // Create weights
    var layers = try allocator.alloc(ViTLayerWeights, n_layer);
    for (0..n_layer) |il| {
        layers[il] = ViTLayerWeights{
            .ln_1_w = try ctx.newTensor1d(ggml.Type.f32, n_embd),
            .q_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .k_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .v_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .o_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .ln_2_w = try ctx.newTensor1d(ggml.Type.f32, n_embd),
            .ff_up_w = try ctx.newTensor2d(ggml.Type.f32, @intCast(n_ff), n_embd),
            .ff_down_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, @intCast(n_ff)),
        };
        // Initialize weights
        for ([_]*ggml.Tensor{ layers[il].ln_1_w.?, layers[il].ln_2_w.? }) |t| {
            const buf = try allocator.alloc(f32, @as(usize, @intCast(t.nElems())));
            defer allocator.free(buf);
            @memset(buf, 1.0);
            try t.dataSet(f32, buf);
        }
        for ([_]*ggml.Tensor{ layers[il].q_w.?, layers[il].k_w.?, layers[il].v_w.?, layers[il].o_w.?, layers[il].ff_up_w.?, layers[il].ff_down_w.? }) |t| {
            const buf = try allocator.alloc(f32, @as(usize, @intCast(t.nElems())));
            defer allocator.free(buf);
            @memset(buf, 0.1);
            try t.dataSet(f32, buf);
        }
    }

    var weights = VisionEncoderWeights{
        .layers = layers,
    };
    var hparams = VisionHParams{
        .n_embd = @intCast(n_embd),
        .n_head = n_head,
        .n_layer = @intCast(n_layer),
        .n_ff = n_ff,
        .eps = 1e-5,
    };

    // Simple add_pos callback that does nothing
    const addPosFn = struct {
        fn f(_: *ggml.Context, cur: *ggml.Tensor, _: *const ViTLayerWeights) *ggml.Tensor {
            return cur;
        }
    }.f;

    // Create a graph for testing
    var gf = try ctx.newGraph();
    defer gf.deinit();

    const result = try buildVit(
        &ctx,
        &gf,
        inp,
        n_patches,
        .rms_norm,
        .silu,
        null,
        &weights,
        &hparams,
        addPosFn,
        .{},
    );

    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
    try testing.expectEqual(n_batch, result.ne()[2]);
}
