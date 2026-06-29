//! Qwen2VL 视觉编码器图构建
//!
//! 实现 Qwen2VL 视觉编码器的计算图构建。
//! 使用 temporal merge (双 Conv2D) + spatial merge + M-RoPE。
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/qwen2vl.cpp

const std = @import("std");
const ggml = @import("ggml");
const graph = @import("..");

const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const BuildVitOpts = graph.BuildVitOpts;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const VisionHParams = graph.VisionHParams;
const ViTLayerWeights = graph.ViTLayerWeights;
const ImageF32 = graph.ImageF32;

const log = std.log.scoped(.qwen2vl_graph);

// 构建 Qwen2VL 完整计算图
//
// 处理流程:
//   1. Temporal merge: 两个 Conv2D 相加 (patch_embeddings_0 + patch_embeddings_1)
//   2. Spatial merge: permute + reshape 合并空间维度
//   3. Pre-LN (可选)
//   4. ViT blocks (LayerNorm + 自注意力 + M-RoPE + FFN)
//   5. Post-LN (可选)
//   6. 多模态投影 (FFN)
//
// 参考: llama.cpp qwen2vl.cpp build()
pub fn buildGraph(
    builder: *GraphBuilder,
) !*ggml.CGraph {
    const ctx = builder.ctx0;
    const w = builder.weights;
    const p = builder.hparams;
    const img = builder.img;

    const n_embd: i64 = @intCast(p.n_embd);
    const n_head: i64 = @intCast(p.n_head);
    const d_head = n_embd / n_head;
    const img_width: u32 = p.image_size;
    const img_height: u32 = p.image_size;
    const patch_size: i32 = @intCast(p.patch_size);
    const n_patches_x: i64 = @divTrunc(@as(i64, @intCast(img_width)), patch_size);
    const n_patches_y: i64 = @divTrunc(@as(i64, @intCast(img_height)), patch_size);
    const n_patches: i64 = n_patches_x * n_patches_y;
    const n_batch: i64 = 1;
    const eps = p.eps;
    const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    // Qwen2VL uses LayerNorm, Qwen2.5VL uses RMSNorm
    const norm_t: NormType = if (p.ffn_op == .gelu_erf) .rms_norm else .layer_norm;

    // M-RoPE sections: 4 equal parts
    const mrope_sections = [_]i32{ @intCast(d_head / 4), @intCast(d_head / 4), @intCast(d_head / 4), @intCast(d_head / 4) };

    const use_window_attn = p.n_wa_pattern > 0;
    const n_wa_pattern: i64 = @intCast(p.n_wa_pattern);

    log.info("Qwen2VL graph: embd={d}, head={d}, d_head={d}, patches={d}x{d}={d}, norm={s}, window_attn={}\n",
        .{ n_embd, n_head, d_head, n_patches_x, n_patches_y, n_patches, @tagName(norm_t), use_window_attn });

    // 1. 创建输入张量
    const inp_raw = try ctx.newTensor3d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), 3);
    inp_raw.setName("inp_raw");
    {
        const dst = inp_raw.dataF32();
        const src = img.buf;
        const H: usize = @intCast(img_height);
        const W: usize = @intCast(img_width);
        for (0..H) |y| {
            for (0..W) |x| {
                const src_idx = (y * W + x) * 3;
                const dst_base = y * W + x;
                dst[dst_base] = src[src_idx];
                dst[dst_base + H * W] = src[src_idx + 1];
                dst[dst_base + 2 * H * W] = src[src_idx + 2];
            }
        }
    }

    // 2. Temporal merge: two Conv2D added together
    var inp: *ggml.Tensor = undefined;
    if (w.patch_embeddings_0) |pe0| {
        const kw: i32 = @intCast(pe0.ne()[0]);
        const kh: i32 = @intCast(pe0.ne()[1]);
        var conv0 = inp_raw.conv2d(ctx, pe0, kw, kh, 0, 0, 1, 1);
        conv0.setName("conv0");

        if (w.patch_embeddings_1) |pe1| {
            var conv1 = inp_raw.conv2d(ctx, pe1, kw, kh, 0, 0, 1, 1);
            conv1.setName("conv1");
            inp = conv0.add(ctx, conv1);
            inp.setName("temporal_merge");
        } else {
            inp = conv0;
        }
    } else {
        return error.MissingPatchEmbedding;
    }

    // 3. Spatial merge
    // [w, h, c, b] -> [c, w, h, b] via permute(1, 2, 0, 3)
    inp = inp.permute(ctx, 1, 2, 0, 3).cont(ctx);
    inp.setName("spatial_permuted");

    // Reshape to [n_embd * 2, n_patches_x / 2, n_patches_y, batch_size]
    inp = inp.cont4d(ctx, n_embd * 2, n_patches_x / 2, n_patches_y, n_batch);
    inp.setName("spatial_reshaped_1");

    // Reshape to [n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2)]
    inp = inp.reshape4d(ctx, n_embd * 2, n_patches_x / 2, 2, n_batch * (n_patches_y / 2));
    inp.setName("spatial_reshaped_2");

    // permute(0, 2, 1, 3)
    inp = inp.permute(ctx, 0, 2, 1, 3).cont(ctx);
    inp.setName("spatial_permuted_2");

    // cont to [n_embd, n_patches_x * n_patches_y, batch_size]
    inp = inp.cont3d(ctx, n_embd, n_patches_x * n_patches_y, n_batch);
    inp.setName("spatial_merged");

    var inpL = inp;

    // 4. Pre-LN (optional)
    if (w.pre_ln_w) |pln_w| {
        inpL = try graph.buildNorm(ctx, inpL, pln_w, w.pre_ln_b, norm_t, eps, "pre_ln");
    }

    // 5. Window attention inputs (if applicable)
    var window_mask: ?*ggml.Tensor = null;
    var inv_window_idx: ?*ggml.Tensor = null;
    if (use_window_attn) {
        // inv_window_idx for reordering
        inv_window_idx = try ctx.newTensor1d(ggml.Type.i32, n_patches / 4);
        inv_window_idx.?.setName("inv_window_idx");

        // window mask
        window_mask = try ctx.newTensor2d(ggml.Type.f32, n_patches, n_patches);
        window_mask.?.setName("window_mask");

        // Reorder patches using inv_window_idx
        inpL = inpL.reshape2d(ctx, n_embd * 4, n_patches * n_batch / 4);
        inpL.setName("window_reorder_flat");
        inpL = inpL.getRows(ctx, inv_window_idx.?);
        inpL.setName("window_reorder");
        inpL = inpL.cont3d(ctx, n_embd, n_patches, n_batch);
        inpL.setName("window_reordered");
    }

    // 6. M-RoPE positions
    const num_position_ids = n_patches * 4;
    const positions = try ctx.newTensor1d(ggml.Type.i32, num_position_ids);
    positions.setName("positions");

    // 7. ViT blocks
    for (w.layers, 0..) |*layer, il| {
        const full_attn = if (use_window_attn) (@mod(@as(i64, @intCast(il + 1)), n_wa_pattern) == 0) else true;

        var cur = inpL;

        // LayerNorm 1
        cur = try graph.buildNorm(ctx, cur, layer.ln_1_w orelse return error.MissingNormWeight, layer.ln_1_b, norm_t, eps, "blk");
        cur.setName("blk");

        // Self-attention
        {
            // QKV projections
            var Qcur = if (layer.q_w) |qw| qw.mulMat(ctx, cur) else return error.MissingQWeight;
            Qcur.setName("blk");
            if (layer.q_b) |qb| {
                Qcur = Qcur.add(ctx, qb);
                Qcur.setName("blk");
            }

            var Kcur = if (layer.k_w) |kw| kw.mulMat(ctx, cur) else return error.MissingKWeight;
            Kcur.setName("blk");
            if (layer.k_b) |kb| {
                Kcur = Kcur.add(ctx, kb);
                Kcur.setName("blk");
            }

            var Vcur = if (layer.v_w) |vw| vw.mulMat(ctx, cur) else return error.MissingVWeight;
            Vcur.setName("blk");
            if (layer.v_b) |vb| {
                Vcur = Vcur.add(ctx, vb);
                Vcur.setName("blk");
            }

            // Reshape to [d_head, n_head, n_patches]
            Qcur = Qcur.reshape3d(ctx, d_head, n_head, n_patches);
            Qcur.setName("blk");
            Kcur = Kcur.reshape3d(ctx, d_head, n_head, n_patches);
            Kcur.setName("blk");
            Vcur = Vcur.reshape3d(ctx, d_head, n_head, n_patches);
            Vcur.setName("blk");

            // M-RoPE
            Qcur = Qcur.ropeMulti(ctx, positions, null, @intCast(d_head / 2), &mrope_sections, 0, 32768, 10000, 1, 0, 1, 32, 1);
            Qcur.setName("blk");
            Kcur = Kcur.ropeMulti(ctx, positions, null, @intCast(d_head / 2), &mrope_sections, 0, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur.setName("blk");

            // Attention
            const attn_mask: ?*ggml.Tensor = if (full_attn) null else window_mask;
            var attn_out = try graph.buildAttn(
                ctx,
                layer.o_w orelse return error.MissingOutputWeight,
                layer.o_b,
                Qcur,
                Kcur,
                Vcur,
                attn_mask,
                kq_scale,
                n_head,
                "blk",
                layer.attn_sinks,
            );
            attn_out.setName("blk");

            // Residual
            cur = inpL.add(ctx, attn_out);
            cur.setName("blk");
        }

        inpL = cur;

        // LayerNorm 2
        cur = try graph.buildNorm(ctx, cur, layer.ln_2_w orelse return error.MissingNormWeight, layer.ln_2_b, norm_t, eps, "blk");
        cur.setName("blk");

        // FFN
        {
            const ffn_out = try graph.buildFFN(
                ctx,
                cur,
                layer.ff_up_w orelse return error.MissingFFNUpWeight,
                layer.ff_up_b,
                layer.ff_gate_w,
                layer.ff_gate_b,
                layer.ff_down_w orelse return error.MissingFFNDownWeight,
                layer.ff_down_b,
                .gelu,
                "blk",
            );
            ffn_out.setName("blk");

            // Residual 2
            inpL = inpL.add(ctx, ffn_out);
            inpL.setName("blk");
        }
    }

    // 8. Post-LN (optional)
    if (w.post_ln_w) |poln_w| {
        inpL = try graph.buildNorm(ctx, inpL, poln_w, w.post_ln_b, norm_t, eps, "post_ln");
    }

    // 9. Multimodal projection
    var embeddings = inpL;
    embeddings = embeddings.cont3d(ctx, n_embd * 4, n_patches / 4, n_batch);
    embeddings.setName("mm_reshape");

    // FFN projection: mm_0 (GELU) -> mm_1
    embeddings = try graph.buildFFN(
        ctx,
        embeddings,
        w.mm_0_w orelse return error.MissingMMWeight,
        w.mm_0_b,
        null,
        null,
        w.mm_1_w orelse return error.MissingMMWeight,
        w.mm_1_b,
        .gelu,
        "mm_proj",
    );
    embeddings.setName("mm_proj");

    // Window attention reorder back (if applicable)
    if (use_window_attn) {
        var window_idx = try ctx.newTensor1d(ggml.Type.i32, n_patches / 4);
        window_idx.setName("window_idx");

        embeddings = embeddings.reshape2d(ctx, p.projection_dim, n_patches / 4);
        embeddings.setName("window_reorder_back_flat");
        embeddings = embeddings.getRows(ctx, window_idx);
        embeddings.setName("window_reorder_back");
        embeddings = embeddings.cont3d(ctx, @intCast(p.projection_dim), n_patches / 4, n_batch);
        embeddings.setName("window_reordered_back");
    }

    embeddings.setName("mm_output");

    // 构建计算图
    builder.gf.buildForwardExpand(embeddings);

    log.info("Qwen2VL graph built successfully", .{});
    return builder.gf;
}
