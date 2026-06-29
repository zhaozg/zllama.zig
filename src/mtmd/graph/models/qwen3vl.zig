//! Qwen3VL 视觉编码器图构建
//!
//! 实现 Qwen3VL 视觉编码器的计算图构建。
//! 继承 Qwen2VL 的 temporal merge + spatial merge，但增加:
//!   - patch_bias
//!   - 可学习位置嵌入 (resize_position_embeddings)
//!   - QKV fused weight
//!   - Deepstack features
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/qwen3vl.cpp

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

const log = std.log.scoped(.qwen3vl_graph);

// 构建 Qwen3VL 完整计算图
//
// 处理流程:
//   1. Temporal merge: 两个 Conv2D 相加 (patch_embeddings_0 + patch_embeddings_1)
//   2. Spatial merge: permute + reshape 合并空间维度
//   3. patch_bias 添加
//   4. 可学习位置嵌入 (resize_position_embeddings)
//   5. Pre-LN (可选)
//   6. ViT blocks (LayerNorm + QKV fused 自注意力 + M-RoPE + FFN + Deepstack)
//   7. Post-LN (可选)
//   8. 多模态投影 (FFN) + Deepstack concat
//
// 参考: llama.cpp qwen3vl.cpp build()
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

    // Qwen3VL uses LayerNorm
    const norm_t: NormType = .layer_norm;

    // M-RoPE sections: 4 equal parts
    const mrope_sections = [_]i32{ @intCast(d_head / 4), @intCast(d_head / 4), @intCast(d_head / 4), @intCast(d_head / 4) };

    // Merge factor for deepstack
    const merge_factor: i64 = if (p.n_merge > 0) @intCast(p.n_merge * p.n_merge) else 4;

    log.info("Qwen3VL graph: embd={d}, head={d}, d_head={d}, patches={d}x{d}={d}, merge_factor={d}\n",
        .{ n_embd, n_head, d_head, n_patches_x, n_patches_y, n_patches, merge_factor });

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

    // 4. Add patch_bias
    if (w.patch_bias) |pb| {
        inp = inp.add(ctx, pb);
        inp.setName("patch_bias_added");
    }

    // 5. 可学习位置嵌入 (resize_position_embeddings)
    if (w.position_embeddings) |pos_embd| {
        // Resize position embeddings to match n_patches
        var learned_pos_embd = try graph.resizePositionEmbeddings(ctx, pos_embd, n_patches, 0);

        // Apply same spatial merge to position embeddings
        learned_pos_embd = learned_pos_embd.cont4d(ctx, n_embd * 2, n_patches_x / 2, n_patches_y, n_batch);
        learned_pos_embd.setName("pos_embd_reshaped_1");
        learned_pos_embd = learned_pos_embd.reshape4d(ctx, n_embd * 2, n_patches_x / 2, 2, n_batch * (n_patches_y / 2));
        learned_pos_embd.setName("pos_embd_reshaped_2");
        learned_pos_embd = learned_pos_embd.permute(ctx, 0, 2, 1, 3).cont(ctx);
        learned_pos_embd.setName("pos_embd_permuted");
        learned_pos_embd = learned_pos_embd.cont3d(ctx, n_embd, n_patches_x * n_patches_y, n_batch);
        learned_pos_embd.setName("pos_embd_merged");

        inp = inp.add(ctx, learned_pos_embd);
        inp.setName("inp_with_pos");
    }

    var inpL = inp;

    // 6. Pre-LN (optional)
    if (w.pre_ln_w) |pln_w| {
        inpL = try graph.buildNorm(ctx, inpL, pln_w, w.pre_ln_b, norm_t, eps, "pre_ln");
    }

    // 7. M-RoPE positions
    const num_position_ids = n_patches * 4;
    const positions = try ctx.newTensor1d(ggml.Type.i32, num_position_ids);
    positions.setName("positions");

    // 8. Deepstack features
    var deepstack_features: ?*ggml.Tensor = null;

    // 9. ViT blocks
    for (w.layers, 0..) |*layer, il| {
        _ = il;


        var cur = inpL;

        // LayerNorm 1
        cur = try graph.buildNorm(ctx, cur, layer.ln_1_w orelse return error.MissingNormWeight, layer.ln_1_b, norm_t, eps, "blk");
        cur.setName("blk");

        // Self-attention (QKV fused)
        {
            // QKV fused projection
            if (layer.qkv_w) |qkv_w| {
                cur = qkv_w.mulMat(ctx, cur);
                cur.setName("blk");
                if (layer.qkv_b) |qkv_b| {
                    cur = cur.add(ctx, qkv_b);
                    cur.setName("blk");
                }

                // Split Q, K, V from fused output
                const row_size_q = ggml.Type.rowSize(cur.dataType(), d_head);
                const row_size_kv = ggml.Type.rowSize(cur.dataType(), d_head);
                const offset_k = @as(usize, @intCast(ggml.Type.rowSize(cur.dataType(), n_embd)));
                const offset_v = @as(usize, @intCast(ggml.Type.rowSize(cur.dataType(), 2 * n_embd)));

                var Qcur = cur.view3d(ctx, d_head, n_head, n_patches, row_size_q, cur.nb()[1], 0);
                Qcur.setName("blk");
                var Kcur = cur.view3d(ctx, d_head, n_head, n_patches, row_size_kv, cur.nb()[1], offset_k);
                Kcur.setName("blk");
                var Vcur = cur.view3d(ctx, d_head, n_head, n_patches, row_size_kv, cur.nb()[1], offset_v);
                Vcur.setName("blk");

                // M-RoPE
                Qcur = Qcur.ropeMulti(ctx, positions, null, @intCast(d_head / 2), &mrope_sections, 0, 32768, 10000, 1, 0, 1, 32, 1);
                Qcur.setName("blk");
                Kcur = Kcur.ropeMulti(ctx, positions, null, @intCast(d_head / 2), &mrope_sections, 0, 32768, 10000, 1, 0, 1, 32, 1);
                Kcur.setName("blk");

                // Attention
                var attn_out = try graph.buildAttn(
                    ctx,
                    layer.o_w orelse return error.MissingOutputWeight,
                    layer.o_b,
                    Qcur,
                    Kcur,
                    Vcur,
                    null,
                    kq_scale,
                    n_head,
                    "blk",
                    layer.attn_sinks,
                );
                attn_out.setName("blk");

                // Residual
                cur = inpL.add(ctx, attn_out);
                cur.setName("blk");
            } else {
                // Fallback to separate Q, K, V (like Qwen2VL)
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

                Qcur = Qcur.reshape3d(ctx, d_head, n_head, n_patches);
                Qcur.setName("blk");
                Kcur = Kcur.reshape3d(ctx, d_head, n_head, n_patches);
                Kcur.setName("blk");
                Vcur = Vcur.reshape3d(ctx, d_head, n_head, n_patches);
                Vcur.setName("blk");

                Qcur = Qcur.ropeMulti(ctx, positions, null, @intCast(d_head / 2), &mrope_sections, 0, 32768, 10000, 1, 0, 1, 32, 1);
                Qcur.setName("blk");
                Kcur = Kcur.ropeMulti(ctx, positions, null, @intCast(d_head / 2), &mrope_sections, 0, 32768, 10000, 1, 0, 1, 32, 1);
                Kcur.setName("blk");

                var attn_out = try graph.buildAttn(
                    ctx,
                    layer.o_w orelse return error.MissingOutputWeight,
                    layer.o_b,
                    Qcur,
                    Kcur,
                    Vcur,
                    null,
                    kq_scale,
                    n_head,
                    "blk",
                    layer.attn_sinks,
                );
                attn_out.setName("blk");

                cur = inpL.add(ctx, attn_out);
                cur.setName("blk");
            }
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

            inpL = inpL.add(ctx, ffn_out);
            inpL.setName("blk");
        }

        // Deepstack feature extraction
        if (layer.hasDeepstack()) {
            var feat = inpL.cont3d(ctx, n_embd * merge_factor, n_patches / merge_factor, n_batch);
            feat.setName("deepstack_reshape");

            feat = try graph.buildNorm(ctx, feat, layer.deepstack_norm_w orelse return error.MissingNormWeight, layer.deepstack_norm_b, norm_t, eps, "deepstack");
            feat.setName("deepstack_norm");

            feat = try graph.buildFFN(
                ctx,
                feat,
                layer.deepstack_fc1_w orelse return error.MissingFFNUpWeight,
                layer.deepstack_fc1_b,
                null,
                null,
                layer.deepstack_fc2_w orelse return error.MissingFFNDownWeight,
                layer.deepstack_fc2_b,
                .gelu,
                "deepstack",
            );
            feat.setName("deepstack_ffn");

            if (deepstack_features) |dsf| {
                deepstack_features = dsf.concat(ctx, feat, 0);
                deepstack_features.?.setName("deepstack_concat");
            } else {
                deepstack_features = feat;
            }
        }
    }

    // 10. Post-LN (optional)
    if (w.post_ln_w) |poln_w| {
        inpL = try graph.buildNorm(ctx, inpL, poln_w, w.post_ln_b, norm_t, eps, "post_ln");
    }

    // 11. Multimodal projection
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

    // Concat deepstack features
    if (deepstack_features) |dsf| {
        embeddings = embeddings.concat(ctx, dsf, 0);
        embeddings.setName("mm_with_deepstack");
    }

    embeddings.setName("mm_output");

    // 构建计算图
    builder.gf.buildForwardExpand(embeddings);

    log.info("Qwen3VL graph built successfully", .{});
    return builder.gf;
}
