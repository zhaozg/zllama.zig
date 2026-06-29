//! Gemma4UV 视觉编码器图构建
//!
//! 实现 Gemma4UnifiedVisionEmbedder 的计算图构建。
//! 与 Gemma4V 不同，Gemma4UV 使用 im2col + patch norms + 投影，
//! 没有 ViT blocks 和 pooling。
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/gemma4uv.cpp

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

const log = std.log.scoped(.gemma4uv_graph);

// 构建 Gemma4UV 完整计算图
//
// 处理流程:
//   1. im2col + patch norm 1 (LayerNorm)
//   2. 投影到 embedding 维度 (patch_embeddings_0)
//   3. patch norm 2 (LayerNorm)
//   4. 2D 位置编码 (X/Y 分别编码)
//   5. patch norm 3 (LayerNorm)
//   6. RMSNorm + 投影到 LLM 嵌入空间
//
// 参考: llama.cpp gemma4uv.cpp build()
pub fn buildGraph(
    builder: *GraphBuilder,
) !*ggml.CGraph {
    const ctx = builder.ctx0;
    const w = builder.weights;
    const p = builder.hparams;
    const img = builder.img;

    const n_embd: i64 = @intCast(p.n_embd);
    const img_width: u32 = p.image_size;
    const img_height: u32 = p.image_size;
    const patch_size: i32 = @intCast(p.patch_size);
    const n_patches_x: i64 = @divTrunc(@as(i64, @intCast(img_width)), patch_size);
    const n_patches_y: i64 = @divTrunc(@as(i64, @intCast(img_height)), patch_size);
    const n_patches: i64 = n_patches_x * n_patches_y;
    const eps: f32 = 1e-5; // Gemma4UV uses pytorch LayerNorm default eps

    log.info("Gemma4UV graph: embd={d}, patches={d}x{d}={d}", .{ n_embd, n_patches_x, n_patches_y, n_patches });

    // 1. 创建输入张量
    // 输入图像: [3, height, width] f32, 值范围 [0, 1]
    const inp_raw = try ctx.newTensor3d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), 3);
    inp_raw.setName("inp_raw");
    // 填充输入数据
    {
        const dst = inp_raw.dataF32();
        const src = img.buf;
        const H: usize = @intCast(img_height);
        const W: usize = @intCast(img_width);
        for (0..H) |y| {
            for (0..W) |x| {
                const src_idx = (y * W + x) * 3;
                const dst_base = y * W + x;
                dst[dst_base] = src[src_idx];           // R
                dst[dst_base + H * W] = src[src_idx + 1]; // G
                dst[dst_base + 2 * H * W] = src[src_idx + 2]; // B
            }
        }
    }

    // 2. im2col + patch norms + projection
    var cur: *ggml.Tensor = undefined;

    // im2col: extract patches
    if (w.patch_norm_1_w) |pn1_w| {
        if (w.patch_embeddings_0) |pe| {
            const kw: i32 = @intCast(pe.ne()[0]);
            const kh: i32 = @intCast(pe.ne()[1]);
            const ic: i32 = @intCast(pe.ne()[2]);

            // Create a dummy kernel for im2col (shape is used for layout)
            const kernel = try ctx.newTensor3d(ggml.Type.f32, kw, kh, ic);
            kernel.setName("im2col_kernel");

            // im2col: [patch_size * patch_size * C, n_patches_w, n_patches_h]
            cur = inp_raw.im2col(ctx, kernel, kw, kh, 0, 0, 1, 1, true, ggml.Type.f32);
            cur.setName("im2col_out");

            // Flatten to [patch_size * patch_size * C, n_patches]
            cur = cur.reshape2d(ctx, cur.ne()[0], n_patches);
            cur.setName("im2col_flat");

            // Patch norm 1 (LayerNorm, not RMSNorm)
            cur = cur.norm(ctx, eps);
            cur.setName("patch_norm_1");
            cur = cur.mul(ctx, pn1_w);
            cur.setName("patch_norm_1_scaled");
            if (w.patch_norm_1_b) |pn1_b| {
                cur = cur.add(ctx, pn1_b);
                cur.setName("patch_norm_1_biased");
            }
        }
    }

    // Project to embedding dimension
    if (w.patch_embeddings_0) |pe| {
        cur = pe.mulMat(ctx, cur);
        cur.setName("patch_proj");
    }
    if (w.patch_bias) |pb| {
        cur = cur.add(ctx, pb);
        cur.setName("patch_bias_added");
    }

    // Patch norm 2 (LayerNorm)
    if (w.patch_norm_2_w) |pn2_w| {
        cur = cur.norm(ctx, eps);
        cur.setName("patch_norm_2");
        cur = cur.mul(ctx, pn2_w);
        cur.setName("patch_norm_2_scaled");
        if (w.patch_norm_2_b) |pn2_b| {
            cur = cur.add(ctx, pn2_b);
            cur.setName("patch_norm_2_biased");
        }
    }

    // 3. 2D 位置编码
    if (w.position_embeddings) |pos_embd| {
        const pos_size = pos_embd.ne()[1];
        const row_size = ggml.Type.rowSize(pos_embd.dataType(), n_embd);

        // X/Y 位置嵌入表
        const tbl_x = pos_embd.view2d(ctx, n_embd, pos_size, row_size, 0);
        tbl_x.setName("pos_tbl_x");
        const tbl_y = pos_embd.view2d(ctx, n_embd, pos_size, row_size, @as(usize, @intCast(pos_size)) * row_size);
        tbl_y.setName("pos_tbl_y");

        // 位置索引
        const indices = try graph.createPositionIndices(ctx, n_patches, n_patches_x);

        // getRows: [n_embd, n_patches]
        const emb_x = tbl_x.getRows(ctx, indices.pos_x);
        emb_x.setName("pos_emb_x");
        const emb_y = tbl_y.getRows(ctx, indices.pos_y);
        emb_y.setName("pos_emb_y");

        cur = cur.add(ctx, emb_x);
        cur.setName("inp_with_pos_x");
        cur = cur.add(ctx, emb_y);
        cur.setName("inp_with_pos");

        // Patch norm 3 (LayerNorm)
        if (w.patch_norm_3_w) |pn3_w| {
            cur = cur.norm(ctx, eps);
            cur.setName("patch_norm_3");
            cur = cur.mul(ctx, pn3_w);
            cur.setName("patch_norm_3_scaled");
            if (w.patch_norm_3_b) |pn3_b| {
                cur = cur.add(ctx, pn3_b);
                cur.setName("patch_norm_3_biased");
            }
        }
    }

    // 4. Gemma4UnifiedMultimodalEmbedder
    //    embedding_pre_projection_norm
    cur = cur.rmsNorm(ctx, p.eps);
    cur.setName("mm_pre_norm");

    // 投影到 LLM 嵌入空间
    if (w.mm_input_proj_w) |proj_w| {
        cur = proj_w.mulMat(ctx, cur);
        cur.setName("mm_proj");
    }

    cur.setName("mm_output");

    // 构建计算图
    builder.gf.buildForwardExpand(cur);

    log.info("Gemma4UV graph built successfully", .{});
    return builder.gf;
}
