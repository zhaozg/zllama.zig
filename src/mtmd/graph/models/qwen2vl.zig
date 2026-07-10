//! Qwen2VL 视觉编码器图构建
//!
//! 实现 Qwen2VL 视觉编码器的计算图构建。
//! 使用 temporal merge (双 Conv2D) + spatial merge + M-RoPE。
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/qwen2vl.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("../mod.zig");

const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const BuildVitOpts = graph.BuildVitOpts;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const VisionHParams = graph.VisionHParams;
const ViTLayerWeights = graph.ViTLayerWeights;
const ImageF32 = graph.ImageF32;

const log = std.log.scoped(.qwen2vl_graph);

// ============================================================================
// 视觉编码器后端注册
// ============================================================================

/// Qwen2VL 视觉编码器后端实例
pub const backend = graph.VisionEncoderBackend{
    .name = "qwen2vl",
    .loadParams = loadParams,
    .loadWeights = loadWeights,
    .loadClampInfo = loadClampInfo,
    .buildGraph = buildGraphFromWeights,
    .estimateOutputTokens = estimateOutputTokens,
};

/// 从 GGUF 元数据读取视觉编码器超参数
pub fn loadParams(io: std.Io, gguf_file: *const gguf.GGUFFile, params: *graph.VisionHParams) void {
    _ = io;
    _ = gguf_file;
    _ = params;
    // Qwen2VL 参数已由 encoder.zig 从 clip.vision.* 前缀加载
}

/// 从 GGUF 加载 clamp 信息（Qwen2VL 不使用 clamp）
pub fn loadClampInfo(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    _ = gguf_file;
    w.clamp_info_map = std.StringHashMap(graph.ClampInfo).init(allocator);
}

/// 从 GGUF 加载 Qwen2VL 视觉编码器所有权重到 VisionEncoderWeights
pub fn loadWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    // Patch embedding (v.patch_embd.*)
    w.patch_embeddings_0 = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight") catch null;
    w.patch_embeddings_1 = findTensorInGGUF(ctx, gguf_file, "v.patch_embd_1.weight") catch null;
    w.patch_bias = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.bias") catch null;

    // 位置编码
    w.position_embeddings = findTensorInGGUF(ctx, gguf_file, "v.position_embd.weight") catch null;

    // Pre/Post LN
    w.pre_ln_w = findTensorInGGUF(ctx, gguf_file, "v.pre_ln.weight") catch null;
    w.pre_ln_b = findTensorInGGUF(ctx, gguf_file, "v.pre_ln.bias") catch null;
    w.post_ln_w = findTensorInGGUF(ctx, gguf_file, "v.post_ln.weight") catch null;
    w.post_ln_b = findTensorInGGUF(ctx, gguf_file, "v.post_ln.bias") catch null;

    // 多模态投影 - 支持两种风格:
    // 1. Qwen2VL 风格: mm.0.weight / mm.1.weight (MLP)
    // 2. Gemma3 风格: mm.input_projection.weight (单线性层)
    w.mm_0_w = findTensorInGGUF(ctx, gguf_file, "mm.0.weight") catch null;
    w.mm_0_b = findTensorInGGUF(ctx, gguf_file, "mm.0.bias") catch null;
    w.mm_1_w = findTensorInGGUF(ctx, gguf_file, "mm.1.weight") catch null;
    w.mm_1_b = findTensorInGGUF(ctx, gguf_file, "mm.1.bias") catch null;
    w.mm_input_proj_w = findTensorInGGUF(ctx, gguf_file, "mm.input_projection.weight") catch null;
    w.mm_soft_emb_norm_w = findTensorInGGUF(ctx, gguf_file, "mm.soft_emb_norm.weight") catch null;

    // 检测实际层数
    var actual_n_layer: u32 = 0;
    for (0..64) |il| {
        var buf: [32]u8 = undefined;
        const test_name = try std.fmt.bufPrint(&buf, "v.blk.{d}.attn_q.weight", .{il});
        if (gguf_file.findTensor(test_name) == null) break;
        actual_n_layer = @intCast(il + 1);
    }

    const n_layer: usize = @intCast(actual_n_layer);
    w.layers = try allocator.alloc(ViTLayerWeights, n_layer);

    for (0..n_layer) |il| {
        const prefix = try std.fmt.allocPrint(allocator, "v.blk.{d}", .{il});
        defer allocator.free(prefix);
        w.layers[il] = loadViTLayer(ctx, gguf_file, prefix) catch |err| {
            log.err("Failed to load ViT layer {d}: {}", .{ il, err });
            return err;
        };
    }

    log.info("Qwen2VL weights loaded: {d} ViT layers", .{n_layer});
}

/// 从 VisionEncoderWeights 构建计算图的包装函数
fn buildGraphFromWeights(
    io: std.Io,
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const graph.VisionHParams,
    image_tensor: *ggml.Tensor,
) !*ggml.CGraph {
    _ = io;
    const img_buf = try image_tensor.dataGet(f32, std.heap.page_allocator);
    // Note: img_buf is intentionally leaked (leak-to-exit) since ImageF32
    // is used during graph construction and the data must remain valid.
    const img = ImageF32{
        .buf = img_buf,
        .nx = p.image_size,
        .ny = p.image_size,
    };

    var hparams = VisionHParams{
        .image_size = p.image_size,
        .patch_size = p.patch_size,
        .n_embd = p.n_embd,
        .n_head = p.n_head,
        .n_layer = p.n_layer,
        .n_ff = p.n_ff,
        .projection_dim = p.projection_dim,
        .n_merge = p.n_merge,
        .eps = p.eps,
        .rope_theta = p.rope_theta,
        .n_wa_pattern = p.n_wa_pattern,
        .wa_layer_indexes = p.wa_layer_indexes,
        .wa_pattern_mode = p.wa_pattern_mode,
    };

    var builder = GraphBuilder{
        .weights = w,
        .hparams = &hparams,
        .proj_type = .qwen2vl,
        .img = &img,
        .ctx0 = ctx,
        .gf = gf,
    };

    return buildGraph(&builder);
}

/// 估算输出 token 数量
pub fn estimateOutputTokens(io: std.Io, img_width: u32, img_height: u32, patch_size: u32, n_merge: u32) u32 {
    _ = io;
    const patches_x = (img_width + patch_size - 1) / patch_size;
    const patches_y = (img_height + patch_size - 1) / patch_size;
    const n_patches = patches_x * patches_y;
    const merge = if (n_merge > 0) n_merge else 1;
    return n_patches / (merge * merge);
}

// ============================================================================
// 权重加载辅助函数
// ============================================================================

fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    const weight_loader = @import("weight_loader");
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name);
}

fn loadViTLayer(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
) !ViTLayerWeights {
    const weight_loader = @import("weight_loader");
    var layer = ViTLayerWeights{};

    // Attention
    layer.ln_1_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln1.weight") catch null;
    layer.ln_1_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln1.bias") catch null;
    layer.q_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight") catch null;
    layer.q_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.bias") catch null;
    layer.k_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch null;
    layer.k_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k.bias") catch null;
    layer.v_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch null;
    layer.v_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_v.bias") catch null;
    layer.o_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_out.weight") catch null;
    layer.o_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_out.bias") catch null;

    // FFN
    layer.ln_2_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln2.weight") catch null;
    layer.ln_2_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln2.bias") catch null;
    layer.ff_up_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch null;
    layer.ff_up_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.bias") catch null;
    layer.ff_gate_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight") catch null;
    layer.ff_gate_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.bias") catch null;
    layer.ff_down_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch null;
    layer.ff_down_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.bias") catch null;

    return layer;
}

// ============================================================================
// 原始 buildGraph 函数（保留向后兼容）
// ============================================================================

/// 构建 Qwen2VL 完整计算图
///
/// 处理流程:
///   1. Temporal merge: 两个 Conv2D 相加 (patch_embeddings_0 + patch_embeddings_1)
///   2. Spatial merge: permute + reshape 合并空间维度
///   3. Pre-LN (可选)
///   4. ViT blocks (LayerNorm + 自注意力 + M-RoPE + FFN)
///   5. Post-LN (可选)
///   6. 多模态投影 (FFN 或单线性层)
///
/// 参考: llama.cpp qwen2vl.cpp build()
pub fn buildGraph(
    builder: *GraphBuilder,
) !*ggml.CGraph {
    const ctx = builder.ctx0;
    const w = builder.weights;
    const p = builder.hparams;
    const img = builder.img;

    const n_embd: i64 = @intCast(p.n_embd);
    const n_head: i64 = @intCast(p.n_head);
    const d_head = @divExact(n_embd, n_head);
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
    const mrope_sections = [_]i32{ @intCast(@divExact(d_head, @as(i64, 4))), @intCast(@divExact(d_head, @as(i64, 4))), @intCast(@divExact(d_head, @as(i64, 4))), @intCast(@divExact(d_head, @as(i64, 4))) };

    const use_window_attn = p.n_wa_pattern > 0;
    const n_wa_pattern: i64 = @intCast(p.n_wa_pattern);

    log.info("Qwen2VL graph: embd={d}, head={d}, d_head={d}, patches={d}x{d}={d}, norm={s}, window_attn={}\n", .{ n_embd, n_head, d_head, n_patches_x, n_patches_y, n_patches, @tagName(norm_t), use_window_attn });

    // 1. 创建输入张量
    const inp_raw = try ctx.newTensor3d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), 3);
    inp_raw.setName("inp_raw");
    {
        const n_elems = @as(usize, @intCast(inp_raw.nElems()));
        const dst = try std.heap.page_allocator.alloc(f32, n_elems);
        defer std.heap.page_allocator.free(dst);
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
        try inp_raw.dataSet(f32, dst);
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
    inp = inp.cont4d(ctx, n_embd * 2, @divExact(n_patches_x, @as(i64, 2)), n_patches_y, n_batch);
    inp.setName("spatial_reshaped_1");

    // Reshape to [n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2)]
    inp = inp.reshape4d(ctx, n_embd * 2, @divExact(n_patches_x, @as(i64, 2)), 2, n_batch * @divExact(n_patches_y, @as(i64, 2)));
    inp.setName("spatial_reshaped_2");

    // permute(0, 2, 1, 3)
    inp = inp.permute(ctx, 0, 2, 1, 3).cont(ctx);
    inp.setName("spatial_permuted_2");

    // cont to [n_embd, n_patches_x * n_patches_y, batch_size]
    inp = ggml.cont(ctx, inp).reshape3d(ctx, n_embd, n_patches_x * n_patches_y, n_batch);
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
        inv_window_idx = try ctx.newTensor1d(ggml.Type.i32, @divExact(n_patches, @as(i64, 4)));
        inv_window_idx.?.setName("inv_window_idx");

        // window mask
        window_mask = try ctx.newTensor2d(ggml.Type.f32, n_patches, n_patches);
        window_mask.?.setName("window_mask");

        // Reorder patches using inv_window_idx
        inpL = inpL.reshape2d(ctx, n_embd * 4, @divExact(n_patches * n_batch, @as(i64, 4)));
        inpL.setName("window_reorder_flat");
        inpL = inpL.getRows(ctx, inv_window_idx.?);
        inpL.setName("window_reorder");
        inpL = ggml.cont(ctx, inpL).reshape3d(ctx, n_embd, n_patches, n_batch);
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

            // M-RoPE (GGML_ROPE_TYPE_VISION = 24)
            Qcur = ggml.ropeMulti(ctx, Qcur, positions, @intCast(@divExact(d_head, @as(i64, 2))), &mrope_sections, 24, 32768, 10000, 1, 0, 1, 32, 1);
            Qcur.setName("blk");
            Kcur = ggml.ropeMulti(ctx, Kcur, positions, @intCast(@divExact(d_head, @as(i64, 2))), &mrope_sections, 24, 32768, 10000, 1, 0, 1, 32, 1);
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

    // 支持两种投影方式:
    // 1. Qwen2VL 风格: mm.0 (GELU) -> mm.1 (MLP), 需要 spatial merge reshape
    // 2. Gemma3 风格: mm.input_projection (单线性层), 使用原始 ViT 输出
    if (w.mm_0_w != null and w.mm_1_w != null) {
        // Qwen2VL 风格: spatial merge + MLP 投影
        embeddings = ggml.cont(ctx, embeddings).reshape3d(ctx, n_embd * 4, @divExact(n_patches, @as(i64, 4)), n_batch);
        embeddings.setName("mm_reshape");

        embeddings = try graph.buildFFN(
            ctx,
            embeddings,
            w.mm_0_w.?,
            w.mm_0_b,
            null,
            null,
            w.mm_1_w.?,
            w.mm_1_b,
            .gelu,
            "mm_proj",
        );
    } else if (w.mm_input_proj_w != null) {
        // Gemma3 风格: 直接使用原始 ViT 输出 [n_embd, n_patches]
        embeddings = embeddings.rmsNorm(ctx, eps);
        embeddings.setName("mm_norm");
        if (w.mm_soft_emb_norm_w) |sn| {
            embeddings = embeddings.mul(ctx, graph.reshapeForBroadcast(ctx, sn));
            embeddings.setName("mm_norm_scaled");
        }
        embeddings = w.mm_input_proj_w.?.mulMat(ctx, embeddings);
    } else {
        return error.MissingMMWeight;
    }
    embeddings.setName("mm_proj");

    // Window attention reorder back (if applicable)
    if (use_window_attn) {
        var window_idx = try ctx.newTensor1d(ggml.Type.i32, @divExact(n_patches, @as(i64, 4)));
        window_idx.setName("window_idx");

        embeddings = embeddings.reshape2d(ctx, p.projection_dim, @divExact(n_patches, @as(i64, 4)));
        embeddings.setName("window_reorder_back_flat");
        embeddings = embeddings.getRows(ctx, window_idx);
        embeddings.setName("window_reorder_back");
        embeddings = ggml.cont(ctx, embeddings).reshape3d(ctx, @intCast(p.projection_dim), @divExact(n_patches, @as(i64, 4)), n_batch);
        embeddings.setName("window_reordered_back");
    }

    embeddings.setName("mm_output");

    // 构建计算图
    builder.gf.buildForwardExpand(embeddings);

    log.info("Qwen2VL graph built successfully", .{});
    return builder.gf;
}
