//! Gemma4UV 视觉编码器图构建
//!
//! 实现 Gemma4UnifiedVisionEmbedder 的计算图构建。
//! 与 Gemma4V 不同，Gemma4UV 使用 im2col + patch norms + 投影，
//! 没有 ViT blocks 和 pooling。
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/gemma4uv.cpp

const std = @import("std");
const graph = @import("../mod.zig");
const gguf = @import("gguf");
const ggml = @import("ggml");
const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const BuildVitOpts = graph.BuildVitOpts;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const VisionHParams = graph.VisionHParams;
const ViTLayerWeights = graph.ViTLayerWeights;
const ImageF32 = graph.ImageF32;
const ClampInfo = graph.ClampInfo;

const log = std.log.scoped(.graph_model_gemma4uv);

// ============================================================================
// 视觉编码器后端注册
pub const backend = graph.VisionEncoderBackend{
    .name = "gemma4uv",
    .loadParams = loadParams,
    .loadWeights = loadWeights,
    .loadClampInfo = loadClampInfo,
    .buildGraph = buildGraphFromWeights,
    .estimateOutputTokens = estimateOutputTokens,
};

pub fn loadParams(io: std.Io, gguf_file: *const gguf.GGUFFile, params: *graph.VisionHParams) void {
    _ = io;
    // 参考: llama.cpp clip.cpp PROJECTOR_TYPE_GEMMA4UV (lines 1434-1448)
    // 与 GEMMA4V 共享同一个 case 分支:
    //   hparams.rope_theta = 100.0f;
    //   hparams.n_merge = 3;
    //   get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
    //   if (proj_type == GEMMA4UV) { patch_size *= n_merge; n_merge = 1; }
    //   hparams.set_limit_image_tokens(40, 280);
    //   hparams.set_warmup_n_tokens(256);
    params.rope_theta = 100.0;
    params.n_merge = 3;
    if (gguf_file.getU32("clip.vision.projector.scale_factor")) |v| {
        params.n_merge = v;
    }
    // 对于 "unified" 变体，token merging 直接在 conv 层完成，
    // 因此使用更大的 patch_size 并将 n_merge 设为 1。
    if (params.n_merge > 0) {
        params.patch_size *= params.n_merge;
        params.n_merge = 1;
    }
    // 限制图像 tokens 范围：最小值 40，最大值 280
    params.setLimitImageTokens(40, 280);
    // warmup: sqrt(256) * patch_size * n_merge
    params.warmup_image_size = 16 * params.patch_size * params.n_merge;
}
/// 从 GGUF 加载 Gemma4UV 视觉编码器所有权重到 VisionEncoderWeights
pub fn loadWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    // Patch embedding
    w.patch_embeddings_0 = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight") catch null;
    w.patch_bias = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.bias") catch null;

    // Patch 归一化
    w.patch_norm_1_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.1.weight") catch null;
    w.patch_norm_1_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.1.bias") catch null;
    w.patch_norm_2_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.2.weight") catch null;
    w.patch_norm_2_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.2.bias") catch null;
    w.patch_norm_3_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.3.weight") catch null;
    w.patch_norm_3_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.3.bias") catch null;

    // 位置编码
    w.position_embeddings = findTensorInGGUF(ctx, gguf_file, "v.position_embd.weight") catch null;

    // 多模态投影
    w.mm_input_proj_w = findTensorInGGUF(ctx, gguf_file, "mm.input_projection.weight") catch null;
    w.mm_soft_emb_norm_w = findTensorInGGUF(ctx, gguf_file, "mm.soft_emb_norm.weight") catch null;

    // Gemma4UV 没有 ViT layers
    w.layers = try allocator.alloc(ViTLayerWeights, 0);

    log.info("Gemma4UV weights loaded (no ViT layers)", .{});
}

/// 从 GGUF 加载 Gemma4UV 视觉编码器的 clamp 信息
pub fn loadClampInfo(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    var weight_names = std.ArrayList([]const u8).initCapacity(allocator, 0) catch |err| return err;
    defer weight_names.deinit(allocator);

    if (w.patch_embeddings_0) |t| try weight_names.append(allocator, t.getName());
    if (w.mm_input_proj_w) |t| try weight_names.append(allocator, t.getName());

    w.clamp_info_map = try graph.clamp.loadClampInfoFromWeightNames(allocator, gguf_file, weight_names.items);
    log.info("Gemma4UV clamp info loaded: {d} entries", .{w.clamp_info_map.count()});
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
    };

    var builder = GraphBuilder{
        .weights = w,
        .hparams = &hparams,
        .proj_type = .gemma4uv,
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

// ============================================================================
// buildGraph 函数
// ============================================================================

/// 构建 Gemma4UV 完整计算图
///
/// 处理流程:
///   1. 创建输入张量 [W, H, C, 1], set_input (match C++ build_inp_raw)
///   2. im2col + patch norm 1 (LayerNorm)
///   3. 投影到 embedding 维度 (patch_embeddings_0 via mul_mat, no clamp)
///   4. patch norm 2 (LayerNorm)
///   5. 2D 位置编码 (X/Y lookup table + get_rows)
///   6. patch norm 3 (LayerNorm)
///   7. RMSNorm + 投影到 LLM 嵌入空间 (mm_input_proj_w via build_mm)
///
/// 参考: llama.cpp gemma4uv.cpp build()
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

    // Determine effective patch_size for im2col.
    // If patch_embeddings_0 is a 4D conv kernel [kh, kw, ic, oc] with kh == 16
    // (gemma4v format), use the original patch_size=16 instead of the modified one.
    // If it's [48, 48, 3, 768] (gemma4uv native format), use the modified patch_size.
    const effective_patch_size: i32 = blk: {
        if (w.patch_embeddings_0) |pe| {
            if (pe.nDims() == 4 and pe.ne()[0] == 16) {
                // gemma4v format weight [16, 16, 3, 768] — use original patch_size
                break :blk 16;
            }
        }
        break :blk @as(i32, @intCast(p.patch_size));
    };
    const patch_size: i32 = effective_patch_size;
    const n_patches_x: i64 = @divTrunc(@as(i64, @intCast(img_width)), patch_size);
    const n_patches_y: i64 = @divTrunc(@as(i64, @intCast(img_height)), patch_size);
    const n_patches: i64 = n_patches_x * n_patches_y;
    const eps: f32 = 1e-5; // Gemma4UV uses pytorch LayerNorm default eps

    log.info("Gemma4UV graph: embd={d}, patches={d}x{d}={d}, patch_size={d}", .{ n_embd, n_patches_x, n_patches_y, n_patches, patch_size });

    // ========================================================================
    // 1. 创建输入张量 (match C++ build_inp_raw: 4D [W, H, C, 1], set_input)
    // ========================================================================
    const n_batch: i64 = 1;
    const inp_raw = try ctx.newTensor4d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), 3, n_batch);
    inp_raw.setName("inp_raw");
    ggml.setInput(inp_raw);

    // 在 no_alloc 模式下，setInput 标记的张量不会被 Gallocr 分配，需要手动分配
    if (ctx.getNoAlloc()) {
        const data_size = @as(usize, @intCast(inp_raw.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
        @memset(buf, 0);
        inp_raw.setDataPtr(buf);
    }

    // 填充输入数据 — HWC→CHW 转换
    // resizeAndNormalize 输出 HWC 布局，而 inp_raw 需要 CHW 布局（ggml im2col 期望）
    // Ref: llama.cpp clip.cpp lines 3645-3665
    {
        const n_elems = @as(usize, @intCast(inp_raw.nElems()));
        const dst = try std.heap.page_allocator.alloc(f32, n_elems);
        defer std.heap.page_allocator.free(dst);
        const src = img.buf;
        const H: usize = @intCast(img_height);
        const W: usize = @intCast(img_width);
        for (0..H) |y| {
            for (0..W) |x| {
                const hwc_idx = (y * W + x) * 3;
                const chw_base = y * W + x;
                dst[chw_base] = src[hwc_idx]; // R
                dst[chw_base + H * W] = src[hwc_idx + 1]; // G
                dst[chw_base + 2 * H * W] = src[hwc_idx + 2]; // B
            }
        }
        try inp_raw.dataSet(f32, dst);
    }

    // ========================================================================
    // 2. im2col + patch norms + projection

    // ========================================================================
    // 2. im2col + patch norms + projection
    // ========================================================================

    // ========================================================================

    // im2col: extract patches
    // Kernel shape: [patch_size, patch_size, channels] — matches C++ gemma4uv.cpp
    // Note: we cannot use ggml_conv_2d here because we need to apply norm after im2col
    var cur: *ggml.Tensor = undefined;
    {
        const c: i32 = @intCast(inp_raw.ne()[2]); // channels (3 for RGB)
        const kernel = try ctx.newTensor3d(ggml.Type.f32, patch_size, patch_size, c);
        kernel.setName("im2col_kernel");

        // im2col: output shape [patch_size * patch_size * c, n_patches_w, n_patches_h]
        cur = inp_raw.im2col(ctx, kernel, patch_size, patch_size, 0, 0, 1, 1, true, ggml.Type.f32);
        cur.setName("im2col_out");

        // Flatten to [patch_size * patch_size * c, n_patches]
        cur = cur.reshape2d(ctx, cur.ne()[0], cur.ne()[1] * cur.ne()[2] * cur.ne()[3]);
        cur.setName("im2col_flat");
    }

    // Patch norm 1 (LayerNorm, not RMSNorm)
    // Note: C++ uses build_norm(inp, patch_norm_1_w, patch_norm_1_b, NORM_TYPE_NORMAL, eps, -1)
    if (w.patch_norm_1_w) |pn1_w| {
        cur = cur.norm(ctx, eps);
        cur.setName("patch_norm_1");
        cur = cur.mul(ctx, pn1_w);
        cur.setName("patch_norm_1_scaled");
        if (w.patch_norm_1_b) |pn1_b| {
            cur = cur.add(ctx, pn1_b);
            cur.setName("patch_norm_1_biased");
        }
    }

    // Project to embedding dimension via mul_mat (no clamp for gemma4uv)
    // Note: C++ clip_graph_gemma4uv does NOT override build_mm, so it uses plain ggml_mul_mat
    if (w.patch_embeddings_0) |pe| {
        // The weight may be stored as 4D conv kernel [kh, kw, ic, oc] (gemma4v format)
        // or as 2D matrix [in_features, out_features] (gemma4uv native format).
        // Reshape to 2D if needed for mul_mat compatibility.
        const pe_reshaped = if (pe.nDims() == 4) blk: {
            // Shape [kh, kw, ic, oc] = [16, 16, 3, 768]
            // Reshape to [kh*kw*ic, oc] = [768, 768]
            const in_features = pe.ne()[0] * pe.ne()[1] * pe.ne()[2];
            const out_features = pe.ne()[3];
            break :blk pe.reshape2d(ctx, in_features, out_features);
        } else pe;
        pe_reshaped.setName("patch_embd_2d");
        cur = pe_reshaped.mulMat(ctx, cur);
        cur.setName("patch_proj");
    }
    // Always add bias (C++ gemma4uv.cpp line 17: unconditional ggml_add)
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

    // ========================================================================
    // 3. 2D 位置编码
    // ========================================================================
    if (w.position_embeddings) |pos_embd| {
        const pos_size = pos_embd.ne()[1];
        const row_size = ggml.Type.rowSize(pos_embd.dataType(), n_embd);

        // X/Y 位置嵌入表 (matching C++ gemma4uv.cpp lookup tables)
        const tbl_x = pos_embd.view2d(ctx, n_embd, pos_size, row_size, 0);
        tbl_x.setName("pos_tbl_x");
        const tbl_y = pos_embd.view2d(ctx, n_embd, pos_size, row_size, @as(usize, @intCast(pos_size)) * row_size);
        tbl_y.setName("pos_tbl_y");

        // Position indices as input tensors (matching C++ gemma4uv.cpp ggml_set_input)
        const pos_x = try ctx.newTensor1d(ggml.Type.i32, n_patches);
        pos_x.setName("pos_x");
        ggml.setInput(pos_x);

        const pos_y = try ctx.newTensor1d(ggml.Type.i32, n_patches);
        pos_y.setName("pos_y");
        ggml.setInput(pos_y);

        // In no_alloc mode, setInput tensors need manually allocated data buffers
        if (ctx.getNoAlloc()) {
            const px_size = @as(usize, @intCast(pos_x.nBytes()));
            const buf_x = @as([*]u8, @ptrCast(std.c.malloc(px_size) orelse return error.OutOfMemory))[0..px_size];
            pos_x.setDataPtr(buf_x);
            const buf_y = @as([*]u8, @ptrCast(std.c.malloc(px_size) orelse return error.OutOfMemory))[0..px_size];
            pos_y.setDataPtr(buf_y);
        }

        // Fill position indices: x = col, y = row
        {
            const px = pos_x.dataI32();
            const py = pos_y.dataI32();
            for (0..@as(usize, @intCast(n_patches_y))) |iy| {
                for (0..@as(usize, @intCast(n_patches_x))) |ix| {
                    const idx = iy * @as(usize, @intCast(n_patches_x)) + ix;
                    px[idx] = @intCast(ix);
                    py[idx] = @intCast(iy);
                }
            }
        }

        // getRows: [n_embd, n_patches]
        const emb_x = tbl_x.getRows(ctx, pos_x);
        emb_x.setName("pos_emb_x");
        const emb_y = tbl_y.getRows(ctx, pos_y);
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

    // ========================================================================
    // 4. Gemma4UnifiedMultimodalEmbedder: RMSNorm + projection
    // ========================================================================
    // embedding_pre_projection_norm
    cur = cur.rmsNorm(ctx, p.eps);
    cur.setName("mm_pre_norm");

    // 投影到 LLM 嵌入空间
    // Note: C++ clip_graph_gemma4uv uses base build_mm (plain ggml_mul_mat, no clamp).
    // We use buildMMWithClamp for safety; if no clamp info exists it falls through to plain mul_mat.
    if (w.mm_input_proj_w) |proj_w| {
        cur = buildMMWithClamp(ctx, proj_w, cur, &w.clamp_info_map);
        cur.setName("mm_proj");
    }

    cur.setName("mm_output");

    // 构建计算图
    builder.gf.buildForwardExpand(cur);

    log.info("Gemma4UV graph built successfully", .{});
    return builder.gf;
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 带 clamp 的矩阵乘法
/// 对应 C++ clip_graph_gemma4v::build_mm().
/// Note: gemma4uv does NOT override build_mm in C++ (uses default ggml_mul_mat),
/// but we support clamp here as a safety enhancement — if no clamp info exists,
/// it falls through to plain mul_mat.
fn buildMMWithClamp(
    ctx: *ggml.Context,
    w: *ggml.Tensor,
    x: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) *ggml.Tensor {
    const name = w.getName();
    if (clamp_map.get(name)) |ci| {
        const clamped = x.clamp(ctx, ci.inp_min, ci.inp_max);
        var out = w.mulMat(ctx, clamped);
        out = out.clamp(ctx, ci.out_min, ci.out_max);
        return out;
    } else {
        return w.mulMat(ctx, x);
    }
}
