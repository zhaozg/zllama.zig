//! Gemma4V 视觉编码器图构建
//!
//! 实现 Gemma4V 视觉编码器的计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/models/gemma4v.cpp

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
const ClampInfo = graph.ClampInfo;

const log = std.log.scoped(.gemma4v_graph);

// ============================================================================
// 2D RoPE add_pos 回调上下文
// ============================================================================

/// 2D RoPE add_pos 回调的上下文数据
/// 参考: gemma4v.cpp add_pos lambda (lines 46-93)
pub const AddPosContext = struct {
    pos_x_tensor: *ggml.Tensor,
    pos_y_tensor: *ggml.Tensor,
    rope_theta_val: f32,
    n_batch_val: i64,

    pub fn callback(
        ctx_ptr: *ggml.Context,
        cur_tensor: *ggml.Tensor,
        _: *const ViTLayerWeights,
        _: *ggml.Tensor,
        user_data: ?*anyopaque,
    ) *ggml.Tensor {
        const self = @as(*AddPosContext, @ptrCast(@alignCast(user_data.?)));
        const n_dim = cur_tensor.ne()[0];
        const n_head_val = cur_tensor.ne()[1];
        const n_pos_val = cur_tensor.ne()[2];
        const n_dim_half = @divExact(n_dim, @as(i64, 2));

        // first half: use pos_x
        var first = cur_tensor.view4d(
            ctx_ptr,
            n_dim_half,
            n_head_val,
            n_pos_val,
            self.n_batch_val,
            cur_tensor.nb()[1],
            cur_tensor.nb()[2],
            cur_tensor.nb()[3],
            0,
        );
        first = first.ropeExt(
            ctx_ptr,
            self.pos_x_tensor,
            null,
            @intCast(n_dim_half),
            2, // GGML_ROPE_TYPE_NEOX
            0,
            self.rope_theta_val,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );

        // second half: use pos_y
        const offset: usize = @intCast(n_dim_half * @sizeOf(f32));
        var second = cur_tensor.view4d(
            ctx_ptr,
            n_dim_half,
            n_head_val,
            n_pos_val,
            self.n_batch_val,
            cur_tensor.nb()[1],
            cur_tensor.nb()[2],
            cur_tensor.nb()[3],
            offset,
        );
        second = second.ropeExt(
            ctx_ptr,
            self.pos_y_tensor,
            null,
            @intCast(n_dim_half),
            2, // GGML_ROPE_TYPE_NEOX
            0,
            self.rope_theta_val,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );

        const result = first.concat(ctx_ptr, second, 0);
        return result;
    }
};

// ============================================================================
// 视觉编码器后端注册
// ============================================================================

/// Gemma4V 视觉编码器后端实例
pub const backend = graph.VisionEncoderBackend{
    .name = "gemma4v",
    .supportBatch = true,
    .loadParams = loadParams,
    .loadWeights = loadWeights,
    .loadClampInfo = loadClampInfo,
    .buildGraph = buildGraphFromWeights,
    .estimateOutputTokens = estimateOutputTokens,
};

/// 从 GGUF 元数据读取视觉编码器超参数
pub fn loadParams(gguf_file: *const gguf.GGUFFile, params: *graph.VisionHParams) void {
    _ = gguf_file;
    _ = params;
    // Gemma4V 参数已由 encoder.zig 从 clip.vision.* 前缀加载
    // 此处可添加模型特定参数覆盖
}

/// 从 GGUF 加载 Gemma4V 视觉编码器所有权重到 VisionEncoderWeights
pub fn loadWeights(
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    w: *VisionEncoderWeights,
) !void {
    // Patch embedding (v.patch_embd.*)
    w.patch_embeddings_0 = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight") catch null;
    w.patch_bias = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.bias") catch null;

    // 位置编码 (v.position_embd.weight)
    w.position_embeddings = findTensorInGGUF(ctx, gguf_file, "v.position_embd.weight") catch null;

    // 标准化 (v.std_bias, v.std_scale)
    w.std_bias = findTensorInGGUF(ctx, gguf_file, "v.std_bias") catch null;
    w.std_scale = findTensorInGGUF(ctx, gguf_file, "v.std_scale") catch null;

    // 多模态投影 (mm.*)
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

    log.info("Gemma4V weights loaded: {d} ViT layers", .{n_layer});
}

/// 从 GGUF 加载 Gemma4V 视觉编码器的 clamp 信息
/// 参考: llama.cpp gemma4a.cpp loadClampInfo() 和 gemma4v.cpp build_mm()
pub fn loadClampInfo(
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    w: *VisionEncoderWeights,
) !void {
    var weight_names = std.ArrayList([]const u8).initCapacity(allocator, 0) catch |err| return err;
    defer weight_names.deinit(allocator);

    // 收集所有权重名称（ViT 层 + MM 投影）
    if (w.patch_embeddings_0) |t| try weight_names.append(allocator, t.getName());
    if (w.mm_input_proj_w) |t| try weight_names.append(allocator, t.getName());

    for (w.layers) |*layer| {
        if (layer.q_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.k_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.v_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.o_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_up_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_down_w) |t| try weight_names.append(allocator, t.getName());
    }

    w.clamp_info_map = try graph.clamp.loadClampInfoFromWeightNames(allocator, gguf_file, weight_names.items);
    log.info("Gemma4V clamp info loaded: {d} entries", .{w.clamp_info_map.count()});
}

/// 从 VisionEncoderWeights 构建计算图的包装函数
fn buildGraphFromWeights(
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const graph.VisionHParams,
    image_tensor: *ggml.Tensor,
) !*ggml.CGraph {
    // 构建 ImageF32 用于 GraphBuilder
    const img = ImageF32{
        .buf = image_tensor.dataF32(),
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
        .proj_type = .gemma4v,
        .img = &img,
        .ctx0 = ctx,
        .gf = gf,
    };

    return buildGraph(&builder);
}

/// 估算输出 token 数量
pub fn estimateOutputTokens(img_width: u32, img_height: u32, patch_size: u32, n_merge: u32) u32 {
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
    layer.k_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch null;
    layer.v_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch null;
    layer.o_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_out.weight") catch null;
    layer.o_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_out.bias") catch null;

    // FFN
    layer.ln_2_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln2.weight") catch null;
    layer.ln_2_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln2.bias") catch null;
    layer.ff_up_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch null;
    layer.ff_down_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch null;

    return layer;
}

// ============================================================================
// 原始 buildGraph 函数（保留向后兼容）
// ============================================================================

/// 构建 Gemma4V 完整计算图
///
/// 处理流程:
///   1. 输入归一化 (scale=2, bias=-1)
///   2. Conv2D patch embedding
///   3. 2D 位置编码 (X/Y 分别编码)
///   4. ViT blocks (RMSNorm + 自注意力 + 2D RoPE + FFN)
///   5. Pooling (平均池化下采样)
///   6. 标准化 (std_bias, std_scale)
///   7. 投影到 LLM 嵌入空间
///
/// 参考: llama.cpp gemma4v.cpp build()
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
    const rope_theta = p.rope_theta;

    log.info("Gemma4V graph: embd={d}, head={d}, d_head={d}, patches={d}x{d}={d}, rope_theta={d}", .{ n_embd, n_head, d_head, n_patches_x, n_patches_y, n_patches, rope_theta });

    // ========================================================================
    // 1. 创建输入张量
    // ========================================================================
    // 创建输入张量
    // 参考: gemma4v.cpp build_inp_raw()
    const inp_raw = try ctx.newTensor4d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), 3, n_batch);
    inp_raw.setName("inp_raw");
    ggml.setInput(inp_raw);

    // 在 no_alloc 模式下，setInput 标记的张量不会被 Gallocr 分配，
    // 所以需要手动分配数据。Gallocr 只为非输入张量分配内存。
    if (ctx.getNoAlloc()) {
        const data_size = @as(usize, @intCast(inp_raw.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
        @memset(buf, 0);
        inp_raw.setDataPtr(buf);
    }

    // 填充输入数据
    // 这里假设 img.buf 包含 RGBRGBRGB... 格式的 f32 数据
    {
        const dst = inp_raw.dataF32();
        const src = img.buf;
        const H: usize = @intCast(img_height);
        const W: usize = @intCast(img_width);
        for (0..H) |y| {
            for (0..W) |x| {
                const src_idx = (y * W + x) * 3;
                const dst_base = y * W + x;
                dst[dst_base] = src[src_idx]; // R
                dst[dst_base + H * W] = src[src_idx + 1]; // G
                dst[dst_base + 2 * H * W] = src[src_idx + 2]; // B
            }
        }
    }

    // ========================================================================
    // 2. Scale + bias: patches * 2 - 1
    // 参考: gemma4v.cpp: ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f)
    // ========================================================================
    var cur = inp_raw;
    cur = cur.scale(ctx, 2.0);
    cur.setName("inp_scaled");
    {
        const bias_t = try ctx.newTensor1d(ggml.Type.f32, 1);
        // 在 no_alloc 模式下，需要手动为 bias_t 分配数据
        if (ctx.getNoAlloc()) {
            const buf = @as([*]u8, @ptrCast(std.c.malloc(@sizeOf(f32)) orelse return error.OutOfMemory))[0..@sizeOf(f32)];
            bias_t.setDataPtr(buf);
        }
        bias_t.dataF32()[0] = -1.0;
        bias_t.setName("inp_bias");
        cur = cur.add(ctx, bias_t);
        cur.setName("inp_raw_scaled");
    }

    // ========================================================================
    // 3. Conv2D patch embedding
    // 参考: gemma4v.cpp: ggml_conv_2d + reshape + transpose
    // ========================================================================
    if (w.patch_embeddings_0) |pe| {
        const kw: i32 = @intCast(pe.ne()[0]);
        const kh: i32 = @intCast(pe.ne()[1]);
        cur = cur.conv2d(ctx, pe, kw, kh, 0, 0, 1, 1);
        cur.setName("inp_conv");

        // Reshape to [n_patches, n_embd, n_batch] then transpose to [n_embd, n_patches, n_batch]
        cur = cur.reshape3d(ctx, n_patches, n_embd, n_batch);
        cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
        cur.setName("inp");
        // note: no patch bias (gemma4v.cpp line 16)
    }

    // ========================================================================
    // 4. 2D 位置编码
    // 参考: gemma4v.cpp: pos_x, pos_y as input tensors + get_rows
    // ========================================================================
    // 创建位置索引张量作为输入（匹配 C++ set_input）
    const pos_x = try ctx.newTensor1d(ggml.Type.i32, n_patches);
    pos_x.setName("pos_x");
    ggml.setInput(pos_x);

    const pos_y = try ctx.newTensor1d(ggml.Type.i32, n_patches);
    pos_y.setName("pos_y");
    ggml.setInput(pos_y);

    // 在 no_alloc 模式下，需要手动为位置索引张量分配数据
    if (ctx.getNoAlloc()) {
        const px_size = @as(usize, @intCast(pos_x.nBytes()));
        const buf_x = @as([*]u8, @ptrCast(std.c.malloc(px_size) orelse return error.OutOfMemory))[0..px_size];
        pos_x.setDataPtr(buf_x);
        const buf_y = @as([*]u8, @ptrCast(std.c.malloc(px_size) orelse return error.OutOfMemory))[0..px_size];
        pos_y.setDataPtr(buf_y);
    }

    // 填充位置索引
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

    if (w.position_embeddings) |pos_embd| {
        const pos_size = pos_embd.ne()[1];
        const row_size = ggml.Type.rowSize(pos_embd.dataType(), n_embd);

        // X/Y 位置嵌入表
        const tbl_x = pos_embd.view2d(ctx, n_embd, pos_size, row_size, 0);
        tbl_x.setName("pos_tbl_x");
        const tbl_y = pos_embd.view2d(ctx, n_embd, pos_size, row_size, @as(usize, @intCast(pos_size)) * row_size);
        tbl_y.setName("pos_tbl_y");

        // getRows: [n_embd, n_patches]
        const emb_x = tbl_x.getRows(ctx, pos_x);
        emb_x.setName("pos_emb_x");
        const emb_y = tbl_y.getRows(ctx, pos_y);
        emb_y.setName("pos_emb_y");

        cur = cur.add(ctx, emb_x);
        cur = cur.add(ctx, emb_y);
        cur.setName("pos_embd");
    }

    // ========================================================================
    // 5. ViT blocks (via buildVit with 2D RoPE add_pos callback)
    // 参考: gemma4v.cpp: build_vit(inp, n_patches, NORM_TYPE_RMS, hparams.ffn_op, nullptr, add_pos)
    // ========================================================================
    var add_pos_ctx = AddPosContext{
        .pos_x_tensor = pos_x,
        .pos_y_tensor = pos_y,
        .rope_theta_val = rope_theta,
        .n_batch_val = n_batch,
    };

    // 使用 buildVit 构建 ViT 主干
    // 注意: gemma4v 使用 kq_scale=1.0 (set in gemma4v.cpp line 95)
    // 并且对 V 应用 RMSNorm (gemma4v.cpp line 448-450)
    const vit_opts = BuildVitOpts{
        .v_norm = true,
        .v_norm_eps = eps,
        .kq_scale = 1.0,
    };

    cur = try graph.buildVit(
        ctx,
        cur,
        n_patches,
        .rms_norm,
        p.ffn_op,
        null, // learned_pos_embd - already handled above
        w,
        p,
        AddPosContext.callback,
        &add_pos_ctx,
        vit_opts,
    );

    // ========================================================================
    // 6. Pooling (平均池化下采样)
    // 参考: gemma4v.cpp: Gemma4VisionPooler (lines 103-119)
    // ========================================================================
    const kernel_size: i64 = @intCast(p.n_merge);
    {
        // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, n_batch]
        cur = ggml.cont4d(ctx, ggml.transpose(ctx, cur), n_patches_x, n_patches_y, n_embd, n_batch);
        cur.setName("pool_4d");

        // Average pooling
        cur = cur.pool2d(
            ctx,
            1, // GGML_OP_POOL_AVG
            @as(i32, @intCast(kernel_size)),
            @as(i32, @intCast(kernel_size)),
            @as(i32, @intCast(kernel_size)),
            @as(i32, @intCast(kernel_size)),
            0,
            0,
        );
        cur.setName("pool_avg");

        const out_x = @divExact(n_patches_x, kernel_size);
        const out_y = @divExact(n_patches_y, kernel_size);
        // [out_x, out_y, n_embd, n_batch] -> [n_embd, out_x * out_y, n_batch]
        cur = cur.reshape3d(ctx, out_x * out_y, n_embd, n_batch);
        cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
        cur = cur.scale(ctx, @sqrt(@as(f32, @floatFromInt(n_embd))));
        cur.setName("pooled");
    }

    // ========================================================================
    // 7. 标准化 (std_bias, std_scale)
    // 参考: gemma4v.cpp: hidden_states = (hidden_states - self.std_bias) * self.std_scale
    // ========================================================================
    if (w.std_bias) |sb| {
        cur = cur.sub(ctx, sb);
        cur.setName("std_sub");
    }
    if (w.std_scale) |ss| {
        cur = cur.mul(ctx, graph.reshapeForBroadcast(ctx, ss));
        cur.setName("std_scaled");
    }

    // ========================================================================
    // 8. 投影到 LLM 嵌入空间
    // 参考: gemma4v.cpp: Gemma4MultimodalEmbedder (lines 122-130)
    // ========================================================================
    {
        // embedding_pre_projection_norm
        cur = cur.rmsNorm(ctx, eps);
        cur.setName("mm_norm");

        // 使用带 clamp 的矩阵乘法（匹配 C++ build_mm）
        if (w.mm_input_proj_w) |proj| {
            cur = buildMMWithClamp(ctx, proj, cur, &w.clamp_info_map);
            cur.setName("mm_output");
        }
    }

    // 构建计算图
    builder.gf.buildForwardExpand(cur);

    log.info("Gemma4V graph built successfully", .{});
    return builder.gf;
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 带 clamp 的矩阵乘法
/// 对应 C++ clip_graph_gemma4v::build_mm()
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
