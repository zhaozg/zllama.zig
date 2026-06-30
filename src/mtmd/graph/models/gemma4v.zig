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
// 视觉编码器后端注册
// ============================================================================

/// Gemma4V 视觉编码器后端实例
pub const backend = graph.VisionEncoderBackend{
    .name = "gemma4v",
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
    var clamp_map = std.StringHashMap(ClampInfo).init(allocator);
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

    const weight_suffix = ".weight";
    const clamp_suffixes = [_][]const u8{ ".input_max", ".input_min", ".output_max", ".output_min" };

    for (weight_names.items) |w_name| {
        if (!std.mem.endsWith(u8, w_name, weight_suffix)) continue;

        const prefix_len = w_name.len - weight_suffix.len;
        var clamp_names: [4][]const u8 = undefined;

        for (&clamp_names, clamp_suffixes) |*out_name, suffix| {
            const new_len = prefix_len + suffix.len;
            const buf = try allocator.alloc(u8, new_len);
            errdefer allocator.free(buf);
            @memcpy(buf[0..prefix_len], w_name[0..prefix_len]);
            @memcpy(buf[prefix_len..][0..suffix.len], suffix);
            out_name.* = buf;
        }
        defer {
            for (clamp_names) |n| allocator.free(n);
        }

        const inp_max_val = readScalarOrDefault(gguf_file, clamp_names[0], std.math.floatMax(f32));
        const inp_min_val = readScalarOrDefault(gguf_file, clamp_names[1], -std.math.floatMax(f32));
        const out_max_val = readScalarOrDefault(gguf_file, clamp_names[2], std.math.floatMax(f32));
        const out_min_val = readScalarOrDefault(gguf_file, clamp_names[3], -std.math.floatMax(f32));

        try clamp_map.put(w_name, ClampInfo{
            .inp_min = inp_min_val,
            .inp_max = inp_max_val,
            .out_min = out_min_val,
            .out_max = out_max_val,
        });
    }

    w.clamp_info_map = clamp_map;
    log.info("Gemma4V clamp info loaded: {d} entries", .{clamp_map.count()});
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

    // 1. 创建输入张量
    // 输入图像: [3, height, width] f32, 值范围 [0, 1]
    const inp_raw = try ctx.newTensor3d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), 3);
    inp_raw.setName("inp_raw");
    // 填充输入数据（由调用者负责）
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

    // 2. Scale + bias: patches * 2 - 1
    var cur = inp_raw;
    cur = cur.scale(ctx, 2.0);
    cur.setName("inp_scaled");
    {
        const bias_t = try ctx.newTensor1d(ggml.Type.f32, 1);
        bias_t.dataF32()[0] = -1.0;
        bias_t.setName("inp_bias");
        cur = cur.add(ctx, bias_t);
        cur.setName("inp_biased");
    }

    // 3. Conv2D patch embedding
    if (w.patch_embeddings_0) |pe| {
        const kw: i32 = @intCast(pe.ne()[0]);
        const kh: i32 = @intCast(pe.ne()[1]);
        cur = cur.conv2d(ctx, pe, kw, kh, 0, 0, 1, 1);
        cur.setName("inp_conv");

        // Reshape to [n_embd, n_patches]
        cur = cur.reshape3d(ctx, n_patches, n_embd, 1);
        cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
        cur.setName("inp_patches");
    }

    // 4. 2D 位置编码
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
    }

    // 5. ViT blocks
    var inpL = cur.reshape2d(ctx, n_embd, n_patches * n_batch);
    inpL.setName("vit_input");

    // 创建 2D RoPE 位置索引
    const vit_indices = try graph.createPositionIndices(ctx, n_patches, n_patches_x);
    const d_head_half = @divExact(d_head, @as(i64, 2));

    for (w.layers, 0..) |*layer, il| {
        var layer_buf: [32]u8 = undefined;
        const layer_name = try std.fmt.bufPrintZ(&layer_buf, "blk.{d}", .{il});

        var residual = inpL;

        // --- Pre-attention RMSNorm ---
        var attn_in = inpL;
        if (layer.ln_1_w) |ln1_w| {
            attn_in = try graph.buildNorm(ctx, attn_in, ln1_w, layer.ln_1_b, .rms_norm, eps, layer_name);
        }

        // --- Self-attention with 2D RoPE ---
        {
            // QKV projections (with clamp, matching C++ clip_graph_gemma4v::build_mm)
            var Qcur = if (layer.q_w) |qw| buildMMWithClamp(ctx, qw, attn_in, &w.clamp_info_map) else return error.MissingQWeight;
            Qcur.setName(layer_name);
            var Kcur = if (layer.k_w) |kw| buildMMWithClamp(ctx, kw, attn_in, &w.clamp_info_map) else return error.MissingKWeight;
            Kcur.setName(layer_name);
            var Vcur = if (layer.v_w) |vw| buildMMWithClamp(ctx, vw, attn_in, &w.clamp_info_map) else return error.MissingVWeight;
            Vcur.setName(layer_name);

            // Reshape to [d_head, n_head, n_patches, n_batch]
            Qcur = Qcur.reshape4d(ctx, d_head, n_head, n_patches, n_batch);
            Qcur.setName(layer_name);
            Kcur = Kcur.reshape4d(ctx, d_head, n_head, n_patches, n_batch);
            Kcur.setName(layer_name);
            Vcur = Vcur.reshape4d(ctx, d_head, n_head, n_patches, n_batch);
            Vcur.setName(layer_name);

            // 2D RoPE: first half uses pos_x, second half uses pos_y
            {
                // First half
                const first_q = Qcur.view4d(ctx, d_head_half, n_head, n_patches, n_batch, Qcur.nb()[1], Qcur.nb()[2], Qcur.nb()[3], 0);
                const first_k = Kcur.view4d(ctx, d_head_half, n_head, n_patches, n_batch, Kcur.nb()[1], Kcur.nb()[2], Kcur.nb()[3], 0);
                const rope_first_q = first_q.ropeExt(ctx, vit_indices.pos_x, null, @intCast(d_head_half), 2, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                rope_first_q.setName(layer_name);
                const rope_first_k = first_k.ropeExt(ctx, vit_indices.pos_x, null, @intCast(d_head_half), 2, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                rope_first_k.setName(layer_name);

                // Second half
                const offset: usize = @intCast(d_head_half * @sizeOf(f32));
                const second_q = Qcur.view4d(ctx, d_head_half, n_head, n_patches, n_batch, Qcur.nb()[1], Qcur.nb()[2], Qcur.nb()[3], offset);
                const second_k = Kcur.view4d(ctx, d_head_half, n_head, n_patches, n_batch, Kcur.nb()[1], Kcur.nb()[2], Kcur.nb()[3], offset);
                const rope_second_q = second_q.ropeExt(ctx, vit_indices.pos_y, null, @intCast(d_head_half), 2, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                rope_second_q.setName(layer_name);
                const rope_second_k = second_k.ropeExt(ctx, vit_indices.pos_y, null, @intCast(d_head_half), 2, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                rope_second_k.setName(layer_name);

                Qcur = rope_first_q.concat(ctx, rope_second_q, 0);
                Qcur.setName(layer_name);
                Kcur = rope_first_k.concat(ctx, rope_second_k, 0);
                Kcur.setName(layer_name);
            }

            // Vcur RMSNorm (gemma4v-specific)
            Vcur = Vcur.rmsNorm(ctx, eps);
            Vcur.setName(layer_name);

            // Attention
            const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
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
                layer_name,
                layer.attn_sinks,
            );
            attn_out.setName(layer_name);

            // Residual
            residual = residual.add(ctx, attn_out);
            residual.setName(layer_name);
        }

        // --- Pre-FFN RMSNorm ---
        var ffn_in = residual;
        if (layer.ln_2_w) |ln2_w| {
            ffn_in = try graph.buildNorm(ctx, ffn_in, ln2_w, layer.ln_2_b, .rms_norm, eps, layer_name);
        }

        // --- FFN (with clamp, matching C++ clip_graph_gemma4v::build_mm) ---
        {
            // Up projection
            var up_result = buildMMWithClamp(ctx, layer.ff_up_w orelse return error.MissingFFNUpWeight, ffn_in, &w.clamp_info_map);
            if (layer.ff_up_b) |b| {
                up_result = up_result.add(ctx, b);
            }

            // Gate projection (optional)
            var gate_result: ?*ggml.Tensor = null;
            if (layer.ff_gate_w) |g| {
                gate_result = buildMMWithClamp(ctx, g, ffn_in, &w.clamp_info_map);
                if (layer.ff_gate_b) |gb| {
                    gate_result = gate_result.?.add(ctx, gb);
                }
            }

            // SiLU activation
            var activated = up_result.silu(ctx);

            // Gate (element-wise multiply)
            if (gate_result) |g| {
                activated = activated.mul(ctx, g);
            }

            // Down projection
            var ffn_out = buildMMWithClamp(ctx, layer.ff_down_w orelse return error.MissingFFNDownWeight, activated, &w.clamp_info_map);
            if (layer.ff_down_b) |b| {
                ffn_out = ffn_out.add(ctx, b);
            }

            ffn_out.setName(layer_name);

            inpL = residual.add(ctx, ffn_out);
            inpL.setName(layer_name);
        }
    }

    // 6. Pooling (平均池化下采样)
    const kernel_size: i64 = @intCast(p.n_merge);
    var pooled = inpL;
    {
        // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, 1]
        pooled = pooled.permute(ctx, 1, 0, 2, 3).cont(ctx);
        pooled.setName("pool_permuted");
        pooled = pooled.cont4d(ctx, n_patches_x, n_patches_y, n_embd, 1);
        pooled.setName("pool_4d");

        // Average pooling
        pooled = pooled.pool2d(
            ctx,
            1,
            @as(i32, @intCast(kernel_size)),
            @as(i32, @intCast(kernel_size)),
            @as(i32, @intCast(kernel_size)),
            @as(i32, @intCast(kernel_size)),
            0,
            0,
        );
        pooled.setName("pool_avg");

        const out_x = @divExact(n_patches_x, kernel_size);
        const out_y = @divExact(n_patches_y, kernel_size);
        pooled = pooled.reshape3d(ctx, out_x * out_y, n_embd, 1);
        pooled.setName("pool_reshaped");
        pooled = pooled.permute(ctx, 1, 0, 2, 3).cont(ctx);
        pooled.setName("pool_result");

        // Scale by sqrt(n_embd)
        pooled = pooled.scale(ctx, @sqrt(@as(f32, @floatFromInt(n_embd))));
        pooled.setName("pool_scaled");
    }

    // 7. 标准化 (std_bias, std_scale)
    var result = pooled;
    if (w.std_bias) |sb| {
        result = result.sub(ctx, sb);
        result.setName("std_sub");
    }
    if (w.std_scale) |ss| {
        result = result.mul(ctx, graph.reshapeForBroadcast(ctx, ss));
        result.setName("std_mul");
    }

    // 8. 投影到 LLM 嵌入空间 (with clamp, matching C++ clip_graph_gemma4v::build_mm)
    // Gemma4MultimodalEmbedder: RMSNorm + linear projection with clamp
    {
        result = result.rmsNorm(ctx, eps);
        result.setName("mm_norm");
        if (w.mm_soft_emb_norm_w) |sn| {
            result = result.mul(ctx, graph.reshapeForBroadcast(ctx, sn));
            result.setName("mm_norm_scaled");
        }
        if (w.mm_input_proj_w) |proj| {
            result = buildMMWithClamp(ctx, proj, result, &w.clamp_info_map);
            result.setName("mm_output");
        }
    }

    // 构建计算图
    builder.gf.buildForwardExpand(result);

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

/// 从 GGUF 读取标量值，如果不存在则返回默认值
fn readScalarOrDefault(gguf_file: *const gguf.GGUFFile, name: []const u8, default_val: f32) f32 {
    const tensor_info = gguf_file.findTensor(name) orelse return default_val;
    const data = gguf_file.getTensorData(tensor_info);
    if (data.len < 4) return default_val;
    return @as(*const f32, @ptrCast(@alignCast(data.ptr))).*;
}
