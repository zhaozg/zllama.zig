//! Gemma4V 视觉编码器图构建
//!
//! 实现 Gemma 4 Vision 的视觉编码器计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/models/gemma4v.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("../mod.zig");
const vit_builder = @import("../vit.zig");
const patch_builder = @import("../patch.zig");
const rope_builder = @import("../rope.zig");
const weight_loader = @import("weight_loader");

const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const BuildVitOpts = graph.BuildVitOpts;
const VisionHParams = graph.VisionHParams;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const ViTLayerWeights = graph.ViTLayerWeights;
const ImageF32 = graph.ImageF32;

const log = std.log.scoped(.graph_gemma4v);

// ============================================================================
// 视觉编码器后端注册
// ============================================================================

/// Gemma4V 视觉编码器后端
pub const backend = graph.VisionEncoderBackend{
    .name = "gemma4v",
    .buildGraph = buildGraph,
    .loadParams = loadParams,
    .loadWeights = loadWeights,
    .loadClampInfo = loadClampInfo,
    .estimateOutputTokens = estimateOutputTokens,
};

/// 加载 Gemma4V 视觉编码器超参数
pub fn loadParams(io: std.Io, gguf_file: *const gguf.GGUFFile, params: *VisionHParams) void {
    _ = io;
    _ = gguf_file;
    _ = params;
    log.info("Gemma4V loadParams: using clip architecture defaults", .{});
}

/// 从 GGUF 查找或创建张量
fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name);
}

/// 加载单层 ViT 权重
fn loadViTLayer(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
    allocator: std.mem.Allocator,
) !ViTLayerWeights {
    var layer = ViTLayerWeights{};

    // Attention
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.ln1.weight", .{prefix});
        defer allocator.free(name);
        layer.ln_1_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.attn_q.weight", .{prefix});
        defer allocator.free(name);
        layer.q_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.attn_k.weight", .{prefix});
        defer allocator.free(name);
        layer.k_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.attn_v.weight", .{prefix});
        defer allocator.free(name);
        layer.v_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.attn_out.weight", .{prefix});
        defer allocator.free(name);
        layer.o_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }

    // FFN
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.ln2.weight", .{prefix});
        defer allocator.free(name);
        layer.ln_2_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.ffn_up.weight", .{prefix});
        defer allocator.free(name);
        layer.ff_up_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.ffn_down.weight", .{prefix});
        defer allocator.free(name);
        layer.ff_down_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }

    // Gemma4V specific: gate, attn_post_norm, ffn_post_norm, attn_k_norm, attn_q_norm
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.ffn_gate.weight", .{prefix});
        defer allocator.free(name);
        layer.ff_gate_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.attn_post_norm.weight", .{prefix});
        defer allocator.free(name);
        layer.attn_post_norm_w = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.attn_k_norm.weight", .{prefix});
        defer allocator.free(name);
        layer.k_norm = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    {
        const name = try std.fmt.allocPrint(allocator, "{s}.attn_q_norm.weight", .{prefix});
        defer allocator.free(name);
        layer.q_norm = findTensorInGGUF(ctx, gguf_file, name) catch null;
    }
    return layer;
}

/// 加载 Gemma4V 视觉编码器权重
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, gguf_file: *const gguf.GGUFFile, ctx: *ggml.Context, w: *VisionEncoderWeights) anyerror!void {
    _ = io;

    // Patch embedding
    w.patch_embeddings_0 = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight") catch null;
    w.patch_bias = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.bias") catch null;

    // 位置编码
    w.position_embeddings = findTensorInGGUF(ctx, gguf_file, "v.position_embd.weight") catch null;

    // 标准化
    w.std_bias = findTensorInGGUF(ctx, gguf_file, "v.std_bias") catch null;
    w.std_scale = findTensorInGGUF(ctx, gguf_file, "v.std_scale") catch null;

    // 多模态投影
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
        w.layers[il] = try loadViTLayer(ctx, gguf_file, prefix, allocator);
    }

    log.info("Gemma4V weights loaded: {d} ViT layers", .{n_layer});
}

/// 加载 clamp 信息
pub fn loadClampInfo(io: std.Io, allocator: std.mem.Allocator, gguf_file: *const gguf.GGUFFile, w: *VisionEncoderWeights) anyerror!void {
    _ = io;
    _ = allocator;
    _ = gguf_file;
    _ = w;
}

/// 估计输出 token 数
pub fn estimateOutputTokens(io: std.Io, img_width: u32, img_height: u32, patch_size: u32, n_merge: u32) u32 {
    _ = io;
    _ = img_width;
    _ = img_height;
    _ = patch_size;
    _ = n_merge;
    return 0;
}

/// 构建 Gemma4V 完整计算图
///
pub fn buildGraph(
    io: std.Io,
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const VisionHParams,
    image_tensor: *ggml.Tensor,
) anyerror!*ggml.CGraph {
    _ = io;
    const eps = p.eps;
    const n_patches: i64 = @intCast((p.image_size / p.patch_size) * (p.image_size / p.patch_size));

    // ========================================================================
    // 1. Conv2D patch embedding (with scale=2, bias=-1)
    // 参考: gemma4v.cpp: inp_raw = ggml_scale(ctx0, inp_raw, 2.0f); inp_raw = ggml_add(ctx0, inp_raw, -1.0f);
    //                    inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, ...);
    // ========================================================================
    // ========================================================================
    // 1. Scale + bias input (scale=2, bias=-1), then Conv2D patch embedding
    // 参考: gemma4v.cpp: inp_raw = ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f);
    //                    inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, ...);
    // ========================================================================
    var cur = image_tensor.scaleBias(ctx, 2.0, -1.0);
    cur.setName("inp_scaled_biased");

    cur = try patch_builder.buildInp(
        ctx,
        cur,
        w.patch_embeddings_0 orelse return error.MissingPatchEmbedding,
        w.patch_bias,
        @intCast(p.image_size),
        @intCast(p.image_size),
        null, // scale - already applied above
        null, // bias - already applied above
    );
    cur.setName("patch_embed");
    // ========================================================================
    // 2. ViT blocks (via buildVit)
    // 参考: gemma4v.cpp: build_vit(inp, n_patches, NORM_TYPE_RMS, hparams.ffn_op, nullptr, add_pos)
    // ========================================================================
    const vit_opts = BuildVitOpts{
        .v_norm = true,
        .v_norm_eps = eps,
        .kq_scale = 1.0,
    };

    cur = try vit_builder.buildVit(
        ctx,
        cur,
        n_patches,
        .rms_norm,
        p.ffn_op,
        null, // learned_pos_embd - already handled above
        w,
        p,
        null, // add_pos - 2D RoPE not yet implemented in this stub
        vit_opts,
    );
    // ========================================================================
    // 3. Pooling (平均池化下采样)
    // 参考: gemma4v.cpp: cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, ...);
    // ========================================================================
    // TODO: 实现池化层

    // ========================================================================
    // 4. 标准化 (std_bias, std_scale)
    // 参考: gemma4v.cpp: cur = ggml_add(ctx0, cur, model.std_bias); cur = ggml_mul(ctx0, cur, model.std_scale);
    // ========================================================================
    // TODO: 实现标准化

    // ========================================================================
    // 5. 多模态投影 (mm_input_proj_w)
    // 将 ViT 输出从 n_embd 投影到 projection_dim
    // ========================================================================
    if (w.mm_input_proj_w) |proj_w| {
        cur = proj_w.mulMat(ctx, cur);
        cur.setName("mm_proj");
    }

    // 设置输出张量名称并添加到计算图
    cur.setName("mm_output");
    gf.buildForwardExpand(cur);

    return gf;
}
