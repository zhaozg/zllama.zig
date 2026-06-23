//! 视觉编码器权重加载器
//!
//! 从 GGUF 文件加载视觉编码器权重。
//! 参考: llama.cpp tools/mtmd/clip.cpp, gemma4v.cpp, gemma4uv.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const weight_loader = @import("weight_loader");
const types = @import("types.zig");
const config = @import("config.zig");

const ViTLayerWeights = types.ViTLayerWeights;
const VisionEncoderWeights = types.VisionEncoderWeights;
const VisionEncoderParams = config.VisionEncoderParams;
const EncoderType = config.EncoderType;

const log = std.log.scoped(.vision_loader);

/// 从 GGUF 文件加载视觉编码器所有权重
pub fn loadWeights(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    params: VisionEncoderParams,
    enc_type: EncoderType,
    allocator: std.mem.Allocator,
) !VisionEncoderWeights {
    _ = enc_type;
    // Patch embedding (v.patch_embd.*)
    const patch_embd = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight") catch null;
    const patch_bias = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.bias") catch null;

    // Patch 归一化（Gemma4UV）
    const pn1_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.1.weight") catch null;
    const pn1_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.1.bias") catch null;
    const pn2_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.2.weight") catch null;
    const pn2_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.2.bias") catch null;
    const pn3_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.3.weight") catch null;
    const pn3_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.3.bias") catch null;

    // 位置编码 (v.position_embd.weight)
    const pos_embd = findTensorInGGUF(ctx, gguf_file, "v.position_embd.weight") catch null;

    // 标准化 (v.std_bias, v.std_scale)
    const std_bias = findTensorInGGUF(ctx, gguf_file, "v.std_bias") catch null;
    const std_scale = findTensorInGGUF(ctx, gguf_file, "v.std_scale") catch null;

    // 多模态投影 (mm.*)
    const mm_proj = findTensorInGGUF(ctx, gguf_file, "mm.input_projection.weight") catch null;
    const mm_soft = findTensorInGGUF(ctx, gguf_file, "mm.soft_emb_norm.weight") catch null;

    // 加载 ViT 层
    const n_layer: usize = @intCast(params.n_layer);
    var layers = try allocator.alloc(ViTLayerWeights, n_layer);
    for (0..n_layer) |il| {
        const prefix = try std.fmt.allocPrint(allocator, "v.blk.{d}", .{il});
        layers[il] = loadViTLayer(ctx, gguf_file, prefix) catch |err| {
            log.err("Failed to load ViT layer {d}: {}", .{ il, err });
            allocator.free(prefix);
            return err;
        };
        allocator.free(prefix);
    }

    log.info("Vision encoder weights loaded: {d} ViT layers", .{n_layer});

    return VisionEncoderWeights{
        .params = params,
        .patch_embeddings_0 = patch_embd,
        .patch_bias = patch_bias,
        .patch_norm_1_w = pn1_w,
        .patch_norm_1_b = pn1_b,
        .patch_norm_2_w = pn2_w,
        .patch_norm_2_b = pn2_b,
        .patch_norm_3_w = pn3_w,
        .patch_norm_3_b = pn3_b,
        .position_embeddings = pos_embd,
        .layers = layers,
        .std_bias = std_bias,
        .std_scale = std_scale,
        .mm_input_proj_w = mm_proj,
        .mm_soft_emb_norm_w = mm_soft,
    };
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
) !ViTLayerWeights {
    var layer = ViTLayerWeights{};

    // Attention
    layer.ln_1_w = findLayerWeight(ctx, gguf_file, prefix, "ln1.weight") catch null;
    layer.ln_1_b = findLayerWeight(ctx, gguf_file, prefix, "ln1.bias") catch null;
    layer.q_w = findLayerWeight(ctx, gguf_file, prefix, "attn_q.weight") catch null;
    layer.k_w = findLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch null;
    layer.v_w = findLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch null;
    layer.o_w = findLayerWeight(ctx, gguf_file, prefix, "attn_out.weight") catch null;
    layer.o_b = findLayerWeight(ctx, gguf_file, prefix, "attn_out.bias") catch null;

    // FFN
    layer.ln_2_w = findLayerWeight(ctx, gguf_file, prefix, "ln2.weight") catch null;
    layer.ln_2_b = findLayerWeight(ctx, gguf_file, prefix, "ln2.bias") catch null;
    layer.ff_up_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch null;
    layer.ff_down_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch null;

    return layer;
}

/// 查找层权重
fn findLayerWeight(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
    name: []const u8,
) !*ggml.Tensor {
    return weight_loader.loadLayerWeight(ctx, gguf_file, prefix, name);
}
