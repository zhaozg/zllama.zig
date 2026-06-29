//! 多模态投影器构建器
//!
//! 提供多模态投影器（MM Projector）的 ggml 计算图构建。
//! 将视觉/音频编码器输出投影到 LLM 嵌入空间。
//!
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_mm()

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");
const norm_builder = @import("norm.zig");

const ProjectorType = types.ProjectorType;
const VisionEncoderWeights = types.VisionEncoderWeights;
const ClampInfo = types.ClampInfo;

const log = std.log.scoped(.graph_mm);

/// 矩阵乘法封装
///
/// 支持钩子（如 LoRA、clamping），具体行为取决于模型类型。
///
/// 参数:
///   - ctx: ggml 上下文
///   - w: 权重矩阵
///   - x: 输入矩阵
///   - clamp_info: 裁剪信息（可选，用于 Gemma4）
///
/// 返回: 矩阵乘法结果
///
/// 参考: clip-graph.h build_mm()
pub fn buildMM(
    ctx: *ggml.Context,
    w: *ggml.Tensor,
    x: *ggml.Tensor,
    clamp_info: ?*const ClampInfo,
) !*ggml.Tensor {
    var result = w.mulMat(ctx, x);

    if (clamp_info) |ci| {
        // Gemma4ClippableLinear: clamp input and output
        result = result.clamp(ctx, ci.inp_min, ci.inp_max);
        result.setName("mm_clamped");
    }

    return result;
}

/// 构建 Gemma3 风格的多模态投影器
///
/// 处理流程:
///   1. RMSNorm (mm_soft_emb_norm_w)
///   2. 线性投影 (mm_input_proj_w)
///
/// 参数:
///   - ctx: ggml 上下文
///   - cur: 输入张量 [n_embd, n_tokens]
///   - weights: 视觉编码器权重
///   - eps: 归一化 epsilon
///
/// 返回: 投影后的张量 [n_output_embd, n_tokens]
pub fn buildGemma3Projector(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    weights: *const VisionEncoderWeights,
    eps: f32,
) !*ggml.Tensor {
    var result = cur;

    // RMSNorm
    result = result.rmsNorm(ctx, eps);
    result.setName("mm_norm");
    if (weights.mm_soft_emb_norm_w) |sn| {
        result = result.mul(ctx, norm_builder.reshapeForBroadcast(ctx, sn));
        result.setName("mm_norm_scaled");
    }

    // Linear projection
    if (weights.mm_input_proj_w) |proj| {
        result = proj.mulMat(ctx, result);
        result.setName("mm_proj");
    }

    return result;
}

/// 构建 MLP 风格的多模态投影器
///
/// 处理流程:
///   1. Up projection + GELU
///   2. Down projection
///
/// 参数:
///   - ctx: ggml 上下文
///   - cur: 输入张量 [n_embd, n_tokens]
///   - fc_w: 第一层权重 [n_hidden, n_embd]
///   - fc_b: 第一层偏置 [n_hidden]（可选）
///   - down_w: 第二层权重 [n_output, n_hidden]
///   - down_b: 第二层偏置 [n_output]（可选）
///
/// 返回: 投影后的张量 [n_output, n_tokens]
pub fn buildMLPProjector(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    fc_w: *ggml.Tensor,
    fc_b: ?*ggml.Tensor,
    down_w: *ggml.Tensor,
    down_b: ?*ggml.Tensor,
) !*ggml.Tensor {
    // Up projection
    var hidden = fc_w.mulMat(ctx, cur);
    hidden.setName("mm_fc");
    if (fc_b) |b| {
        hidden = hidden.add(ctx, b);
        hidden.setName("mm_fc_biased");
    }

    // GELU activation
    hidden = hidden.gelu(ctx);
    hidden.setName("mm_act");

    // Down projection
    var result = down_w.mulMat(ctx, hidden);
    result.setName("mm_down");
    if (down_b) |b| {
        result = result.add(ctx, b);
        result.setName("mm_down_biased");
    }

    return result;
}

/// 构建标准化 + 投影
///
/// 处理流程:
///   1. 标准化 (std_bias, std_scale)
///   2. 投影到 LLM 嵌入空间
///
/// 参数:
///   - ctx: ggml 上下文
///   - cur: 输入张量 [n_embd, n_tokens]
///   - weights: 视觉编码器权重
///   - eps: 归一化 epsilon
///
/// 返回: 投影后的张量 [n_output_embd, n_tokens]
pub fn buildStandardizeAndProject(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    weights: *const VisionEncoderWeights,
    eps: f32,
) !*ggml.Tensor {
    var result = cur;

    // 1. Standardize
    if (weights.std_bias) |sb| {
        result = result.sub(ctx, sb);
        result.setName("mm_std_sub");
    }
    if (weights.std_scale) |ss| {
        result = result.mul(ctx, norm_builder.reshapeForBroadcast(ctx, ss));
        result.setName("mm_std_mul");
    }

    // 2. Project to LLM embedding space
    result = try buildGemma3Projector(ctx, result, weights, eps);

    return result;
}

test "buildMM: basic matrix multiply" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 1024 * 1024 });
    defer ctx.deinit();

    const n_in: i64 = 64;
    const n_out: i64 = 128;
    const n_tokens: i64 = 16;

    const w = try ctx.newTensor2d(ggml.Type.f32, n_out, n_in);
    const x = try ctx.newTensor2d(ggml.Type.f32, n_in, n_tokens);

    @memset(w.dataF32(), 0.1);
    @memset(x.dataF32(), 0.5);

    const result = try buildMM(&ctx, w, x, null);
    try testing.expectEqual(n_out, result.ne()[0]);
    try testing.expectEqual(n_tokens, result.ne()[1]);
}
