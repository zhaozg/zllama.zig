//! 视觉编码器后处理
//!
//! 提供 ViT 输出后的标准化、投影等后处理功能。
//! 参考: llama.cpp tools/mtmd/models/gemma4v.cpp, gemma4uv.cpp

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.vision_postprocess);

/// 对视觉编码器输出进行标准化
///
/// 应用 std_bias 和 std_scale 对编码器输出进行标准化。
/// 对应 llama.cpp 中的 standardization 步骤。
pub fn standardize(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    std_bias: ?*ggml.Tensor,
    std_scale: ?*ggml.Tensor,
) *ggml.Tensor {
    var result = cur;
    if (std_bias) |sb| {
        result = result.sub(ctx, sb);
    }
    if (std_scale) |ss| {
        result = result.mul(ctx, reshapeForBroadcast(ctx, ss));
    }
    return result;
}

/// 将视觉编码器输出投影到 LLM 嵌入空间
///
/// 应用 mm_input_proj_w 和 mm_soft_emb_norm_w 进行投影。
/// 对应 llama.cpp 中的 multimodal embedder 步骤。
pub fn projectToLLM(
    ctx: *ggml.Context,
    cur: *ggml.Tensor,
    norm_eps: f32,
    mm_soft_emb_norm_w: ?*ggml.Tensor,
    mm_input_proj_w: ?*ggml.Tensor,
) *ggml.Tensor {
    var result = cur;

    // RMSNorm
    result = result.rmsNorm(ctx, norm_eps);
    if (mm_soft_emb_norm_w) |sn| {
        result = result.mul(ctx, reshapeForBroadcast(ctx, sn));
    }

    // 投影到 LLM 嵌入空间
    if (mm_input_proj_w) |proj| {
        // weight-first mulMat: proj [n_output_embd, n_embd] @ cur [n_embd, n_tokens]
        // → [n_output_embd, n_tokens]
        result = proj.mulMat(ctx, result);
    }

    return result;
}

/// Reshape a 1D weight tensor [n] to [n, 1] for broadcasting with [n_embd, n_patches] tensors.
/// Vision encoder uses column-major [n_embd, n_patches] layout.
/// ggml broadcasting: b=[n, 1] vs a=[n_embd, n_patches] -> ne[0]: n==n_embd (ok), ne[1]: 1<=n_patches (ok)
pub fn reshapeForBroadcast(ctx: *ggml.Context, t: *ggml.Tensor) *ggml.Tensor {
    const n = t.ne()[0];
    return ctx.view2d(t, n, 1, ggml.Type.rowSize(t.dataType(), n), 0);
}
