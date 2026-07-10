//! 音频嵌入后处理
//!
//! 提供 softcapping 等后处理功能。
//! 匹配 llama.cpp gemma4a 的后处理逻辑。
//!
//! 参考: llama.cpp tools/mtmd/models/gemma4a.cpp

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.audio_postprocess);

// ============================================================================
// 公开 API
// ============================================================================

/// 对注意力分数应用 logit softcapping
/// softcap(x) = tanh(x / cap) * cap
pub fn applySoftcap(ctx: *ggml.Context, scores: *ggml.Tensor, cap: f32) *ggml.Tensor {
    return scores
        .scale(ctx, 1.0 / cap)
        .tanh(ctx)
        .scale(ctx, cap);
}

/// 将 Mel 频谱数据转换为 F32 4D 张量 [n_frames, n_mel_bins, 1, 1]
///
/// 使用 4D 张量与 clip.cpp build_inp_raw() 的约定一致（后者创建 [W, H, C, B] 4D 张量）。
/// 音频编码器（如 Gemma4A）直接使用此 4D 张量，无需额外 reshape。
///
/// 参考: clip.cpp clip_graph::build_inp_raw() → ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, nx, ny, C, B)
pub fn melToTensor(
    ctx: *ggml.Context,
    mel_data: []const f32,
    n_frames: u32,
    n_mel_bins: u32,
) !*ggml.Tensor {
    const tensor = try ctx.newTensor4d(
        ggml.Type.f32,
        @intCast(n_frames),
        @intCast(n_mel_bins),
        1,
        1,
    );
    tensor.setName("mel_input");
    ggml.setInput(tensor);

    // In no_alloc mode, the tensor data pointer is NULL.
    // We need to allocate the data manually so we can write to it.
    // This mirrors the approach used in vision/preprocess.zig:normalizeToTensor.
    const no_alloc = ctx.getNoAlloc();
    if (no_alloc) {
        const data_size = @as(usize, @intCast(tensor.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
        @memset(buf, 0);
        tensor.setDataPtr(buf);
    }

    try tensor.dataSet(f32, mel_data);

    return tensor;
}

/// 计算余弦相似度（用于调试和验证）
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0.0;
    var dot: f64 = 0;
    var norm_a: f64 = 0;
    var norm_b: f64 = 0;
    for (a, b) |va, vb| {
        const fa: f64 = @floatCast(va);
        const fb: f64 = @floatCast(vb);
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }
    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom < 1e-10) return 0.0;
    return @as(f32, @floatCast(dot / denom));
}
