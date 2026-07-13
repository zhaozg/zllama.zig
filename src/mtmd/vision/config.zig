//! 视觉编码器配置参数
//!
//! 定义视觉编码器的超参数和配置结构。
//! 参考: llama.cpp tools/mtmd/models/gemma4v.cpp, gemma4uv.cpp
//!
//! ⚠️ GGUF 键名说明:
//! mmproj GGUF 文件使用 `clip.vision.*` 前缀的键名（非 `gemma4.vision.*`）。
//! 例如: clip.vision.image_size, clip.vision.patch_size, clip.vision.embedding_length 等。
//! 参考: llama.cpp tools/mtmd/clip.cpp 中 clip_hparams 的加载逻辑。

const std = @import("std");

/// FFN 激活函数类型
pub const FfnOp = enum {
    silu,
    gelu,
};

/// 视觉编码器类型
pub const EncoderType = enum {
    /// 标准 ViT + SigLIP
    gemma4v,
    /// 统一视觉编码器（带额外 patch 归一化）
    gemma4uv,
};

/// 视觉编码器超参数
///
/// 所有参数均从 GGUF 元数据读取（mmproj 文件），
/// 使用 `clip.vision.*` 或 `gemma4.vision.*` 前缀。
pub const VisionEncoderParams = struct {
    /// 输入图像尺寸（正方形，边长）
    /// 来自 GGUF: clip.vision.image_size
    /// Gemma 4 E2B: 224
    image_size: u32 = 224,

    /// Patch 大小
    /// 来自 GGUF: clip.vision.patch_size
    /// Gemma 4 E2B: 16
    patch_size: u32 = 16,

    /// 嵌入维度
    /// 来自 GGUF: clip.vision.embedding_length
    /// Gemma 4 E2B: 768
    n_embd: u32 = 768,

    /// 注意力头数
    /// 来自 GGUF: clip.vision.attention.head_count
    /// Gemma 4 E2B: 12
    n_head: u32 = 12,

    /// ViT 层数
    /// 来自 GGUF: clip.vision.block_count
    /// Gemma 4 E2B: 16
    n_layer: u32 = 16,

    /// FFN 中间维度
    /// 来自 GGUF: clip.vision.feed_forward_length
    /// Gemma 4 E2B: 3072
    n_ff: u32 = 3072,

    /// 输出投影维度（匹配 LLM 嵌入维度）
    /// 来自 GGUF: clip.vision.projection_dim
    /// Gemma 4 E2B: 1536 (匹配 LLM n_embd)
    n_output_embd: u32 = 1536,

    /// Pooling kernel size（每侧合并数）
    /// 来自 GGUF: clip.vision.projector.scale_factor 或 gemma4.vision.projector.scale_factor
    /// 参考 llama.cpp: GEMMA4V/GEMMA4UV 默认值为 3，GGUF 中的实际值覆盖此默认值
    n_merge: u32 = 3,
    /// RoPE theta
    rope_theta: f32 = 10000.0,

    /// 归一化 epsilon
    norm_eps: f32 = 1e-6,

    /// FFN activation
    ffn_op: FfnOp = .silu,

    /// 动态分辨率：最小像素数（用于 Qwen3VL 等支持动态尺寸的模型）
    /// 来自 GGUF: clip.vision.image_min_pixels
    /// 参考: llama.cpp mtmd-image.cpp calc_size_preserved_ratio()
    image_min_pixels: u32 = 0,

    /// 动态分辨率：最大像素数（用于 Qwen3VL 等支持动态尺寸的模型）
    /// 来自 GGUF: clip.vision.image_max_pixels
    /// 参考: llama.cpp mtmd-image.cpp calc_size_preserved_ratio()
    image_max_pixels: u32 = 0,

    /// 用户指定的最大像素数（覆盖 GGUF 默认值）
    /// 用于控制内存使用，0 表示使用 GGUF 默认值
    user_max_pixels: u32 = 0,

    /// 视觉编码器类型（如 "gemma4v", "gemma4uv", "qwen3vl" 等）
    /// 由 detectFromGGUF 设置，用于运行时识别后端类型
    projector_type: []const u8 = "",
};
