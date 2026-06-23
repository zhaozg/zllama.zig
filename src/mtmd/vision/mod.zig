//! 视觉编码器模块
//!
//! 提供对 Gemma 4 E2B 内建视觉编码器的支持。
//! 支持可变分辨率输入，使用基于视觉 token 预算的动态扩展方式处理图像。
//!
//! 架构: ViT（Vision Transformer）+ SigLIP 风格
//! - Patch embedding（卷积投影）
//! - 2D 位置编码（X/Y 轴分别编码）
//! - 多层 ViT blocks（RMSNorm + 自注意力 + FFN）
//! - Pooling（平均池化下采样）
//! - 输出投影到 LLM 嵌入空间
//!
//! 支持两种视觉编码器变体:
//! - Gemma4V: 标准 ViT + SigLIP（gemma4v）
//! - Gemma4UV: 统一视觉编码器（gemma4uv, 带额外 patch 归一化）
//!
//! 参考: llama.cpp tools/mtmd/models/gemma4v.cpp, gemma4uv.cpp
//!
//! ⚠️ GGUF 键名说明:
//! mmproj GGUF 文件使用 `clip.vision.*` 前缀的键名（非 `gemma4.vision.*`）。
//! 例如: clip.vision.image_size, clip.vision.patch_size, clip.vision.embedding_length 等。
//! 参考: llama.cpp tools/mtmd/clip.cpp 中 clip_hparams 的加载逻辑。

const std = @import("std");

pub const config = @import("config.zig");
pub const types = @import("types.zig");
pub const loader = @import("loader.zig");
pub const preprocess = @import("preprocess.zig");
pub const postprocess = @import("postprocess.zig");
pub const encoder = @import("encoder.zig");
pub const pipeline = @import("pipeline.zig");

// 重新导出核心类型
pub const VisionEncoderParams = config.VisionEncoderParams;
pub const EncoderType = config.EncoderType;
pub const FfnOp = config.FfnOp;
pub const ViTLayerWeights = types.ViTLayerWeights;
pub const VisionEncoderWeights = types.VisionEncoderWeights;
pub const VisionEncoder = encoder.VisionEncoder;
pub const VisionPipeline = pipeline.VisionPipeline;
pub const NormalizeMode = preprocess.NormalizeMode;
