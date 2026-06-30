//! 视觉编码器类型定义（兼容层）
//!
//! 此文件为兼容层，提供与 graph/types.zig 兼容的类型定义。
//! 新代码应直接使用 @import("graph") 模块。
//!
//! 参考: src/mtmd/graph/types.zig

const ggml = @import("ggml");
const graph = @import("graph");

// ============================================================================
// 重新导出 graph 模块的核心类型
// ============================================================================

/// ViT 单层权重
pub const ViTLayerWeights = graph.ViTLayerWeights;

/// 视觉编码器所有权重
pub const VisionEncoderWeights = graph.VisionEncoderWeights;

/// 图像 F32 数据
pub const ImageF32 = graph.ImageF32;

/// 图像 U8 数据
pub const ImageU8 = graph.ImageU8;

/// 图像 F32 批次
pub const ImageF32Batch = graph.ImageF32Batch;

/// 构建选项
pub const BuildVitOpts = graph.BuildVitOpts;

/// 投影器类型
pub const ProjectorType = graph.ProjectorType;

/// 归一化类型
pub const NormType = graph.NormType;

/// FFN 激活函数类型
pub const FFNOpType = graph.FFNOpType;

/// Patch merge 类型
pub const PatchMergeType = graph.PatchMergeType;

/// 缩放算法
pub const ResizeAlgo = graph.ResizeAlgo;

/// 填充样式
pub const PadStyle = graph.PadStyle;

/// Flash attention 类型
pub const FlashAttnType = graph.FlashAttnType;

/// 模态类型
pub const Modality = graph.Modality;

/// 裁剪信息
pub const ClampInfo = graph.ClampInfo;
