//! 视觉编码器类型定义（兼容层）
//!
//! 此文件为兼容层，提供与 graph/types.zig 兼容的类型定义。
//! 新代码应直接使用 @import("graph") 模块。
//!
//! 参考: src/mtmd/graph/types.zig

const ggml = @import("ggml");

/// ViT 单层权重
pub const ViTLayerWeights = struct {
    ln_1_w: ?*ggml.Tensor = null,
    ln_1_b: ?*ggml.Tensor = null,
    q_w: ?*ggml.Tensor = null,
    k_w: ?*ggml.Tensor = null,
    v_w: ?*ggml.Tensor = null,
    o_w: ?*ggml.Tensor = null,
    o_b: ?*ggml.Tensor = null,
    ln_2_w: ?*ggml.Tensor = null,
    ln_2_b: ?*ggml.Tensor = null,
    ff_up_w: ?*ggml.Tensor = null,
    ff_down_w: ?*ggml.Tensor = null,
};

/// 视觉编码器所有权重
pub const VisionEncoderWeights = struct {
    params: VisionEncoderParams,

    /// Patch embedding
    patch_embeddings_0: ?*ggml.Tensor = null,
    patch_bias: ?*ggml.Tensor = null,

    /// Patch 归一化（Gemma4UV 特有）
    patch_norm_1_w: ?*ggml.Tensor = null,
    patch_norm_1_b: ?*ggml.Tensor = null,
    patch_norm_2_w: ?*ggml.Tensor = null,
    patch_norm_2_b: ?*ggml.Tensor = null,
    patch_norm_3_w: ?*ggml.Tensor = null,
    patch_norm_3_b: ?*ggml.Tensor = null,

    /// 位置编码
    position_embeddings: ?*ggml.Tensor = null,

    /// ViT 层
    layers: []ViTLayerWeights = &.{},

    /// 标准化
    std_bias: ?*ggml.Tensor = null,
    std_scale: ?*ggml.Tensor = null,

    /// 多模态嵌入投影
    mm_input_proj_w: ?*ggml.Tensor = null,
    mm_soft_emb_norm_w: ?*ggml.Tensor = null,
};

// Need VisionEncoderParams for the struct
const config = @import("config.zig");
const VisionEncoderParams = config.VisionEncoderParams;
