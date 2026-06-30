//! 视觉编码器模块入口
//!
//! 提供完整的视觉处理流水线，从图像加载到 ViT 编码器推理。
//! 所有公开 API 通过此模块重新导出。
//!
//! 架构:
//! - config.zig: 配置参数管理
//! - types.zig: 阶段间传递的数据结构
//! - loader.zig: 权重加载
//! - preprocess.zig: 图像预处理（归一化等）
//! - encoder.zig: ViT 编码器框架
//! - postprocess.zig: 后处理（标准化、投影等）
//! - pipeline.zig: 流水线编排器
//!
//! 模型特定实现下沉到 src/mtmd/graph/models/ 中，
//! 通过 VisionEncoderBackend 分发。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
pub const config = @import("config.zig");
pub const types = @import("types.zig");
pub const loader = @import("loader.zig");
pub const preprocess = @import("preprocess.zig");
pub const postprocess = @import("postprocess.zig");
pub const encoder = @import("encoder.zig");
pub const pipeline = @import("pipeline.zig");

// ============================================================================
// 便捷类型别名（保持向后兼容）
// ============================================================================

pub const VisionEncoderParams = config.VisionEncoderParams;
pub const EncoderType = config.EncoderType;
pub const FfnOp = config.FfnOp;

pub const VisionEncoder = encoder.VisionEncoder;
pub const VisionPipeline = pipeline.VisionPipeline;
pub const NormalizeMode = preprocess.NormalizeMode;

/// 视觉编码器后端接口（重新导出自 graph 模块）
pub const VisionEncoderBackend = @import("graph").VisionEncoderBackend;

/// 注册的视觉编码器后端列表
/// 新增视觉模型时在此注册
const registered_backends = struct {
    pub const gemma4v = @import("graph").model_graphs.gemma4v.backend;
    pub const gemma4uv = @import("graph").model_graphs.gemma4uv.backend;
    pub const qwen2vl = @import("graph").model_graphs.qwen2vl.backend;
    pub const qwen3vl = @import("graph").model_graphs.qwen3vl.backend;
};

/// 根据模型类型名称查找对应的后端
pub fn getBackend(name: []const u8) ?*const VisionEncoderBackend {
    if (std.mem.eql(u8, name, "gemma4v")) return &registered_backends.gemma4v;
    if (std.mem.eql(u8, name, "gemma4uv")) return &registered_backends.gemma4uv;
    if (std.mem.eql(u8, name, "qwen2vl")) return &registered_backends.qwen2vl;
    if (std.mem.eql(u8, name, "qwen3vl")) return &registered_backends.qwen3vl;
    return null;
}
