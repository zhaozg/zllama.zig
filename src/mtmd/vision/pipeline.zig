//! 视觉处理流水线编排器
//!
//! 串联视觉处理的各个阶段：加载 → 预处理 → 编码 → 后处理。
//! 参考: llama.cpp tools/mtmd/clip.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const config = @import("config.zig");
const types = @import("types.zig");
const encoder = @import("encoder.zig");
const preprocess = @import("preprocess");
const postprocess = @import("postprocess.zig");

const VisionEncoder = encoder.VisionEncoder;
const VisionEncoderParams = config.VisionEncoderParams;
const EncoderType = config.EncoderType;

const log = std.log.scoped(.vision_pipeline);

/// 视觉处理流水线
///
/// 提供完整的图像处理流程：从文件加载到视觉嵌入生成。
pub const VisionPipeline = struct {
    encoder: VisionEncoder,

    /// 初始化视觉处理流水线
    pub fn init(
        io: std.Io,
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
        backend: *const encoder.VisionEncoderBackend,
    ) !VisionPipeline {
        const enc = try VisionEncoder.init(io, gguf_file, ctx, allocator, backend, backend.name);
        return VisionPipeline{ .encoder = enc };
    }

    /// 处理图像并返回视觉嵌入
    ///
    /// @param ctx ggml 计算上下文
    /// @param cgraph 计算图
    /// @param image_data RGB 图像数据 [height][width][3]，值范围 [0, 255]
    /// @param img_width 图像宽度
    /// @param img_height 图像高度
    /// @returns 视觉嵌入 [n_output_embd, n_tokens]
    pub fn process(
        self: *const VisionPipeline,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        image_data: []const u8,
        img_width: u32,
        img_height: u32,
    ) !*ggml.Tensor {
        return self.encoder.encode(io, ctx, cgraph, image_data, img_width, img_height, 4);
    }

    /// 返回视觉编码器是否可用
    pub fn isAvailable(self: *const VisionPipeline) bool {
        return self.encoder.isAvailable();
    }

    /// 估算给定分辨率图像的 token 数量
    pub fn estimateOutputTokens(self: *const VisionPipeline, io: std.Io, img_width: u32, img_height: u32) u32 {
        return self.encoder.estimateOutputTokens(io, img_width, img_height);
    }

    /// 计算视觉 token 预算下的最佳图像分辨率
    pub fn bestResolution(self: *const VisionPipeline, max_tokens: u32) struct { width: u32, height: u32 } {
        return self.encoder.bestResolution(max_tokens);
    }

    pub fn deinit(self: *VisionPipeline, allocator: std.mem.Allocator) void {
        self.encoder.deinit(allocator);
    }
};
