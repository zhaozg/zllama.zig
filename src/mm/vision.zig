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
//! - Gemma4UV: 统一视觉编码器（gemma4uv）
//!
//! 参考: llama.cpp tools/mtmd/models/gemma4v.cpp, gemma4uv.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");

const log = std.log.scoped(.vision_encoder);

/// 视觉编码器超参数
pub const VisionEncoderParams = struct {
    /// 输入图像尺寸（正方形，边长）
    image_size: u32 = 896,
    /// Patch 大小
    patch_size: u32 = 14,
    /// 嵌入维度
    n_embd: u32 = 1152,
    /// 注意力头数
    n_head: u32 = 16,
    /// ViT 层数
    n_layer: u32 = 27,
    /// FFN 中间维度
    n_ff: u32 = 4304,
    /// 输出投影维度（匹配 LLM 嵌入维度）
    n_output_embd: u32 = 2560,
    /// Pooling kernel size（每侧合并数）
    n_merge: u32 = 2,
    /// RoPE theta
    rope_theta: f32 = 10000.0,
    /// 归一化 epsilon
    norm_eps: f32 = 1e-6,
};

/// ViT 层权重
pub const ViTLayerWeights = struct {
    // 注意力
    ln_1_w: ?*ggml.Tensor,
    ln_1_b: ?*ggml.Tensor,
    q_w: ?*ggml.Tensor,
    k_w: ?*ggml.Tensor,
    v_w: ?*ggml.Tensor,
    o_w: ?*ggml.Tensor,
    o_b: ?*ggml.Tensor,

    // FFN
    ln_2_w: ?*ggml.Tensor,
    ln_2_b: ?*ggml.Tensor,
    ff_up_w: ?*ggml.Tensor,
    ff_down_w: ?*ggml.Tensor,
};

/// 视觉编码器权重
pub const VisionEncoderWeights = struct {
    params: VisionEncoderParams,

    // Patch embedding
    patch_embeddings_0: ?*ggml.Tensor,
    patch_bias: ?*ggml.Tensor,

    // Patch 归一化（Gemma4UV 特有）
    patch_norm_1_w: ?*ggml.Tensor,
    patch_norm_1_b: ?*ggml.Tensor,
    patch_norm_2_w: ?*ggml.Tensor,
    patch_norm_2_b: ?*ggml.Tensor,
    patch_norm_3_w: ?*ggml.Tensor,
    patch_norm_3_b: ?*ggml.Tensor,

    // 位置编码
    position_embeddings: ?*ggml.Tensor,

    // ViT 层
    layers: []ViTLayerWeights,

    // 标准化
    std_bias: ?*ggml.Tensor,
    std_scale: ?*ggml.Tensor,

    // 多模态嵌入投影
    mm_input_proj_w: ?*ggml.Tensor,
};

/// 视觉编码器类型
pub const EncoderType = enum {
    gemma4v, // 标准 ViT + SigLIP
    gemma4uv, // 统一视觉编码器（带额外 patch 归一化）
};

/// 视觉编码器
pub const VisionEncoder = struct {
    params: VisionEncoderParams,
    weights: VisionEncoderWeights,
    encoder_type: EncoderType,

    /// 初始化视觉编码器（从 GGUF 加载权重）
    pub fn init(gguf_file: *const gguf.GGUFFile) !VisionEncoder {
        var params = VisionEncoderParams{};

        // 从 GGUF 元数据读取参数
        params.image_size = gguf_file.getU32("gemma4.vision.image_size") orelse 896;
        params.patch_size = gguf_file.getU32("gemma4.vision.patch_size") orelse 14;
        params.n_embd = gguf_file.getU32("gemma4.vision.embedding_length") orelse 1152;
        params.n_head = gguf_file.getU32("gemma4.vision.attention_head_count") orelse 16;
        params.n_layer = gguf_file.getU32("gemma4.vision.block_count") orelse 27;
        params.n_ff = gguf_file.getU32("gemma4.vision.feed_forward_length") orelse 4304;
        params.n_merge = gguf_file.getU32("gemma4.vision.projection_dim") orelse 2;
        params.rope_theta = gguf_file.getF32("gemma4.vision.rope_theta") orelse 10000.0;

        // 检测编码器类型
        const enc_type: EncoderType = if (gguf_file.findTensor("patch_norm_1.weight") != null)
            .gemma4uv
        else
            .gemma4v;

        log.info("Vision encoder: type={s}, size={d}, patch={d}, embd={d}, heads={d}, layers={d}", .{
            @tagName(enc_type), params.image_size, params.patch_size,
            params.n_embd, params.n_head, params.n_layer,
        });

        // 注意：完整实现需要从 GGUF 加载所有权重张量。
        // 当前为结构占位，权重加载将在后续版本中实现。
        return VisionEncoder{
            .params = params,
            .weights = .{
                .params = params,
                .patch_embeddings_0 = null,
                .patch_bias = null,
                .patch_norm_1_w = null,
                .patch_norm_1_b = null,
                .patch_norm_2_w = null,
                .patch_norm_2_b = null,
                .patch_norm_3_w = null,
                .patch_norm_3_b = null,
                .position_embeddings = null,
                .layers = &[_]ViTLayerWeights{},
                .std_bias = null,
                .std_scale = null,
                .mm_input_proj_w = null,
            },
            .encoder_type = enc_type,
        };
    }

    /// 编码图像数据，返回视觉嵌入 tokens
    /// @param ctx ggml 上下文
    /// @param graph 计算图
    /// @param image_data RGB 图像数据 [height][width][3]，值范围 [0, 255]
    /// @param img_width 图像宽度
    /// @param img_height 图像高度
    /// @returns 视觉嵌入 [n_output_embd, n_tokens]
    pub fn encode(
        self: *VisionEncoder,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        image_data: []const u8,
        img_width: u32,
        img_height: u32,
    ) !*ggml.Tensor {
        _ = self;
        _ = ctx;
        _ = graph;
        _ = image_data;
        _ = img_width;
        _ = img_height;
        // 完整 ViT 编码器实现将在后续版本中添加
        // 参考: llama.cpp tools/mtmd/models/gemma4v.cpp
        return error.NotImplemented;
    }

    /// 返回视觉编码器是否可用（权重已加载）
    pub fn isAvailable(self: *const VisionEncoder) bool {
        return self.weights.patch_embeddings_0 != null;
    }

    /// 估算给定分辨率图像的 token 数量
    /// 公式: ceil(width/patch_size) * ceil(height/patch_size) / (n_merge^2)
    pub fn estimateOutputTokens(self: *const VisionEncoder, img_width: u32, img_height: u32) u32 {
        const patches_x = (img_width + self.params.patch_size - 1) / self.params.patch_size;
        const patches_y = (img_height + self.params.patch_size - 1) / self.params.patch_size;
        const n_patches = patches_x * patches_y;
        const n_merge = if (self.params.n_merge > 0) self.params.n_merge else 1;
        return n_patches / (n_merge * n_merge);
    }

    /// 计算视觉 token 预算下的最佳图像分辨率
    /// 确保 token 数量不超过预算，同时最大化分辨率
    pub fn bestResolution(self: *const VisionEncoder, max_tokens: u32) struct { width: u32, height: u32 } {
        const n_merge = if (self.params.n_merge > 0) self.params.n_merge else 1;
        const max_patches = max_tokens * n_merge * n_merge;
        const side_patches = @max(1, @as(u32, @intFromFloat(@sqrt(@as(f64, @floatFromInt(max_patches))))));
        const side = side_patches * self.params.patch_size;

        return .{ .width = side, .height = side };
    }
};
