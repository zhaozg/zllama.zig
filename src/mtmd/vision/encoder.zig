//! 视觉编码器框架
//!
//! 提供视觉编码器的通用框架，包括：
//! - VisionEncoder 结构体（生命周期管理、编码接口）
//! - 模型无关的权重容器（使用 graph.VisionEncoderWeights）
//!
//! 模型特定的实现（权重加载、图构建、token 估算）下沉到
//! src/mtmd/graph/models/ 中，通过 VisionEncoderBackend 分发。
//!
//! 设计原则：
//! - encoder.zig 只包含框架、流程、架构代码
//! - 具体模型的张量名称、图结构、参数硬编码都在 graph/models/ 中

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const config = @import("config.zig");
const types = @import("types.zig");
const loader = @import("loader.zig");
const preprocess = @import("preprocess.zig");
const postprocess = @import("postprocess.zig");
const graph = @import("graph");

const VisionEncoderParams = config.VisionEncoderParams;
const EncoderType = config.EncoderType;
const VisionEncoderWeights = types.VisionEncoderWeights;
const NormalizeMode = preprocess.NormalizeMode;

const log = std.log.scoped(.vision_encoder);

/// 视觉编码器后端接口（重新导出自 graph 模块）
pub const VisionEncoderBackend = graph.VisionEncoderBackend;

// ============================================================================
// 视觉编码器
// ============================================================================

/// 通用视觉编码器
/// 通过 backend 字段分发到具体模型实现
pub const VisionEncoder = struct {
    params: VisionEncoderParams,
    weights: VisionEncoderWeights,
    ctx_weights: *ggml.Context,
    backend: *const VisionEncoderBackend,

    /// 图像归一化参数（来自 GGUF clip.vision.image_mean / image_std）
    image_mean: [3]f32 = .{ 0.0, 0.0, 0.0 },
    image_std: [3]f32 = .{ 1.0, 1.0, 1.0 },

    /// 从 GGUF 文件初始化视觉编码器
    ///
    /// 注意: mmproj GGUF 使用 `clip.vision.*` 前缀的键名（非 `gemma4.vision.*`）
    /// 参考: llama.cpp tools/mtmd/clip.cpp 中 clip_hparams 的加载逻辑
    pub fn init(
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
        backend: *const VisionEncoderBackend,
    ) !VisionEncoder {
        var params = VisionEncoderParams{};

        // 从 GGUF 元数据读取参数
        // mmproj GGUF 使用 clip.vision.* 前缀
        // 同时也尝试 gemma4.vision.* 前缀以兼容不同格式
        if (gguf_file.getU32("clip.vision.image_size")) |v| params.image_size = v else if (gguf_file.getU32("gemma4.vision.image_size")) |v| params.image_size = v;

        if (gguf_file.getU32("clip.vision.patch_size")) |v| params.patch_size = v else if (gguf_file.getU32("gemma4.vision.patch_size")) |v| params.patch_size = v;

        if (gguf_file.getU32("clip.vision.embedding_length")) |v| params.n_embd = v else if (gguf_file.getU32("gemma4.vision.embedding_length")) |v| params.n_embd = v;

        if (gguf_file.getU32("clip.vision.attention.head_count")) |v| params.n_head = v else if (gguf_file.getU32("gemma4.vision.attention_head_count")) |v| params.n_head = v;

        if (gguf_file.getU32("clip.vision.block_count")) |v| params.n_layer = v else if (gguf_file.getU32("gemma4.vision.block_count")) |v| params.n_layer = v;

        if (gguf_file.getU32("clip.vision.feed_forward_length")) |v| params.n_ff = v else if (gguf_file.getU32("gemma4.vision.feed_forward_length")) |v| params.n_ff = v;

        // projection_dim 是输出投影维度（匹配 LLM n_embd）
        if (gguf_file.getU32("clip.vision.projection_dim")) |v| params.n_output_embd = v else if (gguf_file.getU32("gemma4.vision.projection_dim")) |v| params.n_output_embd = v;

        if (gguf_file.getF32("clip.vision.attention.layer_norm_epsilon")) |v| params.norm_eps = v;
        if (gguf_file.getF32("clip.vision.rope_theta")) |v| params.rope_theta = v;

        // 读取图像归一化参数
        var image_mean: [3]f32 = .{ 0.0, 0.0, 0.0 };
        var image_std: [3]f32 = .{ 1.0, 1.0, 1.0 };
        if (gguf_file.getF32Array("clip.vision.image_mean", 3)) |mean| {
            for (mean, 0..) |v, i| image_mean[i] = v;
        }
        if (gguf_file.getF32Array("clip.vision.image_std", 3)) |std_val| {
            for (std_val, 0..) |v, i| image_std[i] = v;
        }

        // 让 backend 加载模型特定参数
        // 构建 VisionHParams 用于 backend 调用
        var hparams = graph.VisionHParams{
            .image_size = params.image_size,
            .patch_size = params.patch_size,
            .n_embd = params.n_embd,
            .n_head = params.n_head,
            .n_layer = params.n_layer,
            .n_ff = params.n_ff,
            .projection_dim = params.n_output_embd,
            .n_merge = params.n_merge,
            .eps = params.norm_eps,
            .rope_theta = params.rope_theta,
        };
        backend.loadParams(gguf_file, &hparams);

        log.info("Loading vision encoder: backend={s}, size={d}, patch={d}, embd={d}, heads={d}, layers={d}, output_embd={d}", .{
            backend.name, params.image_size, params.patch_size,
            params.n_embd, params.n_head, params.n_layer,
            params.n_output_embd,
        });
        log.info("  image_mean=[{d:.4},{d:.4},{d:.4}] image_std=[{d:.4},{d:.4},{d:.4}]", .{
            image_mean[0], image_mean[1], image_mean[2],
            image_std[0], image_std[1], image_std[2],
        });

        // 加载所有权重
        var weights = VisionEncoderWeights{};
        try backend.loadWeights(allocator, gguf_file, ctx, &weights);

        return VisionEncoder{
            .params = params,
            .weights = weights,
            .ctx_weights = ctx,
            .backend = backend,
            .image_mean = image_mean,
            .image_std = image_std,
        };
    }

    /// 返回视觉编码器是否可用（权重已加载）
    pub fn isAvailable(self: *const VisionEncoder) bool {
        return self.weights.patch_embeddings_0 != null;
    }

    /// 编码 RGB 图像数据，返回视觉嵌入 tokens
    ///
    /// @param ctx ggml 计算上下文
    /// @param cgraph 计算图
    /// @param image_data RGB 图像数据 [height][width][3]，值范围 [0, 255]
    /// @param img_width 图像宽度
    /// @param img_height 图像高度
    /// @returns 视觉嵌入 [n_output_embd, n_tokens]
    pub fn encode(
        self: *const VisionEncoder,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        image_data: []const u8,
        img_width: u32,
        img_height: u32,
    ) !*ggml.Tensor {
        const p = self.params;

        // Validate input size
        const expected_len: usize = @as(usize, @intCast(img_width)) * @as(usize, @intCast(img_height)) * 3;
        if (image_data.len < expected_len) {
            log.err("Image data too small: got {d}, expected {d}", .{ image_data.len, expected_len });
            return error.InvalidImageData;
        }

        // 1. 归一化输入
        var inp = try preprocess.normalizeToTensor(ctx, image_data, img_width, img_height, self.image_mean, self.image_std, .standard);
        inp.setName("vision_input");

        // 2. 构建 VisionHParams 用于 backend 调用
        var hparams = graph.VisionHParams{
            .image_size = p.image_size,
            .patch_size = p.patch_size,
            .n_embd = p.n_embd,
            .n_head = p.n_head,
            .n_layer = p.n_layer,
            .n_ff = p.n_ff,
            .projection_dim = p.n_output_embd,
            .n_merge = p.n_merge,
            .eps = p.norm_eps,
            .rope_theta = p.rope_theta,
        };

        // 3. 通过 backend 构建完整计算图
        _ = try self.backend.buildGraph(ctx, cgraph, &self.weights, &hparams, inp);

        // 4. 从图中获取输出张量
        const ggml_c = @import("ggml").c;
        var name_buf: [64]u8 = undefined;
        const out_name = "mm_output";
        @memcpy(name_buf[0..out_name.len], out_name);
        name_buf[out_name.len] = 0;
        const result = ggml_c.ggml_graph_get_tensor(@ptrCast(cgraph), &name_buf) orelse {
            log.err("Failed to find output tensor '{s}' in graph", .{out_name});
            return error.TensorNotFound;
        };
        return @as(*ggml.Tensor, @ptrCast(result));
    }

    /// 估算给定分辨率图像的 token 数量
    pub fn estimateOutputTokens(self: *const VisionEncoder, img_width: u32, img_height: u32) u32 {
        return self.backend.estimateOutputTokens(img_width, img_height, self.params.patch_size, self.params.n_merge);
    }

    /// 计算视觉 token 预算下的最佳图像分辨率
    pub fn bestResolution(self: *const VisionEncoder, max_tokens: u32) struct { width: u32, height: u32 } {
        const n_merge = if (self.params.n_merge > 0) self.params.n_merge else 1;
        const max_patches = max_tokens * n_merge * n_merge;
        const side_patches = @max(1, @as(u32, @intFromFloat(@sqrt(@as(f64, @floatFromInt(max_patches))))));
        const side = side_patches * self.params.patch_size;
        return .{ .width = side, .height = side };
    }

    pub fn deinit(self: *VisionEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.weights.layers);
    }
};
