//! 视觉编码器框架
//!
//! 提供视觉编码器的通用框架，包括生命周期管理、编码接口。
//! 模型特定实现下沉到 src/mtmd/graph/models/ 中，通过 VisionEncoderBackend 分发。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const config = @import("config.zig");
const types = @import("types.zig");
const preprocess = @import("preprocess.zig");
const graph = @import("graph");

const VisionEncoderParams = config.VisionEncoderParams;
const VisionEncoderWeights = types.VisionEncoderWeights;
const log = std.log.scoped(.vision_encoder);

pub const VisionEncoderBackend = graph.VisionEncoderBackend;

pub const VisionEncoder = struct {
    params: VisionEncoderParams,
    weights: VisionEncoderWeights,
    ctx_weights: *ggml.Context,
    backend: *const VisionEncoderBackend,
    image_mean: [3]f32 = .{ 0.0, 0.0, 0.0 },
    image_std: [3]f32 = .{ 1.0, 1.0, 1.0 },

    pub fn init(
        io: std.Io,
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
        backend: *const VisionEncoderBackend,
    ) !VisionEncoder {
        var params = VisionEncoderParams{};
        if (gguf_file.getU32("clip.vision.image_size")) |v| params.image_size = v else if (gguf_file.getU32("gemma4.vision.image_size")) |v| params.image_size = v;
        if (gguf_file.getU32("clip.vision.patch_size")) |v| params.patch_size = v else if (gguf_file.getU32("gemma4.vision.patch_size")) |v| params.patch_size = v;
        if (gguf_file.getU32("clip.vision.embedding_length")) |v| params.n_embd = v else if (gguf_file.getU32("gemma4.vision.embedding_length")) |v| params.n_embd = v;
        if (gguf_file.getU32("clip.vision.attention.head_count")) |v| params.n_head = v else if (gguf_file.getU32("gemma4.vision.attention_head_count")) |v| params.n_head = v;
        if (gguf_file.getU32("clip.vision.block_count")) |v| params.n_layer = v else if (gguf_file.getU32("gemma4.vision.block_count")) |v| params.n_layer = v;
        if (gguf_file.getU32("clip.vision.feed_forward_length")) |v| params.n_ff = v else if (gguf_file.getU32("gemma4.vision.feed_forward_length")) |v| params.n_ff = v;
        if (gguf_file.getU32("clip.vision.projection_dim")) |v| params.n_output_embd = v else if (gguf_file.getU32("gemma4.vision.projection_dim")) |v| params.n_output_embd = v;
        if (gguf_file.getU32("clip.vision.projector.scale_factor")) |v| {
            params.n_merge = v;
        } else if (gguf_file.getU32("gemma4.vision.projector.scale_factor")) |v| {
            params.n_merge = v;
        }
        if (gguf_file.getF32("clip.vision.rope_theta")) |v| params.rope_theta = v;
        if (gguf_file.getU32("clip.vision.image_min_pixels")) |v| params.image_min_pixels = v;
        if (gguf_file.getU32("clip.vision.image_max_pixels")) |v| params.image_max_pixels = v;

        if (params.image_min_pixels == 0 and params.image_max_pixels == 0) {
            const patch_area = params.patch_size * params.patch_size * params.n_merge * params.n_merge;
            if (patch_area > 0) {
                params.image_min_pixels = 8 * patch_area;
                params.image_max_pixels = 4096 * patch_area;
            }
        }

        var image_mean: [3]f32 = .{ 0.0, 0.0, 0.0 };
        var image_std: [3]f32 = .{ 1.0, 1.0, 1.0 };
        if (gguf_file.getF32Array("clip.vision.image_mean", 3)) |mean| {
            for (mean, 0..) |v, i| image_mean[i] = v;
        }
        if (gguf_file.getF32Array("clip.vision.image_std", 3)) |std_val| {
            for (std_val, 0..) |v, i| image_std[i] = v;
        }

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
        backend.loadParams(io, gguf_file, &hparams);

        var weights = VisionEncoderWeights{};
        try backend.loadWeights(io, allocator, gguf_file, ctx, &weights);
        try backend.loadClampInfo(io, allocator, gguf_file, &weights);

        return VisionEncoder{
            .params = params,
            .weights = weights,
            .ctx_weights = ctx,
            .backend = backend,
            .image_mean = image_mean,
            .image_std = image_std,
        };
    }

    pub fn isAvailable(self: *const VisionEncoder) bool {
        return self.weights.patch_embeddings_0 != null;
    }

    /// Whether this encoder supports batch processing (multiple images in one forward pass).
    pub fn supportBatch(self: *const VisionEncoder) bool {
        return self.backend.supportBatch;
    }

    pub fn setUserMaxPixels(self: *VisionEncoder, max_pixels: u32) void {
        self.params.user_max_pixels = max_pixels;
    }

    /// Backward-compatible encode without n_threads, defaults to 4.
    pub fn encodeSimple(self: *const VisionEncoder, io: std.Io, ctx: *ggml.Context, cgraph: *ggml.CGraph, image_data: []const u8, img_width: u32, img_height: u32) !*ggml.Tensor {
        return self.encode(io, ctx, cgraph, image_data, img_width, img_height, 4);
    }

    /// Encode RGB image data with compute execution.
    pub fn encode(
        self: *const VisionEncoder,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        image_data: []const u8,
        img_width: u32,
        img_height: u32,
        n_threads: i32,
    ) !*ggml.Tensor {
        _ = n_threads;
        const p = self.params;
        const expected_len: usize = @as(usize, @intCast(img_width)) * @as(usize, @intCast(img_height)) * 3;
        if (image_data.len < expected_len) return error.InvalidImageData;

        var inp = try preprocess.normalizeToTensor(ctx, image_data, img_width, img_height, self.image_mean, self.image_std, .standard);
        inp.setName("vision_input");
        ggml.setInput(inp);

        var hparams = graph.VisionHParams{
            .image_size = img_width,
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
        _ = try self.backend.buildGraph(io, ctx, cgraph, &self.weights, &hparams, inp);

        // Return the output tensor by name; caller must compute before reading data.
        return cgraph.getTensor("mm_output") orelse return error.TensorNotFound;
    }

    pub fn estimateOutputTokens(self: *const VisionEncoder, io: std.Io, img_width: u32, img_height: u32) u32 {
        return self.backend.estimateOutputTokens(io, img_width, img_height, self.params.patch_size, self.params.n_merge);
    }

    pub fn deinit(self: *VisionEncoder, allocator: std.mem.Allocator) void {
        self.weights.clamp_info_map.deinit();
        allocator.free(self.weights.layers);
    }
};
