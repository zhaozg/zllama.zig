//! Vision encoder framework.
//!
//! Model-specific implementations live in src/mtmd/graph/models/
//! dispatched via VisionEncoderBackend.

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const config = @import("config.zig");
const types = @import("types.zig");
const preprocess = @import("preprocess");
const graph = @import("graph");
const encoder_debug = @import("encoder_debug");
const VisionEncoderParams = config.VisionEncoderParams;
const VisionEncoderWeights = types.VisionEncoderWeights;
const log = std.log.scoped(.vision_encoder);

pub const VisionEncoderBackend = graph.VisionEncoderBackend;
/// Debug tensor entries: (tensor_name_in_graph, output_filename).
///   1. inp_raw_scaled  — input after scale+bias (patches * 2 - 1)
///   2. inp             — after Conv2D patch embedding
///   3. pos_embd        — after adding 2D position embeddings
///   4. vit_output      — after ViT blocks (with 2D RoPE)
///   5. pooled          — after Pool 2D (avg, kernel=n_merge) + scale(sqrt(n_embd))
///   6. std_scaled      — after standardization: (hidden - std_bias) * std_scale
///   7. projected       — after multimodal embedder (rms_norm + mm_input_proj)
///   8. mm_output       — final output tensor
///
const debug_entries = [_]encoder_debug.DebugTensorEntry{
    .{ .tensor_name = "inp_raw", .filename = "zllama_vision_00_inp_raw.json", .is_input = true },
    .{ .tensor_name = "inp_raw_scaled", .filename = "zllama_vision_01_inp_raw_scaled.json" },
    .{ .tensor_name = "inp_conv_2d", .filename = "zllama_vision_02a_inp_conv_2d.json" },
    .{ .tensor_name = "inp_reshape_3d", .filename = "zllama_vision_02b_inp_reshape_3d.json" },
    .{ .tensor_name = "pos_x", .filename = "zllama_vision_02c_pos_x.json", .is_input = true },
    .{ .tensor_name = "pos_y", .filename = "zllama_vision_02d_pos_y.json", .is_input = true },
    .{ .tensor_name = "inp_final", .filename = "zllama_vision_02z_inp_final.json" },
    .{ .tensor_name = "pos_embd", .filename = "zllama_vision_03_pos_embd.json" },
    .{ .tensor_name = "pre_ln", .filename = "zllama_vision_04a0_pre_ln.json" },
    .{ .tensor_name = "pre_ln_norm", .filename = "zllama_vision_04a1_pre_ln_norm.json" },
    .{ .tensor_name = "pre_ln_norm_w", .filename = "zllama_vision_04a2_pre_ln_norm_w.json" },
    .{ .tensor_name = "pre_ln_norm_b", .filename = "zllama_vision_04a3_pre_ln_norm_b.json" },

    .{ .tensor_name = "layer_inp_normed-0", .filename = "zllama_vision_04b0_layer_inp_normed.json" },
    .{ .tensor_name = "qkv", .filename = "zllama_vision_04b1_qkv.json" },

    .{ .tensor_name = "Qcur_norm-0", .filename = "zllama_vision_04b2_Qcur_norm.json" },
    .{ .tensor_name = "Kcur_norm-0", .filename = "zllama_vision_04b3_Kcur_norm.json" },

    .{ .tensor_name = "Qcur_norm_per_head-0", .filename = "zllama_vision_04b4_Qcur_norm_per_head.json" },
    .{ .tensor_name = "Kcur_norm_per_head-0", .filename = "zllama_vision_04b5_Kcur_norm_per_head.json" },

    .{ .tensor_name = "Qcur_pos-0", .filename = "zllama_vision_04b6_Ocur_pos.json" },
    .{ .tensor_name = "Kcur_pos-0", .filename = "zllama_vision_04b7_Kcur_pos.json" },
    .{ .tensor_name = "Vcur_normed-0", .filename = "zllama_vision_04b8_Vcur_normed.json" },

    .{ .tensor_name = "kqv_out-0", .filename = "zllama_vision_04c1_layer_kqv_out.json" },
    .{ .tensor_name = "attn_out-0", .filename = "zllama_vision_04c2_layer_attn_out.json" },
    .{ .tensor_name = "ffn_inp-0",        .filename = "zllama_vision_04d1_layer_ffn_inp.json" },
    .{ .tensor_name = "ffn_inp_normed-0", .filename = "zllama_vision_04d2_layer_ffn_inp_normed.json" },
    .{ .tensor_name = "ffn_out-0", .filename = "zllama_vision_04e_layer_ffn_out.json" },
    .{ .tensor_name = "layer_out-0", .filename = "zllama_vision_04f1_layer_layer_out.json" },
    .{ .tensor_name = "layer_out_scaled-0", .filename = "zllama_vision_04f1_layer_layer_out.json" },

    .{ .tensor_name = "out_scaled", .filename = "zllama_vision_04g_out_scaled.json" },
    .{ .tensor_name = "vit_output", .filename = "zllama_vision_04z_vit_output.json" },

    .{ .tensor_name = "pooled_cont_4d",     .filename = "zllama_vision_05a_pooled-cont-4d.json"},
    .{ .tensor_name = "pooled_pool_2d",     .filename = "zllama_vision_05b_pooled-pool-2d.json"},
    .{ .tensor_name = "pooled_reshape_3d",  .filename = "zllama_vision_05c_pooled-reshape-3d.json"},
    .{ .tensor_name = "pooled_cont",        .filename = "zllama_vision_05d_pooled-cont.json"},
    .{ .tensor_name = "pooled",             .filename = "zllama_vision_05z_pooled.json"},

    .{ .tensor_name = "std_scaled", .filename = "zllama_vision_06_std_scaled.json" },
    .{ .tensor_name = "mm_output", .filename = "zllama_vision_07_mm_output.json" },
};

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
        projector_type: []const u8,
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
        // Ref: llama.cpp clip.cpp line 1205: get_f32(KEY_LAYER_NORM_EPS, hparams.eps)
        if (gguf_file.getF32("clip.vision.attention.layer_norm_epsilon")) |v| params.norm_eps = v;
        // Ref: llama.cpp clip.cpp lines 1262-1276: use_gelu / use_silu -> ffn_op
        {
            const use_gelu = gguf_file.getBool("clip.use_gelu") orelse false;
            const use_silu = gguf_file.getBool("clip.use_silu") orelse false;
            if (use_gelu and use_silu) {
                log.err("VisionEncoder.init: both clip.use_gelu and clip.use_silu are true", .{});
                return error.InvalidGGUFMetadata;
            }
            if (use_gelu) {
                params.ffn_op = .gelu;
            } else if (use_silu) {
                params.ffn_op = .silu;
            } else {
                params.ffn_op = .gelu_quick;
            }
        }
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
        log.warn("VisionEncoder.init: image_mean=[{d:.6}, {d:.6}, {d:.6}] image_std=[{d:.6}, {d:.6}, {d:.6}]", .{ image_mean[0], image_mean[1], image_mean[2], image_std[0], image_std[1], image_std[2] });

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
            .ffn_op = ffnOpToGraph(params.ffn_op),
        };

        // NOTE: custom_image_min/max_tokens are set by backend.loadParams via setLimitImageTokens,
        // so we should NOT set them from image_min/max_pixels here (they are pixel counts, not token counts).
        // hparams.custom_image_min_tokens = @intCast(params.image_min_pixels);  // BUG: pixel count != token count
        // hparams.custom_image_max_tokens = @intCast(params.image_max_pixels);  // BUG: pixel count != token count

        backend.loadParams(io, gguf_file, &hparams);

        params.n_merge = hparams.n_merge;
        params.patch_size = hparams.patch_size;
        params.rope_theta = hparams.rope_theta;
        params.ffn_op = graphFfnOpToConfig(hparams.ffn_op);
        if (hparams.image_min_pixels > 0) {
            params.image_min_pixels = @intCast(hparams.image_min_pixels);
            params.image_max_pixels = @intCast(hparams.image_max_pixels);
        }

        var weights = VisionEncoderWeights{};
        try backend.loadWeights(io, allocator, gguf_file, ctx, &weights);
        try backend.loadClampInfo(io, allocator, gguf_file, &weights);

        return VisionEncoder{
            .params = VisionEncoderParams{
                .image_size = params.image_size,
                .patch_size = params.patch_size,
                .n_embd = params.n_embd,
                .n_head = params.n_head,
                .n_layer = params.n_layer,
                .n_ff = params.n_ff,
                .n_output_embd = params.n_output_embd,
                .n_merge = params.n_merge,
                .rope_theta = params.rope_theta,
                .norm_eps = params.norm_eps,
                .ffn_op = params.ffn_op,
                .image_min_pixels = params.image_min_pixels,
                .image_max_pixels = params.image_max_pixels,
                .user_max_pixels = params.user_max_pixels,
                .projector_type = projector_type,
            },
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

    pub fn supportBatch(self: *const VisionEncoder) bool {
        return self.backend.supportBatch;
    }

    pub fn setUserMaxPixels(self: *VisionEncoder, max_pixels: u32) void {
        self.params.user_max_pixels = max_pixels;
    }

    pub fn encodeSimple(self: *const VisionEncoder, io: std.Io, ctx: *ggml.Context, cgraph: *ggml.CGraph, image_data: []const u8, img_width: u32, img_height: u32) !*ggml.Tensor {
        return self.encode(io, ctx, cgraph, image_data, img_width, img_height, 4);
    }

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

        log.debug("encode: input {d}x{d}, image_data.len={d}, expected_len={d}", .{ img_width, img_height, image_data.len, expected_len });

        const cur_merge: u32 = if (p.n_merge == 0) 1 else p.n_merge;
        const align_size = p.patch_size * cur_merge;
        const effective_max = if (p.user_max_pixels > 0) p.user_max_pixels else p.image_max_pixels;
        const effective_min = p.image_min_pixels;

        log.debug("encode: align_size={d}, effective_min={d}, effective_max={d}", .{ align_size, effective_min, effective_max });

        const resize_result = try preprocess.resizeAndNormalize(
            ctx,
            std.heap.page_allocator,
            image_data,
            img_width,
            img_height,
            self.image_mean,
            self.image_std,
            align_size,
            effective_min,
            effective_max,
        );

        log.debug("encode: resizeAndNormalize returned {d}x{d}", .{ resize_result.new_width, resize_result.new_height });

        const inp: *ggml.Tensor = resize_result.tensor;
        const new_w: u32 = resize_result.new_width;
        const new_h: u32 = resize_result.new_height;
        log.debug("encode: {d}x{d} -> {d}x{d}", .{ img_width, img_height, new_w, new_h });

        ggml.setInput(inp);

        var hparams = graph.VisionHParams{
            .image_size = new_w,
            .image_height = new_h,
            .patch_size = p.patch_size,
            .n_embd = p.n_embd,
            .n_head = p.n_head,
            .n_layer = p.n_layer,
            .n_ff = p.n_ff,
            .projection_dim = p.n_output_embd,
            .n_merge = p.n_merge,
            .eps = p.norm_eps,
            .rope_theta = p.rope_theta,
            .ffn_op = ffnOpToGraph(p.ffn_op),
        };

        log.debug("encode: calling backend.buildGraph...", .{});
        _ = try self.backend.buildGraph(io, ctx, cgraph, &self.weights, &hparams, inp);
        log.debug("encode: buildGraph completed, looking for mm_output...", .{});
        return cgraph.getTensor("mm_output") orelse return error.TensorNotFound;
    }

    pub fn estimateOutputTokens(self: *const VisionEncoder, io: std.Io, img_width: u32, img_height: u32) u32 {
        return self.backend.estimateOutputTokens(io, img_width, img_height, self.params.patch_size, self.params.n_merge);
    }

    pub fn bestResolution(self: *const VisionEncoder, max_tokens: u32) struct { width: u32, height: u32 } {
        const cur_merge: u32 = if (self.params.n_merge == 0) 1 else self.params.n_merge;
        const align_size = self.params.patch_size * cur_merge;
        const max_pixels = max_tokens * align_size * align_size;
        const size = preprocess.calcSizePreservedRatio(
            4096,
            4096,
            align_size,
            self.params.image_min_pixels,
            @min(max_pixels, self.params.image_max_pixels),
        );
        return .{ .width = size.w, .height = size.h };
    }

    pub fn deinit(self: *VisionEncoder, allocator: std.mem.Allocator) void {
        self.weights.clamp_info_map.deinit();
        allocator.free(self.weights.layers);
    }

    /// under a "debug_vision" subdirectory. The saved tensors correspond to key stages
    /// of the Gemma4V vision encoder pipeline:
    ///
    /// Additionally, weight tensors (patch_embeddings, position_embeddings,
    /// Save debug data for vision encoder alignment analysis with llama.cpp.
    /// Uses shared encoder_debug.saveDebugTensors for graph tensors.
    pub fn saveDebugData(self: *const VisionEncoder, io: std.Io, allocator: std.mem.Allocator, cgraph: *ggml.CGraph) void {
        _ = self;
        const subdir = "debug_vision";

        encoder_debug.saveDebugTensors(io, allocator, subdir, &debug_entries, cgraph, log);
    }
};

/// Convert config.FfnOp to graph.FFNOpType
fn ffnOpToGraph(op: config.FfnOp) graph.FFNOpType {
    return switch (op) {
        .silu => .silu,
        .gelu => .gelu,
        .gelu_quick => .gelu_quick,
    };
}

/// Convert graph.FFNOpType to config.FfnOp
fn graphFfnOpToConfig(op: graph.FFNOpType) config.FfnOp {
    return switch (op) {
        .silu => .silu,
        .gelu => .gelu,
        .gelu_quick => .gelu_quick,
        .gelu_erf => .gelu,
        .relu_sqr => .silu,
    };
}
