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

const VisionEncoderParams = config.VisionEncoderParams;
const VisionEncoderWeights = types.VisionEncoderWeights;
const log = std.log.scoped(.vision_encoder);

pub const VisionEncoderBackend = graph.VisionEncoderBackend;

/// Names of intermediate tensors in the vision encoder graph that can be
/// saved for debug/alignment analysis.
pub const debug_tensor_names = [_][]const u8{
    "inp_raw_scaled",
    "inp",
    "pos_embd",
    "vit_output",
    "pooled",
    "std_scaled",
    "mm_output",
    "Qcur_pos-0",
    "Kcur_pos-0",
    "Vcur_normed-0",
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

    /// Mark intermediate tensors in the graph with ggml.setOutput() so that
    /// their data is preserved after graph computation.
    ///
    /// This MUST be called BEFORE graph allocation/computation (i.e. before
    /// Gallocr.allocGraph or ggml_backend_graph_compute). Without setOutput(),
    /// the graph allocator may reuse the memory buffers of intermediate tensors,
    /// causing saveDebugData() to read stale/overwritten data.
    ///
    /// Call this right after buildGraph() returns, before computeGraph().
    ///
    /// Parameters:
    ///   - cgraph: the computed graph (must have tensors named via setName())
    pub fn markDebugOutputs(cgraph: *ggml.CGraph) void {
        const debug = @import("debug");
        for (debug_tensor_names) |name| {
            debug.markTensorAsOutput(cgraph, name) catch |err| {
                log.warn("markDebugOutputs: failed to mark '{s}': {}", .{ name, err });
            };
        }
    }

    /// Save debug data for vision encoder alignment analysis with llama.cpp.
    ///
    /// This function saves intermediate tensor data from the computed graph to JSON files
    /// under a "debug_vision" subdirectory. The saved tensors correspond to key stages
    /// of the Gemma4V vision encoder pipeline:
    ///
    ///   1. inp_raw_scaled  — input after scale+bias (patches * 2 - 1)
    ///   2. inp             — after Conv2D patch embedding
    ///   3. pos_embd        — after adding 2D position embeddings
    ///   4. vit_output      — after ViT blocks (with 2D RoPE)
    ///   5. pooled          — after Pool 2D (avg, kernel=n_merge) + scale(sqrt(n_embd))
    ///   6. std_scaled      — after standardization: (hidden - std_bias) * std_scale
    ///   7. projected       — after multimodal embedder (rms_norm + mm_input_proj)
    ///   8. mm_output       — final output tensor
    ///
    /// Additionally, weight tensors (patch_embeddings, position_embeddings,
    /// std_bias, std_scale, mm_input_proj_w) are saved for cross-reference.
    ///
    /// Reference: llama.cpp clip.cpp debug_output_embeddings section (line ~4518)
    ///            and gemma4v.cpp build() pipeline.
    ///
    /// NOTE: Before calling this, you MUST call markDebugOutputs(cgraph) BEFORE
    /// graph computation to ensure intermediate tensor data is preserved.
    ///
    /// Parameters:
    ///   - io: I/O instance
    ///   - allocator: memory allocator
    ///   - cgraph: computed graph (must be computed before calling this)
    pub fn saveDebugData(self: *const VisionEncoder, io: std.Io, allocator: std.mem.Allocator, cgraph: *ggml.CGraph) void {
        _ = self;
        const debug = @import("debug");
        const subdir = "debug_vision";

        // === Intermediate activation tensors (from graph, by name) ===
        //
        // NOTE: In gemma4v/gemma4uv, the "projected" tensor is renamed to "mm_output"
        // before being added to the graph via buildForwardExpand. This means
        // ggml_graph_get_tensor(gf, "projected") cannot find it (the tensor's name
        // is now "mm_output"). We save "mm_output" for both 07 and 08 since they
        // represent the same tensor data.
        //
        // Reference: llama.cpp clip.cpp debug_output_embeddings section (line ~4609)
        // In llama.cpp, the tensor is named "projected" (via cb()), and there is
        // no "mm_output" name set, so llama.cpp can find "projected" but not "mm_output".

        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_00_inp_raw.json", "inp_raw", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'inp_raw': {}", .{err});
        };
        // Step 1: Scale+bias input
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_01_inp_raw_scaled.json", "inp_raw_scaled", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'inp_raw_scaled': {}", .{err});
        };

        // Step 2: Conv2D patch embedding output
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_02_inp.json", "inp", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'inp': {}", .{err});
        };

        // Step 3: After position embeddings
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_03_pos_embd.json", "pos_embd", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'pos_embd': {}", .{err});
        };

        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04a_pre_ln.json", "pre_ln", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'pre_ln': {}", .{err});
        };
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04b_layer0_inp_normed.json", "layer_inp_normed", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'layer_inp_normed': {}", .{err});
        };
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04c_layer0_attn_out.json", "attn_out", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'attn_out': {}", .{err});
        };
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04d_layer0_ffn_inp.json", "ffn_inp", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'ffn_inp': {}", .{err});
        };

        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04e_layer0_ffn_out.json", "ffn_out", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'ffn_out': {}", .{err});
        };

        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04f_layer0_layer_out.json", "layer_out", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'layer_out': {}", .{err});
        };

        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04g_out_scaled.json", "out_scaled", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'out_scaled': {}", .{err});
        };

        // Step 4: ViT blocks output
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_04_vit_output.json", "vit_output", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'vit_output': {}", .{err});
        };

        // Step 5: Pool 2D output
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_05_pooled.json", "pooled", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'pooled': {}", .{err});
        };

        // Step 6: Standardization output (only if std_bias/std_scale exist)
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_06_std_scaled.json", "std_scaled", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'std_scaled': {}", .{err});
        };

        // Step 7: Final mm_output
        debug.saveTensorFromGraph(io, allocator, subdir, "zllama_vision_07_mm_output.json", "mm_output", cgraph) catch |err| {
            log.warn("saveDebugData: failed to save 'mm_output': {}", .{err});
        };

        // === Weight tensors (from weights struct, for cross-reference) ===

        // if (self.weights.patch_embeddings_0) |t| {
        //     debug.saveTensor(io, allocator, subdir, "zllama_vision_00_patch_embeddings_weight.json", t) catch {};
        // }
        // if (self.weights.position_embeddings) |t| {
        //     debug.saveTensor(io, allocator, subdir, "zllama_vision_00_position_embeddings_weight.json", t) catch {};
        // }
        // if (self.weights.std_bias) |t| {
        //     debug.saveTensor(io, allocator, subdir, "zllama_vision_00_std_bias.json", t) catch {};
        // }
        // if (self.weights.std_scale) |t| {
        //     debug.saveTensor(io, allocator, subdir, "zllama_vision_00_std_scale.json", t) catch {};
        // }
        // if (self.weights.mm_input_proj_w) |t| {
        //     debug.saveTensor(io, allocator, subdir, "zllama_vision_00_mm_input_proj_weight.json", t) catch {};
        // }
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
