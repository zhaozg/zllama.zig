//! 音频编码器框架
//!
//! 提供音频编码器的通用框架，通过 AudioEncoderBackend 分发到具体模型。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("graph");
const config_mod = @import("config.zig");
const encoder_debug = @import("encoder_debug");
const log = std.log.scoped(.audio_encoder);
pub const AudioEncoderBackend = graph.AudioEncoderBackend;

/// Debug tensor entries: (tensor_name_in_graph, output_filename).
/// Used by saveDebugData.
pub const debug_entries = [_]encoder_debug.DebugTensorEntry{
    .{ .tensor_name = "pos_emb", .filename = "zllama_audio_00_pos_emb.json", .is_input = true },
    .{ .tensor_name = "kq_mask", .filename = "zllama_audio_00_kq_mask.json", .is_input = true },
    .{ .tensor_name = "debug_audio_04_encoder_input", .filename = "zllama_audio_04_mel_input.json" },
    .{ .tensor_name = "debug_audio_conv2d_0_output", .filename = "zllama_audio_05_conv2d_0_output.json" },
    .{ .tensor_name = "debug_audio_conv2d_1_output", .filename = "zllama_audio_06_conv2d_1_output.json" },
    .{ .tensor_name = "debug_audio_after_cont", .filename = "zllama_audio_07_after_cont.json" },
    .{ .tensor_name = "debug_audio_flatten_output", .filename = "zllama_audio_08_flatten_output.json" },
    .{ .tensor_name = "debug_audio_input_proj_output", .filename = "zllama_audio_09_input_proj_output.json" },
    .{ .tensor_name = "debug_audio_half_step_1_output", .filename = "zllama_audio_10_layer0_half_step_1_output.json" },
    .{ .tensor_name = "debug_audio_self_attention_with_RPE_output", .filename = "zllama_audio_11_layer0_self_attention_with_RPE_output.json" },
    .{ .tensor_name = "debug_audio_conv_build_normal_output", .filename = "zllama_audio_12_conv_build_normal_output.json" },
    .{ .tensor_name = "debug_audio_conv_glu_output", .filename = "zllama_audio_13_conv_pw1_glu_output.json" },
    .{ .tensor_name = "debug_audio_conv_dw_output", .filename = "zllama_audio_14_conv_dw_output.json" },
    .{ .tensor_name = "debug_audio_conv_dw_norm_silu_output", .filename = "zllama_audio_15_conv_dw_norm_silu_output.json" },
    .{ .tensor_name = "debug_audio_convolution_output", .filename = "zllama_audio_16_layer0_convolution_output.json" },
    .{ .tensor_name = "debug_audio_half_step_2_output", .filename = "zllama_audio_17_layer0_half_step_2_output.json" },
    .{ .tensor_name = "debug_audio_layer_0_norm_output", .filename = "zllama_audio_18_layer0_norm_output.json" },
    .{ .tensor_name = "debug_audio_conformer_blocks_output", .filename = "zllama_audio_19_conformer_blocks_output.json" },
    .{ .tensor_name = "mm_norm", .filename = "zllama_audio_90_mm_norm.json" },
    .{ .tensor_name = "mm_norm_scaled", .filename = "zllama_audio_91_mm_norm_scaled.json" },
    .{ .tensor_name = "mm_proj", .filename = "zllama_audio_92_mm_proj.json" },
};

pub const AudioEncoder = struct {
    params: config_mod.AudioEncoderParams,
    weights: graph.VisionEncoderWeights,
    ctx_weights: *ggml.Context,
    backend: *const AudioEncoderBackend,

    pub fn init(
        io: std.Io,
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
        backend: *const AudioEncoderBackend,
    ) !AudioEncoder {
        var vparams = graph.VisionHParams{};
        backend.loadParams(io, gguf_file, &vparams);

        const params = config_mod.AudioEncoderParams{
            .n_mel_bins = vparams.n_mel_bins,
            .n_embd = vparams.n_embd,
            .n_head = vparams.n_head,
            .d_head = if (vparams.n_head > 0) vparams.n_embd / vparams.n_head else 64,
            .n_layer = vparams.n_layer,
            .n_ff = vparams.n_ff,
            .norm_eps = vparams.eps,
        };

        var weights = graph.VisionEncoderWeights{};
        try backend.loadWeights(io, allocator, gguf_file, ctx, &weights);
        try backend.loadClampInfo(io, allocator, gguf_file, &weights);

        return AudioEncoder{
            .params = params,
            .weights = weights,
            .ctx_weights = ctx,
            .backend = backend,
        };
    }

    pub fn isAvailable(self: *const AudioEncoder) bool {
        // Gemma4UA 没有 Conformer blocks（sscp_conv_w 为空），
        // 检查 mm_input_proj_w 是否已加载
        return self.weights.sscp_conv_w[0] != null or self.weights.mm_input_proj_w != null;
    }

    /// Encode Mel spectrogram tensor — builds graph only; caller handles compute.
    pub fn encode(
        self: *const AudioEncoder,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        mel_tensor: *ggml.Tensor,
    ) !*ggml.Tensor {
        var vparams = graph.VisionHParams{
            .n_embd = self.params.n_embd,
            .n_head = self.params.n_head,
            .n_layer = self.params.n_layer,
            .n_mel_bins = self.params.n_mel_bins,
            .eps = self.params.norm_eps,
        };
        _ = try self.backend.buildGraph(io, ctx, cgraph, &self.weights, &vparams, mel_tensor, &self.weights.clamp_info_map);

        // Return the output tensor by name; caller must compute before reading data.
        return cgraph.getTensor("mm_proj") orelse return error.TensorNotFound;
    }

    /// Encode raw Mel data by creating a tensor first.
    pub fn encodeRaw(
        self: *const AudioEncoder,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        mel_data: []const f32,
        mel_bins: u32,
        mel_frames: u32,
    ) !*ggml.Tensor {
        const mel_tensor = try ctx.newTensor4d(ggml.Type.f32, @as(i64, @intCast(mel_frames)), @as(i64, @intCast(mel_bins)), 1, 1);
        // In no_alloc mode, allocate data buffer manually before dataSet.
        const no_alloc = ctx.getNoAlloc();
        if (no_alloc) {
            const data_size = @as(usize, @intCast(mel_tensor.nBytes()));
            const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
            @memset(buf, 0);
            mel_tensor.setDataPtr(buf);
        }
        try mel_tensor.dataSet(f32, mel_data);
        ggml.setInput(mel_tensor);
        return self.encode(io, ctx, cgraph, mel_tensor);
    }

    pub fn estimateOutputTokens(self: *const AudioEncoder, io: std.Io, audio_length_sec: f32) u32 {
        const n_samples: u32 = @intFromFloat(@as(f32, @floatFromInt(self.params.sample_rate)) * audio_length_sec);
        const framing = @import("framing.zig");
        const fc = framing.computeFrameCount(n_samples, .{
            .frame_length = config_mod.DEFAULT_FRAME_LENGTH,
            .hop_length = config_mod.DEFAULT_HOP_LENGTH,
            .n_fft = config_mod.DEFAULT_N_FFT,
        });
        const pad_left: u32 = config_mod.DEFAULT_FRAME_LENGTH / 2;
        const n_with_left: u32 = n_samples + pad_left;
        const pt_frames: u32 = if (n_with_left >= config_mod.DEFAULT_FRAME_LENGTH + 1)
            @as(u32, @intCast((n_with_left - (config_mod.DEFAULT_FRAME_LENGTH + 1)) / config_mod.DEFAULT_HOP_LENGTH)) + 1
        else
            0;
        const actual_frames = @min(fc.n_frames, pt_frames);
        return self.backend.estimateOutputTokens(io, actual_frames);
    }

    pub fn deinit(self: *AudioEncoder, allocator: std.mem.Allocator) void {
        self.weights.clamp_info_map.deinit();
        allocator.free(self.weights.layers);
    }

    /// Save debug data for audio encoder alignment analysis with llama.cpp.
    /// Uses shared encoder_debug.saveDebugTensors for graph tensors.
    pub fn saveDebugData(self: *const AudioEncoder, io: std.Io, allocator: std.mem.Allocator, cgraph: *ggml.CGraph) void {
        const subdir = "debug_audio";
        encoder_debug.saveDebugTensors(io, allocator, subdir, &debug_entries, cgraph, log);

        // Save weight tensors (only if available — gemma4ua may not have sscp_conv_w)
        const debug = @import("debug");
        if (self.weights.sscp_conv_w[0]) |t| debug.saveTensor(io, allocator, subdir, "zllama_audio_00_conv1d_0_weight.json", t) catch {};
        if (self.weights.sscp_conv_w[1]) |t| debug.saveTensor(io, allocator, subdir, "zllama_audio_00_conv1d_1_weight.json", t) catch {};
        if (self.weights.sscp_inp_proj_w) |t| debug.saveTensor(io, allocator, subdir, "zllama_audio_00_input_proj_weight.json", t) catch {};
        if (self.weights.audio_out_proj_w) |t| debug.saveTensor(io, allocator, subdir, "zllama_audio_00_out_proj_weight.json", t) catch {};
        if (self.weights.mm_input_proj_w) |t| debug.saveTensor(io, allocator, subdir, "zllama_audio_00_mm_input_proj_weight.json", t) catch {};
    }
};
