//! 音频编码器框架
//!
//! 提供音频编码器的通用框架，通过 AudioEncoderBackend 分发到具体模型。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("graph");
const config_mod = @import("config.zig");
const log = std.log.scoped(.audio_encoder);

pub const AudioEncoderBackend = graph.AudioEncoderBackend;

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
        _ = io;
        var vparams = graph.VisionHParams{};
        backend.loadParams(gguf_file, &vparams);

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
        try backend.loadWeights(allocator, gguf_file, ctx, &weights);
        try backend.loadClampInfo(allocator, gguf_file, &weights);

        return AudioEncoder{
            .params = params,
            .weights = weights,
            .ctx_weights = ctx,
            .backend = backend,
        };
    }

    pub fn isAvailable(self: *const AudioEncoder) bool {
        return self.weights.sscp_conv_w[0] != null;
    }

    /// Encode Mel spectrogram tensor with compute execution.
    pub fn encode(
        self: *const AudioEncoder,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        mel_tensor: *ggml.Tensor,
        n_threads: i32,
    ) !*ggml.Tensor {
        _ = io;
        var vparams = graph.VisionHParams{
            .n_embd = self.params.n_embd,
            .n_head = self.params.n_head,
            .n_layer = self.params.n_layer,
            .n_mel_bins = self.params.n_mel_bins,
            .eps = self.params.norm_eps,
        };
        _ = try self.backend.buildGraph(ctx, cgraph, &self.weights, &vparams, mel_tensor, &self.weights.clamp_info_map);

        ggml.loadBackends();
        const cpu = try ggml.backendCpuInit();
        defer ggml.backendFree(cpu);
        ggml.backendCpuSetNThreads(cpu, n_threads);
        const buft = ggml.backendGetDefaultBufferType(cpu);
        var gallocr = try ggml.Gallocr.init(buft);
        defer gallocr.free();
        _ = gallocr.reserve(cgraph);
        _ = gallocr.allocGraph(cgraph);
        if (!ggml.backendGraphCompute(cpu, cgraph)) return error.ComputeFailed;

        const ggml_c = @import("ggml").c;
        var name_buf: [64]u8 = undefined;
        const out_name = "debug_audio_multimodal_embedder_output";
        @memcpy(name_buf[0..out_name.len], out_name);
        name_buf[out_name.len] = 0;
        const result = ggml_c.ggml_graph_get_tensor(@ptrCast(cgraph), &name_buf) orelse return error.TensorNotFound;
        return @as(*ggml.Tensor, @ptrCast(result));
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
        n_threads: i32,
    ) !*ggml.Tensor {
        const mel_tensor = try ctx.newTensor2d(ggml.Type.f32, @as(i64, @intCast(mel_frames)), @as(i64, @intCast(mel_bins)));
        @memcpy(mel_tensor.dataF32(), mel_data);
        ggml.setInput(mel_tensor);
        return self.encode(io, ctx, cgraph, mel_tensor, n_threads);
    }

    pub fn estimateOutputTokens(self: *const AudioEncoder, audio_length_sec: f32) u32 {
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
        return self.backend.estimateOutputTokens(actual_frames);
    }

    pub fn deinit(self: *AudioEncoder, allocator: std.mem.Allocator) void {
        self.weights.clamp_info_map.deinit();
        allocator.free(self.weights.layers);
    }

    pub fn saveDebugData(self: *const AudioEncoder, io: std.Io, cgraph: *ggml.CGraph) void {
        _ = self;
        const debug = @import("debug");
        const subdir = "debug_audio";
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_multimodal_embedder_output.json", "debug_audio_multimodal_embedder_output", cgraph) catch {};
    }
};
