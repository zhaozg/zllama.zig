//! 音频编码器框架
//!
//! 提供音频编码器的通用框架，包括：
//! - AudioEncoder 结构体（生命周期管理、编码接口）
//! - 模型无关的权重容器（使用 graph.VisionEncoderWeights）
//!
//! 模型特定的实现（权重加载、图构建、token 估算）下沉到
//! src/mtmd/graph/models/ 中，通过 AudioEncoderBackend 分发。
//!
//! 设计原则：
//! - encoder.zig 只包含框架、流程、架构代码
//! - 具体模型的张量名称、图结构、参数硬编码都在 graph/models/ 中

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("graph");
const debug = @import("debug");
const config_mod = @import("config.zig");

const log = std.log.scoped(.audio_encoder);

/// 音频编码器后端接口（定义在 graph/mod.zig 中）
pub const AudioEncoderBackend = graph.AudioEncoderBackend;

// ============================================================================
// 音频编码器
// ============================================================================

/// 通用音频编码器
/// 通过 backend 字段分发到具体模型实现
pub const AudioEncoder = struct {
    params: config_mod.AudioEncoderParams,
    weights: graph.VisionEncoderWeights,
    ctx_weights: *ggml.Context,
    backend: *const AudioEncoderBackend,

    /// 从 GGUF 文件初始化音频编码器
    pub fn init(
        io: std.Io,
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
        backend: *const AudioEncoderBackend,
    ) !AudioEncoder {
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

        log.info("Loading audio encoder: backend={s}, embd={d}, heads={d}, d_head={d}, layers={d}, ff={d}, mel_bins={d}, eps={e}", .{
            backend.name, params.n_embd, params.n_head, params.d_head, params.n_layer, params.n_ff, params.n_mel_bins, params.norm_eps,
        });

        var weights = graph.VisionEncoderWeights{};
        try backend.loadWeights(allocator, gguf_file, ctx, &weights);
        try backend.loadClampInfo(allocator, gguf_file, &weights);

        log.info("Audio encoder loaded: backend={s}, {d} layers, {d} clamp entries", .{ backend.name, weights.layers.len, weights.clamp_info_map.count() });

        // === DEBUG: 保存子采样卷积权重数据 ===
        for (0..2) |i| {
            if (weights.sscp_conv_w[i]) |t| {
                const fname = try std.fmt.allocPrint(allocator, "zllama_audio_conv1d_{d}_weight.json", .{i});
                defer allocator.free(fname);
                debug.saveData(io, "debug_audio", fname, "conv1d_weight", t.dataF32()) catch |err| {
                    log.debug("Failed to save conv1d.{d}.weight debug data: {}", .{ i, err });
                };
            }
            if (weights.sscp_conv_b[i]) |t| {
                const fname = try std.fmt.allocPrint(allocator, "zllama_audio_conv1d_{d}_bias.json", .{i});
                defer allocator.free(fname);
                debug.saveData(io, "debug_audio", fname, "conv1d_bias", t.dataF32()) catch |err| {
                    log.debug("Failed to save conv1d.{d}.bias debug data: {}", .{ i, err });
                };
            }
        }
        if (weights.sscp_inp_proj_w) |t| {
            debug.saveData(io, "debug_audio", "zllama_audio_input_proj_weight.json", "input_proj_weight", t.dataF32()) catch |err| {
                log.debug("Failed to save input_proj.weight debug data: {}", .{err});
            };
        }

        return AudioEncoder{
            .params = params,
            .weights = weights,
            .ctx_weights = ctx,
            .backend = backend,
        };
    }

    /// 返回音频编码器是否可用（权重已加载）
    pub fn isAvailable(self: *const AudioEncoder) bool {
        return self.weights.sscp_conv_w[0] != null;
    }

    /// 编码 Mel 频谱张量为音频嵌入 tokens
    /// 通过 backend.buildGraph 分发到具体模型
    pub fn encode(
        self: *const AudioEncoder,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        mel_tensor: *ggml.Tensor,
    ) !*ggml.Tensor {
        _ = io;
        // 构建 VisionHParams 用于 backend 调用
        var vparams = graph.VisionHParams{
            .n_embd = self.params.n_embd,
            .n_head = self.params.n_head,
            .n_layer = self.params.n_layer,
            .n_mel_bins = self.params.n_mel_bins,
            .eps = self.params.norm_eps,
        };
        _ = try self.backend.buildGraph(ctx, cgraph, &self.weights, &vparams, mel_tensor, &self.weights.clamp_info_map);

        // 从图中获取输出张量（通过 C API ggml_graph_get_tensor）
        const ggml_c = @import("ggml").c;
        var name_buf: [64]u8 = undefined;
        const out_name = "debug_audio_multimodal_embedder_output";
        @memcpy(name_buf[0..out_name.len], out_name);
        name_buf[out_name.len] = 0;
        const result = ggml_c.ggml_graph_get_tensor(@ptrCast(cgraph), &name_buf) orelse {
            log.err("Failed to find output tensor '{s}' in graph", .{out_name});
            return error.TensorNotFound;
        };
        return @as(*ggml.Tensor, @ptrCast(result));
    }

    /// 编码原始 Mel 数据为音频嵌入 tokens
    /// 内部创建 Mel 张量后调用 encode
    pub fn encodeRaw(
        self: *const AudioEncoder,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        mel_data: []const f32,
        mel_bins: u32,
        mel_frames: u32,
    ) !*ggml.Tensor {
        const mel_tensor = try ctx.newTensor2d(ggml.Type.f32, @as(i64, @intCast(mel_frames)), @as(i64, @intCast(mel_bins)));
        @memcpy(mel_tensor.dataF32(), mel_data);
        return self.encode(io, ctx, cgraph, mel_tensor);
    }

    /// 估算编码后的 token 数量
    /// 通过 backend.estimateOutputTokens 分发
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

    /// 保存中间张量的调试数据（需在 graph.compute() 之后调用）
    pub fn saveDebugData(_: *const AudioEncoder, io: std.Io, cgraph: *ggml.CGraph) void {
        const subdir = "debug_audio";
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_conv2d_0_output.json", "debug_audio_conv2d_0_output", cgraph) catch |err| {
            log.debug("Failed to save conv2d_0_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_conv2d_1_output.json", "debug_audio_conv2d_1_output", cgraph) catch |err| {
            log.debug("Failed to save conv2d_1_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_after_cont.json", "debug_audio_after_cont", cgraph) catch |err| {
            log.debug("Failed to save audio_after_cont debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_flatten_output.json", "debug_audio_flatten_output", cgraph) catch |err| {
            log.debug("Failed to save flatten_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_conformer_blocks_output.json", "debug_audio_conformer_blocks_output", cgraph) catch |err| {
            log.debug("Failed to save conformer_blocks_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_multimodal_embedder_output.json", "debug_audio_multimodal_embedder_output", cgraph) catch |err| {
            log.debug("Failed to save multimodal_embedder_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_audio_input_proj_output.json", "debug_audio_input_proj_output", cgraph) catch |err| {
            log.debug("Failed to save input_proj_output debug data: {}", .{err});
        };

        debug.saveTensorFromGraph(io, subdir, "zllama_layer0_half_step_1_output.json", "debug_audio_half_step_1_output", cgraph) catch |err| {
            log.debug("Failed to save half_step_1_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_layer0_self_attention_with_RPE_output.json", "debug_audio_self_attention_with_RPE_output", cgraph) catch |err| {
            log.debug("Failed to save half_step_1_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_layer0_convolution_output.json", "debug_audio_convolution_output", cgraph) catch |err| {
            log.debug("Failed to save half_step_1_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_layer0_half_step_2_output.json", "debug_audio_half_step_2_output", cgraph) catch |err| {
            log.debug("Failed to save half_step_1_output debug data: {}", .{err});
        };
        debug.saveTensorFromGraph(io, subdir, "zllama_layer0_norm_output.json", "debug_audio_layer_0_norm_output", cgraph) catch |err| {
            log.debug("Failed to save half_step_1_output debug data: {}", .{err});
        };

    }

    pub fn deinit(self: *AudioEncoder, allocator: std.mem.Allocator) void {
        self.weights.clamp_info_map.deinit();
        allocator.free(self.weights.layers);
    }
};
