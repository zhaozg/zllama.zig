//! Conformer 音频编码器
//!
//! 提供对 Gemma 4 E2B 内建 Conformer 音频编码器的支持。
//! 该编码器能够直接处理最长 30 秒的 16kHz 单声道音频输入，
//! 将 Mel 频谱特征转换为模型可理解的音频嵌入 tokens。
//!
//! 架构: Conformer（卷积增强 Transformer）
//! - 子采样 Conv2D（2 层，步长 2）
//! - 多层 Conformer blocks
//!   - FFN 1（half-step, res_weight=0.5）
//!   - 分块局部自注意力 + 相对位置编码（RPE）
//!   - 深度可分离卷积模块（GLU gate）
//!   - FFN 2（half-step, res_weight=0.5）
//! - 输出投影到 LLM 嵌入空间
//!
//! 参考: llama.cpp tools/mtmd/models/gemma4a.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const c = @import("ggml").c;
const weight_loader = @import("weight_loader");
const config_mod = @import("config.zig");
const framing = @import("framing.zig");
const helper = @import("helper");

const log = std.log.scoped(.audio_encoder);

// ============================================================================
// Conformer 层权重
// ============================================================================

pub const ConformerLayerWeights = struct {
    // FFN 1 (half-step)
    ff_norm_w: ?*ggml.Tensor = null,
    ff_up_w: ?*ggml.Tensor = null,
    ff_down_w: ?*ggml.Tensor = null,
    ff_post_norm_w: ?*ggml.Tensor = null,

    // 注意力 (pre-norm + Q/K/V/O)
    attn_pre_norm_w: ?*ggml.Tensor = null,
    ln_1_w: ?*ggml.Tensor = null, // fallback norm
    q_w: ?*ggml.Tensor = null,
    k_w: ?*ggml.Tensor = null,
    v_w: ?*ggml.Tensor = null,
    o_w: ?*ggml.Tensor = null,
    o_b: ?*ggml.Tensor = null,
    attn_post_norm_w: ?*ggml.Tensor = null,

    // Q/K per-dimension scaling
    per_dim_scale_w: ?*ggml.Tensor = null,
    per_dim_k_scale_w: ?*ggml.Tensor = null,

    // RPE projection
    attn_k_rel_w: ?*ggml.Tensor = null,

    // 卷积模块
    norm_conv_w: ?*ggml.Tensor = null,
    conv_pw1_w: ?*ggml.Tensor = null,
    conv_dw_w: ?*ggml.Tensor = null,
    conv_dw_b: ?*ggml.Tensor = null,
    conv_norm_w: ?*ggml.Tensor = null,
    conv_pw2_w: ?*ggml.Tensor = null,

    // FFN 2 (half-step)
    ff_norm_1_w: ?*ggml.Tensor = null,
    ff_up_1_w: ?*ggml.Tensor = null,
    ff_down_1_w: ?*ggml.Tensor = null,
    ff_post_norm_1_w: ?*ggml.Tensor = null,

    // Layer output norm
    ln_2_w: ?*ggml.Tensor = null,
};

// ============================================================================
// 音频编码器权重
// ============================================================================

pub const AudioEncoderWeights = struct {
    params: config_mod.AudioEncoderParams,

    // 子采样卷积 (a.conv1d.{0,1}.*)
    sscp_conv_w: [2]?*ggml.Tensor = .{ null, null },
    sscp_conv_b: [2]?*ggml.Tensor = .{ null, null },
    sscp_norm_w: [2]?*ggml.Tensor = .{ null, null },
    // 子采样输入投影 (a.input_projection.*)
    sscp_inp_proj_w: ?*ggml.Tensor = null,
    sscp_inp_proj_b: ?*ggml.Tensor = null,

    // Conformer 层
    layers: []ConformerLayerWeights = &.{},

    // 输出投影 (a.pre_encode.out.*)
    audio_out_proj_w: ?*ggml.Tensor = null,
    audio_out_proj_b: ?*ggml.Tensor = null,

    // 多模态嵌入投影 (mm.a.*)
    mm_soft_emb_norm_w: ?*ggml.Tensor = null,
    mm_input_proj_w: ?*ggml.Tensor = null,
};

// ============================================================================
// 音频编码器
// ============================================================================

pub const AudioEncoder = struct {
    params: config_mod.AudioEncoderParams,
    weights: AudioEncoderWeights,
    ctx_weights: *ggml.Context,

    // Debug: intermediate tensor references for debug data saving
    debug_conv2d_0_raw: ?*ggml.Tensor = null,
    debug_conv2d_0_output: ?*ggml.Tensor = null,
    debug_conv2d_1_output: ?*ggml.Tensor = null,
    debug_flatten_output: ?*ggml.Tensor = null,
    debug_input_proj_output: ?*ggml.Tensor = null,

    // 从 GGUF 文件初始化音频编码器，加载所有权重到 ggml context
    pub fn init(
        io: std.Io,
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
    ) !AudioEncoder {
        var params = config_mod.AudioEncoderParams{};

        // 从 GGUF 元数据读取参数
        if (gguf_file.getU32("clip.audio.embedding_length")) |v| params.n_embd = v;
        if (gguf_file.getU32("clip.audio.attention.head_count")) |v| params.n_head = v;
        if (gguf_file.getU32("clip.audio.block_count")) |v| params.n_layer = v;
        if (gguf_file.getU32("clip.audio.feed_forward_length")) |v| params.n_ff = v;
        if (gguf_file.getU32("gemma4.audio.sample_rate")) |v| params.sample_rate = v;
        if (gguf_file.getU32("clip.audio.num_mel_bins")) |v| params.n_mel_bins = v;
        if (gguf_file.getF32("clip.audio.attention.layer_norm_epsilon")) |v| params.norm_eps = v;
        params.d_head = params.n_embd / params.n_head;

        log.info("Loading audio encoder: embd={d}, heads={d}, d_head={d}, layers={d}, ff={d}, mel_bins={d}, norm_eps={e}", .{
            params.n_embd, params.n_head, params.d_head, params.n_layer, params.n_ff, params.n_mel_bins, params.norm_eps,
        });

        // 加载子采样卷积权重
        var sscp_conv_w: [2]?*ggml.Tensor = .{ null, null };
        var sscp_conv_b: [2]?*ggml.Tensor = .{ null, null };
        var sscp_norm_w: [2]?*ggml.Tensor = .{ null, null };
        for (0..2) |i| {
            var buf: [64]u8 = undefined;
            const conv_name = try std.fmt.bufPrint(&buf, "a.conv1d.{d}.weight", .{i});
            sscp_conv_w[i] = findTensorInGGUF(ctx, gguf_file, conv_name) catch null;

            const bias_name = try std.fmt.bufPrint(&buf, "a.conv1d.{d}.bias", .{i});
            sscp_conv_b[i] = findTensorInGGUF(ctx, gguf_file, bias_name) catch null;

            const norm_name = try std.fmt.bufPrint(&buf, "a.conv1d.{d}.norm.weight", .{i});
            sscp_norm_w[i] = findTensorInGGUF(ctx, gguf_file, norm_name) catch null;
        }

        // 子采样输入投影
        const sscp_inp_proj_w = findTensorInGGUF(ctx, gguf_file, "a.input_projection.weight") catch null;
        const sscp_inp_proj_b = findTensorInGGUF(ctx, gguf_file, "a.input_projection.bias") catch null;

        // 输出投影
        const audio_out_proj_w = findTensorInGGUF(ctx, gguf_file, "a.pre_encode.out.weight") catch null;
        const audio_out_proj_b = findTensorInGGUF(ctx, gguf_file, "a.pre_encode.out.bias") catch null;
        const mm_soft_emb_norm_w = findTensorInGGUF(ctx, gguf_file, "mm.a.soft_emb_norm.weight") catch null;
        const mm_input_proj_w = findTensorInGGUF(ctx, gguf_file, "mm.a.input_projection.weight") catch null;

        // Load Conformer layer weights
        // Detect n_layer from actual tensors rather than metadata
        var actual_n_layer: u32 = 0;
        for (0..64) |il| {
            var buf: [32]u8 = undefined;
            const test_name = try std.fmt.bufPrint(&buf, "a.blk.{d}.attn_q.weight", .{il});
            if (gguf_file.findTensor(test_name) == null) break;
            actual_n_layer = @intCast(il + 1);
        }
        if (actual_n_layer == 0) actual_n_layer = params.n_layer;
        if (actual_n_layer != params.n_layer) {
            log.info("Audio encoder layers: metadata={d}, actual={d}", .{ params.n_layer, actual_n_layer });
            params.n_layer = actual_n_layer;
        }

        const n_layer: usize = @intCast(params.n_layer);
        var layers = try allocator.alloc(ConformerLayerWeights, n_layer);

        for (0..n_layer) |il| {
            const prefix = try std.fmt.allocPrint(allocator, "a.blk.{d}", .{il});
            defer allocator.free(prefix);
            layers[il] = loadConformerLayer(ctx, gguf_file, prefix) catch |err| {
                log.err("Failed to load conformer layer {d}: {}", .{ il, err });
                return err;
            };
        }

        // 日志：子采样卷积权重
        for (0..2) |i| {
            if (sscp_conv_w[i]) |t| {
                log.info("  conv1d.{d}.weight: shape=[{d},{d},{d}], name={s}", .{ i, t.ne()[0], t.ne()[1], t.ne()[2], t.getName() });
            } else {
                log.warn("  conv1d.{d}.weight: NOT FOUND", .{i});
            }
        }
        // 日志：输入投影
        if (sscp_inp_proj_w) |t| {
            log.info("  input_projection: shape=[{d},{d}], name={s}", .{ t.ne()[0], t.ne()[1], t.getName() });
        } else {
            log.warn("  input_projection: NOT FOUND (a.input_projection.weight)", .{});
        }
        // 日志：输出投影矩阵信息
        if (audio_out_proj_w) |t| {
            log.info("  audio_out_proj:  shape=[{d},{d}], name={s}", .{ t.ne()[0], t.ne()[1], t.getName() });
        } else {
            log.warn("  audio_out_proj:  NOT FOUND (a.pre_encode.out.weight)", .{});
        }
        if (mm_soft_emb_norm_w) |t| {
            log.info("  mm_soft_emb_norm: shape=[{d}], name={s}", .{ t.ne()[0], t.getName() });
        } else {
            log.warn("  mm_soft_emb_norm: NOT FOUND (mm.a.soft_emb_norm.weight)", .{});
        }
        if (mm_input_proj_w) |t| {
            log.info("  mm_input_proj:   shape=[{d},{d}], name={s}", .{ t.ne()[0], t.ne()[1], t.getName() });
        } else {
            log.warn("  mm_input_proj:   NOT FOUND (mm.a.input_projection.weight)", .{});
        }

        log.info("Audio encoder loaded: {d} layers, subsampling convs ready", .{n_layer});

        // === DEBUG: 保存子采样卷积权重数据 ===
        for (0..2) |i| {
            if (sscp_conv_w[i]) |t| {
                const fname = try std.fmt.allocPrint(allocator, "zllama_audio_conv1d_{d}_weight.json", .{i});
                defer allocator.free(fname);
                helper.mtmdDebugSaveData(io, "debug_audio", fname, "conv1d_weight", t.dataF32()) catch |err| {
                    log.debug("Failed to save conv1d.{d}.weight debug data: {}", .{ i, err });
                };
            }
            if (sscp_conv_b[i]) |t| {
                const fname = try std.fmt.allocPrint(allocator, "zllama_audio_conv1d_{d}_bias.json", .{i});
                defer allocator.free(fname);
                helper.mtmdDebugSaveData(io, "debug_audio", fname, "conv1d_bias", t.dataF32()) catch |err| {
                    log.debug("Failed to save conv1d.{d}.bias debug data: {}", .{ i, err });
                };
            }
        }
        if (sscp_inp_proj_w) |t| {
            helper.mtmdDebugSaveData(io, "debug_audio", "zllama_audio_input_proj_weight.json", "input_proj_weight", t.dataF32()) catch |err| {
                log.debug("Failed to save input_proj.weight debug data: {}", .{err});
            };
        }

        return AudioEncoder{
            .params = params,
            .weights = .{
                .params = params,
                .sscp_conv_w = sscp_conv_w,
                .sscp_conv_b = sscp_conv_b,
                .sscp_norm_w = sscp_norm_w,
                .sscp_inp_proj_w = sscp_inp_proj_w,
                .sscp_inp_proj_b = sscp_inp_proj_b,
                .layers = layers,
                .audio_out_proj_w = audio_out_proj_w,
                .audio_out_proj_b = audio_out_proj_b,
                .mm_soft_emb_norm_w = mm_soft_emb_norm_w,
                .mm_input_proj_w = mm_input_proj_w,
            },
            .ctx_weights = ctx,
        };
    }

    // 编码音频数据，返回嵌入 tokens
    // @param io I/O 实例
    // @param ctx ggml 计算上下文
    // @param graph 计算图
    // @param mel_data PCM F32 音频样本 [n_mel_bins, n_frames]
    // @returns 音频嵌入 [n_output_embd, n_tokens]
    pub fn encode(
        self: *AudioEncoder,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        mel_data: []const f32,
        n_mel_bins: u32,
        n_frames: u32,
    ) !*ggml.Tensor {
        const w = self.weights;
        const p = self.params;
        // llama.cpp hardcodes 1e-6 for gemma4a (see clip.cpp PROJECTOR_TYPE_GEMMA4A case)
        // due to a mistake in the original conversion code, rms_norm_eps is set to a wrong value
        // since all gemma4a models use 1e-6, we just hardcode it here
        const norm_eps: f32 = 1e-6;
        const res_weight: f32 = 0.5;

        log.info("encode: n_mel_bins={d}, n_frames={d}, mel_data.len={d}", .{ n_mel_bins, n_frames, mel_data.len });
        log.info("encode: n_mel_bins={d}, n_frames={d}, mel_data.len={d}", .{ n_mel_bins, n_frames, mel_data.len });

        // 1. Create input tensor matching llama.cpp layout exactly
        //    llama.cpp: inp = build_inp_raw(1) with shape [n_frames, n_mel, 1, 1] (mel-major)
        //      ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx(), img.ny(), channels, n_batch)
        //      where img.nx() = n_frames, img.ny() = n_mel
        //      → ne[0]=n_frames, ne[1]=n_mel, ne[2]=1, ne[3]=1
        //    then: cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp))
        //      → [n_mel, n_frames, 1, 1] (frame-major in memory)
        //
        //    mel_data layout (from pipeline): mel-major, data[mel_bin * n_frames + frame]
        //    When copied into tensor [n_frames, n_mel] (ne[0]=n_frames varies fastest):
        //      tensor[frame + n_frames * mel_bin] = data[mel_bin * n_frames + frame]
        //      This is exactly mel-major layout, so data can be copied directly.
        //    After transpose + cont: tensor becomes [n_mel, n_frames] with frame-major layout.
        var inp_raw = try ctx.newTensor4d(ggml.Type.f32,
            @intCast(n_frames),  // ne[0] = n_frames (varies fastest)
            @intCast(n_mel_bins), // ne[1] = n_mel
            1,                    // ne[2] = channels
            1);                   // ne[3] = n_batch
        inp_raw.setName("inp_raw");

        // Copy mel data directly (mel-major layout matches ne[0]=n_frames)
        {
            const raw = inp_raw.dataBytes();
            const dst = @as([*]f32, @ptrCast(@alignCast(raw.ptr)));
            const n_mel: usize = @intCast(n_mel_bins);
            const n_fr: usize = @intCast(n_frames);
            // mel_data is mel-major: data[mel_bin * n_frames + frame]
            // tensor[frame + n_frames * mel_bin] = data[mel_bin * n_frames + frame]
            // Direct copy, no transpose needed
            @memcpy(dst[0..mel_data.len], mel_data);
            _ = n_mel;
            _ = n_fr;
        }

        // Transpose + cont to get frame-major layout [n_mel, n_frames, 1, 1]
        // This matches llama.cpp: cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp))
        var cur = ggml.transpose(ctx, inp_raw);
        cur.setName("debug_audio_encoder_input_transposed");
        cur = ggml.cont(ctx, cur);
        cur.setName("debug_audio_encoder_input");

        try helper.mtmdDebugSaveData(io, "debug_audio", "zllama_audio_encoder_input.json",
            "debug_audio_encoder_input",
            cur.dataF32());

        // 2. 子采样 Conv2D (2层，每层 stride=2, padding=1)
        for (0..2) |i| {
            if (w.sscp_conv_w[i]) |conv_w_raw| {
                // conv1d weight from GGUF is 4D [KH, KW, IC, OC] (ne[0]=KH, ne[1]=KW, ne[2]=IC, ne[3]=OC)
                // ggml_conv_2d expects 4D kernel [OC, IC, KH, KW] (ne[0]=KW, ne[1]=KH, ne[2]=IC, ne[3]=OC)
                // When KH==KW (both 3), the data layout matches ggml_conv_2d's expectation
                // because ne[0]=KH=3=KW and ne[1]=KW=3=KH, so no permute needed.
                // llama.cpp passes model.sscp_conv_w[i] directly without permute.
                const ne = conv_w_raw.ne();
                log.debug("conv1d.{d}.weight: ne=[{d},{d},{d},{d}], name={s}", .{ i, ne[0], ne[1], ne[2], ne[3], conv_w_raw.getName() });
                log.info("conv1d.{d}.weight: ne=[{d},{d},{d},{d}], name={s}", .{ i, ne[0], ne[1], ne[2], ne[3], conv_w_raw.getName() });
                // Use kernel directly without permute, matching llama.cpp behavior
                cur = cur.conv2d(ctx, conv_w_raw, 2, 2, 1, 1, 1, 1);

                // === DEBUG: save raw conv2d output (before bias/norm/relu) ===
                if (i == 0) {
                    self.debug_conv2d_0_raw = cur;
                }

                if (w.sscp_conv_b[i]) |conv_b| {
                    cur = cur.add(ctx, conv_b);
                }
                // LayerNorm: permute to [C, H, W, N] for norm on C axis
                if (w.sscp_norm_w[i]) |norm_w| {
                    cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
                    cur = cur.norm(ctx, norm_eps);
                    cur = cur.mul(ctx, norm_w);
                    cur = cur.permute(ctx, 2, 0, 1, 3).cont(ctx);
                }

                cur = cur.relu(ctx);

                // === DEBUG: save after relu (matches llama.cpp ggml_set_name("conv2d_0_out") after relu) ===
                if (i == 0) {
                    self.debug_conv2d_0_output = cur;
                }

                // === DEBUG: save conv2d_1 after relu ===
                if (i == 1) {
                    self.debug_conv2d_1_output = cur;
                }
            }
        }

        // Flatten: [freq, time, channels, 1] -> [channels*freq, time]
        cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
        const flat_dim0 = cur.ne()[0] * cur.ne()[1];
        cur = cur.reshape2d(ctx, flat_dim0, cur.ne()[2]);

        // === DEBUG: 保存 flatten 输出 ===
        self.debug_flatten_output = cur;

        // Input projection: map conv output dim -> Conformer embedding dim
        if (w.sscp_inp_proj_w) |proj_w| {
            cur = proj_w.mulMat(ctx, cur);
            if (w.sscp_inp_proj_b) |proj_b| {
                cur = cur.add(ctx, proj_b);
            }
        }

        // === DEBUG: 保存 input_proj 输出 ===
        self.debug_input_proj_output = cur;

        // 输入投影
        const n_pos = cur.ne()[1];

        // Chunked local attention parameters (matching llama.cpp gemma4a.cpp)
        const C: i64 = 12; // chunk_size
        const P: i64 = 12; // max_past_horizon (context_left - 1)
        const S: i64 = C + P; // context_size = 24
        const R: i64 = P + 1; // RPE positions = 13
        const B: i64 = @divTrunc((n_pos + C - 1), C); // num_blocks
        const Np: i64 = B * C; // padded sequence length
        const pad_seq: i64 = Np - n_pos;

        log.info("encode: n_pos={d}, C={d}, P={d}, S={d}, R={d}, B={d}, Np={d}, pad_seq={d}", .{ n_pos, C, P, S, R, B, Np, pad_seq });

        // Create input tensors for RPE and mask (filled with data, matching C++ set_input)
        // Ensure allocations are enabled before creating tensors that we'll write into
        ctx.setNoAlloc(false);
        const pos_emb = try ctx.newTensor2d(ggml.Type.f32, p.n_embd, @intCast(R));
        pos_emb.setName("pos_emb");
        fillSinusoidalPosEmb(pos_emb, @intCast(R), @intCast(p.n_embd), @intCast(P));

        // === DEBUG: 保存 pos_emb 数据 ===
        {
            helper.mtmdDebugSaveData(io, "debug_audio", "zllama_audio_pos_emb.json", "pos_emb", pos_emb.dataF32()) catch |err| {
                log.debug("Failed to save pos_emb debug data: {}", .{err});
            };
        }

        // Create 4D mask [S, C, B, 1] so it can broadcast to [S, C, B, H] scores
        const kq_mask = try ctx.newTensor4d(ggml.Type.f32, @intCast(S), @intCast(C), @intCast(B), 1);
        kq_mask.setName("kq_mask");
        fillChunkedAttentionMask(kq_mask, @intCast(S), @intCast(C), @intCast(B), @intCast(P), @intCast(n_pos));

        // === DEBUG: 保存 kq_mask 数据 ===
        {
            helper.mtmdDebugSaveData(io, "debug_audio", "zllama_audio_attn_mask.json", "kq_mask", kq_mask.dataF32()) catch |err| {
                log.debug("Failed to save kq_mask debug data: {}", .{err});
            };
        }

        ctx.setNoAlloc(true);

        // 3. Conformer Blocks
        for (w.layers, 0..) |*layer, il| {
            _ = il;
            var residual = cur;

            // FFN 1 (half-step)
            if (layer.ff_norm_w != null and layer.ff_up_w != null and layer.ff_down_w != null) {
                cur = rmsNorm(ctx, residual, layer.ff_norm_w.?, norm_eps);
                cur = ffnSilu(ctx, cur, layer.ff_up_w.?, layer.ff_down_w.?);
                if (layer.ff_post_norm_w) |post_norm| {
                    cur = rmsNorm(ctx, cur, post_norm, norm_eps);
                }
                residual = residual.add(ctx, cur.scale(ctx, res_weight));
            }

            // Chunked local self-attention with RPE
            if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
                const q_scale: f32 = (1.0 / @sqrt(@as(f32, @floatFromInt(p.d_head)))) / @log2(2.0);
                const k_scale: f32 = @log2(1.0 + @exp(1.0));
                const softcap: f32 = 50.0;

                const attn_norm = if (layer.attn_pre_norm_w) |w2| w2 else layer.ln_1_w;
                const attn_in = if (attn_norm) |norm_w|
                    rmsNorm(ctx, residual, norm_w, norm_eps)
                else
                    residual;

                // Q, K, V 投影
                var Qcur = layer.q_w.?.mulMat(ctx, attn_in);
                var Kcur = layer.k_w.?.mulMat(ctx, attn_in);
                var Vcur = layer.v_w.?.mulMat(ctx, attn_in);

                const d_head_i: i64 = @intCast(p.d_head);
                const n_head_i: i64 = @intCast(p.n_head);
                const n_pos_i: i64 = n_pos;

                // Reshape: [n_embd, n_pos] -> [d_head, n_head, n_pos]
                Qcur = Qcur.reshape3d(ctx, d_head_i, n_head_i, n_pos_i);
                Kcur = Kcur.reshape3d(ctx, d_head_i, n_head_i, n_pos_i);
                Vcur = Vcur.reshape3d(ctx, d_head_i, n_head_i, n_pos_i);

                // Q/K scaling
                Qcur = Qcur.scale(ctx, q_scale);
                if (layer.per_dim_scale_w) |ps| {
                    Qcur = Qcur.mul(ctx, ps.reshape3d(ctx, d_head_i, 1, 1));
                }
                Kcur = Kcur.scale(ctx, k_scale);
                if (layer.per_dim_k_scale_w) |pks| {
                    Kcur = Kcur.mul(ctx, pks.reshape3d(ctx, d_head_i, 1, 1));
                }

                // Q blocking: [D, H, N] -> pad to Np -> reshape [D, H, C, B]
                Qcur = Qcur.pad(ctx, 0, 0, @as(i32, @intCast(pad_seq)), 0); // [D, H, Np]
                Qcur = Qcur.reshape4d(ctx, d_head_i, n_head_i, C, B); // [D, H, C, B]
                // llama.cpp: permute(0,3,1,2) -> [D, C, B, H]
                Qcur = Qcur.permute(ctx, 0, 3, 1, 2).cont(ctx); // [D, C, B, H]

                // K/V block context extraction via overlapping view
                const pad_kv: i64 = S * B - n_pos_i;
                Kcur = Kcur.pad(ctx, 0, 0, @as(i32, @intCast(pad_kv)), 0); // [D, H, S*B]
                Kcur = Kcur.roll(ctx, 0, 0, P, 0); // left-pad by P
                Kcur = Kcur.cont(ctx); // materialize roll
                // Overlapping view: stride for B dim is C positions, not S
                Kcur = Kcur.view4d(ctx, d_head_i, n_head_i, S, B, Kcur.nb()[1], Kcur.nb()[2], @as(usize, @intCast(C)) * Kcur.nb()[2], 0);
                Kcur = Kcur.cont(ctx); // materialize overlapping windows
                // llama.cpp: permute(0,3,1,2): ne[0]=D, ne[3]=H, ne[1]=S, ne[2]=B → [D,S,B,H]
                var Kblk = Kcur.permute(ctx, 0, 3, 1, 2).cont(ctx); // [D, S, B, H]

                Vcur = Vcur.pad(ctx, 0, 0, @as(i32, @intCast(pad_kv)), 0);
                Vcur = Vcur.roll(ctx, 0, 0, P, 0);
                Vcur = Vcur.cont(ctx);
                Vcur = Vcur.view4d(ctx, d_head_i, n_head_i, S, B, Vcur.nb()[1], Vcur.nb()[2], @as(usize, @intCast(C)) * Vcur.nb()[2], 0);
                Vcur = Vcur.cont(ctx);
                // llama.cpp: permute(1,3,0,2): ne[1]=D, ne[3]=H, ne[0]=S, ne[2]=B → [S,D,B,H]
                var Vblk = Vcur.permute(ctx, 1, 3, 0, 2).cont(ctx); // [S, D, B, H]

                // Content attention: Kblk=[D,S,B,H] @ Qcur=[D,C,B,H] → contracts on D → [S, C, B, H]
                var scores = Kblk.mulMat(ctx, Qcur); // [S, C, B, H]

                // Relative position attention
                if (layer.attn_k_rel_w) |k_rel| {
                    // RPE: k_rel=[n_embd,n_embd] @ pos_emb=[D,R] → [n_embd,R] → reshape [D,H,R]
                    var p_rpe = k_rel.mulMat(ctx, pos_emb);
                    p_rpe = p_rpe.reshape3d(ctx, d_head_i, n_head_i, R);
                    // llama.cpp: permute(0,2,1,3) on [D,H,R] → [D,R,H]
                    p_rpe = p_rpe.permute(ctx, 0, 2, 1, 3).cont(ctx); // [D, R, H]

                    // Q_flat @ RPE^T: Qcur=[D,C,B,H] → flatten to [D, C*B, H]
                    // p_rpe=[D,R,H] @ Q_flat=[D,C*B,H] → [R, C*B, H]
                    const Q_flat = Qcur.reshape3d(ctx, d_head_i, C * B, n_head_i);
                    var matrix_bd = p_rpe.mulMat(ctx, Q_flat); // [R, C*B, H]
                    // Reshape to [R, C, B, H] (note: H is the last dim, matching llama.cpp)
                    matrix_bd = matrix_bd.reshape4d(ctx, R, C, B, n_head_i); // [R, C, B, H]

                    // Blocked relative shift (appendix B of Transformer-XL): produce [S, C, B, H]
                    matrix_bd = matrix_bd.pad(ctx, S + 1 - R, 0, 0, 0); // [S+1, C, B, H]
                    matrix_bd = matrix_bd.reshape3d(ctx, (S + 1) * C, B, n_head_i);
                    matrix_bd = matrix_bd.view3d(ctx, C * S, B, n_head_i, matrix_bd.nb()[1], matrix_bd.nb()[2], 0);
                    matrix_bd = matrix_bd.cont(ctx); // [C*S, B, H]
                    matrix_bd = matrix_bd.reshape4d(ctx, S, C, B, n_head_i); // [S, C, B, H] matches scores

                    scores = scores.add(ctx, matrix_bd);
                }

                // Softcap
                scores = scores.scale(ctx, 1.0 / softcap);
                scores = scores.tanh(ctx);
                scores = scores.scale(ctx, softcap);

                // Blocked attention mask: [S, C, B] broadcasts to [S, C, B, H]
                scores = scores.add(ctx, kq_mask);

                const attn = scores.softMax(ctx); // [S, C, B, H]

                // llama.cpp: V.mulMat(attn): V=[S,D,B,H], attn=[S,C,B,H] → [D, C, B, H]
                var x = Vblk.mulMat(ctx, attn); // [D, C, B, H]

                // llama.cpp: output permute(0,2,3,1): [D,C,B,H] → [D,H,C,B]
                x = x.permute(ctx, 0, 2, 3, 1).cont(ctx); // [D, H, C, B]
                x = x.cont2d(ctx, d_head_i * n_head_i, C * B); // [D*H, C*B]
                if (pad_seq > 0) {
                    x = x.view2d(ctx, d_head_i * n_head_i, n_pos_i, x.nb()[1], 0);
                    x = x.cont(ctx);
                }

                x = layer.o_w.?.mulMat(ctx, x);
                if (layer.o_b) |ob| {
                    x = x.add(ctx, ob);
                }
                if (layer.attn_post_norm_w) |post_norm| {
                    x = rmsNorm(ctx, x, post_norm, norm_eps);
                }
                residual = residual.add(ctx, x);
            }

            // Convolution Module (GLU + depthwise conv)
            if (layer.norm_conv_w != null and layer.conv_pw1_w != null and
                layer.conv_dw_w != null and layer.conv_pw2_w != null)
            {
                cur = rmsNorm(ctx, residual, layer.norm_conv_w.?, norm_eps);
                var x_conv = layer.conv_pw1_w.?.mulMat(ctx, cur);

                // GLU gate (sigmoid)
                const d_gate = @divExact(x_conv.ne()[0], 2);
                const gate = x_conv.view2d(ctx, d_gate, x_conv.ne()[1], x_conv.nb()[1], @as(usize, @intCast(@sizeOf(f32) * d_gate))).cont(ctx).sigmoid(ctx);
                const act = x_conv.view2d(ctx, d_gate, x_conv.ne()[1], x_conv.nb()[1], 0);
                x_conv = act.mul(ctx, gate);
                x_conv = x_conv.cont(ctx).permute(ctx, 1, 0, 2, 3).cont(ctx);

                // Causal depthwise conv1d via ssm_conv
                x_conv = x_conv.pad(ctx, 4, 0, 0, 0);
                x_conv = x_conv.roll(ctx, 4, 0, 0, 0);
                x_conv = x_conv.ssmConv(ctx, layer.conv_dw_w.?);
                if (layer.conv_dw_b) |dw_b| {
                    x_conv = x_conv.add(ctx, dw_b);
                }

                if (layer.conv_norm_w) |cn_w| {
                    x_conv = x_conv.rmsNorm(ctx, norm_eps);
                    x_conv = x_conv.mul(ctx, cn_w);
                }
                x_conv = x_conv.silu(ctx);
                x_conv = layer.conv_pw2_w.?.mulMat(ctx, x_conv);
                residual = residual.add(ctx, x_conv);
            }

            // FFN 2 (half-step)
            if (layer.ff_norm_1_w != null and layer.ff_up_1_w != null and layer.ff_down_1_w != null) {
                cur = rmsNorm(ctx, residual, layer.ff_norm_1_w.?, norm_eps);
                cur = ffnSilu(ctx, cur, layer.ff_up_1_w.?, layer.ff_down_1_w.?);
                if (layer.ff_post_norm_1_w) |post_norm| {
                    cur = rmsNorm(ctx, cur, post_norm, norm_eps);
                }
                residual = residual.add(ctx, cur.scale(ctx, res_weight));
            }

            // Layer output norm
            cur = if (layer.ln_2_w) |ln2|
                rmsNorm(ctx, residual, ln2, norm_eps)
            else
                residual;
        }

        // 4. 输出投影
        if (w.audio_out_proj_w) |out_w| {
            cur = out_w.mulMat(ctx, cur);
            if (w.audio_out_proj_b) |out_b| {
                cur = cur.add(ctx, out_b);
            }
        }

        // 5. 多模态嵌入器
        cur = cur.rmsNorm(ctx, norm_eps);
        if (w.mm_soft_emb_norm_w) |sn_w| {
            cur = cur.mul(ctx, sn_w);
        }
        if (w.mm_input_proj_w) |proj_w| {
            cur = proj_w.mulMat(ctx, cur);
        }

        cgraph.buildForwardExpand(cur);
        return cur;
    }

    // 返回音频编码器是否可用（权重已加载）
    pub fn isAvailable(self: *const AudioEncoder) bool {
        return self.weights.sscp_conv_w[0] != null;
    }

    // 估算编码后的 token 数量
    // Gemma 4 E2B 的 Conformer 使用 2 层步长为 2 的子采样，
    // 因此每 4 帧 mel 特征产生 1 个输出 token
    pub fn estimateOutputTokens(self: *const AudioEncoder, audio_length_sec: f32) u32 {
        const n_samples: u32 = @intFromFloat(@as(f32, @floatFromInt(self.params.sample_rate)) * audio_length_sec);
        // 使用 framing.computeFrameCount 复用分帧计数逻辑
        const fc = framing.computeFrameCount(n_samples, .{
            .frame_length = config_mod.DEFAULT_FRAME_LENGTH,
            .hop_length = config_mod.DEFAULT_HOP_LENGTH,
            .n_fft = config_mod.DEFAULT_N_FFT,
        });
        // 匹配 llama.cpp: 裁剪到 PyTorch 帧数
        const pad_left: u32 = config_mod.DEFAULT_FRAME_LENGTH / 2;
        const n_with_left: u32 = n_samples + pad_left;
        const pt_frames: u32 = if (n_with_left >= config_mod.DEFAULT_FRAME_LENGTH + 1)
            @as(u32, @intCast((n_with_left - (config_mod.DEFAULT_FRAME_LENGTH + 1)) / config_mod.DEFAULT_HOP_LENGTH)) + 1
        else
            0;
        const actual_frames = @min(fc.n_frames, pt_frames);
        return actual_frames / 4;
    }

    // 保存中间张量的调试数据（需在 graph.compute() 之后调用）
    // 所有文件保存到 debug_audio/ 子目录
    pub fn saveDebugData(self: *const AudioEncoder, io: std.Io) void {
        const subdir = "debug_audio";
        if (self.debug_conv2d_0_output) |t| {
            helper.mtmdDebugSaveData(io, subdir, "zllama_audio_conv2d_0_output.json", "conv2d_0_output", t.dataF32()) catch |err| {
                log.debug("Failed to save conv2d_0_output debug data: {}", .{err});
            };
        }
        if (self.debug_conv2d_1_output) |t| {
            helper.mtmdDebugSaveData(io, subdir, "zllama_audio_conv2d_1_output.json", "conv2d_1_output", t.dataF32()) catch |err| {
                log.debug("Failed to save conv2d_1_output debug data: {}", .{err});
            };
        }
        if (self.debug_input_proj_output) |t| {
            helper.mtmdDebugSaveData(io, subdir, "zllama_audio_input_proj_output.json", "input_proj_output", t.dataF32()) catch |err| {
                log.debug("Failed to save input_proj_output debug data: {}", .{err});
            };
        }

        if (self.debug_flatten_output) | t | {
            helper.mtmdDebugSaveData(io, subdir, "zllama_audio_flatten_output.json", "flatten_output", t.dataF32()) catch |err| {
                log.debug("Failed to save input_proj_output debug data: {}", .{err});
            };
        }
    }

    pub fn deinit(self: *AudioEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.weights.layers);
    }
};

// ============================================================================
// 辅助函数
// ============================================================================

fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name);
}

// 加载单个 Conformer 层
fn loadConformerLayer(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
) !ConformerLayerWeights {
    var layer = ConformerLayerWeights{};

    // 标准 attention 权重
    layer.q_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_q.weight");
    layer.k_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_k.weight");
    layer.v_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_v.weight");
    layer.o_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_out.weight");
    layer.o_b = findLayerWeight(ctx, gguf_file, prefix, "attn_out.bias") catch null;
    layer.ln_1_w = findLayerWeight(ctx, gguf_file, prefix, "ln1.weight") catch null;

    // FFN 1
    layer.ff_norm_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight") catch null;
    layer.ff_up_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight");
    layer.ff_down_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight");

    // Conformer-specific
    layer.attn_pre_norm_w = findLayerWeight(ctx, gguf_file, prefix, "attn_pre_norm.weight") catch null;
    layer.attn_post_norm_w = findLayerWeight(ctx, gguf_file, prefix, "attn_post_norm.weight") catch null;
    layer.per_dim_scale_w = findLayerWeight(ctx, gguf_file, prefix, "per_dim_scale.weight") catch null;
    layer.per_dim_k_scale_w = findLayerWeight(ctx, gguf_file, prefix, "per_dim_k_scale.weight") catch null;
    layer.attn_k_rel_w = findLayerWeight(ctx, gguf_file, prefix, "attn_k_rel.weight") catch null;
    layer.ff_post_norm_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_post_norm.weight") catch null;

    // Convolution module
    layer.norm_conv_w = findLayerWeight(ctx, gguf_file, prefix, "norm_conv.weight") catch null;
    layer.conv_pw1_w = findLayerWeight(ctx, gguf_file, prefix, "conv_pw1.weight") catch null;
    layer.conv_dw_w = findLayerWeight(ctx, gguf_file, prefix, "conv_dw.weight") catch null;
    layer.conv_dw_b = findLayerWeight(ctx, gguf_file, prefix, "conv_dw.bias") catch null;
    layer.conv_norm_w = findLayerWeight(ctx, gguf_file, prefix, "conv_norm.weight") catch null;
    layer.conv_pw2_w = findLayerWeight(ctx, gguf_file, prefix, "conv_pw2.weight") catch null;

    // FFN 2
    layer.ff_norm_1_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_norm_1.weight") catch null;
    layer.ff_up_1_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_up_1.weight") catch null;
    layer.ff_down_1_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_down_1.weight") catch null;
    layer.ff_post_norm_1_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_post_norm_1.weight") catch null;

    // Layer output
    layer.ln_2_w = findLayerWeight(ctx, gguf_file, prefix, "ln2.weight") catch null;

    return layer;
}

fn findLayerWeight(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
    name: []const u8,
) !*ggml.Tensor {
    return weight_loader.loadLayerWeight(ctx, gguf_file, prefix, name);
}

// RMS 归一化
fn rmsNorm(ctx: *ggml.Context, x: *ggml.Tensor, weight: *ggml.Tensor, eps: f32) *ggml.Tensor {
    return x.rmsNorm(ctx, eps).mul(ctx, weight);
}

// SiLU FFN: down(up(x) * silu(gate(x)))
fn ffnSilu(ctx: *ggml.Context, x: *ggml.Tensor, up_w: *ggml.Tensor, down_w: *ggml.Tensor) *ggml.Tensor {
    const h = up_w.mulMat(ctx, x);
    return down_w.mulMat(ctx, h.silu(ctx));
}

// Fill a 2D tensor [n_embd, n_pos] with sinusoidal position encodings.
// Used for Relative Position Encoding (RPE) in Conformer attention.
// Matches llama.cpp clip.cpp: positions are in DESCENDING order (max_past, max_past-1, ..., 0).
fn fillSinusoidalPosEmb(tensor: *ggml.Tensor, n_pos: usize, n_embd: usize, max_past: usize) void {
    const data = tensor.dataF32();
    const num_timescales = n_embd / 2;
    const log_timescale_increment = @log(10000.0) / @max(@as(f32, @floatFromInt(num_timescales - 1)), 1.0);

    for (0..n_pos) |p| {
        // Position in DESCENDING order: max_past, max_past-1, ..., 0
        const position: f32 = @floatFromInt(max_past - p);
        for (0..num_timescales) |i| {
            const inv_ts: f32 = @exp(-@as(f32, @floatFromInt(i)) * log_timescale_increment);
            const scaled: f32 = position * inv_ts;
            data[p * n_embd + i] = @sin(scaled);
            data[p * n_embd + i + num_timescales] = @cos(scaled);
        }
    }
}

// Fill a 3D attention mask tensor [S, C, B] for chunked self-attention.
// Matches llama.cpp clip.cpp lines 4244-4263.
fn fillChunkedAttentionMask(
    tensor: *ggml.Tensor,
    n_ctx: usize,
    chunk_size: usize,
    n_blocks: usize,
    past: usize,
    n_pos: usize,
) void {
    const data = tensor.dataF32();
    const neg_val: f32 = -1e9; // Use -1e9 instead of -inf, matching C++ code
    const S = n_ctx;
    const C = chunk_size;

    for (0..n_blocks) |b| {
        const bC: i64 = @intCast(b * C);
        for (0..C) |cc| {
            const gq: i64 = @intCast(b * C + cc); // global query position
            for (0..S) |s| {
                const s_i64: i64 = @intCast(s);
                const gk: i64 = s_i64 + bC - @as(i64, @intCast(past)); // global key position
                const idx = s + cc * S + b * S * C;
                // Condition matching C++: gq < n_pos && gk >= 0 && gk < n_pos && gk <= gq && (gq - gk) < past
                if (gq < n_pos and gk >= 0 and gk < n_pos and gk <= gq and (gq - gk) < past) {
                    data[idx] = 0.0;
                } else {
                    data[idx] = neg_val;
                }
            }
        }
    }
}
