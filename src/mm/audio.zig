//! 音频编码器模块
//!
//! 提供对 Gemma 4 E2B 内建 Conformer 音频编码器的支持。
//! 该编码器能够直接处理最长 30 秒的 16kHz 单声道音频输入，
//! 将原始 PCM 音频转换为模型可理解的音频嵌入 tokens。
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

const log = std.log.scoped(.audio_encoder);

// ============================================================================
// 音频编码器超参数
// ============================================================================

pub const AudioEncoderParams = struct {
    /// 输入特征维度（mel bins）
    n_mel_bins: u32 = 128,
    /// 模型嵌入维度
    n_embd: u32 = 512,
    /// 注意力头数
    n_head: u32 = 8,
    /// 每头维度
    d_head: u32 = 64,
    /// Conformer 层数
    n_layer: u32 = 16,
    /// FFN 中间维度
    n_ff: u32 = 2048,
    /// 输出投影维度（匹配 LLM 嵌入维度）
    n_output_embd: u32 = 2560,
    /// 音频采样率
    sample_rate: u32 = 16000,
    /// 最大音频长度（秒）
    max_audio_length_sec: f32 = 30.0,
    /// 归一化 epsilon
    norm_eps: f32 = 1e-6,
};

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
    params: AudioEncoderParams,

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
    params: AudioEncoderParams,
    weights: AudioEncoderWeights,
    ctx_weights: *ggml.Context,

    /// 从 GGUF 文件初始化音频编码器，加载所有权重到 ggml context
    pub fn init(
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
    ) !AudioEncoder {
        var params = AudioEncoderParams{};

        // 从 GGUF 元数据读取参数
        if (gguf_file.getU32("clip.audio.embedding_length")) |v| params.n_embd = v;
        if (gguf_file.getU32("clip.audio.attention.head_count")) |v| params.n_head = v;
        if (gguf_file.getU32("clip.audio.block_count")) |v| params.n_layer = v;
        if (gguf_file.getU32("clip.audio.feed_forward_length")) |v| params.n_ff = v;
        if (gguf_file.getU32("gemma4.audio.sample_rate")) |v| params.sample_rate = v;
        params.d_head = params.n_embd / params.n_head;

        log.info("Loading audio encoder: embd={d}, heads={d}, d_head={d}, layers={d}, ff={d}", .{
            params.n_embd, params.n_head, params.d_head, params.n_layer, params.n_ff,
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

        log.info("Audio encoder loaded: {d} layers, subsampling convs ready", .{n_layer});

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

    /// 编码音频数据，返回嵌入 tokens
    /// @param ctx ggml 计算上下文
    /// @param graph 计算图
    /// @param audio_data PCM F32 音频样本 [n_mel_bins, n_frames]
    /// @returns 音频嵌入 [n_output_embd, n_tokens]
    pub fn encode(
        self: *const AudioEncoder,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        mel_data: []const f32,
        n_mel_bins: u32,
        n_frames: u32,
    ) !*ggml.Tensor {
        const w = self.weights;
        const p = self.params;
        const norm_eps = p.norm_eps;
        const res_weight: f32 = 0.5;

        // 1. 创建输入张量并填充实际 Mel 数据
        var cur = try ctx.newTensor4d(ggml.Type.f32, @intCast(n_frames), @intCast(n_mel_bins), 1, 1);
        cur.setName("audio_mel_input");

        const raw_bytes = cur.dataBytes();
        const mel_byte_len = mel_data.len * @sizeOf(f32);
        @memcpy(raw_bytes[0..mel_byte_len], @as([*]const u8, @ptrCast(mel_data.ptr))[0..mel_byte_len]);

        // 转置: [n_mel_bins, n_frames] -> [n_frames, n_mel_bins, 1, 1]
        // 以匹配 Conv2D 的 [H, W, C, N] 布局
        cur = ggml.cont(ctx, ggml.transpose(ctx, cur));

        // 2. 子采样 Conv2D (2层，每层 stride=2, padding=1)
        for (0..2) |i| {
            if (w.sscp_conv_w[i]) |conv_w| {
                // Conv2D: [OH, OW, OC, N] = conv2d(input, kernel, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w)
                cur = cur.conv2d(ctx, conv_w, 2, 2, 1, 1, 1, 1);
                if (w.sscp_conv_b[i]) |conv_b| {
                    cur = cur.add(ctx, conv_b);
                }
                // LayerNorm: permute to [N, H, W, C] -> [N, C, H, W] for norm on C axis
                if (w.sscp_norm_w[i]) |norm_w| {
                    cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
                    cur = cur.norm(ctx, norm_eps);
                    cur = cur.mul(ctx, norm_w);
                    cur = cur.permute(ctx, 2, 0, 1, 3).cont(ctx);
                }
                cur = cur.relu(ctx);
            }
        }

        // Flatten: [freq, time, channels, 1] -> [channels*freq, time]
        cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
        const flat_dim0 = cur.ne()[0] * cur.ne()[1];
        cur = cur.reshape2d(ctx, flat_dim0, cur.ne()[2]);

        // 输入投影
        if (w.sscp_inp_proj_w) |proj_w| {
            cur = proj_w.mulMat(ctx, cur);
            if (w.sscp_inp_proj_b) |proj_b| {
                cur = cur.add(ctx, proj_b);
            }
        }

        const n_pos = cur.ne()[1];

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

                // Chunked attention parameters
                const C: i64 = 12; // chunk_size
                const P: i64 = 12; // max_past_horizon
                const S: i64 = C + P; // context_size = 24
                const B: i64 = @divTrunc((n_pos_i + C - 1), C); // num_blocks
                const Np: i64 = B * C; // padded sequence length
                const pad_seq: i64 = Np - n_pos_i;
                const R: i64 = P + 1; // RPE positions

                // Q blocking: pad to Np, reshape to [D, H, C, B], then permute
                Qcur = Qcur.pad(ctx, 0, 0, @as(i32, @intCast(pad_seq)), 0);
                Qcur = Qcur.reshape4d(ctx, d_head_i, n_head_i, C, B);
                Qcur = Qcur.permute(ctx, 0, 3, 1, 2).cont(ctx); // [D, C, B, H]

                // K/V: extract overlapping blocks
                const pad_kv: i64 = S * B - n_pos_i;
                Kcur = Kcur.pad(ctx, 0, 0, @as(i32, @intCast(pad_kv)), 0); // [D, H, S*B]
                Kcur = Kcur.roll(ctx, 0, 0, P, 0); // left-pad by P
                Kcur = Kcur.cont(ctx);
                Kcur = Kcur.view4d(ctx, d_head_i, n_head_i, S, B, Kcur.nb()[1], Kcur.nb()[2], @as(usize, @intCast(C)) * Kcur.nb()[2], 0);
                Kcur = Kcur.cont(ctx);
                var Kblk = Kcur.permute(ctx, 0, 3, 1, 2).cont(ctx); // [D, S, B, H]

                Vcur = Vcur.pad(ctx, 0, 0, @as(i32, @intCast(pad_kv)), 0);
                Vcur = Vcur.roll(ctx, 0, 0, P, 0);
                Vcur = Vcur.cont(ctx);
                Vcur = Vcur.view4d(ctx, d_head_i, n_head_i, S, B, Vcur.nb()[1], Vcur.nb()[2], @as(usize, @intCast(C)) * Vcur.nb()[2], 0);
                Vcur = Vcur.cont(ctx);
                var Vblk = Vcur.permute(ctx, 1, 3, 0, 2).cont(ctx); // [S, D, B, H]

                // Content attention: Q @ K^T
                var scores = Kblk.mulMat(ctx, Qcur); // [S, C, B, H]

                // Relative position attention
                if (layer.attn_k_rel_w) |k_rel| {
                    // Create position embedding input
                    var pos_emb = try ctx.newTensor2d(ggml.Type.f32, d_head_i * n_head_i, R);
                    pos_emb.setName("pos_emb");
                    // RPE projection
                    var p_rpe = k_rel.mulMat(ctx, pos_emb); // [n_embd, R]
                    p_rpe = p_rpe.reshape3d(ctx, d_head_i, n_head_i, R);
                    p_rpe = p_rpe.permute(ctx, 0, 2, 1, 3).cont(ctx); // [D, R, H]

                    // Q_flat @ RPE^T
                    const Q_flat = Qcur.reshape3d(ctx, d_head_i, C * B, n_head_i);
                    var matrix_bd = p_rpe.mulMat(ctx, Q_flat); // [R, C*B, H]
                    matrix_bd = matrix_bd.reshape4d(ctx, R, C, B, n_head_i); // [R, C, B, H]

                    // Blocked relative shift
                    matrix_bd = matrix_bd.pad(ctx, S + 1 - R, 0, 0, 0);
                    matrix_bd = matrix_bd.reshape3d(ctx, (S + 1) * C, B, n_head_i);
                    matrix_bd = matrix_bd.view3d(ctx, C * S, B, n_head_i, matrix_bd.nb()[1], matrix_bd.nb()[2], 0);
                    matrix_bd = matrix_bd.cont(ctx);
                    matrix_bd = matrix_bd.reshape4d(ctx, S, C, B, n_head_i);

                    scores = scores.add(ctx, matrix_bd);
                }

                // Softcap + mask + softmax
                scores = scores.scale(ctx, 1.0 / softcap);
                scores = scores.tanh(ctx);
                scores = scores.scale(ctx, softcap);

                // Create attention mask [S, C, B]
                var kq_mask = try ctx.newTensor3d(ggml.Type.f32, S, C, B);
                kq_mask.setName("kq_mask");
                scores = scores.add(ctx, kq_mask);
                const attn = scores.softMax(ctx);

                // attn @ V
                var x = Vblk.mulMat(ctx, attn); // [D, C, B, H]
                x = x.permute(ctx, 0, 2, 3, 1).cont(ctx); // [D, H, C, B]
                x = x.cont2d(ctx, d_head_i * n_head_i, C * B);
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

    /// 返回音频编码器是否可用（权重已加载）
    pub fn isAvailable(self: *const AudioEncoder) bool {
        return self.weights.sscp_conv_w[0] != null;
    }

    /// 估算编码后的 token 数量
    /// Gemma 4 E2B 的 Conformer 使用 2 层步长为 2 的子采样，
    /// 因此每 4 帧 mel 特征产生 1 个输出 token
    pub fn estimateOutputTokens(self: *const AudioEncoder, audio_length_sec: f32) u32 {
        const n_frames: u32 = @intFromFloat(@as(f32, @floatFromInt(self.params.sample_rate)) * audio_length_sec / 160.0);
        return n_frames / 4;
    }

    pub fn deinit(self: *AudioEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.weights.layers);
    }
};

// ============================================================================
// 辅助函数
// ============================================================================

/// 从 GGUF 查找张量并加载到 ggml context
fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    const info = gguf_file.findTensor(name) orelse return error.TensorNotFound;
    const n_dims = info.n_dims;
    const typ: ggml.Type = @enumFromInt(@intFromEnum(info.data_type));

    ctx.setNoAlloc(false);
    const tensor = switch (n_dims) {
        1 => try ctx.newTensor1d(typ, @intCast(info.dims[0])),
        2 => try ctx.newTensor2d(typ, @intCast(info.dims[0]), @intCast(info.dims[1])),
        3 => try ctx.newTensor3d(typ, @intCast(info.dims[0]), @intCast(info.dims[1]), @intCast(info.dims[2])),
        4 => try ctx.newTensor4d(typ, @intCast(info.dims[0]), @intCast(info.dims[1]), @intCast(info.dims[2]), @intCast(info.dims[3])),
        else => return error.UnsupportedTensorDims,
    };
    ctx.setNoAlloc(true);

    tensor.setName(@ptrCast(name));

    const tensor_data = gguf_file.getTensorData(info);
    @memcpy(tensor.dataBytes(), tensor_data);

    return tensor;
}

/// 加载单个 Conformer 层
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

/// 查找层权重（带前缀）
fn findLayerWeight(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
    name: []const u8,
) !*ggml.Tensor {
    var buf: [256]u8 = undefined;
    const full_name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, name });
    return findTensorInGGUF(ctx, gguf_file, full_name);
}

/// RMS 归一化
fn rmsNorm(ctx: *ggml.Context, x: *ggml.Tensor, weight: *ggml.Tensor, eps: f32) *ggml.Tensor {
    return x.rmsNorm(ctx, eps).mul(ctx, weight);
}

/// SiLU FFN: down(up(x) * silu(gate(x)))
fn ffnSilu(ctx: *ggml.Context, x: *ggml.Tensor, up_w: *ggml.Tensor, down_w: *ggml.Tensor) *ggml.Tensor {
    const h = up_w.mulMat(ctx, x);
    return down_w.mulMat(ctx, h.silu(ctx));
}
