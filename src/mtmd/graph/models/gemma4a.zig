//! Gemma4A 音频编码器 — 模型特定实现
//!
//! 实现 Gemma4A Conformer 音频编码器的：
//! - 权重结构定义（AudioEncoderWeights）
//! - 权重加载（loadWeights）
//! - 计算图构建（buildGraph / buildGraphEx）
//! - Token 估算（estimateOutputTokens）
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/gemma4a.cpp
//!
//! 本模块提供两种调用方式：
//! 1. buildGraph() — 接受 GraphBuilder（用于 vision 管线兼容）
//! 2. buildGraphEx() — 接受独立参数（用于 audio encoder 直接调用）

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("../mod.zig");
const weight_loader = @import("weight_loader");

const debug_mod = @import("debug");

const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const VisionHParams = graph.VisionHParams;
const ViTLayerWeights = graph.ViTLayerWeights;
const ClampInfo = graph.ClampInfo;

const log = std.log.scoped(.graph_model_gemma4a);

/// 音频编码器权重接口（与 VisionEncoderWeights 兼容的字段子集）
/// 供 buildGraphEx 使用，避免直接依赖 VisionEncoderWeights
pub const AudioWeights = struct {
    sscp_conv_w: [2]?*ggml.Tensor,
    sscp_conv_b: [2]?*ggml.Tensor,
    sscp_norm_w: [2]?*ggml.Tensor,
    sscp_inp_proj_w: ?*ggml.Tensor,
    sscp_inp_proj_b: ?*ggml.Tensor,
    layers: []const ViTLayerWeights,
    audio_out_proj_w: ?*ggml.Tensor,
    audio_out_proj_b: ?*ggml.Tensor,
    mm_soft_emb_norm_w: ?*ggml.Tensor,
    mm_input_proj_w: ?*ggml.Tensor,
};

/// 音频编码器超参数接口
pub const AudioParams = struct {
    n_embd: u32,
    n_head: u32,
    n_layer: u32,
    d_head: u32,
};

/// 从 VisionEncoderWeights 构建 AudioWeights
pub fn weightsFromVision(w: *const VisionEncoderWeights) AudioWeights {
    return .{
        .sscp_conv_w = w.sscp_conv_w,
        .sscp_conv_b = w.sscp_conv_b,
        .sscp_norm_w = w.sscp_norm_w,
        .sscp_inp_proj_w = w.sscp_inp_proj_w,
        .sscp_inp_proj_b = w.sscp_inp_proj_b,
        .layers = w.layers,
        .audio_out_proj_w = w.audio_out_proj_w,
        .audio_out_proj_b = w.audio_out_proj_b,
        .mm_soft_emb_norm_w = w.mm_soft_emb_norm_w,
        .mm_input_proj_w = w.mm_input_proj_w,
    };
}

/// 从 VisionHParams 构建 AudioParams
pub fn paramsFromVision(p: *const VisionHParams) AudioParams {
    return .{
        .n_embd = p.n_embd,
        .n_head = p.n_head,
        .n_layer = p.n_layer,
        .d_head = if (p.n_head > 0) p.n_embd / p.n_head else 64,
    };
}

/// 构建 Gemma4A 完整计算图（GraphBuilder 版本）
pub fn buildGraph(
    builder: *GraphBuilder,
    mel_tensor: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) !*ggml.CGraph {
    const audio_w = weightsFromVision(builder.weights);
    const audio_p = paramsFromVision(builder.hparams);
    return buildGraphEx(builder.ctx0, builder.gf, &audio_w, &audio_p, mel_tensor, clamp_map);
}

/// 构建 Gemma4A 完整计算图（独立参数版本）
/// 可被 audio encoder 直接调用，无需 GraphBuilder
///
/// 处理流程:
///   1. 子采样 Conv2D（2 层，每层 stride=2, padding=1）
///   2. 输入投影到 Conformer 嵌入维度
///   3. Conformer blocks（FFN1 → Chunked Attention → Conv Module → FFN2）
///   4. 输出投影
///   5. 多模态嵌入器（RMSNorm + 线性投影）
///
/// 参考: llama.cpp gemma4a.cpp build()
pub fn buildGraphEx(
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const AudioWeights,
    p: *const AudioParams,
    mel_tensor: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) !*ggml.CGraph {
    const norm_eps: f32 = 1e-6;
    const res_weight: f32 = 0.5;

    const n_frames: i64 = mel_tensor.ne()[0];
    const n_mel_bins: i64 = mel_tensor.ne()[1];

    log.info("Gemma4A graph: frames={d}, mel_bins={d}, embd={d}, heads={d}, layers={d}", .{ n_frames, n_mel_bins, p.n_embd, p.n_head, p.n_layer });

    // 1. 输入张量已是 4D [n_frames, n_mel_bins, 1, 1]（由 melToTensor 创建）
    //    参考: clip.cpp build_inp_raw() → ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, nx, ny, C, B)
    var cur = mel_tensor;

    // Transpose to frame-major layout [n_mel, n_frames, 1, 1]
    cur = ggml.transpose(ctx, cur);
    cur = ggml.cont(ctx, cur);
    cur.setName("debug_audio_04_encoder_input");
    ggml.setOutput(cur);

    // 2. Subsampling Conv2D (2 layers, stride=2, padding=1)
    for (0..2) |i| {
        if (w.sscp_conv_w[i]) |conv_w| {
            cur = cur.conv2d(ctx, conv_w, 2, 2, 1, 1, 1, 1);

            if (w.sscp_conv_b[i]) |conv_b| {
                cur = cur.add(ctx, conv_b);
            }

            // LayerNorm
            if (w.sscp_norm_w[i]) |norm_w| {
                cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
                cur = cur.norm(ctx, norm_eps);
                cur = cur.mul(ctx, norm_w);
                cur = cur.permute(ctx, 2, 0, 1, 3).cont(ctx);
            }

            cur = cur.relu(ctx);

            // Debug output for each conv layer (matching C++)
            if (i == 0) {
                cur.setName("debug_audio_conv2d_0_output");
                ggml.setOutput(cur);
            } else {
                cur.setName("debug_audio_conv2d_1_output");
                ggml.setOutput(cur);
            }
        }
    }

    // 3. Flatten: [freq, time, ch, 1] -> [ch*freq, time]
    cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
    cur.setName("debug_audio_after_cont");
    ggml.setOutput(cur);

    const flat_dim0 = cur.ne()[0] * cur.ne()[1];
    cur = cur.reshape2d(ctx, flat_dim0, cur.ne()[2]);
    cur.setName("debug_audio_flatten_output");
    ggml.setOutput(cur);

    log.debug("sscp_inp_proj_w={any}, sscp_inp_proj_b={any}", .{ w.sscp_inp_proj_w, w.sscp_inp_proj_b });
    // 4. Input projection to Conformer embedding dim
    if (w.sscp_inp_proj_w) |proj_w| {
        cur = buildMMWithClamp(ctx, proj_w, cur, clamp_map);
        if (w.sscp_inp_proj_b) |proj_b| {
            cur = cur.add(ctx, proj_b);
        }
    }
    cur.setName("debug_audio_input_proj_output");
    ggml.setOutput(cur);

    const n_pos = cur.ne()[1];

    // 5. Chunked local attention parameters
    const C: i64 = 12; // chunk_size
    const P: i64 = 12; // max_past_horizon
    const S: i64 = C + P; // context_size = 24
    const R: i64 = P + 1; // RPE positions = 13
    const B: i64 = @divTrunc((n_pos + C - 1), C); // num_blocks
    const Np: i64 = B * C; // padded sequence length
    const pad_seq: i64 = Np - n_pos;

    log.debug("  chunked attn: n_pos={d} C={d}, P={d}, S={d}, R={d}, B={d}, Np={d}, pad={d}", .{ n_pos, C, P, S, R, B, Np, pad_seq });

    // Create RPE and mask tensors
    // C++: ggml_set_input(pos_emb) and ggml_set_input(kq_mask) — caller fills them.
    // Zig: We fill them inline during graph construction for simplicity.
    // Save and restore caller's no_alloc state to avoid side effects.
    const prev_no_alloc = ctx.getNoAlloc();
    ctx.setNoAlloc(false);
    const pos_emb = try ctx.newTensor2d(ggml.Type.f32, p.n_embd, R);
    pos_emb.setName("pos_emb");
    ggml.setInput(pos_emb);
    fillSinusoidalPosEmb(pos_emb, @intCast(R), @intCast(p.n_embd), @intCast(P));

    const kq_mask = try ctx.newTensor3d(ggml.Type.f32, S, C, B);
    kq_mask.setName("kq_mask");
    ggml.setInput(kq_mask);
    fillChunkedAttentionMask(kq_mask, @intCast(S), @intCast(C), @intCast(B), @intCast(P), @intCast(n_pos));
    ctx.setNoAlloc(prev_no_alloc);

    // 6. Conformer Blocks
    for (w.layers, 0..) |*layer, il| {
        var residual = cur;

        // FFN 1 (half-step) — with clamp, matching C++ clip_graph_gemma4a::build_mm
        if (layer.ff_norm_w != null and layer.ff_up_w != null and layer.ff_down_w != null) {
            cur = try graph.buildNorm(ctx, residual, layer.ff_norm_w.?, null, .rms_norm, norm_eps, "blk");
            cur = buildFFNWithClamp(ctx, layer.ff_up_w.?, null, null, null, layer.ff_down_w.?, null, .silu, cur, clamp_map);
            if (layer.ff_post_norm_w) |post_norm| {
                cur = try graph.buildNorm(ctx, cur, post_norm, null, .rms_norm, norm_eps, "blk");
            }
            residual = residual.add(ctx, cur.scale(ctx, res_weight));
        }
        if (il == 0) {
            residual.setName("debug_audio_half_step_1_output");
            ggml.setOutput(residual);
        }

        // Chunked local self-attention with RPE
        if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
            const q_scale: f32 = (1.0 / @sqrt(@as(f32, @floatFromInt(p.d_head)))) / @log(2.0);
            const k_scale: f32 = @log2(1.0 + @exp(1.0));
            const softcap: f32 = 50.0;
            const attn_norm = if (layer.attn_pre_norm_w) |w2| w2 else layer.ln_1_w;
            const attn_in = if (attn_norm) |norm_w|
                try graph.buildNorm(ctx, residual, norm_w, null, .rms_norm, norm_eps, "blk")
            else
                residual;

            // Q, K, V projections
            var Qcur = buildMMWithClamp(ctx, layer.q_w.?, attn_in, clamp_map);
            var Kcur = buildMMWithClamp(ctx, layer.k_w.?, attn_in, clamp_map);
            var Vcur = buildMMWithClamp(ctx, layer.v_w.?, attn_in, clamp_map);

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
            Qcur = Qcur.pad(ctx, 0, 0, @as(i32, @intCast(pad_seq)), 0);
            Qcur = Qcur.reshape4d(ctx, d_head_i, n_head_i, C, B);
            Qcur = Qcur.permute(ctx, 0, 3, 1, 2).cont(ctx); // [D, C, B, H]

            // K/V block context extraction
            var Kblk = extractBlocks(ctx, Kcur, d_head_i, n_head_i, S, B, C, P, n_pos_i);
            Kblk = Kblk.permute(ctx, 0, 3, 1, 2).cont(ctx); // [D, S, B, H]

            var Vblk = extractBlocks(ctx, Vcur, d_head_i, n_head_i, S, B, C, P, n_pos_i);
            Vblk = Vblk.permute(ctx, 1, 3, 0, 2).cont(ctx); // [S, D, B, H]

            // Content attention
            var scores = Kblk.mulMat(ctx, Qcur); // [S, C, B, H]

            // Relative position attention
            // NOTE: C++ uses pure ggml_mul_mat for attn_k_rel_w (no clamp)
            if (layer.attn_k_rel_w) |k_rel| {
                var p_rpe = k_rel.mulMat(ctx, pos_emb);
                p_rpe = p_rpe.reshape3d(ctx, d_head_i, n_head_i, R);
                p_rpe = p_rpe.permute(ctx, 0, 2, 1, 3).cont(ctx); // [D, R, H]

                const Q_flat = Qcur.reshape3d(ctx, d_head_i, C * B, n_head_i);
                var matrix_bd = p_rpe.mulMat(ctx, Q_flat); // [R, C*B, H]
                matrix_bd = matrix_bd.reshape4d(ctx, R, C, B, n_head_i);

                // Blocked relative shift
                matrix_bd = matrix_bd.pad(ctx, S + 1 - R, 0, 0, 0);
                matrix_bd = matrix_bd.reshape3d(ctx, (S + 1) * C, B, n_head_i);
                matrix_bd = matrix_bd.view3d(ctx, C * S, B, n_head_i, matrix_bd.nb()[1], matrix_bd.nb()[2], 0);
                matrix_bd = matrix_bd.cont(ctx);
                matrix_bd = matrix_bd.reshape4d(ctx, S, C, B, n_head_i);

                scores = scores.add(ctx, matrix_bd);
            }

            // Softcap
            scores = scores.scale(ctx, 1.0 / softcap);
            scores = scores.tanh(ctx);
            scores = scores.scale(ctx, softcap);

            // Mask
            // C++: scores = ggml_add(ctx0, scores, kq_mask) — 3D [S,C,B] broadcasts over H
            scores = scores.add(ctx, kq_mask);

            const attn = scores.softMax(ctx);

            // Attention output
            var x = Vblk.mulMat(ctx, attn); // [D, C, B, H]
            x = x.permute(ctx, 0, 2, 3, 1).cont(ctx); // [D, H, C, B]
            x = x.cont2d(ctx, d_head_i * n_head_i, C * B);
            if (pad_seq > 0) {
                x = x.view2d(ctx, d_head_i * n_head_i, n_pos_i, x.nb()[1], 0);
                x = x.cont(ctx);
            }

            // Output projection
            x = buildMMWithClamp(ctx, layer.o_w.?, x, clamp_map);
            if (layer.o_b) |ob| {
                x = x.add(ctx, ob);
            }
            if (layer.attn_post_norm_w) |post_norm| {
                x = try graph.buildNorm(ctx, x, post_norm, null, .rms_norm, norm_eps, "blk");
            }
            residual = residual.add(ctx, x);
        }
        if (il == 0) {
            residual.setName("debug_audio_self_attention_with_RPE_output");
            ggml.setOutput(residual);
        }

        // Convolution Module (GLU + depthwise conv)
        // conv_pw1: [n_embd, n_pos] -> [intermediate, n_pos] (intermediate=2*n_embd for GLU)
        // GLU gate: split intermediate into two halves, gate = sigmoid(half1), act = half2
        // Depthwise conv1d: causal conv on time dimension
        // conv_pw2: [n_embd, n_pos] -> [n_embd, n_pos] (project back to model dim)
        //
        // Weight shapes (from GGUF):
        //   conv_pw1.weight: [1024, 2048]  (n_embd=1024, intermediate=2048)
        //   conv_dw.weight:  [5, 1024]      (kernel_size=5, channels=1024)
        //   conv_pw2.weight: [1024, 1024]   (back to n_embd)
        //
        // Reference: llama.cpp gemma4a.cpp build() — conv block (lines 264-296)
        if (layer.norm_conv_w != null and layer.conv_pw1_w != null and
            layer.conv_dw_w != null and layer.conv_pw2_w != null)
        {
            cur = try graph.buildNorm(ctx, residual, layer.norm_conv_w.?, null, .rms_norm, norm_eps, "blk");
            var x_conv = buildMMWithClamp(ctx, layer.conv_pw1_w.?, cur, clamp_map);
            // x_conv shape: [intermediate=2048, n_pos]

            if (il == 0) {
                cur.setName("debug_audio_conv_build_normal_output");
                ggml.setOutput(cur);
            }
            // GLU gate: split intermediate dim in half
            // GLU
            // {
            //     int64_t d = x->ne[0] / 2;
            //     ggml_tensor * gate = ggml_sigmoid(ctx0,
            //         ggml_cont(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0])));
            //     x = ggml_mul(ctx0,
            //         ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
            //     x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            // }
            {
                const d_gate = @divExact(x_conv.ne()[0], 2);
                var gate = x_conv.view2d(ctx, d_gate, x_conv.ne()[1], x_conv.nb()[1], x_conv.nb()[0] * @as(usize, @intCast(d_gate)));
                gate = gate.cont(ctx).sigmoid(ctx);
                const act = x_conv.view2d(ctx, d_gate, x_conv.ne()[1], x_conv.nb()[1], 0);
                x_conv = act.mul(ctx, gate);
                // x_conv shape: [d_gate=1024, n_pos]
                // Transpose to [n_pos, d_gate] for causal depthwise conv1d along time axis
                // C++: x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
                x_conv = ggml.cont(ctx, ggml.transpose(ctx, x_conv));
                // x_conv shape: [n_pos, 1024]
            }

            if (il == 0) {
                x_conv.setName("debug_audio_conv_glu_output");
                ggml.setOutput(x_conv);
            }

            // Causal depthwise Conv1D via ggml_ssm_conv (pad+roll for left-only padding).
            // C++ (gemma4a.cpp:278-283):
            //   x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            //   x = ggml_roll(ctx0, x, 4, 0, 0, 0);
            //   x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            // NOTE: gemma4a.cpp uses only ONE pad(4) before ssm_conv (unlike conformer.cpp
            // which uses two pads). With kernel_size=5 and input padded to n_pos+4,
            // ssm_conv output has n_pos+4-5+1 = n_pos time steps, matching the residual.
            x_conv = x_conv.pad(ctx, 4, 0, 0, 0);
            x_conv = x_conv.roll(ctx, 4, 0, 0, 0);
            x_conv = x_conv.ssmConv(ctx, layer.conv_dw_w.?);
            if (layer.conv_dw_b) |dw_b| {
                x_conv = x_conv.add(ctx, dw_b);
            }

            if (il == 0) {
                x_conv.setName("debug_audio_conv_dw_output");
                ggml.setOutput(x_conv);
            }

            // C++ (gemma4a.cpp:286-288):
            //   if (layer.conv_norm_w) {
            //       x = ggml_rms_norm(ctx0, x, norm_eps);
            //       x = ggml_mul(ctx0, x, layer.conv_norm_w);
            //   }
            // NOTE: gemma4a.cpp uses RMSNorm (not affine transform like conformer.cpp)
            if (layer.conv_norm_w) |cn_w| {
                x_conv = x_conv.rmsNorm(ctx, norm_eps);
                x_conv = x_conv.mul(ctx, cn_w);
            }
            x_conv = x_conv.silu(ctx);
            // x_conv shape after ssmConv: [d_inner=1024, n_t=n_pos, 1] = [1024, n_pos]
            // This is already in [d_gate, n_pos] layout, no need to transpose back

            if (il == 0) {
                x_conv.setName("debug_audio_conv_dw_norm_silu_output");
                ggml.setOutput(x_conv);
            }

            // conv_pw2: project back to n_embd (1024 -> 1024)
            // C++ (gemma4a.cpp:291): x = build_mm(layer.conv_pw2_w, x);
            // NOTE: gemma4a.cpp does NOT add conv_pw2_b here (unlike conformer.cpp)
            x_conv = buildMMWithClamp(ctx, layer.conv_pw2_w.?, x_conv, clamp_map);
            // x_conv shape: [1024, n_pos] = [n_embd, n_pos] — matches residual

            residual = residual.add(ctx, x_conv);
        }
        if (il == 0) {
            residual.setName("debug_audio_convolution_output");
            ggml.setOutput(residual);
        }

        // FFN 2 (half-step) — with clamp, matching C++ clip_graph_gemma4a::build_mm
        if (layer.ff_norm_1_w != null and layer.ff_up_1_w != null and layer.ff_down_1_w != null) {
            cur = try graph.buildNorm(ctx, residual, layer.ff_norm_1_w.?, null, .rms_norm, norm_eps, "blk");
            cur = buildFFNWithClamp(ctx, layer.ff_up_1_w.?, null, null, null, layer.ff_down_1_w.?, null, .silu, cur, clamp_map);
            if (layer.ff_post_norm_1_w) |post_norm| {
                cur = try graph.buildNorm(ctx, cur, post_norm, null, .rms_norm, norm_eps, "blk");
            }
            residual = residual.add(ctx, cur.scale(ctx, res_weight));
        }
        if (il == 0) {
            residual.setName("debug_audio_half_step_2_output");
            ggml.setOutput(residual);
        }

        // Layer output norm
        cur = if (layer.ln_2_w) |ln2|
            try graph.buildNorm(ctx, residual, ln2, null, .rms_norm, norm_eps, "blk")
        else
            residual;

        if (il == 0 and layer.ln_2_w != null) {
            cur.setName("debug_audio_layer_0_norm_output");
            ggml.setOutput(cur);
        }
    }

    // Always name and mark conformer blocks output (matching C++)
    cur.setName("debug_audio_conformer_blocks_output");
    ggml.setOutput(cur);

    // 7. Output projection
    if (w.audio_out_proj_w) |out_w| {
        cur = buildMMWithClamp(ctx, out_w, cur, clamp_map);
        if (w.audio_out_proj_b) |out_b| {
            cur = cur.add(ctx, out_b);
        }
    }

    // 8. Multimodal embedder
    cur = cur.rmsNorm(ctx, norm_eps);
    cur.setName("mm_norm");
    ggml.setOutput(cur);
    if (w.mm_soft_emb_norm_w) |sn_w| {
        cur = cur.mul(ctx, sn_w);
        cur.setName("mm_norm_scaled");
        ggml.setOutput(cur);
    }
    if (w.mm_input_proj_w) |proj_w| {
        cur = buildMMWithClamp(ctx, proj_w, cur, clamp_map);
        cur.setName("mm_proj");
        ggml.setOutput(cur);
    }

    gf.buildForwardExpand(cur);

    log.info("Gemma4A graph built successfully", .{});
    return gf;
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 带 clamp 的矩阵乘法
fn buildMMWithClamp(
    ctx: *ggml.Context,
    w: *ggml.Tensor,
    x: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) *ggml.Tensor {
    const name = w.getName();
    if (clamp_map.get(name)) |ci| {
        log.debug("[gemma4a] ggml_clamp('{s}'): min=[{}, {}] max=[{}, {}]", .{ name, ci.inp_min, ci.inp_max, ci.out_min, ci.out_max });
        const clamped = x.clamp(ctx, ci.inp_min, ci.inp_max);
        var out = w.mulMat(ctx, clamped);
        out = out.clamp(ctx, ci.out_min, ci.out_max);
        return out;
    } else {
        log.debug("[gemma4a] no clamp info for '{s}'", .{name});
        return w.mulMat(ctx, x);
    }
}

/// 带 clamp 的 FFN 构建
/// 对应 C++ clip_graph_gemma4a::build_ffn() — 使用 build_mm（带 clamp）进行所有权重投影
fn buildFFNWithClamp(
    ctx: *ggml.Context,
    up: *ggml.Tensor,
    up_b: ?*ggml.Tensor,
    gate: ?*ggml.Tensor,
    gate_b: ?*ggml.Tensor,
    down: *ggml.Tensor,
    down_b: ?*ggml.Tensor,
    type_op: FFNOpType,
    cur: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) *ggml.Tensor {
    // Up projection (with clamp)
    var tmp = buildMMWithClamp(ctx, up, cur, clamp_map);
    if (up_b) |b| {
        tmp = tmp.add(ctx, b);
    }

    // Gate projection (optional, for GLU variants)
    var x: *ggml.Tensor = undefined;
    if (gate) |g| {
        x = buildMMWithClamp(ctx, g, cur, clamp_map);
        if (gate_b) |gb| {
            x = x.add(ctx, gb);
        }
    } else {
        x = tmp;
    }

    // Activation: when gate exists, use fused GLU split op;
    // when no gate, apply plain activation.
    // Reference: clip_graph::build_ffn() in deps/llama.cpp/tools/mtmd/clip.cpp
    const activated = if (gate != null) blk: {
        break :blk switch (type_op) {
            .silu => x.swigluSplit(ctx, tmp),
            .gelu => x.gegluSplit(ctx, tmp),
            .gelu_erf => x.gegluErfSplit(ctx, tmp),
            .gelu_quick => x.gegluQuickSplit(ctx, tmp),
            .relu_sqr => blk2: {
                // llama.cpp: cur = ggml_relu(ctx0, cur); cur = ggml_sqr(ctx0, cur);
                // 其中 cur 是 gate 分支的输出（如果 gate 存在）或 up 分支的输出
                // 注意：与 llama.cpp 一致，relu_sqr 总是 relu(x) * relu(x)，
                // 无论 gate 是否存在（即对 gate 分支或 up 分支的输出做 relu 平方）
                const relu = x.relu(ctx);
                break :blk2 relu.mul(ctx, relu);
            },
        };
    } else blk: {
        break :blk switch (type_op) {
            .silu => x.silu(ctx),
            .gelu => x.gelu(ctx),
            .gelu_erf => x.geluErf(ctx),
            .gelu_quick => x.geluQuick(ctx),
            .relu_sqr => blk2: {
                const relu = x.relu(ctx);
                break :blk2 relu.mul(ctx, relu);
            },
        };
    };

    // Down projection (with clamp)
    var result = buildMMWithClamp(ctx, down, activated, clamp_map);
    if (down_b) |b| {
        result = result.add(ctx, b);
    }

    return result;
}

/// K/V block context extraction via overlapping view
fn extractBlocks(
    ctx: *ggml.Context,
    t: *ggml.Tensor,
    d_head: i64,
    n_head: i64,
    S: i64,
    B: i64,
    C: i64,
    P: i64,
    n_pos: i64,
) *ggml.Tensor {
    const pad_kv: i64 = S * B - n_pos;
    var result = t.pad(ctx, 0, 0, @as(i32, @intCast(pad_kv)), 0);
    result = result.roll(ctx, 0, 0, @as(i32, @intCast(P)), 0);
    result = result.cont(ctx);
    result = result.view4d(ctx, d_head, n_head, S, B, result.nb()[1], result.nb()[2], @as(usize, @intCast(C)) * result.nb()[2], 0);
    result = result.cont(ctx);
    return result;
}

/// Fill sinusoidal position embeddings (descending order)
fn fillSinusoidalPosEmb(tensor: *ggml.Tensor, n_pos: usize, n_embd: usize, max_past: usize) void {
    const n_elems = @as(usize, @intCast(tensor.nElems()));
    const data = std.heap.page_allocator.alloc(f32, n_elems) catch return;
    defer std.heap.page_allocator.free(data);
    const num_timescales = n_embd / 2;
    const log_timescale_increment = @log(10000.0) / @max(@as(f32, @floatFromInt(num_timescales -| 1)), 1.0);

    for (0..n_pos) |p| {
        const position: f32 = @floatFromInt(max_past - p);
        for (0..num_timescales) |i| {
            const inv_ts: f32 = @exp(-@as(f32, @floatFromInt(i)) * log_timescale_increment);
            const scaled: f32 = position * inv_ts;
            data[p * n_embd + i] = @sin(scaled);
            data[p * n_embd + i + num_timescales] = @cos(scaled);
        }
    }
    log.debug("pos_emb: loop={d}, n_embd={d}, max_past={d} [0-4]=[{d}, {d}, {d}, {d}]", .{ n_pos, n_embd, max_past, data[0], data[1], data[2], data[3] });
    tensor.dataSet(f32, data) catch {};
}

/// Fill chunked attention mask
fn fillChunkedAttentionMask(
    tensor: *ggml.Tensor,
    n_ctx: usize,
    chunk_size: usize,
    n_blocks: usize,
    past: usize,
    n_pos: usize,
) void {
    const n_elems = @as(usize, @intCast(tensor.nElems()));
    const data = std.heap.page_allocator.alloc(f32, n_elems) catch return;
    defer std.heap.page_allocator.free(data);
    const neg_val: f32 = -1e9;
    const S = n_ctx;
    const C = chunk_size;

    for (0..n_blocks) |b| {
        const bC: i64 = @intCast(b * C);
        for (0..C) |cc| {
            const gq: i64 = @intCast(b * C + cc);
            for (0..S) |s| {
                const s_i64: i64 = @intCast(s);
                const gk: i64 = s_i64 + bC - @as(i64, @intCast(past));
                const idx = s + cc * S + b * S * C;
                if (gq < n_pos and gk >= 0 and gk < n_pos and gk <= gq and (gq - gk) < past) {
                    data[idx] = 0.0;
                } else {
                    data[idx] = neg_val;
                }
            }
        }
    }
    tensor.dataSet(f32, data) catch {};
}

// ============================================================================
// 音频编码器后端注册
// ============================================================================

/// Gemma4A 音频编码器后端实例
pub const backend = graph.AudioEncoderBackend{
    .name = "gemma4a",
    .loadParams = loadParams,
    .loadWeights = loadWeights,
    .loadClampInfo = loadClampInfo,
    .buildGraph = buildGraphFromWeights,
    .estimateOutputTokens = estimateOutputTokens,
};

/// 从 VisionEncoderWeights 构建计算图的包装函数
fn buildGraphFromWeights(
    io: std.Io,
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const VisionHParams,
    mel_tensor: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) !*ggml.CGraph {
    _ = io;
    const audio_w = weightsFromVision(w);
    const audio_p = paramsFromVision(p);
    return buildGraphEx(ctx, gf, &audio_w, &audio_p, mel_tensor, clamp_map);
}

// ============================================================================
// 权重加载（从 GGUF 加载 Gemma4A 音频编码器权重到 VisionEncoderWeights）
// ============================================================================

/// 从 GGUF 元数据读取音频编码器超参数
pub fn loadParams(io: std.Io, gguf_file: *const gguf.GGUFFile, params: *VisionHParams) void {
    _ = io;
    if (gguf_file.getU32("clip.audio.embedding_length")) |v| params.n_embd = v;
    if (gguf_file.getU32("clip.audio.attention.head_count")) |v| params.n_head = v;
    if (gguf_file.getU32("clip.audio.block_count")) |v| params.n_layer = v;
    if (gguf_file.getU32("clip.audio.feed_forward_length")) |v| params.n_ff = v;
    if (gguf_file.getU32("clip.audio.num_mel_bins")) |v| params.n_mel_bins = v;
    if (gguf_file.getF32("clip.audio.attention.layer_norm_epsilon")) |v| params.eps = v;
}

/// 从 GGUF 加载 Gemma4A 音频编码器所有权重到 VisionEncoderWeights
pub fn loadWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    w: *VisionEncoderWeights,
) !void {
    log.info("Loading Gemma4A audio encoder weights...", .{});

    // 加载子采样卷积权重
    for (0..2) |i| {
        var buf: [64]u8 = undefined;
        const conv_name = try std.fmt.bufPrint(&buf, "a.conv1d.{d}.weight", .{i});
        // conv1d.weight 是必需张量（参考 clip.cpp get_tensor 默认 required=true）
        w.sscp_conv_w[i] = (try findTensorInGGUFWithRequired(ctx, gguf_file, conv_name, true)) orelse return error.TensorNotFound;
        log.info("  conv1d.{d}.weight: shape=[{d},{d},{d}], name={s}", .{ i, w.sscp_conv_w[i].?.ne()[0], w.sscp_conv_w[i].?.ne()[1], w.sscp_conv_w[i].?.ne()[2], w.sscp_conv_w[i].?.getName() });

        const bias_name = try std.fmt.bufPrint(&buf, "a.conv1d.{d}.bias", .{i});
        w.sscp_conv_b[i] = findTensorInGGUF(ctx, gguf_file, bias_name) catch null;
        if (w.sscp_conv_b[i]) |t| {
            log.info("  conv1d.{d}.bias: shape=[{d}], name={s}", .{ i, t.ne()[0], t.getName() });
        } else {
            log.debug("  conv1d.{d}.bias: not found (optional)", .{i});
        }

        const norm_name = try std.fmt.bufPrint(&buf, "a.conv1d.{d}.norm.weight", .{i});
        w.sscp_norm_w[i] = findTensorInGGUF(ctx, gguf_file, norm_name) catch null;
        if (w.sscp_norm_w[i]) |t| {
            log.info("  conv1d.{d}.norm.weight: shape=[{d}], name={s}", .{ i, t.ne()[0], t.getName() });
        } else {
            log.debug("  conv1d.{d}.norm.weight: not found (optional)", .{i});
        }
    }

    // 子采样输入投影
    // a.input_projection.weight 是必需张量（参考 clip.cpp get_tensor 默认 required=true）
    w.sscp_inp_proj_w = (try findTensorInGGUFWithRequired(ctx, gguf_file, "a.input_projection.weight", true)) orelse return error.TensorNotFound;
    log.info("  input_projection.weight: shape=[{d},{d}], name={s}", .{ w.sscp_inp_proj_w.?.ne()[0], w.sscp_inp_proj_w.?.ne()[1], w.sscp_inp_proj_w.?.getName() });
    w.sscp_inp_proj_b = findTensorInGGUF(ctx, gguf_file, "a.input_projection.bias") catch null;
    if (w.sscp_inp_proj_b) |t| {
        log.info("  input_projection.bias: shape=[{d}], name={s}", .{ t.ne()[0], t.getName() });
    } else {
        log.debug("  input_projection.bias: not found (optional)", .{});
    }

    // 输出投影
    w.audio_out_proj_w = findTensorInGGUF(ctx, gguf_file, "a.pre_encode.out.weight") catch null;
    if (w.audio_out_proj_w) |t| {
        log.info("  audio_out_proj.weight: shape=[{d},{d}], name={s}", .{ t.ne()[0], t.ne()[1], t.getName() });
    } else {
        log.debug("  audio_out_proj.weight: not found (optional)", .{});
    }
    w.audio_out_proj_b = findTensorInGGUF(ctx, gguf_file, "a.pre_encode.out.bias") catch null;
    if (w.audio_out_proj_b) |t| {
        log.info("  audio_out_proj.bias: shape=[{d}], name={s}", .{ t.ne()[0], t.getName() });
    } else {
        log.debug("  audio_out_proj.bias: not found (optional)", .{});
    }

    // 多模态嵌入器
    w.mm_soft_emb_norm_w = findTensorInGGUF(ctx, gguf_file, "mm.a.soft_emb_norm.weight") catch null;
    if (w.mm_soft_emb_norm_w) |t| {
        log.info("  mm_soft_emb_norm.weight: shape=[{d}], name={s}", .{ t.ne()[0], t.getName() });
    } else {
        log.debug("  mm_soft_emb_norm.weight: not found (optional)", .{});
    }
    w.mm_input_proj_w = findTensorInGGUF(ctx, gguf_file, "mm.a.input_projection.weight") catch null;
    if (w.mm_input_proj_w) |t| {
        log.info("  mm_input_proj.weight: shape=[{d},{d}], name={s}", .{ t.ne()[0], t.ne()[1], t.getName() });
    } else {
        log.debug("  mm_input_proj.weight: not found (optional)", .{});
    }

    // 检测实际层数
    var actual_n_layer: u32 = 0;
    for (0..64) |il| {
        var buf: [32]u8 = undefined;
        const test_name = try std.fmt.bufPrint(&buf, "a.blk.{d}.attn_q.weight", .{il});
        if (gguf_file.findTensor(test_name) == null) break;
        actual_n_layer = @intCast(il + 1);
    }
    // n_layer 由调用者从 VisionHParams 设置，这里使用检测值覆盖
    const n_layer: usize = @intCast(actual_n_layer);
    log.info("  detected {d} conformer layers", .{n_layer});
    w.layers = try allocator.alloc(ViTLayerWeights, n_layer);

    for (0..n_layer) |il| {
        const prefix = try std.fmt.allocPrint(allocator, "a.blk.{d}", .{il});
        defer allocator.free(prefix);
        log.debug("loadWeights: loading layer {d}, prefix='{s}'", .{ il, prefix });
        w.layers[il] = loadConformerLayer(io, allocator, ctx, gguf_file, prefix, il) catch |err| {
            log.err("Failed to load conformer layer {d}: {}", .{ il, err });
            return err;
        };
        log.debug("loadWeights: freeing prefix '{s}' (ptr=0x{x}, len={d})", .{ prefix, @intFromPtr(prefix.ptr), prefix.len });
        log.debug("loadWeights: layer {d} done", .{il});
    }
}

/// 加载 Clamp 信息到 VisionEncoderWeights.clamp_info_map
pub fn loadClampInfo(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    var weight_names = std.ArrayList([]const u8).initCapacity(allocator, 0) catch |err| return err;
    defer weight_names.deinit(allocator);

    // 收集所有权重名称
    for (w.sscp_conv_w) |t| {
        if (t) |wt| try weight_names.append(allocator, wt.getName());
    }
    if (w.sscp_inp_proj_w) |t| try weight_names.append(allocator, t.getName());
    if (w.audio_out_proj_w) |t| try weight_names.append(allocator, t.getName());
    if (w.mm_input_proj_w) |t| try weight_names.append(allocator, t.getName());

    for (w.layers) |*layer| {
        if (layer.q_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.k_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.v_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.o_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_up_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_down_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.conv_pw1_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.conv_dw_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.conv_pw2_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_up_1_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_down_1_w) |t| try weight_names.append(allocator, t.getName());
    }

    w.clamp_info_map = try graph.clamp.loadClampInfoFromWeightNames(allocator, gguf_file, weight_names.items);
    log.info("Gemma4A clamp info loaded: {d} entries", .{w.clamp_info_map.count()});
}

/// 估算 Gemma4A 编码后的 token 数量
/// Gemma4A 使用 2 层步长为 2 的子采样，每 4 帧 mel 特征产生 1 个输出 token
pub fn estimateOutputTokens(io: std.Io, n_frames: u32) u32 {
    _ = io;
    return n_frames / 4;
}

// ============================================================================
// 内部辅助函数
// ============================================================================

fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name);
}

/// 从 GGUF 查找张量，支持 required 参数（类似 clip.cpp 的 get_tensor）
/// 如果 required=true 且张量不存在，返回 error.TensorNotFound
/// 如果 required=false 且张量不存在，返回 null
fn findTensorInGGUFWithRequired(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8, required: bool) !?*ggml.Tensor {
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name) catch |err| {
        if (err == error.TensorNotFound) {
            if (required) {
                log.err("Required tensor '{s}' not found in GGUF file", .{name});
                return error.TensorNotFound;
            }
            return null;
        }
        return err;
    };
}
fn loadConformerLayer(io: std.Io, allocator: std.mem.Allocator, ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, prefix: []const u8, il: usize) !ViTLayerWeights {
    var layer = ViTLayerWeights{};

    layer.q_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_q.weight");
    layer.k_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_k.weight");
    layer.v_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_v.weight");
    layer.o_w = try findLayerWeight(ctx, gguf_file, prefix, "attn_out.weight");
    layer.o_b = findLayerWeight(ctx, gguf_file, prefix, "attn_out.bias") catch null;
    layer.ln_1_w = findLayerWeight(ctx, gguf_file, prefix, "ln1.weight") catch null;

    // FFN 1 (required — matching C++ get_tensor without false)
    layer.ff_norm_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight");
    layer.ff_up_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight");
    layer.ff_down_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight");

    // Conformer-specific (all optional — matching C++ get_tensor with false)
    layer.attn_pre_norm_w = findLayerWeight(ctx, gguf_file, prefix, "attn_pre_norm.weight") catch null;
    layer.attn_post_norm_w = findLayerWeight(ctx, gguf_file, prefix, "attn_post_norm.weight") catch null;
    layer.per_dim_scale_w = findLayerWeight(ctx, gguf_file, prefix, "per_dim_scale.weight") catch null;
    layer.per_dim_k_scale_w = findLayerWeight(ctx, gguf_file, prefix, "per_dim_k_scale.weight") catch null;
    layer.attn_k_rel_w = findLayerWeight(ctx, gguf_file, prefix, "attn_k_rel.weight") catch null;
    layer.ff_post_norm_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_post_norm.weight") catch null;

    // Convolution module — matching clip.cpp PROJECTOR_TYPE_GEMMA4A (lines 2577-2586)
    //
    // NOTE: The tensor names are SWAPPED relative to the logical meaning due to
    // upstream tensor_mapping.py. From clip.cpp comment:
    //   "conv_norm / norm_conv are swapped in GGUF due to
    //    upstream tensor_mapping.py, so we load them in reverse order"
    //
    // TN_CONV_NORM = "%s.blk.%d.conv_norm.%s"  →  layer.norm_conv_w (logical norm before conv)
    // TN_NORM_CONV = "%s.blk.%d.norm_conv.%s"  →  layer.conv_norm_w  (logical norm after conv)
    layer.norm_conv_w = findLayerWeight(ctx, gguf_file, prefix, "conv_norm.weight") catch null;
    layer.norm_conv_b = findLayerWeight(ctx, gguf_file, prefix, "conv_norm.bias") catch null;
    layer.conv_pw1_w = try findLayerWeight(ctx, gguf_file, prefix, "conv_pw1.weight");
    layer.conv_pw1_b = findLayerWeight(ctx, gguf_file, prefix, "conv_pw1.bias") catch null;
    layer.conv_dw_w = try findLayerWeight(ctx, gguf_file, prefix, "conv_dw.weight");
    layer.conv_dw_b = findLayerWeight(ctx, gguf_file, prefix, "conv_dw.bias") catch null;
    layer.conv_norm_w = findLayerWeight(ctx, gguf_file, prefix, "norm_conv.weight") catch null;
    layer.conv_norm_b = findLayerWeight(ctx, gguf_file, prefix, "norm_conv.bias") catch null;
    layer.conv_pw2_w = try findLayerWeight(ctx, gguf_file, prefix, "conv_pw2.weight");
    layer.conv_pw2_b = findLayerWeight(ctx, gguf_file, prefix, "conv_pw2.bias") catch null;

    // FFN 2 (ff_norm_1_w, ff_up_1_w, ff_down_1_w required — matching C++ get_tensor without false)
    layer.ff_norm_1_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_norm_1.weight");
    layer.ff_up_1_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_up_1.weight");
    layer.ff_up_1_b = findLayerWeight(ctx, gguf_file, prefix, "ffn_up_1.bias") catch null;
    layer.ff_down_1_w = try findLayerWeight(ctx, gguf_file, prefix, "ffn_down_1.weight");
    layer.ff_down_1_b = findLayerWeight(ctx, gguf_file, prefix, "ffn_down_1.bias") catch null;
    layer.ff_post_norm_1_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_post_norm_1.weight") catch null;

    // Layer output
    layer.ln_2_w = findLayerWeight(ctx, gguf_file, prefix, "ln2.weight") catch null;

    if (il == 0) {
        // Debug save — only for f32 tensors; quantized tensors (Q4_K_M etc.) are skipped
        if (layer.norm_conv_w) |t| {
            const data = try t.dataGet(f32, allocator);
            defer allocator.free(data);
            debug_mod.saveData(io, "debug_audio", "zllama_audio_00_norm_conv_w.json", "norm_conv_w", data) catch {};
        }
        if (layer.conv_pw1_w) |t| {
            const data = try t.dataGet(f32, allocator);
            defer allocator.free(data);
            debug_mod.saveData(io, "debug_audio", "zllama_audio_00_conv_pw1_w.json", "conv_pw1_w", data) catch {};
        }
        if (layer.conv_dw_w) |t| {
            const data = try t.dataGet(f32, allocator);
            defer allocator.free(data);
            debug_mod.saveData(io, "debug_audio", "zllama_audio_00_conv_dw_w.json", "conv_dw_w", data) catch {};
        }
        if (layer.conv_pw2_w) |t| {
            const data = try t.dataGet(f32, allocator);
            defer allocator.free(data);
            debug_mod.saveData(io, "debug_audio", "zllama_audio_00_conv_pw2_w.json", "conv_pw2_w", data) catch {};
        }
    }

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
