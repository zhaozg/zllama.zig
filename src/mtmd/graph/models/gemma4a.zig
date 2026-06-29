//! Gemma4A 音频编码器图构建
//!
//! 实现 Gemma4A Conformer 音频编码器的计算图构建。
//! 参考: deps/llama.cpp/tools/mtmd/models/gemma4a.cpp

const std = @import("std");
const ggml = @import("ggml");
const graph = @import("../mod.zig");

const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const VisionHParams = graph.VisionHParams;
const ViTLayerWeights = graph.ViTLayerWeights;
const ClampInfo = graph.ClampInfo;

const log = std.log.scoped(.gemma4a_graph);

/// 构建 Gemma4A 完整计算图
///
/// 处理流程:
///   1. 子采样 Conv2D（2 层，每层 stride=2, padding=1）
///   2. 输入投影到 Conformer 嵌入维度
///   3. Conformer blocks（FFN1 → Chunked Attention → Conv Module → FFN2）
///   4. 输出投影
///   5. 多模态嵌入器（RMSNorm + 线性投影）
///
/// 参考: llama.cpp gemma4a.cpp build()
pub fn buildGraph(
    builder: *GraphBuilder,
    mel_tensor: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) !*ggml.CGraph {
    const ctx = builder.ctx0;
    const w = builder.weights;
    const p = builder.hparams;

    const norm_eps: f32 = 1e-6;
    const res_weight: f32 = 0.5;

    const n_frames: i64 = mel_tensor.ne()[0];
    const n_mel_bins: i64 = mel_tensor.ne()[1];

    log.info("Gemma4A graph: frames={d}, mel_bins={d}, embd={d}, heads={d}, layers={d}",
        .{ n_frames, n_mel_bins, p.n_embd, p.n_head, p.n_layer });

    // 1. Reshape mel tensor to 4D [n_frames, n_mel_bins, 1, 1]
    var cur = mel_tensor.reshape4d(ctx, n_frames, n_mel_bins, 1, 1);
    cur.setName("inp_raw");

    // Transpose to frame-major layout [n_mel, n_frames, 1, 1]
    cur = ggml.transpose(ctx, cur);
    cur = ggml.cont(ctx, cur);
    cur.setName("inp_transposed");
    ggml.setInput(cur);

    // 2. Subsampling Conv2D (2 layers, stride=2, padding=1)
    for (0..2) |i| {
        if (w.sscp_conv_w[i]) |conv_w| {
            cur = cur.conv2d(ctx, conv_w, 2, 2, 1, 1, 1, 1);
            cur.setName("conv");

            if (w.sscp_conv_b[i]) |conv_b| {
                cur = cur.add(ctx, conv_b);
                cur.setName("conv_biased");
            }

            // LayerNorm
            if (w.sscp_norm_w[i]) |norm_w| {
                cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
                cur = cur.norm(ctx, norm_eps);
                cur = cur.mul(ctx, norm_w);
                cur = cur.permute(ctx, 2, 0, 1, 3).cont(ctx);
                cur.setName("conv_normed");
            }

            cur = cur.relu(ctx);
            cur.setName("conv_act");
        }
    }

    // 3. Flatten: [freq, time, ch, 1] -> [ch*freq, time]
    cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
    cur.setName("flattened");

    const flat_dim0 = cur.ne()[0] * cur.ne()[1];
    cur = cur.reshape2d(ctx, flat_dim0, cur.ne()[2]);
    cur.setName("flatten_2d");

    // 4. Input projection to Conformer embedding dim
    if (w.sscp_inp_proj_w) |proj_w| {
        cur = buildMMWithClamp(ctx, proj_w, cur, clamp_map);
        cur.setName("inp_proj");
        if (w.sscp_inp_proj_b) |proj_b| {
            cur = cur.add(ctx, proj_b);
            cur.setName("inp_proj_biased");
        }
    }

    const n_pos = cur.ne()[1];

    // 5. Chunked local attention parameters
    const C: i64 = 12; // chunk_size
    const P: i64 = 12; // max_past_horizon
    const S: i64 = C + P; // context_size = 24
    const R: i64 = P + 1; // RPE positions = 13
    const B: i64 = @divTrunc((n_pos + C - 1), C); // num_blocks
    const Np: i64 = B * C; // padded sequence length
    const pad_seq: i64 = Np - n_pos;

    log.info("  chunked attn: C={d}, P={d}, S={d}, R={d}, B={d}, Np={d}, pad={d}",
        .{ C, P, S, R, B, Np, pad_seq });

    // Create RPE and mask tensors
    ctx.setNoAlloc(false);
    const pos_emb = try ctx.newTensor2d(ggml.Type.f32, @as(i64, @intCast(p.n_embd)), R);
    pos_emb.setName("pos_emb");
    fillSinusoidalPosEmb(pos_emb, @intCast(R), @intCast(p.n_embd), @intCast(P));

    const kq_mask = try ctx.newTensor4d(ggml.Type.f32, S, C, B, 1);
    kq_mask.setName("attn_mask");
    fillChunkedAttentionMask(kq_mask, @intCast(S), @intCast(C), @intCast(B), @intCast(P), @intCast(n_pos));
    ctx.setNoAlloc(true);

    // 6. Conformer Blocks
    for (w.layers, 0..) |*layer, il| {
        _ = il;


        var residual = cur;

        // FFN 1 (half-step)
        if (layer.ff_norm_w != null and layer.ff_up_w != null and layer.ff_down_w != null) {
            cur = try graph.buildNorm(ctx, residual, layer.ff_norm_w.?, null, .rms_norm, norm_eps, "blk");
            cur = try graph.buildFFN(ctx, cur, layer.ff_up_w.?, null, null, null, layer.ff_down_w.?, null, .silu, "blk");
            if (layer.ff_post_norm_w) |post_norm| {
                cur = try graph.buildNorm(ctx, cur, post_norm, null, .rms_norm, norm_eps, "blk");
            }
            residual = residual.add(ctx, cur.scale(ctx, res_weight));
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
            if (layer.attn_k_rel_w) |k_rel| {
                var p_rpe = buildMMWithClamp(ctx, k_rel, pos_emb, clamp_map);
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
            const kq_mask_4d = ggml.repeat(ctx, kq_mask, scores);
            scores = scores.add(ctx, kq_mask_4d);

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

        // Convolution Module (GLU + depthwise conv)
        if (layer.norm_conv_w != null and layer.conv_pw1_w != null and
            layer.conv_dw_w != null and layer.conv_pw2_w != null)
        {
            cur = try graph.buildNorm(ctx, residual, layer.norm_conv_w.?, null, .rms_norm, norm_eps, "blk");
            var x_conv = buildMMWithClamp(ctx, layer.conv_pw1_w.?, cur, clamp_map);

            // GLU gate
            const d_gate = @divExact(x_conv.ne()[0], 2);
            const gate = x_conv.view2d(ctx, d_gate, x_conv.ne()[1], x_conv.nb()[1], @as(usize, @intCast(@sizeOf(f32) * d_gate))).cont(ctx).sigmoid(ctx);
            const act = x_conv.view2d(ctx, d_gate, x_conv.ne()[1], x_conv.nb()[1], 0);
            x_conv = act.mul(ctx, gate);
            x_conv = x_conv.cont(ctx).permute(ctx, 1, 0, 2, 3).cont(ctx);

            // Causal depthwise conv1d
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
            x_conv = buildMMWithClamp(ctx, layer.conv_pw2_w.?, x_conv, clamp_map);
            residual = residual.add(ctx, x_conv);
        }

        // FFN 2 (half-step)
        if (layer.ff_norm_1_w != null and layer.ff_up_1_w != null and layer.ff_down_1_w != null) {
            cur = try graph.buildNorm(ctx, residual, layer.ff_norm_1_w.?, null, .rms_norm, norm_eps, "blk");
            cur = try graph.buildFFN(ctx, cur, layer.ff_up_1_w.?, null, null, null, layer.ff_down_1_w.?, null, .silu, "blk");
            if (layer.ff_post_norm_1_w) |post_norm| {
                cur = try graph.buildNorm(ctx, cur, post_norm, null, .rms_norm, norm_eps, "blk");
            }
            residual = residual.add(ctx, cur.scale(ctx, res_weight));
        }

        // Layer output norm
        cur = if (layer.ln_2_w) |ln2|
            try graph.buildNorm(ctx, residual, ln2, null, .rms_norm, norm_eps, "blk")
        else
            residual;
    }

    // 7. Output projection
    if (w.audio_out_proj_w) |out_w| {
        cur = buildMMWithClamp(ctx, out_w, cur, clamp_map);
        cur.setName("out_proj");
        if (w.audio_out_proj_b) |out_b| {
            cur = cur.add(ctx, out_b);
            cur.setName("out_proj_biased");
        }
    }

    // 8. Multimodal embedder
    cur = cur.rmsNorm(ctx, norm_eps);
    cur.setName("mm_norm");
    if (w.mm_soft_emb_norm_w) |sn_w| {
        cur = cur.mul(ctx, sn_w);
        cur.setName("mm_norm_scaled");
    }
    if (w.mm_input_proj_w) |proj_w| {
        cur = buildMMWithClamp(ctx, proj_w, cur, clamp_map);
        cur.setName("mm_proj");
    }

    cur.setName("audio_output");
    builder.gf.buildForwardExpand(cur);

    log.info("Gemma4A graph built successfully", .{});
    return builder.gf;
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
        const clamped = x.clamp(ctx, ci.inp_min, ci.inp_max);
        var out = w.mulMat(ctx, clamped);
        out = out.clamp(ctx, ci.out_min, ci.out_max);
        return out;
    } else {
        return w.mulMat(ctx, x);
    }
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
    const data = tensor.dataF32();
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
    const data = tensor.dataF32();
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
}
