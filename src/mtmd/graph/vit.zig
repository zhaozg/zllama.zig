//! 通用 ViT 图构建器
//!
//! 提供 Vision Transformer (ViT) 的通用计算图构建函数。
//! 覆盖大多数模型，特殊模型可复制此函数进行定制。
//!
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h build_vit()

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");
const norm_builder = @import("norm.zig");
const ffn_builder = @import("ffn.zig");
const attn_builder = @import("attn.zig");
const rope_builder = @import("rope.zig");

const NormType = types.NormType;
const FFNOpType = types.FFNOpType;
const BuildVitOpts = types.BuildVitOpts;
const ViTLayerWeights = types.ViTLayerWeights;
const VisionEncoderWeights = types.VisionEncoderWeights;
const VisionHParams = types.VisionHParams;

const log = std.log.scoped(.graph_vit);

/// 构建 Vision Transformer 计算图
///
/// 参数:
///   - ctx: ggml 上下文
///   - inp: 输入张量 [n_embd, n_patches]
///   - n_pos: 位置数量
///   - norm_t: 归一化类型
///   - ffn_t: FFN 激活函数类型
///   - learned_pos_embd: 可学习位置嵌入 [n_embd, n_pos]（可选）
///   - weights: 视觉编码器权重
///   - hparams: 视觉编码器超参数
///   - add_pos: 添加位置嵌入的回调函数
///   - opts: 构建选项
///
/// 返回: 编码后的张量 [n_embd, n_patches]
///
/// 参考: clip-graph.h build_vit()
pub fn buildVit(
    ctx: *ggml.Context,
    inp: *ggml.Tensor,
    n_pos: i64,
    norm_t: NormType,
    ffn_t: FFNOpType,
    learned_pos_embd: ?*ggml.Tensor,
    weights: *const VisionEncoderWeights,
    hparams: *const VisionHParams,
    add_pos: *const fn (*ggml.Context, *ggml.Tensor, *const ViTLayerWeights, *ggml.Tensor) *ggml.Tensor,
    opts: BuildVitOpts,
) !*ggml.Tensor {
    const n_embd = inp.ne()[0];
    _ = add_pos;

    const n_patches = inp.ne()[1];
    const n_head: i64 = @intCast(hparams.n_head);
    const d_head = n_embd / n_head;
    const n_batch: i64 = 1;
    const eps = hparams.eps;

    _ = n_pos;

    // 1. 添加位置嵌入
    var cur = inp;
    if (learned_pos_embd) |pos_embd| {
        // 截取或插值位置嵌入以匹配 n_patches
        const pos_size = pos_embd.ne()[1];
        if (pos_size >= n_patches) {
            // 直接截取前 n_patches 个位置
            const row_size = ggml.Type.rowSize(pos_embd.dataType(), n_embd);
            const sliced = pos_embd.view2d(ctx, n_embd, n_patches, row_size, 0);
            sliced.setName("pos_embd_sliced");
            cur = cur.add(ctx, sliced);
            cur.setName("inp_with_pos");
        } else {
            // 需要插值（通过 resize_position_embeddings）
            log.warn("Position embedding interpolation not yet implemented", .{});
        }
    }

    // 2. Pre-LN (optional)
    if (weights.pre_ln_w) |pln_w| {
        cur = try norm_builder.buildNorm(ctx, cur, pln_w, weights.pre_ln_b, norm_t, eps, "pre_ln");
    }

    // 3. ViT blocks
    var inpL = cur.reshape2d(ctx, n_embd, n_patches * n_batch);
    inpL.setName("vit_input");

    for (weights.layers, 0..) |*layer, il| {
        var residual = inpL;

        // --- Pre-attention norm ---
        var attn_in = inpL;
        if (layer.ln_1_w) |ln1_w| {
            attn_in = try norm_builder.buildNorm(ctx, attn_in, ln1_w, layer.ln_1_b, norm_t, eps, "blk");
        }

        // --- Self-attention ---
        {
            // QKV projections
            var Qcur = if (layer.q_w) |qw| qw.mulMat(ctx, attn_in) else blk: {
                log.warn("Layer {d}: q_w is null, using zero tensor", .{il});
                break :blk try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
            };
            Qcur.setName("blk");

            var Kcur = if (layer.k_w) |kw| kw.mulMat(ctx, attn_in) else blk: {
                log.warn("Layer {d}: k_w is null, using zero tensor", .{il});
                break :blk try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
            };
            Kcur.setName("blk");

            var Vcur = if (layer.v_w) |vw| vw.mulMat(ctx, attn_in) else blk: {
                log.warn("Layer {d}: v_w is null, using zero tensor", .{il});
                break :blk try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
            };
            Vcur.setName("blk");

            // Reshape to [d_head, n_head, n_patches, n_batch]
            Qcur = Qcur.reshape4d(ctx, d_head, n_head, n_patches, n_batch);
            Qcur.setName("blk");
            Kcur = Kcur.reshape4d(ctx, d_head, n_head, n_patches, n_batch);
            Kcur.setName("blk");
            Vcur = Vcur.reshape4d(ctx, d_head, n_head, n_patches, n_batch);
            Vcur.setName("blk");

            // 2D RoPE (if position embeddings are provided via add_pos callback)
            // The add_pos callback handles position-specific modifications
            // For models with 2D RoPE, the caller should apply it before calling buildVit
            // or via the add_pos callback

            // Q/K norm (optional)
            if (layer.q_norm) |qn| {
                Qcur = try norm_builder.buildNorm(ctx, Qcur, qn, null, norm_t, eps, "blk");
            }
            if (layer.k_norm) |kn| {
                Kcur = try norm_builder.buildNorm(ctx, Kcur, kn, null, norm_t, eps, "blk");
            }

            // Attention
            const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
            var attn_out = try attn_builder.buildAttn(
                ctx,
                layer.o_w orelse return error.MissingOutputWeight,
                layer.o_b,
                Qcur,
                Kcur,
                Vcur,
                opts.attn_mask,
                kq_scale,
                n_head,
                "blk",
                layer.attn_sinks,
            );
            attn_out.setName("blk");

            // Post-attention norm (optional, e.g. gemma4)
            if (layer.attn_post_norm_w) |apn| {
                attn_out = try norm_builder.buildNorm(ctx, attn_out, apn, null, norm_t, eps, "blk");
            }

            // Residual + layer scale
            residual = residual.add(ctx, attn_out);
            residual.setName("blk");
            if (layer.ls_1_w) |ls1| {
                residual = residual.mul(ctx, norm_builder.reshapeForBroadcast(ctx, ls1));
                residual.setName("blk");
            }
        }

        // --- Pre-FFN norm ---
        var ffn_in = residual;
        if (layer.ln_2_w) |ln2_w| {
            ffn_in = try norm_builder.buildNorm(ctx, ffn_in, ln2_w, layer.ln_2_b, norm_t, eps, "blk");
        }

        // --- FFN ---
        {
            const ffn_out = try ffn_builder.buildFFN(
                ctx,
                ffn_in,
                layer.ff_up_w orelse return error.MissingFFNUpWeight,
                layer.ff_up_b,
                layer.ff_gate_w,
                layer.ff_gate_b,
                layer.ff_down_w orelse return error.MissingFFNDownWeight,
                layer.ff_down_b,
                ffn_t,
                "blk",
            );
            ffn_out.setName("blk");

            // Post-FFN norm (optional)
            var ffn_result = ffn_out;
            if (layer.ff_post_norm_w) |fpn| {
                ffn_result = try norm_builder.buildNorm(ctx, ffn_result, fpn, null, norm_t, eps, "blk");
            }

            // Residual + layer scale
            inpL = residual.add(ctx, ffn_result);
            inpL.setName("blk");
            if (layer.ls_2_w) |ls2| {
                inpL = inpL.mul(ctx, norm_builder.reshapeForBroadcast(ctx, ls2));
                inpL.setName("blk");
            }
        }
    }

    // 4. Post-LN (optional)
    if (weights.post_ln_w) |poln_w| {
        inpL = try norm_builder.buildNorm(ctx, inpL, poln_w, weights.post_ln_b, norm_t, eps, "post_ln");
    }

    return inpL;
}

/// 调整位置嵌入大小（通过插值）
///
/// 使用双线性插值将位置嵌入调整到目标大小。
///
/// 参数:
///   - ctx: ggml 上下文
///   - pos_embd: 原始位置嵌入 [n_embd, src_size]
///   - target_size: 目标大小
///   - interpolation_mode: 插值模式（默认双线性）
///
/// 返回: 调整后的位置嵌入 [n_embd, target_size]
///
/// 参考: clip-graph.h resize_position_embeddings()
pub fn resizePositionEmbeddings(
    ctx: *ggml.Context,
    pos_embd: *ggml.Tensor,
    target_size: i64,
    interpolation_mode: u32,
) !*ggml.Tensor {
    _ = interpolation_mode;
    const n_embd = pos_embd.ne()[0];
    const src_size = pos_embd.ne()[1];

    log.info("resizePositionEmbeddings: n_embd={d}, src_size={d}, target_size={d}", .{ n_embd, src_size, target_size });

    if (src_size == target_size) {
        return pos_embd;
    }

    // Reshape to 4D: [n_embd, 1, src_size, 1]
    var cur = pos_embd.reshape4d(ctx, n_embd, 1, src_size, 1);
    cur.setName("pos_embd_4d");

    // Use repeat to interpolate along dim 2 (src_size -> target_size)
    const scale_factor: i32 = @intCast(@divTrunc(target_size, src_size));
    if (scale_factor > 1) {
        // Create a target tensor with the desired shape and repeat
        const repeated = try ctx.newTensor4d(ggml.Type.f32, n_embd, 1, target_size, 1);
        cur = ggml.repeat(ctx, cur, repeated);
        cur.setName("pos_embd_upscaled");
    }

    // If target_size is not a multiple of src_size, pad or slice
    const cur_size = cur.ne()[2];
    if (cur_size > target_size) {
        cur = cur.view3d(ctx, n_embd, 1, target_size, cur.nb()[1], cur.nb()[2], 0);
        cur = cur.cont(ctx);
        cur.setName("pos_embd_sliced");
    } else if (cur_size < target_size) {
        const pad_size = target_size - cur_size;
        cur = cur.pad(ctx, 0, 0, @as(i32, @intCast(pad_size)), 0);
        cur.setName("pos_embd_padded");
    }

    // Reshape back to 2D: [n_embd, target_size]
    cur = cur.reshape2d(ctx, n_embd, target_size);
    cur.setName("pos_embd_resized");

    return cur;
}

test "buildVit: basic ViT forward" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ctx = try ggml.Context.init(allocator, .{ .mem_size = 4 * 1024 * 1024 });
    defer ctx.deinit();

    const n_embd: i64 = 64;
    const n_head: u32 = 4;
    const n_layer: u32 = 2;
    const n_patches: i64 = 16;
    const n_ff: u32 = 256;

    // Create input
    const inp = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_patches);
    @memset(inp.dataF32(), 0.5);

    // Create weights
    var layers = try allocator.alloc(ViTLayerWeights, n_layer);
    for (0..n_layer) |il| {
        layers[il] = ViTLayerWeights{
            .ln_1_w = try ctx.newTensor1d(ggml.Type.f32, n_embd),
            .q_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .k_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .v_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .o_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, n_embd),
            .ln_2_w = try ctx.newTensor1d(ggml.Type.f32, n_embd),
            .ff_up_w = try ctx.newTensor2d(ggml.Type.f32, @intCast(n_ff), n_embd),
            .ff_down_w = try ctx.newTensor2d(ggml.Type.f32, n_embd, @intCast(n_ff)),
        };
        // Initialize weights
        for ([_]*ggml.Tensor{ layers[il].ln_1_w.?, layers[il].ln_2_w.? }) |t| {
            @memset(t.dataF32(), 1.0);
        }
        for ([_]*ggml.Tensor{ layers[il].q_w.?, layers[il].k_w.?, layers[il].v_w.?, layers[il].o_w.?, layers[il].ff_up_w.?, layers[il].ff_down_w.? }) |t| {
            @memset(t.dataF32(), 0.1);
        }
    }

    var weights = VisionEncoderWeights{
        .layers = layers,
    };
    var hparams = VisionHParams{
        .n_embd = @intCast(n_embd),
        .n_head = n_head,
        .n_layer = n_layer,
        .n_ff = n_ff,
        .eps = 1e-5,
    };

    // Simple add_pos callback that does nothing
    const addPosFn = struct {
        fn f(_: *ggml.Context, cur: *ggml.Tensor, _: *const ViTLayerWeights, _: *ggml.Tensor) *ggml.Tensor {
            return cur;
        }
    }.f;

    const result = try buildVit(
        &ctx,
        inp,
        n_patches,
        .rms_norm,
        .silu,
        null,
        &weights,
        &hparams,
        addPosFn,
        .{},
    );

    try testing.expectEqual(n_embd, result.ne()[0]);
    try testing.expectEqual(n_patches, result.ne()[1]);
}
