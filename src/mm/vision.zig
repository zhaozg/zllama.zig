//! 视觉编码器模块
//!
//! 提供对 Gemma 4 E2B 内建视觉编码器的支持。
//! 支持可变分辨率输入，使用基于视觉 token 预算的动态扩展方式处理图像。
//!
//! 架构: ViT（Vision Transformer）+ SigLIP 风格
//! - Patch embedding（卷积投影）
//! - 2D 位置编码（X/Y 轴分别编码）
//! - 多层 ViT blocks（RMSNorm + 自注意力 + FFN）
//! - Pooling（平均池化下采样）
//! - 输出投影到 LLM 嵌入空间
//!
//! 支持两种视觉编码器变体:
//! - Gemma4V: 标准 ViT + SigLIP（gemma4v）
//! - Gemma4UV: 统一视觉编码器（gemma4uv, 带额外 patch 归一化）
//!
//! 参考: llama.cpp tools/mtmd/models/gemma4v.cpp, gemma4uv.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");

const log = std.log.scoped(.vision_encoder);

// ============================================================================
// 视觉编码器超参数
// ============================================================================

pub const VisionEncoderParams = struct {
    /// 输入图像尺寸（正方形，边长）
    image_size: u32 = 896,
    /// Patch 大小
    patch_size: u32 = 14,
    /// 嵌入维度
    n_embd: u32 = 1152,
    /// 注意力头数
    n_head: u32 = 16,
    /// ViT 层数
    n_layer: u32 = 27,
    /// FFN 中间维度
    n_ff: u32 = 4304,
    /// 输出投影维度（匹配 LLM 嵌入维度）
    n_output_embd: u32 = 2560,
    /// Pooling kernel size（每侧合并数）
    n_merge: u32 = 2,
    /// RoPE theta
    rope_theta: f32 = 10000.0,
    /// 归一化 epsilon
    norm_eps: f32 = 1e-6,
    // FFN activation
    ffn_op: FfnOp = .silu,
};

pub const FfnOp = enum {
    silu,
    gelu,
};

// ============================================================================
// ViT 层权重
// ============================================================================

pub const ViTLayerWeights = struct {
    // 注意力
    ln_1_w: ?*ggml.Tensor = null,
    ln_1_b: ?*ggml.Tensor = null,
    q_w: ?*ggml.Tensor = null,
    k_w: ?*ggml.Tensor = null,
    v_w: ?*ggml.Tensor = null,
    o_w: ?*ggml.Tensor = null,
    o_b: ?*ggml.Tensor = null,

    // FFN
    ln_2_w: ?*ggml.Tensor = null,
    ln_2_b: ?*ggml.Tensor = null,
    ff_up_w: ?*ggml.Tensor = null,
    ff_down_w: ?*ggml.Tensor = null,
};

// ============================================================================
// 视觉编码器权重
// ============================================================================

pub const VisionEncoderWeights = struct {
    params: VisionEncoderParams,

    // Patch embedding
    patch_embeddings_0: ?*ggml.Tensor = null,
    patch_bias: ?*ggml.Tensor = null,

    // Patch 归一化（Gemma4UV 特有）
    patch_norm_1_w: ?*ggml.Tensor = null,
    patch_norm_1_b: ?*ggml.Tensor = null,
    patch_norm_2_w: ?*ggml.Tensor = null,
    patch_norm_2_b: ?*ggml.Tensor = null,
    patch_norm_3_w: ?*ggml.Tensor = null, // pos_norm
    patch_norm_3_b: ?*ggml.Tensor = null, // pos_norm

    // 位置编码
    position_embeddings: ?*ggml.Tensor = null,

    // ViT 层
    layers: []ViTLayerWeights = &.{},

    // 标准化
    std_bias: ?*ggml.Tensor = null,
    std_scale: ?*ggml.Tensor = null,

    // 多模态嵌入投影
    mm_input_proj_w: ?*ggml.Tensor = null,
    mm_soft_emb_norm_w: ?*ggml.Tensor = null,
};

// ============================================================================
// 视觉编码器类型
// ============================================================================

pub const EncoderType = enum {
    gemma4v, // 标准 ViT + SigLIP
    gemma4uv, // 统一视觉编码器（带额外 patch 归一化）
};

// ============================================================================
// 视觉编码器
// ============================================================================

pub const VisionEncoder = struct {
    params: VisionEncoderParams,
    weights: VisionEncoderWeights,
    encoder_type: EncoderType,
    ctx_weights: *ggml.Context,

    /// 从 GGUF 文件初始化视觉编码器
    pub fn init(
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
    ) !VisionEncoder {
        var params = VisionEncoderParams{};

        // 从 GGUF 元数据读取参数
        if (gguf_file.getU32("gemma4.vision.image_size")) |v| params.image_size = v;
        if (gguf_file.getU32("gemma4.vision.patch_size")) |v| params.patch_size = v;
        if (gguf_file.getU32("gemma4.vision.embedding_length")) |v| params.n_embd = v;
        if (gguf_file.getU32("gemma4.vision.attention_head_count")) |v| params.n_head = v;
        if (gguf_file.getU32("gemma4.vision.block_count")) |v| params.n_layer = v;
        if (gguf_file.getU32("gemma4.vision.feed_forward_length")) |v| params.n_ff = v;
        if (gguf_file.getU32("gemma4.vision.projection_dim")) |v| params.n_merge = v;
        if (gguf_file.getF32("gemma4.vision.rope_theta")) |v| params.rope_theta = v;

        // 检测编码器类型
        const enc_type: EncoderType = if (gguf_file.findTensor("patch_norm_1.weight") != null)
            .gemma4uv
        else
            .gemma4v;

        log.info("Loading vision encoder: type={s}, size={d}, patch={d}, embd={d}, heads={d}, layers={d}", .{
            @tagName(enc_type), params.image_size, params.patch_size,
            params.n_embd, params.n_head, params.n_layer,
        });

        // 加载 Patch embedding (v.patch_embd.*)
        const patch_embd = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight") catch null;
        const patch_bias = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.bias") catch null;

        // Patch 归一化（Gemma4UV）
        const pn1_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.1.weight") catch null;
        const pn1_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.1.bias") catch null;
        const pn2_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.2.weight") catch null;
        const pn2_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.2.bias") catch null;
        const pn3_w = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.3.weight") catch null;
        const pn3_b = findTensorInGGUF(ctx, gguf_file, "v.patch_norm.3.bias") catch null;

        // 位置编码 (v.position_embd.weight)
        const pos_embd = findTensorInGGUF(ctx, gguf_file, "v.position_embd.weight") catch null;

        // 标准化 (v.std_bias, v.std_scale)
        const std_bias = findTensorInGGUF(ctx, gguf_file, "v.std_bias") catch null;
        const std_scale = findTensorInGGUF(ctx, gguf_file, "v.std_scale") catch null;

        // 多模态投影 (mm.*)
        const mm_proj = findTensorInGGUF(ctx, gguf_file, "mm.input_projection.weight") catch null;
        const mm_soft = findTensorInGGUF(ctx, gguf_file, "mm.soft_emb_norm.weight") catch null;

        // 加载 ViT 层
        const n_layer: usize = @intCast(params.n_layer);
        var layers = try allocator.alloc(ViTLayerWeights, n_layer);

        for (0..n_layer) |il| {
            const prefix = try std.fmt.allocPrint(allocator, "v.blk.{d}", .{il});
            layers[il] = loadViTLayer(ctx, gguf_file, prefix) catch |err| {
                log.err("Failed to load ViT layer {d}: {}", .{ il, err });
                allocator.free(prefix);
                return err;
            };
            allocator.free(prefix);
        }

        log.info("Vision encoder loaded: {d} ViT layers", .{n_layer});

        return VisionEncoder{
            .params = params,
            .weights = .{
                .params = params,
                .patch_embeddings_0 = patch_embd,
                .patch_bias = patch_bias,
                .patch_norm_1_w = pn1_w,
                .patch_norm_1_b = pn1_b,
                .patch_norm_2_w = pn2_w,
                .patch_norm_2_b = pn2_b,
                .patch_norm_3_w = pn3_w,
                .patch_norm_3_b = pn3_b,
                .position_embeddings = pos_embd,
                .layers = layers,
                .std_bias = std_bias,
                .std_scale = std_scale,
                .mm_input_proj_w = mm_proj,
                .mm_soft_emb_norm_w = mm_soft,
            },
            .encoder_type = enc_type,
            .ctx_weights = ctx,
        };
    }

    /// 编码 RGB 图像数据，返回视觉嵌入 tokens
    /// @param ctx ggml 计算上下文
    /// @param graph 计算图
    /// @param image_data RGB 图像数据 [height][width][3]，值范围 [0, 255]
    /// @param img_width 图像宽度
    /// @param img_height 图像高度
    /// @returns 视觉嵌入 [n_output_embd, n_tokens]
    pub fn encode(
        self: *const VisionEncoder,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        image_data: []const u8,
        img_width: u32,
        img_height: u32,
    ) !*ggml.Tensor {
        _ = image_data;
        const w = self.weights;
        const p = self.params;
        const patch_size: i64 = @intCast(p.patch_size);
        const n_embd: i64 = @intCast(p.n_embd);
        const n_head: i64 = @intCast(p.n_head);
        const d_head: i64 = n_embd / n_head;
        const n_patches_x = @divTrunc(@as(i64, @intCast(img_width)), patch_size);
        const n_patches_y = @divTrunc(@as(i64, @intCast(img_height)), patch_size);
        const n_patches = n_patches_x * n_patches_y;

        // 1. 创建输入张量 [channels, height, width] -> [C, H, W]
        // 输入为 RGB: [3, height, width]
        var inp = try ctx.newTensor3d(ggml.Type.F32, n_patches_x, n_patches_y, 3);
        inp.setName("vision_input");

        switch (self.encoder_type) {
            .gemma4v => {
                // 标准化: patches * 2 - 1
                inp = inp.scale(ctx, 2.0);
                inp = inp.add(ctx, try ctx.newTensor1d(ggml.Type.F32, 1)); // bias=-1 as add

                // Conv2D patch embedding
                if (w.patch_embeddings_0) |pe| {
                    inp = inp.conv2d(ctx, pe, patch_size, patch_size, 0, 0, 1, 1);
                }
                // [out_c, out_h, out_w] -> [n_embd, n_patches_y * n_patches_x] -> transpose to [n_patches, n_embd]
                inp = inp.reshape2d(n_embd, n_patches);
                inp = inp.permute(1, 0, 2, 3).cont(ctx);
            },
            .gemma4uv => {
                // Gemma4UV: im2col + patch norms + projection
                if (w.patch_norm_1_w) |pn1_w| {
                    // im2col: [C, H, W] -> [patch_size*patch_size*C, n_patches_y, n_patches_x]
                    var kernel = try ctx.newTensor3d(ggml.Type.F32, patch_size, patch_size, 3);
                    inp = inp.im2col(ctx, kernel, patch_size, patch_size, 0, 0, 1, 1, true, ggml.Type.F32);
                    // Flatten to [patch_size*patch_size*C, n_patches]
                    inp = inp.reshape2d(inp.ne[0], n_patches);
                    // Patch norm 1
                    inp = inp.norm(ctx, 1e-5);
                    inp = inp.mul(ctx, pn1_w);
                    if (w.patch_norm_1_b) |pn1_b| {
                        inp = inp.add(ctx, pn1_b);
                    }
                }
                // Project to embedding dimension
                if (w.patch_embeddings_0) |pe| {
                    inp = inp.mulMat(ctx, pe);
                }
                if (w.patch_bias) |pb| {
                    inp = inp.add(ctx, pb);
                }
                // Patch norm 2
                if (w.patch_norm_2_w) |pn2_w| {
                    inp = inp.norm(ctx, 1e-5);
                    inp = inp.mul(ctx, pn2_w);
                    if (w.patch_norm_2_b) |pn2_b| {
                        inp = inp.add(ctx, pn2_b);
                    }
                }
            },
        }

        // 2. 位置编码 (lookup from position_embeddings)
        if (w.position_embeddings) |pos_embd| {
            const pos_size = pos_embd.ne[1];
            const row_size = ggml.rowSize(pos_embd.getType(), n_embd);

            // Position embeddings stored as [n_embd, 2*pos_size]: first half=X, second half=Y
            const tbl_x = pos_embd.view2d(n_embd, pos_size, row_size, 0);
            const tbl_y = pos_embd.view2d(n_embd, pos_size, row_size, pos_size * row_size);

            // Create position indices [n_patches]
            var pos_x = try ctx.newTensor1d(ggml.Type.I32, n_patches);
            pos_x.setName("pos_x");
            var pos_y = try ctx.newTensor1d(ggml.Type.I32, n_patches);
            pos_y.setName("pos_y");

            // Lookup position embeddings
            const emb_x = tbl_x.getRows(ctx, pos_x);
            const emb_y = tbl_y.getRows(ctx, pos_y);

            inp = inp.add(ctx, emb_x);
            inp = inp.add(ctx, emb_y);

            // Pos norm (Gemma4UV only)
            if (self.encoder_type == .gemma4uv) {
                if (w.patch_norm_3_w) |pn3_w| {
                    inp = inp.norm(ctx, 1e-5);
                    inp = inp.mul(ctx, pn3_w);
                    if (w.patch_norm_3_b) |pn3_b| {
                        inp = inp.add(ctx, pn3_b);
                    }
                }
            }
        }

        // 3. ViT blocks (only for gemma4v - gemma4uv skips ViT)
        var cur = inp;
        if (self.encoder_type == .gemma4v) {
            for (w.layers) |*layer| {
                // Self-attention with RoPE
                if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
                    // Pre-norm
                    var attn_in = cur;
                    if (layer.ln_1_w) |ln1_w| {
                        attn_in = attn_in.rmsNorm(ctx, p.norm_eps);
                        attn_in = attn_in.mul(ctx, ln1_w);
                        if (layer.ln_1_b) |ln1_b| {
                            attn_in = attn_in.add(ctx, ln1_b);
                        }
                    }

                    // Q, K, V projections
                    var Q = attn_in.mulMat(ctx, layer.q_w.?);
                    var K = attn_in.mulMat(ctx, layer.k_w.?);
                    var V = attn_in.mulMat(ctx, layer.v_w.?);

                    // Reshape to [d_head, n_head, n_patches]
                    Q = Q.reshape3d(d_head, n_head, n_patches);
                    K = K.reshape3d(d_head, n_head, n_patches);
                    V = V.reshape3d(d_head, n_head, n_patches);

                    // Apply 2D RoPE
                    var pos_x = try ctx.newTensor1d(ggml.Type.I32, n_patches);
                    pos_x.setName("rope_pos_x");
                    var pos_y = try ctx.newTensor1d(ggml.Type.I32, n_patches);
                    pos_y.setName("rope_pos_y");

                    // First half RoPE with pos_x (neox)
                    const half_d = d_head / 2;
                    var Q_first = Q.view3d(half_d, n_head, n_patches, Q.nb[1], Q.nb[2], 0);
                    Q_first = Q_first.ropeExt(ctx, pos_x, null, half_d, .neox, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                    var Q_second = Q.view3d(half_d, n_head, n_patches, Q.nb[1], Q.nb[2], half_d * @sizeOf(f32));
                    Q_second = Q_second.ropeExt(ctx, pos_y, null, half_d, .neox, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                    Q = Q_first.concat(ctx, Q_second, 0);

                    var K_first = K.view3d(half_d, n_head, n_patches, K.nb[1], K.nb[2], 0);
                    K_first = K_first.ropeExt(ctx, pos_x, null, half_d, .neox, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                    var K_second = K.view3d(half_d, n_head, n_patches, K.nb[1], K.nb[2], half_d * @sizeOf(f32));
                    K_second = K_second.ropeExt(ctx, pos_y, null, half_d, .neox, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
                    K = K_first.concat(ctx, K_second, 0);

                    // K transpose: [d_head, n_head, n_patches] -> [d_head, n_patches, n_head]
                    K = K.permute(0, 2, 1, 3).cont(ctx);

                    // V transpose: [d_head, n_head, n_patches] -> [n_patches, d_head, n_head]
                    V = V.permute(2, 0, 1, 3).cont(ctx);

                    // Q @ K^T: [d_head, n_head, n_patches] @ [d_head, n_patches, n_head] -> [n_patches, n_head, n_head] -> [n_head, n_patches, n_patches]
                    var scores = K.mulMat(ctx, Q);
                    scores = scores.scale(ctx, 1.0 / @sqrt(@as(f32, @floatFromInt(d_head))));
                    scores = scores.softMax(ctx);

                    // attn @ V: [n_head, n_patches, n_patches] @ [n_patches, d_head, n_head] -> [n_head, n_patches, d_head]
                    var x = V.mulMat(ctx, scores);
                    x = x.permute(2, 0, 1, 3).cont(ctx); // [d_head, n_head, n_patches]
                    x = x.cont2d(ctx, n_embd, n_patches);

                    x = x.mulMat(ctx, layer.o_w.?);
                    if (layer.o_b) |ob| {
                        x = x.add(ctx, ob);
                    }

                    cur = cur.add(ctx, x);
                }

                // FFN
                if (layer.ff_up_w != null and layer.ff_down_w != null) {
                    var ffn_in = cur;
                    if (layer.ln_2_w) |ln2_w| {
                        ffn_in = ffn_in.rmsNorm(ctx, p.norm_eps);
                        ffn_in = ffn_in.mul(ctx, ln2_w);
                        if (layer.ln_2_b) |ln2_b| {
                            ffn_in = ffn_in.add(ctx, ln2_b);
                        }
                    }

                    const h = ffn_in.mulMat(ctx, layer.ff_up_w.?);
                    const activated = switch (p.ffn_op) {
                        .silu => h.silu(ctx),
                        .gelu => h.gelu(ctx),
                    };
                    const ffn_out = activated.mulMat(ctx, layer.ff_down_w.?);
                    cur = cur.add(ctx, ffn_out);
                }
            }
        }

        // 4. Gemma4VisionPooler (average pool + scale)
        if (self.encoder_type == .gemma4v) {
            const kernel_size: i64 = @intCast(p.n_merge);
            // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, 1]
            cur = cur.permute(1, 0, 2, 3).cont(ctx);
            cur = cur.cont4d(ctx, n_patches_x, n_patches_y, n_embd, 1);
            cur = cur.pool2d(ctx, .avg, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
            const out_x = n_patches_x / kernel_size;
            const out_y = n_patches_y / kernel_size;
            cur = cur.reshape3d(out_x * out_y, n_embd, 1);
            cur = cur.permute(1, 0, 2, 3).cont(ctx);
            cur = cur.scale(ctx, @sqrt(@as(f32, @floatFromInt(n_embd))));
        }

        // 5. Standardization
        if (w.std_bias) |sb| {
            cur = cur.sub(ctx, sb);
        }
        if (w.std_scale) |ss| {
            cur = cur.mul(ctx, ss);
        }

        // 6. Multimodal embedder
        cur = cur.rmsNorm(ctx, p.norm_eps);
        if (w.mm_soft_emb_norm_w) |sn| {
            cur = cur.mul(ctx, sn);
        }
        if (w.mm_input_proj_w) |proj| {
            cur = cur.mulMat(ctx, proj);
        }

        cgraph.buildForwardExpand(cur);
        return cur;
    }

    /// 返回视觉编码器是否可用（权重已加载）
    pub fn isAvailable(self: *const VisionEncoder) bool {
        return self.weights.patch_embeddings_0 != null;
    }

    /// 估算给定分辨率图像的 token 数量
    pub fn estimateOutputTokens(self: *const VisionEncoder, img_width: u32, img_height: u32) u32 {
        const patches_x = (img_width + self.params.patch_size - 1) / self.params.patch_size;
        const patches_y = (img_height + self.params.patch_size - 1) / self.params.patch_size;
        const n_patches = patches_x * patches_y;
        const n_merge = if (self.params.n_merge > 0) self.params.n_merge else 1;
        return n_patches / (n_merge * n_merge);
    }

    /// 计算视觉 token 预算下的最佳图像分辨率
    pub fn bestResolution(self: *const VisionEncoder, max_tokens: u32) struct { width: u32, height: u32 } {
        const n_merge = if (self.params.n_merge > 0) self.params.n_merge else 1;
        const max_patches = max_tokens * n_merge * n_merge;
        const side_patches = @max(1, @as(u32, @intFromFloat(@sqrt(@as(f64, @floatFromInt(max_patches))))));
        const side = side_patches * self.params.patch_size;
        return .{ .width = side, .height = side };
    }

    pub fn deinit(self: *VisionEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.weights.layers);
    }
};

// ============================================================================
// 辅助函数
// ============================================================================

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

fn loadViTLayer(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
) !ViTLayerWeights {
    var layer = ViTLayerWeights{};

    // Attention
    layer.ln_1_w = findLayerWeight(ctx, gguf_file, prefix, "ln1.weight") catch null;
    layer.ln_1_b = findLayerWeight(ctx, gguf_file, prefix, "ln1.bias") catch null;
    layer.q_w = findLayerWeight(ctx, gguf_file, prefix, "attn_q.weight") catch null;
    layer.k_w = findLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch null;
    layer.v_w = findLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch null;
    layer.o_w = findLayerWeight(ctx, gguf_file, prefix, "attn_out.weight") catch null;
    layer.o_b = findLayerWeight(ctx, gguf_file, prefix, "attn_out.bias") catch null;

    // FFN
    layer.ln_2_w = findLayerWeight(ctx, gguf_file, prefix, "ln2.weight") catch null;
    layer.ln_2_b = findLayerWeight(ctx, gguf_file, prefix, "ln2.bias") catch null;
    layer.ff_up_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch null;
    layer.ff_down_w = findLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch null;

    return layer;
}

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
