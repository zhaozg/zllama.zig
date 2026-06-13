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
const weight_loader = @import("weight_loader");
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
        const w = self.weights;
        const p = self.params;
        var effective_n_embd: i64 = @intCast(p.n_embd);
        const effective_n_head: i64 = @intCast(p.n_head);
        var d_head: i64 = @divExact(effective_n_embd, effective_n_head);
        var n_patches_x: i64 = 0;
        var n_patches_y: i64 = 0;
        var n_patches: i64 = 0;
        // Validate input size
        const expected_len: usize = @as(usize, @intCast(img_width)) * @as(usize, @intCast(img_height)) * 3;
        if (image_data.len < expected_len) {
            log.err("Image data too small: got {d}, expected {d}", .{ image_data.len, expected_len });
            return error.InvalidImageData;
        }
        // 1. Create input tensor [width, height, channels] = [W, H, C]
        // ggml conv2d WHCN: kernel=[KW,KH,IC,OC], input=[IW,IH,IC]
        var inp = try ctx.newTensor3d(ggml.Type.f32, @intCast(img_width), @intCast(img_height), 3);
        inp.setName("vision_input");
        // Fill tensor with image data: HWC u8 [0,255] -> WHC f32 [0,1]
        {
            const src = image_data;
            const W: usize = @intCast(img_width);
            const H: usize = @intCast(img_height);
            const wh: usize = W * H;
            const dst = inp.dataF32();
            for (0..H) |y| {
                for (0..W) |x| {
                    const src_idx = (y * W + x) * 3;
                    const dst_base = y * W + x;
                    dst[dst_base] = @as(f32, @floatFromInt(src[src_idx + 0])) / 255.0;
                    dst[dst_base + wh] = @as(f32, @floatFromInt(src[src_idx + 1])) / 255.0;
                    dst[dst_base + 2 * wh] = @as(f32, @floatFromInt(src[src_idx + 2])) / 255.0;
                }
            }
        }
        switch (self.encoder_type) {
            .gemma4v => {
                // 标准化: patches * 2 - 1
                inp = ggml.scale(ctx, inp, 2.0);
                {
                    const bias = try ctx.newTensor1d(ggml.Type.f32, 1);
                    bias.dataF32()[0] = -1.0;
                    bias.setName("vision_bias");
                    inp = inp.add(ctx, bias);
                }
                // Conv2D patch embedding — use kernel dimensions for stride
                if (w.patch_embeddings_0) |pe| {
                    const kw: i32 = @intCast(pe.ne()[0]);
                    const kh: i32 = @intCast(pe.ne()[1]);
                    effective_n_embd = pe.ne()[3];
                    d_head = @divExact(effective_n_embd, effective_n_head);
                    n_patches_x = @divTrunc(@as(i64, @intCast(img_width)), kw);
                    n_patches_y = @divTrunc(@as(i64, @intCast(img_height)), kh);
                    n_patches = n_patches_x * n_patches_y;
                    inp = inp.conv2d(ctx, pe, kw, kh, 0, 0, 1, 1);
                    // Reshape to [n_embd, n_patches] — matches llama.cpp convention
                    inp = inp.reshape2d(ctx, effective_n_embd, n_patches);
                    inp = inp.cont(ctx);
                }
            },
            .gemma4uv => {
                // Gemma4UV: im2col + patch norms + projection
                if (w.patch_norm_1_w) |pn1_w| {
                    if (w.patch_embeddings_0) |pe| {
                        const kw: i32 = @intCast(pe.ne()[0]);
                        const kh: i32 = @intCast(pe.ne()[1]);
                        const ic: i32 = @intCast(pe.ne()[2]);
                        n_patches_x = @divTrunc(@as(i64, @intCast(img_width)), kw);
                        n_patches_y = @divTrunc(@as(i64, @intCast(img_height)), kh);
                        n_patches = n_patches_x * n_patches_y;
                        const im2col_kernel = try ctx.newTensor4d(ggml.Type.f32, kw, kh, ic, 1);
                        inp = inp.im2col(ctx, im2col_kernel, kw, kh, 0, 0, 1, 1, true, ggml.Type.f32);
                    }
                    // Flatten to [patch_size*patch_size*C, n_patches]
                    inp = inp.reshape2d(ctx, inp.ne()[0], n_patches);
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
            const pos_size = pos_embd.ne()[1];
            const row_size = ggml.Type.rowSize(pos_embd.dataType(), effective_n_embd);
            // Position embeddings stored as [n_embd, 2*pos_size]: first half=X, second half=Y
            const tbl_x = pos_embd.view2d(ctx, effective_n_embd, pos_size, row_size, 0);
            const tbl_y = pos_embd.view2d(ctx, effective_n_embd, pos_size, row_size, @as(usize, @intCast(pos_size)) * row_size);
            // Create and fill position index tensors (need alloc enabled)
            ctx.setNoAlloc(false);
            var pos_x = try ctx.newTensor1d(ggml.Type.i32, n_patches);
            pos_x.setName("pos_x");
            var pos_y = try ctx.newTensor1d(ggml.Type.i32, n_patches);
            pos_y.setName("pos_y");
            {
                const px = pos_x.dataI32();
                const py = pos_y.dataI32();
                for (0..@as(usize, @intCast(n_patches))) |i| {
                    px[i] = @mod(@as(i32, @intCast(i)), @as(i32, @intCast(n_patches_x)));
                    py[i] = @divTrunc(@as(i32, @intCast(i)), @as(i32, @intCast(n_patches_x)));
                }
            }
            ctx.setNoAlloc(true);
            // getRows produces [n_embd, n_patches]; matches inp format
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
        var cur = inp; // [n_embd, n_patches]
        if (self.encoder_type == .gemma4v) {
            for (w.layers) |*layer| {
                // Self-attention with RoPE
                if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
                    // Pre-norm
                    var attn_in = cur;
                    if (layer.ln_1_w) |ln1_w| {
                        attn_in = attn_in.rmsNorm(ctx, p.norm_eps);
                        attn_in = attn_in.mul(ctx, reshapeForBroadcast(ctx, ln1_w));
                        if (layer.ln_1_b) |ln1_b| {
                            attn_in = attn_in.add(ctx, ln1_b);
                        }
                    }

                    // Q, K, V projections: weight-first mulMat (llama.cpp convention)
                    // attn_in is [n_embd, n_patches], weights are [n_embd, n_embd]
                    // Result: [n_embd, n_patches] — correct layout for reshape
                    var Q = layer.q_w.?.mulMat(ctx, attn_in);
                    const K = layer.k_w.?.mulMat(ctx, attn_in);
                    var V = layer.v_w.?.mulMat(ctx, attn_in);

                    // Scores: Q @ K (contracts on n_embd) → [n_patches, n_patches]
                    // Note: ggml_mul_mat with Q [n_embd, n_patches] @ K [n_embd, n_patches]
                    // contracts on ne[0]=n_embd, producing [n_patches, n_patches]
                    // This computes Q^T @ K (same as standard attention up to transpose)
                    var scores = Q.mulMat(ctx, K);
                    scores = scores.scale(ctx, 1.0 / @sqrt(@as(f32, @floatFromInt(d_head))));
                    scores = scores.softMax(ctx);

                    // Output: scores @ V^T
                    // V^T = V.permute(1,0) → [n_patches, n_embd]
                    // scores [n_patches, n_patches] @ V^T [n_patches, n_embd] → [n_patches, n_embd]
                    const Vt = V.permute(ctx, 1, 0, 2, 3).cont(ctx);
                    var x = scores.mulMat(ctx, Vt); // [n_patches, n_embd]

                    // Output projection: o_w @ x^T
                    // x^T = x.permute(1,0) → [n_embd, n_patches]
                    // o_w [n_embd, n_embd] @ x^T [n_embd, n_patches] → [n_embd, n_patches]
                    const xt = x.permute(ctx, 1, 0, 2, 3).cont(ctx);
                    x = layer.o_w.?.mulMat(ctx, xt);
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
                        ffn_in = ffn_in.mul(ctx, reshapeForBroadcast(ctx, ln2_w));
                        if (layer.ln_2_b) |ln2_b| {
                            ffn_in = ffn_in.add(ctx, ln2_b);
                        }
                    }
                    // ff_up_w [n_ff, n_embd] @ ffn_in [n_embd, n_patches] → [n_ff, n_patches]
                    const h = layer.ff_up_w.?.mulMat(ctx, ffn_in);
                    const activated = switch (p.ffn_op) {
                        .silu => h.silu(ctx),
                        .gelu => h.gelu(ctx),
                    };
                    // ff_down_w [n_embd, n_ff] @ activated [n_ff, n_patches] → [n_embd, n_patches]
                    const ffn_out = layer.ff_down_w.?.mulMat(ctx, activated);
                    cur = cur.add(ctx, ffn_out);
                }
            }
        }
        // 4. Gemma4VisionPooler (average pool + scale)
        if (self.encoder_type == .gemma4v) {
            const kernel_size: i64 = @intCast(p.n_merge);
            // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, 1]
            cur = cur.permute(ctx, 1, 0, 2, 3).cont(ctx);
            cur = cur.cont4d(ctx, n_patches_x, n_patches_y, effective_n_embd, 1);
            cur = cur.pool2d(ctx, 1, @as(i32, @intCast(kernel_size)), @as(i32, @intCast(kernel_size)), @as(i32, @intCast(kernel_size)), @as(i32, @intCast(kernel_size)), 0, 0);
            const out_x = @divTrunc(n_patches_x, kernel_size);
            const out_y = @divTrunc(n_patches_y, kernel_size);
            cur = cur.reshape3d(ctx, out_x * out_y, effective_n_embd, 1);
            cur = cur.permute(ctx, 1, 0, 2, 3).cont(ctx);
            cur = cur.scale(ctx, @sqrt(@as(f32, @floatFromInt(effective_n_embd))));
        }
        // 5. Standardization
        if (w.std_bias) |sb| {
            cur = cur.sub(ctx, sb);
        }
        if (w.std_scale) |ss| {
            cur = cur.mul(ctx, reshapeForBroadcast(ctx, ss));
        }
        // 6. Multimodal embedder
        cur = cur.rmsNorm(ctx, p.norm_eps);
        if (w.mm_soft_emb_norm_w) |sn| {
            cur = cur.mul(ctx, reshapeForBroadcast(ctx, sn));
        }
        if (w.mm_input_proj_w) |proj| {
            // weight-first mulMat: proj [n_output_embd, n_embd] @ cur [n_embd, n_tokens]
            // → [n_output_embd, n_tokens], matching caller expectation (ne[1] = n_tokens)
            cur = proj.mulMat(ctx, cur);
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
fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name);
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
/// Reshape a 1D weight tensor [n] to [n, 1] for broadcasting with [n_embd, n_patches] tensors.
/// Vision encoder uses column-major [n_embd, n_patches] layout.
/// ggml broadcasting: b=[n, 1] vs a=[n_embd, n_patches] -> ne[0]: n==n_embd (ok), ne[1]: 1<=n_patches (ok)
fn reshapeForBroadcast(ctx: *ggml.Context, t: *ggml.Tensor) *ggml.Tensor {
    const n = t.ne()[0];
    return ctx.view2d(t, n, 1, ggml.Type.rowSize(t.dataType(), n), 0);
}
fn findLayerWeight(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
    name: []const u8,
) !*ggml.Tensor {
    return weight_loader.loadLayerWeight(ctx, gguf_file, prefix, name);
}
