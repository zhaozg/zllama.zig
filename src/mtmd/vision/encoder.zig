//! 视觉编码器
//!
//! 提供 Vision Transformer (ViT) 编码器实现，支持 Gemma4V 和 Gemma4UV 两种变体。
//! 参考: llama.cpp tools/mtmd/models/gemma4v.cpp, gemma4uv.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const config = @import("config.zig");
const types = @import("types.zig");
const loader = @import("loader.zig");
const preprocess = @import("preprocess.zig");
const postprocess = @import("postprocess.zig");

const VisionEncoderParams = config.VisionEncoderParams;
const EncoderType = config.EncoderType;
const VisionEncoderWeights = types.VisionEncoderWeights;
const ViTLayerWeights = types.ViTLayerWeights;
const NormalizeMode = preprocess.NormalizeMode;

const log = std.log.scoped(.vision_encoder);

/// 视觉编码器
pub const VisionEncoder = struct {
    params: VisionEncoderParams,
    weights: VisionEncoderWeights,
    encoder_type: EncoderType,
    ctx_weights: *ggml.Context,

    /// 图像归一化参数（来自 GGUF clip.vision.image_mean / image_std）
    image_mean: [3]f32 = .{ 0.0, 0.0, 0.0 },
    image_std: [3]f32 = .{ 1.0, 1.0, 1.0 },

    /// 从 GGUF 文件初始化视觉编码器
    ///
    /// 注意: mmproj GGUF 使用 `clip.vision.*` 前缀的键名（非 `gemma4.vision.*`）
    /// 参考: llama.cpp tools/mtmd/clip.cpp 中 clip_hparams 的加载逻辑
    pub fn init(
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        allocator: std.mem.Allocator,
    ) !VisionEncoder {
        var params = VisionEncoderParams{};

        // 从 GGUF 元数据读取参数
        // mmproj GGUF 使用 clip.vision.* 前缀
        // 同时也尝试 gemma4.vision.* 前缀以兼容不同格式
        if (gguf_file.getU32("clip.vision.image_size")) |v| params.image_size = v else if (gguf_file.getU32("gemma4.vision.image_size")) |v| params.image_size = v;

        if (gguf_file.getU32("clip.vision.patch_size")) |v| params.patch_size = v else if (gguf_file.getU32("gemma4.vision.patch_size")) |v| params.patch_size = v;

        if (gguf_file.getU32("clip.vision.embedding_length")) |v| params.n_embd = v else if (gguf_file.getU32("gemma4.vision.embedding_length")) |v| params.n_embd = v;

        if (gguf_file.getU32("clip.vision.attention.head_count")) |v| params.n_head = v else if (gguf_file.getU32("gemma4.vision.attention_head_count")) |v| params.n_head = v;

        if (gguf_file.getU32("clip.vision.block_count")) |v| params.n_layer = v else if (gguf_file.getU32("gemma4.vision.block_count")) |v| params.n_layer = v;

        if (gguf_file.getU32("clip.vision.feed_forward_length")) |v| params.n_ff = v else if (gguf_file.getU32("gemma4.vision.feed_forward_length")) |v| params.n_ff = v;

        // projection_dim 是输出投影维度（匹配 LLM n_embd），不是 n_merge
        if (gguf_file.getU32("clip.vision.projection_dim")) |v| params.n_output_embd = v else if (gguf_file.getU32("gemma4.vision.projection_dim")) |v| params.n_output_embd = v;

        if (gguf_file.getF32("clip.vision.attention.layer_norm_epsilon")) |v| params.norm_eps = v else if (gguf_file.getF32("gemma4.vision.rope_theta")) |v| params.rope_theta = v;

        // 读取图像归一化参数
        var image_mean: [3]f32 = .{ 0.0, 0.0, 0.0 };
        var image_std: [3]f32 = .{ 1.0, 1.0, 1.0 };
        if (gguf_file.getF32Array("clip.vision.image_mean", 3)) |mean| {
            for (mean, 0..) |v, i| image_mean[i] = v;
        }
        if (gguf_file.getF32Array("clip.vision.image_std", 3)) |std_val| {
            for (std_val, 0..) |v, i| image_std[i] = v;
        }

        // 检测编码器类型
        const enc_type: EncoderType = if (gguf_file.findTensor("v.patch_norm.1.weight") != null or
            gguf_file.findTensor("patch_norm_1.weight") != null)
            .gemma4uv
        else
            .gemma4v;

        log.info("Loading vision encoder: type={s}, size={d}, patch={d}, embd={d}, heads={d}, layers={d}, output_embd={d}", .{
            @tagName(enc_type),   params.image_size, params.patch_size,
            params.n_embd,        params.n_head,     params.n_layer,
            params.n_output_embd,
        });
        log.info("  image_mean=[{d:.4},{d:.4},{d:.4}] image_std=[{d:.4},{d:.4},{d:.4}]", .{
            image_mean[0], image_mean[1], image_mean[2],
            image_std[0],  image_std[1],  image_std[2],
        });

        // 加载所有权重
        const weights = try loader.loadWeights(ctx, gguf_file, params, enc_type, allocator);

        return VisionEncoder{
            .params = params,
            .weights = weights,
            .encoder_type = enc_type,
            .ctx_weights = ctx,
            .image_mean = image_mean,
            .image_std = image_std,
        };
    }

    /// 编码 RGB 图像数据，返回视觉嵌入 tokens
    ///
    /// @param ctx ggml 计算上下文
    /// @param cgraph 计算图
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

        // 1. 归一化输入
        var inp = try preprocess.normalizeToTensor(ctx, image_data, img_width, img_height, self.image_mean, self.image_std, .standard);
        inp.setName("vision_input");

        // 2. Patch embedding
        inp = self.patchEmbed(ctx, inp, img_width, img_height, &effective_n_embd, &d_head, &n_patches_x, &n_patches_y, &n_patches);

        // 3. 位置编码
        inp = self.addPositionEmbeddings(ctx, inp, n_patches, n_patches_x, n_patches_y, effective_n_embd);

        // 4. ViT blocks (only for gemma4v)
        var cur = inp;
        if (self.encoder_type == .gemma4v) {
            cur = self.applyViTBlocks(ctx, cur, effective_n_embd, effective_n_head, d_head, n_patches, n_patches_x, n_patches_y);
        }

        // 5. Pooling (gemma4v only)
        if (self.encoder_type == .gemma4v) {
            cur = self.applyPooling(ctx, cur, effective_n_embd, n_patches_x, n_patches_y);
        }

        // 6. 标准化
        cur = postprocess.standardize(ctx, cur, w.std_bias, w.std_scale);

        // 7. 投影到 LLM 嵌入空间
        cur = postprocess.projectToLLM(ctx, cur, p.norm_eps, w.mm_soft_emb_norm_w, w.mm_input_proj_w);

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

    // ============================================================================
    // 内部方法
    // ============================================================================

    /// Patch embedding 阶段
    fn patchEmbed(
        self: *const VisionEncoder,
        ctx: *ggml.Context,
        inp: *ggml.Tensor,
        img_width: u32,
        img_height: u32,
        effective_n_embd: *i64,
        d_head: *i64,
        n_patches_x: *i64,
        n_patches_y: *i64,
        n_patches: *i64,
    ) *ggml.Tensor {
        const w = self.weights;
        var cur = inp;

        switch (self.encoder_type) {
            .gemma4v => {
                // Scale: patches * 2 - 1
                // Matches llama.cpp gemma4v.cpp: ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f)
                cur = ggml.scale(ctx, cur, 2.0);
                {
                    const bias = ctx.newTensor1d(ggml.Type.f32, 1) catch unreachable;
                    bias.dataF32()[0] = -1.0;
                    bias.setName("vision_bias");
                    cur = cur.add(ctx, bias);
                }

                // Conv2D patch embedding
                if (w.patch_embeddings_0) |pe| {
                    const kw: i32 = @intCast(pe.ne()[0]);
                    const kh: i32 = @intCast(pe.ne()[1]);
                    effective_n_embd.* = pe.ne()[3];
                    d_head.* = @divExact(effective_n_embd.*, @as(i64, @intCast(self.params.n_head)));
                    n_patches_x.* = @divTrunc(@as(i64, @intCast(img_width)), kw);
                    n_patches_y.* = @divTrunc(@as(i64, @intCast(img_height)), kh);
                    n_patches.* = n_patches_x.* * n_patches_y.*;

                    cur = cur.conv2d(ctx, pe, kw, kh, 0, 0, 1, 1);
                    // Reshape to [n_embd, n_patches]
                    cur = cur.reshape3d(ctx, n_patches.*, effective_n_embd.*, 1);
                    cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
                    cur.setName("inp_patches");
                }
            },
            .gemma4uv => {
                // Gemma4UV: im2col + patch norms + projection
                if (w.patch_norm_1_w) |pn1_w| {
                    if (w.patch_embeddings_0) |pe| {
                        const kw: i32 = @intCast(pe.ne()[0]);
                        const kh: i32 = @intCast(pe.ne()[1]);
                        const ic: i32 = @intCast(pe.ne()[2]);
                        n_patches_x.* = @divTrunc(@as(i64, @intCast(img_width)), kw);
                        n_patches_y.* = @divTrunc(@as(i64, @intCast(img_height)), kh);
                        n_patches.* = n_patches_x.* * n_patches_y.*;

                        const im2col_kernel = ctx.newTensor4d(ggml.Type.f32, kw, kh, ic, 1) catch unreachable;
                        cur = cur.im2col(ctx, im2col_kernel, kw, kh, 0, 0, 1, 1, true, ggml.Type.f32);
                    }

                    // Flatten to [patch_size*patch_size*C, n_patches]
                    cur = cur.reshape2d(ctx, cur.ne()[0], n_patches.*);

                    // Patch norm 1
                    cur = cur.norm(ctx, 1e-5);
                    cur = cur.mul(ctx, pn1_w);
                    if (w.patch_norm_1_b) |pn1_b| {
                        cur = cur.add(ctx, pn1_b);
                    }
                }

                // Project to embedding dimension
                if (w.patch_embeddings_0) |pe| {
                    cur = cur.mulMat(ctx, pe);
                }
                if (w.patch_bias) |pb| {
                    cur = cur.add(ctx, pb);
                }

                // Patch norm 2
                if (w.patch_norm_2_w) |pn2_w| {
                    cur = cur.norm(ctx, 1e-5);
                    cur = cur.mul(ctx, pn2_w);
                    if (w.patch_norm_2_b) |pn2_b| {
                        cur = cur.add(ctx, pn2_b);
                    }
                }
            },
        }

        return cur;
    }

    /// 添加位置编码
    fn addPositionEmbeddings(
        self: *const VisionEncoder,
        ctx: *ggml.Context,
        inp: *ggml.Tensor,
        n_patches: i64,
        n_patches_x: i64,
        n_patches_y: i64,
        effective_n_embd: i64,
    ) *ggml.Tensor {
        _ = n_patches_y;
        const w = self.weights;
        var cur = inp;

        if (w.position_embeddings) |pos_embd| {
            const pos_size = pos_embd.ne()[1];
            const row_size = ggml.Type.rowSize(pos_embd.dataType(), effective_n_embd);

            // Position embeddings stored as [n_embd, 2*pos_size]: first half=X, second half=Y
            const tbl_x = pos_embd.view2d(ctx, effective_n_embd, pos_size, row_size, 0);
            const tbl_y = pos_embd.view2d(ctx, effective_n_embd, pos_size, row_size, @as(usize, @intCast(pos_size)) * row_size);

            // Create and fill position index tensors
            ctx.setNoAlloc(false);
            var pos_x = ctx.newTensor1d(ggml.Type.i32, n_patches) catch unreachable;
            pos_x.setName("pos_x");
            var pos_y = ctx.newTensor1d(ggml.Type.i32, n_patches) catch unreachable;
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
            cur = cur.add(ctx, emb_x);
            cur = cur.add(ctx, emb_y);

            // Pos norm (Gemma4UV only)
            if (self.encoder_type == .gemma4uv) {
                if (w.patch_norm_3_w) |pn3_w| {
                    cur = cur.norm(ctx, 1e-5);
                    cur = cur.mul(ctx, pn3_w);
                    if (w.patch_norm_3_b) |pn3_b| {
                        cur = cur.add(ctx, pn3_b);
                    }
                }
            }
        }

        return cur;
    }
    fn applyViTBlocks(
        self: *const VisionEncoder,
        ctx: *ggml.Context,
        cur: *ggml.Tensor,
        effective_n_embd: i64,
        effective_n_head: i64,
        d_head: i64,
        n_patches: i64,
        n_patches_x: i64,
        n_patches_y: i64,
    ) *ggml.Tensor {
        _ = effective_n_embd;
        _ = effective_n_head;
        _ = n_patches;
        _ = n_patches_x;
        _ = n_patches_y;
        const w = self.weights;
        const p = self.params;

        var result = cur;

        for (w.layers) |*layer| {
            // Self-attention with RoPE
            if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
                // Pre-norm
                var attn_in = result;
                if (layer.ln_1_w) |ln1_w| {
                    attn_in = attn_in.rmsNorm(ctx, p.norm_eps);
                    attn_in = attn_in.mul(ctx, postprocess.reshapeForBroadcast(ctx, ln1_w));
                    if (layer.ln_1_b) |ln1_b| {
                        attn_in = attn_in.add(ctx, ln1_b);
                    }
                }

                // Q, K, V projections: weight-first mulMat
                var Q = layer.q_w.?.mulMat(ctx, attn_in);
                const K = layer.k_w.?.mulMat(ctx, attn_in);
                var V = layer.v_w.?.mulMat(ctx, attn_in);

                // Scores: Q @ K
                var scores = Q.mulMat(ctx, K);
                scores = scores.scale(ctx, 1.0 / @sqrt(@as(f32, @floatFromInt(d_head))));
                scores = scores.softMax(ctx);

                // Output: scores @ V^T
                const Vt = V.permute(ctx, 1, 0, 2, 3).cont(ctx);
                var x = scores.mulMat(ctx, Vt);

                // Output projection
                const xt = x.permute(ctx, 1, 0, 2, 3).cont(ctx);
                x = layer.o_w.?.mulMat(ctx, xt);
                if (layer.o_b) |ob| {
                    x = x.add(ctx, ob);
                }
                result = result.add(ctx, x);
            }

            // FFN
            if (layer.ff_up_w != null and layer.ff_down_w != null) {
                var ffn_in = result;
                if (layer.ln_2_w) |ln2_w| {
                    ffn_in = ffn_in.rmsNorm(ctx, p.norm_eps);
                    ffn_in = ffn_in.mul(ctx, postprocess.reshapeForBroadcast(ctx, ln2_w));
                    if (layer.ln_2_b) |ln2_b| {
                        ffn_in = ffn_in.add(ctx, ln2_b);
                    }
                }

                const h = layer.ff_up_w.?.mulMat(ctx, ffn_in);
                const activated = switch (p.ffn_op) {
                    .silu => h.silu(ctx),
                    .gelu => h.gelu(ctx),
                };
                const ffn_out = layer.ff_down_w.?.mulMat(ctx, activated);
                result = result.add(ctx, ffn_out);
            }
        }

        return result;
    }

    /// 应用 Pooling（平均池化下采样）
    fn applyPooling(
        self: *const VisionEncoder,
        ctx: *ggml.Context,
        cur: *ggml.Tensor,
        effective_n_embd: i64,
        n_patches_x: i64,
        n_patches_y: i64,
    ) *ggml.Tensor {
        const p = self.params;
        const kernel_size: i64 = @intCast(p.n_merge);

        var result = cur;
        // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, 1]
        result = result.permute(ctx, 1, 0, 2, 3).cont(ctx);
        result = result.cont4d(ctx, n_patches_x, n_patches_y, effective_n_embd, 1);
        result = result.pool2d(ctx, 1, @as(i32, @intCast(kernel_size)), @as(i32, @intCast(kernel_size)), @as(i32, @intCast(kernel_size)), @as(i32, @intCast(kernel_size)), 0, 0);

        const out_x = @divTrunc(n_patches_x, kernel_size);
        const out_y = @divTrunc(n_patches_y, kernel_size);
        result = result.reshape3d(ctx, out_x * out_y, effective_n_embd, 1);
        result = result.permute(ctx, 1, 0, 2, 3).cont(ctx);
        result = result.scale(ctx, @sqrt(@as(f32, @floatFromInt(effective_n_embd))));

        return result;
    }
};
