//! Gemma 4 模型实现
//!
//! Gemma 4 是增强版 Transformer 解码器，支持：
//! - SWA（滑动窗口注意力）+ Full Attention 混合层
//! - 共享 KV 层（部分层复用前面的 KV）
//! - MoE（Mixture of Experts）FFN
//! - Per-layer token embedding
//! - Proportional RoPE（全注意力层使用 rope_freqs）
//! - Q/K pre-norm + Post norms
//! - Layer output scale
//! - Final logit softcapping
//!
//! 参考 llama.cpp gemma4.cpp 实现。
const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const kv_cache = @import("kv_cache");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const attention = @import("attention");
const embed = @import("embed");
const weight_loader = @import("weight_loader");

const model = @import("../model.zig");

const log = std.log.scoped(.gemma4);

// ============================================================================
// Gemma 4 超参数
// ============================================================================

pub const Gemma4Params = struct {
    base: model.ModelParams = .{},

    /// SWA 滑动窗口大小
    n_swa: u32 = 4096,
    /// SWA 模式（每层是否为 SWA）
    is_swa_layer: std.ArrayList(bool),
    /// SWA 的 RoPE base frequency
    rope_freq_base_swa: f32 = 0.0,
    /// 共享 KV 的层数（最后 n_kv_shared_layers 层复用前面层的 KV）
    /// SWA 的 RoPE 维度（与 full attention 的 rope_dim 可能不同，因为 SWA 层的 head_dim 较小）
    /// 如果未指定则默认与 base.rope_dim 相同
    rope_dim_swa: u32 = 0,

    n_kv_shared_layers: u32 = 0,
    /// 从开始算起有多少层拥有自己的 KV
    n_layer_kv_from_start: u32 = 0,
    /// 注意力缩放因子（默认 1.0，Gemma4 不做 pre-attn scaling）
    f_attention_scale: f32 = 1.0,
    /// 最终 logit softcapping 阈值（0 表示禁用）
    final_logit_softcapping: f32 = 0.0,
    /// 注意力 logit softcapping 阈值（0 表示禁用，Gemma 默认 ~50.0）
    attn_logit_softcapping: f32 = 0.0,
    /// per-layer embedding 维度（0 表示禁用）
    n_embd_per_layer: u32 = 0,

    pub fn deinit(self: *Gemma4Params, allocator: std.mem.Allocator) void {
        self.is_swa_layer.deinit(allocator);
    }
};

// ============================================================================
// Gemma 4 层权重
// ============================================================================

pub const LayerWeights = struct {
    prefix: []const u8,

    // 归一化
    attn_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor,

    // Q/K 预归一化
    attn_q_norm_weight: *ggml.Tensor,
    attn_k_norm_weight: *ggml.Tensor,

    // Post 归一化
    attn_post_norm_weight: *ggml.Tensor,
    ffn_post_norm_weight: *ggml.Tensor,

    // 注意力
    attn_q_weight: *ggml.Tensor,
    attn_k_weight: ?*ggml.Tensor,
    attn_v_weight: ?*ggml.Tensor,
    attn_output_weight: *ggml.Tensor,

    // FFN (GeGLU)
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,

    // 层输出缩放（可选）
    out_scale: ?*ggml.Tensor,

    // RoPE freqs（全注意力层使用 proportional rope）
    rope_freqs: ?*ggml.Tensor,

    // Per-layer embedding 权重（仅当 n_embd_per_layer > 0 时加载）
    per_layer_inp_gate: ?*ggml.Tensor = null,
    per_layer_proj: ?*ggml.Tensor = null,
    per_layer_post_norm: ?*ggml.Tensor = null,

    // 该层是否有自己的 KV
    has_kv: bool,
};
pub const Gemma4Weights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,

    // Per-layer embedding global weights（仅当 n_embd_per_layer > 0 时加载）
    per_layer_token_embd: ?*ggml.Tensor = null,
    per_layer_model_proj: ?*ggml.Tensor = null,
    per_layer_proj_norm: ?*ggml.Tensor = null,

    pub fn deinit(self: *Gemma4Weights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| {
            allocator.free(layer.prefix);
        }
        allocator.free(self.layers);
    }
};

// ============================================================================
// Gemma 4 图构建类
// ============================================================================
//
// 参考 llama.cpp 的 llm_build_context 模式：将 forward() 拆分为独立的图构建类。
// Gemma4Graph 持有图构建过程中所有中间张量引用，并将 transformerForward 分解为
// buildPerLayerInputs / buildLayer / buildAttention / buildOutput 等命名方法。
// Gemma4Model 的 forward 方法变为 thin wrapper，委托给 Gemma4Graph。

pub const Gemma4Graph = struct {
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    params: *const Gemma4Params,
    weights: *const Gemma4Weights,

    // ------ 中间张量（一次 forward 有效）------
    cur: *ggml.Tensor,
    pos_tensor: *ggml.Tensor,
    inp_per_layer: ?*ggml.Tensor,

    const Self = @This();

    /// 初始化图构建器
    pub fn init(
        ctx: *ggml.Context,
        gf: *ggml.CGraph,
        params: *const Gemma4Params,
        weights: *const Gemma4Weights,
        inp_embd: *ggml.Tensor,
        pos_tensor: *ggml.Tensor,
    ) Self {
        return Self{
            .ctx = ctx,
            .gf = gf,
            .params = params,
            .weights = weights,
            .cur = inp_embd,
            .pos_tensor = pos_tensor,
            .inp_per_layer = null,
        };
    }

    // ====================================================================
    // 公共构建入口
    // ====================================================================

    /// 纯文本 forward：查表嵌入 → transformer 层 → logits
    pub fn build(
        self: *Self,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
    ) !*ggml.Tensor {
        const p = self.params;
        const w = self.weights;
        const n_tokens_i64: i64 = n_tokens;

        var embd = embed.tokenEmbedding(self.ctx, w.base.token_embd, input_tokens);
        embd = ggml.scale(self.ctx, embd, @sqrt(@as(f32, @floatFromInt(p.base.n_embd))));
        embd.setName("inp_scaled");

        const pos_tensor = rope.buildPositionTensor(self.ctx, @intCast(n_tokens), start_pos);

        var g = Self.init(self.ctx, self.gf, p, w, embd, pos_tensor);
        return try g.transformerForward(n_tokens_i64, start_pos, kv_cache_mgr, input_tokens, true);
    }

    /// 混合文本+媒体 forward：在指定位置注入预计算嵌入
    pub fn buildWithEmbd(
        self: *Self,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        embd_override: *ggml.Tensor,
        embd_offset: i32,
        causal: bool,
    ) !*ggml.Tensor {
        const p = self.params;
        const w = self.weights;
        const n_tokens_i64: i64 = n_tokens;
        const n_override: i64 = embd_override.ne()[1];
        const override_offset: i64 = @intCast(embd_offset);
        const n_text_pre: i64 = override_offset;
        const n_text_post: i64 = n_tokens_i64 - n_override - override_offset;
        const n_embd_i64: i64 = @intCast(p.base.n_embd);
        const override_embd_dim: i64 = embd_override.ne()[0];

        if (override_embd_dim != n_embd_i64) {
            log.err("buildWithEmbd: embedding dim mismatch — override={d} model={d}", .{ override_embd_dim, n_embd_i64 });
            return error.EmbeddingDimensionMismatch;
        }
        log.debug("buildWithEmbd: n_tokens={d}, embd_offset={d}, n_override={d}, causal={}, embd_dim={d} ✓", .{
            n_tokens, embd_offset, n_override, causal, override_embd_dim,
        });

        var all_embd = embed.tokenEmbedding(self.ctx, w.base.token_embd, input_tokens);
        all_embd = ggml.scale(self.ctx, all_embd, @sqrt(@as(f32, @floatFromInt(p.base.n_embd))));
        const scaled_override = embd_override; // Already in model embedding space — NO scale

        var cur: *ggml.Tensor = undefined;
        if (n_text_pre > 0 and n_text_post > 0) {
            const prefix_embd = all_embd.view2d(self.ctx, n_embd_i64, n_text_pre, all_embd.nb()[1], 0);
            const suffix_off: usize = @as(usize, @intCast(override_offset + n_override)) * @sizeOf(f32) * @as(usize, @intCast(n_embd_i64));
            const suffix_embd = all_embd.view2d(self.ctx, n_embd_i64, n_text_post, all_embd.nb()[1], suffix_off);
            const mid = ggml.concat(self.ctx, ggml.cont(self.ctx, prefix_embd), scaled_override, 1);
            cur = ggml.concat(self.ctx, mid, ggml.cont(self.ctx, suffix_embd), 1);
        } else if (n_text_pre > 0) {
            const prefix_embd = all_embd.view2d(self.ctx, n_embd_i64, n_text_pre, all_embd.nb()[1], 0);
            cur = ggml.concat(self.ctx, ggml.cont(self.ctx, prefix_embd), scaled_override, 1);
        } else if (n_text_post > 0) {
            const suffix_off: usize = @as(usize, @intCast(override_offset + n_override)) * @sizeOf(f32) * @as(usize, @intCast(n_embd_i64));
            const suffix_embd = all_embd.view2d(self.ctx, n_embd_i64, n_text_post, all_embd.nb()[1], suffix_off);
            cur = ggml.concat(self.ctx, scaled_override, ggml.cont(self.ctx, suffix_embd), 1);
        } else {
            cur = scaled_override;
        }
        cur.setName("inp_scaled_mm");

        const pos_tensor = rope.buildPositionTensor(self.ctx, @intCast(n_tokens), start_pos);

        var g = Self.init(self.ctx, self.gf, p, w, cur, pos_tensor);
        return try g.transformerForward(n_tokens_i64, start_pos, kv_cache_mgr, input_tokens, causal);
    }

    /// 纯媒体 forward：使用预计算嵌入，不做 token 查表。非因果注意力。
    /// NOTE: input_tokens is only used for per-layer embedding lookup.
    /// For pure-media forward, per-layer injection is skipped (passing null)
    /// because the placeholder token's per-layer embedding is untrained noise
    /// that would corrupt the audio/vision hidden states.
    pub fn buildMediaOnly(
        self: *Self,
        embd_override: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        input_tokens: *ggml.Tensor,
    ) !*ggml.Tensor {
        const p = self.params;
        const w = self.weights;
        const n_tokens_i64: i64 = n_tokens;

        // —— 嵌入维度检查 ——
        const override_embd_dim: i64 = embd_override.ne()[0];
        const model_n_embd: i64 = @intCast(p.base.n_embd);
        log.debug("buildMediaOnly (non-causal): n_tokens={d}, start_pos={d}, embd_dim={d}, model_n_embd={d}", .{ n_tokens, start_pos, override_embd_dim, model_n_embd });
        if (override_embd_dim != model_n_embd) {
            log.err("buildMediaOnly: embedding dim mismatch — override={d} model={d}", .{ override_embd_dim, model_n_embd });
            return error.EmbeddingDimensionMismatch;
        }
        log.debug("  ✓ embedding dimension check passed", .{});

        const scaled = embd_override; // Already in model embedding space — NO scale
        scaled.setName("inp_media");

        const pos_tensor = rope.buildPositionTensor(self.ctx, @intCast(n_tokens), start_pos);

        var g = Self.init(self.ctx, self.gf, p, w, scaled, pos_tensor);
        // For multimodal embedding path, use the placeholder token IDs from input_tokens
        // for per-layer embedding lookup. This matches llama.cpp behavior where
        // ubatch.token is null (embedding path), so build_inp_per_layer uses
        // padding token (row 0) from per_layer_tok_embd.
        // Reference: llama.cpp gemma4.cpp build_inp_per_layer() else branch
        // Note: We pass input_tokens (placeholder token IDs) for per-layer embedding lookup.
        // The ggml context must NOT be in setNoAlloc(false) mode here — the graph is built
        // with view tensors and the actual allocation happens via Gallocr.
        return try g.transformerForward(n_tokens_i64, start_pos, kv_cache_mgr, input_tokens, false);
    }

    // ====================================================================
    // Transformer 主循环
    // ====================================================================

    fn transformerForward(
        self: *Self,
        n_tokens_i64: i64,
        start_pos: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        input_tokens: ?*ggml.Tensor,
        causal: bool,
    ) !*ggml.Tensor {
        log.debug("transformerForward: n_tokens={d}, start_pos={d}, causal={}, kv_cache={}", .{
            n_tokens_i64, start_pos, causal, kv_cache_mgr != null,
        });

        // --- Per-layer embedding 预计算 ---
        self.inp_per_layer = try self.buildPerLayerInputs(self.cur, input_tokens, n_tokens_i64);

        // --- 逐层计算 ---
        for (self.weights.layers, 0..) |*layer, i| {
            self.cur = try self.buildLayer(layer, i, n_tokens_i64, start_pos, kv_cache_mgr, causal);
        }

        // --- 输出投影 ---
        return self.buildOutput();
    }

    // ====================================================================
    // Per-layer embedding 预计算
    // ====================================================================

    fn buildPerLayerInputs(
        self: *Self,
        cur_in: *ggml.Tensor,
        input_tokens: ?*ggml.Tensor,
        n_tokens_i64: i64,
    ) !?*ggml.Tensor {
        const p = self.params;
        const w = self.weights;

        if (p.n_embd_per_layer == 0) return null;

        const n_embd_pl: i64 = @intCast(p.n_embd_per_layer);
        const n_layer_i64: i64 = @intCast(p.base.n_layer);
        const ctx = self.ctx;
        const gf = self.gf;

        // Per-layer token embedding lookup.
        // For text path (input_tokens != null): use actual token IDs.
        // For multimodal path (input_tokens == null): use padding token (ID=0).
        // This matches llama.cpp gemma4.cpp build_inp_per_layer() behavior.
        var inp_pl: *ggml.Tensor = undefined;
        if (input_tokens) |tokens| {
            if (w.per_layer_token_embd) |pl_embd| {
                inp_pl = ggml.getRows(ctx, pl_embd, tokens);
                inp_pl = ggml.reshape3d(ctx, inp_pl, n_embd_pl, n_layer_i64, n_tokens_i64);
                inp_pl = ggml.scale(ctx, inp_pl, @sqrt(@as(f32, @floatFromInt(p.n_embd_per_layer))));
            } else {
                return null;
            }
        } else {
            // Multimodal embedding path without input_tokens: use padding token (ID=0).
            // Matches llama.cpp gemma4.cpp build_inp_per_layer() else branch:
            //   ggml_view_1d(ctx0, model.per_layer_tok_embd, embd_size, 0)
            //   -> cast to F32 -> scale -> reshape to [n_embd_per_layer, n_layer, 1]
            // The padding token is row 0 of per_layer_tok_embd.
            // ggml_add with proj ([n_embd_pl, n_layer, n_tokens]) will broadcast
            // this [n_embd_pl, n_layer, 1] tensor automatically.
            if (w.per_layer_token_embd) |pl_embd| {
                log.debug("buildPerLayerInputs: multimodal path, pl_embd type={s} ne=[{d},{d}]", .{ pl_embd.dataType().name(), pl_embd.ne()[0], pl_embd.ne()[1] });
                const embd_size = pl_embd.ne()[0]; // n_embd_per_layer * n_layer
                const padding = ctx.view1d(pl_embd, embd_size, 0);
                inp_pl = ggml.cast(ctx, padding, .f32);
                inp_pl = ggml.scale(ctx, inp_pl, @sqrt(@as(f32, @floatFromInt(p.n_embd_per_layer))));
                inp_pl = ggml.reshape3d(ctx, inp_pl, n_embd_pl, n_layer_i64, 1);
            } else {
                log.debug("buildPerLayerInputs: multimodal path, per_layer_token_embd is null, returning null", .{});
                return null;
            }
        }

        if (w.per_layer_model_proj) |proj_w| {
            var proj = ggml.mulMat(ctx, proj_w, cur_in);
            proj = ggml.scale(ctx, proj, 1.0 / @sqrt(@as(f32, @floatFromInt(p.base.n_embd))));
            proj = ggml.reshape3d(ctx, proj, n_embd_pl, n_layer_i64, n_tokens_i64);
            proj = ggml.reshape2d(ctx, proj, n_embd_pl, n_layer_i64 * n_tokens_i64);
            proj = ggml.rmsNorm(ctx, proj, p.base.norm_eps);
            if (w.per_layer_proj_norm) |proj_norm| {
                proj = ggml.mul(ctx, proj, ggml.reshape2d(ctx, proj_norm, n_embd_pl, 1));
            }
            proj = ggml.reshape3d(ctx, proj, n_embd_pl, n_layer_i64, n_tokens_i64);

            inp_pl = ggml.add(ctx, proj, inp_pl);
            inp_pl = ggml.scale(ctx, inp_pl, 1.0 / @sqrt(2.0));
        }

        inp_pl = ggml.cont(ctx, ggml.permute(ctx, inp_pl, 0, 2, 1, 3));
        inp_pl.setName("inp_per_layer");

        gf.buildForwardExpand(inp_pl);
        return inp_pl;
    }

    // ====================================================================
    // 单层计算
    // ====================================================================

    fn buildLayer(
        self: *Self,
        layer: *const LayerWeights,
        il: usize,
        n_tokens_i64: i64,
        start_pos: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        causal: bool,
    ) !*ggml.Tensor {
        const p = self.params;
        const ctx = self.ctx;

        const head_dim: i64 = layer.attn_q_norm_weight.ne()[0];
        const n_head: i64 = @divExact(layer.attn_q_weight.ne()[1], head_dim);
        const layer_is_swa = p.is_swa_layer.items[il];

        const freq_base_l: f32 = if (layer_is_swa and p.rope_freq_base_swa > 0) p.rope_freq_base_swa else p.base.rope_theta;
        const rope_dim_full: u32 = if (p.rope_dim_swa > 0 and layer_is_swa)
            @min(p.rope_dim_swa, @as(u32, @intCast(head_dim)))
        else
            @min(p.base.rope_dim, @as(u32, @intCast(head_dim)));
        const rope_dim: i64 = @intCast(rope_dim_full);

        // log.debug("Layer {d}: is_swa={}, has_kv={}, head_dim={d}", .{ il, layer_is_swa, layer.has_kv, head_dim });

        // --- Pre-attention RMSNorm ---
        const attn_input = rms_norm.rmsNorm(ctx, self.cur, layer.attn_norm_weight, p.base.norm_eps);

        // --- Q 投影 + pre-norm + RoPE ---
        var q = ggml.mulMat(ctx, layer.attn_q_weight, attn_input);
        q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);
        q = ggml.reshape2d(ctx, q, head_dim, n_head * n_tokens_i64);
        q = ggml.rmsNorm(ctx, q, p.base.norm_eps);
        q = ggml.mul(ctx, q, ggml.reshape2d(ctx, layer.attn_q_norm_weight, head_dim, 1));
        q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);

        const rope_freqs: ?*ggml.Tensor = if (!layer_is_swa) layer.rope_freqs else null;
        q = ggml.ropeExt(ctx, q, self.pos_tensor, rope_freqs, @intCast(rope_dim), 2, 0, freq_base_l, 1.0, 0.0, 1.0, 0.0, 0.0);

        // --- K/V 投影 + 注意力 ---
        const attn_out = try self.buildAttention(
            layer,
            il,
            q,
            attn_input,
            n_head,
            head_dim,
            layer_is_swa,
            n_tokens_i64,
            start_pos,
            kv_cache_mgr,
            causal,
            freq_base_l,
            rope_freqs,
            rope_dim,
        );

        // 残差
        var cur = ggml.add(ctx, self.cur, attn_out);

        // --- FFN ---
        const ffn_input = rms_norm.rmsNorm(ctx, cur, layer.ffn_norm_weight, p.base.norm_eps);
        const ffn_out = gegluFFN(ctx, ffn_input, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
        const ffn_normed = rms_norm.rmsNorm(ctx, ffn_out, layer.ffn_post_norm_weight, p.base.norm_eps);
        cur = ggml.add(ctx, cur, ffn_normed);

        // --- Per-layer embedding ---
        if (self.inp_per_layer) |inp_pl| {
            if (layer.per_layer_inp_gate) |pl_gate| {
                _ = pl_gate;
                cur = buildPerLayerInjection(ctx, p, layer, cur, inp_pl, il, n_tokens_i64);
            }
        }

        // --- 层输出缩放 ---
        if (layer.out_scale) |scale| {
            cur = ggml.mul(ctx, cur, scale);
        }

        return cur;
    }

    // ====================================================================
    // 注意力子图
    // ====================================================================

    fn buildAttention(
        self: *Self,
        layer: *const LayerWeights,
        il: usize,
        q: *ggml.Tensor,
        attn_input: *ggml.Tensor,
        n_head: i64,
        head_dim: i64,
        layer_is_swa: bool,
        n_tokens_i64: i64,
        start_pos: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        causal: bool,
        freq_base_l: f32,
        rope_freqs: ?*ggml.Tensor,
        rope_dim: i64,
    ) !*ggml.Tensor {
        const p = self.params;
        const ctx = self.ctx;
        const gf = self.gf;

        var attn_out: *ggml.Tensor = undefined;

        if (layer.has_kv) {
            const head_dim_k: i64 = layer.attn_k_norm_weight.ne()[0];
            const n_kv_head: i64 = if (layer.attn_k_weight) |kw|
                @divExact(kw.ne()[1], head_dim_k)
            else
                n_head;

            var k = ggml.mulMat(ctx, layer.attn_k_weight.?, attn_input);
            var v_tensor: *ggml.Tensor = if (layer.attn_v_weight) |vw|
                ggml.mulMat(ctx, vw, attn_input)
            else
                k;

            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);
            v_tensor = ggml.reshape3d(ctx, v_tensor, head_dim_k, n_kv_head, n_tokens_i64);

            // K pre-norm
            k = ggml.reshape2d(ctx, k, head_dim_k, n_kv_head * n_tokens_i64);
            k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
            k = ggml.mul(ctx, k, ggml.reshape2d(ctx, layer.attn_k_norm_weight, head_dim_k, 1));
            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);

            // V RMSNorm
            v_tensor = ggml.rmsNorm(ctx, v_tensor, p.base.norm_eps);

            // RoPE on K
            k = ggml.ropeExt(ctx, k, self.pos_tensor, rope_freqs, @intCast(rope_dim), 2, 0, freq_base_l, 1.0, 0.0, 1.0, 0.0, 0.0);

            // Q reshape for head_dim match
            var n_head_eff = n_head;
            var q_use = q;
            if (head_dim != head_dim_k) {
                n_head_eff = @divExact(n_head * head_dim, head_dim_k);
                q_use = ggml.reshape3d(ctx, q, head_dim_k, n_head_eff, n_tokens_i64);
            }

            // KV Cache
            var k_attn = k;
            var v_attn = v_tensor;
            if (kv_cache_mgr) |cache| {
                cache.setKv(ctx, gf, il, k, v_tensor, @intCast(n_tokens_i64));
                k_attn = cache.getKView(ctx, il);
                v_attn = cache.getVView(ctx, il);
            }

            const cache_len: i64 = if (kv_cache_mgr) |cache| @as(i64, @intCast(cache.currentLen())) else n_tokens_i64;

            attn_out = attention.scaledDotProductAttention(ctx, q_use, k_attn, v_attn, .{
                .n_head = n_head_eff,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim_k,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = p.f_attention_scale,
                .attn_logit_softcap = p.attn_logit_softcapping,
                .causal = causal,
            }, if (layer_is_swa) @as(i64, @intCast(p.n_swa)) else null);

            attn_out = ggml.reshape2d(ctx, attn_out, n_head * head_dim, n_tokens_i64);
        } else {
            const kv_layer_idx = findKVLayer(p, il);
            const cache = kv_cache_mgr orelse @panic("Shared KV layer requires KV cache");
            const k_cache = cache.getKView(ctx, kv_layer_idx);
            const v_cache = cache.getVView(ctx, kv_layer_idx);
            const head_dim_k_cache: i64 = k_cache.ne()[0];
            const n_kv_head_cache: i64 = k_cache.ne()[1];

            var n_head_eff = n_head;
            var q_use = q;
            if (head_dim != head_dim_k_cache) {
                n_head_eff = @divExact(n_head * head_dim, head_dim_k_cache);
                q_use = ggml.reshape3d(ctx, q, head_dim_k_cache, n_head_eff, n_tokens_i64);
            }

            const cache_len: i64 = @as(i64, @intCast(cache.currentLen()));

            attn_out = attention.scaledDotProductAttention(ctx, q_use, k_cache, v_cache, .{
                .n_head = n_head_eff,
                .n_kv_head = n_kv_head_cache,
                .head_dim = head_dim_k_cache,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = p.f_attention_scale,
                .attn_logit_softcap = p.attn_logit_softcapping,
                .causal = causal,
            }, if (layer_is_swa) @as(i64, @intCast(p.n_swa)) else null);

            attn_out = ggml.reshape2d(ctx, attn_out, n_head * head_dim, n_tokens_i64);
        }

        // Output projection + post-norm
        attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);
        attn_out = rms_norm.rmsNorm(ctx, attn_out, layer.attn_post_norm_weight, p.base.norm_eps);
        return attn_out;
    }

    // ====================================================================
    // 输出投影
    // ====================================================================

    fn buildOutput(self: *Self) *ggml.Tensor {
        const p = self.params;
        const w = self.weights;
        const ctx = self.ctx;
        const gf = self.gf;

        var cur = rms_norm.rmsNorm(ctx, self.cur, w.base.output_norm_weight, p.base.norm_eps);
        cur.setName("output_norm");

        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);

        if (p.final_logit_softcapping > 0.0) {
            const cap = p.final_logit_softcapping;
            logits_tensor = ggml.scale(ctx, logits_tensor, 1.0 / cap);
            logits_tensor = ggml.tanh(ctx, logits_tensor);
            logits_tensor = ggml.scale(ctx, logits_tensor, cap);
        }
        logits_tensor.setName("logits");

        gf.buildForwardExpand(logits_tensor);
        return logits_tensor;
    }
};

// ============================================================================
// Gemma 4 模型
// ============================================================================

pub const Gemma4Model = struct {
    params: Gemma4Params,
    weights: Gemma4Weights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *Gemma4Model, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.params = try parseParams(gguf_file, allocator);

        const mem_size_estimate = estimateMemSize(gguf_file);
        self.ctx_weights = try ggml.Context.init(mem_size_estimate);

        self.weights = try loadWeights(gguf_file, self.ctx_weights, &self.params, allocator);
    }

    pub fn deinit(self: *Gemma4Model, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
        self.params.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn getParams(self: *const Gemma4Model) *const model.ModelParams {
        return &self.params.base;
    }

    pub fn getWeights(self: *const Gemma4Model) *const model.ModelWeights {
        return &self.weights.base;
    }

    // ====================================================================
    // Forward 方法 — 委托给 Gemma4Graph
    // ====================================================================

    /// Forward with embedding override (multimodal).  Delegates to Gemma4Graph.
    pub fn forwardWithEmbdOverride(
        self: *Gemma4Model,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        embd_override: *ggml.Tensor,
        embd_offset: i32,
        causal: bool,
    ) !*ggml.Tensor {
        var g = Gemma4Graph{
            .ctx = ctx,
            .gf = graph,
            .params = &self.params,
            .weights = &self.weights,
            .cur = undefined,
            .pos_tensor = undefined,
            .inp_per_layer = null,
        };
        return try g.buildWithEmbd(input_tokens, n_tokens, kv_cache_mgr, start_pos, embd_override, embd_offset, causal);
    }

    /// Media-only forward.  Delegates to Gemma4Graph.
    pub fn forwardMediaOnly(
        self: *Gemma4Model,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        embd_override: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        input_tokens: *ggml.Tensor,
    ) !*ggml.Tensor {
        var g = Gemma4Graph{
            .ctx = ctx,
            .gf = graph,
            .params = &self.params,
            .weights = &self.weights,
            .cur = undefined,
            .pos_tensor = undefined,
            .inp_per_layer = null,
        };
        return try g.buildMediaOnly(embd_override, n_tokens, kv_cache_mgr, start_pos, input_tokens);
    }

    /// Media forward adapter (MediaForwardFn signature).  Delegates to Gemma4Graph.
    pub fn mediaForward(
        self: *Gemma4Model,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        embd_override: *ggml.Tensor,
        embd_offset: i32,
        causal: bool,
    ) !*ggml.Tensor {
        // —— 音频嵌入诊断 ——
        {
            const eo_data = embd_override.dataF32();
            const n_total: usize = @as(usize, @intCast(embd_override.ne()[0] * embd_override.ne()[1]));
            const n_preview: usize = @min(n_total, 8);
            var all_zero = true;
            var has_nan = false;
            for (eo_data[0..n_total]) |v| {
                if (v != 0.0) all_zero = false;
                if (std.math.isNan(v)) has_nan = true;
            }
            log.debug("mediaForward: ne=[{d},{d}] start_pos={d} n_tokens={d} causal={}", .{
                embd_override.ne()[0], embd_override.ne()[1], start_pos, n_tokens, causal,
            });
            log.debug("  embed preview[0..{d}]: {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}", .{
                @min(n_preview, @as(usize, 8)),
                eo_data[0], if (n_total > 1) eo_data[1] else @as(f32, 0),
                if (n_total > 2) eo_data[2] else @as(f32, 0), if (n_total > 3) eo_data[3] else @as(f32, 0),
                if (n_total > 4) eo_data[4] else @as(f32, 0), if (n_total > 5) eo_data[5] else @as(f32, 0),
                if (n_total > 6) eo_data[6] else @as(f32, 0), if (n_total > 7) eo_data[7] else @as(f32, 0),
            });
            log.debug("  all_zero={} has_nan={}", .{ all_zero, has_nan });
            if (all_zero) log.warn("  ⚠ embd_override is ALL ZEROS!", .{});
            if (has_nan) log.warn("  ⚠ embd_override contains NaN!", .{});
        }

        _ = embd_offset;
        var g = Gemma4Graph{
            .ctx = ctx,
            .gf = graph,
            .params = &self.params,
            .weights = &self.weights,
            .cur = undefined,
            .pos_tensor = undefined,
            .inp_per_layer = null,
        };
        return try g.buildMediaOnly(embd_override, n_tokens, kv_cache_mgr, start_pos, input_tokens);
    }

    /// Pure text forward.  Delegates to Gemma4Graph.
    pub fn forward(
        self: *Gemma4Model,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
    ) !*ggml.Tensor {
        var g = Gemma4Graph{
            .ctx = ctx,
            .gf = graph,
            .params = &self.params,
            .weights = &self.weights,
            .cur = undefined,
            .pos_tensor = undefined,
            .inp_per_layer = null,
        };
        return try g.build(input_tokens, n_tokens, kv_cache_mgr, start_pos);
    }

    pub fn buildGraph(
        self: *Gemma4Model,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*anyopaque,
        start_pos: i32,
    ) !*ggml.Tensor {
        const gctx = builder.ctx;
        const graph = builder.gf;
        const kv_cache_mgr: ?*kv_cache.KVCache = if (mem_ctx) |ptr| @ptrCast(@alignCast(ptr)) else null;
        return self.forward(gctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos);
    }

    pub const vtable = model.ModelVTable{
        .deinit = deinitAdapter,
        .buildGraph = buildGraphAdapter,
        .getParams = getParamsAdapter,
        .resetSSMStates = resetSSMStatesAdapter,
    };

    fn deinitAdapter(data: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Gemma4Model = @ptrCast(@alignCast(data));
        self.weights.deinit(allocator);
        self.params.deinit(allocator);
        self.ctx_weights.deinit();
        allocator.destroy(self);
    }

    fn buildGraphAdapter(
        data: *anyopaque,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*anyopaque,
        start_pos: i32,
    ) anyerror!*ggml.Tensor {
        const self: *Gemma4Model = @ptrCast(@alignCast(data));
        return self.buildGraph(builder, input_tokens, n_tokens, mem_ctx, start_pos);
    }

    fn getParamsAdapter(data: *anyopaque) *const model.ModelParams {
        const self: *Gemma4Model = @ptrCast(@alignCast(data));
        return self.getParams();
    }

    fn resetSSMStatesAdapter(data: *anyopaque) void {
        _ = data;
    }
};
/// GeGLU FFN: gelu(h @ W_gate) * (h @ W_up) -> down
/// Gemma 4 uses GeGLU (GELU-gated Linear Unit) activation,
/// matching llama.cpp's LLM_FFN_GELU.
fn gegluFFN(
    ctx: *ggml.Context,
    x: *ggml.Tensor,
    gate_w: *ggml.Tensor,
    up_w: *ggml.Tensor,
    down_w: *ggml.Tensor,
) *ggml.Tensor {
    const gate_out = ggml.mulMat(ctx, gate_w, x);
    const up_out = ggml.mulMat(ctx, up_w, x);
    const gelu_out = ggml.gelu(ctx, gate_out);
    const mul_out = ggml.mul(ctx, gelu_out, up_out);
    return ggml.mulMat(ctx, down_w, mul_out);
}

/// 查找非 KV 层应该复用哪个层的 KV
/// 参考 llama.cpp gemma4 的 reuse lambda:
///   if il >= n_layer_kv_from_start:
///     // Find the last KV layer that matches the SWA/non-SWA pattern
///     for (int j = n_layer_kv_from_start - 1; j >= 0; j--) {
///         if (is_swa_layer[il] == is_swa_layer[j]) return j;
///     }
///     return n_layer_kv_from_start - 1;
/// 调用者保证 layer_idx >= n_layer_kv_from_start（即该层没有自己的 KV）
fn findKVLayer(p: *const Gemma4Params, layer_idx: usize) usize {
    if (layer_idx >= p.n_layer_kv_from_start and p.n_layer_kv_from_start > 0) {
        const is_swa = p.is_swa_layer.items[layer_idx];
        // 从最后一个 KV 层向前搜索，找到与当前层 SWA 类型匹配的层
        var j: usize = p.n_layer_kv_from_start - 1;
        while (true) {
            if (p.is_swa_layer.items[j] == is_swa) {
                return j;
            }
            if (j == 0) break;
            j -= 1;
        }
        // Fallback: 返回最后一个 KV 层
        return p.n_layer_kv_from_start - 1;
    }
    // 理论上不应该走到这里，因为该层应该 has_kv==false
    // 返回 0 作为 fallback（第 0 层通常有 KV）
    return 0;
}

/// Per-layer embedding injection at position `il` in the transformer layer.
/// Used by Gemma4Graph.buildLayer.
fn buildPerLayerInjection(
    ctx: *ggml.Context,
    p: *const Gemma4Params,
    layer: *const LayerWeights,
    cur: *ggml.Tensor,
    inp_pl: *ggml.Tensor,
    il: usize,
    n_tokens_i64: i64,
) *ggml.Tensor {
    const n_embd_pl: i64 = @intCast(p.n_embd_per_layer);
    const pe_in = cur;

    var c = ggml.mulMat(ctx, layer.per_layer_inp_gate.?, cur);
    c = ggml.gelu(ctx, c);

    const elem_size = @sizeOf(f32);
    const slice_offset: usize = @as(usize, @intCast(il)) * @as(usize, @intCast(n_embd_pl)) * @as(usize, @intCast(n_tokens_i64)) * elem_size;
    const inp_this = ctx.view2d(inp_pl, n_embd_pl, n_tokens_i64, inp_pl.nb()[1], slice_offset);

    c = ggml.mul(ctx, c, inp_this);
    c = ggml.mulMat(ctx, layer.per_layer_proj.?, c);
    c = rms_norm.rmsNorm(ctx, c, layer.per_layer_post_norm.?, p.base.norm_eps);

    return ggml.add(ctx, pe_in, c);
}

// ============================================================================
// 参数解析
// ============================================================================

pub fn parseParams(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Gemma4Params {
    var p = Gemma4Params{
        .is_swa_layer = try std.ArrayList(bool).initCapacity(allocator, 0),
    };
    errdefer p.is_swa_layer.deinit(allocator);

    p.base.n_vocab = gguf_file.getU32("gemma4.vocab_size") orelse
        gguf_file.getU32("llama.vocab_size") orelse
        blk: {
            if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
                if (val.value_type == .array) break :blk @intCast(val.array_val.len);
            }
            break :blk 0;
        };
    p.base.n_embd = gguf_file.getU32("gemma4.embedding_length") orelse
        gguf_file.getU32("llama.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("gemma4.attention.head_count") orelse
        gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("gemma4.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse 0;

    // head_count_kv is an array in Gemma 4 (per-layer KV heads)
    if ((@constCast(gguf_file)).getU32Array("gemma4.attention.head_count_kv")) |kv_arr| {
        var max_kv: u32 = 0;
        for (kv_arr) |h| {
            if (h > max_kv) max_kv = h;
        }
        p.base.n_kv_head = max_kv;
    } else {
        p.base.n_kv_head = gguf_file.getU32("gemma4.attention.head_count_kv") orelse
            gguf_file.getU32("llama.attention.head_count_kv") orelse
            gguf_file.getU32("gemma4.head_count_kv") orelse
            gguf_file.getU32("llama.head_count_kv") orelse p.base.n_head;
    }

    p.base.n_layer = gguf_file.getU32("gemma4.block_count") orelse
        gguf_file.getU32("llama.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("gemma4.feed_forward_length") orelse
        gguf_file.getU32("llama.feed_forward_length") orelse 0;

    // Fallback: if n_ff is still 0, derive from n_embd (Gemma 4 uses 4 * n_embd)
    if (p.base.n_ff == 0 and p.base.n_embd > 0) {
        p.base.n_ff = 4 * p.base.n_embd;
        log.info("n_ff not specified in GGUF, using default 4 * n_embd = {d}", .{p.base.n_ff});
    }

    // Gemma 4 uses explicit key_length/value_length for head_dim
    // n_head_dim_k/n_head_dim_v 使用 SWA 维度（所有层的 K/V norm 维度一致）
    // n_head_dim 保持 full attention 维度（用于 rope_dim 等）
    const key_len = gguf_file.getU32("gemma4.attention.key_length") orelse
        gguf_file.getU32("llama.attention.key_length") orelse 0;
    const key_len_swa = gguf_file.getU32("gemma4.attention.key_length_swa") orelse 0;
    const val_len_swa = gguf_file.getU32("gemma4.attention.value_length_swa") orelse 0;

    if (key_len > 0) {
        p.base.n_head_dim = key_len;
    } else if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }
    // KV cache 使用 SWA 维度（所有层的 K norm 均为 key_length_swa）
    p.base.n_head_dim_k = if (key_len_swa > 0) key_len_swa else p.base.n_head_dim;
    p.base.n_head_dim_v = if (val_len_swa > 0) val_len_swa else p.base.n_head_dim;

    p.base.max_seq_len = gguf_file.getU32("gemma4.context_length") orelse
        gguf_file.getU32("llama.context_length") orelse 32768;
    p.base.rope_theta = gguf_file.getF32("gemma4.rope.freq_base") orelse
        gguf_file.getF32("llama.rope.freq_base") orelse 10000.0;
    p.base.rope_dim = gguf_file.getU32("gemma4.rope.dimension_count") orelse
        gguf_file.getU32("llama.rope.dimension_count") orelse
        @divExact(p.base.n_head_dim, @as(u32, 2));
    // SWA 层可能使用不同的 rope_dim（因为 head_dim 更小）
    p.rope_dim_swa = gguf_file.getU32("gemma4.rope.dimension_count_swa") orelse
        gguf_file.getU32("llama.rope.dimension_count_swa") orelse p.base.rope_dim;

    p.base.norm_eps = gguf_file.getF32("gemma4.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-6;
    p.base.model_name = gguf_file.getString("general.name") orelse "";
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gemma";

    // Gemma 4 特定参数
    p.n_swa = gguf_file.getU32("gemma4.attention.sliding_window") orelse 0;
    p.rope_freq_base_swa = gguf_file.getF32("gemma4.rope.freq_base_swa") orelse p.base.rope_theta;
    p.f_attention_scale = 1.0;

    // 解析 SWA pattern（Gemma 4 使用 bool 数组）
    const n_layer_i64 = @as(usize, @intCast(p.base.n_layer));
    if ((@constCast(gguf_file)).getBoolArray("gemma4.attention.sliding_window_pattern")) |swa_pattern| {
        try p.is_swa_layer.ensureTotalCapacity(allocator, n_layer_i64);
        for (swa_pattern, 0..) |is_swa, idx| {
            if (idx < n_layer_i64) {
                p.is_swa_layer.appendAssumeCapacity(is_swa);
            }
        }
        while (p.is_swa_layer.items.len < n_layer_i64) {
            p.is_swa_layer.appendAssumeCapacity(false);
        }
    } else if (p.n_swa > 0) {
        // 没有显式 pattern，使用周期模式：每 6 层一个全注意力层
        const swa_period: u32 = 6;
        try p.is_swa_layer.ensureTotalCapacity(allocator, n_layer_i64);
        for (0..n_layer_i64) |idx| {
            const is_swa = (idx % swa_period) != 0;
            p.is_swa_layer.appendAssumeCapacity(is_swa);
        }
    } else {
        try p.is_swa_layer.ensureTotalCapacity(allocator, n_layer_i64);
        for (0..n_layer_i64) |_| {
            p.is_swa_layer.appendAssumeCapacity(false);
        }
    }

    // 共享 KV 层
    p.n_kv_shared_layers = gguf_file.getU32("gemma4.attention.shared_kv_layers") orelse 0;
    if (p.base.n_layer > p.n_kv_shared_layers) {
        p.n_layer_kv_from_start = p.base.n_layer - p.n_kv_shared_layers;
    } else {
        p.n_layer_kv_from_start = p.base.n_layer;
    }

    p.final_logit_softcapping = gguf_file.getF32("gemma4.final_logit_softcapping") orelse 0.0;
    p.attn_logit_softcapping = gguf_file.getF32("gemma4.attn_logit_softcapping") orelse 50.0;
    p.n_embd_per_layer = gguf_file.getU32("gemma4.embedding_length_per_layer_input") orelse
        gguf_file.getU32("gemma4.embedding_length_per_layer") orelse 0;

    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required Gemma 4 parameters", .{});
        return error.InvalidModelParams;
    }

    log.info("Gemma4: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}, swa={d}, shared_kv={d}, softcap={d}, embd_per_layer={d}", .{
        p.base.n_vocab,            p.base.n_embd,      p.base.n_head, p.base.n_kv_head,
        p.base.n_layer,            p.base.n_ff,        p.n_swa,       p.n_kv_shared_layers,
        p.final_logit_softcapping, p.n_embd_per_layer,
    });

    return p;
}

// ============================================================================
// 权重加载
// ============================================================================

fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *Gemma4Params,
    allocator: std.mem.Allocator,
) !Gemma4Weights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Gemma 4 weights ({d} layers)...", .{n_layer});

    const token_embd = findOrCreateTensor(ctx, gguf_file, "token_embd.weight") catch |err| {
        log.err("Failed to load token_embd.weight: {}", .{err});
        return error.MissingWeight;
    };
    token_embd.setName("token_embd.weight");

    const output_weight = findOrCreateTensor(ctx, gguf_file, "output.weight") catch null;
    if (output_weight) |ow| ow.setName("output.weight");

    const output_norm_weight = findOrCreateTensor(ctx, gguf_file, "output_norm.weight") catch |err| {
        log.err("Failed to load output_norm.weight: {}", .{err});
        return error.MissingWeight;
    };
    output_norm_weight.setName("output_norm.weight");

    // Per-layer embedding global weights (only when n_embd_per_layer > 0)
    const per_layer_token_embd: ?*ggml.Tensor = if (params.n_embd_per_layer > 0) blk: {
        const t = findOrCreateTensor(ctx, gguf_file, "per_layer_token_embd.weight") catch null;
        if (t) |tt| tt.setName("per_layer_token_embd.weight");
        break :blk t;
    } else null;
    const per_layer_model_proj: ?*ggml.Tensor = if (params.n_embd_per_layer > 0) blk: {
        const t = findOrCreateTensor(ctx, gguf_file, "per_layer_model_proj.weight") catch null;
        if (t) |tt| tt.setName("per_layer_model_proj.weight");
        break :blk t;
    } else null;
    const per_layer_proj_norm: ?*ggml.Tensor = if (params.n_embd_per_layer > 0) blk: {
        const t = findOrCreateTensor(ctx, gguf_file, "per_layer_proj_norm.weight") catch null;
        if (t) |tt| tt.setName("per_layer_proj_norm.weight");
        break :blk t;
    } else null;

    if (per_layer_token_embd != null) {
        log.info("Gemma4: per-layer embedding enabled (n_embd_per_layer={d})", .{params.n_embd_per_layer});
    }

    var layers = try allocator.alloc(LayerWeights, n_layer);
    var layers_loaded: usize = 0;
    errdefer {
        for (0..layers_loaded) |j| {
            allocator.free(layers[j].prefix);
        }
        allocator.free(layers);
    }

    // Gemma 4: rope_freqs is per-layer (blk.0.rope_freqs.weight), shared
    // across all full-attention layers (llama.cpp uses TENSOR_DUPLICATED).
    // Try multiple possible names for rope_freqs.
    var global_rope_freqs = findOrCreateTensor(ctx, gguf_file, "blk.0.rope_freqs.weight") catch null;
    if (global_rope_freqs == null) {
        global_rope_freqs = findOrCreateTensor(ctx, gguf_file, "rope_freqs.weight") catch null;
    }
    if (global_rope_freqs == null) {
        global_rope_freqs = findOrCreateTensor(ctx, gguf_file, "blk.0.rope_freqs") catch null;
    }
    if (global_rope_freqs) |_| {
        log.info("Gemma4: global_rope_freqs loaded successfully", .{});
    } else {
        log.warn("Gemma4: global_rope_freqs not found, full-attention layers will use standard RoPE", .{});
    }

    // Derive n_ff from first layer's FFN gate weight shape
    var n_ff_derived: i64 = params.base.n_ff;

    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});

        // Detect has_kv: use n_layer_kv_from_start to determine which layers have their own KV.
        // Do NOT rely on attn_k.weight existence in GGUF, as E2B models may store it for all layers.
        const has_kv: bool = i < params.n_layer_kv_from_start;
        const k_weight = if (has_kv)
            loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch null
        else
            null;
        const v_weight = if (has_kv)
            loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch null
        else
            null;

        const q_norm = loadLayerWeight(ctx, gguf_file, prefix, "attn_q_norm.weight") catch |err| {
            log.warn("Layer {d}: missing attn_q_norm.weight: {}", .{ i, err });
            continue;
        };
        const k_norm = if (has_kv)
            loadLayerWeight(ctx, gguf_file, prefix, "attn_k_norm.weight") catch null
        else
            null;

        const out_scale = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "layer_output_scale.weight") catch null;

        // Per-layer embedding weights
        const per_layer_inp_gate: ?*ggml.Tensor = if (params.n_embd_per_layer > 0)
            loadLayerWeight(ctx, gguf_file, prefix, "inp_gate.weight") catch null
        else
            null;
        const per_layer_proj: ?*ggml.Tensor = if (params.n_embd_per_layer > 0)
            loadLayerWeight(ctx, gguf_file, prefix, "proj.weight") catch null
        else
            null;
        const per_layer_post_norm: ?*ggml.Tensor = if (params.n_embd_per_layer > 0)
            loadLayerWeight(ctx, gguf_file, prefix, "post_norm.weight") catch null
        else
            null;

        const ffn_gate = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight");

        // Derive n_ff from first layer
        if (i == 0) {
            n_ff_derived = ffn_gate.ne()[1];
            log.info("Layer 0 FFN gate: [{d}, {d}], n_embd={d}", .{ ffn_gate.ne()[0], ffn_gate.ne()[1], params.base.n_embd });
        }

        // Gemma 4: all full-attention layers share the global rope_freqs
        layers[i] = LayerWeights{
            .prefix = prefix,
            .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
            .ffn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight"),
            .attn_q_norm_weight = q_norm,
            .attn_k_norm_weight = if (k_norm) |kn| kn else q_norm,
            .attn_post_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "post_attention_norm.weight"),
            .ffn_post_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "post_ffw_norm.weight"),
            .attn_q_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight"),
            .attn_k_weight = k_weight,
            .attn_v_weight = v_weight,
            .attn_output_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight"),
            .ffn_gate_weight = ffn_gate,
            .ffn_up_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight"),
            .ffn_down_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight"),
            .out_scale = out_scale,
            .rope_freqs = if (!params.is_swa_layer.items[i]) global_rope_freqs else null,
            .per_layer_inp_gate = per_layer_inp_gate,
            .per_layer_proj = per_layer_proj,
            .per_layer_post_norm = per_layer_post_norm,
            .has_kv = has_kv,
        };

        layers_loaded = i + 1;
    }

    // Update n_ff from actual weight shape if metadata was missing
    const n_embd_per_layer = params.n_embd_per_layer;
    const n_kv_layers = blk: {
        var count: u32 = 0;
        for (layers) |l| {
            if (l.has_kv) count += 1;
        }
        break :blk count;
    };

    log.info("Gemma4 weights: {d} layers, {d} with KV, n_ff={d}, per_layer={d}", .{
        n_layer, n_kv_layers, n_ff_derived, n_embd_per_layer,
    });

    return Gemma4Weights{
        .base = .{
            .params = params.base,
            .token_embd = token_embd,
            .output_weight = output_weight,
            .output_norm_weight = output_norm_weight,
        },
        .layers = layers,
        .per_layer_token_embd = per_layer_token_embd,
        .per_layer_model_proj = per_layer_model_proj,
        .per_layer_proj_norm = per_layer_proj_norm,
    };
}

fn loadLayerWeight(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, prefix: []const u8, name: []const u8) !*ggml.Tensor {
    var buf: [256]u8 = undefined;
    const full_name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, name });
    buf[full_name.len] = 0;
    return findOrCreateTensor(ctx, gguf_file, buf[0..full_name.len :0]);
}

fn findOrCreateTensor(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    if (gguf_file.findTensor(name)) |info| {
        const n_dims = info.n_dims;
        const dims = info.dims;
        const typ: ggml.Type = @enumFromInt(@intFromEnum(info.data_type));
        ctx.setNoAlloc(false);
        const tensor = switch (n_dims) {
            1 => try ctx.newTensor1d(typ, @intCast(dims[0])),
            2 => try ctx.newTensor2d(typ, @intCast(dims[0]), @intCast(dims[1])),
            3 => try ctx.newTensor3d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2])),
            4 => try ctx.newTensor4d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2]), @intCast(dims[3])),
            else => return error.UnsupportedTensorDims,
        };
        ctx.setNoAlloc(true);

        tensor.setName(@ptrCast(name));

        const tensor_data = gguf_file.getTensorData(info);
        const tensor_bytes = tensor.dataBytes();
        if (tensor_bytes.len != tensor_data.len) {
            log.warn("Tensor '{s}' size mismatch: expected {d} bytes, got {d} bytes", .{ name, tensor_bytes.len, tensor_data.len });
        }
        @memcpy(tensor_bytes, tensor_data);

        return tensor;
    }
    return error.TensorNotFound;
}

fn estimateMemSize(gguf_file: *const gguf.GGUFFile) usize {
    const raw_data_size = gguf_file.totalTensorDataSize();
    const n_tensors = gguf_file.tensors.items.len;
    // ggml 内部每个张量需要: ggml_tensor (~256B) + ggml_object (~64B) + 对齐
    // 使用 384 字节/tensor 以确保覆盖
    const overhead: usize = n_tensors * 384;
    const with_overhead = raw_data_size + overhead;
    // 33% 安全余量 + 64MB 固定缓冲
    const total = with_overhead + with_overhead / 3 + 64 * 1024 * 1024;

    log.info("Estimated Gemma 4 weights memory: {d} MB (raw: {d} MB, {d} tensors)", .{
        total / (1024 * 1024),
        raw_data_size / (1024 * 1024),
        n_tensors,
    });
    return total;
}

const testing = std.testing;

test "Gemma4Params defaults" {
    var p = Gemma4Params{
        .is_swa_layer = std.ArrayList(bool).init(testing.allocator),
    };
    defer p.deinit();
    try testing.expectEqual(@as(u32, 0), p.base.n_vocab);
    try testing.expectEqual(@as(f32, 1.0), p.f_attention_scale);
}
