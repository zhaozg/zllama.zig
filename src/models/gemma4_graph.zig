//! Gemma 4 图构建类 — 管理 transformer 前向传播的图构建。
//!
//! 从 gemma4.zig 拆分（refact.md §1）以保持文件 ≤600 行。
//! 参考 llama.cpp gemma4.cpp 的 llm_build_context 模式。
//!
//! Gemma4Graph 持有图构建过程中所有中间张量引用，并将 transformerForward
//! 分解为 buildPerLayerInputs / buildLayer / buildAttention / buildOutput 等命名方法。

const std = @import("std");
const ggml = @import("ggml");
const kv_cache = @import("kv_cache");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const embed = @import("embed");
const attention = @import("attention");

const log = std.log.scoped(.model_gemma4);

const Gemma4Params = @import("./gemma4.zig").Gemma4Params;
const Gemma4Weights = @import("./gemma4.zig").Gemma4Weights;
const LayerWeights = @import("./gemma4.zig").LayerWeights;

// ============================================================================
// Gemma 4 图构建类
// ============================================================================

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

        var all_embd = embed.tokenEmbedding(self.ctx, w.base.token_embd, input_tokens);
        all_embd = ggml.scale(self.ctx, all_embd, @sqrt(@as(f32, @floatFromInt(p.base.n_embd))));
        const scaled_override = embd_override;

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
    pub fn buildMediaOnly(
        self: *Self,
        embd_override: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        input_tokens: *ggml.Tensor,
    ) !*ggml.Tensor {
        _ = input_tokens;
        const p = self.params;
        const w = self.weights;
        const n_tokens_i64: i64 = n_tokens;

        const override_embd_dim: i64 = embd_override.ne()[0];
        const model_n_embd: i64 = @intCast(p.base.n_embd);
        if (override_embd_dim != model_n_embd) {
            log.err("buildMediaOnly: embedding dim mismatch — override={d} model={d}", .{ override_embd_dim, model_n_embd });
            return error.EmbeddingDimensionMismatch;
        }

        const scaled = embd_override;
        scaled.setName("inp_media");

        const pos_tensor = rope.buildPositionTensor(self.ctx, @intCast(n_tokens), start_pos);

        var g = Self.init(self.ctx, self.gf, p, w, scaled, pos_tensor);
        return try g.transformerForward(n_tokens_i64, start_pos, kv_cache_mgr, null, false);
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
        // Always build per-layer embedding inputs — matching llama.cpp gemma4.cpp:194-200.
        // buildPerLayerInputs internally handles both token-based (input_tokens != null)
        // and multimodal (input_tokens == null, padding token) paths.
        self.inp_per_layer = try self.buildPerLayerInputs(self.cur, input_tokens, n_tokens_i64);

        for (self.weights.layers, 0..) |*layer, i| {
            self.cur = try self.buildLayer(layer, i, n_tokens_i64, start_pos, kv_cache_mgr, causal);
        }

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
            if (w.per_layer_token_embd) |pl_embd| {
                const embd_size = pl_embd.ne()[0];
                const padding = ctx.view1d(pl_embd, embd_size, 0);
                inp_pl = ggml.cast(ctx, padding, .f32);
                inp_pl = ggml.scale(ctx, inp_pl, @sqrt(@as(f32, @floatFromInt(p.n_embd_per_layer))));
                inp_pl = ggml.reshape3d(ctx, inp_pl, n_embd_pl, n_layer_i64, 1);
            } else {
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

        const attn_input = rms_norm.rmsNorm(ctx, self.cur, layer.attn_norm_weight, p.base.norm_eps);

        var q = ggml.mulMat(ctx, layer.attn_q_weight, attn_input);
        q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);
        q = ggml.reshape2d(ctx, q, head_dim, n_head * n_tokens_i64);
        q = ggml.rmsNorm(ctx, q, p.base.norm_eps);
        q = ggml.mul(ctx, q, ggml.reshape2d(ctx, layer.attn_q_norm_weight, head_dim, 1));
        q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);

        const rope_freqs: ?*ggml.Tensor = if (!layer_is_swa) layer.rope_freqs else null;
        q = ggml.ropeExt(ctx, q, self.pos_tensor, rope_freqs, @intCast(rope_dim), 2, 0, freq_base_l, 1.0, 0.0, 1.0, 0.0, 0.0);

        const attn_out = try self.buildAttention(layer, il, q, attn_input, n_head, head_dim, layer_is_swa, n_tokens_i64, start_pos, kv_cache_mgr, causal, freq_base_l, rope_freqs, rope_dim);

        var cur = ggml.add(ctx, self.cur, attn_out);

        const ffn_input = rms_norm.rmsNorm(ctx, cur, layer.ffn_norm_weight, p.base.norm_eps);
        const ffn_out = gegluFFN(ctx, ffn_input, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
        const ffn_normed = rms_norm.rmsNorm(ctx, ffn_out, layer.ffn_post_norm_weight, p.base.norm_eps);
        cur = ggml.add(ctx, cur, ffn_normed);

        if (self.inp_per_layer) |inp_pl| {
            if (layer.per_layer_inp_gate) |pl_gate| {
                _ = pl_gate;
                cur = buildPerLayerInjection(ctx, p, layer, cur, inp_pl, il, n_tokens_i64);
            }
        }

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
            const n_kv_head: i64 = if (layer.attn_k_weight) |kw| @divExact(kw.ne()[1], head_dim_k) else n_head;

            var k = ggml.mulMat(ctx, layer.attn_k_weight.?, attn_input);
            var v_tensor: *ggml.Tensor = if (layer.attn_v_weight) |vw| ggml.mulMat(ctx, vw, attn_input) else k;

            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);
            v_tensor = ggml.reshape3d(ctx, v_tensor, head_dim_k, n_kv_head, n_tokens_i64);

            k = ggml.reshape2d(ctx, k, head_dim_k, n_kv_head * n_tokens_i64);
            k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
            k = ggml.mul(ctx, k, ggml.reshape2d(ctx, layer.attn_k_norm_weight, head_dim_k, 1));
            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);

            v_tensor = ggml.rmsNorm(ctx, v_tensor, p.base.norm_eps);

            k = ggml.ropeExt(ctx, k, self.pos_tensor, rope_freqs, @intCast(rope_dim), 2, 0, freq_base_l, 1.0, 0.0, 1.0, 0.0, 0.0);

            var n_head_eff = n_head;
            var q_use = q;
            if (head_dim != head_dim_k) {
                n_head_eff = @divExact(n_head * head_dim, head_dim_k);
                q_use = ggml.reshape3d(ctx, q, head_dim_k, n_head_eff, n_tokens_i64);
            }

            var k_attn = k;
            var v_attn = v_tensor;
            if (kv_cache_mgr) |cache| {
                if (causal) {
                    // Causal pass: write K/V to cache, then read back with history
                    cache.setKv(ctx, gf, il, k, v_tensor, @intCast(n_tokens_i64));
                    k_attn = cache.getKView(ctx, il);
                    v_attn = cache.getVView(ctx, il);
                } else {
                    // Non-causal (media) pass: write K/V to cache so the suffix
                    // causal pass can attend to media tokens. Use the current
                    // cache length as the write offset.
                    cache.setKv(ctx, gf, il, k, v_tensor, @intCast(n_tokens_i64));
                    // For non-causal attention, attend to all media tokens directly
                    // (not through cache views, since cache may not be large enough
                    // for the full media sequence in a single view).
                }
            }

            // Use the actual K tensor's sequence length (ne[2]) for cache_len,
            // not the global currentLen(), because getKView may clamp to per-layer max_seq_len.
            const cache_len: i64 = if (causal and kv_cache_mgr != null) blk: {
                break :blk k_attn.ne()[2];
            } else if (!causal and kv_cache_mgr != null) blk: {
                // For non-causal pass, use n_tokens as the effective length
                // since we're attending to all media tokens simultaneously.
                break :blk n_tokens_i64;
            } else n_tokens_i64;

            // For SWA layers, the cache view may be truncated to max_seq_len (window size).
            // cache_start_abs is the absolute position of the first element in the cache view.
            // When cache_len < currentLen(), the view starts at (currentLen - cache_len).
            const cache_start_abs: i64 = if (causal and layer_is_swa and kv_cache_mgr != null) blk: {
                const current_len = kv_cache_mgr.?.currentLen();
                if (current_len > cache_len) {
                    break :blk @as(i64, @intCast(current_len)) - cache_len;
                }
                break :blk 0;
            } else 0;

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
                .cache_start_abs = cache_start_abs,
            }, if (layer_is_swa) @as(i64, @intCast(p.n_swa)) else null);

            attn_out = ggml.reshape2d(ctx, attn_out, n_head * head_dim, n_tokens_i64);
        } else {
            if (causal) {
                // Causal pass: read K/V from cache (written by the corresponding KV layer)
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

                // Use the actual K cache tensor's sequence length (ne[2]),
                // not the global currentLen(), because getKView may clamp to per-layer max_seq_len.
                const cache_len: i64 = k_cache.ne()[2];

                // For SWA shared KV layers, compute cache_start_abs similarly.
                const cache_start_abs: i64 = if (layer_is_swa) blk: {
                    const current_len = cache.currentLen();
                    if (current_len > cache_len) {
                        break :blk @as(i64, @intCast(current_len)) - cache_len;
                    }
                    break :blk 0;
                } else 0;

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
                    .cache_start_abs = cache_start_abs,
                }, if (layer_is_swa) @as(i64, @intCast(p.n_swa)) else null);

                attn_out = ggml.reshape2d(ctx, attn_out, n_head * head_dim, n_tokens_i64);
            } else {
                // Non-causal (media) pass: shared KV layers cannot read from cache
                // because the KV layer's K/V was not cached. Use a zero attention output
                // as a placeholder (the shared layers don't contribute to media encoding).
                log.debug("Shared KV layer {d} in non-causal pass: using zero attention", .{il});
                attn_out = try ctx.newTensor2d(.f32, n_head * head_dim, n_tokens_i64);
                // In no_alloc mode, manually allocate data for the zero tensor
                if (ctx.getNoAlloc()) {
                    const data_size = @as(usize, @intCast(attn_out.nBytes()));
                    const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
                    @memset(buf, 0);
                    attn_out.setDataPtr(buf);
                } else {
                    const n_elems = @as(usize, @intCast(attn_out.nElems()));
                    const buf = try std.heap.page_allocator.alloc(f32, n_elems);
                    defer std.heap.page_allocator.free(buf);
                    @memset(buf, 0);
                    try attn_out.dataSet(f32, buf);
                }
            }
        }

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
// 辅助函数
// ============================================================================

/// GeGLU FFN: gelu(h @ W_gate) * (h @ W_up) -> down
pub fn gegluFFN(
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
pub fn findKVLayer(p: *const Gemma4Params, layer_idx: usize) usize {
    if (layer_idx >= p.n_layer_kv_from_start and p.n_layer_kv_from_start > 0) {
        const is_swa = p.is_swa_layer.items[layer_idx];
        var j: usize = p.n_layer_kv_from_start - 1;
        while (true) {
            if (p.is_swa_layer.items[j] == is_swa) return j;
            if (j == 0) break;
            j -= 1;
        }
        return p.n_layer_kv_from_start - 1;
    }
    return 0;
}

/// Per-layer embedding injection at position `il` in the transformer layer.
pub fn buildPerLayerInjection(
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
