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
//!
//! 注意：此实现为初始版本，暂不支持 MoE 路由和 per-layer embedding。
//! 这些特性将在后续版本中实现。

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

    // 该层是否有自己的 KV
    has_kv: bool,
};

pub const Gemma4Weights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,

    pub fn deinit(self: *Gemma4Weights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| {
            allocator.free(layer.prefix);
        }
        allocator.free(self.layers);
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
        self.ctx_weights = try ggml.Context.initNoAlloc(mem_size_estimate);

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


    /// Forward pass with pre-computed embedding override (for multimodal input).
    /// When embd_override is non-null, the first embd_override.ne()[1] token positions
    /// use the override embeddings instead of looking up token embeddings.
    /// Text tokens (from input_tokens) follow after the override positions.
    pub fn forwardWithEmbdOverride(
        self: *Gemma4Model,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        embd_override: *ggml.Tensor,
    ) !*ggml.Tensor {
        const p = &self.params;
        const w = &self.weights;
        const n_tokens_i64: i64 = n_tokens;
        const n_override: i64 = embd_override.ne()[1];
        const n_text: i64 = n_tokens_i64 - n_override;
        const n_embd_i64: i64 = @intCast(p.base.n_embd);

        // Mixed embeddings: first n_override from override, rest from token embeddings
        var cur: *ggml.Tensor = undefined;
        if (n_text > 0) {
            // Get embeddings for ALL tokens, then replace first n_override positions
            var all_embd = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
            all_embd = ggml.scale(ctx, all_embd, @sqrt(@as(f32, @floatFromInt(p.base.n_embd))));
            // Extract text-only portion: [n_embd, n_text]
            const text_embd = all_embd.view2d(ctx, n_embd_i64, n_text, all_embd.nb()[1], @as(usize, @intCast(n_override)) * @sizeOf(f32) * @as(usize, @intCast(n_embd_i64)));
            cur = ggml.concat(ctx, embd_override, ggml.cont(ctx, text_embd), 1);
        } else {
            cur = embd_override;
        }
        cur.setName("inp_scaled_mm");

        // Position encoding
        const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens), start_pos);

        // Main transformer loop (identical to forward())
        return self.transformerForward(ctx, graph, cur, pos_tensor, n_tokens_i64, start_pos, kv_cache_mgr);
    }

    pub fn forward(
        self: *Gemma4Model,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
    ) !*ggml.Tensor {
        const p = &self.params;
        const w = &self.weights;
        const n_tokens_i64: i64 = n_tokens;

        // Token 嵌入 + 缩放
        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur = ggml.scale(ctx, cur, @sqrt(@as(f32, @floatFromInt(p.base.n_embd))));
        cur.setName("inp_scaled");

        // 位置编码
        const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens), start_pos);

        return self.transformerForward(ctx, graph, cur, pos_tensor, n_tokens_i64, start_pos, kv_cache_mgr);
    }

    /// Shared transformer loop (used by both forward and forwardWithEmbdOverride).
    fn transformerForward(
        self: *Gemma4Model,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        cur_in: *ggml.Tensor,
        pos_tensor: *ggml.Tensor,
        n_tokens_i64: i64,
        start_pos: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
    ) !*ggml.Tensor {
        const p = &self.params;
        const w = &self.weights;
        var cur = cur_in;

        for (w.layers, 0..) |*layer, i| {
            const n_head: i64 = @intCast(p.base.n_head);

            // Gemma 4: head_dim 由 attn_q_norm 维度决定（per-layer 可能不同）
            const head_dim: i64 = layer.attn_q_norm_weight.ne()[0];

            // RoPE 参数
            const layer_is_swa = p.is_swa_layer.items[i];
            const freq_base_l: f32 = if (layer_is_swa and p.rope_freq_base_swa > 0)
                p.rope_freq_base_swa
            else
                p.base.rope_theta;
            const freq_scale_l: f32 = 1.0;
            // SWA 层使用较小的 rope_dim_swa（因为 head_dim 更小），全注意力层使用 base.rope_dim
            // 同时确保 rope_dim 不超过 head_dim（防止 ggml_rope 断言 n_dims <= ne0）
            const rope_dim_full: u32 = if (p.rope_dim_swa > 0 and layer_is_swa)
                @min(p.rope_dim_swa, @as(u32, @intCast(head_dim)))
            else
                @min(p.base.rope_dim, @as(u32, @intCast(head_dim)));
            const rope_dim: i64 = @intCast(rope_dim_full);

            // --- Pre-attention RMSNorm ---
            const attn_input = rms_norm.rmsNorm(ctx, cur, layer.attn_norm_weight, p.base.norm_eps);

            // --- Q 投影（所有层都需要） ---
            var q = ggml.mulMat(ctx, layer.attn_q_weight, attn_input);
            q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);

            // Q Pre-norm
            q = ggml.reshape2d(ctx, q, head_dim, n_head * n_tokens_i64);
            q = ggml.rmsNorm(ctx, q, p.base.norm_eps);
            q = ggml.mul(ctx, q, ggml.reshape2d(ctx, layer.attn_q_norm_weight, head_dim, 1));
            q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);

            // RoPE on Q（全注意力层使用 rope_freqs 做 proportional rope）
            const rope_freqs: ?*ggml.Tensor = if (!layer_is_swa) layer.rope_freqs else null;
            q = ggml.ropeExt(ctx, q, pos_tensor, rope_freqs, @intCast(rope_dim), 0, 0, freq_base_l, freq_scale_l, 0.0, 1.0, 0.0, 0.0);

            // --- K/V 投影 + 注意力 ---
            var attn_out: *ggml.Tensor = undefined;
            if (layer.has_kv) {
                // head_dim_k from K norm weight (all layers use 256 for Gemma 4)
                const head_dim_k: i64 = layer.attn_k_norm_weight.ne()[0];
                const n_kv_head: i64 = if (layer.attn_k_weight) |kw|
                    @divExact(kw.ne()[1], head_dim_k)
                else
                    n_head;

                var k = ggml.mulMat(ctx, layer.attn_k_weight.?, attn_input);
                var v_tensor: *ggml.Tensor = if (layer.attn_v_weight) |vw|
                    ggml.mulMat(ctx, vw, attn_input)
                else
                    k; // 共享 K/V

                k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);
                v_tensor = ggml.reshape3d(ctx, v_tensor, head_dim_k, n_kv_head, n_tokens_i64);

                // K Pre-norm
                k = ggml.reshape2d(ctx, k, head_dim_k, n_kv_head * n_tokens_i64);
                k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
                k = ggml.mul(ctx, k, ggml.reshape2d(ctx, layer.attn_k_norm_weight, head_dim_k, 1));
                k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);

                // V RMSNorm (no weight)
                v_tensor = ggml.rmsNorm(ctx, v_tensor, p.base.norm_eps);

                // RoPE on K
                k = ggml.ropeExt(ctx, k, pos_tensor, rope_freqs, @intCast(rope_dim), 0, 0, freq_base_l, freq_scale_l, 0.0, 1.0, 0.0, 0.0);

                // Gemma 4: Q may have larger head_dim than K.
                // Reshape Q to match K's head_dim BEFORE permute.
                var n_head_eff = n_head;
                if (head_dim != head_dim_k) {
                    n_head_eff = @divExact(n_head * head_dim, head_dim_k);
                    q = ggml.reshape3d(ctx, q, head_dim_k, n_head_eff, n_tokens_i64);
                    log.debug("Layer {d}: reshape Q from [{d},{d}] -> [{d},{d}]", .{ i, head_dim, n_head, head_dim_k, n_head_eff });
                }
                // KV Cache: store K/V, then read back full cache for attention across all tokens
                if (kv_cache_mgr) |cache| {
                    cache.setKv(ctx, graph, i, k, v_tensor, @intCast(n_tokens_i64));
                    k = cache.getKView(ctx, i);
                    v_tensor = cache.getVView(ctx, i);
                }

                const cache_len: i64 = if (kv_cache_mgr) |cache|
                    @as(i64, @intCast(cache.currentLen()))
                else
                    n_tokens_i64;

                attn_out = attention.scaledDotProductAttention(ctx, q, k, v_tensor, .{
                    .n_head = n_head_eff,
                    .n_kv_head = n_kv_head,
                    .head_dim = head_dim_k,
                    .n_tokens = n_tokens_i64,
                    .cache_len = cache_len,
                    .start_pos = start_pos,
                    .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim_k))),
                }, if (layer_is_swa) @as(i64, @intCast(p.n_swa)) else null);

                // Reshape attn_out back to original Q dimension
                attn_out = ggml.reshape2d(ctx, attn_out, n_head * head_dim, n_tokens_i64);
            } else {
                // 非 KV 层：复用前面层的 KV（来自 cache 或实时计算）
                const kv_layer_idx = findKVLayer(w, i);
                const kv_layer = &w.layers[kv_layer_idx];

                var k: *ggml.Tensor = undefined;
                var v_tensor: *ggml.Tensor = undefined;
                var head_dim_k_cache: i64 = undefined;
                var n_kv_head_cache: i64 = undefined;

                if (kv_cache_mgr) |cache| {
                    k = cache.getKView(ctx, kv_layer_idx);
                    v_tensor = cache.getVView(ctx, kv_layer_idx);
                    head_dim_k_cache = k.ne()[0];
                    n_kv_head_cache = k.ne()[1];
                } else {
                    // No KV cache: compute K/V from nearest KV layer's weights
                    head_dim_k_cache = kv_layer.attn_k_norm_weight.ne()[0];
                    n_kv_head_cache = if (kv_layer.attn_k_weight) |kw|
                        @divExact(kw.ne()[1], head_dim_k_cache)
                    else
                        n_head;

                    k = ggml.mulMat(ctx, kv_layer.attn_k_weight.?, attn_input);
                    v_tensor = if (kv_layer.attn_v_weight) |vw|
                        ggml.mulMat(ctx, vw, attn_input)
                    else
                        k;

                    k = ggml.reshape3d(ctx, k, head_dim_k_cache, n_kv_head_cache, n_tokens_i64);
                    v_tensor = ggml.reshape3d(ctx, v_tensor, head_dim_k_cache, n_kv_head_cache, n_tokens_i64);

                    k = ggml.reshape2d(ctx, k, head_dim_k_cache, n_kv_head_cache * n_tokens_i64);
                    k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
                    k = ggml.mul(ctx, k, ggml.reshape2d(ctx, kv_layer.attn_k_norm_weight, head_dim_k_cache, 1));
                    k = ggml.reshape3d(ctx, k, head_dim_k_cache, n_kv_head_cache, n_tokens_i64);

                    v_tensor = ggml.rmsNorm(ctx, v_tensor, p.base.norm_eps);
                    k = ggml.ropeExt(ctx, k, pos_tensor, rope_freqs, @intCast(rope_dim), 0, 0, freq_base_l, freq_scale_l, 0.0, 1.0, 0.0, 0.0);
                }

                var n_head_eff = n_head;
                if (head_dim != head_dim_k_cache) {
                    n_head_eff = @divExact(n_head * head_dim, head_dim_k_cache);
                    q = ggml.reshape3d(ctx, q, head_dim_k_cache, n_head_eff, n_tokens_i64);
                }

                const cache_len: i64 = if (kv_cache_mgr) |cache|
                    @as(i64, @intCast(cache.currentLen()))
                else
                    n_tokens_i64;

                attn_out = attention.scaledDotProductAttention(ctx, q, k, v_tensor, .{
                    .n_head = n_head_eff,
                    .n_kv_head = n_kv_head_cache,
                    .head_dim = head_dim_k_cache,
                    .n_tokens = n_tokens_i64,
                    .cache_len = cache_len,
                    .start_pos = start_pos,
                    .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim_k_cache))),
                }, if (layer_is_swa) @as(i64, @intCast(p.n_swa)) else null);

                // Reshape attn_out back to original Q dimension
                attn_out = ggml.reshape2d(ctx, attn_out, n_head * head_dim, n_tokens_i64);
            }

            // 输出投影
            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);

            // --- Attention Post-norm ---
            attn_out = rms_norm.rmsNorm(ctx, attn_out, layer.attn_post_norm_weight, p.base.norm_eps);

            // 残差连接
            cur = ggml.add(ctx, cur, attn_out);

            // --- FFN ---
            const ffn_input = rms_norm.rmsNorm(ctx, cur, layer.ffn_norm_weight, p.base.norm_eps);
            const ffn_out = gegluFFN(ctx, ffn_input, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);

            // --- FFN Post-norm ---
            const ffn_normed = rms_norm.rmsNorm(ctx, ffn_out, layer.ffn_post_norm_weight, p.base.norm_eps);

            // 残差连接
            cur = ggml.add(ctx, cur, ffn_normed);

            // --- Layer output scale（可选） ---
            if (layer.out_scale) |scale| {
                cur = ggml.mul(ctx, cur, scale);
            }
        }

        // --- 最终 RMSNorm ---
        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.base.norm_eps);
        cur.setName("output_norm");

        // --- 输出投影 ---
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);

        // --- Final logit softcapping ---
        if (p.final_logit_softcapping > 0.0) {
            const cap = p.final_logit_softcapping;
            logits_tensor = ggml.scale(ctx, logits_tensor, 1.0 / cap);
            logits_tensor = ggml.tanh(ctx, logits_tensor);
            logits_tensor = ggml.scale(ctx, logits_tensor, cap);
        }
        logits_tensor.setName("logits");

        graph.buildForwardExpand(logits_tensor);
        return logits_tensor;
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

// ============================================================================
// 辅助函数
// ============================================================================

/// GeGLU FFN: gate(h) * up(h) -> down
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

/// 查找指定层之前最近的有 KV 的层索引
fn findKVLayer(w: *const Gemma4Weights, layer_idx: usize) usize {
    var idx: usize = layer_idx;
    while (idx > 0) {
        idx -= 1;
        if (w.layers[idx].has_kv) return idx;
    }
    return 0;
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

    log.info("Gemma4: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}, swa={d}, shared_kv={d}, softcap={d}", .{
        p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head,
        p.base.n_layer, p.base.n_ff, p.n_swa, p.n_kv_shared_layers, p.final_logit_softcapping,
    });

    return p;
}

// ============================================================================
// 权重加载
// ============================================================================

fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *const Gemma4Params,
    allocator: std.mem.Allocator,
) !Gemma4Weights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Gemma 4 weights...", .{});

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

    var layers = try allocator.alloc(LayerWeights, n_layer);
    var layers_loaded: usize = 0;
    errdefer {
        for (0..layers_loaded) |j| {
            allocator.free(layers[j].prefix);
        }
        allocator.free(layers);
    }

    // Gemma 4: rope_freqs is global, shared across all full-attention layers
    const global_rope_freqs = findOrCreateTensor(ctx, gguf_file, "rope_freqs.weight") catch null;

    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});

        const has_kv: bool = if (params.n_layer_kv_from_start > 0)
            i < params.n_layer_kv_from_start
        else
            true;

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

        // Gemma 4: all full-attention layers share the global rope_freqs
        layers[i] = LayerWeights{
            .prefix = prefix,
            .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
            .ffn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight"),
            .attn_q_norm_weight = q_norm,
            .attn_k_norm_weight = if (k_norm) |kn| kn else q_norm,
            // Gemma 4 使用 post_attention_norm / post_ffw_norm 命名
            .attn_post_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "post_attention_norm.weight"),
            .ffn_post_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "post_ffw_norm.weight"),
            .attn_q_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight"),
            .attn_k_weight = k_weight,
            .attn_v_weight = v_weight,
            .attn_output_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight"),
            .ffn_gate_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight"),
            .ffn_up_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight"),
            .ffn_down_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight"),
            .out_scale = out_scale,
            .rope_freqs = if (!params.is_swa_layer.items[i]) global_rope_freqs else null,
            .has_kv = has_kv,
        };
        layers_loaded = i + 1;
    }

    log.info("All Gemma 4 weights loaded ({d} layers, {d} with KV)", .{ n_layer, params.n_layer_kv_from_start });

    return Gemma4Weights{
        .base = .{
            .params = params.base,
            .token_embd = token_embd,
            .output_weight = output_weight,
            .output_norm_weight = output_norm_weight,
        },
        .layers = layers,
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
