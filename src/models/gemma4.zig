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

const gemma4_graph = @import("gemma4_graph.zig");
const model = @import("../model.zig");

const log = std.log.scoped(.model_gemma4);

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

// ============================================================================
// Gemma 4 模型
// ============================================================================

pub const Gemma4Model = struct {
    params: Gemma4Params,
    weights: Gemma4Weights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *Gemma4Model, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.params = try gemma4_loader.parseParams(gguf_file, allocator);

        const mem_size_estimate = gemma4_loader.estimateMemSize(gguf_file);
        self.ctx_weights = try ggml.Context.init(mem_size_estimate);

        self.weights = try gemma4_loader.loadWeights(gguf_file, self.ctx_weights, &self.params, allocator);
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
        var g = gemma4_graph.Gemma4Graph{
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
        var g = gemma4_graph.Gemma4Graph{
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
        log.debug("mediaForward: ne=[{d},{d}] start_pos={d} n_tokens={d} causal={}", .{
            embd_override.ne()[0], embd_override.ne()[1], start_pos, n_tokens, causal,
        });

        // embd_offset is unused here because the embedding override is already
        // sliced to the correct chunk by the caller (threeStagePrefill).
        // Position encoding uses start_pos for the chunk's absolute position.
        _ = embd_offset;
        var g = gemma4_graph.Gemma4Graph{
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
        var g = gemma4_graph.Gemma4Graph{
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
        .getPerLayerMaxSeqLen = getPerLayerMaxSeqLenAdapter,
        .buildMM = buildMMAdapter,
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

    fn getPerLayerMaxSeqLenAdapter(data: *anyopaque, allocator: std.mem.Allocator) ?[]u32 {
        const self: *Gemma4Model = @ptrCast(@alignCast(data));
        const n_layer = self.params.base.n_layer;
        const n_swa = self.params.n_swa;
        const max_seq = self.params.base.max_seq_len;

        // 如果没有 SWA 或 SWA 窗口为 0，所有层使用相同的 max_seq_len
        if (n_swa == 0) return null;

        const lens = allocator.alloc(u32, n_layer) catch return null;
        for (0..n_layer) |i| {
            if (self.params.is_swa_layer.items[i]) {
                // SWA 层使用滑动窗口大小
                lens[i] = n_swa;
            } else {
                // 非 SWA 层使用完整上下文长度
                lens[i] = max_seq;
            }
        }
        return lens;
    }

    fn buildMMAdapter(
        data: *anyopaque,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        cache: ?*anyopaque,
        pos: i32,
        embd_override: *ggml.Tensor,
        embd_offset: i32,
        causal: bool,
    ) anyerror!*ggml.Tensor {
        const self: *Gemma4Model = @ptrCast(@alignCast(data));
        const kv_cache_mgr: ?*kv_cache.KVCache = if (cache) |c| @ptrCast(@alignCast(c)) else null;
        return self.mediaForward(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, pos, embd_override, embd_offset, causal);
    }
};

const gemma4_loader = @import("gemma4_loader.zig");

const testing = std.testing;

test "Gemma4Params defaults" {
    var p = Gemma4Params{
        .is_swa_layer = std.ArrayList(bool).init(testing.allocator),
    };
    defer p.deinit();
    try testing.expectEqual(@as(u32, 0), p.base.n_vocab);
    try testing.expectEqual(@as(f32, 1.0), p.f_attention_scale);
}
