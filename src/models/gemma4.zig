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
                eo_data[0],
                if (n_total > 1) eo_data[1] else @as(f32, 0),
                if (n_total > 2) eo_data[2] else @as(f32, 0),
                if (n_total > 3) eo_data[3] else @as(f32, 0),
                if (n_total > 4) eo_data[4] else @as(f32, 0),
                if (n_total > 5) eo_data[5] else @as(f32, 0),
                if (n_total > 6) eo_data[6] else @as(f32, 0),
                if (n_total > 7) eo_data[7] else @as(f32, 0),
            });
            log.debug("  all_zero={} has_nan={}", .{ all_zero, has_nan });
            if (all_zero) log.warn("  ⚠ embd_override is ALL ZEROS!", .{});
            if (has_nan) log.warn("  ⚠ embd_override contains NaN!", .{});
        }

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
