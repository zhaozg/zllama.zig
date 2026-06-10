//! Qwen2 系列模型实现
//!
//! 标准 Transformer 架构，支持 Qwen2 / Qwen2.5 / Qwen3-Embedding。
//! 特点：QKV 投影 + RoPE + SwiGLU FFN。

const std = @import("std");
const ggml = @import("ggml");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const gguf = @import("gguf");
const kv_cache = @import("kv_cache");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const swiglu = @import("swiglu");
const attention = @import("attention");
const embed = @import("embed");
const weight_loader = @import("weight_loader");

const model = @import("../model.zig");

const log = std.log.scoped(.qwen2);

pub const Qwen2Params = struct {
    base: model.ModelParams = .{},
};

pub const LayerWeights = struct {
    prefix: []const u8,
    attn_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor,
    attn_q_weight: *ggml.Tensor,
    attn_k_weight: *ggml.Tensor,
    attn_v_weight: *ggml.Tensor,
    attn_output_weight: *ggml.Tensor,
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,
};

pub const Qwen2Weights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,
    pub fn deinit(self: *Qwen2Weights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| allocator.free(layer.prefix);
        allocator.free(self.layers);
    }
};

pub const Qwen2Model = struct {
    params: Qwen2Params,
    weights: Qwen2Weights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *Qwen2Model, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.params = try parseParams(gguf_file, allocator);
        self.ctx_weights = try ggml.Context.initNoAlloc(weight_loader.estimateMemSize(gguf_file));
        self.weights = try loadWeights(gguf_file, self.ctx_weights, &self.params, allocator);
    }

    pub fn deinit(self: *Qwen2Model, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn getParams(self: *const Qwen2Model) *const model.ModelParams {
        return &self.params.base;
    }

    pub fn getWeights(self: *const Qwen2Model) *const model.ModelWeights {
        return &self.weights.base;
    }

    pub fn forward(self: *Qwen2Model, ctx: *ggml.Context, graph: *ggml.CGraph, input_tokens: *ggml.Tensor, n_tokens: i32, kv_cache_mgr: ?*kv_cache.KVCache, start_pos: i32) !*ggml.Tensor {
        const p = &self.params.base;
        const w = &self.weights;
        const n_head: i64 = @intCast(p.n_head);
        const n_kv_head: i64 = @intCast(p.n_kv_head);
        const head_dim: i64 = @intCast(p.n_head_dim);
        const head_dim_k: i64 = if (p.n_head_dim_k > 0) @intCast(p.n_head_dim_k) else head_dim;
        const head_dim_v: i64 = if (p.n_head_dim_v > 0) @intCast(p.n_head_dim_v) else head_dim;
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.rope_dim);

        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur.setName("token_embd");

        for (w.layers, 0..) |*layer, i| {
            var name_buf: [128]u8 = undefined;
            var attn_input = rms_norm.rmsNorm(ctx, cur, layer.attn_norm_weight, p.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_input.setName(name_buf[0..name.len :0]);
            }

            var q = ggml.mulMat(ctx, layer.attn_q_weight, attn_input);
            var k = ggml.mulMat(ctx, layer.attn_k_weight, attn_input);
            var v = ggml.mulMat(ctx, layer.attn_v_weight, attn_input);

            q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);
            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);
            v = ggml.reshape3d(ctx, v, head_dim_v, n_kv_head, n_tokens_i64);

            const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens_i64), start_pos);
            q = ggml.ropeExt(ctx, q, pos_tensor, null, @as(i32, @intCast(@min(rope_dim, head_dim))), 0, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k = ggml.ropeExt(ctx, k, pos_tensor, null, @as(i32, @intCast(@min(rope_dim, head_dim_k))), 0, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

            q = ggml.cont(ctx, ggml.permute(ctx, q, 0, 2, 1, 3));
            k = ggml.cont(ctx, ggml.permute(ctx, k, 0, 2, 1, 3));
            v = ggml.cont(ctx, ggml.permute(ctx, v, 0, 2, 1, 3));

            if (kv_cache_mgr) |cache| {
                cache.setKv(ctx, graph, i, k, v, @intCast(n_tokens_i64));
                k = cache.getKView(ctx, i);
                v = cache.getVView(ctx, i);
            }

            const cache_len: i64 = if (kv_cache_mgr) |cache| @as(i64, @intCast(cache.currentLen())) else n_tokens_i64;

            var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
                .n_head = n_head,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            }, null);

            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }
            cur = ggml.add(ctx, cur, attn_out);

            var ffn_input = rms_norm.rmsNorm(ctx, cur, layer.ffn_norm_weight, p.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_input.setName(name_buf[0..name.len :0]);
            }
            const ffn_out = swiglu.swiGLU(ctx, ffn_input, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_out.setName(name_buf[0..name.len :0]);
            }
            cur = ggml.add(ctx, cur, ffn_out);
        }

        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.norm_eps);
        cur.setName("output_norm");
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);
        logits_tensor.setName("logits");
        graph.buildForwardExpand(logits_tensor);
        return logits_tensor;
    }

    pub fn buildGraph(
        self: *Qwen2Model,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*memory.MemoryContext,
        start_pos: i32,
    ) !*ggml.Tensor {
        const ctx = builder.ctx;
        const graph = builder.gf;
        const kv_cache_mgr: ?*kv_cache.KVCache = if (mem_ctx) |ptr| @ptrCast(@alignCast(ptr)) else null;
        return self.forward(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos);
    }

    pub const vtable = model.ModelVTable{
        .deinit = deinitAdapter,
        .buildGraph = buildGraphAdapter,
        .getParams = getParamsAdapter,
        .resetSSMStates = resetSSMStatesAdapter,
    };

    fn deinitAdapter(data: *anyopaque, allocator: std.mem.Allocator) void {
        const self = @as(*Qwen2Model, @ptrCast(@alignCast(data)));
        self.weights.deinit(allocator);
        self.ctx_weights.deinit();
        allocator.destroy(self);
    }

    fn buildGraphAdapter(
        data: *anyopaque,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        cache: ?*anyopaque,
        pos: i32,
    ) anyerror!*ggml.Tensor {
        const self = @as(*Qwen2Model, @ptrCast(@alignCast(data)));
        return self.buildGraph(builder, input_tokens, n_tokens, @as(?*memory.MemoryContext, @ptrCast(@alignCast(cache))), pos);
    }

    fn getParamsAdapter(data: *anyopaque) *const model.ModelParams {
        const self = @as(*Qwen2Model, @ptrCast(@alignCast(data)));
        return self.getParams();
    }

    fn resetSSMStatesAdapter(data: *anyopaque) void {
        const self = @as(*Qwen2Model, @ptrCast(@alignCast(data)));
        _ = self;
    }
};

pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !Qwen2Params {
    var p = Qwen2Params{};

    // Try qwen3.* / qwen2.* / llama.* prefix for each parameter
    p.base.n_vocab = gguf_file.getU32("qwen3.vocab_size") orelse
        gguf_file.getU32("qwen2.vocab_size") orelse
        gguf_file.getU32("llama.vocab_size") orelse blk: {
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            if (val.value_type == .array) break :blk @intCast(val.array_val.len);
        }
        break :blk 0;
    };
    p.base.n_embd = gguf_file.getU32("qwen3.embedding_length") orelse
        gguf_file.getU32("qwen2.embedding_length") orelse
        gguf_file.getU32("llama.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("qwen3.attention.head_count") orelse
        gguf_file.getU32("qwen2.attention.head_count") orelse
        gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("qwen3.head_count") orelse
        gguf_file.getU32("qwen2.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse 0;
    p.base.n_kv_head = gguf_file.getU32("qwen3.attention.head_count_kv") orelse
        gguf_file.getU32("qwen2.attention.head_count_kv") orelse
        gguf_file.getU32("llama.attention.head_count_kv") orelse
        gguf_file.getU32("qwen3.head_count_kv") orelse
        gguf_file.getU32("qwen2.head_count_kv") orelse
        gguf_file.getU32("llama.head_count_kv") orelse p.base.n_head;
    p.base.n_layer = gguf_file.getU32("qwen3.block_count") orelse
        gguf_file.getU32("qwen2.block_count") orelse
        gguf_file.getU32("llama.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("qwen3.feed_forward_length") orelse
        gguf_file.getU32("qwen2.feed_forward_length") orelse
        gguf_file.getU32("llama.feed_forward_length") orelse 0;
    p.base.n_expert = gguf_file.getU32("qwen3.expert_count") orelse
        gguf_file.getU32("qwen2.expert_count") orelse
        gguf_file.getU32("llama.expert_count") orelse 0;
    p.base.n_expert_used = gguf_file.getU32("qwen3.expert_used_count") orelse
        gguf_file.getU32("qwen2.expert_used_count") orelse
        gguf_file.getU32("llama.expert_used_count") orelse 0;
    if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }

    // K/V head dimensions (e.g. Qwen3-Embedding: key_length=128 but n_embd/n_head=64)
    p.base.n_head_dim_k = gguf_file.getU32("qwen3.attention.key_length") orelse
        gguf_file.getU32("qwen2.attention.key_length") orelse
        gguf_file.getU32("llama.attention.key_length") orelse 0;
    p.base.n_head_dim_v = gguf_file.getU32("qwen3.attention.value_length") orelse
        gguf_file.getU32("qwen2.attention.value_length") orelse
        gguf_file.getU32("llama.attention.value_length") orelse 0;

    // When key_length > n_embd/n_head, Q also uses key_length
    // (Q weight is [n_head * key_length, n_embd])
    if (p.base.n_head_dim_k > 0 and p.base.n_head_dim_k != p.base.n_head_dim) {
        p.base.n_head_dim = p.base.n_head_dim_k;
    }

    p.base.max_seq_len = gguf_file.getU32("qwen3.context_length") orelse
        gguf_file.getU32("qwen2.context_length") orelse
        gguf_file.getU32("llama.context_length") orelse 32768;
    p.base.rope_theta = gguf_file.getF32("qwen3.rope.freq_base") orelse
        gguf_file.getF32("qwen2.rope.freq_base") orelse
        gguf_file.getF32("llama.rope.freq_base") orelse 10000000.0;
    p.base.rope_dim = gguf_file.getU32("qwen3.rope.dimension_count") orelse
        gguf_file.getU32("qwen2.rope.dimension_count") orelse
        gguf_file.getU32("llama.rope.dimension_count") orelse @divExact(p.base.n_head_dim, @as(u32, 2));
    p.base.norm_eps = gguf_file.getF32("qwen3.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("qwen2.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-6;
    p.base.model_name = gguf_file.getString("general.name") orelse "";
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";

    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required model parameters", .{});
        return error.InvalidModelParams;
    }
    log.info("Qwen2: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}", .{ p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head, p.base.n_layer, p.base.n_ff });
    return p;
}

pub fn loadWeights(gguf_file: *const gguf.GGUFFile, ctx: *ggml.Context, params: *const Qwen2Params, allocator: std.mem.Allocator) !Qwen2Weights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Qwen2 weights...", .{});
    const token_embd = try weight_loader.findOrCreateTensor(ctx, gguf_file, "token_embd.weight");
    token_embd.setName("token_embd.weight");
    const output_weight = weight_loader.findOrCreateTensor(ctx, gguf_file, "output.weight") catch null;
    if (output_weight) |ow| ow.setName("output.weight");
    const output_norm_weight = try weight_loader.findOrCreateTensor(ctx, gguf_file, "output_norm.weight");
    output_norm_weight.setName("output_norm.weight");
    var layers = try allocator.alloc(LayerWeights, n_layer);
    var layers_loaded: usize = 0;
    errdefer {
        for (0..layers_loaded) |j| allocator.free(layers[j].prefix);
        allocator.free(layers);
    }
    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});
        layers[i] = LayerWeights{
            .prefix = prefix,
            .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
            .ffn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight"),
            .attn_q_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight"),
            .attn_k_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight"),
            .attn_v_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight"),
            .attn_output_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight"),
            .ffn_gate_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight"),
            .ffn_up_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight"),
            .ffn_down_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight"),
        };
        layers_loaded = i + 1;
    }
    log.info("All Qwen2 weights loaded ({d} layers)\n", .{n_layer});
    return Qwen2Weights{
        .base = .{ .params = params.base, .token_embd = token_embd, .output_weight = output_weight, .output_norm_weight = output_norm_weight },
        .layers = layers,
    };
}
