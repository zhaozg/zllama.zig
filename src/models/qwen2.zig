//! Qwen2 系列模型实现
//!
//! 标准 Transformer 架构，支持 Qwen2 / Qwen2.5。
//! 特点：QKV 投影 + RoPE + SwiGLU FFN，无 Q/K norm，无 gate。

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
        self.ctx_weights = try ggml.Context.initNoAlloc(1024 * 1024 * 1024 * 2);
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
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.rope_dim);

        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur.setName("token_embd");

        for (w.layers, 0..) |*layer, i| {
            var name_buf: [128]u8 = undefined;

            // Pre-attention norm
            var attn_input = rms_norm.rmsNorm(ctx, cur, layer.attn_norm_weight, p.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_input.setName(name_buf[0..name.len :0]);
            }

            // QKV projections
            var q = ggml.mulMat(ctx, layer.attn_q_weight, attn_input);
            var k = ggml.mulMat(ctx, layer.attn_k_weight, attn_input);
            var v = ggml.mulMat(ctx, layer.attn_v_weight, attn_input);

            // Reshape for RoPE: [head_dim, n_head, n_tokens]
            q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);
            k = ggml.reshape3d(ctx, k, head_dim, n_kv_head, n_tokens_i64);
            v = ggml.reshape3d(ctx, v, head_dim, n_kv_head, n_tokens_i64);

            // RoPE
            const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens_i64), start_pos);
            q = ggml.ropeExt(ctx, q, pos_tensor, @as(i32, @intCast(rope_dim)), 0, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k = ggml.ropeExt(ctx, k, pos_tensor, @as(i32, @intCast(rope_dim)), 0, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

            // Transpose to [head_dim, n_tokens, n_head] for attention
            q = ggml.cont(ctx, ggml.permute(ctx, q, 0, 2, 1, 3));
            k = ggml.cont(ctx, ggml.permute(ctx, k, 0, 2, 1, 3));
            v = ggml.cont(ctx, ggml.permute(ctx, v, 0, 2, 1, 3));

            // KV Cache
            if (kv_cache_mgr) |cache| {
                cache.setKv(ctx, graph, i, k, v, @intCast(n_tokens_i64));
                k = cache.getKView(ctx, i);
                v = cache.getVView(ctx, i);
            }

            const cache_len: i64 = if (kv_cache_mgr) |cache| @as(i64, @intCast(cache.currentLen())) else n_tokens_i64;

            // Scaled dot-product attention
            var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
                .n_head = n_head,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            });

            // Output projection
            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }

            // Residual
            cur = ggml.add(ctx, cur, attn_out);

            // FFN norm
            var ffn_input = rms_norm.rmsNorm(ctx, cur, layer.ffn_norm_weight, p.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_input.setName(name_buf[0..name.len :0]);
            }

            // SwiGLU FFN
            const ffn_out = swiglu.swiGLU(ctx, ffn_input, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_out.setName(name_buf[0..name.len :0]);
            }
            cur = ggml.add(ctx, cur, ffn_out);
        }

        // Output norm & projection
        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.norm_eps);
        cur.setName("output_norm");
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);
        logits_tensor.setName("logits");
        graph.buildForwardExpand(logits_tensor);
        return logits_tensor;

    }

    /// 适配 buildGraph 接口（通过 GraphBuilder 调用）
    pub fn buildGraph(
        self: *Qwen2Model,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*memory.MemoryContext,
        start_pos: i32,
    ) !*ggml.Tensor {
        _ = mem_ctx;
        const ctx = builder.ctx;
        const graph = builder.gf;
        return self.forward(ctx, graph, input_tokens, n_tokens, null, start_pos);
    }

    /// 虚表定义（用于 ModelInstance 运行时多态）
    pub const vtable = model.ModelVTable{
        .deinit = deinitAdapter,
        .buildGraph = buildGraphAdapter,
        .getParams = getParamsAdapter,
        .resetSSMStates = resetSSMStatesAdapter,
    };

    fn deinitAdapter(data: *anyopaque, allocator: std.mem.Allocator) void {
        const self = @as(*Qwen2Model, @ptrCast(@alignCast(data)));
        // 释放 weights 中的分配内存
        self.weights.deinit(allocator);
        // 释放 ctx_weights（ggml 上下文）
        self.ctx_weights.deinit();
        // 释放 Qwen2Model 结构体本身
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
    p.base.n_vocab = gguf_file.getU32("llama.vocab_size") orelse blk: {
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            if (val.value_type == .array) break :blk @intCast(val.array_val.len);
        }
        break :blk 0;
    };
    p.base.n_embd = gguf_file.getU32("llama.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("llama.attention.head_count") orelse gguf_file.getU32("llama.head_count") orelse 0;
    p.base.n_kv_head = gguf_file.getU32("llama.attention.head_count_kv") orelse gguf_file.getU32("llama.head_count_kv") orelse p.base.n_head;
    p.base.n_layer = gguf_file.getU32("llama.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("llama.feed_forward_length") orelse 0;
    p.base.n_expert = gguf_file.getU32("llama.expert_count") orelse 0;
    p.base.n_expert_used = gguf_file.getU32("llama.expert_used_count") orelse 0;
    if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }
    p.base.max_seq_len = gguf_file.getU32("llama.context_length") orelse 32768;
    p.base.rope_theta = gguf_file.getF32("llama.rope.freq_base") orelse 10000000.0;
    p.base.rope_dim = gguf_file.getU32("llama.rope.dimension_count") orelse @divExact(p.base.n_head_dim, @as(u32, 2));
    p.base.norm_eps = gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-6;
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
    const token_embd = findOrCreateTensor(ctx, gguf_file, "token_embd.weight") catch |err| {
        log.err("Failed to load token_embd.weight: {}\n", .{err});
        return error.MissingWeight;
    };
    token_embd.setName("token_embd.weight");
    const output_weight = findOrCreateTensor(ctx, gguf_file, "output.weight") catch null;
    if (output_weight) |ow| ow.setName("output.weight");
    const output_norm_weight = findOrCreateTensor(ctx, gguf_file, "output_norm.weight") catch |err| {
        log.err("Failed to load output_norm.weight: {}\n", .{err});
        return error.MissingWeight;
    };
    output_norm_weight.setName("output_norm.weight");
    var layers = try allocator.alloc(LayerWeights, n_layer);
    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});
        layers[i] = LayerWeights{
            .prefix = prefix,
            .attn_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_norm.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_norm.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .attn_q_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_q.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .attn_k_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_k.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .attn_v_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_v.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .attn_output_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_output.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_gate_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_gate.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_up_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_up.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_down_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_down.weight: {}\n", .{ i, err });
                return error.MissingWeight;
            },
        };
    }
    log.info("All Qwen2 weights loaded ({d} layers)\n", .{n_layer});
    return Qwen2Weights{
        .base = .{ .params = params.base, .token_embd = token_embd, .output_weight = output_weight, .output_norm_weight = output_norm_weight },
        .layers = layers,
    };
}

fn loadLayerWeight(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, prefix: []const u8, name: []const u8) !*ggml.Tensor {
    var buf: [256]u8 = undefined;
    const full_name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, name });
    buf[full_name.len] = 0;
    return findOrCreateTensor(ctx, gguf_file, full_name[0..full_name.len :0]);
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
        if (tensor_bytes.len != tensor_data.len) log.warn("Tensor '{s}' size mismatch: expected {d} bytes, got {d} bytes", .{ name, tensor_bytes.len, tensor_data.len });
        @memcpy(tensor_bytes, tensor_data);
        return tensor;
    }
    return error.TensorNotFound;
}
