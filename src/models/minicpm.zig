//! MiniCPM 系列模型实现
//!
//! 支持 MiniCPM / MiniCPM5 架构。
//! MiniCPM 架构与 Granite 共享图构建逻辑，本质上与 LLaMA 架构相同，
//! 但额外支持 f_embedding_scale、f_residual_scale、f_logit_scale 等缩放参数。
//!
//! 参考 llama.cpp: src/models/minicpm.cpp, src/models/granite.cpp
//!
//! 注意：MiniCPM5 的 GGUF 文件可能使用 general.architecture = 'llama'，
//! 此时会走 LLaMA 模型路径。本模块处理 general.architecture = 'minicpm' 的情况。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const kv_cache = @import("kv_cache");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const swiglu = @import("swiglu");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const attention = @import("attention");
const embed = @import("embed");
const weight_loader = @import("weight_loader");

const model = @import("../model.zig");

const log = std.log.scoped(.model_minicpm);

// ============================================================================
// MiniCPM 特有超参数
// ============================================================================

pub const MiniCPMParams = struct {
    base: model.ModelParams = .{},
    rope_scaling: ?model.RopeScaling = null,
    // MiniCPM / Granite 缩放参数
    embedding_scale: f32 = 12.0,
    residual_scale: f32 = 1.4,
    logit_scale: f32 = 1.0,
    attention_scale: f32 = 0.0,
};

// ============================================================================
// MiniCPM 模型权重
// ============================================================================

pub const LayerWeights = struct {
    prefix: []const u8,

    // 归一化
    attn_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor,

    // 注意力
    attn_q_weight: *ggml.Tensor,
    attn_k_weight: *ggml.Tensor,
    attn_v_weight: *ggml.Tensor,
    attn_output_weight: *ggml.Tensor,

    // FFN (SwiGLU)
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,
};

pub const MiniCPMWeights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,

    pub fn deinit(self: *MiniCPMWeights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| {
            allocator.free(layer.prefix);
        }
        allocator.free(self.layers);
    }
};

// ============================================================================
pub const MiniCPMModel = struct {
    params: MiniCPMParams,
    weights: MiniCPMWeights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *MiniCPMModel, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.params = try parseParams(gguf_file, allocator);

        // 根据模型参数动态估计所需内存
        const mem_size_estimate = weight_loader.estimateMemSize(gguf_file);
        self.ctx_weights = try ggml.Context.initNoAlloc(mem_size_estimate);

        self.weights = try loadWeights(gguf_file, self.ctx_weights, &self.params, allocator);
    }

    pub fn deinit(self: *MiniCPMModel, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn getParams(self: *const MiniCPMModel) *const model.ModelParams {
        return &self.params.base;
    }

    pub fn getWeights(self: *const MiniCPMModel) *const model.ModelWeights {
        return &self.weights.base;
    }

    pub fn forward(
        self: *MiniCPMModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
    ) !*ggml.Tensor {
        const p = &self.params;
        const w = &self.weights;
        const n_head: i64 = @intCast(p.base.n_head);
        const n_kv_head: i64 = @intCast(p.base.n_kv_head);
        const head_dim: i64 = @intCast(p.base.n_head_dim);
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.base.rope_dim);

        // Token 嵌入
        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur.setName("token_embd");

        // MiniCPM: scale the input embeddings
        cur = ggml.scale(ctx, cur, p.embedding_scale);
        cur.setName("inp_scaled");

        // 逐层处理
        for (w.layers, 0..) |*layer, i| {
            var name_buf: [128]u8 = undefined;

            // --- 注意力前的 RMSNorm ---
            var attn_input = rms_norm.rmsNorm(ctx, cur, layer.attn_norm_weight, p.base.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_input.setName(name_buf[0..name.len :0]);
            }

            // --- QKV 投影 ---
            var q = ggml.mulMat(ctx, layer.attn_q_weight, attn_input);
            var k = ggml.mulMat(ctx, layer.attn_k_weight, attn_input);
            var v = ggml.mulMat(ctx, layer.attn_v_weight, attn_input);

            // 重塑为 [head_dim, n_head/kv_head, n_tokens]
            q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);
            k = ggml.reshape3d(ctx, k, head_dim, n_kv_head, n_tokens_i64);
            v = ggml.reshape3d(ctx, v, head_dim, n_kv_head, n_tokens_i64);

            // RoPE 位置编码
            const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens_i64), start_pos);
            const rope_result = rope.applyRope(ctx, q, k, pos_tensor, .{
                .rope_dim = rope_dim,
                .rope_theta = p.base.rope_theta,
                .mode = 0, // LLAMA_ROPE_TYPE_NORM
            });
            q = rope_result.q;
            k = rope_result.k;

            // 处理 KV Cache
            if (kv_cache_mgr) |cache| {
                cache.setKv(ctx, graph, i, k, v, @intCast(n_tokens_i64));
                k = cache.getKView(ctx, i);
                v = cache.getVView(ctx, i);
            }

            const cache_len: i64 = if (kv_cache_mgr) |cache|
                @as(i64, @intCast(cache.currentLen()))
            else
                n_tokens_i64;

            // 注意力计算
            const kq_scale = if (p.attention_scale == 0.0)
                1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)))
            else
                p.attention_scale;

            var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
                .n_head = n_head,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = kq_scale,
            }, null);

            // 输出投影
            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }

            // MiniCPM: scale residual
            if (p.residual_scale != 0.0) {
                attn_out = ggml.scale(ctx, attn_out, p.residual_scale);
            }

            // 残差连接
            cur = ggml.add(ctx, cur, attn_out);

            // --- FFN Norm + SwiGLU FFN ---
            var ffn_input = rms_norm.rmsNorm(ctx, cur, layer.ffn_norm_weight, p.base.norm_eps);
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

            // MiniCPM: scale residual
            if (p.residual_scale != 0.0) {
                _ = ggml.scale(ctx, ffn_out, p.residual_scale);
            }

            cur = ggml.add(ctx, cur, ffn_out);
        }

        // --- 最终 RMSNorm ---
        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.base.norm_eps);
        cur.setName("output_norm");

        // --- 输出投影 ---
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);
        logits_tensor.setName("logits");

        // MiniCPM: scale logits
        if (p.logit_scale != 0.0) {
            logits_tensor = ggml.scale(ctx, logits_tensor, 1.0 / p.logit_scale);
        }

        graph.buildForwardExpand(logits_tensor);

        return logits_tensor;
    }

    pub fn buildGraph(
        self: *MiniCPMModel,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*anyopaque,
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
        const self = @as(*MiniCPMModel, @ptrCast(@alignCast(data)));
        self.weights.deinit(allocator);
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
        const self = @as(*MiniCPMModel, @ptrCast(@alignCast(data)));
        return self.buildGraph(builder, input_tokens, n_tokens, mem_ctx, start_pos);
    }

    fn getParamsAdapter(data: *anyopaque) *const model.ModelParams {
        const self = @as(*MiniCPMModel, @ptrCast(@alignCast(data)));
        return self.getParams();
    }

    fn resetSSMStatesAdapter(data: *anyopaque) void {
        const self = @as(*MiniCPMModel, @ptrCast(@alignCast(data)));
        _ = self;
    }
};

// ============================================================================
// 辅助函数
// ============================================================================

pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !MiniCPMParams {
    var p = MiniCPMParams{};

    p.base.n_vocab = gguf_file.getU32("minicpm.vocab_size") orelse
        gguf_file.getU32("llama.vocab_size") orelse
        blk: {
            if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
                if (val.value_type == .array) break :blk @intCast(val.array_val.len);
            }
            break :blk 0;
        };
    p.base.n_embd = gguf_file.getU32("minicpm.embedding_length") orelse
        gguf_file.getU32("llama.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("minicpm.attention.head_count") orelse
        gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("minicpm.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse 0;
    p.base.n_kv_head = gguf_file.getU32("minicpm.attention.head_count_kv") orelse
        gguf_file.getU32("llama.attention.head_count_kv") orelse
        gguf_file.getU32("minicpm.head_count_kv") orelse
        gguf_file.getU32("llama.head_count_kv") orelse p.base.n_head;
    p.base.n_layer = gguf_file.getU32("minicpm.block_count") orelse
        gguf_file.getU32("llama.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("minicpm.feed_forward_length") orelse
        gguf_file.getU32("llama.feed_forward_length") orelse 0;

    // MiniCPM5 uses attention.key_length and attention.value_length
    if (gguf_file.getU32("minicpm.attention.key_length")) |key_len| {
        p.base.n_head_dim = key_len;
    } else if (gguf_file.getU32("llama.attention.key_length")) |key_len| {
        p.base.n_head_dim = key_len;
    } else if (gguf_file.getU32("minicpm.attention.head_dim")) |head_dim| {
        p.base.n_head_dim = head_dim;
    } else if (gguf_file.getU32("llama.attention.head_dim")) |head_dim| {
        p.base.n_head_dim = head_dim;
    } else if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }
    p.base.n_head_dim_k = p.base.n_head_dim;
    p.base.n_head_dim_v = gguf_file.getU32("minicpm.attention.value_length") orelse
        gguf_file.getU32("llama.attention.value_length") orelse p.base.n_head_dim;

    p.base.max_seq_len = gguf_file.getU32("minicpm.context_length") orelse
        gguf_file.getU32("llama.context_length") orelse 4096;
    p.base.rope_theta = gguf_file.getF32("minicpm.rope.freq_base") orelse
        gguf_file.getF32("llama.rope.freq_base") orelse 10000.0;
    p.base.rope_dim = gguf_file.getU32("minicpm.rope.dimension_count") orelse
        gguf_file.getU32("llama.rope.dimension_count") orelse
        @divExact(p.base.n_head_dim, 2);
    p.base.norm_eps = gguf_file.getF32("minicpm.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-5;
    p.base.model_name = gguf_file.getString("general.name") orelse "";
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "llama";

    // MiniCPM / Granite 缩放参数
    p.embedding_scale = gguf_file.getF32("minicpm.embedding_scale") orelse
        gguf_file.getF32("llama.embedding_scale") orelse 12.0;
    p.residual_scale = gguf_file.getF32("minicpm.residual_scale") orelse
        gguf_file.getF32("llama.residual_scale") orelse
        if (p.base.n_layer > 0) 1.4 / @sqrt(@as(f32, @floatFromInt(p.base.n_layer))) else 1.4;
    p.logit_scale = gguf_file.getF32("minicpm.logit_scale") orelse
        gguf_file.getF32("llama.logit_scale") orelse
        if (p.base.n_embd > 0) 256.0 / @as(f32, @floatFromInt(p.base.n_embd)) else 1.0;
    p.attention_scale = gguf_file.getF32("minicpm.attention_scale") orelse
        gguf_file.getF32("llama.attention_scale") orelse 0.0;

    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required MiniCPM parameters", .{});
        return error.InvalidModelParams;
    }

    log.info("MiniCPM: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}, head_dim={d}", .{
        p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head, p.base.n_layer, p.base.n_ff, p.base.n_head_dim,
    });
    log.info("MiniCPM scales: embd={d:.4}, residual={d:.4}, logit={d:.4}, attn={d:.4}", .{
        p.embedding_scale, p.residual_scale, p.logit_scale, p.attention_scale,
    });

    return p;
}

fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *const MiniCPMParams,
    allocator: std.mem.Allocator,
) !MiniCPMWeights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading MiniCPM weights...", .{});

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
    var layers_loaded: usize = 0;
    errdefer {
        for (0..layers_loaded) |j| {
            allocator.free(layers[j].prefix);
        }
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

    log.info("All MiniCPM weights loaded ({d} layers)\n", .{n_layer});

    return MiniCPMWeights{
        .base = .{
            .params = params.base,
            .token_embd = token_embd,
            .output_weight = output_weight,
            .output_norm_weight = output_norm_weight,
        },
        .layers = layers,
    };
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

        // 从 GGUF 文件加载张量数据
        const tensor_data = gguf_file.getTensorData(info);
        const tensor_bytes = tensor.dataBytes();
        if (tensor_bytes.len != tensor_data.len) {
            log.warn("Tensor '{s}' size mismatch: expected {d} bytes, got {d} bytes\n", .{ name, tensor_bytes.len, tensor_data.len });
        }
        @memcpy(tensor_bytes, tensor_data);

        return tensor;
    }
    return error.TensorNotFound;
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "MiniCPMParams defaults" {
    const p = MiniCPMParams{};
    try testing.expectEqual(@as(u32, 0), p.base.n_vocab);
    try testing.expectEqual(@as(f32, 12.0), p.embedding_scale);
}
