//! LLaMA 系列模型实现
//!
//! 支持 LLaMA 2 / 3 / 3.1 架构。
//! 标准 Transformer 架构：RMSNorm + RoPE + GQA + SwiGLU FFN。

const std = @import("std");
const ggml = @import("../ggml.zig");
const gguf = @import("../gguf.zig");
const kv_cache = @import("../kv_cache.zig");
const model = @import("../model.zig");
const rms_norm = @import("../layers/rms_norm.zig");
const rope = @import("../layers/rope.zig");
const swiglu = @import("../layers/swiglu.zig");
const attention = @import("../layers/attention.zig");
const embed = @import("../layers/embed.zig");

const log = std.log.scoped(.llama);

// ============================================================================
// LLaMA 特有超参数
// ============================================================================

pub const LlamaParams = struct {
    base: model.ModelParams = .{},
    rope_scaling: ?model.RopeScaling = null,
};

// ============================================================================
// LLaMA 模型权重
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

pub const LlamaWeights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,

    pub fn deinit(self: *LlamaWeights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| {
            allocator.free(layer.prefix);
        }
        allocator.free(self.layers);
    }
};

// ============================================================================
// LLaMA 模型实现
// ============================================================================

pub const LlamaModel = struct {
    llm_params: LlamaParams,
    llm_weights: LlamaWeights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *LlamaModel, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.llm_params = try parseParams(gguf_file, allocator);

        // 根据模型参数动态估计所需内存
        // 对于 Q4_K_M 量化，每个参数约 0.5 字节，加上元数据开销
        const mem_size_estimate = estimateMemSize(&self.llm_params);
        self.ctx_weights = try ggml.Context.initNoAlloc(mem_size_estimate);

        self.llm_weights = try loadWeights(gguf_file, self.ctx_weights, &self.llm_params, allocator);
    }

    pub fn deinit(self: *LlamaModel, allocator: std.mem.Allocator) void {
        self.llm_weights.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn getParams(self: *const LlamaModel) *const model.ModelParams {
        return &self.llm_params.base;
    }

    pub fn getWeights(self: *const LlamaModel) *const model.ModelWeights {
        return &self.llm_weights.base;
    }

    pub fn forward(
        self: *LlamaModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
    ) !*ggml.Tensor {
        const p = &self.llm_params;
        const w = &self.llm_weights;
        const n_head: i64 = @intCast(p.base.n_head);
        const n_kv_head: i64 = @intCast(p.base.n_kv_head);
        const head_dim: i64 = @intCast(p.base.n_head_dim);
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.base.rope_dim);

        // Token 嵌入
        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur.setName("token_embd");

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

            // 重塑为 [head_dim, n_head/kv_head, n_tokens]（与 llama.cpp 布局一致）
            q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);
            k = ggml.reshape3d(ctx, k, head_dim, n_kv_head, n_tokens_i64);
            v = ggml.reshape3d(ctx, v, head_dim, n_kv_head, n_tokens_i64);

            // RoPE 位置编码
            const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens_i64), start_pos);
            const rope_result = rope.applyRope(ctx, q, k, pos_tensor, .{
                .rope_dim = rope_dim,
                .rope_theta = p.base.rope_theta,
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
            var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
                .n_head = n_head,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            });

            // 输出投影
            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
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

            cur = ggml.add(ctx, cur, ffn_out);
        }

        // --- 最终 RMSNorm ---
        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.base.norm_eps);
        cur.setName("output_norm");

        // --- 输出投影 ---
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);
        logits_tensor.setName("logits");

        // graph output determined by buildForwardExpand
        graph.buildForwardExpand(logits_tensor);

        return logits_tensor;
    }
};

// ============================================================================
// 辅助函数
// ============================================================================

pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !LlamaParams {
    var p = LlamaParams{};

    p.base.n_vocab = gguf_file.getU32("llama.vocab_size") orelse
        blk: {
            if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
                if (val.value_type == .array) break :blk @intCast(val.array_val.len);
            }
            break :blk 0;
        };
    p.base.n_embd = gguf_file.getU32("llama.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse 0;
    p.base.n_kv_head = gguf_file.getU32("llama.attention.head_count_kv") orelse
        gguf_file.getU32("llama.head_count_kv") orelse p.base.n_head;
    p.base.n_layer = gguf_file.getU32("llama.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("llama.feed_forward_length") orelse 0;

    if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }

    p.base.max_seq_len = gguf_file.getU32("llama.context_length") orelse 4096;
    p.base.rope_theta = gguf_file.getF32("llama.rope.freq_base") orelse 10000.0;
    p.base.rope_dim = gguf_file.getU32("llama.rope.dimension_count") orelse
        @divExact(p.base.n_head_dim, 2);
    p.base.norm_eps = gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-5;
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "llama";

    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required LLaMA parameters", .{});
        return error.InvalidModelParams;
    }

    log.info("LLaMA: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}", .{
        p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head, p.base.n_layer, p.base.n_ff,
    });

    return p;
}

fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *const LlamaParams,
    allocator: std.mem.Allocator,
) !LlamaWeights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading LLaMA weights...", .{});

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
    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});

        layers[i] = LayerWeights{
            .prefix = prefix,
            .attn_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_norm.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_norm.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .attn_q_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_q.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .attn_k_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_k.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .attn_v_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_v.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .attn_output_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_output.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_gate_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_gate.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_up_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_up.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_down_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_down.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
        };
    }

    log.info("All LLaMA weights loaded ({d} layers)", .{n_layer});

    return LlamaWeights{
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

        // 从 GGUF 文件加载张量数据
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

/// 估计模型权重所需的内存大小
/// 对于 Q4_K_M 量化，每个参数约 0.5 字节，加上 ggml 元数据开销
/// 对于 f32 张量，每个参数 4 字节
fn estimateMemSize(params: *const LlamaParams) usize {
    const n_vocab = params.base.n_vocab;
    const n_embd = params.base.n_embd;
    _ = params.base.n_head;
    const n_kv_head = params.base.n_kv_head;
    const n_layer = params.base.n_layer;
    const n_ff = params.base.n_ff;
    const head_dim = params.base.n_head_dim;

    // 对于量化模型，每个元素平均约 0.5 字节（Q4_K_M）
    // 对于 f32 张量（如 norm），每个元素 4 字节
    const bytes_per_elem_quant: f64 = 0.6; // 略高于 0.5 以留余量
    const bytes_per_elem_f32: u64 = 4;

    // token_embd: [n_embd, n_vocab]
    var total: u64 = @intFromFloat(@as(f64, @floatFromInt(n_embd * n_vocab)) * bytes_per_elem_quant);

    // output.weight: [n_embd, n_vocab]（如果存在）
    total += @intFromFloat(@as(f64, @floatFromInt(n_embd * n_vocab)) * bytes_per_elem_quant);

    // output_norm.weight: [n_embd] f32
    total += n_embd * bytes_per_elem_f32;

    // 每层权重
    const kv_dim = n_kv_head * head_dim;
    const per_layer_quant: u64 = @as(u64, @intFromFloat(@as(f64, @floatFromInt(n_embd * n_embd)) * bytes_per_elem_quant)) + // attn_q
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(n_embd * kv_dim)) * bytes_per_elem_quant)) + // attn_k
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(n_embd * kv_dim)) * bytes_per_elem_quant)) + // attn_v
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(n_embd * n_embd)) * bytes_per_elem_quant)) + // attn_output
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(n_embd * n_ff)) * bytes_per_elem_quant)) + // ffn_gate
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(n_embd * n_ff)) * bytes_per_elem_quant)) + // ffn_up
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(n_ff * n_embd)) * bytes_per_elem_quant)); // ffn_down
    const per_layer_f32: u64 = 2 * n_embd * bytes_per_elem_f32; // attn_norm + ffn_norm

    total += (per_layer_quant + per_layer_f32) * n_layer;

    // 加上 ggml 元数据开销（每个张量约 200 字节）
    const n_tensors: u64 = 3 + n_layer * 9; // 3 个全局 + 每层 9 个
    total += n_tensors * 256;

    // 额外预留 20% 空间
    total = @intFromFloat(@as(f64, @floatFromInt(total)) * 1.2);

    log.info("Estimated weights memory: {d} MB", .{total / (1024 * 1024)});

    return total;
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "LlamaParams defaults" {
    const p = LlamaParams{};
    try testing.expectEqual(@as(u32, 0), p.base.n_vocab);
    try testing.expectEqual(@as(f32, 10000.0), p.base.rope_theta);
}
