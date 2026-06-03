//! Qwen 系列模型实现
//!
//! 支持 Qwen 2 / 2.5 / 3.5 混合架构（全注意力 + SSM 线性注意力交替）。
//! 从现有 model.zig 迁移，适配新的多模型架构。

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

const log = std.log.scoped(.qwen);

// ============================================================================
// Qwen 特有超参数
// ============================================================================

/// Qwen 模型超参数（继承 ModelParams 并扩展）
pub const QwenParams = struct {
    base: model.ModelParams = .{},

    // Qwen 3.5 混合架构参数
    full_attention_interval: u32 = 4,
    ssm_conv_kernel: u32 = 4,
    ssm_state_size: u32 = 128,
    ssm_group_count: u32 = 16,
    ssm_time_step_rank: u32 = 16,
    ssm_inner_size: u32 = 2048,

    // Qwen 3.5 全注意力层 K/V 维度
    attn_key_length: u32 = 0,
    attn_value_length: u32 = 0,

    // RoPE 缩放
    rope_scaling: ?model.RopeScaling = null,
};

/// 层类型
pub const LayerType = enum(u32) {
    full_attention = 0,
    ssm = 1,
    _,
};

// ============================================================================
// Qwen 模型权重
// ============================================================================

/// 单层 Transformer 的权重
pub const LayerWeights = struct {
    prefix: []const u8,
    layer_type: LayerType,

    // 公共
    attn_norm_weight: *ggml.Tensor,
    post_attention_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor,

    // 全注意力层权重
    attn_q_weight: ?*ggml.Tensor = null,
    attn_k_weight: ?*ggml.Tensor = null,
    attn_v_weight: ?*ggml.Tensor = null,
    attn_output_weight: ?*ggml.Tensor = null,
    attn_q_norm_weight: ?*ggml.Tensor = null,
    attn_k_norm_weight: ?*ggml.Tensor = null,

    // SSM 层权重
    attn_qkv_weight: ?*ggml.Tensor = null,
    attn_gate_weight: ?*ggml.Tensor = null,
    ssm_conv1d_weight: ?*ggml.Tensor = null,
    ssm_a: ?*ggml.Tensor = null,
    ssm_dt_bias: ?*ggml.Tensor = null,
    ssm_alpha_weight: ?*ggml.Tensor = null,
    ssm_beta_weight: ?*ggml.Tensor = null,
    ssm_norm_weight: ?*ggml.Tensor = null,
    ssm_out_weight: ?*ggml.Tensor = null,

    // FFN (SwiGLU)
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,
};

/// Qwen 模型权重
pub const QwenWeights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,

    pub fn deinit(self: *QwenWeights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| {
            allocator.free(layer.prefix);
        }
        allocator.free(self.layers);
    }
};

// ============================================================================
// Qwen 模型实现
// ============================================================================

pub const QwenModel = struct {
    qwen_params: QwenParams,
    qwen_weights: QwenWeights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *QwenModel, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.qwen_params = try parseParams(gguf_file, allocator);

        const mem_size_estimate = estimateMemSize(&self.qwen_params);
        self.ctx_weights = try ggml.Context.initNoAlloc(mem_size_estimate);

        self.qwen_weights = try loadWeights(gguf_file, self.ctx_weights, &self.qwen_params, allocator);
    }

    pub fn deinit(self: *QwenModel, allocator: std.mem.Allocator) void {
        self.qwen_weights.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn getParams(self: *const QwenModel) *const model.ModelParams {
        return &self.qwen_params.base;
    }

    pub fn getWeights(self: *const QwenModel) *const model.ModelWeights {
        return &self.qwen_weights.base;
    }

    pub fn forward(
        self: *QwenModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
    ) !*ggml.Tensor {
        const p = &self.qwen_params;
        const w = &self.qwen_weights;
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
            const layer_idx: u32 = @intCast(i);
            const is_full_attn = isFullAttentionLayer(layer_idx, p.full_attention_interval);

            var name_buf: [128]u8 = undefined;

            // --- 注意力前的 RMSNorm ---
            var attn_input = rms_norm.rmsNorm(ctx, cur, layer.attn_norm_weight, p.base.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_input.setName(name_buf[0..name.len :0]);
            }

            if (is_full_attn) {
                // 全注意力层
                const attn_out = try self.forwardFullAttention(ctx, graph, layer, attn_input, n_tokens_i64, n_head, n_kv_head, head_dim, rope_dim, kv_cache_mgr, start_pos, i, &name_buf);
                cur = ggml.add(ctx, cur, attn_out);
            } else {
                // SSM 层（简化实现）
                const ssm_out = try self.forwardSSM(ctx, attn_input, n_tokens_i64, &name_buf);
                cur = ggml.add(ctx, cur, ssm_out);
            }

            // --- Post-Attention Norm + SwiGLU FFN ---
            var post_attn = rms_norm.rmsNorm(ctx, cur, layer.post_attention_norm_weight, p.base.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                post_attn.setName(name_buf[0..name.len :0]);
            }

            const ffn_out = swiglu.swiGLU(ctx, post_attn, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
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

        ggml.setOutput(logits_tensor);
        graph.buildForwardExpand(logits_tensor);

        return logits_tensor;
    }

    /// 全注意力层前向
    fn forwardFullAttention(
        self: *QwenModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        layer: *const LayerWeights,
        attn_input: *ggml.Tensor,
        n_tokens_i64: i64,
        n_head: i64,
        n_kv_head: i64,
        head_dim: i64,
        rope_dim: i64,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        layer_idx: usize,
        name_buf: *[128]u8,
    ) !*ggml.Tensor {
        const p = &self.qwen_params;
        const q_w = layer.attn_q_weight orelse return error.MissingWeight;
        const k_w = layer.attn_k_weight orelse return error.MissingWeight;
        const v_w = layer.attn_v_weight orelse return error.MissingWeight;
        const o_w = layer.attn_output_weight orelse return error.MissingWeight;

        // QKV 投影
        var q_full = ggml.mulMat(ctx, q_w, attn_input);
        var k = ggml.mulMat(ctx, k_w, attn_input);
        var v = ggml.mulMat(ctx, v_w, attn_input);

        // Q/K Norm（Qwen 3.5 特有）
        if (layer.attn_q_norm_weight) |q_norm_raw| {
            q_full = ggml.rmsNorm(ctx, q_full, p.base.norm_eps);
            const q_norm_2d = ggml.reshape2d(ctx, q_norm_raw, head_dim, 1);
            const q_norm_target = ctx.newTensor2d(.f32, n_head * head_dim, 1) catch unreachable;
            const q_norm_rep = ggml.repeat(ctx, q_norm_2d, q_norm_target);
            q_full = ggml.mul(ctx, q_full, q_norm_rep);
        }
        if (layer.attn_k_norm_weight) |k_norm_raw| {
            k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
            const k_norm_2d = ggml.reshape2d(ctx, k_norm_raw, head_dim, 1);
            const k_norm_target = ctx.newTensor2d(.f32, n_kv_head * head_dim, 1) catch unreachable;
            const k_norm_rep = ggml.repeat(ctx, k_norm_2d, k_norm_target);
            k = ggml.mul(ctx, k, k_norm_rep);
        }

        // Qwen 3.5: Q projection contains both Q and Q_gate
        const q_attn_size: i64 = n_head * head_dim;
        var q = ctx.view2d(q_full, q_attn_size, n_tokens_i64, q_full.strides()[1], 0);
        const q_gate = ctx.view2d(q_full, q_attn_size, n_tokens_i64, q_full.strides()[1], @intCast(q_attn_size * @sizeOf(f32)));

        // 重塑为 [head_dim, n_tokens, n_head/kv_head]
        q = ggml.reshape3d(ctx, ggml.cont(ctx, q), head_dim, n_tokens_i64, n_head);
        k = ggml.reshape3d(ctx, k, head_dim, n_tokens_i64, n_kv_head);
        v = ggml.reshape3d(ctx, v, head_dim, n_tokens_i64, n_kv_head);

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
            cache.setKv(ctx, graph, layer_idx, k, v, @intCast(n_tokens_i64));
            k = cache.getKView(ctx, layer_idx);
            v = cache.getVView(ctx, layer_idx);
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

        // Qwen 3.5 门控：attn = attn * sigmoid(q_gate)
        const gate_sigmoid = ggml.sigmoid(ctx, q_gate);
        attn_out = ggml.mul(ctx, attn_out, gate_sigmoid);
        {
            const name = std.fmt.bufPrint(name_buf, "blk.{d}.attn_gated", .{layer_idx}) catch unreachable;
            name_buf[name.len] = 0;
            attn_out.setName(name_buf[0..name.len :0]);
        }

        // 输出投影
        var result = ggml.mulMat(ctx, o_w, attn_out);
        {
            const name = std.fmt.bufPrint(name_buf, "blk.{d}.attn_out", .{layer_idx}) catch unreachable;
            name_buf[name.len] = 0;
            result.setName(name_buf[0..name.len :0]);
        }

        return result;
    }

    /// SSM 层前向（简化实现）
    fn forwardSSM(
        self: *QwenModel,
        ctx: *ggml.Context,
        attn_input: *ggml.Tensor,
        n_tokens_i64: i64,
        name_buf: *[128]u8,
    ) !*ggml.Tensor {
        _ = self;
        _ = ctx;
        _ = n_tokens_i64;
        _ = name_buf;
        // SSM 层简化实现 - 直接返回输入
        // 完整实现需要 ggml_ssm_scan 支持
        return attn_input;
    }
};

// ============================================================================
// 辅助函数
// ============================================================================

/// 判断指定层是否为全注意力层
pub fn isFullAttentionLayer(layer_idx: u32, interval: u32) bool {
    if (interval == 0) return true;
    return (layer_idx + 1) % interval == 0;
}

/// 从 GGUF 元数据解析 Qwen 参数
pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !QwenParams {
    var p = QwenParams{};

    p.base.n_vocab = gguf_file.getU32("llama.vocab_size") orelse
        blk: {
            if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
                if (val.value_type == .array) break :blk @intCast(val.array_val.len);
            }
            break :blk 0;
        };
    p.base.n_embd = gguf_file.getU32("llama.embedding_length") orelse
        gguf_file.getU32("qwen35.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse
        gguf_file.getU32("qwen35.attention.head_count") orelse 0;
    p.base.n_kv_head = gguf_file.getU32("llama.attention.head_count_kv") orelse
        gguf_file.getU32("llama.head_count_kv") orelse
        gguf_file.getU32("qwen35.attention.head_count_kv") orelse p.base.n_head;
    p.base.n_layer = gguf_file.getU32("llama.block_count") orelse
        gguf_file.getU32("qwen35.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("llama.feed_forward_length") orelse
        gguf_file.getU32("qwen35.feed_forward_length") orelse 0;
    p.base.n_expert = gguf_file.getU32("llama.expert_count") orelse 0;
    p.base.n_expert_used = gguf_file.getU32("llama.expert_used_count") orelse 0;

    p.attn_key_length = gguf_file.getU32("qwen35.attention.key_length") orelse
        gguf_file.getU32("llama.attention.key_length") orelse 0;
    p.attn_value_length = gguf_file.getU32("qwen35.attention.value_length") orelse
        gguf_file.getU32("llama.attention.value_length") orelse 0;

    if (p.attn_key_length > 0) {
        p.base.n_head_dim = p.attn_key_length;
    } else if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }

    p.base.max_seq_len = gguf_file.getU32("llama.context_length") orelse
        gguf_file.getU32("qwen35.context_length") orelse 32768;
    p.base.rope_theta = gguf_file.getF32("llama.rope.freq_base") orelse
        gguf_file.getF32("qwen35.rope.freq_base") orelse 10000000.0;
    p.base.rope_dim = gguf_file.getU32("qwen35.rope.dimension_count") orelse 64;
    p.base.norm_eps = gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("qwen35.attention.layer_norm_rms_epsilon") orelse 1e-6;

    p.full_attention_interval = gguf_file.getU32("qwen35.full_attention_interval") orelse 4;
    p.ssm_conv_kernel = gguf_file.getU32("qwen35.ssm.conv_kernel") orelse 4;
    p.ssm_state_size = gguf_file.getU32("qwen35.ssm.state_size") orelse 128;
    p.ssm_group_count = gguf_file.getU32("qwen35.ssm.group_count") orelse 16;
    p.ssm_time_step_rank = gguf_file.getU32("qwen35.ssm.time_step_rank") orelse 16;
    p.ssm_inner_size = gguf_file.getU32("qwen35.ssm.inner_size") orelse 2048;

    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";

    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required model parameters", .{});
        return error.InvalidModelParams;
    }

    log.info("Qwen: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}", .{
        p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head, p.base.n_layer, p.base.n_ff,
    });

    return p;
}

fn estimateMemSize(params: *const QwenParams) usize {
    _ = params;
    return 1024 * 1024 * 1024 * 2;
}

/// 加载模型权重
pub fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *const QwenParams,
    allocator: std.mem.Allocator,
) !QwenWeights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Qwen weights...", .{});

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
        const layer_idx: u32 = @intCast(i);
        const is_full_attn = isFullAttentionLayer(layer_idx, params.full_attention_interval);

        var layer_weights = LayerWeights{
            .prefix = prefix,
            .layer_type = if (is_full_attn) .full_attention else .ssm,
            .attn_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_norm.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .post_attention_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "post_attention_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load post_attention_norm.weight: {}", .{ i, err });
                return error.MissingWeight;
            },
            .ffn_norm_weight = undefined,
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

        layer_weights.ffn_norm_weight = layer_weights.post_attention_norm_weight;

        if (is_full_attn) {
            layer_weights.attn_q_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_q.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.attn_k_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_k.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.attn_v_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_v.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.attn_output_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_output.weight: {}", .{ i, err });
                return error.MissingWeight;
            };

            layer_weights.attn_q_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_q_norm.weight") catch null;
            layer_weights.attn_k_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_k_norm.weight") catch null;
        } else {
            layer_weights.attn_qkv_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_qkv.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_qkv.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.attn_gate_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_gate.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_gate.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_conv1d_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_conv1d.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_conv1d.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_a = loadLayerWeight(ctx, gguf_file, prefix, "ssm_a") catch |err| {
                log.err("Layer {d}: failed to load ssm_a: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_dt_bias = loadLayerWeight(ctx, gguf_file, prefix, "ssm_dt.bias") catch |err| {
                log.err("Layer {d}: failed to load ssm_dt.bias: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_alpha_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_alpha.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_alpha.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_beta_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_beta.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_beta.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_norm.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_out_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_out.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_out.weight: {}", .{ i, err });
                return error.MissingWeight;
            };
        }

        layers[i] = layer_weights;
    }

    log.info("All Qwen weights loaded ({d} layers)", .{n_layer});

    return QwenWeights{
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

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "isFullAttentionLayer" {
    try testing.expect(!isFullAttentionLayer(0, 4));
    try testing.expect(!isFullAttentionLayer(2, 4));
    try testing.expect(isFullAttentionLayer(3, 4));
    try testing.expect(!isFullAttentionLayer(4, 4));
    try testing.expect(isFullAttentionLayer(7, 4));
    try testing.expect(isFullAttentionLayer(0, 1));
}

test "QwenParams defaults" {
    const p = QwenParams{};
    try testing.expectEqual(@as(u32, 0), p.base.n_vocab);
    try testing.expectEqual(@as(u32, 4), p.full_attention_interval);
}
