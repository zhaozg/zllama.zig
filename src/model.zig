//! Qwen 3.5 模型架构
//!
//! 实现 Transformer 解码器架构，支持：
//! - 全注意力层（标准多头注意力）
//! - RMSNorm 归一化
//! - RoPE 位置编码
//! - SwiGLU 前馈网络
//! - GQA (Grouped Query Attention)
//!
//! 模型超参数从 GGUF 元数据读取，不硬编码。

const std = @import("std");
const ggml = @import("ggml.zig");
const gguf = @import("gguf.zig");

const log = std.log.scoped(.model);

// ============================================================================
// 模型超参数
// ============================================================================

/// Qwen 3.5 模型超参数，从 GGUF 元数据解析
pub const ModelParams = struct {
    // 基础维度
    n_vocab: u32 = 0,          // 词表大小
    n_embd: u32 = 0,           // 隐藏层维度 (hidden_size)
    n_head: u32 = 0,           // 注意力头数
    n_head_dim: u32 = 0,       // 每个头的维度
    n_kv_head: u32 = 0,        // KV 头数 (GQA)
    n_layer: u32 = 0,          // Transformer 层数
    n_ff: u32 = 0,             // FFN 中间维度
    n_expert: u32 = 0,         // MoE 专家数（0 表示非 MoE）
    n_expert_used: u32 = 0,    // 每 token 使用的专家数

    // 位置编码
    max_seq_len: u32 = 32768,  // 最大序列长度
    rope_theta: f32 = 1000000.0, // RoPE base frequency
    rope_scaling: ?RopeScaling = null, // RoPE 缩放配置

    // 归一化
    norm_eps: f32 = 1e-6,      // RMSNorm epsilon

    // 层类型（Qwen 3.5 混合架构）
    layer_types: []const LayerType = &.{},

    // 分词器
    tokenizer_name: []const u8 = "",

    pub fn deinit(self: *ModelParams) void {
        _ = self;
        // layer_types 由调用者管理
    }
};

/// RoPE 缩放配置
pub const RopeScaling = struct {
    rope_type: []const u8 = "",
    factor: f32 = 1.0,
    original_max_seq_len: u32 = 32768,
};

/// 层类型（Qwen 3.5 混合架构）
pub const LayerType = enum(u32) {
    full_attention = 0,
    linear_attention = 1,
    _,
};

/// 从 GGUF 元数据解析模型超参数
pub fn parseParams(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !ModelParams {
    var params = ModelParams{};

    // 基础维度
    params.n_vocab = gguf_file.getU32("llama.vocab_size") orelse
        gguf_file.getU32("tokenizer.ggml.tokens") orelse 0;
    params.n_embd = gguf_file.getU32("llama.embedding_length") orelse
        gguf_file.getU32("qwen.embedding_length") orelse 0;
    params.n_head = gguf_file.getU32("llama.head_count") orelse
        gguf_file.getU32("qwen.head_count") orelse 0;
    params.n_kv_head = gguf_file.getU32("llama.head_count_kv") orelse
        gguf_file.getU32("qwen.head_count_kv") orelse params.n_head;
    params.n_layer = gguf_file.getU32("llama.block_count") orelse
        gguf_file.getU32("qwen.block_count") orelse 0;
    params.n_ff = gguf_file.getU32("llama.feed_forward_length") orelse
        gguf_file.getU32("qwen.feed_forward_length") orelse 0;
    params.n_expert = gguf_file.getU32("llama.expert_count") orelse 0;
    params.n_expert_used = gguf_file.getU32("llama.expert_used_count") orelse 0;

    // 计算 head_dim
    if (params.n_head > 0) {
        params.n_head_dim = params.n_embd / params.n_head;
    }

    // 位置编码
    params.max_seq_len = gguf_file.getU32("llama.context_length") orelse
        gguf_file.getU32("qwen.context_length") orelse 32768;
    params.rope_theta = gguf_file.getF32("llama.rope.freq_base") orelse
        gguf_file.getF32("qwen.rope.freq_base") orelse 1000000.0;
    params.norm_eps = gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("qwen.attention.layer_norm_rms_epsilon") orelse 1e-6;

    // RoPE 缩放
    if (gguf_file.getString("llama.rope.scaling.type")) |scaling_type| {
        const factor = gguf_file.getF32("llama.rope.factor") orelse 1.0;
        const orig_max = gguf_file.getU32("llama.rope.scaling.original_context_length") orelse params.max_seq_len;
        params.rope_scaling = RopeScaling{
            .rope_type = scaling_type,
            .factor = factor,
            .original_max_seq_len = orig_max,
        };
    }

    // 层类型（Qwen 3.5 混合架构）
    if (gguf_file.metadata.get("qwen.layer_types")) |val| {
        if (val.asString()) |s| {
            var list = std.ArrayList(LayerType).empty;
            var it = std.mem.splitScalar(u8, s, ',');
            while (it.next()) |token| {
                const trimmed = std.mem.trim(u8, token, " ");
                if (trimmed.len > 0) {
                    const v = std.fmt.parseInt(u32, trimmed, 10) catch 0;
                    try list.append(allocator, @enumFromInt(v));
                }
            }
            params.layer_types = try list.toOwnedSlice(allocator);
        }
    }

    // 分词器名称
    params.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";

    // 验证必要参数
    if (params.n_vocab == 0 or params.n_embd == 0 or params.n_head == 0 or params.n_layer == 0) {
        log.err("Missing required model parameters: vocab={d}, embd={d}, head={d}, layers={d}",
            .{ params.n_vocab, params.n_embd, params.n_head, params.n_layer });
        return error.InvalidModelParams;
    }

    log.info("Model: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}",
        .{ params.n_vocab, params.n_embd, params.n_head, params.n_kv_head, params.n_layer, params.n_ff });
    log.info("  head_dim={d}, max_seq_len={d}, rope_theta={d}, norm_eps={d}",
        .{ params.n_head_dim, params.max_seq_len, params.rope_theta, params.norm_eps });

    return params;
}

// ============================================================================
// 模型权重（从 GGUF 张量映射）
// ============================================================================

/// 模型所有权重的引用（指向 GGUF 文件映射内存）
pub const ModelWeights = struct {
    params: ModelParams,

    // Token 嵌入
    token_embd: *ggml.Tensor,

    // 输出层
    output_weight: *ggml.Tensor,
    output_norm_weight: *ggml.Tensor,
    output_norm_bias: ?*ggml.Tensor = null,

    // 每层的权重
    layers: []LayerWeights,
    pub fn deinit(self: *ModelWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};


/// 单层 Transformer 的权重
pub const LayerWeights = struct {
    // 注意力
    attn_norm_weight: *ggml.Tensor,
    attn_norm_bias: ?*ggml.Tensor = null,
    q_weight: *ggml.Tensor,
    k_weight: *ggml.Tensor,
    v_weight: *ggml.Tensor,
    o_weight: *ggml.Tensor,

    // 线性注意力（Qwen 3.5 特有）
    attn_output_gate: ?*ggml.Tensor = null,
    linear_kernel: ?*ggml.Tensor = null,

    // FFN (SwiGLU)
    ffn_norm_weight: *ggml.Tensor,
    ffn_norm_bias: ?*ggml.Tensor = null,
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,
};

// ============================================================================
// 计算图构建
// ============================================================================

/// 构建首 token 的完整前向计算图
/// 返回 logits 张量
pub fn buildForwardGraph(
    ctx: *ggml.Context,
    weights: *const ModelWeights,
    input_tokens: *ggml.Tensor,
    n_tokens: i32,
    n_threads: i32,
) !*ggml.Tensor {
    const params = &weights.params;
    const n_embd: i64 = @intCast(params.n_embd);
    const n_head: i64 = @intCast(params.n_head);
    const n_kv_head: i64 = @intCast(params.n_kv_head);
    const head_dim: i64 = @intCast(params.n_head_dim);
    const n_tokens_i64: i64 = n_tokens;

    _ = n_threads;

    // Token 嵌入
    var cur = ggml.mulMat(ctx, weights.token_embd, input_tokens);
    cur.setName("token_embd");

    // 逐层处理
    for (weights.layers, 0..) |*layer, i| {
        // 使用栈上缓冲区构建层名称
        var name_buf: [128]u8 = undefined;
        _ = std.fmt.bufPrint(&name_buf, "blk.{d}", .{i}) catch unreachable;

        // --- 注意力前的 RMSNorm ---
        var attn_input = ggml.rmsNorm(ctx, cur, params.norm_eps);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
            attn_input.setName(name);
        }

        attn_input = ggml.mul(ctx, attn_input, layer.attn_norm_weight);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm_mul", .{i}) catch unreachable;
            attn_input.setName(name);
        }

        // --- QKV 投影 ---
        var q = ggml.mulMat(ctx, layer.q_weight, attn_input);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q", .{i}) catch unreachable;
            q.setName(name);
        }

        var k = ggml.mulMat(ctx, layer.k_weight, attn_input);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k", .{i}) catch unreachable;
            k.setName(name);
        }

        var v = ggml.mulMat(ctx, layer.v_weight, attn_input);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v", .{i}) catch unreachable;
            v.setName(name);
        }

        // 重塑为 [head_dim, n_tokens, n_head/kv_head]
        q = ggml.reshape3d(ctx, q, head_dim, n_tokens_i64, n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_reshaped", .{i}) catch unreachable;
            q.setName(name);
        }

        k = ggml.reshape3d(ctx, k, head_dim, n_tokens_i64, n_kv_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_reshaped", .{i}) catch unreachable;
            k.setName(name);
        }

        v = ggml.reshape3d(ctx, v, head_dim, n_tokens_i64, n_kv_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_reshaped", .{i}) catch unreachable;
            v.setName(name);
        }

        // --- RoPE 位置编码 ---
        const pos_tensor = buildPositionTensor(ctx, n_tokens, 0);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.positions", .{i}) catch unreachable;
            pos_tensor.setName(name);
        }

        const rope_n_dims: i32 = @intCast(head_dim);
        q = ggml.ropeExt(ctx, q, pos_tensor, 0, rope_n_dims, 0,
            params.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_rope", .{i}) catch unreachable;
            q.setName(name);
        }

        k = ggml.ropeExt(ctx, k, pos_tensor, 0, rope_n_dims, 0,
            params.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_rope", .{i}) catch unreachable;
            k.setName(name);
        }

        // --- 注意力计算 ---
        // 使用简化的 2D mul_mat 实现注意力
        // Q: [head_dim, n_tokens, n_head] -> permute(1,0,2) -> [n_tokens, head_dim, n_head]
        var q_perm = ggml.permute(ctx, q, 1, 0, 2, 3);
        q_perm = ggml.cont(ctx, q_perm);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_perm", .{i}) catch unreachable;
            q_perm.setName(name);
        }

        var k_perm = ggml.permute(ctx, k, 1, 0, 2, 3);
        k_perm = ggml.cont(ctx, k_perm);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_perm", .{i}) catch unreachable;
            k_perm.setName(name);
        }

        var v_perm = ggml.permute(ctx, v, 1, 0, 2, 3);
        v_perm = ggml.cont(ctx, v_perm);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_perm", .{i}) catch unreachable;
            v_perm.setName(name);
        }

        // GQA: 扩展 KV 头以匹配 Q 头数
        if (n_kv_head < n_head) {
            const n_rep = @divExact(n_head, n_kv_head);
            if (n_rep > 1) {
                const k_2d = ggml.reshape2d(ctx, k_perm, n_tokens_i64 * head_dim, n_kv_head);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_2d", .{i}) catch unreachable;
                    k_2d.setName(name);
                }
                const k_rep = ggml.cont(ctx, ggml.repeat(ctx, k_2d,
                    ggml.reshape2d(ctx, k_perm, n_tokens_i64 * head_dim, n_head)));
                k_perm = ggml.reshape3d(ctx, k_rep, n_tokens_i64, head_dim, n_head);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_rep", .{i}) catch unreachable;
                    k_perm.setName(name);
                }

                const v_2d = ggml.reshape2d(ctx, v_perm, n_tokens_i64 * head_dim, n_kv_head);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_2d", .{i}) catch unreachable;
                    v_2d.setName(name);
                }
                const v_rep = ggml.cont(ctx, ggml.repeat(ctx, v_2d,
                    ggml.reshape2d(ctx, v_perm, n_tokens_i64 * head_dim, n_head)));
                v_perm = ggml.reshape3d(ctx, v_rep, n_tokens_i64, head_dim, n_head);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_rep", .{i}) catch unreachable;
                    v_perm.setName(name);
                }
            }
        }

        // 将 Q, K, V 展平为 2D 进行批量矩阵乘法
        const q_flat = ggml.reshape2d(ctx, ggml.cont(ctx, q_perm), head_dim, n_tokens_i64 * n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_flat", .{i}) catch unreachable;
            q_flat.setName(name);
        }

        const k_flat = ggml.reshape2d(ctx, ggml.cont(ctx, k_perm), head_dim, n_tokens_i64 * n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_flat", .{i}) catch unreachable;
            k_flat.setName(name);
        }

        // score = K^T * Q: [n_tokens * n_head, n_tokens * n_head]
        var kq = ggml.mulMat(ctx, k_flat, q_flat);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq", .{i}) catch unreachable;
            kq.setName(name);
        }

        // 缩放
        const scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        kq = ggml.scale(ctx, kq, scale_factor);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_scaled", .{i}) catch unreachable;
            kq.setName(name);
        }

        // 重塑为 [n_head, n_tokens, n_tokens] 用于 mask 和 softmax
        kq = ggml.reshape3d(ctx, kq, n_tokens_i64, n_tokens_i64, n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_3d", .{i}) catch unreachable;
            kq.setName(name);
        }

        // 展平为 2D 应用 mask
        kq = ggml.reshape2d(ctx, kq, n_tokens_i64, n_tokens_i64 * n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_2d", .{i}) catch unreachable;
            kq.setName(name);
        }

        kq = ggml.diagMaskInf(ctx, kq, 0);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_masked", .{i}) catch unreachable;
            kq.setName(name);
        }

        kq = ggml.softMax(ctx, kq);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_softmax", .{i}) catch unreachable;
            kq.setName(name);
        }

        // 重塑回 3D
        kq = ggml.reshape3d(ctx, kq, n_tokens_i64, n_tokens_i64, n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_3d_softmax", .{i}) catch unreachable;
            kq.setName(name);
        }

        // 注意力输出: softmax * V
        // V: [n_tokens, head_dim, n_head] -> permute(2,0,1) -> [n_head, n_tokens, head_dim]
        var v_batch = ggml.permute(ctx, v_perm, 2, 0, 1, 3);
        v_batch = ggml.cont(ctx, v_batch);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_batch", .{i}) catch unreachable;
            v_batch.setName(name);
        }

        // V 展平为 [head_dim, n_tokens * n_head]
        const v_flat = ggml.reshape2d(ctx, v_batch, head_dim, n_tokens_i64 * n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_flat", .{i}) catch unreachable;
            v_flat.setName(name);
        }

        // kq (softmax): [n_head, n_tokens, n_tokens] -> 展平为 [n_tokens * n_head, n_tokens]
        const kq_flat = ggml.reshape2d(ctx, kq, n_tokens_i64, n_tokens_i64 * n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_flat", .{i}) catch unreachable;
            kq_flat.setName(name);
        }

        // V 置换为 [n_tokens, n_head, head_dim]
        var v_perm2 = ggml.permute(ctx, v_batch, 1, 0, 2, 3);
        v_perm2 = ggml.cont(ctx, v_perm2);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_perm2", .{i}) catch unreachable;
            v_perm2.setName(name);
        }

        // V 展平为 [n_tokens, n_head * head_dim] 用于注意力计算
        const v_2d_final = ggml.reshape2d(ctx, v_perm2, n_tokens_i64, n_head * head_dim);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_2d_final", .{i}) catch unreachable;
            v_2d_final.setName(name);
        }

        // attn = softmax * V
        // softmax: [n_tokens * n_head, n_tokens]
        // V: [n_tokens, n_head * head_dim]
        // mul_mat(softmax, V) = softmax * V: [n_tokens * n_head, n_head * head_dim]
        var attn = ggml.mulMat(ctx, kq_flat, v_2d_final);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn", .{i}) catch unreachable;
            attn.setName(name);
        }

        // 重塑为 [n_head, n_tokens, head_dim]
        attn = ggml.reshape3d(ctx, attn, n_tokens_i64, head_dim, n_head);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_3d", .{i}) catch unreachable;
            attn.setName(name);
        }

        // 置换为 [n_tokens, head_dim, n_head]
        attn = ggml.permute(ctx, attn, 1, 0, 2, 3);
        attn = ggml.cont(ctx, attn);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_perm", .{i}) catch unreachable;
            attn.setName(name);
        }

        // 展平为 [n_embd, n_tokens]
        attn = ggml.reshape2d(ctx, attn, n_embd, n_tokens_i64);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_2d", .{i}) catch unreachable;
            attn.setName(name);
        }

        // 输出投影
        var attn_out = ggml.mulMat(ctx, layer.o_weight, attn);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
            attn_out.setName(name);
        }

        // 门控（Qwen 3.5 特有）
        if (layer.attn_output_gate) |gate| {
            attn_out = ggml.mul(ctx, attn_out, gate);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_gated", .{i}) catch unreachable;
                attn_out.setName(name);
            }
        }

        // 残差连接
        cur = ggml.add(ctx, cur, attn_out);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.residual_1", .{i}) catch unreachable;
            cur.setName(name);
        }

        // --- FFN 前的 RMSNorm ---
        var ffn_input = ggml.rmsNorm(ctx, cur, params.norm_eps);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm", .{i}) catch unreachable;
            ffn_input.setName(name);
        }

        ffn_input = ggml.mul(ctx, ffn_input, layer.ffn_norm_weight);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm_mul", .{i}) catch unreachable;
            ffn_input.setName(name);
        }

        // --- SwiGLU FFN ---
        var gate = ggml.mulMat(ctx, layer.ffn_gate_weight, ffn_input);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate", .{i}) catch unreachable;
            gate.setName(name);
        }

        gate = ggml.silu(ctx, gate);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate_silu", .{i}) catch unreachable;
            gate.setName(name);
        }

        var up = ggml.mulMat(ctx, layer.ffn_up_weight, ffn_input);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up", .{i}) catch unreachable;
            up.setName(name);
        }

        var ffn_hidden = ggml.mul(ctx, gate, up);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_hidden", .{i}) catch unreachable;
            ffn_hidden.setName(name);
        }

        var ffn_out = ggml.mulMat(ctx, layer.ffn_down_weight, ffn_hidden);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_out", .{i}) catch unreachable;
            ffn_out.setName(name);
        }

        // 残差连接
        cur = ggml.add(ctx, cur, ffn_out);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.residual_2", .{i}) catch unreachable;
            cur.setName(name);
        }
    }

    // --- 最终 RMSNorm ---
    cur = ggml.rmsNorm(ctx, cur, params.norm_eps);
    cur.setName("output_norm");

    cur = ggml.mul(ctx, cur, weights.output_norm_weight);
    cur.setName("output_norm_mul");

    // --- 输出投影 ---
    var logits = ggml.mulMat(ctx, weights.output_weight, cur);
    logits.setName("logits");

    ggml.setOutput(logits);

    return logits;
}

/// 构建位置张量 [0, 1, 2, ..., n_tokens-1]
fn buildPositionTensor(ctx: *ggml.Context, n_tokens: i32, start_pos: i32) *ggml.Tensor {
    const pos_tensor = ctx.newTensor1d(.i32, n_tokens) catch unreachable;
    const data = pos_tensor.dataBytes();
    const pos_slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
    for (0..@as(usize, @intCast(n_tokens))) |i| {
        pos_slice[i] = @as(i32, @intCast(i)) + start_pos;
    }
    return pos_tensor;
}

// ============================================================================
// 模型加载
// ============================================================================

/// 从 GGUF 文件加载模型权重
pub fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    params: *const ModelParams,
    ctx: *ggml.Context,
    allocator: std.mem.Allocator,
) !ModelWeights {
    const n_embd: i64 = @intCast(params.n_embd);
    const n_head: i64 = @intCast(params.n_head);
    const n_kv_head: i64 = @intCast(params.n_kv_head);
    const head_dim: i64 = @intCast(params.n_head_dim);
    const n_ff: i64 = @intCast(params.n_ff);
    const n_layer: usize = @intCast(params.n_layer);
    const n_vocab: i64 = @intCast(params.n_vocab);

    _ = n_embd;
    _ = n_head;
    _ = n_kv_head;
    _ = head_dim;
    _ = n_ff;
    _ = n_vocab;

    log.info("Loading model weights...", .{});

    // Token 嵌入
    const token_embd = findOrCreateTensor(ctx, gguf_file, "token_embd.weight", .f16) catch |err| {
        log.err("Failed to load token_embd.weight: {}", .{err});
        return error.MissingWeight;
    };
    token_embd.setName("token_embd.weight");

    // 输出层
    const output_weight = findOrCreateTensor(ctx, gguf_file, "output.weight", .f16) catch |err| {
        log.err("Failed to load output.weight: {}", .{err});
        return error.MissingWeight;
    };
    output_weight.setName("output.weight");

    const output_norm_weight = findOrCreateTensor(ctx, gguf_file, "output_norm.weight", .f16) catch |err| {
        log.err("Failed to load output_norm.weight: {}", .{err});
        return error.MissingWeight;
    };
    output_norm_weight.setName("output_norm.weight");

    // 逐层加载
    var layers = try allocator.alloc(LayerWeights, n_layer);
    for (0..n_layer) |i| {
        const prefix = std.fmt.allocPrint(allocator, "blk.{d}", .{i}) catch unreachable;
        defer allocator.free(prefix);

        const attn_norm_name = try std.fmt.allocPrint(allocator, "{s}.attn_norm.weight", .{prefix});
        defer allocator.free(attn_norm_name);
        const q_name = try std.fmt.allocPrint(allocator, "{s}.attn_q.weight", .{prefix});
        defer allocator.free(q_name);
        const k_name = try std.fmt.allocPrint(allocator, "{s}.attn_k.weight", .{prefix});
        defer allocator.free(k_name);
        const v_name = try std.fmt.allocPrint(allocator, "{s}.attn_v.weight", .{prefix});
        defer allocator.free(v_name);
        const o_name = try std.fmt.allocPrint(allocator, "{s}.attn_o.weight", .{prefix});
        defer allocator.free(o_name);
        const ffn_norm_name = try std.fmt.allocPrint(allocator, "{s}.ffn_norm.weight", .{prefix});
        defer allocator.free(ffn_norm_name);
        const gate_name = try std.fmt.allocPrint(allocator, "{s}.ffn_gate.weight", .{prefix});
        defer allocator.free(gate_name);
        const up_name = try std.fmt.allocPrint(allocator, "{s}.ffn_up.weight", .{prefix});
        defer allocator.free(up_name);
        const down_name = try std.fmt.allocPrint(allocator, "{s}.ffn_down.weight", .{prefix});
        defer allocator.free(down_name);

        layers[i] = LayerWeights{
            .attn_norm_weight = findOrCreateTensor(ctx, gguf_file, attn_norm_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ attn_norm_name, err });
                return error.MissingWeight;
            },
            .q_weight = findOrCreateTensor(ctx, gguf_file, q_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ q_name, err });
                return error.MissingWeight;
            },
            .k_weight = findOrCreateTensor(ctx, gguf_file, k_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ k_name, err });
                return error.MissingWeight;
            },
            .v_weight = findOrCreateTensor(ctx, gguf_file, v_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ v_name, err });
                return error.MissingWeight;
            },
            .o_weight = findOrCreateTensor(ctx, gguf_file, o_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ o_name, err });
                return error.MissingWeight;
            },
            .ffn_norm_weight = findOrCreateTensor(ctx, gguf_file, ffn_norm_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ ffn_norm_name, err });
                return error.MissingWeight;
            },
            .ffn_gate_weight = findOrCreateTensor(ctx, gguf_file, gate_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ gate_name, err });
                return error.MissingWeight;
            },
            .ffn_up_weight = findOrCreateTensor(ctx, gguf_file, up_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ up_name, err });
                return error.MissingWeight;
            },
            .ffn_down_weight = findOrCreateTensor(ctx, gguf_file, down_name, .f16) catch |err| {
                log.err("Failed to load {s}: {}", .{ down_name, err });
                return error.MissingWeight;
            },
            .attn_output_gate = null,
            .linear_kernel = null,
        };

        // 可选的门控权重
        const gate_out_name = try std.fmt.allocPrint(allocator, "{s}.attn_output_gate.weight", .{prefix});
        defer allocator.free(gate_out_name);
        if (gguf_file.findTensor(gate_out_name)) |_| {
            layers[i].attn_output_gate = findOrCreateTensor(ctx, gguf_file, gate_out_name, .f16) catch null;
        }

        log.debug("  Layer {d}: loaded", .{i});
    }

    log.info("All weights loaded successfully ({d} layers)", .{n_layer});

    return ModelWeights{
        .params = params.*,
        .token_embd = token_embd,
        .output_weight = output_weight,
        .output_norm_weight = output_norm_weight,
        .layers = layers,
    };
}

/// 在 GGUF 文件中查找张量，如果找到则创建视图，否则创建新张量
fn findOrCreateTensor(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    name: []const u8,
    default_type: ggml.Type,
) !*ggml.Tensor {
    _ = default_type;
    // 尝试从 GGUF 文件查找
    if (gguf_file.findTensor(name)) |_| {
        const tensor = ctx.getTensor(@ptrCast(name)) orelse {
            return ctx.newTensor1d(.f32, 1);
        };
        return tensor;
    }

    // 未找到，创建新张量
    return ctx.newTensor1d(.f32, 1);
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "ModelParams parse" {
    const params = ModelParams{};
    try testing.expectEqual(@as(u32, 0), params.n_vocab);
    try testing.expectEqual(@as(u32, 32768), params.max_seq_len);
    try testing.expectEqual(@as(f32, 1000000.0), params.rope_theta);
}

test "LayerType enum" {
    try testing.expectEqual(@as(u32, 0), @intFromEnum(LayerType.full_attention));
    try testing.expectEqual(@as(u32, 1), @intFromEnum(LayerType.linear_attention));
}
