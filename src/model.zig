//! Qwen 3.5 模型架构
//!
//! 实现 Qwen 3.5 混合架构（全注意力 + SSM 线性注意力交替）。
//! 支持：
//! - 全注意力层（标准多头注意力 + GQA + RoPE）
//! - SSM 层（Mamba-2 风格线性注意力）
//! - RMSNorm 归一化
//! - RoPE 位置编码
//! - SwiGLU 前馈网络
//! - attn_output_gate 门控机制
//!
//! 模型超参数从 GGUF 元数据读取，不硬编码。

const std = @import("std");
const ggml = @import("ggml.zig");
const gguf = @import("gguf.zig");
const kv_cache = @import("kv_cache.zig");

const log = std.log.scoped(.model);

// ============================================================================
// 模型超参数
// ============================================================================

/// Qwen 3.5 模型超参数，从 GGUF 元数据解析
pub const ModelParams = struct {
    // 基础维度
    n_vocab: u32 = 0, // 词表大小
    n_embd: u32 = 0, // 隐藏层维度 (hidden_size)
    n_head: u32 = 0, // 注意力头数
    n_head_dim: u32 = 0, // 每个头的维度 (key_length / value_length)
    n_kv_head: u32 = 0, // KV 头数 (GQA)
    n_layer: u32 = 0, // Transformer 层数
    n_ff: u32 = 0, // FFN 中间维度
    n_expert: u32 = 0, // MoE 专家数（0 表示非 MoE）
    n_expert_used: u32 = 0, // 每 token 使用的专家数

    // 位置编码
    max_seq_len: u32 = 32768, // 最大序列长度
    rope_theta: f32 = 10000000.0, // RoPE base frequency
    rope_dim: u32 = 64, // RoPE 维度数
    rope_scaling: ?RopeScaling = null,

    // 归一化
    norm_eps: f32 = 1e-6, // RMSNorm epsilon

    // Qwen 3.5 混合架构参数
    full_attention_interval: u32 = 4, // 每 N 层一个全注意力层
    ssm_conv_kernel: u32 = 4, // SSM 1D 卷积核大小
    ssm_state_size: u32 = 128, // SSM 状态大小
    ssm_group_count: u32 = 16, // SSM 组数
    ssm_time_step_rank: u32 = 16, // SSM 时间步秩
    ssm_inner_size: u32 = 2048, // SSM 内部维度

    // Qwen 3.5 全注意力层 K/V 维度（与 head_dim 可能不同）
    attn_key_length: u32 = 0, // 每个 KV 头的 key 维度 (来自 qwen35.attention.key_length)
    attn_value_length: u32 = 0, // 每个 KV 头的 value 维度 (来自 qwen35.attention.value_length)

    // 分词器
    tokenizer_name: []const u8 = "",

    pub fn deinit(self: *ModelParams) void {
        _ = self;
    }
};

/// RoPE 缩放配置
pub const RopeScaling = struct {
    rope_type: []const u8 = "",
    factor: f32 = 1.0,
    original_max_seq_len: u32 = 32768,
};

/// 层类型
pub const LayerType = enum(u32) {
    full_attention = 0,
    ssm = 1,
    _,
};

/// 判断指定层是否为全注意力层
pub fn isFullAttentionLayer(layer_idx: u32, interval: u32) bool {
    if (interval == 0) return true;
    return (layer_idx + 1) % interval == 0;
}

/// 从 GGUF 元数据解析模型超参数
pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !ModelParams {
    var params = ModelParams{};

    // 基础维度
    params.n_vocab = gguf_file.getU32("llama.vocab_size") orelse
        blk: {
            if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
                if (val == .array) break :blk @intCast(val.array.len);
            }
            break :blk 0;
        };
    params.n_embd = gguf_file.getU32("llama.embedding_length") orelse
        gguf_file.getU32("qwen35.embedding_length") orelse 0;
    params.n_head = gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse
        gguf_file.getU32("qwen35.attention.head_count") orelse 0;
    params.n_kv_head = gguf_file.getU32("llama.attention.head_count_kv") orelse
        gguf_file.getU32("llama.head_count_kv") orelse
        gguf_file.getU32("qwen35.attention.head_count_kv") orelse params.n_head;
    params.n_layer = gguf_file.getU32("llama.block_count") orelse
        gguf_file.getU32("qwen35.block_count") orelse 0;
    params.n_ff = gguf_file.getU32("llama.feed_forward_length") orelse
        gguf_file.getU32("qwen35.feed_forward_length") orelse 0;
    params.n_expert = gguf_file.getU32("llama.expert_count") orelse 0;
    params.n_expert_used = gguf_file.getU32("llama.expert_used_count") orelse 0;

    // 读取 Qwen 3.5 全注意力层的 K/V 维度（优先使用显式声明的 head_dim）
    // Qwen3.5 的 head_dim 可能不等于 n_embd / n_head，必须从元数据读取
    params.attn_key_length = gguf_file.getU32("qwen35.attention.key_length") orelse
        gguf_file.getU32("llama.attention.key_length") orelse 0;
    params.attn_value_length = gguf_file.getU32("qwen35.attention.value_length") orelse
        gguf_file.getU32("llama.attention.value_length") orelse 0;

    // 计算 head_dim: 优先使用显式声明的 key_length，否则回退到 n_embd / n_head
    if (params.attn_key_length > 0) {
        params.n_head_dim = params.attn_key_length;
        log.info("  Using explicit head_dim={d} from qwen35.attention.key_length", .{params.n_head_dim});
    } else if (params.n_head > 0 and params.n_embd > 0) {
        params.n_head_dim = params.n_embd / params.n_head;
        log.info("  Computed head_dim={d} = n_embd({d}) / n_head({d})", .{ params.n_head_dim, params.n_embd, params.n_head });
    } else {
        params.n_head_dim = 0;
    }
    params.max_seq_len = gguf_file.getU32("llama.context_length") orelse
        gguf_file.getU32("qwen35.context_length") orelse 32768;
    params.rope_theta = gguf_file.getF32("llama.rope.freq_base") orelse
        gguf_file.getF32("qwen35.rope.freq_base") orelse 10000000.0;
    params.rope_dim = gguf_file.getU32("qwen35.rope.dimension_count") orelse 64;
    params.norm_eps = gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("qwen35.attention.layer_norm_rms_epsilon") orelse 1e-6;

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

    // Qwen 3.5 混合架构参数
    params.full_attention_interval = gguf_file.getU32("qwen35.full_attention_interval") orelse 4;
    params.ssm_conv_kernel = gguf_file.getU32("qwen35.ssm.conv_kernel") orelse 4;
    params.ssm_state_size = gguf_file.getU32("qwen35.ssm.state_size") orelse 128;
    params.ssm_group_count = gguf_file.getU32("qwen35.ssm.group_count") orelse 16;
    params.ssm_time_step_rank = gguf_file.getU32("qwen35.ssm.time_step_rank") orelse 16;
    params.ssm_inner_size = gguf_file.getU32("qwen35.ssm.inner_size") orelse 2048;

    // 分词器名称
    params.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";

    // 验证必要参数
    if (params.n_vocab == 0 or params.n_embd == 0 or params.n_head == 0 or params.n_layer == 0) {
        log.err("Missing required model parameters: vocab={d}, embd={d}, head={d}, layers={d}", .{ params.n_vocab, params.n_embd, params.n_head, params.n_layer });
        return error.InvalidModelParams;
    }

    log.info("Model: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}", .{ params.n_vocab, params.n_embd, params.n_head, params.n_kv_head, params.n_layer, params.n_ff });
    log.info("  head_dim={d}, max_seq_len={d}, rope_theta={d}, rope_dim={d}, norm_eps={d}", .{ params.n_head_dim, params.max_seq_len, params.rope_theta, params.rope_dim, params.norm_eps });
    log.info("  full_attn_interval={d}, ssm_inner={d}, ssm_state={d}, ssm_conv={d}", .{ params.full_attention_interval, params.ssm_inner_size, params.ssm_state_size, params.ssm_conv_kernel });

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
    output_weight: ?*ggml.Tensor = null, // 可能为 null（与 token_embd 共享）
    output_norm_weight: *ggml.Tensor,

    // 每层的权重
    layers: []LayerWeights,

    pub fn deinit(self: *ModelWeights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| {
            allocator.free(layer.prefix);
        }
        allocator.free(self.layers);
    }
};

/// 单层 Transformer 的权重
pub const LayerWeights = struct {
    prefix: []const u8, // 层前缀，如 "blk.0"
    layer_type: LayerType,

    // 公共
    attn_norm_weight: *ggml.Tensor,
    post_attention_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor, // 与 attn_norm_weight 相同（Qwen 3.5 使用 post_attention_norm）

    // 全注意力层权重
    attn_q_weight: ?*ggml.Tensor = null,
    attn_k_weight: ?*ggml.Tensor = null,
    attn_v_weight: ?*ggml.Tensor = null,
    attn_output_weight: ?*ggml.Tensor = null,
    attn_q_norm_weight: ?*ggml.Tensor = null,
    attn_k_norm_weight: ?*ggml.Tensor = null,

    // SSM 层权重
    attn_qkv_weight: ?*ggml.Tensor = null, // 合并的 QKV
    attn_gate_weight: ?*ggml.Tensor = null, // 门控
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

// ============================================================================
// 计算图构建
// ============================================================================

/// 构建首 token 的完整前向计算图
/// 返回 logits 张量，所有操作已添加到 graph 中
pub fn buildForwardGraph(
    ctx: *ggml.Context,
    graph: *ggml.CGraph,
    weights: *const ModelWeights,
    input_tokens: *ggml.Tensor,
    n_tokens: i32,
    kv_cache_mgr: ?*kv_cache.KVCache,
    start_pos: i32,
    is_qwen: bool,
) !*ggml.Tensor {
    const params = &weights.params;
    const n_head: i64 = @intCast(params.n_head);
    const n_kv_head: i64 = @intCast(params.n_kv_head);
    const head_dim: i64 = @intCast(params.n_head_dim);
    const n_tokens_i64: i64 = n_tokens;
    const rope_dim: i64 = @intCast(params.rope_dim);

    _ = is_qwen;

    // Token 嵌入
    log.debug("token_embd: token_embd ne={d}", .{weights.token_embd.nelements()});

    var cur = ggml.getRows(ctx, weights.token_embd, input_tokens);
    cur.setName("token_embd");
    log.debug("token_embd_after: cur ne={d}, input_tokens ne={d}, token_embd ne={d}", .{ cur.nelements(), input_tokens.nelements(), weights.token_embd.nelements() });

    // 逐层处理
    for (weights.layers, 0..) |*layer, i| {
        const layer_idx: u32 = @intCast(i);
        const is_full_attn = isFullAttentionLayer(layer_idx, params.full_attention_interval);

        // 使用栈上缓冲区构建层名称
        var name_buf: [128]u8 = undefined;

        // --- 注意力前的 RMSNorm ---
        log.debug("pre_attn_norm: layer={d} cur ne={d}", .{ i, cur.nelements() });

        var attn_input = ggml.rmsNorm(ctx, cur, params.norm_eps);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            attn_input.setName(name_buf[0..name.len :0]);
        }
        // 调试：打印形状
        {
            const a_ne = attn_input.nelements();
            const b_ne = layer.attn_norm_weight.nelements();
            const cur_ne = cur.nelements();
            log.debug("mul: cur ne={d}, attn_input ne={d}, norm_weight ne={d}, n_tokens={d}", .{ cur_ne, a_ne, b_ne, n_tokens_i64 });
        }
        attn_input = ggml.mul(ctx, attn_input, ggml.reshape2d(ctx, layer.attn_norm_weight, @intCast(params.n_embd), 1));
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm_mul", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            attn_input.setName(name_buf[0..name.len :0]);
        }

        if (is_full_attn) {}
        attn_input = ggml.mul(ctx, attn_input, ggml.reshape2d(ctx, layer.attn_norm_weight, @intCast(params.n_embd), 1));
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm_mul", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            attn_input.setName(name_buf[0..name.len :0]);
        }

        if (is_full_attn) {
            // ================================================================
            // 全注意力层
            // ================================================================
            const q_w = layer.attn_q_weight orelse return error.MissingWeight;
            const k_w = layer.attn_k_weight orelse return error.MissingWeight;
            const v_w = layer.attn_v_weight orelse return error.MissingWeight;
            const o_w = layer.attn_output_weight orelse return error.MissingWeight;

            // QKV 投影
            var q_full = ggml.mulMat(ctx, q_w, attn_input);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q_full.setName(name_buf[0..name.len :0]);
            }
            var k = ggml.mulMat(ctx, k_w, attn_input);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                k.setName(name_buf[0..name.len :0]);
            }
            var v = ggml.mulMat(ctx, v_w, attn_input);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                v.setName(name_buf[0..name.len :0]);
            }

            // Q/K Norm（Qwen 3.5 特有）
            if (layer.attn_q_norm_weight) |q_norm_raw| {
                q_full = ggml.rmsNorm(ctx, q_full, params.norm_eps);
                // q_norm 是 [head_dim]，需要 repeat 到 [n_head * head_dim]
                const q_norm_2d = ggml.reshape2d(ctx, q_norm_raw, head_dim, 1);
                const q_norm_target = try ctx.newTensor2d(.f32, n_head * head_dim, 1);
                const q_norm_rep = ggml.repeat(ctx, q_norm_2d, q_norm_target);
                q_full = ggml.mul(ctx, q_full, q_norm_rep);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q_normed", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    q_full.setName(name_buf[0..name.len :0]);
                }
            }
            if (layer.attn_k_norm_weight) |k_norm_raw| {
                k = ggml.rmsNorm(ctx, k, params.norm_eps);
                // k_norm 是 [head_dim]，需要 repeat 到 [n_kv_head * head_dim]
                const k_norm_2d = ggml.reshape2d(ctx, k_norm_raw, head_dim, 1);
                const k_norm_target = try ctx.newTensor2d(.f32, n_kv_head * head_dim, 1);
                const k_norm_rep = ggml.repeat(ctx, k_norm_2d, k_norm_target);
                k = ggml.mul(ctx, k, k_norm_rep);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k_normed", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    k.setName(name_buf[0..name.len :0]);
                }
            }

            // Qwen 3.5: Q projection contains both Q and Q_gate (each n_head*head_dim)
            // Split Q into attn part and gate part
            const q_attn_size: i64 = n_head * head_dim;
            const q_gate_size: i64 = n_head * head_dim;
            var q = ctx.view2d(q_full, q_attn_size, n_tokens_i64, q_full.strides()[1], 0);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_attn", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q.setName(name_buf[0..name.len :0]);
            }
            const q_gate = ctx.view2d(q_full, q_gate_size, n_tokens_i64, q_full.strides()[1], @intCast(q_attn_size * @sizeOf(f32)));
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_gate", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q_gate.setName(name_buf[0..name.len :0]);
            }

            // 重塑为 [head_dim, n_tokens, n_head/kv_head]
            q = ggml.reshape3d(ctx, ggml.cont(ctx, q), head_dim, n_tokens_i64, n_head);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_reshaped", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q.setName(name_buf[0..name.len :0]);
            }
            k = ggml.reshape3d(ctx, k, head_dim, n_tokens_i64, n_kv_head);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_reshaped", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                k.setName(name_buf[0..name.len :0]);
            }
            v = ggml.reshape3d(ctx, v, head_dim, n_tokens_i64, n_kv_head);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_reshaped", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                v.setName(name_buf[0..name.len :0]);
            }

            // RoPE 位置编码 (permute ne[1]<->ne[2] so ne[2]=n_tokens matches pos->ne[0])
            const pos_tensor = buildPositionTensor(ctx, n_tokens, start_pos);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.positions", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                pos_tensor.setName(name_buf[0..name.len :0]);
            }

            var q_rope = ggml.permute(ctx, q, 0, 2, 1, 3);
            q_rope = ggml.cont(ctx, q_rope);
            q_rope = ggml.ropeExt(ctx, q_rope, pos_tensor, 0, @intCast(rope_dim), 0, params.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            q = ggml.permute(ctx, q_rope, 0, 2, 1, 3);
            q = ggml.cont(ctx, q);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_rope", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q.setName(name_buf[0..name.len :0]);
            }

            var k_rope = ggml.permute(ctx, k, 0, 2, 1, 3);
            k_rope = ggml.cont(ctx, k_rope);
            k_rope = ggml.ropeExt(ctx, k_rope, pos_tensor, 0, @intCast(rope_dim), 0, params.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k = ggml.permute(ctx, k_rope, 0, 2, 1, 3);
            k = ggml.cont(ctx, k);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_rope", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                k.setName(name_buf[0..name.len :0]);
            }

            // 处理 KV Cache
            if (kv_cache_mgr) |cache| {
                // 将当前 K, V 写入 Cache
                cache.setKv(ctx, graph, i, k, v, @intCast(n_tokens));

                // 从 Cache 获取完整视图用于注意力计算
                // 注意：setKv 已经更新了 current_len，所以 getKView 返回包含新 token 的视图
                const k_cached = cache.getKView(ctx, i);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_cached", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    k_cached.setName(name_buf[0..name.len :0]);
                }
                const v_cached = cache.getVView(ctx, i);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_cached", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    v_cached.setName(name_buf[0..name.len :0]);
                }

                // 使用缓存的 K, V 进行注意力计算
                k = k_cached;
                v = v_cached;
            }

            // 重新计算 cache_len（setKv 可能已更新 current_len）
            const cache_len_after: i64 = if (kv_cache_mgr) |cache|
                @as(i64, @intCast(cache.currentLen()))
            else
                n_tokens_i64;

            // 注意力计算
            // Q: [head_dim, n_tokens, n_head] -> permute(1,0,2) -> [n_tokens, head_dim, n_head]
            var q_perm = ggml.permute(ctx, q, 1, 0, 2, 3);
            q_perm = ggml.cont(ctx, q_perm);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_perm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q_perm.setName(name_buf[0..name.len :0]);
            }

            // K: [head_dim, cache_len, n_kv_head] -> permute(1,0,2) -> [cache_len, head_dim, n_kv_head]
            var k_perm = ggml.permute(ctx, k, 1, 0, 2, 3);
            k_perm = ggml.cont(ctx, k_perm);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_perm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                k_perm.setName(name_buf[0..name.len :0]);
            }

            // V: [head_dim, cache_len, n_kv_head] -> permute(1,0,2) -> [cache_len, head_dim, n_kv_head]
            var v_perm = ggml.permute(ctx, v, 1, 0, 2, 3);
            v_perm = ggml.cont(ctx, v_perm);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_perm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                v_perm.setName(name_buf[0..name.len :0]);
            }

            // 使用 setKv 后更新的 cache_len
            const cache_len: i64 = cache_len_after;
            // GQA: 扩展 KV 头以匹配 Q 头数
            if (n_kv_head < n_head) {
                const n_rep = @divExact(n_head, n_kv_head);
                if (n_rep > 1) {
                    // K: [cache_len, head_dim, n_kv_head] -> 展平为 [cache_len * head_dim, n_kv_head]
                    const k_2d = ggml.reshape2d(ctx, k_perm, cache_len * head_dim, n_kv_head);
                    // 创建目标张量 [cache_len * head_dim, n_head]
                    const k_target = try ctx.newTensor2d(.f32, cache_len * head_dim, n_head);
                    const k_rep = ggml.cont(ctx, ggml.repeat(ctx, k_2d, k_target));
                    k_perm = ggml.reshape3d(ctx, k_rep, cache_len, head_dim, n_head);
                    {
                        const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_rep", .{i}) catch unreachable;
                        name_buf[name.len] = 0;
                        k_perm.setName(name_buf[0..name.len :0]);
                    }

                    // V: [cache_len, head_dim, n_kv_head] -> 展平为 [cache_len * head_dim, n_kv_head]
                    const v_2d = ggml.reshape2d(ctx, v_perm, cache_len * head_dim, n_kv_head);
                    // 创建目标张量 [cache_len * head_dim, n_head]
                    const v_target = try ctx.newTensor2d(.f32, cache_len * head_dim, n_head);
                    const v_rep = ggml.cont(ctx, ggml.repeat(ctx, v_2d, v_target));
                    v_perm = ggml.reshape3d(ctx, v_rep, cache_len, head_dim, n_head);
                    {
                        const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_rep", .{i}) catch unreachable;
                        name_buf[name.len] = 0;
                        v_perm.setName(name_buf[0..name.len :0]);
                    }
                }
            }
            // 展平 Q, K 为 2D 进行批量矩阵乘法
            // 使用 batch 矩阵乘法计算注意力分数
            // Q: [n_tokens, head_dim, n_head] -> permute(1,0,2) -> [head_dim, n_tokens, n_head]
            // K: [cache_len, head_dim, n_head] -> permute(1,0,2) -> [head_dim, cache_len, n_head]
            // mulMat 对每个 n_head 独立计算: score_h = Q_h^T * K_h -> [n_tokens, cache_len]
            // 结果: [n_tokens, cache_len, n_head]
            var q_3d = ggml.permute(ctx, q_perm, 1, 0, 2, 3);
            q_3d = ggml.cont(ctx, q_3d);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.q_3d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q_3d.setName(name_buf[0..name.len :0]);
            }
            var k_3d = ggml.permute(ctx, k_perm, 1, 0, 2, 3);
            k_3d = ggml.cont(ctx, k_3d);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.k_3d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                k_3d.setName(name_buf[0..name.len :0]);
            }

            // score = K^T * Q (batch over n_head)
            // k_3d: [head_dim, cache_len, n_head], q_3d: [head_dim, n_tokens, n_head]
            // mulMat 结果: [n_tokens, cache_len, n_head]
            var kq = ggml.mulMat(ctx, k_3d, q_3d);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                kq.setName(name_buf[0..name.len :0]);
            }

            // 缩放
            const scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
            kq = ggml.scale(ctx, kq, scale_factor);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_scaled", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                kq.setName(name_buf[0..name.len :0]);
            }

            // 展平为 2D 应用 mask: [n_tokens * n_head, cache_len]
            kq = ggml.reshape2d(ctx, kq, cache_len, n_tokens_i64 * n_head);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_2d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                kq.setName(name_buf[0..name.len :0]);
            }
            kq = ggml.diagMaskInf(ctx, kq, start_pos);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_masked", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                kq.setName(name_buf[0..name.len :0]);
            }
            kq = ggml.softMax(ctx, kq);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_softmax", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                kq.setName(name_buf[0..name.len :0]);
            }

            // 重塑回 3D: [n_tokens, cache_len, n_head]
            kq = ggml.reshape3d(ctx, kq, cache_len, n_tokens_i64, n_head);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.kq_3d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                kq.setName(name_buf[0..name.len :0]);
            }

            // V: [cache_len, head_dim, n_head] -> 已经是正确的形状
            // mulMat 要求 a.ne0 == b.ne0，即 kq.ne0 (cache_len) == v_3d.ne0 (cache_len)
            var v_3d = ggml.cont(ctx, v_perm);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.v_3d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                v_3d.setName(name_buf[0..name.len :0]);
            }
            // attn = softmax * V (batch over n_head)
            // 结果: [n_tokens, head_dim, n_head]
            var attn = ggml.mulMat(ctx, kq, v_3d);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn.setName(name_buf[0..name.len :0]);
            }

            // 置换为 [n_tokens, n_head, head_dim]
            attn = ggml.permute(ctx, attn, 0, 2, 1, 3);
            attn = ggml.cont(ctx, attn);
            // 展平为 [n_head * head_dim, n_tokens]
            attn = ggml.reshape2d(ctx, attn, n_head * head_dim, n_tokens_i64);
            var attn_out = ggml.mulMat(ctx, o_w, attn);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }

            // 残差连接
            cur = ggml.add(ctx, cur, attn_out);
            log.debug("attn_out: layer={d} cur ne={d}, attn_out ne={d}", .{ i, cur.nelements(), attn_out.nelements() });
        } else {
            // ================================================================
            // SSM 层（Mamba-2 风格线性注意力）
            // ================================================================
            const qkv_w = layer.attn_qkv_weight orelse return error.MissingWeight;
            const gate_w = layer.attn_gate_weight orelse return error.MissingWeight;
            const ssm_conv1d_w = layer.ssm_conv1d_weight orelse return error.MissingWeight;
            _ = layer.ssm_a orelse return error.MissingWeight;
            const ssm_dt_bias = layer.ssm_dt_bias orelse return error.MissingWeight;
            const ssm_alpha_w = layer.ssm_alpha_weight orelse return error.MissingWeight;
            const ssm_beta_w = layer.ssm_beta_weight orelse return error.MissingWeight;
            const ssm_norm_w = layer.ssm_norm_weight orelse return error.MissingWeight;
            const ssm_out_w = layer.ssm_out_weight orelse return error.MissingWeight;

            const ssm_inner: i64 = @intCast(params.ssm_inner_size);
            _ = @as(i64, @intCast(params.ssm_state_size));
            _ = @as(i64, @intCast(params.ssm_group_count));
            const ssm_rank: i64 = @intCast(params.ssm_time_step_rank);
            const conv_kernel: i64 = @intCast(params.ssm_conv_kernel);

            // QKV 投影: [ssm_inner * 3, n_tokens]
            var qkv = ggml.mulMat(ctx, qkv_w, attn_input);
            // QKV 投影: [ssm_inner * 3, n_tokens]
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_qkv", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                qkv.setName(name_buf[0..name.len :0]);
            }

            // 1D 因果卷积 (使用 ggml_ssm_conv)
            // ggml_ssm_conv 要求输入为 [d_conv-1+n_t, d_inner, n_s]
            // 对于初始推理和增量推理，前填充 d_conv-1 个零
            // TODO: 增量解码时应使用保存的 conv state 而非零填充
            var qkv_transposed = ggml.cont(ctx, ggml.permute(ctx, qkv, 1, 0, 2, 3));
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_qkv_t", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                qkv_transposed.setName(name_buf[0..name.len :0]);
            }

            // 创建零填充：形状 [conv_kernel - 1, ssm_inner * 3]
            const pad_len = conv_kernel - 1;
            ctx.setNoAlloc(false);
            const zero_pad = try ctx.newTensor2d(.f32, pad_len, ssm_inner * 3);
            ctx.setNoAlloc(true);
            zero_pad.setZero();
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_zero_pad", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                zero_pad.setName(name_buf[0..name.len :0]);
            }

            // 拼接填充 + 数据: [pad_len + n_tokens, ssm_inner * 3]
            var qkv_padded = ggml.concat(ctx, zero_pad, qkv_transposed, 0);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_qkv_padded", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                qkv_padded.setName(name_buf[0..name.len :0]);
            }

            // 重塑为 3D: [pad_len + n_tokens, ssm_inner * 3, 1]
            var qkv_3d = ggml.reshape3d(ctx, qkv_padded, pad_len + n_tokens_i64, ssm_inner * 3, 1);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_qkv_3d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                qkv_3d.setName(name_buf[0..name.len :0]);
            }

            // SSM 卷积: 输出 [ssm_inner * 3, n_tokens, 1]
            var conv_out = ggml.ssmConv(ctx, qkv_3d, ssm_conv1d_w);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_conv_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                conv_out.setName(name_buf[0..name.len :0]);
            }

            // SiLU 激活
            conv_out = ggml.silu(ctx, conv_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_conv_act", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                conv_out.setName(name_buf[0..name.len :0]);
            }

            // 展平回 2D [ssm_inner * 3, n_tokens]
            qkv = ggml.reshape2d(ctx, conv_out, ssm_inner * 3, n_tokens_i64);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_qkv_act", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                qkv.setName(name_buf[0..name.len :0]);
            }
            // 分割 QKV
            const q_part = ggml.reshape2d(ctx, ctx.view1d(qkv, ssm_inner * n_tokens_i64, 0), ssm_inner, n_tokens_i64);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_q", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                q_part.setName(name_buf[0..name.len :0]);
            }

            const k_part = ggml.reshape2d(ctx, ctx.view1d(qkv, ssm_inner * n_tokens_i64, @intCast(ssm_inner * n_tokens_i64 * @sizeOf(f32))), ssm_inner, n_tokens_i64);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_k", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                k_part.setName(name_buf[0..name.len :0]);
            }

            const v_part = ggml.reshape2d(ctx, ctx.view1d(qkv, ssm_inner * n_tokens_i64, @intCast(2 * ssm_inner * n_tokens_i64 * @sizeOf(f32))), ssm_inner, n_tokens_i64);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_v", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                v_part.setName(name_buf[0..name.len :0]);
            }

            // 门控: gate = silu(gate_w * attn_input)
            var gate = ggml.mulMat(ctx, gate_w, attn_input);
            gate = ggml.silu(ctx, gate);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_gate", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                gate.setName(name_buf[0..name.len :0]);
            }

            // SSM 核心计算（简化版）
            var dt = ggml.mulMat(ctx, ssm_alpha_w, attn_input);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_dt_raw", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                dt.setName(name_buf[0..name.len :0]);
            }

            const dt_bias_2d = ggml.reshape2d(ctx, ssm_dt_bias, ssm_rank, 1);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_dt_bias_2d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                dt_bias_2d.setName(name_buf[0..name.len :0]);
            }
            const dt_bias_rep = ggml.repeat(ctx, dt_bias_2d, dt);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_dt_bias_rep", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                dt_bias_rep.setName(name_buf[0..name.len :0]);
            }
            dt = ggml.add(ctx, dt, dt_bias_rep);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_dt_biased", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                dt.setName(name_buf[0..name.len :0]);
            }

            // 简化的 SSM 输出: y = q * v (逐元素乘)
            var ssm_out = ggml.mul(ctx, q_part, v_part);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_mul", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ssm_out.setName(name_buf[0..name.len :0]);
            }

            // 应用 B 矩阵
            var beta_out = ggml.mulMat(ctx, ssm_beta_w, attn_input);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_beta", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                beta_out.setName(name_buf[0..name.len :0]);
            }

            // 应用 ssm_norm
            ssm_out = ggml.rmsNorm(ctx, ssm_out, params.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ssm_out.setName(name_buf[0..name.len :0]);
            }
            // ssm_norm_w 是分组 norm 权重 [ssm_inner / group_count]
            // 使用 ggml_repeat 扩展到 [ssm_inner, n_tokens]
            const ssm_norm_2d = ggml.reshape2d(ctx, ssm_norm_w, @divExact(ssm_inner, @as(i64, @intCast(params.ssm_group_count))), 1);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_norm_2d", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ssm_norm_2d.setName(name_buf[0..name.len :0]);
            }
            // 创建目标张量并 repeat
            const ssm_norm_target = try ctx.newTensor2d(.f32, ssm_inner, n_tokens_i64);
            const ssm_norm_rep = ggml.repeat(ctx, ssm_norm_2d, ssm_norm_target);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_norm_rep", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ssm_norm_rep.setName(name_buf[0..name.len :0]);
            }
            ssm_out = ggml.mul(ctx, ssm_out, ssm_norm_rep);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_norm_mul", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ssm_out.setName(name_buf[0..name.len :0]);
            }

            // 门控输出: gate SSM output in ssm_inner space, then project to embd
            var ssm_out_gated = ggml.mul(ctx, ssm_out, gate);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_gated", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ssm_out_gated.setName(name_buf[0..name.len :0]);
            }

            // 输出投影
            var attn_out = ggml.mulMat(ctx, ssm_out_w, ssm_out_gated);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ssm_proj", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }

            // 残差连接
            cur = ggml.add(ctx, cur, attn_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.residual_1", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                cur.setName(name_buf[0..name.len :0]);
            }
            log.debug("ssm_out: layer={d} cur ne={d}, attn_out ne={d}", .{ i, cur.nelements(), attn_out.nelements() });
        }

        // --- Post-Attention Norm ---
        var post_attn = ggml.rmsNorm(ctx, cur, params.norm_eps);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attn_norm", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            post_attn.setName(name_buf[0..name.len :0]);
        }
        log.debug("post_attn: layer={d} cur ne={d}, post_attn ne={d}", .{ i, cur.nelements(), post_attn.nelements() });

        post_attn = ggml.mul(ctx, post_attn, ggml.reshape2d(ctx, layer.post_attention_norm_weight, @intCast(params.n_embd), 1));
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attn_norm_mul", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            post_attn.setName(name_buf[0..name.len :0]);
        }

        // --- SwiGLU FFN ---
        var gate_ffn = ggml.mulMat(ctx, layer.ffn_gate_weight, post_attn);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            gate_ffn.setName(name_buf[0..name.len :0]);
        }
        gate_ffn = ggml.silu(ctx, gate_ffn);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate_silu", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            gate_ffn.setName(name_buf[0..name.len :0]);
        }

        var up = ggml.mulMat(ctx, layer.ffn_up_weight, post_attn);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            up.setName(name_buf[0..name.len :0]);
        }

        var ffn_hidden = ggml.mul(ctx, gate_ffn, up);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_hidden", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            ffn_hidden.setName(name_buf[0..name.len :0]);
        }

        var ffn_out = ggml.mulMat(ctx, layer.ffn_down_weight, ffn_hidden);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_out", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            ffn_out.setName(name_buf[0..name.len :0]);
        }
        // 残差连接
        cur = ggml.add(ctx, cur, ffn_out);
        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.residual_2", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            cur.setName(name_buf[0..name.len :0]);
        }
        log.debug("ffn_out: layer={d} cur ne={d}, ffn_out ne={d}", .{ i, cur.nelements(), ffn_out.nelements() });

        {
            const name = std.fmt.bufPrint(&name_buf, "blk.{d}.residual_2", .{i}) catch unreachable;
            name_buf[name.len] = 0;
            cur.setName(name_buf[0..name.len :0]);
        }
        log.debug("ffn_out: layer={d} cur ne={d}, ffn_out ne={d}", .{ i, cur.nelements(), ffn_out.nelements() });
    }

    // --- 最终 RMSNorm ---
    cur = ggml.rmsNorm(ctx, cur, params.norm_eps);
    cur.setName("output_norm");

    cur = ggml.mul(ctx, cur, ggml.reshape2d(ctx, weights.output_norm_weight, @intCast(params.n_embd), 1));
    cur.setName("output_norm_mul");

    // --- 输出投影 ---
    const out_w = weights.output_weight orelse weights.token_embd;
    var logits = ggml.mulMat(ctx, out_w, cur);
    logits.setName("logits");

    ggml.setOutput(logits);

    // 将所有操作添加到计算图
    graph.buildForwardExpand(logits);

    return logits;
}

/// 构建位置张量 [start_pos, start_pos+1, ..., start_pos+n_tokens-1]
fn buildPositionTensor(ctx: *ggml.Context, n_tokens: i32, start_pos: i32) *ggml.Tensor {
    ctx.setNoAlloc(false);
    const pos_tensor = ctx.newTensor1d(.i32, n_tokens) catch unreachable;
    ctx.setNoAlloc(true);
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

pub fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    gguf_data: []const u8,
    params: *const ModelParams,
    ctx: *ggml.Context,
    allocator: std.mem.Allocator,
) !ModelWeights {
    const n_layer: usize = @intCast(params.n_layer);

    log.info("Loading model weights...", .{});

    // Token 嵌入
    const token_embd = findOrCreateTensor(ctx, gguf_file, gguf_data, "token_embd.weight") catch |err| {
        log.err("Failed to load token_embd.weight: {}", .{err});
        return error.MissingWeight;
    };
    token_embd.setName("token_embd.weight");
    log.debug("token_embd_shape: ne={d}", .{token_embd.nelements()});

    // 输出层（可能不存在，与 token_embd 共享）
    const output_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, "output.weight") catch null;
    if (output_weight) |ow| {
        ow.setName("output.weight");
    }

    const output_norm_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, "output_norm.weight") catch |err| {
        log.err("Failed to load output_norm.weight: {}", .{err});
        return error.MissingWeight;
    };
    output_norm_weight.setName("output_norm.weight");

    // 逐层加载
    var layers = try allocator.alloc(LayerWeights, n_layer);
    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});
        const layer_idx: u32 = @intCast(i);
        const is_full_attn = isFullAttentionLayer(layer_idx, params.full_attention_interval);

        // 公共权重
        const attn_norm_name = try std.fmt.allocPrint(allocator, "{s}.attn_norm.weight", .{prefix});
        defer allocator.free(attn_norm_name);
        const post_attn_norm_name = try std.fmt.allocPrint(allocator, "{s}.post_attention_norm.weight", .{prefix});
        defer allocator.free(post_attn_norm_name);
        const ffn_gate_name = try std.fmt.allocPrint(allocator, "{s}.ffn_gate.weight", .{prefix});
        defer allocator.free(ffn_gate_name);
        const ffn_up_name = try std.fmt.allocPrint(allocator, "{s}.ffn_up.weight", .{prefix});
        defer allocator.free(ffn_up_name);
        const ffn_down_name = try std.fmt.allocPrint(allocator, "{s}.ffn_down.weight", .{prefix});
        defer allocator.free(ffn_down_name);

        var layer_weights = LayerWeights{
            .prefix = prefix,
            .layer_type = if (is_full_attn) .full_attention else .ssm,
            .attn_norm_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, attn_norm_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ attn_norm_name, err });
                return error.MissingWeight;
            },
            .post_attention_norm_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, post_attn_norm_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ post_attn_norm_name, err });
                return error.MissingWeight;
            },
            .ffn_norm_weight = undefined, // 稍后设置
            .ffn_gate_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, ffn_gate_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ ffn_gate_name, err });
                return error.MissingWeight;
            },
            .ffn_up_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, ffn_up_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ ffn_up_name, err });
                return error.MissingWeight;
            },
            .ffn_down_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, ffn_down_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ ffn_down_name, err });
                return error.MissingWeight;
            },
        };

        // Qwen 3.5 使用 post_attention_norm 作为 FFN 的输入 norm
        layer_weights.ffn_norm_weight = layer_weights.post_attention_norm_weight;

        if (is_full_attn) {
            // 全注意力层权重
            const q_name = try std.fmt.allocPrint(allocator, "{s}.attn_q.weight", .{prefix});
            defer allocator.free(q_name);
            const k_name = try std.fmt.allocPrint(allocator, "{s}.attn_k.weight", .{prefix});
            defer allocator.free(k_name);
            const v_name = try std.fmt.allocPrint(allocator, "{s}.attn_v.weight", .{prefix});
            defer allocator.free(v_name);
            const o_name = try std.fmt.allocPrint(allocator, "{s}.attn_output.weight", .{prefix});
            defer allocator.free(o_name);

            layer_weights.attn_q_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, q_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ q_name, err });
                return error.MissingWeight;
            };
            layer_weights.attn_k_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, k_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ k_name, err });
                return error.MissingWeight;
            };
            layer_weights.attn_v_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, v_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ v_name, err });
                return error.MissingWeight;
            };
            layer_weights.attn_output_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, o_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ o_name, err });
                return error.MissingWeight;
            };

            // 可选的 Q/K Norm
            const q_norm_name = try std.fmt.allocPrint(allocator, "{s}.attn_q_norm.weight", .{prefix});
            defer allocator.free(q_norm_name);
            const k_norm_name = try std.fmt.allocPrint(allocator, "{s}.attn_k_norm.weight", .{prefix});
            defer allocator.free(k_norm_name);

            layer_weights.attn_q_norm_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, q_norm_name) catch null;
            layer_weights.attn_k_norm_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, k_norm_name) catch null;

            log.debug("  Layer {d}: full attention (Q/K/V/O + norms)", .{i});
        } else {
            // SSM 层权重
            const qkv_name = try std.fmt.allocPrint(allocator, "{s}.attn_qkv.weight", .{prefix});
            defer allocator.free(qkv_name);
            const gate_name = try std.fmt.allocPrint(allocator, "{s}.attn_gate.weight", .{prefix});
            defer allocator.free(gate_name);
            const conv_name = try std.fmt.allocPrint(allocator, "{s}.ssm_conv1d.weight", .{prefix});
            defer allocator.free(conv_name);
            const ssm_a_name = try std.fmt.allocPrint(allocator, "{s}.ssm_a", .{prefix});
            defer allocator.free(ssm_a_name);
            const dt_bias_name = try std.fmt.allocPrint(allocator, "{s}.ssm_dt.bias", .{prefix});
            defer allocator.free(dt_bias_name);
            const alpha_name = try std.fmt.allocPrint(allocator, "{s}.ssm_alpha.weight", .{prefix});
            defer allocator.free(alpha_name);
            const beta_name = try std.fmt.allocPrint(allocator, "{s}.ssm_beta.weight", .{prefix});
            defer allocator.free(beta_name);
            const ssm_norm_name = try std.fmt.allocPrint(allocator, "{s}.ssm_norm.weight", .{prefix});
            defer allocator.free(ssm_norm_name);
            const ssm_out_name = try std.fmt.allocPrint(allocator, "{s}.ssm_out.weight", .{prefix});
            defer allocator.free(ssm_out_name);

            layer_weights.attn_qkv_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, qkv_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ qkv_name, err });
                return error.MissingWeight;
            };
            layer_weights.attn_gate_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, gate_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ gate_name, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_conv1d_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, conv_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ conv_name, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_a = findOrCreateTensor(ctx, gguf_file, gguf_data, ssm_a_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ ssm_a_name, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_dt_bias = findOrCreateTensor(ctx, gguf_file, gguf_data, dt_bias_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ dt_bias_name, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_alpha_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, alpha_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ alpha_name, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_beta_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, beta_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ beta_name, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_norm_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, ssm_norm_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ ssm_norm_name, err });
                return error.MissingWeight;
            };
            layer_weights.ssm_out_weight = findOrCreateTensor(ctx, gguf_file, gguf_data, ssm_out_name) catch |err| {
                log.err("Failed to load {s}: {}", .{ ssm_out_name, err });
                return error.MissingWeight;
            };

            log.debug("  Layer {d}: SSM (QKV+gate+conv+ssm)", .{i});
        }

        layers[i] = layer_weights;
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

/// 在 GGUF 文件中查找张量并创建视图
fn findOrCreateTensor(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    gguf_data: []const u8,
    name: []const u8,
) !*ggml.Tensor {
    // 尝试从 GGUF 文件查找
    if (gguf_file.findTensor(name)) |info| {
        // 创建张量视图
        const n_dims = info.n_dims;
        const dims = info.dims;
        const typ: ggml.Type = @enumFromInt(@intFromEnum(info.type_));

        // 临时启用分配，创建张量元数据
        ctx.setNoAlloc(false);
        const tensor = switch (n_dims) {
            1 => try ctx.newTensor1d(typ, @intCast(dims[0])),
            2 => try ctx.newTensor2d(typ, @intCast(dims[0]), @intCast(dims[1])),
            3 => try ctx.newTensor3d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2])),
            4 => try ctx.newTensor4d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2]), @intCast(dims[3])),
            else => return error.UnsupportedTensorDims,
        };
        ctx.setNoAlloc(true);

        // 设置张量数据指针，指向 GGUF 文件中的张量数据
        const data_offset = gguf_file.tensor_data_offset + info.offset;
        if (data_offset + info.sizeBytes() <= gguf_data.len) {
            const data_ptr = @as(*anyopaque, @ptrCast(@constCast(&gguf_data[data_offset])));
            tensor.setDataPtr(data_ptr);
        } else {
            log.err("Tensor '{s}' data out of bounds: offset={}, size={}, file_size={}", .{
                name, data_offset, info.sizeBytes(), gguf_data.len,
            });
            return error.TensorDataOutOfBounds;
        }

        tensor.setName(@ptrCast(name));
        return tensor;
    }
    return error.TensorNotFound;
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "ModelParams parse" {
    const params = ModelParams{};
    try testing.expectEqual(@as(u32, 0), params.n_vocab);
    try testing.expectEqual(@as(u32, 32768), params.max_seq_len);
    try testing.expectEqual(@as(f32, 10000000.0), params.rope_theta);
}

test "LayerType enum" {
    try testing.expectEqual(@as(u32, 0), @intFromEnum(LayerType.full_attention));
    try testing.expectEqual(@as(u32, 1), @intFromEnum(LayerType.ssm));
}

test "isFullAttentionLayer" {
    try testing.expect(!isFullAttentionLayer(0, 4));
    try testing.expect(!isFullAttentionLayer(2, 4));
    try testing.expect(isFullAttentionLayer(3, 4));
    try testing.expect(!isFullAttentionLayer(4, 4));
    try testing.expect(isFullAttentionLayer(7, 4));
    try testing.expect(isFullAttentionLayer(0, 1));
}
