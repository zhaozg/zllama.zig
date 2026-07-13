//! Gemma 3 模型实现
//!
//! Gemma 3 是标准 Transformer 解码器，支持：
//! - GeGLU FFN（GELU 激活函数）
//! - Q/K pre-norm（RoPE 前的 RMSNorm）
//! - Attention post-norm + FFN post-norm
//! - 可选的 SWA（滑动窗口注意力）
//! - 最终 logit softcapping
//!
//! 参考 llama.cpp gemma3.cpp 实现。

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

const log = std.log.scoped(.model_gemma3);

// ============================================================================
// Gemma 3 超参数
// ============================================================================

pub const Gemma3Params = struct {
    base: model.ModelParams = .{},

    /// 最终 logit softcapping 阈值（0 表示禁用）
    final_logit_softcapping: f32 = 0.0,
    /// SWA 层滑动窗口大小（0 表示全注意力）
    n_swa: u32 = 0,
    /// SWA 模式周期
    swa_period: u32 = 1,
    /// SWA 的 RoPE base frequency
    rope_freq_base_swa: f32 = 10000.0,
    /// RoPE frequency scale (= 1.0 / rope_scaling_factor)
    rope_freq_scale: f32 = 1.0,
    /// 注意力缩放因子
    f_attention_scale: f32 = 1.0,
    /// 是否使用 SWA
    use_swa: bool = false,
};
// ============================================================================
// Gemma 3 层权重
// ============================================================================

pub const LayerWeights = struct {
    prefix: []const u8,

    // 归一化
    attn_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor,

    // Q/K 预归一化
    attn_q_norm_weight: ?*ggml.Tensor,
    attn_k_norm_weight: ?*ggml.Tensor,

    // Post 归一化
    attn_post_norm_weight: ?*ggml.Tensor,
    ffn_post_norm_weight: ?*ggml.Tensor,

    // 注意力
    attn_q_weight: *ggml.Tensor,
    attn_k_weight: *ggml.Tensor,
    attn_v_weight: *ggml.Tensor,
    attn_output_weight: *ggml.Tensor,

    // FFN (GeGLU)
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,
};

pub const Gemma3Weights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,

    pub fn deinit(self: *Gemma3Weights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| {
            allocator.free(layer.prefix);
        }
        allocator.free(self.layers);
    }
};

// ============================================================================
// Gemma 3 模型
// ============================================================================

pub const Gemma3Model = struct {
    params: Gemma3Params,
    weights: Gemma3Weights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *Gemma3Model, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.params = try parseParams(gguf_file, allocator);

        self.ctx_weights = try ggml.Context.initNoAlloc(weight_loader.estimateMemSize(gguf_file));

        self.weights = try loadWeights(gguf_file, self.ctx_weights, &self.params, allocator);
    }

    pub fn deinit(self: *Gemma3Model, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn getParams(self: *const Gemma3Model) *const model.ModelParams {
        return &self.params.base;
    }

    pub fn getWeights(self: *const Gemma3Model) *const model.ModelWeights {
        return &self.weights.base;
    }

    pub fn forward(
        self: *Gemma3Model,
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
        const head_dim_k: i64 = if (p.base.n_head_dim_k > 0) @intCast(p.base.n_head_dim_k) else head_dim;
        const head_dim_v: i64 = if (p.base.n_head_dim_v > 0) @intCast(p.base.n_head_dim_v) else head_dim;
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.base.rope_dim);

        // Token 嵌入 + 缩放
        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur = ggml.scale(ctx, cur, @sqrt(@as(f32, @floatFromInt(p.base.n_embd))));
        cur.setName("inp_scaled");

        // 位置编码
        const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens), start_pos);

        for (w.layers, 0..) |*layer, i| {
            var name_buf: [128]u8 = undefined;

            // 确定该层的 RoPE base frequency
            // SWA 层可能使用不同的 freq_base
            const layer_is_swa = isSWALayer(p, i);
            const freq_base_l: f32 = if (layer_is_swa and p.use_swa)
                p.rope_freq_base_swa
            else
                p.base.rope_theta;
            // SWA layers use freq_scale=1.0, non-SWA layers use rope_freq_scale
            const freq_scale_l: f32 = if (layer_is_swa and p.use_swa)
                1.0
            else
                p.rope_freq_scale;
            const attn_scale = p.f_attention_scale;

            // --- Pre-attention RMSNorm ---
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

            // 重塑为 [head_dim_k/v, n_head/kv_head, n_tokens]
            // Gemma3 使用 key_length/value_length 而非标准 head_dim
            q = ggml.reshape3d(ctx, q, head_dim_k, n_head, n_tokens_i64);
            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);
            v = ggml.reshape3d(ctx, v, head_dim_v, n_kv_head, n_tokens_i64);

            // --- Q/K Pre-norm (Gemma 3 特性) ---
            // Q: [head_dim_k, n_head, n_tokens], weight: [head_dim_k]
            // 重塑为 2D 进行归一化
            q = ggml.reshape2d(ctx, q, head_dim_k, n_head * n_tokens_i64);
            q = ggml.rmsNorm(ctx, q, p.base.norm_eps);
            if (layer.attn_q_norm_weight) |q_norm| {
                q = ggml.mul(ctx, q, ggml.reshape2d(ctx, q_norm, head_dim_k, 1));
            }
            q = ggml.reshape3d(ctx, q, head_dim_k, n_head, n_tokens_i64);

            k = ggml.reshape2d(ctx, k, head_dim_k, n_kv_head * n_tokens_i64);
            k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
            if (layer.attn_k_norm_weight) |k_norm| {
                k = ggml.mul(ctx, k, ggml.reshape2d(ctx, k_norm, head_dim_k, 1));
            }
            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);

            // --- RoPE (使用 rope_dim，与 head_dim_k 可能不同) ---
            // Gemma3 uses NEOX-style RoPE (mode=2), same as LLaMA
            q = ggml.ropeExt(ctx, q, pos_tensor, null, @intCast(rope_dim), 2, 0, freq_base_l, freq_scale_l, 0.0, 1.0, 0.0, 0.0);
            k = ggml.ropeExt(ctx, k, pos_tensor, null, @intCast(rope_dim), 2, 0, freq_base_l, freq_scale_l, 0.0, 1.0, 0.0, 0.0);

            // 应用注意力缩放到 Q（在进入 attention 之前）
            q = ggml.scale(ctx, q, attn_scale);

            // --- KV Cache ---
            // 在 permute 之前存入 cache，保持 [head_dim, n_kv_head, n_tokens] 布局
            // attention 层内部会自动执行 permute(0,2,1,3)
            if (kv_cache_mgr) |cache| {
                cache.setKv(ctx, graph, i, k, v, @intCast(n_tokens_i64));
                k = cache.getKView(ctx, i);
                v = cache.getVView(ctx, i);
            }

            const cache_len: i64 = if (kv_cache_mgr) |cache|
                @as(i64, @intCast(cache.currentLen()))
            else
                n_tokens_i64;

            // --- 缩放点积注意力 ---
            var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
                .n_head = n_head,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim_k,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = 1.0, // 已经通过 ggml_scale 应用了
            }, if (layer_is_swa) @as(i64, @intCast(p.n_swa)) else null); // SWA mask for sliding window layers

            // 输出投影
            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);

            // --- Attention Post-norm (Gemma 3 特性) ---
            if (layer.attn_post_norm_weight) |post_norm| {
                attn_out = rms_norm.rmsNorm(ctx, attn_out, post_norm, p.base.norm_eps);
                const ap_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_post", .{i}) catch unreachable;
                name_buf[ap_name.len] = 0;
                attn_out.setName(name_buf[0..ap_name.len :0]);
            }

            // 残差连接（post-norm 后）
            cur = ggml.add(ctx, cur, attn_out);

            // --- FFN Pre-norm ---
            const ffn_input = rms_norm.rmsNorm(ctx, cur, layer.ffn_norm_weight, p.base.norm_eps);

            // --- GeGLU FFN ---
            const ffn_out = gegluFFN(ctx, ffn_input, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);

            // --- FFN Post-norm (Gemma 3 特性) ---
            const ffn_residual = if (layer.ffn_post_norm_weight) |ffn_post_norm| blk: {
                var ffn_normed = rms_norm.rmsNorm(ctx, ffn_out, ffn_post_norm, p.base.norm_eps);
                const fp_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_post", .{i}) catch unreachable;
                name_buf[fp_name.len] = 0;
                ffn_normed.setName(name_buf[0..fp_name.len :0]);
                break :blk ffn_normed;
            } else ffn_out;

            // 残差连接
            cur = ggml.add(ctx, cur, ffn_residual);
            const out_name = std.fmt.bufPrint(&name_buf, "blk.{d}.out", .{i}) catch unreachable;
            name_buf[out_name.len] = 0;
            cur.setName(name_buf[0..out_name.len :0]);
        }

        // --- 最终 RMSNorm ---
        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.base.norm_eps);
        cur.setName("output_norm");

        // --- 输出投影 ---
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);

        // --- Final logit softcapping (Gemma 3 特性) ---
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

    // 适配 buildGraph 接口
    pub fn buildGraph(
        self: *Gemma3Model,
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

    // 虚表
    pub const vtable = model.ModelVTable{
        .deinit = deinitAdapter,
        .buildGraph = buildGraphAdapter,
        .getParams = getParamsAdapter,
        .resetSSMStates = resetSSMStatesAdapter,
    };

    fn deinitAdapter(data: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Gemma3Model = @ptrCast(@alignCast(data));
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
        const self: *Gemma3Model = @ptrCast(@alignCast(data));
        return self.buildGraph(builder, input_tokens, n_tokens, mem_ctx, start_pos);
    }

    fn getParamsAdapter(data: *anyopaque) *const model.ModelParams {
        const self: *Gemma3Model = @ptrCast(@alignCast(data));
        return self.getParams();
    }

    fn resetSSMStatesAdapter(data: *anyopaque) void {
        _ = data;
    }
};

// ============================================================================
// GeGLU FFN（Gemma 使用 GELU 而非 SiLU）
// ============================================================================

/// GeGLU FFN: gate(h) * up(h) -> down
/// 输入 x: [n_embd, n_tokens]
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

// ============================================================================
// 参数解析
// ============================================================================

fn isSWALayer(p: *const Gemma3Params, layer_idx: usize) bool {
    if (!p.use_swa) return false;
    if (p.swa_period == 0) return false;
    return (layer_idx % p.swa_period) != 0;
}

pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !Gemma3Params {
    var p = Gemma3Params{};

    p.base.n_vocab = gguf_file.getU32("gemma3.vocab_size") orelse
        gguf_file.getU32("llama.vocab_size") orelse
        blk: {
            if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
                if (val.value_type == .array) break :blk @intCast(val.array_val.len);
            }
            break :blk 0;
        };
    p.base.n_embd = gguf_file.getU32("gemma3.embedding_length") orelse
        gguf_file.getU32("llama.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("gemma3.attention.head_count") orelse
        gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("gemma3.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse 0;
    p.base.n_kv_head = gguf_file.getU32("gemma3.attention.head_count_kv") orelse
        gguf_file.getU32("llama.attention.head_count_kv") orelse
        gguf_file.getU32("gemma3.head_count_kv") orelse
        gguf_file.getU32("llama.head_count_kv") orelse p.base.n_head;
    p.base.n_layer = gguf_file.getU32("gemma3.block_count") orelse
        gguf_file.getU32("llama.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("gemma3.feed_forward_length") orelse
        gguf_file.getU32("llama.feed_forward_length") orelse 0;

    if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }

    // Gemma3 特有的 head dim（key_length / value_length）
    // 必须在 rope_dim 之前读取，因为 rope_dim 的回退逻辑依赖 n_head_dim_k
    p.base.n_head_dim_k = gguf_file.getU32("gemma3.attention.key_length") orelse
        gguf_file.getU32("gemma3.attention.head_dim_k") orelse p.base.n_head_dim;
    p.base.n_head_dim_v = gguf_file.getU32("gemma3.attention.value_length") orelse
        gguf_file.getU32("gemma3.attention.head_dim_v") orelse p.base.n_head_dim;

    p.base.max_seq_len = gguf_file.getU32("gemma3.context_length") orelse
        gguf_file.getU32("llama.context_length") orelse 4096;
    p.base.rope_theta = gguf_file.getF32("gemma3.rope.freq_base") orelse
        gguf_file.getF32("llama.rope.freq_base") orelse 10000.0;
    // RoPE dimension: Gemma 3 applies RoPE to all key/query dimensions (head_dim_k)
    // Standard LLaMA models use llama.rope.dimension_count or full head_dim
    p.base.rope_dim = gguf_file.getU32("gemma3.rope.dimension_count") orelse
        gguf_file.getU32("llama.rope.dimension_count") orelse
        if (p.base.n_head_dim_k > 0 and p.base.n_head_dim_k != p.base.n_head_dim)
            p.base.n_head_dim_k // Gemma 3 style: RoPE on full key length
        else
            p.base.n_head_dim; // Standard: RoPE on full head dim
    p.base.norm_eps = gguf_file.getF32("gemma3.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-6;
    p.base.model_name = gguf_file.getString("general.name") orelse "";
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gemma";

    // Gemma 3 特定参数
    // 尝试从 gemma3. 前缀读取
    const n_swa = gguf_file.getU32("gemma3.attention.sliding_window") orelse 0;
    if (n_swa > 0) {
        p.n_swa = n_swa;
        p.use_swa = true;
        p.swa_period = gguf_file.getU32("gemma3.attention.sliding_window_pattern") orelse 6;
        // SWA freq_base: GGUF override or Gemma default 10000.0
        p.rope_freq_base_swa = gguf_file.getF32("gemma3.rope.freq_base_swa") orelse 10000.0;
    }

    p.final_logit_softcapping = gguf_file.getF32("gemma3.final_logit_softcapping") orelse 0.0;

    // RoPE frequency scale = 1.0 / rope_scaling_factor (for linear scaling)
    const rope_scaling_factor = gguf_file.getF32("gemma3.rope.scaling.factor") orelse 1.0;
    if (rope_scaling_factor != 1.0) {
        p.rope_freq_scale = 1.0 / rope_scaling_factor;
    }

    // 注意力缩放因子
    // Gemma 3 使用 key_length (head_dim_k) 而非 n_embd/n_head 计算缩放
    // 参考: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py
    p.f_attention_scale = if (p.base.n_head_dim_k > 0)
        1.0 / @sqrt(@as(f32, @floatFromInt(p.base.n_head_dim_k)))
    else if (p.base.n_head_dim > 0)
        1.0 / @sqrt(@as(f32, @floatFromInt(p.base.n_head_dim)))
    else
        1.0;

    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required Gemma 3 parameters", .{});
        return error.InvalidModelParams;
    }

    log.info("Gemma3: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}, swa={d}, softcap={d}", .{
        p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head,
        p.base.n_layer, p.base.n_ff,   p.n_swa,       p.final_logit_softcapping,
    });
    log.info("Gemma3: head_dim={d}, head_dim_k={d}, head_dim_v={d}, rope_dim={d}", .{
        p.base.n_head_dim, p.base.n_head_dim_k, p.base.n_head_dim_v, p.base.rope_dim,
    });
    log.info("Gemma3: rope_theta={d:.1}, freq_scale={d:.4}, freq_base_swa={d:.1}, attn_scale={d:.4}", .{
        p.base.rope_theta, p.rope_freq_scale, p.rope_freq_base_swa, p.f_attention_scale,
    });

    return p;
}

// ============================================================================
// 权重加载
// ============================================================================

fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *const Gemma3Params,
    allocator: std.mem.Allocator,
) !Gemma3Weights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Gemma 3 weights...", .{});

    const token_embd = try weight_loader.findOrCreateTensor(ctx, gguf_file, "token_embd.weight");
    token_embd.setName("token_embd.weight");

    const output_weight = weight_loader.findOrCreateTensor(ctx, gguf_file, "output.weight") catch null;
    if (output_weight) |ow| ow.setName("output.weight");

    const output_norm_weight = try weight_loader.findOrCreateTensor(ctx, gguf_file, "output_norm.weight");
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

        // 辅助函数，加载层权重，缺失权重非致命（可选权重）
        const loadOpt = struct {
            fn load(ctx2: *ggml.Context, gf: *const gguf.GGUFFile, pfx: []const u8, name: []const u8, _: usize) ?*ggml.Tensor {
                return weight_loader.loadLayerWeight(ctx2, gf, pfx, name) catch return null;
            }
        }.load;

        layers[i] = LayerWeights{
            .prefix = prefix,
            .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
            .ffn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight"),
            .attn_q_norm_weight = loadOpt(ctx, gguf_file, prefix, "attn_q_norm.weight", i) orelse blk: {
                log.warn("Layer {d}: missing attn_q_norm.weight, skipping Q norm", .{i});
                break :blk null;
            },
            .attn_k_norm_weight = loadOpt(ctx, gguf_file, prefix, "attn_k_norm.weight", i) orelse blk: {
                log.warn("Layer {d}: missing attn_k_norm.weight", .{i});
                break :blk null;
            },
            .attn_post_norm_weight = loadOpt(ctx, gguf_file, prefix, "post_attention_norm.weight", i) orelse blk: {
                log.warn("Layer {d}: missing post_attention_norm.weight", .{i});
                break :blk null;
            },
            .ffn_post_norm_weight = loadOpt(ctx, gguf_file, prefix, "post_ffw_norm.weight", i) orelse blk: {
                log.warn("Layer {d}: missing post_ffw_norm.weight", .{i});
                break :blk null;
            },
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

    log.info("All Gemma 3 weights loaded ({d} layers)", .{n_layer});

    return Gemma3Weights{
        .base = .{
            .params = params.base,
            .token_embd = token_embd,
            .output_weight = output_weight,
            .output_norm_weight = output_norm_weight,
        },
        .layers = layers,
    };
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "Gemma3Params defaults" {
    const p = Gemma3Params{};
    try testing.expectEqual(@as(u32, 0), p.base.n_vocab);
    try testing.expectEqual(@as(f32, 0.0), p.final_logit_softcapping);
}
