//! Qwen35 混合架构模型实现（全注意力 + SSM/GDN 线性注意力交替）
//!
//! 参考: deps/llama.cpp/src/models/qwen35.cpp, deps/llama.cpp/src/models/delta-net-base.cpp

const std = @import("std");
const gguf = @import("gguf");
const ggml = @import("ggml");
const kv_cache = @import("kv_cache");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const swiglu = @import("swiglu");
const attention = @import("attention");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const embed = @import("embed");

const model = @import("../model.zig");

const log = std.log.scoped(.qwen35);

pub const QwenParams = struct {
    base: model.ModelParams = .{},
    full_attention_interval: u32 = 4,
    ssm_conv_kernel: u32 = 4,
    ssm_state_size: u32 = 128,
    ssm_group_count: u32 = 16,
    ssm_time_step_rank: u32 = 16,
    ssm_inner_size: u32 = 2048,
    attn_key_length: u32 = 0,
    attn_value_length: u32 = 0,
    rope_scaling: ?model.RopeScaling = null,
};

pub const LayerType = enum(u32) {
    full_attention = 0,
    ssm = 1,
    _,
};

pub const LayerWeights = struct {
    prefix: []const u8,
    layer_type: LayerType,
    attn_norm_weight: *ggml.Tensor,
    post_attention_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor,
    attn_q_weight: ?*ggml.Tensor = null,
    attn_k_weight: ?*ggml.Tensor = null,
    attn_v_weight: ?*ggml.Tensor = null,
    attn_output_weight: ?*ggml.Tensor = null,
    attn_q_norm_weight: ?*ggml.Tensor = null,
    attn_k_norm_weight: ?*ggml.Tensor = null,
    attn_qkv_weight: ?*ggml.Tensor = null,
    attn_gate_weight: ?*ggml.Tensor = null,
    ssm_conv1d_weight: ?*ggml.Tensor = null,
    ssm_a: ?*ggml.Tensor = null,
    ssm_dt_bias: ?*ggml.Tensor = null,
    ssm_alpha_weight: ?*ggml.Tensor = null,
    ssm_beta_weight: ?*ggml.Tensor = null,
    ssm_norm_weight: ?*ggml.Tensor = null,
    ssm_out_weight: ?*ggml.Tensor = null,
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,
};

pub const QwenWeights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,
    pub fn deinit(self: *QwenWeights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| allocator.free(layer.prefix);
        allocator.free(self.layers);
    }
};

/// 每层的 SSM 状态（conv_state 和 ssm_state）
const LayerSSMState = struct {
    conv_state: ?*ggml.Tensor, // [d_conv-1, conv_dim]
    ssm_state: ?*ggml.Tensor,  // [S_v*S_v*H, 1, n_seqs]
};

pub const QwenModel = struct {
    qwen_params: QwenParams,
    qwen_weights: QwenWeights,
    ctx_weights: *ggml.Context,
    ctx_kv_cache: ?*ggml.Context = null, // 用于分配持久状态的 context
    ssm_states: []LayerSSMState, // 每层的 SSM 状态

    pub fn init(self: *QwenModel, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.qwen_params = try parseParams(gguf_file, allocator);
        self.ctx_weights = try ggml.Context.initNoAlloc(estimateMemSize(&self.qwen_params));
        self.qwen_weights = try loadWeights(gguf_file, self.ctx_weights, &self.qwen_params, allocator);

        // 初始化 SSM 状态（在推理上下文中分配）
        const n_layer = self.qwen_params.base.n_layer;
        self.ssm_states = try allocator.alloc(LayerSSMState, n_layer);
        // 初始化为空，在 forward 中按需创建
        for (0..n_layer) |i| {
            self.ssm_states[i] = .{
                .conv_state = null,
                .ssm_state = null,
            };
        }
    }

    pub fn deinit(self: *QwenModel, allocator: std.mem.Allocator) void {
        self.qwen_weights.deinit(allocator);
        self.ctx_weights.deinit();
        allocator.free(self.ssm_states);
    }

    pub fn getParams(self: *const QwenModel) *const model.ModelParams {
        return &self.qwen_params.base;
    }
    pub fn getWeights(self: *const QwenModel) *const model.ModelWeights {
        return &self.qwen_weights.base;
    }
    pub fn resetSSMStates(self: *QwenModel) void {
        for (self.ssm_states) |*state| {
            state.conv_state = null;
            state.ssm_state = null;
        }
    }



    pub fn forward(self: *QwenModel, ctx: *ggml.Context, graph: *ggml.CGraph, input_tokens: *ggml.Tensor, n_tokens: i32, kv_cache_mgr: ?*kv_cache.KVCache, start_pos: i32) !*ggml.Tensor {
        const p = &self.qwen_params;
        const w = &self.qwen_weights;
        const n_head: i64 = @intCast(p.base.n_head);
        const n_kv_head: i64 = @intCast(p.base.n_kv_head);
        const head_dim: i64 = @intCast(p.base.n_head_dim);
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.base.rope_dim);

        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur.setName("token_embd");

        for (w.layers, 0..) |*layer, i| {
            const layer_idx: u32 = @intCast(i);
            const is_full_attn = isFullAttentionLayer(layer_idx, p.full_attention_interval);
            var name_buf: [128]u8 = undefined;

            // Pre-attention norm
            var attn_input = rms_norm.rmsNorm(ctx, cur, layer.attn_norm_weight, p.base.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_input.setName(name_buf[0..name.len :0]);
            }

            if (is_full_attn) {
                const attn_out = try self.forwardFullAttention(ctx, graph, layer, attn_input, n_tokens_i64, n_head, n_kv_head, head_dim, rope_dim, kv_cache_mgr, start_pos, i, &name_buf);
                cur = ggml.add(ctx, cur, attn_out);
            } else {
                const ssm_out = try self.forwardSSM(ctx, graph, layer, attn_input, n_tokens_i64, i);
                cur = ggml.add(ctx, cur, ssm_out);
            }

            // Post-attention norm (Qwen35 uses this instead of ffn_norm)
            var post_attn = rms_norm.rmsNorm(ctx, cur, layer.post_attention_norm_weight, p.base.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                post_attn.setName(name_buf[0..name.len :0]);
            }

            // SwiGLU FFN
            const ffn_out = swiglu.swiGLU(ctx, post_attn, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_out.setName(name_buf[0..name.len :0]);
            }
            cur = ggml.add(ctx, cur, ffn_out);
        }

        // Output norm & projection
        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.base.norm_eps);
        cur.setName("output_norm");
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);
        logits_tensor.setName("logits");
        graph.buildForwardExpand(logits_tensor);
        return logits_tensor;
    }

    fn forwardFullAttention(self: *QwenModel, ctx: *ggml.Context, graph: *ggml.CGraph, layer: *const LayerWeights, attn_input: *ggml.Tensor, n_tokens_i64: i64, n_head: i64, n_kv_head: i64, head_dim: i64, rope_dim: i64, kv_cache_mgr: ?*kv_cache.KVCache, start_pos: i32, layer_idx: usize, name_buf: *[128]u8) !*ggml.Tensor {
        const p = &self.qwen_params;
        const q_w = layer.attn_q_weight orelse return error.MissingWeight;
        const k_w = layer.attn_k_weight orelse return error.MissingWeight;
        const v_w = layer.attn_v_weight orelse return error.MissingWeight;
        const o_w = layer.attn_output_weight orelse return error.MissingWeight;

        // Q projection outputs [Q, gate] combined: [n_embd_head*2*n_head, n_tokens]
        // Qwen3.5: Q and gate are interleaved: Q[0], gate[0], Q[1], gate[1], ...
        const q_full = ggml.mulMat(ctx, q_w, attn_input);
        var k = ggml.mulMat(ctx, k_w, attn_input);
        var v = ggml.mulMat(ctx, v_w, attn_input);

        // View Q as 3D with interleaved stride: [head_dim, n_head, n_tokens] with stride = head_dim * 2
        var q = ctx.view3d(q_full, head_dim, n_head, n_tokens_i64,
            @as(usize, @intCast(head_dim * 2 * @sizeOf(f32))),
            @as(usize, @intCast(head_dim * 2 * n_head * @sizeOf(f32))),
            0);
        // Gate is the same view but with offset = head_dim
        const q_gate_view = ctx.view3d(q_full, head_dim, n_head, n_tokens_i64,
            @as(usize, @intCast(head_dim * 2 * @sizeOf(f32))),
            @as(usize, @intCast(head_dim * 2 * n_head * @sizeOf(f32))),
            @as(usize, @intCast(head_dim * @sizeOf(f32))));

        // Make Q contiguous for norm
        q = ggml.cont(ctx, q);

        // Q norm: apply RMS norm + Q norm weight to Q only (not gate)
        if (layer.attn_q_norm_weight) |q_norm_raw| {
            q = ggml.rmsNorm(ctx, q, p.base.norm_eps);
            // q_norm_raw: [head_dim] -> reshape to [head_dim, 1, 1] for broadcasting
            const q_norm_3d = ggml.reshape3d(ctx, q_norm_raw, head_dim, 1, 1);
            const q_norm_target = ctx.newTensor3d(.f32, head_dim, n_head, n_tokens_i64) catch unreachable;
            const q_norm_rep = ggml.repeat(ctx, q_norm_3d, q_norm_target);
            q = ggml.mul(ctx, q, q_norm_rep);
        }

        // K norm: reshape K to 3D first, then apply norm (llama.cpp style)
        k = ggml.reshape3d(ctx, k, head_dim, n_kv_head, n_tokens_i64);
        if (layer.attn_k_norm_weight) |k_norm_raw| {
            k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
            const k_norm_3d = ggml.reshape3d(ctx, k_norm_raw, head_dim, 1, 1);
            const k_norm_target = ctx.newTensor3d(.f32, head_dim, n_kv_head, n_tokens_i64) catch unreachable;
            const k_norm_rep = ggml.repeat(ctx, k_norm_3d, k_norm_target);
            k = ggml.mul(ctx, k, k_norm_rep);
        }

        // V reshape
        v = ggml.reshape3d(ctx, v, head_dim, n_kv_head, n_tokens_i64);

        // MRoPE (Multi-dimensional RoPE)
        const pos_tensor = rope.buildMultiPositionTensor(ctx, @intCast(n_tokens_i64), start_pos);
        const rope_sections = [4]i32{ 11, 11, 10, 0 };
        const rope_type: i32 = 40; // GGML_ROPE_TYPE_IMROPE for Qwen 3.5
        q = ggml.ropeMulti(ctx, q, pos_tensor, @intCast(rope_dim), &rope_sections, rope_type, 0, p.base.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        k = ggml.ropeMulti(ctx, k, pos_tensor, @intCast(rope_dim), &rope_sections, rope_type, 0, p.base.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

        // KV Cache: K/V 形状为 [head_dim, n_kv_head, n_tokens]
        // Cache 布局: [head_dim, n_kv_head, max_seq_len]
        // 直接存储，不需要 permute（与 cache 布局一致）
        if (kv_cache_mgr) |cache| {
            cache.setKv(ctx, graph, layer_idx, k, v, @intCast(n_tokens_i64));
            k = cache.getKView(ctx, layer_idx);
            v = cache.getVView(ctx, layer_idx);
        }

        const cache_len: i64 = if (kv_cache_mgr) |cache| @as(i64, @intCast(cache.currentLen())) else n_tokens_i64;

        // Scaled dot-product attention
        // attention.scaledDotProductAttention 期望输入:
        //   q: [head_dim, n_head, n_tokens]
        //   k: [head_dim, n_kv_head, cache_len]
        //   v: [head_dim, n_kv_head, cache_len]
        var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
            .n_head = n_head,
            .n_kv_head = n_kv_head,
            .head_dim = head_dim,
            .n_tokens = n_tokens_i64,
            .cache_len = cache_len,
            .start_pos = start_pos,
            .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        });

        // Gate: sigmoid(gate) * attn_out
        // Ensure gate is contiguous before sigmoid
        // gate_view: [head_dim, n_head, n_tokens] -> reshape to [n_head*head_dim, n_tokens]
        const gate_cont = ggml.cont(ctx, q_gate_view);
        const gate_2d = ggml.reshape2d(ctx, gate_cont, n_head * head_dim, n_tokens_i64);
        const gate_sigmoid = ggml.sigmoid(ctx, gate_2d);
        attn_out = ggml.mul(ctx, attn_out, gate_sigmoid);
        {
            const name = std.fmt.bufPrint(name_buf, "blk.{d}.attn_gated", .{layer_idx}) catch unreachable;
            name_buf[name.len] = 0;
            attn_out.setName(name_buf[0..name.len :0]);
        }

        // Output projection
        var result = ggml.mulMat(ctx, o_w, attn_out);
        {
            const name = std.fmt.bufPrint(name_buf, "blk.{d}.attn_out", .{layer_idx}) catch unreachable;
            name_buf[name.len] = 0;
            result.setName(name_buf[0..name.len :0]);
        }
        return result;
    }

    fn forwardSSM(self: *QwenModel, ctx: *ggml.Context, graph: *ggml.CGraph, layer: *const LayerWeights, attn_input: *ggml.Tensor, n_tokens_i64: i64, layer_idx: usize) !*ggml.Tensor {
        const p = &self.qwen_params;
        const d_inner: i64 = @intCast(p.ssm_inner_size);
        const d_state: i64 = @intCast(p.ssm_state_size);
        const d_conv: i64 = @intCast(p.ssm_conv_kernel);
        const n_group: i64 = @intCast(p.ssm_group_count);
        const dt_rank: i64 = @intCast(p.ssm_time_step_rank);
        const n_seqs: i64 = 1;

        const head_k_dim = d_state;
        const head_v_dim = @divExact(d_inner, dt_rank);
        const num_k_heads = n_group;
        const num_v_heads = dt_rank;
        const key_dim = head_k_dim * num_k_heads;
        const value_dim = head_v_dim * num_v_heads;
        const conv_dim = key_dim * 2 + value_dim;

        // === Step 1: Input projections (QKV mixed + Z gate) ===
        const qkv_mixed = ggml.mulMat(ctx, layer.attn_qkv_weight.?, attn_input);
        qkv_mixed.setName("ssm_qkv_mixed");
        const z = ggml.mulMat(ctx, layer.attn_gate_weight.?, attn_input);
        z.setName("ssm_z");

        // === Step 2: Beta (sigmoid) ===
        var beta = ggml.mulMat(ctx, layer.ssm_beta_weight.?, attn_input);
        beta = ggml.reshape4d(ctx, beta, 1, num_v_heads, n_tokens_i64, n_seqs);
        beta = ggml.sigmoid(ctx, beta);
        beta.setName("ssm_beta");

        // === Step 3: Alpha -> gate (softplus(alpha + dt_bias) * A) ===
        var alpha = ggml.mulMat(ctx, layer.ssm_alpha_weight.?, attn_input);
        alpha = ggml.reshape3d(ctx, alpha, num_v_heads, n_tokens_i64, n_seqs);
        alpha.setName("ssm_alpha");

        const alpha_biased = ggml.add(ctx, alpha, layer.ssm_dt_bias.?);
        var alpha_softplus = ggml.softplus(ctx, alpha_biased);
        alpha_softplus.setName("ssm_alpha_softplus");

        var gate = ggml.mul(ctx, alpha_softplus, layer.ssm_a.?);
        gate = ggml.reshape4d(ctx, gate, 1, num_v_heads, n_tokens_i64, n_seqs);
        gate.setName("ssm_gate");

        // === Step 4: Conv1d processing ===
        // Build conv input: [conv_state | qkv_mixed]
        // ggml_ssm_conv expects sx: 3D [d_conv-1+n_t, d_inner, n_s]
        const qkv_2d = ggml.reshape2d(ctx, qkv_mixed, conv_dim, n_tokens_i64);
        const qkv_transposed = ggml.cont(ctx, ggml.permute(ctx, qkv_2d, 1, 0, 2, 3));

        // Get or create conv state for this layer
        if (self.ssm_states[layer_idx].conv_state == null) {
            if (self.ctx_kv_cache) |kv_ctx| {
                kv_ctx.setNoAlloc(false);
                self.ssm_states[layer_idx].conv_state = try kv_ctx.newTensor2d(.f32, d_conv - 1, conv_dim);
                kv_ctx.setNoAlloc(true);
            } else {
                ctx.setNoAlloc(false);
                self.ssm_states[layer_idx].conv_state = try ctx.newTensor2d(.f32, d_conv - 1, conv_dim);
                ctx.setNoAlloc(true);
            }
            self.ssm_states[layer_idx].conv_state.?.setZero();
            {
                var buf: [64]u8 = undefined;
                const slice = try std.fmt.bufPrint(&buf, "ssm_conv_state.{d}", .{layer_idx});
                buf[slice.len] = 0;
                self.ssm_states[layer_idx].conv_state.?.setName(buf[0..slice.len :0]);
            }
        }

        const conv_state = self.ssm_states[layer_idx].conv_state.?;

        // Concat conv_state and qkv_mixed along dim 0
        const concat_result = ggml.concat(ctx, conv_state, qkv_transposed, 0);
        const concat_cont = ggml.cont(ctx, concat_result);

        // View as 3D for ggml_ssm_conv: [d_conv-1+n_t, conv_dim, 1]
        const sx_3d = ctx.view3d(concat_cont, d_conv - 1 + n_tokens_i64, conv_dim, 1,
            @as(usize, @intCast((d_conv - 1 + n_tokens_i64) * @sizeOf(f32))),
            @as(usize, @intCast((d_conv - 1 + n_tokens_i64) * @sizeOf(f32) * conv_dim)),
            0);
        sx_3d.setName("ssm_conv_sx");

        // Apply conv1d
        var conv_output = ggml.ssmConv(ctx, sx_3d, layer.ssm_conv1d_weight.?);
        conv_output.setName("ssm_conv_output");
        var conv_silu = ggml.silu(ctx, conv_output);
        conv_silu.setName("ssm_conv_silu");
        // Update conv state: extract last (d_conv-1) rows from concat for next iteration
        // concat_cont: [d_conv-1+n_tokens, conv_dim], last d_conv-1 rows become new conv_state
        const row_stride = @as(usize, @intCast(@as(usize, @intCast(d_conv - 1 + n_tokens_i64)) * @sizeOf(f32)));
        const conv_state_update = ctx.view2d(concat_cont, d_conv - 1, conv_dim,
            row_stride,
            @as(usize, @intCast(@as(usize, @intCast(n_tokens_i64)) * @sizeOf(f32))));
        const conv_state_cpy = ggml.cpy(ctx, conv_state_update, conv_state);
        graph.buildForwardExpand(conv_state_cpy);

        // === Step 5: Extract Q, K, V from conv output ===
        // conv_silu: [conv_dim, n_tokens, n_seqs] (after ggml_ssm_conv)
        // Layout: [Q | K | V] along dim 0
        const qkv_stride = conv_silu.strides()[1];
        var q_conv = ctx.view2d(conv_silu, head_k_dim * num_k_heads, n_tokens_i64, qkv_stride, 0);
        var k_conv = ctx.view2d(conv_silu, head_k_dim * num_k_heads, n_tokens_i64, qkv_stride,
            @intCast(head_k_dim * num_k_heads * @sizeOf(f32)));
        var v_conv = ctx.view2d(conv_silu, head_v_dim * num_v_heads, n_tokens_i64, qkv_stride,
            @intCast(2 * head_k_dim * num_k_heads * @sizeOf(f32)));
        // Reshape to 4D: [head_dim, n_heads, n_tokens, n_seqs]
        // Need contiguous for reshape
        q_conv = ggml.cont(ctx, q_conv);
        k_conv = ggml.cont(ctx, k_conv);
        v_conv = ggml.cont(ctx, v_conv);
        q_conv = ggml.reshape4d(ctx, q_conv, head_k_dim, num_k_heads, n_tokens_i64, n_seqs);
        k_conv = ggml.reshape4d(ctx, k_conv, head_k_dim, num_k_heads, n_tokens_i64, n_seqs);
        v_conv = ggml.reshape4d(ctx, v_conv, head_v_dim, num_v_heads, n_tokens_i64, n_seqs);

        // L2 normalize Q and K
        q_conv = ggml.l2Norm(ctx, q_conv, p.base.norm_eps);
        k_conv = ggml.l2Norm(ctx, k_conv, p.base.norm_eps);
        q_conv.setName("ssm_q_norm");
        k_conv.setName("ssm_k_norm");

        // Repeat heads if num_k_heads != num_v_heads
        if (num_k_heads != num_v_heads) {
            q_conv = ggml.repeat4d(ctx, q_conv, head_k_dim, num_v_heads, n_tokens_i64, n_seqs);
            k_conv = ggml.repeat4d(ctx, k_conv, head_k_dim, num_v_heads, n_tokens_i64, n_seqs);
        }

        // === Step 6: Gated Delta Net ===
        // Get or create SSM state for this layer
        if (self.ssm_states[layer_idx].ssm_state == null) {
            const alloc_ctx = if (self.ctx_kv_cache) |kv_ctx| kv_ctx else ctx;
            alloc_ctx.setNoAlloc(false);
            // state: [S_v*S_v*H, K, n_seqs] where K=1 for autoregressive
            const state_size = head_v_dim * head_v_dim * num_v_heads;
            self.ssm_states[layer_idx].ssm_state = try alloc_ctx.newTensor3d(.f32, state_size, 1, n_seqs);
            alloc_ctx.setNoAlloc(true);
            self.ssm_states[layer_idx].ssm_state.?.setZero();
            {
                var buf: [64]u8 = undefined;
                const slice = try std.fmt.bufPrint(&buf, "ssm_state.{d}", .{layer_idx});
                buf[slice.len] = 0;
                self.ssm_states[layer_idx].ssm_state.?.setName(buf[0..slice.len :0]);
            }
        }

        const ssm_state = self.ssm_states[layer_idx].ssm_state.?;

        // ggml_gatedDeltaNet expects state: [S_v*S_v*H, K, n_seqs]
        // Output: [S_v*H, n_tokens*n_seqs + state_rows, 1, 1]
        // where state_rows = K * S_v * n_seqs
        const gdn_output = ggml.gatedDeltaNet(ctx, q_conv, k_conv, v_conv, gate, beta, ssm_state);
        gdn_output.setName("ssm_gdn_output");

        // Extract attention output from gdn_output
        // gdn_output layout: [S_v*H, n_tokens*n_seqs + state_rows, 1, 1]
        // First S_v*H*n_tokens*n_seqs elements are the attention output
        // The data is laid out as [S_v, H, n_tokens, n_seqs] in column-major order:
        //   for each seq: for each token: for each head: S_v values
        // So the correct strides for a [S_v, H_v, n_tokens, n_seqs] view are:
        //   nb1 = S_v * sizeof(f32)  (stride between heads)
        //   nb2 = S_v * H_v * sizeof(f32)  (stride between tokens)
        //   nb3 = S_v * H_v * n_tokens * sizeof(f32)  (stride between sequences)
        const attn_output = ctx.view4d(gdn_output,
            head_v_dim, num_v_heads, n_tokens_i64, n_seqs,
            @as(usize, @intCast(head_v_dim * @sizeOf(f32))),
            @as(usize, @intCast(head_v_dim * num_v_heads * @sizeOf(f32))),
            @as(usize, @intCast(head_v_dim * num_v_heads * n_tokens_i64 * @sizeOf(f32))),
            0);
        attn_output.setName("ssm_attn_output");

        // Extract new state from gdn_output and copy to persistent state
        // new state starts at offset = S_v * H_v * n_tokens * n_seqs * sizeof(f32)
        const state_offset = @as(usize, @intCast(head_v_dim * num_v_heads * n_tokens_i64 * n_seqs * @sizeOf(f32)));
        // The state is laid out as [S_v*S_v, H_v, n_seqs] in the output:
        //   for each seq: for each head: S_v*S_v values
        // We view it as 3D [S_v*S_v*H_v, 1, n_seqs] to match the persistent state shape.
        // nb1 = S_v*S_v*H_v * sizeof(f32) (all heads for one seq)
        // nb2 = S_v*S_v*H_v * sizeof(f32) (same, since ne1=1)
        const new_state = ctx.view3d(gdn_output,
            head_v_dim * head_v_dim * num_v_heads, 1, n_seqs,
            @as(usize, @intCast(head_v_dim * head_v_dim * num_v_heads * @sizeOf(f32))),
            @as(usize, @intCast(head_v_dim * head_v_dim * num_v_heads * @sizeOf(f32))),
            state_offset);
        const state_cpy = ggml.cpy(ctx, new_state, ssm_state);
        graph.buildForwardExpand(state_cpy);

        // === Step 7: Gated normalization (norm(attn_out, z) ===
        const z_4d = ggml.reshape4d(ctx, z, head_v_dim, num_v_heads, n_tokens_i64, n_seqs);
        const attn_out_norm = ggml.rmsNorm(ctx, attn_output, p.base.norm_eps);
        const z_silu = ggml.silu(ctx, z_4d);
        const gated = ggml.mul(ctx, attn_out_norm, z_silu);
        gated.setName("ssm_gated_norm");

        // === Step 8: Output projection ===
        const final_output = ggml.reshape3d(ctx, gated, head_v_dim * num_v_heads, n_tokens_i64, n_seqs);
        var result = ggml.mulMat(ctx, layer.ssm_out_weight.?, final_output);
        result.setName("ssm_out");
        result = ggml.reshape2d(ctx, result, p.base.n_embd, n_tokens_i64);
        return result;
    }

    /// 适配 buildGraph 接口（通过 GraphBuilder 调用）
    pub fn buildGraph(
        self: *QwenModel,
        builder: *graph_builder.GraphBuilder,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        mem_ctx: ?*anyopaque,
        start_pos: i32,
    ) !*ggml.Tensor {
        // 从 builder 中提取 ctx 和 graph
        const ctx = builder.ctx;
        const graph = builder.gf;
        // 将 mem_ctx 转换为 kv_cache_mgr
        const kv_cache_mgr: ?*kv_cache.KVCache = if (mem_ctx) |ptr| @ptrCast(@alignCast(ptr)) else null;
        return self.forward(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos);
    }

    /// 虚表定义（用于 ModelInstance 运行时多态）
    pub const vtable = model.ModelVTable{
        .deinit = deinitAdapter,
        .buildGraph = buildGraphAdapter,
        .getParams = getParamsAdapter,
        .resetSSMStates = resetSSMStatesAdapter,
        .setKVCacheContext = setKVCacheContextAdapter,
    };

    fn deinitAdapter(data: *anyopaque, allocator: std.mem.Allocator) void {
        const self = @as(*QwenModel, @ptrCast(@alignCast(data)));
        // 释放 qwen_weights 中的 prefix 字符串和 layers 数组
        self.qwen_weights.deinit(allocator);
        // 释放 ssm_states 数组
        allocator.free(self.ssm_states);
        // 释放 ctx_weights（ggml 上下文）
        self.ctx_weights.deinit();
        // 释放 QwenModel 结构体本身
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
        const self = @as(*QwenModel, @ptrCast(@alignCast(data)));
        return self.buildGraph(builder, input_tokens, n_tokens, mem_ctx, start_pos);
    }

    fn getParamsAdapter(data: *anyopaque) *const model.ModelParams {
        const self = @as(*QwenModel, @ptrCast(@alignCast(data)));
        return self.getParams();
    }

    fn resetSSMStatesAdapter(data: *anyopaque) void {
        const self = @as(*QwenModel, @ptrCast(@alignCast(data)));
        self.resetSSMStates();
    }

    fn setKVCacheContextAdapter(data: *anyopaque, ctx: *ggml.Context) void {
        const self = @as(*QwenModel, @ptrCast(@alignCast(data)));
        self.ctx_kv_cache = ctx;
    }
};

pub fn isFullAttentionLayer(layer_idx: u32, interval: u32) bool {
    if (interval == 0) return true;
    return (layer_idx + 1) % interval == 0;
}

pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !QwenParams {
    var p = QwenParams{};
    p.base.n_vocab = gguf_file.getU32("llama.vocab_size") orelse blk: {
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            if (val.value_type == .array) break :blk @intCast(val.array_val.len);
        }
        break :blk 0;
    };
    p.base.n_embd = gguf_file.getU32("llama.embedding_length") orelse gguf_file.getU32("qwen35.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("llama.attention.head_count") orelse gguf_file.getU32("llama.head_count") orelse gguf_file.getU32("qwen35.attention.head_count") orelse 0;
    p.base.n_kv_head = gguf_file.getU32("llama.attention.head_count_kv") orelse gguf_file.getU32("llama.head_count_kv") orelse gguf_file.getU32("qwen35.attention.head_count_kv") orelse p.base.n_head;
    p.base.n_layer = gguf_file.getU32("llama.block_count") orelse gguf_file.getU32("qwen35.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("llama.feed_forward_length") orelse gguf_file.getU32("qwen35.feed_forward_length") orelse 0;
    p.base.n_expert = gguf_file.getU32("llama.expert_count") orelse 0;
    p.base.n_expert_used = gguf_file.getU32("llama.expert_used_count") orelse 0;
    p.attn_key_length = gguf_file.getU32("qwen35.attention.key_length") orelse gguf_file.getU32("llama.attention.key_length") orelse 0;
    p.attn_value_length = gguf_file.getU32("qwen35.attention.value_length") orelse gguf_file.getU32("llama.attention.value_length") orelse 0;
    if (p.attn_key_length > 0) {
        p.base.n_head_dim = p.attn_key_length;
    } else if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }
    p.base.max_seq_len = gguf_file.getU32("llama.context_length") orelse gguf_file.getU32("qwen35.context_length") orelse 32768;
    p.base.rope_theta = gguf_file.getF32("llama.rope.freq_base") orelse gguf_file.getF32("qwen35.rope.freq_base") orelse 10000000.0;
    p.base.rope_dim = gguf_file.getU32("qwen35.rope.dimension_count") orelse 64;
    p.base.norm_eps = gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse gguf_file.getF32("qwen35.attention.layer_norm_rms_epsilon") orelse 1e-6;
    p.full_attention_interval = gguf_file.getU32("qwen35.full_attention_interval") orelse 4;
    p.ssm_conv_kernel = gguf_file.getU32("qwen35.ssm.conv_kernel") orelse 4;
    p.ssm_state_size = gguf_file.getU32("qwen35.ssm.state_size") orelse 128;
    p.ssm_group_count = gguf_file.getU32("qwen35.ssm.group_count") orelse 16;
    p.ssm_time_step_rank = gguf_file.getU32("qwen35.ssm.time_step_rank") orelse 16;
    p.ssm_inner_size = gguf_file.getU32("qwen35.ssm.inner_size") orelse 2048;
    p.base.model_name = gguf_file.getString("general.name") orelse "";
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";
    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required model parameters", .{});
        return error.InvalidModelParams;
    }
    log.info("Qwen35: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}", .{ p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head, p.base.n_layer, p.base.n_ff });
    return p;
}

fn estimateMemSize(params: *const QwenParams) usize {
    _ = params;
    return 1024 * 1024 * 1024 * 2;
}

pub fn loadWeights(gguf_file: *const gguf.GGUFFile, ctx: *ggml.Context, params: *const QwenParams, allocator: std.mem.Allocator) !QwenWeights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Qwen35 weights...", .{});
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
        const is_full_attn = isFullAttentionLayer(@intCast(i), params.full_attention_interval);
        var lw = LayerWeights{
            .prefix = prefix,
            .layer_type = if (is_full_attn) .full_attention else .ssm,
            .attn_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_norm.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            },
            .post_attention_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "post_attention_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load post_attention_norm.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            },
            .ffn_norm_weight = undefined,
            .ffn_gate_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_gate.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            },
            .ffn_up_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_up.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            },
            .ffn_down_weight = loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch |err| {
                log.err("Layer {d}: failed to load ffn_down.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            },
        };
        lw.ffn_norm_weight = lw.post_attention_norm_weight;
        if (is_full_attn) {
            lw.attn_q_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_q.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.attn_k_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_k.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.attn_v_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_v.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.attn_output_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_output.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.attn_q_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_q_norm.weight") catch null;
            lw.attn_k_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_k_norm.weight") catch null;
        } else {
            lw.attn_qkv_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_qkv.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_qkv.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.attn_gate_weight = loadLayerWeight(ctx, gguf_file, prefix, "attn_gate.weight") catch |err| {
                log.err("Layer {d}: failed to load attn_gate.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.ssm_conv1d_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_conv1d.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_conv1d.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.ssm_a = loadLayerWeight(ctx, gguf_file, prefix, "ssm_a") catch |err| {
                log.err("Layer {d}: failed to load ssm_a: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.ssm_dt_bias = loadLayerWeight(ctx, gguf_file, prefix, "ssm_dt.bias") catch |err| {
                log.err("Layer {d}: failed to load ssm_dt.bias: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.ssm_alpha_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_alpha.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_alpha.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.ssm_beta_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_beta.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_beta.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.ssm_norm_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_norm.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_norm.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
            lw.ssm_out_weight = loadLayerWeight(ctx, gguf_file, prefix, "ssm_out.weight") catch |err| {
                log.err("Layer {d}: failed to load ssm_out.weight: {}\n", .{ i, err });
                allocator.free(prefix);
                return error.MissingWeight;
            };
        }
        layers[i] = lw;
        layers_loaded = i + 1;
    }
    log.info("All Qwen35 weights loaded ({d} layers)\n", .{n_layer});
    return QwenWeights{
        .base = .{ .params = params.base, .token_embd = token_embd, .output_weight = output_weight, .output_norm_weight = output_norm_weight },
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
        const tensor_data = gguf_file.getTensorData(info);
        const tensor_bytes = tensor.dataBytes();
        if (tensor_bytes.len != tensor_data.len) log.warn("Tensor '{s}' size mismatch: expected {d} bytes, got {d} bytes", .{ name, tensor_bytes.len, tensor_data.len });
        @memcpy(tensor_bytes, tensor_data);
        return tensor;
    }
    return error.TensorNotFound;
}

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
