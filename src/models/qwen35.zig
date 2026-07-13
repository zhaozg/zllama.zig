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
const weight_loader = @import("weight_loader");

const model = @import("../model.zig");

const log = std.log.scoped(.model_qwen35);

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
/// 在 setKVCacheContext 时预分配，forward 中直接使用，无需 lazy-init。
const LayerSSMState = struct {
    conv_state: ?*ggml.Tensor, // [d_conv-1, conv_dim] — SSM 层预分配，全注意力层为 null
    ssm_state: ?*ggml.Tensor, // [S_v, S_v, H, 1] — SSM 层预分配，全注意力层为 null
    is_ssm_layer: bool = false,
};

pub const QwenModel = struct {
    qwen_params: QwenParams,
    qwen_weights: QwenWeights,
    ctx_weights: *ggml.Context,
    ctx_kv_cache: ?*ggml.Context = null, // 用于分配持久状态的 context
    ssm_states: []LayerSSMState, // 每层的 SSM 状态（预分配后持久化）
    ssm_states_allocated: bool = false,

    pub fn init(self: *QwenModel, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.qwen_params = try qwen35_loader.parseParams(gguf_file, allocator);
        self.ctx_weights = try ggml.Context.initNoAlloc(qwen35_loader.estimateMemSize(gguf_file));
        self.qwen_weights = try qwen35_loader.loadWeights(gguf_file, self.ctx_weights, &self.qwen_params, allocator);

        // 初始化 SSM 状态数组（张量在 setKVCacheContext 时预分配）
        const n_layer = self.qwen_params.base.n_layer;
        self.ssm_states = try allocator.alloc(LayerSSMState, n_layer);
        for (0..n_layer) |i| {
            const idx: u32 = @intCast(i);
            const is_ssm = !isFullAttentionLayer(idx, self.qwen_params.full_attention_interval);
            self.ssm_states[i] = .{
                .conv_state = null,
                .ssm_state = null,
                .is_ssm_layer = is_ssm,
            };
        }
        self.ssm_states_allocated = false;
    }

    /// 在 ctx_kv_cache 中预分配所有 SSM 层的持久状态张量。
    /// 由 setKVCacheContext 自动调用；也可在 /reset 后手动调用重建状态。
    pub fn allocateSSMStates(self: *QwenModel) !void {
        const ctx = self.ctx_kv_cache orelse return error.NoKVCacheContext;
        const p = self.qwen_params;

        const d_conv: i64 = @intCast(p.ssm_conv_kernel);
        const d_inner: i64 = @intCast(p.ssm_inner_size);
        const d_state: i64 = @intCast(p.ssm_state_size);
        const n_group: i64 = @intCast(p.ssm_group_count);
        const dt_rank: i64 = @intCast(p.ssm_time_step_rank);

        const head_v_dim = @divExact(d_inner, dt_rank);
        const num_v_heads = dt_rank;
        const key_dim = d_state * n_group;
        const value_dim = head_v_dim * num_v_heads;
        const conv_dim = key_dim * 2 + value_dim;

        for (self.ssm_states) |*state| {
            if (!state.is_ssm_layer) continue;

            // Pre-allocate conv_state: [d_conv-1, conv_dim]
            state.conv_state = try ctx.newTensor2d(.f32, d_conv - 1, conv_dim);
            state.conv_state.?.setZero();

            // Pre-allocate ssm_state: [head_v_dim, head_v_dim, num_v_heads, 1]
            state.ssm_state = try ctx.newTensor4d(.f32, head_v_dim, head_v_dim, num_v_heads, 1);
            state.ssm_state.?.setZero();
        }
        self.ssm_states_allocated = true;
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
    /// 重置所有 SSM 状态为零（不清除预分配的张量，避免重新分配开销）
    pub fn resetSSMStates(self: *QwenModel) void {
        for (self.ssm_states) |*state| {
            if (state.conv_state) |t| t.setZero();
            if (state.ssm_state) |t| t.setZero();
        }
    }

    // ============================================================================
    // Qwen 3.5 图构建类
    // ============================================================================
    //
    // 参考 llama.cpp 的 llm_build_context 模式：将 forward() 拆分为独立的图构建类。
    // Qwen35Graph 将 forward 分解为 build / buildFullAttnLayer / buildSSMLayer / buildOutput。

    pub const Qwen35Graph = struct {
        ctx: *ggml.Context,
        gf: *ggml.CGraph,
        params: *const QwenParams,
        weights: *const QwenWeights,
        ssm_states: []LayerSSMState,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,

        const Self = @This();

        pub fn init(
            ctx: *ggml.Context,
            gf: *ggml.CGraph,
            params: *const QwenParams,
            weights: *const QwenWeights,
            ssm_states: []LayerSSMState,
            kv_cache_mgr: ?*kv_cache.KVCache,
            start_pos: i32,
        ) Self {
            return Self{
                .ctx = ctx,
                .gf = gf,
                .params = params,
                .weights = weights,
                .ssm_states = ssm_states,
                .kv_cache_mgr = kv_cache_mgr,
                .start_pos = start_pos,
            };
        }

        pub fn build(
            self: *Self,
            input_tokens: *ggml.Tensor,
            n_tokens: i32,
        ) !*ggml.Tensor {
            const p = self.params;
            const w = self.weights;
            const n_head: i64 = @intCast(p.base.n_head);
            const n_kv_head: i64 = @intCast(p.base.n_kv_head);
            const head_dim: i64 = @intCast(p.base.n_head_dim);
            const n_tokens_i64: i64 = n_tokens;
            const rope_dim: i64 = @intCast(p.base.rope_dim);

            var cur = embed.tokenEmbedding(self.ctx, w.base.token_embd, input_tokens);
            cur.setName("token_embd");

            for (w.layers, 0..) |*layer, i| {
                const layer_idx: u32 = @intCast(i);
                const is_full_attn = isFullAttentionLayer(layer_idx, p.full_attention_interval);
                var name_buf: [128]u8 = undefined;

                var attn_input = rms_norm.rmsNorm(self.ctx, cur, layer.attn_norm_weight, p.base.norm_eps);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    attn_input.setName(name_buf[0..name.len :0]);
                }

                if (is_full_attn) {
                    const attn_out = try self.buildFullAttnLayer(layer, attn_input, n_tokens_i64, n_head, n_kv_head, head_dim, rope_dim, i, &name_buf);
                    cur = ggml.add(self.ctx, cur, attn_out);
                } else {
                    const ssm_out = try self.buildSSMLayer(layer, attn_input, n_tokens_i64, i);
                    cur = ggml.add(self.ctx, cur, ssm_out);
                }

                var post_attn = rms_norm.rmsNorm(self.ctx, cur, layer.post_attention_norm_weight, p.base.norm_eps);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attn_norm", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    post_attn.setName(name_buf[0..name.len :0]);
                }

                const ffn_out = swiglu.swiGLU(self.ctx, post_attn, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_out", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    ffn_out.setName(name_buf[0..name.len :0]);
                }
                cur = ggml.add(self.ctx, cur, ffn_out);
            }

            return self.buildOutput(cur);
        }

        fn buildFullAttnLayer(
            self: *Self,
            layer: *const LayerWeights,
            attn_input: *ggml.Tensor,
            n_tokens_i64: i64,
            n_head: i64,
            n_kv_head: i64,
            head_dim: i64,
            rope_dim: i64,
            layer_idx: usize,
            name_buf: *[128]u8,
        ) !*ggml.Tensor {
            const p = self.params;
            const ctx = self.ctx;
            const gf = self.gf;

            const q_w = layer.attn_q_weight orelse return error.MissingWeight;
            const k_w = layer.attn_k_weight orelse return error.MissingWeight;
            const v_w = layer.attn_v_weight orelse return error.MissingWeight;
            const o_w = layer.attn_output_weight orelse return error.MissingWeight;

            const q_full = ggml.mulMat(ctx, q_w, attn_input);
            var k = ggml.mulMat(ctx, k_w, attn_input);
            var v = ggml.mulMat(ctx, v_w, attn_input);

            var q = ctx.view3d(q_full, head_dim, n_head, n_tokens_i64, @as(usize, @intCast(head_dim * 2 * @sizeOf(f32))), @as(usize, @intCast(head_dim * 2 * n_head * @sizeOf(f32))), 0);
            const q_gate_view = ctx.view3d(q_full, head_dim, n_head, n_tokens_i64, @as(usize, @intCast(head_dim * 2 * @sizeOf(f32))), @as(usize, @intCast(head_dim * 2 * n_head * @sizeOf(f32))), @as(usize, @intCast(head_dim * @sizeOf(f32))));

            q = ggml.cont(ctx, q);

            if (layer.attn_q_norm_weight) |q_norm_raw| {
                q = ggml.rmsNorm(ctx, q, p.base.norm_eps);
                const q_norm_3d = ggml.reshape3d(ctx, q_norm_raw, head_dim, 1, 1);
                const q_norm_target = ctx.newTensor3d(.f32, head_dim, n_head, n_tokens_i64) catch unreachable;
                q = ggml.mul(ctx, q, ggml.repeat(ctx, q_norm_3d, q_norm_target));
            }

            k = ggml.reshape3d(ctx, k, head_dim, n_kv_head, n_tokens_i64);
            if (layer.attn_k_norm_weight) |k_norm_raw| {
                k = ggml.rmsNorm(ctx, k, p.base.norm_eps);
                const k_norm_3d = ggml.reshape3d(ctx, k_norm_raw, head_dim, 1, 1);
                const k_norm_target = ctx.newTensor3d(.f32, head_dim, n_kv_head, n_tokens_i64) catch unreachable;
                k = ggml.mul(ctx, k, ggml.repeat(ctx, k_norm_3d, k_norm_target));
            }
            v = ggml.reshape3d(ctx, v, head_dim, n_kv_head, n_tokens_i64);

            const pos_tensor = rope.buildMultiPositionTensor(ctx, @intCast(n_tokens_i64), self.start_pos);
            const rope_sections = [4]i32{ 11, 11, 10, 0 };
            q = ggml.ropeMulti(ctx, q, pos_tensor, @intCast(rope_dim), &rope_sections, 40, 0, p.base.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k = ggml.ropeMulti(ctx, k, pos_tensor, @intCast(rope_dim), &rope_sections, 40, 0, p.base.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

            if (self.kv_cache_mgr) |cache| {
                cache.setKv(ctx, gf, layer_idx, k, v, @intCast(n_tokens_i64));
                k = cache.getKView(ctx, layer_idx);
                v = cache.getVView(ctx, layer_idx);
            }

            const cache_len: i64 = if (self.kv_cache_mgr) |cache| @as(i64, @intCast(cache.currentLen())) else n_tokens_i64;

            var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
                .n_head = n_head,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = self.start_pos,
                .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            }, null);

            const gate_cont = ggml.cont(ctx, q_gate_view);
            const gate_sigmoid = ggml.sigmoid(ctx, ggml.reshape2d(ctx, gate_cont, n_head * head_dim, n_tokens_i64));
            attn_out = ggml.mul(ctx, attn_out, gate_sigmoid);
            {
                const name = std.fmt.bufPrint(name_buf, "blk.{d}.attn_gated", .{layer_idx}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }

            var result = ggml.mulMat(ctx, o_w, attn_out);
            {
                const name = std.fmt.bufPrint(name_buf, "blk.{d}.attn_out", .{layer_idx}) catch unreachable;
                name_buf[name.len] = 0;
                result.setName(name_buf[0..name.len :0]);
            }
            return result;
        }

        fn buildSSMLayer(
            self: *Self,
            layer: *const LayerWeights,
            attn_input: *ggml.Tensor,
            n_tokens_i64: i64,
            layer_idx: usize,
        ) !*ggml.Tensor {
            const p = self.params;
            const ctx = self.ctx;
            const gf = self.gf;
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

            // Step 1: Input projections
            const qkv_mixed = ggml.mulMat(ctx, layer.attn_qkv_weight.?, attn_input);
            qkv_mixed.setName("ssm_qkv_mixed");
            const z = ggml.mulMat(ctx, layer.attn_gate_weight.?, attn_input);
            z.setName("ssm_z");

            // Step 2: Beta
            var beta = ggml.mulMat(ctx, layer.ssm_beta_weight.?, attn_input);
            beta = ggml.reshape4d(ctx, beta, 1, num_v_heads, n_tokens_i64, n_seqs);
            beta = ggml.sigmoid(ctx, beta);
            beta.setName("ssm_beta");

            // Step 3: Alpha -> gate
            var alpha = ggml.mulMat(ctx, layer.ssm_alpha_weight.?, attn_input);
            alpha = ggml.reshape3d(ctx, alpha, num_v_heads, n_tokens_i64, n_seqs);
            const alpha_biased = ggml.add(ctx, alpha, layer.ssm_dt_bias.?);
            const alpha_softplus = ggml.softplus(ctx, alpha_biased);
            var gate = ggml.mul(ctx, alpha_softplus, layer.ssm_a.?);
            gate = ggml.reshape4d(ctx, gate, 1, num_v_heads, n_tokens_i64, n_seqs);
            gate.setName("ssm_gate");

            // Step 4: Conv1d
            const qkv_2d = ggml.reshape2d(ctx, qkv_mixed, conv_dim, n_tokens_i64);
            const qkv_transposed = ggml.cont(ctx, ggml.permute(ctx, qkv_2d, 1, 0, 2, 3));

            // SSM conv_state 已在 setKVCacheContext 时预分配，直接使用
            if (self.ssm_states[layer_idx].conv_state == null) {
                return error.SSMStateNotPreallocated;
            }

            const conv_state = self.ssm_states[layer_idx].conv_state.?;
            const concat_result = ggml.concat(ctx, conv_state, qkv_transposed, 0);
            const concat_cont = ggml.cont(ctx, concat_result);

            const sx_3d = ctx.view3d(concat_cont, d_conv - 1 + n_tokens_i64, conv_dim, 1, @as(usize, @intCast((d_conv - 1 + n_tokens_i64) * @sizeOf(f32))), @as(usize, @intCast((d_conv - 1 + n_tokens_i64) * @sizeOf(f32) * conv_dim)), 0);

            const conv_output = ggml.ssmConv(ctx, sx_3d, layer.ssm_conv1d_weight.?);
            var conv_silu = ggml.silu(ctx, conv_output);
            conv_silu.setName("ssm_conv_silu");

            // Update conv state
            const row_stride = @as(usize, @intCast(@as(usize, @intCast(d_conv - 1 + n_tokens_i64)) * @sizeOf(f32)));
            const conv_state_update = ctx.view2d(concat_cont, d_conv - 1, conv_dim, row_stride, @as(usize, @intCast(@as(usize, @intCast(n_tokens_i64)) * @sizeOf(f32))));
            const conv_state_cpy = ggml.cpy(ctx, conv_state_update, conv_state);
            gf.buildForwardExpand(conv_state_cpy);

            // Step 5: Extract Q, K, V
            const qkv_stride = conv_silu.strides()[1];
            var q_conv = ctx.view2d(conv_silu, key_dim, n_tokens_i64, qkv_stride, 0);
            var k_conv = ctx.view2d(conv_silu, key_dim, n_tokens_i64, qkv_stride, @intCast(key_dim * @sizeOf(f32)));
            var v_conv = ctx.view2d(conv_silu, value_dim, n_tokens_i64, qkv_stride, @intCast(2 * key_dim * @sizeOf(f32)));

            q_conv = ggml.cont(ctx, q_conv);
            k_conv = ggml.cont(ctx, k_conv);
            v_conv = ggml.cont(ctx, v_conv);
            q_conv = ggml.reshape4d(ctx, q_conv, head_k_dim, num_k_heads, n_tokens_i64, n_seqs);
            k_conv = ggml.reshape4d(ctx, k_conv, head_k_dim, num_k_heads, n_tokens_i64, n_seqs);
            v_conv = ggml.reshape4d(ctx, v_conv, head_v_dim, num_v_heads, n_tokens_i64, n_seqs);

            q_conv = ggml.l2Norm(ctx, q_conv, p.base.norm_eps);
            k_conv = ggml.l2Norm(ctx, k_conv, p.base.norm_eps);

            if (num_k_heads != num_v_heads) {
                q_conv = ggml.repeat4d(ctx, q_conv, head_k_dim, num_v_heads, n_tokens_i64, n_seqs);
                k_conv = ggml.repeat4d(ctx, k_conv, head_k_dim, num_v_heads, n_tokens_i64, n_seqs);
            }

            // SSM ssm_state 已在 setKVCacheContext 时预分配，直接使用
            if (self.ssm_states[layer_idx].ssm_state == null) {
                return error.SSMStateNotPreallocated;
            }
            const ssm_state = self.ssm_states[layer_idx].ssm_state.?;

            const gdn_output = ggml.gatedDeltaNet(ctx, q_conv, k_conv, v_conv, gate, beta, ssm_state, 1);
            gdn_output.setName("ssm_gdn_output");

            const attn_output = ctx.view4d(gdn_output, head_v_dim, num_v_heads, n_tokens_i64, n_seqs, @as(usize, @intCast(head_v_dim * @sizeOf(f32))), @as(usize, @intCast(head_v_dim * num_v_heads * @sizeOf(f32))), @as(usize, @intCast(head_v_dim * num_v_heads * n_tokens_i64 * @sizeOf(f32))), 0);

            const state_offset = @as(usize, @intCast(head_v_dim * num_v_heads * n_tokens_i64 * n_seqs * @sizeOf(f32)));
            const new_state = ctx.view4d(gdn_output, head_v_dim, head_v_dim, num_v_heads, n_seqs, @as(usize, @intCast(head_v_dim * @sizeOf(f32))), @as(usize, @intCast(head_v_dim * head_v_dim * @sizeOf(f32))), @as(usize, @intCast(head_v_dim * head_v_dim * num_v_heads * @sizeOf(f32))), state_offset);
            const state_cpy = ggml.cpy(ctx, new_state, ssm_state);
            gf.buildForwardExpand(state_cpy);

            // Step 7: Gated normalization
            const z_4d = ggml.reshape4d(ctx, z, head_v_dim, num_v_heads, n_tokens_i64, n_seqs);
            const attn_out_norm = ggml.rmsNorm(ctx, attn_output, p.base.norm_eps);
            const z_silu = ggml.silu(ctx, z_4d);
            const gated = ggml.mul(ctx, attn_out_norm, z_silu);

            // Step 8: Output projection
            const final_output = ggml.reshape3d(ctx, gated, head_v_dim * num_v_heads, n_tokens_i64, n_seqs);
            var result = ggml.mulMat(ctx, layer.ssm_out_weight.?, final_output);
            result.setName("ssm_out");
            return ggml.reshape2d(ctx, result, p.base.n_embd, n_tokens_i64);
        }

        fn buildOutput(self: *Self, cur: *ggml.Tensor) *ggml.Tensor {
            const p = self.params;
            const w = self.weights;
            const ctx = self.ctx;
            const gf = self.gf;

            var c = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.base.norm_eps);
            c.setName("output_norm");
            const out_w = w.base.output_weight orelse w.base.token_embd;
            var logits_tensor = ggml.mulMat(ctx, out_w, c);
            logits_tensor.setName("logits");
            gf.buildForwardExpand(logits_tensor);
            return logits_tensor;
        }
    };

    /// Thin delegate to Qwen35Graph.
    pub fn forward(self: *QwenModel, ctx: *ggml.Context, graph: *ggml.CGraph, input_tokens: *ggml.Tensor, n_tokens: i32, kv_cache_mgr: ?*kv_cache.KVCache, start_pos: i32) !*ggml.Tensor {
        var g = Qwen35Graph.init(ctx, graph, &self.qwen_params, &self.qwen_weights, self.ssm_states, kv_cache_mgr, start_pos);
        return try g.build(input_tokens, n_tokens);
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
        // 自动预分配 SSM 状态张量（避免 forward 中的 lazy-init 开销）
        self.allocateSSMStates() catch |err| {
            log.err("Failed to pre-allocate SSM states: {}", .{err});
        };
    }
};

pub fn isFullAttentionLayer(layer_idx: u32, interval: u32) bool {
    if (interval == 0) return true;
    return (layer_idx + 1) % interval == 0;
}

const qwen35_loader = @import("qwen35_loader.zig");

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
