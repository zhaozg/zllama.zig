//! 嵌入模型 — Qwen3-Embedding / BGE 等
//!
//! 基于 Qwen2 骨干，区别：
//! - 双向注意力（无 causal mask），利用 ggml mulMat 自动 GQA 广播
//! - 无 KV Cache（单次前向）
//! - Q/K normalization（RMS norm + per-head weight）
//! - 池化 + L2 归一化输出
//! - 无 output.weight（不需要 lm_head）

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const swiglu = @import("swiglu");
const embed = @import("embed");
const weight_loader = @import("weight_loader");
const pooling = @import("pooling");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const model = @import("../model.zig");
const qwen2 = @import("qwen2.zig");

const log = std.log.scoped(.embedding);

pub const EmbeddingWeights = struct {
    base: model.ModelWeights,
    layers: []qwen2.LayerWeights,
    pub fn deinit(self: *EmbeddingWeights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| allocator.free(layer.prefix);
        allocator.free(self.layers);
    }
};

pub const EmbeddingModel = struct {
    params: model.ModelParams,
    weights: EmbeddingWeights,
    ctx_weights: *ggml.Context,
    pooling_type: pooling.PoolingType = .mean,
    normalize_output: bool = true,

    pub fn init(self: *EmbeddingModel, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        const qwen2_params = try qwen2.parseParams(gguf_file, allocator);
        self.params = qwen2_params.base;
        if (gguf_file.getString("general.pooling_type")) |pt_str| {
            self.pooling_type = pooling.PoolingType.fromString(pt_str);
        } else if (gguf_file.getU32("qwen3.pooling_type")) |pt_val| {
            self.pooling_type = switch (pt_val) {
                2 => .cls,
                3 => .last,
                else => .mean,
            };
        }
        self.ctx_weights = try ggml.Context.initNoAlloc(weight_loader.estimateMemSize(gguf_file));
        const qwen2_weights = try qwen2.loadWeights(gguf_file, self.ctx_weights, &qwen2_params, allocator);
        self.weights = EmbeddingWeights{ .base = qwen2_weights.base, .layers = qwen2_weights.layers };
        log.info("Embedding: layers={d}, embd={d}, heads={d}, kv_heads={d}, head_dim={d}", .{
            self.params.n_layer,    self.params.n_embd,
            self.params.n_head,     self.params.n_kv_head,
            self.params.n_head_dim,
        });
        log.info("Pooling: {s}, normalize: {}", .{ @tagName(self.pooling_type), self.normalize_output });
    }

    pub fn deinit(self: *EmbeddingModel, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn forward(
        self: *EmbeddingModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
    ) !*ggml.Tensor {
        const p = &self.params;
        const w = &self.weights;
        const n_head: i64 = @intCast(p.n_head);
        const n_kv_head: i64 = @intCast(p.n_kv_head);
        const head_dim: i64 = @intCast(p.n_head_dim);
        const head_dim_k: i64 = if (p.n_head_dim_k > 0) @intCast(p.n_head_dim_k) else head_dim;
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.rope_dim);
        const rope_n_dims: i32 = @intCast(@min(rope_dim, head_dim));

        var cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
        cur.setName("token_embd");
        const pos_tensor = rope.buildPositionTensor(ctx, @intCast(n_tokens_i64), 0);

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

            // Reshape to [head_dim, n_head/kv_head, n_tokens]
            q = ggml.reshape3d(ctx, q, head_dim, n_head, n_tokens_i64);
            k = ggml.reshape3d(ctx, k, head_dim_k, n_kv_head, n_tokens_i64);
            v = ggml.reshape3d(ctx, v, head_dim, n_kv_head, n_tokens_i64);

            // Q/K normalization (Qwen3-Embedding)
            if (layer.attn_q_norm_weight) |q_norm| {
                q = ggml.rmsNorm(ctx, q, p.norm_eps);
                const q_norm_3d = ggml.reshape3d(ctx, q_norm, head_dim, 1, 1);
                const q_norm_target = ctx.newTensor3d(.f32, head_dim, n_head, n_tokens_i64) catch unreachable;
                const q_norm_rep = ggml.repeat(ctx, q_norm_3d, q_norm_target);
                q = ggml.mul(ctx, q, q_norm_rep);
            }
            if (layer.attn_k_norm_weight) |k_norm| {
                k = ggml.rmsNorm(ctx, k, p.norm_eps);
                const k_norm_3d = ggml.reshape3d(ctx, k_norm, head_dim_k, 1, 1);
                const k_norm_target = ctx.newTensor3d(.f32, head_dim_k, n_kv_head, n_tokens_i64) catch unreachable;
                const k_norm_rep = ggml.repeat(ctx, k_norm_3d, k_norm_target);
                k = ggml.mul(ctx, k, k_norm_rep);
            }

            // RoPE
            q = ggml.ropeExt(ctx, q, pos_tensor, null, rope_n_dims, 0, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k = ggml.ropeExt(ctx, k, pos_tensor, null, rope_n_dims, 0, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

            // Permute for mulMat-based attention: [head_dim, n_tokens, n_head/kv_head]
            // ggml mulMat automatically handles GQA by broadcasting ne[2]
            q = ggml.cont(ctx, ggml.permute(ctx, q, 0, 2, 1, 3));
            k = ggml.cont(ctx, ggml.permute(ctx, k, 0, 2, 1, 3));
            v = ggml.cont(ctx, ggml.permute(ctx, v, 0, 2, 1, 3));

            // kq = k^T @ q : [head_dim_k, n_tokens, n_kv_head] @ [head_dim, n_tokens, n_head]
            // → [n_tokens, n_tokens, n_head]  (GQA: k broadcasts from n_kv_head to n_head)
            var kq = ggml.mulMat(ctx, k, q);
            kq = ggml.scale(ctx, kq, 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
            // Bidirectional: no causal mask
            kq = ggml.softMax(ctx, kq);

            // v^T: [head_dim, n_tokens, n_kv_head] → [n_tokens, head_dim, n_kv_head]
            const v_t = ggml.cont(ctx, ggml.transpose(ctx, v));
            // kqv = v^T @ kq : [n_tokens, head_dim, n_kv_head] @ [n_tokens, n_tokens, n_head]
            // → [head_dim, n_tokens, n_head]  (GQA: v_t broadcasts)
            var kqv = ggml.mulMat(ctx, v_t, kq);
            // Permute back: [head_dim, n_tokens, n_head] → [head_dim, n_head, n_tokens]
            kqv = ggml.cont(ctx, ggml.permute(ctx, kqv, 0, 2, 1, 3));

            var attn_out = ggml.reshape2d(ctx, kqv, n_head * head_dim, n_tokens_i64);
            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }
            cur = ggml.add(ctx, cur, attn_out);

            // FFN
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
        cur = pooling.poolHidden(ctx, cur, self.pooling_type);
        if (self.normalize_output) {
            cur = pooling.normalize(ctx, cur);
        }
        cur.setName("embedding_vector");
        graph.buildForwardExpand(cur);
        return cur;
    }

    pub fn buildGraph(self: *EmbeddingModel, builder: *graph_builder.GraphBuilder, input_tokens: *ggml.Tensor, n_tokens: i32, mem_ctx: ?*memory.MemoryContext, start_pos: i32) !*ggml.Tensor {
        _ = mem_ctx;
        _ = start_pos;
        return self.forward(builder.ctx, builder.gf, input_tokens, n_tokens);
    }

    pub const vtable = model.ModelVTable{
        .deinit = deinitAdapter,
        .buildGraph = buildGraphAdapter,
        .getParams = getParamsAdapter,
        .resetSSMStates = resetSSMStatesAdapter,
    };
    fn deinitAdapter(data: *anyopaque, allocator: std.mem.Allocator) void {
        const self = @as(*EmbeddingModel, @ptrCast(@alignCast(data)));
        self.deinit(allocator);
        allocator.destroy(self);
    }
    fn buildGraphAdapter(data: *anyopaque, builder: *graph_builder.GraphBuilder, input_tokens: *ggml.Tensor, n_tokens: i32, mem_ctx: ?*anyopaque, start_pos: i32) !*ggml.Tensor {
        const self = @as(*EmbeddingModel, @ptrCast(@alignCast(data)));
        _ = mem_ctx;
        _ = start_pos;
        return self.forward(builder.ctx, builder.gf, input_tokens, n_tokens);
    }
    fn getParamsAdapter(data: *anyopaque) *const model.ModelParams {
        const self = @as(*EmbeddingModel, @ptrCast(@alignCast(data)));
        return &self.params;
    }
    fn resetSSMStatesAdapter(data: *anyopaque) void {
        _ = data;
    }
};
