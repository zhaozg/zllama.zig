//! Qwen3VL 模型实现
//!
//! 纯注意力 Transformer，支持 fused QKV 权重和视觉嵌入注入。
//! 与 Qwen2 类似但使用 attn_qkv.weight（fused QKV）而非 separate Q/K/V。
//! 支持 deepstack 层（前 n_deepstack_layers 层添加嵌入残差）。
//!
//! 参考: deps/llama.cpp/src/models/qwen3vl.cpp

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
const weight_loader = @import("weight_loader");

const model = @import("../model.zig");

const log = std.log.scoped(.model_qwen3vl);

pub const Qwen3VLParams = struct {
    base: model.ModelParams = .{},
    rope_sections: [4]i32 = .{ 0, 0, 0, 0 },
    n_deepstack_layers: u32 = 0,
};

pub const LayerWeights = struct {
    prefix: []const u8,
    attn_norm_weight: *ggml.Tensor,
    ffn_norm_weight: *ggml.Tensor,
    // Fused QKV (attn_qkv.weight) or separate Q/K/V
    attn_qkv_weight: ?*ggml.Tensor = null, // fused QKV
    attn_q_weight: ?*ggml.Tensor = null, // separate Q
    attn_k_weight: ?*ggml.Tensor = null, // separate K
    attn_v_weight: ?*ggml.Tensor = null, // separate V
    attn_q_norm_weight: ?*ggml.Tensor = null,
    attn_k_norm_weight: ?*ggml.Tensor = null,
    attn_output_weight: *ggml.Tensor,
    ffn_gate_weight: *ggml.Tensor,
    ffn_up_weight: *ggml.Tensor,
    ffn_down_weight: *ggml.Tensor,
};

pub const Qwen3VLWeights = struct {
    base: model.ModelWeights,
    layers: []LayerWeights,
    pub fn deinit(self: *Qwen3VLWeights, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| allocator.free(layer.prefix);
        allocator.free(self.layers);
    }
};

pub const Qwen3VLModel = struct {
    params: Qwen3VLParams,
    weights: Qwen3VLWeights,
    ctx_weights: *ggml.Context,

    pub fn init(self: *Qwen3VLModel, allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void {
        _ = io;
        self.params = try parseParams(gguf_file, allocator);
        self.ctx_weights = try ggml.Context.initNoAlloc(weight_loader.estimateMemSize(gguf_file));
        self.weights = try loadWeights(gguf_file, self.ctx_weights, &self.params, allocator);
    }

    pub fn deinit(self: *Qwen3VLModel, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
        self.ctx_weights.deinit();
    }

    pub fn getParams(self: *const Qwen3VLModel) *const model.ModelParams {
        return &self.params.base;
    }

    pub fn getWeights(self: *const Qwen3VLModel) *const model.ModelWeights {
        return &self.weights.base;
    }

    /// Standard forward pass with token embedding lookup.
    pub fn forward(
        self: *Qwen3VLModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
    ) !*ggml.Tensor {
        return self.forwardWithEmbdOverride(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos, null, 0);
    }

    /// Forward pass with optional embedding override.
    /// When embd_override is provided, it replaces the token embedding lookup
    /// for the first n_override tokens. This is used for vision/audio media tokens.
    pub fn forwardWithEmbdOverride(
        self: *Qwen3VLModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        embd_override: ?*ggml.Tensor,
        embd_offset: i32,
    ) !*ggml.Tensor {
        _ = embd_offset;
        const p = &self.params.base;
        const w = &self.weights;
        const n_embd: i64 = @intCast(p.n_embd);
        const n_head: i64 = @intCast(p.n_head);
        const n_kv_head: i64 = @intCast(p.n_kv_head);
        const head_dim: i64 = @intCast(p.n_head_dim);
        const n_tokens_i64: i64 = n_tokens;
        const rope_dim: i64 = @intCast(p.rope_dim);
        const rope_sections = self.params.rope_sections;
        const n_deepstack_layers: i64 = @intCast(self.params.n_deepstack_layers);

        // Token embedding or override
        var cur: *ggml.Tensor = undefined;
        if (embd_override) |eo| {
            // The embedding override may contain deepstack features concatenated
            // along the feature dimension: [n_embd * (1 + n_deepstack_layers), n_tokens]
            // We need to use only the first n_embd values as the initial embedding.
            // The deepstack slices will be added at the appropriate layers.
            if (eo.ne()[0] > n_embd) {
                // Extract the first n_embd rows as the base embedding
                cur = ctx.view2d(eo, n_embd, n_tokens_i64, @as(usize, @intCast(eo.nb()[1])), 0);
                cur.setName("embd_override_base");
            } else {
                cur = eo;
                cur.setName("embd_override");
            }
        } else {
            cur = embed.tokenEmbedding(ctx, w.base.token_embd, input_tokens);
            cur.setName("token_embd");
        }

        // inp_pos - contains the positions
        const pos_tensor = rope.buildMultiPositionTensor(ctx, @intCast(n_tokens_i64), start_pos);

        // inp_out_ids for last layer output selection (like reference)
        const inp_out_ids: ?*ggml.Tensor = null; // out_ids support reserved for future multi-token selection

        for (w.layers, 0..) |*layer, i| {
            const il: i64 = @intCast(i);
            _ = il;
            var name_buf: [128]u8 = undefined;

            // inpSA = residual (like reference: ggml_tensor * inpSA = inpL;)
            const inpSA = cur;

            // Attention norm (RMS norm)
            var attn_input = rms_norm.rmsNorm(ctx, cur, layer.attn_norm_weight, p.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_input.setName(name_buf[0..name.len :0]);
            }

            // QKV projection: fused or separate
            var q: *ggml.Tensor = undefined;
            var k: *ggml.Tensor = undefined;
            var v: *ggml.Tensor = undefined;

            if (layer.attn_qkv_weight) |qkv_w| {
                // Fused QKV projection
                const qkv = ggml.mulMat(ctx, qkv_w, attn_input);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.qkv", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    qkv.setName(name_buf[0..name.len :0]);
                }

                // Split Q, K, V from fused output
                const q_dim: i64 = n_head * head_dim;
                const kv_dim: i64 = n_kv_head * head_dim;
                const qkv_stride = qkv.nb()[1];
                const qv = ctx.view2d(qkv, q_dim, n_tokens_i64, qkv_stride, 0);
                const kv = ctx.view2d(qkv, kv_dim, n_tokens_i64, qkv_stride, @as(usize, @intCast(q_dim * @sizeOf(f32))));
                const vv = ctx.view2d(qkv, kv_dim, n_tokens_i64, qkv_stride, @as(usize, @intCast((q_dim + kv_dim) * @sizeOf(f32))));

                q = ggml.cont(ctx, qv);
                k = ggml.cont(ctx, kv);
                v = ggml.cont(ctx, vv);
            } else {
                // Separate Q/K/V projection
                q = ggml.mulMat(ctx, layer.attn_q_weight orelse return error.MissingQWeight, attn_input);
                k = ggml.mulMat(ctx, layer.attn_k_weight orelse return error.MissingKWeight, attn_input);
                v = ggml.mulMat(ctx, layer.attn_v_weight orelse return error.MissingVWeight, attn_input);
            }

            // Reshape to 3D: [head_dim, n_heads, n_tokens]
            q = ggml.reshape3d(ctx, ggml.cont(ctx, q), head_dim, n_head, n_tokens_i64);
            k = ggml.reshape3d(ctx, ggml.cont(ctx, k), head_dim, n_kv_head, n_tokens_i64);
            v = ggml.reshape3d(ctx, ggml.cont(ctx, v), head_dim, n_kv_head, n_tokens_i64);

            // Q/K normalization (RMS norm) — reference: build_norm(Qcur, attn_q_norm, NULL, LLM_NORM_RMS, il)
            if (layer.attn_q_norm_weight) |q_norm| {
                q = ggml.rmsNorm(ctx, q, p.norm_eps);
                const q_norm_3d = ctx.view3d(q_norm, head_dim, 1, 1, @sizeOf(f32), @sizeOf(f32), 0);
                const q_norm_target = ctx.newTensor3d(.f32, head_dim, n_head, n_tokens_i64) catch unreachable;
                const q_norm_rep = ggml.repeat(ctx, q_norm_3d, q_norm_target);
                q = ggml.mul(ctx, q, q_norm_rep);
            }
            if (layer.attn_k_norm_weight) |k_norm| {
                k = ggml.rmsNorm(ctx, k, p.norm_eps);
                const k_norm_3d = ctx.view3d(k_norm, head_dim, 1, 1, @sizeOf(f32), @sizeOf(f32), 0);
                const k_norm_target = ctx.newTensor3d(.f32, head_dim, n_kv_head, n_tokens_i64) catch unreachable;
                const k_norm_rep = ggml.repeat(ctx, k_norm_3d, k_norm_target);
                k = ggml.mul(ctx, k, k_norm_rep);
            }

            // RoPE with multi sections — reference: ggml_rope_multi on Qcur and Kcur
            q = ggml.ropeMulti(ctx, q, pos_tensor, @intCast(rope_dim), &rope_sections, 40, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k = ggml.ropeMulti(ctx, k, pos_tensor, @intCast(rope_dim), &rope_sections, 40, 0, p.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

            // KV Cache: setKv expects [head_dim, n_kv_head, n_tokens]
            if (kv_cache_mgr) |cache| {
                cache.setKv(ctx, graph, i, k, v, @intCast(n_tokens_i64));
                k = cache.getKView(ctx, i);
                v = cache.getVView(ctx, i);
            }

            const cache_len: i64 = if (kv_cache_mgr) |cache| @as(i64, @intCast(cache.currentLen())) else n_tokens_i64;

            // Attention — reference: build_attn with 1.0f/sqrtf(float(n_embd_head))
            var attn_out = attention.scaledDotProductAttention(ctx, q, k, v, .{
                .n_head = n_head,
                .n_kv_head = n_kv_head,
                .head_dim = head_dim,
                .n_tokens = n_tokens_i64,
                .cache_len = cache_len,
                .start_pos = start_pos,
                .scale_factor = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            }, null);

            // Output projection
            attn_out = ggml.mulMat(ctx, layer.attn_output_weight, attn_out);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                attn_out.setName(name_buf[0..name.len :0]);
            }

            // Last layer output selection (like reference: if il == n_layer - 1 && inp_out_ids)
            var ffn_inp: *ggml.Tensor = undefined;
            var ffn_residual: *ggml.Tensor = undefined;
            if (i == w.layers.len - 1) {
                if (inp_out_ids) |out_ids| {
                    attn_out = ggml.getRows(ctx, attn_out, out_ids);
                    ffn_residual = ggml.getRows(ctx, inpSA, out_ids);
                    ffn_inp = ggml.add(ctx, attn_out, ffn_residual);
                } else {
                    ffn_inp = ggml.add(ctx, attn_out, inpSA);
                }
            } else {
                ffn_inp = ggml.add(ctx, attn_out, inpSA);
            }
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_inp", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_inp.setName(name_buf[0..name.len :0]);
            }

            // FFN norm
            var ffn_input = rms_norm.rmsNorm(ctx, ffn_inp, layer.ffn_norm_weight, p.norm_eps);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_input.setName(name_buf[0..name.len :0]);
            }

            // FFN (SwiGLU) — reference: build_ffn with LLM_FFN_SILU, LLM_FFN_PAR
            const ffn_out = swiglu.swiGLU(ctx, ffn_input, layer.ffn_gate_weight, layer.ffn_up_weight, layer.ffn_down_weight);
            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                ffn_out.setName(name_buf[0..name.len :0]);
            }

            // Residual connection
            cur = ggml.add(ctx, ffn_inp, ffn_out);
            // build_cvec equivalent: ggml_cont (reference: cur = build_cvec(cur, il))
            cur = ggml.cont(ctx, cur);

            {
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.l_out", .{i}) catch unreachable;
                name_buf[name.len] = 0;
                cur.setName(name_buf[0..name.len :0]);
            }

            // Deepstack: add embedding slice for first n_deepstack_layers
            // Reference: ggml_view_2d(ctx0, res->t_inp_embd, n_embd, n_tokens, ..., (il + 1) * n_embd * sizeof(float))
            // res->t_inp_embd is the f32 input embedding (embd_override when provided)
            if (embd_override != null and @as(u64, @intCast(i)) < n_deepstack_layers) {
                const ds_offset = (@as(u64, @intCast(i)) + 1) * @as(u64, @intCast(n_embd)) * @sizeOf(f32);
                const ds = ctx.view2d(
                    embd_override.?,
                    n_embd,
                    n_tokens_i64,
                    @as(usize, @intCast(embd_override.?.nb()[1])),
                    ds_offset,
                );
                cur = ggml.add(ctx, cur, ds);
                {
                    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.deepstack_out", .{i}) catch unreachable;
                    name_buf[name.len] = 0;
                    cur.setName(name_buf[0..name.len :0]);
                }
            }
        }

        // Output norm (RMS norm)
        cur = rms_norm.rmsNorm(ctx, cur, w.base.output_norm_weight, p.norm_eps);
        cur.setName("result_norm");

        // lm_head — reference: build_lora_mm(model.output, cur, model.output_s)
        const out_w = w.base.output_weight orelse w.base.token_embd;
        var logits_tensor = ggml.mulMat(ctx, out_w, cur);
        logits_tensor.setName("result_output");
        graph.buildForwardExpand(logits_tensor);
        return logits_tensor;
    }

    /// Media forward adapter (MediaForwardFn signature).
    /// Delegates to forwardWithEmbdOverride with embedding override.
    pub fn mediaForward(
        self: *Qwen3VLModel,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        kv_cache_mgr: ?*kv_cache.KVCache,
        start_pos: i32,
        embd_override: *ggml.Tensor,
        embd_offset: i32,
        causal: bool,
    ) !*ggml.Tensor {
        _ = embd_offset;
        _ = causal;
        return self.forwardWithEmbdOverride(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos, embd_override, 0);
    }

    pub fn buildGraph(
        self: *Qwen3VLModel,
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
        .resetSSMStates = null,
        .setKVCacheContext = null,
        .buildMM = buildMMAdapter,
    };

    fn deinitAdapter(data: *anyopaque, allocator: std.mem.Allocator) void {
        const self = @as(*Qwen3VLModel, @ptrCast(@alignCast(data)));
        self.weights.deinit(allocator);
        self.ctx_weights.deinit();
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
        const self = @as(*Qwen3VLModel, @ptrCast(@alignCast(data)));
        return self.buildGraph(builder, input_tokens, n_tokens, cache, pos);
    }

    fn getParamsAdapter(data: *anyopaque) *const model.ModelParams {
        const self = @as(*Qwen3VLModel, @ptrCast(@alignCast(data)));
        return self.getParams();
    }

    fn buildMMAdapter(
        data: *anyopaque,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input_tokens: *ggml.Tensor,
        n_tokens: i32,
        cache: ?*anyopaque,
        pos: i32,
        embd_override: *ggml.Tensor,
        embd_offset: i32,
        causal: bool,
    ) anyerror!*ggml.Tensor {
        const self = @as(*Qwen3VLModel, @ptrCast(@alignCast(data)));
        const kv_cache_mgr: ?*kv_cache.KVCache = if (cache) |c| @ptrCast(@alignCast(c)) else null;
        return self.mediaForward(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, pos, embd_override, embd_offset, causal);
    }
};

/// Helper: try multiple key prefixes for a u32 value
fn getU32Prefixed(gguf_file: *const gguf.GGUFFile, key_suffix: []const u8) ?u32 {
    const prefixes = [_][]const u8{ "qwen3vl", "qwen3", "llama" };
    var buf: [128]u8 = undefined;
    for (prefixes) |prefix| {
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, key_suffix }) catch continue;
        if (gguf_file.getU32(key)) |v| return v;
    }
    return null;
}

/// Helper: try multiple key prefixes for a f32 value
fn getF32Prefixed(gguf_file: *const gguf.GGUFFile, key_suffix: []const u8) ?f32 {
    const prefixes = [_][]const u8{ "qwen3vl", "qwen3", "llama" };
    var buf: [128]u8 = undefined;
    for (prefixes) |prefix| {
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, key_suffix }) catch continue;
        if (gguf_file.getF32(key)) |v| return v;
    }
    return null;
}

pub fn parseParams(gguf_file: *const gguf.GGUFFile, _: std.mem.Allocator) !Qwen3VLParams {
    var p = Qwen3VLParams{};

    p.base.n_vocab = getU32Prefixed(gguf_file, "vocab_size") orelse blk: {
        if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
            if (val.value_type == .array) break :blk @intCast(val.array_val.len);
        }
        break :blk 0;
    };
    p.base.n_embd = getU32Prefixed(gguf_file, "embedding_length") orelse 0;
    p.base.n_head = getU32Prefixed(gguf_file, "attention.head_count") orelse
        getU32Prefixed(gguf_file, "head_count") orelse 0;
    p.base.n_kv_head = getU32Prefixed(gguf_file, "attention.head_count_kv") orelse
        getU32Prefixed(gguf_file, "head_count_kv") orelse p.base.n_head;
    p.base.n_layer = getU32Prefixed(gguf_file, "block_count") orelse 0;
    p.base.n_ff = getU32Prefixed(gguf_file, "feed_forward_length") orelse 0;
    p.base.n_expert = getU32Prefixed(gguf_file, "expert_count") orelse 0;
    p.base.n_expert_used = getU32Prefixed(gguf_file, "expert_used_count") orelse 0;
    // Read key_length and value_length for head_dim (may differ from n_embd / n_head)
    const key_length = getU32Prefixed(gguf_file, "attention.key_length") orelse 0;
    const value_length = getU32Prefixed(gguf_file, "attention.value_length") orelse 0;
    if (key_length > 0) {
        p.base.n_head_dim = key_length;
    } else if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }
    _ = value_length;

    p.base.max_seq_len = getU32Prefixed(gguf_file, "context_length") orelse 32768;
    p.base.rope_theta = getF32Prefixed(gguf_file, "rope.freq_base") orelse 10000000.0;
    p.base.rope_dim = getU32Prefixed(gguf_file, "rope.dimension_count") orelse p.base.n_head_dim;
    p.base.norm_eps = getF32Prefixed(gguf_file, "attention.layer_norm_rms_epsilon") orelse 1e-6;
    p.base.model_name = gguf_file.getString("general.name") orelse "";
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gpt2";

    // Read rope sections from metadata
    {
        const section_keys = [_][]const u8{
            "qwen3vl.rope.dimension_sections",
            "qwen3.rope.dimension_sections",
            "llama.rope.dimension_sections",
            "qwen3vl.rope.sections",
            "qwen3.rope.sections",
            "llama.rope.sections",
        };
        var found = false;
        for (section_keys) |key| {
            if (gguf_file.getF32Array(key, 4)) |sections| {
                for (sections, 0..) |v, i| p.rope_sections[i] = @intFromFloat(v);
                found = true;
                break;
            }
        }
        if (!found) {
            for (section_keys) |key| {
                if (gguf_file.metadata.get(key)) |val| {
                    if (val.value_type == .array and val.array_val.len >= 4) {
                        for (val.array_val, 0..) |item, i| {
                            if (item.asI32()) |v| {
                                p.rope_sections[i] = v;
                            }
                        }
                        found = true;
                        break;
                    }
                }
            }
        }
        if (!found or (p.rope_sections[0] == 0 and p.rope_sections[1] == 0 and p.rope_sections[2] == 0 and p.rope_sections[3] == 0)) {
            const section_size = @divExact(@as(i32, @intCast(p.base.rope_dim)), @as(i32, 4));
            p.rope_sections = .{ section_size, section_size, section_size, section_size };
        }
    }

    // Read deepstack layers — reference: ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false)
    p.n_deepstack_layers = getU32Prefixed(gguf_file, "n_deepstack_layers") orelse 0;

    log.info("Qwen3VL params: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}, deepstack={d}", .{
        p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_kv_head, p.base.n_layer, p.base.n_ff, p.n_deepstack_layers,
    });
    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required model parameters", .{});
        log.err("  n_vocab={d} n_embd={d} n_head={d} n_layer={d}", .{ p.base.n_vocab, p.base.n_embd, p.base.n_head, p.base.n_layer });
        return error.InvalidModelParams;
    }
    return p;
}

pub fn loadWeights(gguf_file: *const gguf.GGUFFile, ctx: *ggml.Context, params: *const Qwen3VLParams, allocator: std.mem.Allocator) !Qwen3VLWeights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Qwen3VL weights...", .{});
    for (gguf_file.tensors.items, 0..) |t, i| {
        if (i >= 5) break;
        log.debug("  tensor[{d}]: '{s}' dims={any}", .{ i, t.name, t.dims[0..t.n_dims] });
    }
    const token_embd = try weight_loader.findOrCreateTensor(ctx, gguf_file, "token_embd.weight");
    token_embd.setName("token_embd.weight");
    const output_weight = weight_loader.findOrCreateTensor(ctx, gguf_file, "output.weight") catch null;
    if (output_weight) |ow| ow.setName("output.weight");
    const output_norm_weight = try weight_loader.findOrCreateTensor(ctx, gguf_file, "output_norm.weight");
    output_norm_weight.setName("output_norm.weight");
    var layers = try allocator.alloc(LayerWeights, n_layer);
    var layers_loaded: usize = 0;
    errdefer {
        for (0..layers_loaded) |j| allocator.free(layers[j].prefix);
        allocator.free(layers);
    }
    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});

        // Reference uses create_tensor_qkv which creates fused QKV
        // Try fused QKV first, fall back to separate Q/K/V
        const attn_qkv_weight = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_qkv.weight") catch null;
        if (attn_qkv_weight) |qkv| {
            qkv.setName("attn_qkv.weight");
            layers[i] = LayerWeights{
                .prefix = prefix,
                .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
                .ffn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight"),
                .attn_qkv_weight = qkv,
                .attn_q_norm_weight = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q_norm.weight") catch null,
                .attn_k_norm_weight = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k_norm.weight") catch null,
                .attn_output_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight"),
                .ffn_gate_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight"),
                .ffn_up_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight"),
                .ffn_down_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight"),
            };
        } else {
            // Fall back to separate Q/K/V (create_tensor_qkv in reference creates separate Q/K/V tensors)
            log.debug("  Layer {d}: using separate Q/K/V weights", .{i});
            const q_w = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight");
            const k_w = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight");
            const v_w = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight");
            layers[i] = LayerWeights{
                .prefix = prefix,
                .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
                .ffn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight"),
                .attn_q_weight = q_w,
                .attn_k_weight = k_w,
                .attn_v_weight = v_w,
                .attn_q_norm_weight = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q_norm.weight") catch null,
                .attn_k_norm_weight = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k_norm.weight") catch null,
                .attn_output_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight"),
                .ffn_gate_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight"),
                .ffn_up_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight"),
                .ffn_down_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight"),
            };
        }
        layers_loaded = i + 1;
    }
    log.info("All Qwen3VL weights loaded ({d} layers)\n", .{n_layer});
    return Qwen3VLWeights{
        .base = .{ .params = params.base, .token_embd = token_embd, .output_weight = output_weight, .output_norm_weight = output_norm_weight },
        .layers = layers,
    };
}

const testing = std.testing;

test "Qwen3VLParams defaults" {
    const p = Qwen3VLParams{};
    try testing.expectEqual(@as(u32, 0), p.base.n_vocab);
    try testing.expectEqual(@as(u32, 0), p.n_deepstack_layers);
}
