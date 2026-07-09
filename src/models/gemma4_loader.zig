//! Gemma 4 参数解析与权重加载
//! 从 gemma4.zig 拆分，保持文件 ≤600 行。
const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const weight_loader = @import("weight_loader");

const Gemma4Params = @import("./gemma4.zig").Gemma4Params;
const Gemma4Weights = @import("./gemma4.zig").Gemma4Weights;
const LayerWeights = @import("./gemma4.zig").LayerWeights;

const log = std.log.scoped(.gemma4);

pub fn parseParams(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Gemma4Params {
    var p = Gemma4Params{
        .is_swa_layer = try std.ArrayList(bool).initCapacity(allocator, 0),
    };
    errdefer p.is_swa_layer.deinit(allocator);

    p.base.n_vocab = gguf_file.getU32("gemma4.vocab_size") orelse
        gguf_file.getU32("llama.vocab_size") orelse
        blk: {
            if (gguf_file.metadata.get("tokenizer.ggml.tokens")) |val| {
                if (val.value_type == .array) break :blk @intCast(val.array_val.len);
            }
            break :blk 0;
        };
    p.base.n_embd = gguf_file.getU32("gemma4.embedding_length") orelse
        gguf_file.getU32("llama.embedding_length") orelse 0;
    p.base.n_head = gguf_file.getU32("gemma4.attention.head_count") orelse
        gguf_file.getU32("llama.attention.head_count") orelse
        gguf_file.getU32("gemma4.head_count") orelse
        gguf_file.getU32("llama.head_count") orelse 0;

    // head_count_kv is an array in Gemma 4 (per-layer KV heads)
    if ((@constCast(gguf_file)).getU32Array("gemma4.attention.head_count_kv")) |kv_arr| {
        var max_kv: u32 = 0;
        for (kv_arr) |h| {
            if (h > max_kv) max_kv = h;
        }
        p.base.n_kv_head = max_kv;
    } else {
        p.base.n_kv_head = gguf_file.getU32("gemma4.attention.head_count_kv") orelse
            gguf_file.getU32("llama.attention.head_count_kv") orelse
            gguf_file.getU32("gemma4.head_count_kv") orelse
            gguf_file.getU32("llama.head_count_kv") orelse p.base.n_head;
    }

    p.base.n_layer = gguf_file.getU32("gemma4.block_count") orelse
        gguf_file.getU32("llama.block_count") orelse 0;
    p.base.n_ff = gguf_file.getU32("gemma4.feed_forward_length") orelse
        gguf_file.getU32("llama.feed_forward_length") orelse 0;

    // Fallback: if n_ff is still 0, derive from n_embd (Gemma 4 uses 4 * n_embd)
    if (p.base.n_ff == 0 and p.base.n_embd > 0) {
        p.base.n_ff = 4 * p.base.n_embd;
        log.info("n_ff not specified in GGUF, using default 4 * n_embd = {d}", .{p.base.n_ff});
    }

    // Gemma 4 uses explicit key_length/value_length for head_dim
    // n_head_dim_k/n_head_dim_v 使用 SWA 维度（所有层的 K/V norm 维度一致）
    // n_head_dim 保持 full attention 维度（用于 rope_dim 等）
    const key_len = gguf_file.getU32("gemma4.attention.key_length") orelse
        gguf_file.getU32("llama.attention.key_length") orelse 0;
    const key_len_swa = gguf_file.getU32("gemma4.attention.key_length_swa") orelse 0;
    const val_len_swa = gguf_file.getU32("gemma4.attention.value_length_swa") orelse 0;

    if (key_len > 0) {
        p.base.n_head_dim = key_len;
    } else if (p.base.n_head > 0 and p.base.n_embd > 0) {
        p.base.n_head_dim = p.base.n_embd / p.base.n_head;
    }
    // KV cache 使用 SWA 维度（所有层的 K norm 均为 key_length_swa）
    p.base.n_head_dim_k = if (key_len_swa > 0) key_len_swa else p.base.n_head_dim;
    p.base.n_head_dim_v = if (val_len_swa > 0) val_len_swa else p.base.n_head_dim;

    p.base.max_seq_len = gguf_file.getU32("gemma4.context_length") orelse
        gguf_file.getU32("llama.context_length") orelse 32768;
    p.base.rope_theta = gguf_file.getF32("gemma4.rope.freq_base") orelse
        gguf_file.getF32("llama.rope.freq_base") orelse 10000.0;
    p.base.rope_dim = gguf_file.getU32("gemma4.rope.dimension_count") orelse
        gguf_file.getU32("llama.rope.dimension_count") orelse
        @divExact(p.base.n_head_dim, @as(u32, 2));
    // SWA 层可能使用不同的 rope_dim（因为 head_dim 更小）
    p.rope_dim_swa = gguf_file.getU32("gemma4.rope.dimension_count_swa") orelse
        gguf_file.getU32("llama.rope.dimension_count_swa") orelse p.base.rope_dim;

    p.base.norm_eps = gguf_file.getF32("gemma4.attention.layer_norm_rms_epsilon") orelse
        gguf_file.getF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-6;
    p.base.model_name = gguf_file.getString("general.name") orelse "";
    p.base.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "gemma";

    // Gemma 4 特定参数
    p.n_swa = gguf_file.getU32("gemma4.attention.sliding_window") orelse 0;
    p.rope_freq_base_swa = gguf_file.getF32("gemma4.rope.freq_base_swa") orelse p.base.rope_theta;
    p.f_attention_scale = 1.0;

    // 解析 SWA pattern（Gemma 4 使用 bool 数组）
    const n_layer_i64 = @as(usize, @intCast(p.base.n_layer));
    if ((@constCast(gguf_file)).getBoolArray("gemma4.attention.sliding_window_pattern")) |swa_pattern| {
        try p.is_swa_layer.ensureTotalCapacity(allocator, n_layer_i64);
        for (swa_pattern, 0..) |is_swa, idx| {
            if (idx < n_layer_i64) {
                p.is_swa_layer.appendAssumeCapacity(is_swa);
            }
        }
        while (p.is_swa_layer.items.len < n_layer_i64) {
            p.is_swa_layer.appendAssumeCapacity(false);
        }
    } else if (p.n_swa > 0) {
        // 没有显式 pattern，使用周期模式：每 6 层一个全注意力层
        const swa_period: u32 = 6;
        try p.is_swa_layer.ensureTotalCapacity(allocator, n_layer_i64);
        for (0..n_layer_i64) |idx| {
            const is_swa = (idx % swa_period) != 0;
            p.is_swa_layer.appendAssumeCapacity(is_swa);
        }
    } else {
        try p.is_swa_layer.ensureTotalCapacity(allocator, n_layer_i64);
        for (0..n_layer_i64) |_| {
            p.is_swa_layer.appendAssumeCapacity(false);
        }
    }

    // 共享 KV 层
    p.n_kv_shared_layers = gguf_file.getU32("gemma4.attention.shared_kv_layers") orelse 0;
    if (p.base.n_layer > p.n_kv_shared_layers) {
        p.n_layer_kv_from_start = p.base.n_layer - p.n_kv_shared_layers;
    } else {
        p.n_layer_kv_from_start = p.base.n_layer;
    }

    p.final_logit_softcapping = gguf_file.getF32("gemma4.final_logit_softcapping") orelse 0.0;
    p.attn_logit_softcapping = gguf_file.getF32("gemma4.attn_logit_softcapping") orelse 50.0;
    p.n_embd_per_layer = gguf_file.getU32("gemma4.embedding_length_per_layer_input") orelse
        gguf_file.getU32("gemma4.embedding_length_per_layer") orelse 0;

    if (p.base.n_vocab == 0 or p.base.n_embd == 0 or p.base.n_head == 0 or p.base.n_layer == 0) {
        log.err("Missing required Gemma 4 parameters", .{});
        return error.InvalidModelParams;
    }

    log.info("Gemma4: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}, swa={d}, shared_kv={d}, softcap={d}, embd_per_layer={d}", .{
        p.base.n_vocab,            p.base.n_embd,      p.base.n_head, p.base.n_kv_head,
        p.base.n_layer,            p.base.n_ff,        p.n_swa,       p.n_kv_shared_layers,
        p.final_logit_softcapping, p.n_embd_per_layer,
    });

    return p;
}

// ============================================================================
// 权重加载
// ============================================================================

pub fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *Gemma4Params,
    allocator: std.mem.Allocator,
) !Gemma4Weights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Gemma 4 weights ({d} layers)...", .{n_layer});

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

    // Per-layer embedding global weights (only when n_embd_per_layer > 0)
    const per_layer_token_embd: ?*ggml.Tensor = if (params.n_embd_per_layer > 0) blk: {
        const t = findOrCreateTensor(ctx, gguf_file, "per_layer_token_embd.weight") catch null;
        if (t) |tt| tt.setName("per_layer_token_embd.weight");
        break :blk t;
    } else null;
    const per_layer_model_proj: ?*ggml.Tensor = if (params.n_embd_per_layer > 0) blk: {
        const t = findOrCreateTensor(ctx, gguf_file, "per_layer_model_proj.weight") catch null;
        if (t) |tt| tt.setName("per_layer_model_proj.weight");
        break :blk t;
    } else null;
    const per_layer_proj_norm: ?*ggml.Tensor = if (params.n_embd_per_layer > 0) blk: {
        const t = findOrCreateTensor(ctx, gguf_file, "per_layer_proj_norm.weight") catch null;
        if (t) |tt| tt.setName("per_layer_proj_norm.weight");
        break :blk t;
    } else null;

    if (per_layer_token_embd != null) {
        log.info("Gemma4: per-layer embedding enabled (n_embd_per_layer={d})", .{params.n_embd_per_layer});
    }

    var layers = try allocator.alloc(LayerWeights, n_layer);
    var layers_loaded: usize = 0;
    errdefer {
        for (0..layers_loaded) |j| {
            allocator.free(layers[j].prefix);
        }
        allocator.free(layers);
    }

    // Gemma 4: rope_freqs is per-layer (blk.0.rope_freqs.weight), shared
    // across all full-attention layers (llama.cpp uses TENSOR_DUPLICATED).
    // Try multiple possible names for rope_freqs.
    var global_rope_freqs = findOrCreateTensor(ctx, gguf_file, "blk.0.rope_freqs.weight") catch null;
    if (global_rope_freqs == null) {
        global_rope_freqs = findOrCreateTensor(ctx, gguf_file, "rope_freqs.weight") catch null;
    }
    if (global_rope_freqs == null) {
        global_rope_freqs = findOrCreateTensor(ctx, gguf_file, "blk.0.rope_freqs") catch null;
    }
    if (global_rope_freqs) |_| {
        log.info("Gemma4: global_rope_freqs loaded successfully", .{});
    } else {
        log.warn("Gemma4: global_rope_freqs not found, full-attention layers will use standard RoPE", .{});
    }

    // Derive n_ff from first layer's FFN gate weight shape
    var n_ff_derived: i64 = params.base.n_ff;

    for (0..n_layer) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});

        // Detect has_kv: use n_layer_kv_from_start to determine which layers have their own KV.
        // Do NOT rely on attn_k.weight existence in GGUF, as E2B models may store it for all layers.
        const has_kv: bool = i < params.n_layer_kv_from_start;
        const k_weight = if (has_kv)
            loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight") catch null
        else
            null;
        const v_weight = if (has_kv)
            loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight") catch null
        else
            null;

        const q_norm = loadLayerWeight(ctx, gguf_file, prefix, "attn_q_norm.weight") catch |err| {
            log.warn("Layer {d}: missing attn_q_norm.weight: {}", .{ i, err });
            continue;
        };
        const k_norm = if (has_kv)
            loadLayerWeight(ctx, gguf_file, prefix, "attn_k_norm.weight") catch null
        else
            null;

        const out_scale = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "layer_output_scale.weight") catch null;

        // Per-layer embedding weights
        const per_layer_inp_gate: ?*ggml.Tensor = if (params.n_embd_per_layer > 0)
            loadLayerWeight(ctx, gguf_file, prefix, "inp_gate.weight") catch null
        else
            null;
        const per_layer_proj: ?*ggml.Tensor = if (params.n_embd_per_layer > 0)
            loadLayerWeight(ctx, gguf_file, prefix, "proj.weight") catch null
        else
            null;
        const per_layer_post_norm: ?*ggml.Tensor = if (params.n_embd_per_layer > 0)
            loadLayerWeight(ctx, gguf_file, prefix, "post_norm.weight") catch null
        else
            null;

        const ffn_gate = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight");

        // Derive n_ff from first layer
        if (i == 0) {
            n_ff_derived = ffn_gate.ne()[1];
            log.info("Layer 0 FFN gate: [{d}, {d}], n_embd={d}", .{ ffn_gate.ne()[0], ffn_gate.ne()[1], params.base.n_embd });
        }

        // Gemma 4: all full-attention layers share the global rope_freqs
        layers[i] = LayerWeights{
            .prefix = prefix,
            .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
            .ffn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_norm.weight"),
            .attn_q_norm_weight = q_norm,
            .attn_k_norm_weight = if (k_norm) |kn| kn else q_norm,
            .attn_post_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "post_attention_norm.weight"),
            .ffn_post_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "post_ffw_norm.weight"),
            .attn_q_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight"),
            .attn_k_weight = k_weight,
            .attn_v_weight = v_weight,
            .attn_output_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight"),
            .ffn_gate_weight = ffn_gate,
            .ffn_up_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight"),
            .ffn_down_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight"),
            .out_scale = out_scale,
            .rope_freqs = if (!params.is_swa_layer.items[i]) global_rope_freqs else null,
            .per_layer_inp_gate = per_layer_inp_gate,
            .per_layer_proj = per_layer_proj,
            .per_layer_post_norm = per_layer_post_norm,
            .has_kv = has_kv,
        };

        layers_loaded = i + 1;
    }

    // Update n_ff from actual weight shape if metadata was missing
    const n_embd_per_layer = params.n_embd_per_layer;
    const n_kv_layers = blk: {
        var count: u32 = 0;
        for (layers) |l| {
            if (l.has_kv) count += 1;
        }
        break :blk count;
    };

    log.info("Gemma4 weights: {d} layers, {d} with KV, n_ff={d}, per_layer={d}", .{
        n_layer, n_kv_layers, n_ff_derived, n_embd_per_layer,
    });

    return Gemma4Weights{
        .base = .{
            .params = params.base,
            .token_embd = token_embd,
            .output_weight = output_weight,
            .output_norm_weight = output_norm_weight,
        },
        .layers = layers,
        .per_layer_token_embd = per_layer_token_embd,
        .per_layer_model_proj = per_layer_model_proj,
        .per_layer_proj_norm = per_layer_proj_norm,
    };
}

pub fn loadLayerWeight(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, prefix: []const u8, name: []const u8) !*ggml.Tensor {
    var buf: [256]u8 = undefined;
    const full_name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, name });
    buf[full_name.len] = 0;
    return findOrCreateTensor(ctx, gguf_file, buf[0..full_name.len :0]);
}

pub fn findOrCreateTensor(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
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
        if (tensor_bytes.len != tensor_data.len) {
            log.warn("Tensor '{s}' size mismatch: expected {d} bytes, got {d} bytes", .{ name, tensor_bytes.len, tensor_data.len });
        }
        @memcpy(tensor_bytes, tensor_data);

        return tensor;
    }
    return error.TensorNotFound;
}

pub fn estimateMemSize(gguf_file: *const gguf.GGUFFile) usize {
    _ = gguf_file.totalTensorDataSize(); // 保留供调试
    const n_tensors = gguf_file.tensors.items.len;

    // 计算反量化后的内存需求：量化张量需要 f32 空间（4x）
    // 注意：raw_data_size 是 GGUF 中压缩后的数据大小。
    // 对于量化张量，我们需要 f32 的完整大小，而不是压缩后的大小。
    var total_data_size: usize = 0;
    for (gguf_file.tensors.items) |tensor_info| {
        const typ: ggml.Type = @enumFromInt(@intFromEnum(tensor_info.data_type));
        if (typ.isQuantized()) {
            // 量化张量：计算 f32 大小
            var n_elems: u64 = 1;
            for (tensor_info.dims[0..tensor_info.n_dims]) |d| {
                n_elems *= d;
            }
            total_data_size += @as(usize, n_elems) * @sizeOf(f32);
        } else {
            // 非量化张量：使用 GGUF 中的压缩大小
            total_data_size += tensor_info.sizeBytes();
        }
    }

    // ggml 内部每个张量需要: ggml_tensor (~256B) + ggml_object (~64B) + 对齐
    // 使用 512 字节/tensor 以确保覆盖（含对齐填充）
    const overhead: usize = n_tensors * 512;
    const with_overhead = total_data_size + overhead;
    // 50% 安全余量 + 128MB 固定缓冲（反量化后内存需求更大）
    const total = with_overhead + with_overhead / 2 + 128 * 1024 * 1024;

    log.info("Estimated Gemma 4 weights memory: {d} MB (data: {d} MB, overhead: {d} MB, {d} tensors)", .{
        @divTrunc(total, 1024 * 1024),
        @divTrunc(total_data_size, 1024 * 1024),
        @divTrunc(overhead, 1024 * 1024),
        n_tensors,
    });
    return total;
}
