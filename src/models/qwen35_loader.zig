//! Qwen 3.5 参数解析与权重加载
const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const weight_loader = @import("weight_loader");

const QwenParams = @import("./qwen35.zig").QwenParams;
const QwenWeights = @import("./qwen35.zig").QwenWeights;
const LayerWeights = @import("./qwen35.zig").LayerWeights;
const LayerType = @import("./qwen35.zig").LayerType;
const isFullAttentionLayer = @import("./qwen35.zig").isFullAttentionLayer;

const log = std.log.scoped(.qwen35);

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


/// 根据 GGUF 文件中实际张量数据大小估计所需内存
/// 加上 ggml 元数据开销（每个张量 ~256 字节）和 20% 安全余量
pub fn estimateMemSize(gguf_file: *const gguf.GGUFFile) usize {
    const raw_data_size = gguf_file.totalTensorDataSize();
    const n_tensors = gguf_file.tensors.items.len;
    // ggml 内部每个张量需要: ggml_tensor (~256B) + ggml_object (~64B) + 对齐
    // 使用 384 字节/tensor 以确保覆盖
    const overhead: usize = n_tensors * 384;
    const with_overhead = raw_data_size + overhead;
    // 33% 安全余量 + 64MB 固定缓冲
    const total = with_overhead + with_overhead / 3 + 64 * 1024 * 1024;
    log.info("Estimated weights memory: {d} MB (raw: {d} MB, {d} tensors)", .{ total / (1024 * 1024), raw_data_size / (1024 * 1024), n_tensors });
    return total;
}


pub fn loadWeights(gguf_file: *const gguf.GGUFFile, ctx: *ggml.Context, params: *const QwenParams, allocator: std.mem.Allocator) !QwenWeights {
    const n_layer: usize = @intCast(params.base.n_layer);
    log.info("Loading Qwen35 weights...", .{});
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
        const is_full_attn = isFullAttentionLayer(@intCast(i), params.full_attention_interval);
        var lw = LayerWeights{
            .prefix = prefix,
            .layer_type = if (is_full_attn) .full_attention else .ssm,
            .attn_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_norm.weight"),
            .post_attention_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "post_attention_norm.weight"),
            .ffn_norm_weight = undefined,
            .ffn_gate_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_gate.weight"),
            .ffn_up_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight"),
            .ffn_down_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight"),
        };
        lw.ffn_norm_weight = lw.post_attention_norm_weight;
        if (is_full_attn) {
            lw.attn_q_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q.weight");
            lw.attn_k_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k.weight");
            lw.attn_v_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_v.weight");
            lw.attn_output_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_output.weight");
            lw.attn_q_norm_weight = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_q_norm.weight") catch null;
            lw.attn_k_norm_weight = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_k_norm.weight") catch null;
        } else {
            lw.attn_qkv_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_qkv.weight");
            lw.attn_gate_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_gate.weight");
            lw.ssm_conv1d_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ssm_conv1d.weight");
            lw.ssm_a = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ssm_a");
            lw.ssm_dt_bias = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ssm_dt.bias");
            lw.ssm_alpha_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ssm_alpha.weight");
            lw.ssm_beta_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ssm_beta.weight");
            lw.ssm_norm_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ssm_norm.weight");
            lw.ssm_out_weight = try weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ssm_out.weight");
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
