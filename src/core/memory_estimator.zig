//! 自适应内存估算器
//!
//! 根据模型参数动态计算所需内存大小。
//! 替代硬编码的 estimateKVCacheSize / estimateGraphSize 函数。
//!
//! 支持三种估算模式：
//! - KV Cache: 根据 n_layer, n_kv_head, head_dim, max_seq_len 计算
//! - Graph: 根据 n_layer, n_embd, max_seq_len 计算图元数据大小
//! - Inc: 根据 n_layer, n_embd 计算增量解码图大小
//!
//! 设计原则：
//! - 精确估算：根据模型参数动态计算，而非硬编码
//! - 自适应：根据 max_seq_len 自动调整初始分配大小
//! - 安全余量：包含元数据开销、对齐开销和安全余量
//!
//! 参考：llama.cpp llama_context 内存管理

const std = @import("std");
const model_if = @import("model");

const log = std.log.scoped(.core_mem_estimator);

/// 内存估算器
///
/// 根据模型参数动态计算所需内存大小。
/// 替代硬编码的 estimateKVCacheSize / estimateGraphSize 函数。
pub const MemoryEstimator = struct {
    /// 模型参数引用
    params: *const model_if.ModelParams,

    /// 创建估算器
    pub fn init(params: *const model_if.ModelParams) MemoryEstimator {
        return .{ .params = params };
    }

    /// 估算 KV Cache 所需内存大小。
    ///
    /// KV cache: 2 (K/V) × n_layer × n_kv_head × head_dim × max_seq_len × sizeof(f32)
    /// 加上 ggml 元数据开销和对齐。
    ///
    /// 使用 n_head_dim_k / n_head_dim_v 如果可用（Gemma 3 等模型），
    /// 否则回退到 n_head_dim。
    pub fn estimateKVCache(self: *const MemoryEstimator) usize {
        const p = self.params;
        const hdim_k = @max(p.n_head_dim, p.n_head_dim_k);
        const hdim_v = if (p.n_head_dim_v > 0) @max(p.n_head_dim, p.n_head_dim_v) else hdim_k;

        // K cache: n_layer × n_kv_head × head_dim_k × max_seq_len × sizeof(f32)
        const k_per_layer: usize = @as(usize, @intCast(p.n_kv_head)) *
            @as(usize, @intCast(hdim_k)) *
            @as(usize, @intCast(p.max_seq_len)) *
            @sizeOf(f32);

        // V cache: n_layer × n_kv_head × head_dim_v × max_seq_len × sizeof(f32)
        const v_per_layer: usize = @as(usize, @intCast(p.n_kv_head)) *
            @as(usize, @intCast(hdim_v)) *
            @as(usize, @intCast(p.max_seq_len)) *
            @sizeOf(f32);

        const kv_total = (k_per_layer + v_per_layer) * @as(usize, @intCast(p.n_layer));

        // ggml 元数据开销：每个张量 ~416 字节，每层 2 个张量（K/V）
        const n_tensors = @as(usize, @intCast(p.n_layer)) * 2;
        const metadata_overhead = n_tensors * 512; // 每个张量 ~512 字节元数据

        // 对齐开销：~10%
        const alignment_overhead = kv_total / 10;

        // 安全余量：64 MB
        const safety_margin: usize = 64 * 1024 * 1024;

        return kv_total + metadata_overhead + alignment_overhead + safety_margin;
    }

    /// 估算图上下文所需内存大小（no_alloc 模式）。
    ///
    /// 图上下文只存储张量元数据（tensor descriptors）和图节点，
    /// 不存储张量数据。每个 tensor descriptor ~416 字节。
    ///
    /// 对于大模型（n_layer=62, n_embd=8192）的完整 prefill 图，
    /// 每层约产生 50-200 个中间张量，总计 ~10K-30K 个张量。
    /// 每个张量 ~416 字节元数据，加上图节点 ~128 字节。
    ///
    /// 多模态视觉图额外增加 ~2K-10K 个张量。
    pub fn estimateGraph(self: *const MemoryEstimator) usize {
        const p = self.params;

        // 每层产生的中间张量数（估算）
        // 注意力层：Q/K/V/输出 ~10 个
        // FFN 层：gate/up/down ~8 个
        // 归一化层：~4 个
        // 总计每层 ~22 个中间张量
        const tensors_per_layer: usize = 22;

        // 输入/输出张量
        const io_tensors: usize = 10;

        // 总张量数
        const total_tensors = io_tensors + @as(usize, @intCast(p.n_layer)) * tensors_per_layer;

        // 每个 tensor descriptor ~416 字节
        const tensor_metadata = total_tensors * 416;

        // 图节点：每个操作 ~128 字节
        // 每个张量对应约 1.5 个操作（创建 + 可能的视图）
        const graph_nodes = total_tensors + total_tensors / 2;
        const graph_metadata = graph_nodes * 128;

        // 多模态视觉图额外开销（如果适用）
        // 视觉编码器图可能增加 ~5K 个额外张量
        const vision_overhead: usize = 5000 * 416;

        // 安全余量：50%
        const total = tensor_metadata + graph_metadata + vision_overhead;
        return total + total / 2;
    }

    /// 估算增量解码图所需内存大小。
    ///
    /// 增量解码图比 prefill 图小得多（单 token），
    /// 但需要为每层分配中间张量。
    pub fn estimateInc(self: *const MemoryEstimator) usize {
        const p = self.params;

        // 单 token 解码：每层约 15 个中间张量
        const tensors_per_layer: usize = 15;
        const total_tensors = 10 + @as(usize, @intCast(p.n_layer)) * tensors_per_layer;

        // 每个 tensor descriptor ~416 字节
        const tensor_metadata = total_tensors * 416;

        // 图节点：每个操作 ~128 字节
        const graph_nodes = total_tensors + total_tensors / 2;
        const graph_metadata = graph_nodes * 128;

        // 安全余量：100%（增量解码图可能因模型架构而异）
        return (tensor_metadata + graph_metadata) * 2;
    }

    /// 估算多模态视觉编码器图所需内存大小。
    ///
    /// 视觉编码器图可能非常大（ViT + 投影器），
    /// 取决于图像分辨率和模型架构。
    pub fn estimateVision(self: *const MemoryEstimator, image_tokens: u32) usize {
        const p = self.params;

        // 视觉编码器每图像 token 约产生 5 个中间张量
        const tensors_from_image = @as(usize, @intCast(image_tokens)) * 5;

        // LLM 层处理视觉 token 的中间张量
        const tensors_per_layer: usize = 10;
        const llm_tensors = @as(usize, @intCast(p.n_layer)) * tensors_per_layer;

        const total_tensors = tensors_from_image + llm_tensors + 20; // +20 for I/O

        const tensor_metadata = total_tensors * 416;
        const graph_nodes = total_tensors + total_tensors / 2;
        const graph_metadata = graph_nodes * 128;

        // 安全余量：50%
        const total = tensor_metadata + graph_metadata;
        return total + total / 2;
    }

    /// 获取推荐的最小 KV Cache 大小（基于 max_seq_len 的 25%）。
    /// 用于初始分配，后续由 GrowableContext 自动扩容。
    pub fn estimateMinKVCache(self: *const MemoryEstimator) usize {
        const p = self.params;
        const full_size = self.estimateKVCache();

        // 如果 max_seq_len 很大（>32K），初始分配 25%
        // 否则分配完整大小
        if (p.max_seq_len > 32768) {
            return full_size / 4;
        }
        return full_size;
    }

    /// 获取推荐的最小图大小（基于完整估算的 50%）。
    pub fn estimateMinGraph(self: *const MemoryEstimator) usize {
        return self.estimateGraph() / 2;
    }

    /// 格式化输出估算结果
    pub fn format(self: *const MemoryEstimator, writer: anytype) !void {
        const kv = self.estimateKVCache();
        const graph = self.estimateGraph();
        const inc = self.estimateInc();
        const min_kv = self.estimateMinKVCache();
        const min_graph = self.estimateMinGraph();

        try writer.print("MemoryEstimator for {s}:\n", .{@tagName(self.params.tokenizer_name)});
        try writer.print("  KV Cache  (full): {d:.1} MB\n", .{@as(f64, @floatFromInt(kv)) / (1024.0 * 1024.0)});
        try writer.print("  KV Cache  (min):  {d:.1} MB\n", .{@as(f64, @floatFromInt(min_kv)) / (1024.0 * 1024.0)});
        try writer.print("  Graph     (full): {d:.1} MB\n", .{@as(f64, @floatFromInt(graph)) / (1024.0 * 1024.0)});
        try writer.print("  Graph     (min):  {d:.1} MB\n", .{@as(f64, @floatFromInt(min_graph)) / (1024.0 * 1024.0)});
        try writer.print("  Inc Decode:      {d:.1} MB\n", .{@as(f64, @floatFromInt(inc)) / (1024.0 * 1024.0)});
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "MemoryEstimator init" {
    const params = model_if.ModelParams{
        .n_vocab = 32000,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 8,
        .n_layer = 32,
        .n_ff = 11008,
        .max_seq_len = 4096,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };
    const estimator = MemoryEstimator.init(&params);
    try testing.expect(estimator.estimateKVCache() > 0);
    try testing.expect(estimator.estimateGraph() > 0);
    try testing.expect(estimator.estimateInc() > 0);
}

test "MemoryEstimator estimateKVCache" {
    // Llama 3 8B: n_layer=32, n_kv_head=8, n_head_dim=128, max_seq_len=8192
    const params = model_if.ModelParams{
        .n_vocab = 128256,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 8,
        .n_layer = 32,
        .n_ff = 14336,
        .max_seq_len = 8192,
        .rope_theta = 500000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };
    const estimator = MemoryEstimator.init(&params);

    // K: 32 * 8 * 128 * 8192 * 4 = 1,073,741,824
    // V: 32 * 8 * 128 * 8192 * 4 = 1,073,741,824
    // Total data: ~2 GB
    const kv_size = estimator.estimateKVCache();
    try testing.expect(kv_size > 2 * 1024 * 1024 * 1024); // > 2 GB
    try testing.expect(kv_size < 4 * 1024 * 1024 * 1024); // < 4 GB (with overhead)
}

test "MemoryEstimator estimateKVCache with separate K/V dims" {
    // Gemma 3: uses key_length / value_length
    const params = model_if.ModelParams{
        .n_vocab = 256000,
        .n_embd = 2560,
        .n_head = 20,
        .n_head_dim = 256,
        .n_head_dim_k = 256,
        .n_head_dim_v = 256,
        .n_kv_head = 4,
        .n_layer = 42,
        .n_ff = 10240,
        .max_seq_len = 32768,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-6,
    };
    const estimator = MemoryEstimator.init(&params);
    const kv_size = estimator.estimateKVCache();
    try testing.expect(kv_size > 0);
}

test "MemoryEstimator estimateGraph" {
    // Large model: n_layer=62, n_embd=8192
    const params = model_if.ModelParams{
        .n_vocab = 128256,
        .n_embd = 8192,
        .n_head = 64,
        .n_head_dim = 128,
        .n_kv_head = 8,
        .n_layer = 62,
        .n_ff = 28672,
        .max_seq_len = 131072,
        .rope_theta = 500000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };
    const estimator = MemoryEstimator.init(&params);
    const graph_size = estimator.estimateGraph();
    // Graph context is no_alloc, so it should be much smaller than KV cache
    try testing.expect(graph_size > 0);
    try testing.expect(graph_size < 4 * 1024 * 1024 * 1024); // < 4 GB
}

test "MemoryEstimator estimateInc" {
    const params = model_if.ModelParams{
        .n_vocab = 32000,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 8,
        .n_layer = 32,
        .n_ff = 11008,
        .max_seq_len = 4096,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };
    const estimator = MemoryEstimator.init(&params);
    const inc_size = estimator.estimateInc();
    // Inc decode graph is small (single token)
    try testing.expect(inc_size > 0);
    try testing.expect(inc_size < 512 * 1024 * 1024); // < 512 MB
}

test "MemoryEstimator estimateVision" {
    const params = model_if.ModelParams{
        .n_vocab = 256000,
        .n_embd = 3584,
        .n_head = 16,
        .n_head_dim = 256,
        .n_kv_head = 8,
        .n_layer = 40,
        .n_ff = 14336,
        .max_seq_len = 8192,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-6,
    };
    const estimator = MemoryEstimator.init(&params);
    // 256 image tokens
    const vision_size = estimator.estimateVision(256);
    try testing.expect(vision_size > 0);
    // Vision graph should be larger than inc decode
    const inc_size = estimator.estimateInc();
    try testing.expect(vision_size > inc_size);
}

test "MemoryEstimator estimateMinKVCache" {
    // Large max_seq_len: should use 25%
    const params_large = model_if.ModelParams{
        .n_vocab = 32000,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 8,
        .n_layer = 32,
        .n_ff = 11008,
        .max_seq_len = 131072,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };
    const estimator_large = MemoryEstimator.init(&params_large);
    const min_kv = estimator_large.estimateMinKVCache();
    const full_kv = estimator_large.estimateKVCache();
    try testing.expect(min_kv < full_kv);
    try testing.expect(min_kv >= full_kv / 4);

    // Small max_seq_len: should use full size
    const params_small = model_if.ModelParams{
        .n_vocab = 32000,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 8,
        .n_layer = 32,
        .n_ff = 11008,
        .max_seq_len = 4096,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };
    const estimator_small = MemoryEstimator.init(&params_small);
    try testing.expectEqual(estimator_small.estimateKVCache(), estimator_small.estimateMinKVCache());
}

test "MemoryEstimator estimateMinGraph" {
    const params = model_if.ModelParams{
        .n_vocab = 32000,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 8,
        .n_layer = 32,
        .n_ff = 11008,
        .max_seq_len = 4096,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };
    const estimator = MemoryEstimator.init(&params);
    const min_graph = estimator.estimateMinGraph();
    const full_graph = estimator.estimateGraph();
    try testing.expect(min_graph < full_graph);
    try testing.expect(min_graph >= full_graph / 2);
}
