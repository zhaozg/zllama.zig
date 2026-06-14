const std = @import("std");
const ggml = @import("ggml");
const memory = @import("memory");

const log = std.log.scoped(.kv_cache);

// ============================================================================
// KV Cache
// ============================================================================

/// 单层的 KV Cache
pub const LayerCache = struct {
    k: *ggml.Tensor, // [head_dim_k, n_kv_head, max_seq_len]
    v: *ggml.Tensor, // [head_dim_v, n_kv_head, max_seq_len]
    current_len: u32, // 当前已使用的长度
    /// 实际存储在本层的 n_kv_head（可能小于全局 max）
    n_kv_head_actual: u32,
    /// 实际存储在本层的 K head_dim
    head_dim_k_actual: u32,
    /// 实际存储在本层的 V head_dim
    head_dim_v_actual: u32,
};

/// KV Cache 管理器
/// 同时实现了 MemoryContext 接口（通过 toMemoryContext()）
pub const KVCache = struct {
    layers: []LayerCache,
    max_seq_len: u32,
    n_kv_head: u32,
    head_dim: u32,
    head_dim_k: u32,
    head_dim_v: u32,

    /// 初始化 KV Cache
    /// K/V 形状: [head_dim, n_kv_head, max_seq_len] (与 llama.cpp 一致)
    pub fn init(
        ctx: *ggml.Context,
        n_layer: u32,
        n_kv_head: u32,
        head_dim: u32,
        max_seq_len: u32,
        allocator: std.mem.Allocator,
    ) !KVCache {
        return initWithKVDim(ctx, n_layer, n_kv_head, head_dim, head_dim, max_seq_len, allocator);
    }

    /// 初始化 KV Cache，支持 K 和 V 不同 head dim
    pub fn initWithKVDim(
        ctx: *ggml.Context,
        n_layer: u32,
        n_kv_head: u32,
        head_dim_k: u32,
        head_dim_v: u32,
        max_seq_len: u32,
        allocator: std.mem.Allocator,
    ) !KVCache {
        var layers = try allocator.alloc(LayerCache, n_layer);

        for (0..n_layer) |i| {
            // K Cache: [head_dim_k, n_kv_head, max_seq_len]
            const k = try ctx.newTensor3d(.f32, @intCast(head_dim_k), @intCast(n_kv_head), @intCast(max_seq_len));
            {
                var buf: [64]u8 = undefined;
                const slice = try std.fmt.bufPrint(&buf, "cache.k.{d}", .{i});
                buf[slice.len] = 0;
                k.setName(buf[0..slice.len :0]);
            }

            // V Cache: [head_dim_v, n_kv_head, max_seq_len]
            const v = try ctx.newTensor3d(.f32, @intCast(head_dim_v), @intCast(n_kv_head), @intCast(max_seq_len));
            {
                var buf: [64]u8 = undefined;
                const slice = try std.fmt.bufPrint(&buf, "cache.v.{d}", .{i});
                buf[slice.len] = 0;
                v.setName(buf[0..slice.len :0]);
            }

            layers[i] = LayerCache{
                .k = k,
                .v = v,
                .current_len = 0,
                .n_kv_head_actual = n_kv_head,
                .head_dim_k_actual = head_dim_k,
                .head_dim_v_actual = head_dim_v,
            };
        }

        log.info("KV Cache initialized: {d} layers, K:[{d}, {d}, {d}] V:[{d}, {d}, {d}]", .{ n_layer, head_dim_k, n_kv_head, max_seq_len, head_dim_v, n_kv_head, max_seq_len });

        return KVCache{
            .layers = layers,
            .max_seq_len = max_seq_len,
            .n_kv_head = n_kv_head,
            .head_dim = head_dim_k,
            .head_dim_k = head_dim_k,
            .head_dim_v = head_dim_v,
        };
    }

    /// 释放 KV Cache
    pub fn deinit(self: *const KVCache, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    /// 重置所有层的 Cache
    pub fn reset(self: *KVCache) void {
        for (self.layers) |*layer| {
            layer.current_len = 0;
        }
    }

    /// 获取指定层的 K Cache 视图 [head_dim_k_actual, n_kv_head_actual, seq_len]
    /// 对于共享 KV 层（current_len == 0），使用全局最大长度。
    pub fn getKView(self: *KVCache, ctx: *ggml.Context, layer_idx: usize) *ggml.Tensor {
        const layer = &self.layers[layer_idx];
        // 使用全局最大长度而非层自身长度，支持共享 KV 层
        const len: i64 = @intCast(self.currentLen());
        const hdim: i64 = @intCast(layer.head_dim_k_actual);
        const nkv: i64 = @intCast(layer.n_kv_head_actual);

        // 使用父张量的实际 stride（支持 per-layer head_dim 不同的情况）
        const parent_nb = layer.k.nb();
        return ctx.view3d(layer.k, hdim, nkv, len, parent_nb[1], // nb1: 父张量沿 dim 1 的 stride
            parent_nb[2], // nb2: 父张量沿 dim 2 的 stride
            0);
    }

    /// 获取指定层的 V Cache 视图 [head_dim_v_actual, n_kv_head_actual, seq_len]
    /// 对于共享 KV 层（current_len == 0），使用全局最大长度。
    pub fn getVView(self: *KVCache, ctx: *ggml.Context, layer_idx: usize) *ggml.Tensor {
        const layer = &self.layers[layer_idx];
        const len: i64 = @intCast(self.currentLen());
        const hdim: i64 = @intCast(layer.head_dim_v_actual);
        const nkv: i64 = @intCast(layer.n_kv_head_actual);

        const parent_nb = layer.v.nb();
        return ctx.view3d(layer.v, hdim, nkv, len, parent_nb[1], parent_nb[2], 0);
    }

    /// 将新的 K, V 写入 Cache
    /// new_k: [head_dim, n_kv_head, n_tokens]（RoPE 后，permute 前的形状）
    /// new_v: [head_dim, n_kv_head, n_tokens]
    /// 自动适配 per-layer 维度差异，使用父张量 stride 创建视图
    pub fn setKv(
        self: *KVCache,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        layer_idx: usize,
        new_k: *ggml.Tensor,
        new_v: *ggml.Tensor,
        n_tokens: u32,
    ) void {
        const layer = &self.layers[layer_idx];
        const offset = layer.current_len;

        // 使用实际张量维度（支持 per-layer 变化的 n_kv_head/head_dim）
        const actual_hdim_k: i64 = new_k.ne()[0];
        const actual_nkv_k: i64 = new_k.ne()[1];
        const actual_hdim_v: i64 = new_v.ne()[0];
        const actual_nkv_v: i64 = new_v.ne()[1];

        // 首次写入时更新本层的实际维度
        if (offset == 0) {
            layer.n_kv_head_actual = @intCast(actual_nkv_k);
            layer.head_dim_k_actual = @intCast(actual_hdim_k);
            layer.head_dim_v_actual = @intCast(actual_hdim_v);
        }

        // 使用父张量的实际 stride（支持 per-layer head_dim 不同）
        const k_parent_nb = layer.k.nb();
        const v_parent_nb = layer.v.nb();

        // layer.k: [head_dim_k_cache, n_kv_head_cache, max_seq_len]
        const k_dst = ctx.view3d(layer.k, actual_hdim_k, actual_nkv_k, @intCast(n_tokens), k_parent_nb[1], k_parent_nb[2], @intCast(offset * k_parent_nb[2]));
        const k_cpy = ggml.cpy(ctx, new_k, k_dst);
        graph.buildForwardExpand(k_cpy);

        const v_dst = ctx.view3d(layer.v, actual_hdim_v, actual_nkv_v, @intCast(n_tokens), v_parent_nb[1], v_parent_nb[2], @intCast(offset * v_parent_nb[2]));
        const v_cpy = ggml.cpy(ctx, new_v, v_dst);
        graph.buildForwardExpand(v_cpy);

        layer.current_len += n_tokens;
    }

    /// 设置所有层的 cache 长度为指定值。
    /// 用于构建 worst-case 计算图进行 gallocr 预规划。
    /// 调用者负责在预规划完成后恢复原始长度。
    pub fn setAllLengths(self: *KVCache, len: u32) void {
        for (self.layers) |*layer| {
            layer.current_len = len;
        }
    }

    /// 获取所有层当前长度的快照，用于 save/restore。
    pub fn getAllLengths(self: *const KVCache, allocator: std.mem.Allocator) ![]u32 {
        const lens = try allocator.alloc(u32, self.layers.len);
        for (self.layers, 0..) |layer, i| {
            lens[i] = layer.current_len;
        }
        return lens;
    }

    /// 获取当前 Cache 长度（取所有层的最大值）
    pub fn currentLen(self: *KVCache) u32 {
        var max_len: u32 = 0;
        for (self.layers) |layer| {
            if (layer.current_len > max_len) {
                max_len = layer.current_len;
            }
        }
        return max_len;
    }

    // ======================================================================
    // MemoryContext 接口适配
    // ======================================================================

    /// 转换为 MemoryContext 接口
    /// 使得 KVCache 可以通过通用内存接口访问
    pub fn toMemoryContext(self: *KVCache) memory.MemoryContext {
        return memory.MemoryContext{
            .vtable = &memory_vtable,
            .data = @as(*anyopaque, @ptrCast(self)),
        };
    }

    /// MemoryContext 适配器：deinit
    fn memDeinit(data: *anyopaque) void {
        _ = data;
        // KVCache 的 deinit 需要 allocator，由外部管理
    }

    /// MemoryContext 适配器：reset
    fn memReset(data: *anyopaque) void {
        const self = @as(*KVCache, @ptrCast(@alignCast(data)));
        self.reset();
    }

    /// MemoryContext 适配器：get_k
    fn memGetK(data: *anyopaque, layer: usize) *ggml.Tensor {
        const self = @as(*KVCache, @ptrCast(@alignCast(data)));
        return self.layers[layer].k;
    }

    /// MemoryContext 适配器：get_v
    fn memGetV(data: *anyopaque, layer: usize) *ggml.Tensor {
        const self = @as(*KVCache, @ptrCast(@alignCast(data)));
        return self.layers[layer].v;
    }

    /// MemoryContext 适配器：set_kv
    fn memSetKv(
        data: *anyopaque,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        layer: usize,
        new_k: *ggml.Tensor,
        new_v: *ggml.Tensor,
        n_tokens: u32,
    ) void {
        const self = @as(*KVCache, @ptrCast(@alignCast(data)));
        self.setKv(ctx, graph, layer, new_k, new_v, n_tokens);
    }

    /// MemoryContext 适配器：current_len
    fn memCurrentLen(data: *anyopaque) u32 {
        const self = @as(*KVCache, @ptrCast(@alignCast(data)));
        return self.currentLen();
    }

    /// MemoryContext 适配器：n_layers
    fn memNLayers(data: *anyopaque) u32 {
        const self = @as(*KVCache, @ptrCast(@alignCast(data)));
        return @intCast(self.layers.len);
    }

    const memory_vtable = memory.MemoryContext.MemoryVTable{
        .deinit = memDeinit,
        .reset = memReset,
        .get_k = memGetK,
        .get_v = memGetV,
        .set_kv = memSetKv,
        .current_len = memCurrentLen,
        .n_layers = memNLayers,
    };
};

const testing = std.testing;

test "KVCache init" {
    try testing.expectEqual(@as(usize, @sizeOf(LayerCache)), @sizeOf(LayerCache));
}

test "KVCache toMemoryContext" {
    const ctx = try ggml.Context.initNoAlloc(256 * 1024);
    defer ctx.deinit();

    var kv = try KVCache.init(ctx, 2, 4, 32, 64, testing.allocator);
    defer kv.deinit(testing.allocator);

    var mem_ctx = kv.toMemoryContext();

    // 验证接口方法
    try testing.expectEqual(@as(u32, 2), mem_ctx.nLayers());
    try testing.expectEqual(@as(u32, 0), mem_ctx.currentLen());

    const k = mem_ctx.getK(0);
    const v = mem_ctx.getV(0);
    try testing.expect(k != undefined);
    try testing.expect(v != undefined);

    // 测试 reset
    mem_ctx.reset();
    try testing.expectEqual(@as(u32, 0), mem_ctx.currentLen());
}

test "KVCache basic lifecycle" {
    const ctx = try ggml.Context.initNoAlloc(256 * 1024);
    defer ctx.deinit();

    var kv = try KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    try testing.expectEqual(@as(u32, 0), kv.currentLen());

    // 验证 layers 初始化
    try testing.expectEqual(@as(u32, 0), kv.layers[0].current_len);
    try testing.expect(kv.layers[0].k != undefined);
    try testing.expect(kv.layers[0].v != undefined);

    // 验证 per-layer 维度
    try testing.expectEqual(@as(u32, 2), kv.layers[0].n_kv_head_actual);
    try testing.expectEqual(@as(u32, 16), kv.layers[0].head_dim_k_actual);

    // 测试 reset
    kv.reset();
    try testing.expectEqual(@as(u32, 0), kv.currentLen());
}
