//! 内存抽象接口
//!
//! 定义 MemoryContext 接口，支持多种内存类型：
//! - KV Cache（标准注意力）
//! - Recurrent State（SSM/DeltaNet）
//! - Hybrid（混合架构）
//!
//! 参考 llama.cpp 的 llama_memory_context_i 接口设计。

const std = @import("std");
const ggml = @import("../ggml.zig");

/// 内存参数
pub const MemoryParams = struct {
    type_k: ggml.Type = .f32,
    type_v: ggml.Type = .f32,
    max_seq_len: u32 = 2048,
    n_layer: u32 = 0,
    n_kv_head: u32 = 0,
    head_dim: u32 = 0,
};

/// 内存状态码
pub const MemoryStatus = enum(u32) {
    success = 0,
    no_update = 1,
    failed_prepare = 2,
    failed_compute = 3,
};

/// 内存上下文接口（虚表模式）
/// 每个内存类型实现此接口，通过函数指针表分发。
pub const MemoryContext = struct {
    vtable: *const MemoryVTable,
    data: *anyopaque,

    pub const MemoryVTable = struct {
        deinit: *const fn (data: *anyopaque) void,
        reset: *const fn (data: *anyopaque) void,
        get_k: *const fn (data: *anyopaque, layer: usize) *ggml.Tensor,
        get_v: *const fn (data: *anyopaque, layer: usize) *ggml.Tensor,
        set_kv: *const fn (data: *anyopaque, ctx: *ggml.Context, graph: *ggml.CGraph, layer: usize, new_k: *ggml.Tensor, new_v: *ggml.Tensor, n_tokens: u32) void,
        current_len: *const fn (data: *anyopaque) u32,
        n_layers: *const fn (data: *anyopaque) u32,
    };

    pub fn deinit(self: *MemoryContext) void {
        self.vtable.deinit(self.data);
    }

    pub fn reset(self: *MemoryContext) void {
        self.vtable.reset(self.data);
    }

    pub fn getK(self: *MemoryContext, layer: usize) *ggml.Tensor {
        return self.vtable.get_k(self.data, layer);
    }

    pub fn getV(self: *MemoryContext, layer: usize) *ggml.Tensor {
        return self.vtable.get_v(self.data, layer);
    }

    pub fn setKv(self: *MemoryContext, ctx: *ggml.Context, graph: *ggml.CGraph, layer: usize, new_k: *ggml.Tensor, new_v: *ggml.Tensor, n_tokens: u32) void {
        self.vtable.set_kv(self.data, ctx, graph, layer, new_k, new_v, n_tokens);
    }

    pub fn currentLen(self: *MemoryContext) u32 {
        return self.vtable.current_len(self.data);
    }

    pub fn nLayers(self: *MemoryContext) u32 {
        return self.vtable.n_layers(self.data);
    }
};

/// 标准 KV Cache 实现
pub const KVCacheMemory = struct {
    layers: []struct {
        k: *ggml.Tensor,
        v: *ggml.Tensor,
        current_len: u32,
    },
    max_seq_len: u32,
    n_kv_head: u32,
    head_dim: u32,

    pub fn init(
        ctx: *ggml.Context,
        n_layer: u32,
        n_kv_head: u32,
        head_dim: u32,
        max_seq_len: u32,
        allocator: std.mem.Allocator,
    ) !KVCacheMemory {
        var layers = try allocator.alloc(struct {
            k: *ggml.Tensor,
            v: *ggml.Tensor,
            current_len: u32,
        }, n_layer);

        for (0..n_layer) |i| {
            const k = try ctx.newTensor3d(.f32, @intCast(head_dim), @intCast(n_kv_head), @intCast(max_seq_len));
            const v = try ctx.newTensor3d(.f32, @intCast(head_dim), @intCast(n_kv_head), @intCast(max_seq_len));
            layers[i] = .{
                .k = k,
                .v = v,
                .current_len = 0,
            };
        }

        return KVCacheMemory{
            .layers = layers,
            .max_seq_len = max_seq_len,
            .n_kv_head = n_kv_head,
            .head_dim = head_dim,
        };
    }

    pub fn deinit(data: *anyopaque) void {
        const self = @as(*KVCacheMemory, @ptrCast(@alignCast(data)));
        // layers 由调用者释放
        _ = self;
    }

    pub fn reset(data: *anyopaque) void {
        const self = @as(*KVCacheMemory, @ptrCast(@alignCast(data)));
        for (self.layers) |*layer| {
            layer.current_len = 0;
        }
    }

    pub fn getK(data: *anyopaque, layer: usize) *ggml.Tensor {
        const self = @as(*KVCacheMemory, @ptrCast(@alignCast(data)));
        return self.layers[layer].k;
    }

    pub fn getV(data: *anyopaque, layer: usize) *ggml.Tensor {
        const self = @as(*KVCacheMemory, @ptrCast(@alignCast(data)));
        return self.layers[layer].v;
    }

    pub fn setKv(data: *anyopaque, ctx: *ggml.Context, graph: *ggml.CGraph, layer: usize, new_k: *ggml.Tensor, new_v: *ggml.Tensor, n_tokens: u32) void {
        const self = @as(*KVCacheMemory, @ptrCast(@alignCast(data)));
        const layer_data = &self.layers[layer];
        const offset = layer_data.current_len;

        const hdim: i64 = @intCast(self.head_dim);
        const nkv: i64 = @intCast(self.n_kv_head);
        const max_len: i64 = @intCast(self.max_seq_len);

        // 写入 K
        const k_dst = ctx.view3d(layer_data.k, hdim, nkv, @intCast(n_tokens),
            @intCast(nkv * @sizeOf(f32)),
            @intCast(max_len * @sizeOf(f32)),
            @intCast(offset * @sizeOf(f32)));
        const k_cpy = ggml.cpy(ctx, new_k, k_dst);
        graph.buildForwardExpand(k_cpy);

        // 写入 V
        const v_dst = ctx.view3d(layer_data.v, hdim, nkv, @intCast(n_tokens),
            @intCast(nkv * @sizeOf(f32)),
            @intCast(max_len * @sizeOf(f32)),
            @intCast(offset * @sizeOf(f32)));
        const v_cpy = ggml.cpy(ctx, new_v, v_dst);
        graph.buildForwardExpand(v_cpy);

        layer_data.current_len += n_tokens;
    }

    pub fn currentLen(data: *anyopaque) u32 {
        const self = @as(*KVCacheMemory, @ptrCast(@alignCast(data)));
        var max_len: u32 = 0;
        for (self.layers) |layer| {
            if (layer.current_len > max_len) max_len = layer.current_len;
        }
        return max_len;
    }

    pub fn nLayers(data: *anyopaque) u32 {
        const self = @as(*KVCacheMemory, @ptrCast(@alignCast(data)));
        return @intCast(self.layers.len);
    }

    /// 创建 MemoryContext 包装
    pub fn toMemoryContext(self: *KVCacheMemory) MemoryContext {
        return MemoryContext{
            .vtable = &vtable,
            .data = @as(*anyopaque, @ptrCast(self)),
        };
    }

    const vtable = MemoryContext.MemoryVTable{
        .deinit = deinit,
        .reset = reset,
        .get_k = getK,
        .get_v = getV,
        .set_kv = setKv,
        .current_len = currentLen,
        .n_layers = nLayers,
    };
};

const testing = std.testing;

test "MemoryParams defaults" {
    const p = MemoryParams{};
    try testing.expectEqual(@as(u32, 2048), p.max_seq_len);
}

test "MemoryVTable size" {
    try testing.expectEqual(@as(usize, @sizeOf(MemoryContext.MemoryVTable)), @sizeOf(MemoryContext.MemoryVTable));
}
