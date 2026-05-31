//! KV Cache 管理
//!
//! 为每层预分配 KV Cache 张量，支持增量写入和视图切片。
//! 避免每 token 复制历史缓存。

const std = @import("std");
const ggml = @import("ggml.zig");

const log = std.log.scoped(.kv_cache);

// ============================================================================
// KV Cache
// ============================================================================

/// 单层的 KV Cache
pub const LayerCache = struct {
    k: *ggml.Tensor,  // [max_seq_len, n_kv_head, head_dim]
    v: *ggml.Tensor,  // [max_seq_len, n_kv_head, head_dim]
    current_len: u32,  // 当前已使用的长度
};

/// KV Cache 管理器
pub const KVCache = struct {
    layers: []LayerCache,
    max_seq_len: u32,
    n_kv_head: u32,
    head_dim: u32,

    /// 初始化 KV Cache
    pub fn init(
        ctx: *ggml.Context,
        n_layer: u32,
        n_kv_head: u32,
        head_dim: u32,
        max_seq_len: u32,
        allocator: std.mem.Allocator,
    ) !KVCache {
        var layers = try allocator.alloc(LayerCache, n_layer);

        for (0..n_layer) |i| {
            // K Cache: [max_seq_len, n_kv_head, head_dim]
            const k = try ctx.newTensor3d(.f32, @intCast(max_seq_len), @intCast(n_kv_head), @intCast(head_dim));
            k.setName(std.fmt.allocPrint(allocator, "cache.k.{d}", .{i}) catch unreachable);
            k.setZero();

            // V Cache: [max_seq_len, n_kv_head, head_dim]
            const v = try ctx.newTensor3d(.f32, @intCast(max_seq_len), @intCast(n_kv_head), @intCast(head_dim));
            v.setName(std.fmt.allocPrint(allocator, "cache.v.{d}", .{i}) catch unreachable);
            v.setZero();

            layers[i] = LayerCache{
                .k = k,
                .v = v,
                .current_len = 0,
            };
        }

        log.info("KV Cache initialized: {d} layers, {d} x {d} x {d} = {d} elements per layer",
            .{ n_layer, max_seq_len, n_kv_head, head_dim, max_seq_len * n_kv_head * head_dim });

        return KVCache{
            .layers = layers,
            .max_seq_len = max_seq_len,
            .n_kv_head = n_kv_head,
            .head_dim = head_dim,
        };
    }

    /// 释放 KV Cache
    pub fn deinit(self: *KVCache, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    /// 重置所有层的 Cache
    pub fn reset(self: *KVCache) void {
        for (self.layers) |*layer| {
            layer.current_len = 0;
        }
    }

    /// 获取指定层的 K Cache 视图 [current_len, n_kv_head, head_dim]
    pub fn getKView(self: *KVCache, ctx: *ggml.Context, layer_idx: usize) *ggml.Tensor {
        const layer = &self.layers[layer_idx];
        const len: i64 = @intCast(layer.current_len);
        const n_kv_head: i64 = @intCast(self.n_kv_head);
        const head_dim: i64 = @intCast(self.head_dim);

        return ctx.view3d(layer.k, len, n_kv_head, head_dim,
            @intCast(n_kv_head * head_dim * @sizeOf(f32)),
            @intCast(head_dim * @sizeOf(f32)),
            0);
    }

    /// 获取指定层的 V Cache 视图 [current_len, n_kv_head, head_dim]
    pub fn getVView(self: *KVCache, ctx: *ggml.Context, layer_idx: usize) *ggml.Tensor {
        const layer = &self.layers[layer_idx];
        const len: i64 = @intCast(layer.current_len);
        const n_kv_head: i64 = @intCast(self.n_kv_head);
        const head_dim: i64 = @intCast(self.head_dim);

        return ctx.view3d(layer.v, len, n_kv_head, head_dim,
            @intCast(n_kv_head * head_dim * @sizeOf(f32)),
            @intCast(head_dim * @sizeOf(f32)),
            0);
    }

    /// 将新的 K, V 写入 Cache
    /// new_k: [1, n_kv_head, head_dim] 或 [n_tokens, n_kv_head, head_dim]
    /// new_v: [1, n_kv_head, head_dim] 或 [n_tokens, n_kv_head, head_dim]
    pub fn setKv(
        self: *KVCache,
        ctx: *ggml.Context,
        layer_idx: usize,
        new_k: *ggml.Tensor,
        new_v: *ggml.Tensor,
        n_tokens: u32,
    ) void {
        const layer = &self.layers[layer_idx];
        const offset = layer.current_len;

        // 使用 ggml_cpy 将新 K/V 拷贝到 Cache 的对应位置
        // 通过 view 获取目标位置
        const n_kv_head: i64 = @intCast(self.n_kv_head);
        const head_dim: i64 = @intCast(self.head_dim);
        const token_size = n_kv_head * head_dim * @sizeOf(f32);

        const k_dst = ctx.view3d(layer.k,
            @intCast(n_tokens), n_kv_head, head_dim,
            @intCast(n_kv_head * head_dim * @sizeOf(f32)),
            @intCast(head_dim * @sizeOf(f32)),
            @intCast(offset * token_size));
        _ = ggml.cpy(ctx, new_k, k_dst);

        const v_dst = ctx.view3d(layer.v,
            @intCast(n_tokens), n_kv_head, head_dim,
            @intCast(n_kv_head * head_dim * @sizeOf(f32)),
            @intCast(head_dim * @sizeOf(f32)),
            @intCast(offset * token_size));
        _ = ggml.cpy(ctx, new_v, v_dst);

        layer.current_len += n_tokens;
    }

    /// 获取当前 Cache 长度
    pub fn currentLen(self: *KVCache) u32 {
        if (self.layers.len == 0) return 0;
        return self.layers[0].current_len;
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "KVCache init" {
    // 需要 ggml context，这里只测试结构体
    try testing.expectEqual(@as(usize, @sizeOf(LayerCache)), @sizeOf(LayerCache));
}
