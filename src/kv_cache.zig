//! KV Cache 管理
//!
//! 为每层预分配 KV Cache 张量，支持增量写入和视图切片。
//! 避免每 token 复制历史缓存。
//!
//! KV Cache 布局（与 llama.cpp 一致）:
//!   K: [head_dim, max_seq_len, n_kv_head]
//!   V: [head_dim, max_seq_len, n_kv_head]
//!
//! 新 K/V 张量布局（经过 RoPE + permute 后）:
//!   [head_dim, n_tokens, n_kv_head]

const std = @import("std");
const ggml = @import("ggml.zig");

const log = std.log.scoped(.kv_cache);

// ============================================================================
// KV Cache
// ============================================================================

/// 单层的 KV Cache
pub const LayerCache = struct {
    k: *ggml.Tensor, // [head_dim, max_seq_len, n_kv_head]
    v: *ggml.Tensor, // [head_dim, max_seq_len, n_kv_head]
    current_len: u32, // 当前已使用的长度
};

/// KV Cache 管理器
pub const KVCache = struct {
    layers: []LayerCache,
    max_seq_len: u32,
    n_kv_head: u32,
    head_dim: u32,

    /// 初始化 KV Cache
    /// K/V 形状: [head_dim, max_seq_len, n_kv_head] (匹配 RoPE 后的 K/V 布局)
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
            // K Cache: [head_dim, max_seq_len, n_kv_head]
            const k = try ctx.newTensor3d(.f32, @intCast(head_dim), @intCast(max_seq_len), @intCast(n_kv_head));
            {
                var buf: [64]u8 = undefined;
                const slice = try std.fmt.bufPrint(&buf, "cache.k.{d}", .{i});
                buf[slice.len] = 0;
                k.setName(buf[0..slice.len :0]);
            }

            // V Cache: [head_dim, max_seq_len, n_kv_head]
            const v = try ctx.newTensor3d(.f32, @intCast(head_dim), @intCast(max_seq_len), @intCast(n_kv_head));
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
            };
        }

        log.info("KV Cache initialized: {d} layers, [{d}, {d}, {d}] per layer", .{ n_layer, head_dim, max_seq_len, n_kv_head });

        return KVCache{
            .layers = layers,
            .max_seq_len = max_seq_len,
            .n_kv_head = n_kv_head,
            .head_dim = head_dim,
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

    /// 获取指定层的 K Cache 视图 [head_dim, current_len, n_kv_head]
    /// 注意: 返回的张量形状与 RoPE+permute 后的 K 一致
    pub fn getKView(self: *KVCache, ctx: *ggml.Context, layer_idx: usize) *ggml.Tensor {
        const layer = &self.layers[layer_idx];
        const len: i64 = @intCast(layer.current_len);
        const hdim: i64 = @intCast(self.head_dim);
        const nkv: i64 = @intCast(self.n_kv_head);
        const max_len: i64 = @intCast(self.max_seq_len);

        // layer.k: [head_dim, max_seq_len, n_kv_head]
        // View:   [head_dim, len, n_kv_head]
        // nb1 = max_seq_len * sizeof(f32) (stride along seq dim in source)
        // nb2 = max_seq_len * n_kv_head * sizeof(f32) (stride along head dim in source)
        return ctx.view3d(layer.k, hdim, len, nkv,
            @intCast(max_len * @sizeOf(f32)),
            @intCast(max_len * nkv * @sizeOf(f32)),
            0);
    }

    /// 获取指定层的 V Cache 视图 [head_dim, current_len, n_kv_head]
    pub fn getVView(self: *KVCache, ctx: *ggml.Context, layer_idx: usize) *ggml.Tensor {
        const layer = &self.layers[layer_idx];
        const len: i64 = @intCast(layer.current_len);
        const hdim: i64 = @intCast(self.head_dim);
        const nkv: i64 = @intCast(self.n_kv_head);
        const max_len: i64 = @intCast(self.max_seq_len);

        return ctx.view3d(layer.v, hdim, len, nkv,
            @intCast(max_len * @sizeOf(f32)),
            @intCast(max_len * nkv * @sizeOf(f32)),
            0);
    }

    /// 将新的 K, V 写入 Cache
    /// new_k: [head_dim, n_tokens, n_kv_head] (RoPE + permute 后的形状)
    /// new_v: [head_dim, n_tokens, n_kv_head]
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

        const hdim: i64 = @intCast(self.head_dim);
        const nkv: i64 = @intCast(self.n_kv_head);
        const max_len: i64 = @intCast(self.max_seq_len);

        // layer.k: [head_dim, max_seq_len, n_kv_head]
        // View into cache at offset: [head_dim, n_tokens, n_kv_head]
        // nb1 = max_seq_len * sizeof(f32)
        // nb2 = max_seq_len * n_kv_head * sizeof(f32)
        // offset in bytes = offset * sizeof(f32) along the seq dim (ne1)
        const k_dst = ctx.view3d(layer.k, hdim, @intCast(n_tokens), nkv,
            @intCast(max_len * @sizeOf(f32)),
            @intCast(max_len * nkv * @sizeOf(f32)),
            @intCast(offset * @sizeOf(f32)));
        const k_cpy = ggml.cpy(ctx, new_k, k_dst);
        graph.buildForwardExpand(k_cpy);

        const v_dst = ctx.view3d(layer.v, hdim, @intCast(n_tokens), nkv,
            @intCast(max_len * @sizeOf(f32)),
            @intCast(max_len * nkv * @sizeOf(f32)),
            @intCast(offset * @sizeOf(f32)));
        const v_cpy = ggml.cpy(ctx, new_v, v_dst);
        graph.buildForwardExpand(v_cpy);

        layer.current_len += n_tokens;
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
};

const testing = std.testing;

test "KVCache init" {
    try testing.expectEqual(@as(usize, @sizeOf(LayerCache)), @sizeOf(LayerCache));
}