//! 内存抽象接口
//!
//! 定义 MemoryContext 接口，支持多种内存类型：
//! - KV Cache（标准注意力）
//! - Recurrent State（SSM/DeltaNet）
//! - Hybrid（混合架构：KV Cache + SSM State 统一管理）
//!
//! 参考 llama.cpp 的 llama_memory_context_i 接口设计。

const std = @import("std");
const ggml = @import("ggml");

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
        /// Optional: get SSM conv state for a layer (returns null if not applicable)
        get_conv_state: *const fn (data: *anyopaque, layer: usize) ?*ggml.Tensor = &getConvStateNotSupported,
        /// Optional: get SSM recurrent state for a layer
        get_ssm_state: *const fn (data: *anyopaque, layer: usize) ?*ggml.Tensor = &getSSMStateNotSupported,
        /// Optional: reset SSM states (zero them out)
        reset_ssm: *const fn (data: *anyopaque) void = &resetSSMNotSupported,
    };

    fn getConvStateNotSupported(_: *anyopaque, _: usize) ?*ggml.Tensor { return null; }
    fn getSSMStateNotSupported(_: *anyopaque, _: usize) ?*ggml.Tensor { return null; }
    fn resetSSMNotSupported(_: *anyopaque) void {}

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

    /// Get SSM conv state for a layer (hybrid memory only).
    pub fn getConvState(self: *MemoryContext, layer: usize) ?*ggml.Tensor {
        return self.vtable.get_conv_state(self.data, layer);
    }

    /// Get SSM recurrent state for a layer (hybrid memory only).
    pub fn getSSMState(self: *MemoryContext, layer: usize) ?*ggml.Tensor {
        return self.vtable.get_ssm_state(self.data, layer);
    }

    /// Reset all SSM states (hybrid memory only).
    pub fn resetSSM(self: *MemoryContext) void {
        self.vtable.reset_ssm(self.data);
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

/// SSM 状态参数
pub const SSMStateParams = struct {
    d_conv: u32 = 4,       // conv kernel size
    d_state: u32 = 128,     // state dimension
    d_inner: u32 = 2048,    // inner dimension
    n_group: u32 = 16,      // number of groups
    dt_rank: u32 = 16,      // time step rank
};

/// 混合内存实现 — 统一管理 KV Cache + SSM 状态
///
/// 用于 Qwen3.5 混合架构（全注意力 + SSM/GDN 线性注意力交替），
/// 在 init 时预分配所有持久张量，避免推理时 lazily 分配。
pub const HybridMemory = struct {
    /// KV cache layers (same as KVCacheMemory, only used by full-attention layers)
    kv_layers: []struct {
        k: *ggml.Tensor,
        v: *ggml.Tensor,
        current_len: u32,
    },
    /// SSM conv states per layer [d_conv-1, conv_dim] (SSM layers only, null for full-attn layers)
    ssm_conv_states: []?*ggml.Tensor,
    /// SSM recurrent states per layer [S_v, S_v, H, 1] (SSM layers only)
    ssm_recurrent_states: []?*ggml.Tensor,

    max_seq_len: u32,
    n_kv_head: u32,
    head_dim: u32,
    ssm_params: SSMStateParams,
    /// Which layers are full-attention (true) vs SSM (false)
    layer_is_full_attn: []bool,

    pub fn init(
        ctx: *ggml.Context,
        n_layer: u32,
        n_kv_head: u32,
        head_dim: u32,
        max_seq_len: u32,
        ssm_params: SSMStateParams,
        /// Predicate: isFullAttentionLayer(layer_idx)
        is_full_attn_fn: *const fn (layer_idx: u32, interval: u32) bool,
        full_attn_interval: u32,
        allocator: std.mem.Allocator,
    ) !HybridMemory {
        var kv_layers = try allocator.alloc(struct {
            k: *ggml.Tensor,
            v: *ggml.Tensor,
            current_len: u32,
        }, n_layer);
        var ssm_conv_states = try allocator.alloc(?*ggml.Tensor, n_layer);
        var ssm_recurrent_states = try allocator.alloc(?*ggml.Tensor, n_layer);
        var layer_is_full_attn = try allocator.alloc(bool, n_layer);
        errdefer {
            allocator.free(layer_is_full_attn);
            allocator.free(ssm_recurrent_states);
            allocator.free(ssm_conv_states);
            allocator.free(kv_layers);
        }

        const d_conv: i64 = @intCast(ssm_params.d_conv);
        const d_inner: i64 = @intCast(ssm_params.d_inner);
        const d_state: i64 = @intCast(ssm_params.d_state);
        const n_group: i64 = @intCast(ssm_params.n_group);
        const dt_rank: i64 = @intCast(ssm_params.dt_rank);

        const head_k_dim = d_state;
        const head_v_dim = @divExact(d_inner, dt_rank);
        const num_v_heads = dt_rank;
        const key_dim = head_k_dim * n_group;
        const value_dim = head_v_dim * num_v_heads;
        const conv_dim = key_dim * 2 + value_dim;

        for (0..n_layer) |i| {
            const idx: u32 = @intCast(i);
            const is_full = is_full_attn_fn(idx, full_attn_interval);
            layer_is_full_attn[i] = is_full;

            // Allocate KV cache per layer (needed for full-attention layers)
            const k = try ctx.newTensor3d(.f32, @intCast(head_dim), @intCast(n_kv_head), @intCast(max_seq_len));
            const v = try ctx.newTensor3d(.f32, @intCast(head_dim), @intCast(n_kv_head), @intCast(max_seq_len));
            kv_layers[i] = .{ .k = k, .v = v, .current_len = 0 };

            if (!is_full) {
                // Pre-allocate SSM states for SSM layers
                ssm_conv_states[i] = try ctx.newTensor2d(.f32, d_conv - 1, conv_dim);
                ssm_conv_states[i].?.setZero();
                ssm_recurrent_states[i] = try ctx.newTensor4d(.f32, head_v_dim, head_v_dim, num_v_heads, 1);
                ssm_recurrent_states[i].?.setZero();
            } else {
                ssm_conv_states[i] = null;
                ssm_recurrent_states[i] = null;
            }
        }

        return HybridMemory{
            .kv_layers = kv_layers,
            .ssm_conv_states = ssm_conv_states,
            .ssm_recurrent_states = ssm_recurrent_states,
            .max_seq_len = max_seq_len,
            .n_kv_head = n_kv_head,
            .head_dim = head_dim,
            .ssm_params = ssm_params,
            .layer_is_full_attn = layer_is_full_attn,
        };
    }

    // ---- VTable callbacks ----

    pub fn deinit(data: *anyopaque) void {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        // Layers/state arrays owned by caller; freed via the allocator passed to init
        _ = self;
    }

    pub fn reset(data: *anyopaque) void {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        for (self.kv_layers) |*layer| {
            layer.current_len = 0;
        }
        for (self.ssm_conv_states) |maybe_state| {
            if (maybe_state) |state| state.setZero();
        }
        for (self.ssm_recurrent_states) |maybe_state| {
            if (maybe_state) |state| state.setZero();
        }
    }

    pub fn getK(data: *anyopaque, layer: usize) *ggml.Tensor {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        return self.kv_layers[layer].k;
    }

    pub fn getV(data: *anyopaque, layer: usize) *ggml.Tensor {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        return self.kv_layers[layer].v;
    }

    pub fn setKv(data: *anyopaque, ctx: *ggml.Context, graph: *ggml.CGraph, layer: usize, new_k: *ggml.Tensor, new_v: *ggml.Tensor, n_tokens: u32) void {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        const layer_data = &self.kv_layers[layer];
        const offset = layer_data.current_len;

        const hdim: i64 = @intCast(self.head_dim);
        const nkv: i64 = @intCast(self.n_kv_head);
        const max_len: i64 = @intCast(self.max_seq_len);

        const k_dst = ctx.view3d(layer_data.k, hdim, nkv, @intCast(n_tokens),
            @intCast(nkv * @sizeOf(f32)),
            @intCast(max_len * @sizeOf(f32)),
            @intCast(offset * @sizeOf(f32)));
        graph.buildForwardExpand(ggml.cpy(ctx, new_k, k_dst));

        const v_dst = ctx.view3d(layer_data.v, hdim, nkv, @intCast(n_tokens),
            @intCast(nkv * @sizeOf(f32)),
            @intCast(max_len * @sizeOf(f32)),
            @intCast(offset * @sizeOf(f32)));
        graph.buildForwardExpand(ggml.cpy(ctx, new_v, v_dst));

        layer_data.current_len += n_tokens;
    }

    pub fn currentLen(data: *anyopaque) u32 {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        var max_len: u32 = 0;
        for (self.kv_layers) |layer| {
            if (layer.current_len > max_len) max_len = layer.current_len;
        }
        return max_len;
    }

    pub fn nLayers(data: *anyopaque) u32 {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        return @intCast(self.kv_layers.len);
    }

    pub fn getConvState(data: *anyopaque, layer: usize) ?*ggml.Tensor {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        return self.ssm_conv_states[layer];
    }

    pub fn getSSMState(data: *anyopaque, layer: usize) ?*ggml.Tensor {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        return self.ssm_recurrent_states[layer];
    }

    pub fn resetSSM(data: *anyopaque) void {
        const self = @as(*HybridMemory, @ptrCast(@alignCast(data)));
        for (self.ssm_conv_states) |maybe_state| {
            if (maybe_state) |state| state.setZero();
        }
        for (self.ssm_recurrent_states) |maybe_state| {
            if (maybe_state) |state| state.setZero();
        }
    }

    /// Create MemoryContext wrapper
    pub fn toMemoryContext(self: *HybridMemory) MemoryContext {
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
        .get_conv_state = getConvState,
        .get_ssm_state = getSSMState,
        .reset_ssm = resetSSM,
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
