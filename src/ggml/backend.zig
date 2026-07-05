//! Backend 与 Graph Allocator 封装
//!
//! 提供 ggml_backend 和 ggml_gallocr 的类型安全 Zig 封装。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Context = @import("context.zig").Context;
const Tensor = @import("tensor.zig").Tensor;
const CGraph = @import("graph.zig").CGraph;

const log = std.log.scoped(.ggml);

// ============================================================================
// 全局状态
// ============================================================================

var backends_loaded: bool = false;

// ============================================================================
// Backend 封装
// ============================================================================

/// ggml_backend 结构体引用
pub const Backend = c.struct_ggml_backend;
/// ggml_backend_buffer_type 结构体引用
pub const BackendBufferType = c.struct_ggml_backend_buffer_type;

/// 初始化 CPU backend
pub fn backendCpuInit() !*Backend {
    const backend = c.ggml_backend_cpu_init() orelse return error.GgmlBackendInitFailed;
    return backend;
}

/// 获取 CPU backend 的默认 buffer type
pub fn backendCpuBufferType() *BackendBufferType {
    return c.ggml_backend_cpu_buffer_type().?;
}

/// 获取 backend 的默认 buffer type
pub fn backendGetDefaultBufferType(backend: *Backend) *BackendBufferType {
    return c.ggml_backend_get_default_buffer_type(backend).?;
}

/// 按类型初始化 backend (CPU/GPU)
pub fn backendInitByType(device_type: c_uint, device_id: ?[]const u8) ?*Backend {
    const id_ptr = if (device_id) |id| id.ptr else null;
    return c.ggml_backend_init_by_type(@intCast(device_type), id_ptr);
}

/// 初始化最佳可用 backend
pub fn backendInitBest() ?*Backend {
    return c.ggml_backend_init_best();
}

/// 设置 CPU backend 线程数
pub fn backendCpuSetNThreads(backend: *Backend, n_threads: i32) void {
    c.ggml_backend_cpu_set_n_threads(backend, n_threads);
}

/// 在指定 backend 上执行计算图
/// ggml_backend_graph_compute 返回 ggml_status: GGML_STATUS_SUCCESS = 0
pub fn backendGraphCompute(backend: *Backend, graph: *CGraph) bool {
    return c.ggml_backend_graph_compute(backend, @ptrCast(graph)) == 0;
}

/// 将数据从 tensor 拷贝到 host 内存
pub fn backendTensorGet(tensor: *Tensor, data: []u8, offset: usize) void {
    c.ggml_backend_tensor_get(@ptrCast(@alignCast(tensor)), data.ptr, offset, data.len);
}

/// 检查 buffer type 是否为 host 内存（CPU 可访问）
/// 对于 CPU 和 Metal 后端返回 true，对于 CUDA 等设备后端返回 false
pub fn backendBuftIsHost(buft: *BackendBufferType) bool {
    return c.ggml_backend_buft_is_host(@ptrCast(buft));
}

/// 将数据从 host 内存设置到 tensor（支持 device 内存）
/// 对于 device 内存的 tensor，内部处理 host->device 拷贝
pub fn backendTensorSet(tensor: *Tensor, data: []const u8, offset: usize) void {
    c.ggml_backend_tensor_set(@ptrCast(@alignCast(tensor)), data.ptr, offset, data.len);
}

/// 为 context 中所有未分配的张量分配内存
/// 权重张量（已通过 setDataPtr 设置）不会被重新分配
pub fn backendAllocCtxTensors(ctx: *Context, backend: *Backend) !void {
    const buft = backendGetDefaultBufferType(backend);
    return backendAllocCtxTensorsFromBuft(ctx, buft);
}

/// 使用指定的 buffer type 为 context 中所有未分配的张量分配内存
pub fn backendAllocCtxTensorsFromBuft(ctx: *Context, buft: *BackendBufferType) !void {
    const buf = c.ggml_backend_alloc_ctx_tensors_from_buft(
        @ptrCast(ctx),
        @ptrCast(buft),
    );
    if (buf == null) {
        return error.GgmlBackendAllocFailed;
    }
}

/// 释放 backend
pub fn backendFree(backend: *Backend) void {
    c.ggml_backend_free(backend);
}

/// 加载所有可用的 ggml backend
pub fn loadBackends() void {
    if (backends_loaded) return;
    c.ggml_backend_load_all();
    backends_loaded = true;
    log.info("Backends loaded", .{});
}

// ============================================================================
// Backend Scheduler 封装
// ============================================================================

/// ggml_backend_sched 不透明句柄
pub const Scheduler = opaque {
    pub fn init(backends: []const *Backend, bufts: []const *BackendBufferType, graph_size: usize, parallel: bool) !*Scheduler {
        const sched = c.ggml_backend_sched_new(@constCast(backends.ptr), @constCast(bufts.ptr), @intCast(backends.len), @intCast(graph_size), parallel, true);
        if (sched == null) return error.SchedulerInitFailed;
        return @as(*Scheduler, @ptrCast(sched));
    }
    pub fn free(self: *Scheduler) void { c.ggml_backend_sched_free(@ptrCast(self)); }
    pub fn allocGraph(self: *Scheduler, graph: *CGraph) bool { return c.ggml_backend_sched_alloc_graph(@ptrCast(self), @ptrCast(graph)); }
    pub fn reserve(self: *Scheduler, graph: *CGraph) bool { return c.ggml_backend_sched_reserve(@ptrCast(self), @ptrCast(graph)); }
    pub fn graphCompute(self: *Scheduler, graph: *CGraph) bool { return c.ggml_backend_sched_graph_compute(@ptrCast(self), @ptrCast(graph)) == 1; }
};

// ============================================================================
// Graph Allocator (gallocr) 封装
// ============================================================================

/// Graph allocator 句柄
pub const Gallocr = opaque {
    /// 初始化 graph allocator
    pub fn init(buft: *BackendBufferType) !*Gallocr {
        const ga = c.ggml_gallocr_new(@ptrCast(buft));
        if (ga == null) return error.GgmlGallocrInitFailed;
        return @as(*Gallocr, @ptrCast(ga));
    }

    /// 为计算图分配内存
    pub fn allocGraph(self: *Gallocr, graph: *CGraph) bool {
        return c.ggml_gallocr_alloc_graph(@ptrCast(self), @ptrCast(graph));
    }

    /// Pre-allocate buffers from a measure graph to avoid reallocations.
    /// Call once with a worst-case graph before the main loop.
    /// Returns false if the buffer allocation failed.
    pub fn reserve(self: *Gallocr, graph: *CGraph) bool {
        return c.ggml_gallocr_reserve(@ptrCast(self), @ptrCast(graph));
    }

    /// Get the buffer size for the given buffer index (for diagnostics).
    pub fn getBufferSize(self: *Gallocr, buffer_id: u32) usize {
        return c.ggml_gallocr_get_buffer_size(@ptrCast(self), @intCast(buffer_id));
    }

    /// 释放 graph allocator
    pub fn free(self: *Gallocr) void {
        c.ggml_gallocr_free(@as(*c.struct_ggml_gallocr, @ptrCast(self)));
    }
};

/// 设置输入张量（告诉 allocator 此张量不会被覆盖）
pub fn setInput(tensor: *Tensor) void {
    c.ggml_set_input(@ptrCast(@alignCast(tensor)));
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "Backend basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*Backend)), @sizeOf(*Backend));
}

test "Gallocr basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*Gallocr)), @sizeOf(*Gallocr));
}
