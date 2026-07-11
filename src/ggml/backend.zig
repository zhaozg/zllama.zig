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

/// Backend device types for init_by_type
pub const DeviceType = enum(c_uint) {
    cpu = 0,
    gpu = 1,
    igpu = 2,
    accel = 3,
    meta = 4,
};

/// Detect and return the best available backend (tries GPU first, falls back to CPU).
/// Caller owns the returned backend and must call backendFree() when done.
pub fn detectBestBackend() !*Backend {
    // Try best auto-detected backend first
    if (backendInitBest()) |best| {
        const name = backendName(best);
        log.info("Best backend: {s}", .{name});
        return best;
    }
    // Try GPU explicitly
    if (backendInitByType(@intFromEnum(DeviceType.gpu), null)) |gpu| {
        const name = backendName(gpu);
        log.info("GPU backend: {s}", .{name});
        return gpu;
    }
    // Fallback to CPU
    log.info("Using CPU backend", .{});
    return backendCpuInit();
}

/// Get backend name string (caller must not free)
pub fn backendName(backend: *Backend) []const u8 {
    const ptr = c.ggml_backend_name(@ptrCast(backend));
    if (ptr == null) return "(unknown)";
    return std.mem.span(ptr);
}

/// Check if a backend is a GPU (Metal/CUDA/Vulkan)
pub fn backendIsGpu(backend: *Backend) bool {
    const dev = c.ggml_backend_get_device(@ptrCast(backend)) orelse return false;
    const dev_type = c.ggml_backend_dev_type(dev);
    return dev_type == @intFromEnum(DeviceType.gpu) or dev_type == @intFromEnum(DeviceType.igpu);
}

/// Get name for a backend device
pub fn backendDevName(dev: *c.struct_ggml_backend_device) []const u8 {
    const ptr = c.ggml_backend_dev_name(dev);
    if (ptr == null) return "(unknown)";
    return std.mem.span(ptr);
}

/// Get description for a backend device
pub fn backendDevDescription(dev: *c.struct_ggml_backend_device) []const u8 {
    const ptr = c.ggml_backend_dev_description(dev);
    if (ptr == null) return "(unknown)";
    return std.mem.span(ptr);
}

/// Get total/free memory for a backend device
pub fn backendDevMemory(dev: *c.struct_ggml_backend_device) struct { free: usize, total: usize } {
    var free: usize = 0;
    var total: usize = 0;
    c.ggml_backend_dev_memory(dev, &free, &total);
    return .{ .free = free, .total = total };
}

/// Log available backends for diagnostics
pub fn logAvailableBackends() void {
    const reg_count = c.ggml_backend_reg_count();
    log.info("Available ggml backends: {d}", .{reg_count});
    var i: usize = 0;
    while (i < reg_count) : (i += 1) {
        const reg = c.ggml_backend_reg_get(i) orelse continue;
        const reg_name = c.ggml_backend_reg_name(reg);
        const dev_count = c.ggml_backend_reg_dev_count(reg);
        log.info("  [{d}] {s} ({d} devices)", .{ i, if (reg_name != null) std.mem.span(reg_name) else "(unnamed)", dev_count });
        var j: usize = 0;
        while (j < dev_count) : (j += 1) {
            const dev = c.ggml_backend_reg_dev_get(reg, j) orelse continue;
            const dev_name = c.ggml_backend_dev_name(dev);
            const dev_desc = c.ggml_backend_dev_description(dev);
            var mem_free: usize = 0;
            var mem_total: usize = 0;
            c.ggml_backend_dev_memory(dev, &mem_free, &mem_total);
            if (mem_total > 0) {
                log.info("    [{d}] {s} — {s} (memory: {d:.1} GB free / {d:.1} GB total)", .{
                    j,
                    if (dev_name != null) std.mem.span(dev_name) else "(unnamed)",
                    if (dev_desc != null) std.mem.span(dev_desc) else "",
                    @as(f64, @floatFromInt(mem_free)) / 1e9,
                    @as(f64, @floatFromInt(mem_total)) / 1e9,
                });
            } else {
                log.info("    [{d}] {s} — {s}", .{
                    j,
                    if (dev_name != null) std.mem.span(dev_name) else "(unnamed)",
                    if (dev_desc != null) std.mem.span(dev_desc) else "",
                });
            }
        }
    }
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
    pub fn free(self: *Scheduler) void {
        c.ggml_backend_sched_free(@ptrCast(self));
    }
    pub fn allocGraph(self: *Scheduler, graph: *CGraph) bool {
        return c.ggml_backend_sched_alloc_graph(@ptrCast(self), @ptrCast(graph));
    }
    pub fn reserve(self: *Scheduler, graph: *CGraph) bool {
        return c.ggml_backend_sched_reserve(@ptrCast(self), @ptrCast(graph));
    }
    pub fn graphCompute(self: *Scheduler, graph: *CGraph) bool {
        return c.ggml_backend_sched_graph_compute(@ptrCast(self), @ptrCast(graph)) == 1;
    }
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
    /// 释放后内部指针置空，防止重复释放。
    pub fn free(self: *Gallocr) void {
        c.ggml_gallocr_free(@as(*c.struct_ggml_gallocr, @ptrCast(self)));
    }

    /// 检查当前是否已分配内存。
    /// 由于 Gallocr 是 opaque 类型，无法直接访问内部字段。
    /// 调用者应自行维护状态跟踪（如 IncContext.galloc_reserved）。
    pub fn isAllocated(self: *Gallocr) bool {
        _ = self;
        // opaque 类型无法可靠判断，调用者自行跟踪。
        return true;
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
