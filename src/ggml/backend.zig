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
    const backend = c.ggml_backend_cpu_init();
    if (backend == null) return error.GgmlBackendInitFailed;
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
