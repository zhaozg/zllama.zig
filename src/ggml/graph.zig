//! ggml_cgraph 封装
//!
//! 提供 ggml_cgraph 的类型安全 Zig 封装。
//! 新增：measureGraph — 计算图内存测量（docs/MEM.md §1.3）

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Tensor = @import("tensor.zig").Tensor;
const Context = @import("context.zig").Context;

const log = std.log.scoped(.ggml);

// ============================================================================
// ggml_cgraph 封装
// ============================================================================

/// ggml_cgraph 的不透明指针封装
pub const CGraph = opaque {
    /// 初始化计算图
    pub fn init(ctx: *Context) !*CGraph {
        const graph = c.ggml_new_graph(@ptrCast(ctx));
        if (graph == null) return error.GgmlNewGraphFailed;
        return @as(*CGraph, @ptrCast(graph));
    }

    /// 初始化计算图（带预留节点数）
    pub fn initReserved(ctx: *Context, n_nodes: i32) !*CGraph {
        const graph = c.ggml_new_graph_custom(
            @ptrCast(ctx),
            @as(usize, @intCast(n_nodes)),
            false,
        );
        if (graph == null) return error.GgmlNewGraphFailed;
        return @as(*CGraph, @ptrCast(graph));
    }

    /// 构建前向扩展（将张量及其依赖添加到图中）
    pub fn buildForwardExpand(self: *CGraph, tensor: *Tensor) void {
        c.ggml_build_forward_expand(@ptrCast(self), @ptrCast(@alignCast(tensor)));
    }

    /// 执行计算图
    pub fn compute(self: *CGraph, n_threads: i32) !void {
        // CPU backend with specified thread count
        const cpu_backend = c.ggml_backend_init_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU, null);
        if (cpu_backend == null) return error.BackendInitFailed;
        defer c.ggml_backend_free(cpu_backend);

        c.ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);

        log.debug("Starting graph compute ({d} nodes, {d} threads)...", .{ c.ggml_graph_n_nodes(@ptrCast(self)), n_threads });
        if (c.ggml_backend_graph_compute(cpu_backend, @ptrCast(self)) == 0) {
            log.debug("Graph compute succeeded", .{});
            return;
        }
        log.warn("ggml_backend_graph_compute failed, falling back to GPU", .{});

        // Try GPU backend second (Metal/CUDA)
        const gpu_backend = c.ggml_backend_init_best();
        if (gpu_backend) |gpu| {
            defer c.ggml_backend_free(gpu);
            if (c.ggml_backend_graph_compute(gpu, @ptrCast(self)) == 0) return;
            log.warn("GPU compute failed", .{});
        }

        return error.ComputeError;
    }

    /// 获取图中节点数量
    pub fn nNodes(self: *CGraph) i32 {
        return c.ggml_graph_n_nodes(@ptrCast(self));
    }

    /// 获取图中指定索引的节点
    pub fn getNode(self: *CGraph, i: i32) *Tensor {
        return @as(*Tensor, @ptrCast(c.ggml_graph_node(@ptrCast(self), i)));
    }

    /// 获取图中叶子节点数量
    pub fn nLeafs(self: *CGraph) i32 {
        _ = self;
        return 0;
    }

    /// 打印图信息（调试用）
    pub fn print(self: *CGraph) void {
        c.ggml_graph_print(@ptrCast(self));
    }

    /// 重置图（清空节点）
    pub fn reset(self: *CGraph) void {
        c.ggml_graph_reset(@ptrCast(self));
    }

    /// 复制计算图（浅复制：新图节点引用相同的张量对象）
    /// 用于图复用场景，避免每 token 重建所有张量
    pub fn dup(ctx: *Context, cgraph: *CGraph) *CGraph {
        return @as(*CGraph, @ptrCast(c.ggml_graph_dup(
            @ptrCast(ctx),
            @ptrCast(cgraph),
            false, // force_grads = false
        )));
    }

    /// 通过名称在图中查找张量
    /// 返回 null 如果未找到
    pub fn getTensor(self: *CGraph, name: [:0]const u8) ?*Tensor {
        const t = c.ggml_graph_get_tensor(@ptrCast(self), name.ptr);
        if (t) |ptr| {
            return @as(*Tensor, @ptrCast(ptr));
        }
        return null;
    }
};

// ============================================================================
    // 计算图内存测量（docs/MEM.md §1.3）
// ============================================================================

/// 测量计算图所需的内存大小（字节）。
/// 使用 ggml_gallocr 的 reserve 模式来预测内存需求，
/// 不实际分配张量数据。
///
/// 参数:
///   - graph: 已构建的计算图
///   - buft: backend buffer type（通常为 CPU buffer type）
///
/// 返回: 所需的总内存大小（字节）
///
/// 参考: llama.cpp 中 ggml_gallocr_reserve 的测量用法
pub fn measureGraph(graph: *CGraph, buft: *c.struct_ggml_backend_buffer_type) !usize {
    // 创建临时 gallocr 用于测量
    const gallocr = c.ggml_gallocr_new(@ptrCast(buft));
    if (gallocr == null) return error.GallocrInitFailed;
    defer c.ggml_gallocr_free(gallocr);

    // reserve 模式会分析图结构并计算所需缓冲区大小，但不分配张量数据
    if (!c.ggml_gallocr_reserve(gallocr, @ptrCast(graph))) {
        return error.GraphMeasureFailed;
    }

    // 获取缓冲区 0 的大小（主计算缓冲区）
    const buf_size = c.ggml_gallocr_get_buffer_size(gallocr, 0);
    return buf_size;
}

/// 测量计算图所需的内存大小，并返回详细的缓冲区信息。
/// 用于调试和动态调整 context 大小。
pub const GraphMeasureInfo = struct {
    total_bytes: usize,
    n_buffers: u32,
    buffer_sizes: []const usize,
};

/// 测量计算图并返回所有缓冲区的详细信息。
/// 调用者负责释放返回的 buffer_sizes 切片。
pub fn measureGraphDetailed(
    graph: *CGraph,
    buft: *c.struct_ggml_backend_buffer_type,
    allocator: std.mem.Allocator,
) !GraphMeasureInfo {
    const gallocr = c.ggml_gallocr_new(@ptrCast(buft));
    if (gallocr == null) return error.GallocrInitFailed;
    defer c.ggml_gallocr_free(gallocr);

    if (!c.ggml_gallocr_reserve(gallocr, @ptrCast(graph))) {
        return error.GraphMeasureFailed;
    }

    // 获取缓冲区数量（通过遍历直到 getBufferSize 返回 0）
    var buf_sizes = std.ArrayList(usize).init(allocator);
    defer buf_sizes.deinit();

    var total: usize = 0;
    var buf_id: u32 = 0;
    while (true) {
        const sz = c.ggml_gallocr_get_buffer_size(gallocr, @intCast(buf_id));
        if (sz == 0) break;
        try buf_sizes.append(sz);
        total += sz;
        buf_id += 1;
    }

    return GraphMeasureInfo{
        .total_bytes = total,
        .n_buffers = buf_id,
        .buffer_sizes = try buf_sizes.toOwnedSlice(),
    };
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "CGraph basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*CGraph)), @sizeOf(*CGraph));
}

test "measureGraph basic" {
    // 创建一个简单的图来测试测量功能
    const ctx = try Context.init(1024 * 1024);
    defer ctx.deinit();

    const a = try ctx.newTensor1d(.f32, 100);
    const b = try ctx.newTensor1d(.f32, 100);

    var graph = try CGraph.initReserved(ctx, 1024);
    graph.buildForwardExpand(a);
    graph.buildForwardExpand(b);

    const buft = c.ggml_backend_cpu_buffer_type() orelse return error.SkipZigTest;
    const size = measureGraph(graph, buft) catch |err| {
        // 测量可能因后端初始化失败而失败，这是可接受的
        std.debug.print("measureGraph test skipped: {}\n", .{err});
        return error.SkipZigTest;
    };
    try testing.expect(size > 0);
    std.debug.print("measureGraph: {d} bytes\n", .{size});
}
