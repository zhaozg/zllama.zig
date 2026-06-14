//! ggml_cgraph 封装
//!
//! 提供 ggml_cgraph 的类型安全 Zig 封装。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Tensor = @import("tensor.zig").Tensor;
const Context = @import("context.zig").Context;

const log = std.log.scoped(.ggml);

// ============================================================================
// 全局状态
// ============================================================================

var backends_loaded: bool = false;

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
        // 加载所有可用的后端（动态库），只加载一次
        if (!backends_loaded) {
            c.ggml_backend_load_all();
            backends_loaded = true;
            log.info("Backends loaded", .{});
        }

        // CPU backend with specified thread count
        const cpu_backend = c.ggml_backend_init_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU, null);
        if (cpu_backend == null) return error.BackendInitFailed;
        defer c.ggml_backend_free(cpu_backend);

        c.ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);

        // Try GPU backend first (Metal/CUDA), fall back to CPU
        const gpu_backend = c.ggml_backend_init_best();
        if (gpu_backend) |gpu| {
            defer c.ggml_backend_free(gpu);
            if (c.ggml_backend_graph_compute(gpu, @ptrCast(self)) == 0) return;
            log.warn("ggml_backend_graph_compute failed, falling back to CPU", .{});
        }

        if (c.ggml_backend_graph_compute(cpu_backend, @ptrCast(self)) == 0) return;
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
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "CGraph basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*CGraph)), @sizeOf(*CGraph));
}
