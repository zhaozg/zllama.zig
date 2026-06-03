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
    pub fn compute(self: *CGraph, _n_threads: i32) !void {
        _ = _n_threads;
        // 加载所有可用的后端（动态库），只加载一次
        if (!backends_loaded) {
            c.ggml_backend_load_all();
            backends_loaded = true;
            log.info("Backends loaded", .{});
        }

        // 先尝试使用 ggml_backend API（Metal/CUDA 加速）
        const backend = c.ggml_backend_init_best();
        if (backend) |b| {
            defer c.ggml_backend_free(b);
            if (c.ggml_backend_graph_compute(b, @ptrCast(self)) == 0) return;
            log.warn("ggml_backend_graph_compute failed, falling back to CPU", .{});
        } else {
            log.info("No GPU backend available, using CPU", .{});
        }

        // CPU 回退路径：使用 ggml_backend CPU 后端
        const cpu_backend = c.ggml_backend_init_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU, null);
        if (cpu_backend) |b| {
            defer c.ggml_backend_free(b);
            if (c.ggml_backend_graph_compute(b, @ptrCast(self)) == 0) return;
            return error.ComputeError;
        }

        return error.BackendInitFailed;
    }

    /// 获取图中节点数量
    pub fn nNodes(self: *CGraph) i32 {
        const g = @as(*c.struct_ggml_cgraph, @ptrCast(self));
        return @as(i32, @intCast(g.n_nodes));
    }

    /// 获取图中叶子节点数量
    pub fn nLeafs(self: *CGraph) i32 {
        const g = @as(*c.struct_ggml_cgraph, @ptrCast(self));
        return @as(i32, @intCast(g.n_leafs));
    }

    /// 打印图信息（调试用）
    pub fn print(self: *CGraph) void {
        c.ggml_graph_print(@ptrCast(self));
    }

    /// 重置图（清空节点）
    pub fn reset(self: *CGraph) void {
        c.ggml_graph_reset(@ptrCast(self));
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "CGraph basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*CGraph)), @sizeOf(*CGraph));
}
