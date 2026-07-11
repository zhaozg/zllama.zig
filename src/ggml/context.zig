//! ggml_context 封装
//!
//! 提供 ggml_context 的类型安全 Zig 封装。
//! 所有分配类操作返回 `!*T` 错误联合。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Type = cmod.Type;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.ggml);

// ============================================================================
// ggml_context 封装
// ============================================================================

/// ggml_context 的不透明指针封装
pub const Context = opaque {
    /// 初始化 ggml context（分配内存）
    pub fn init(mem_size: usize) !*Context {
        const ctx = c.ggml_init(.{
            .mem_size = mem_size,
            .mem_buffer = null,
            .no_alloc = false,
        });
        if (ctx == null) return error.GgmlInitFailed;
        return @as(*Context, @ptrCast(ctx));
    }

    /// 初始化 ggml context（不分配内存，no_alloc 模式）
    pub fn initNoAlloc(mem_size: usize) !*Context {
        const ctx = c.ggml_init(.{
            .mem_size = mem_size,
            .mem_buffer = null,
            .no_alloc = true,
        });
        if (ctx == null) return error.GgmlInitFailed;
        return @as(*Context, @ptrCast(ctx));
    }

    /// 释放 ggml context
    pub fn deinit(self: *Context) void {
        c.ggml_free(@ptrCast(self));
    }

    /// 重置 ggml context（释放所有张量，重用内存池）
    pub fn reset(self: *Context) void {
        c.ggml_reset(@ptrCast(self));
    }

    /// 获取 no_alloc 模式
    pub fn getNoAlloc(self: *Context) bool {
        return c.ggml_get_no_alloc(@ptrCast(self));
    }

    /// 设置 no_alloc 模式
    pub fn setNoAlloc(self: *Context, no_alloc: bool) void {
        c.ggml_set_no_alloc(@ptrCast(self), no_alloc);
    }

    /// 创建 1D 张量
    pub fn newTensor1d(self: *Context, typ: Type, ne0: i64) !*Tensor {
        const t = c.ggml_new_tensor_1d(@ptrCast(self), @intFromEnum(typ), ne0);
        if (t == null) return error.GgmlNewTensorFailed;
        return @as(*Tensor, @ptrCast(t));
    }

    /// 创建 2D 张量
    pub fn newTensor2d(self: *Context, typ: Type, ne0: i64, ne1: i64) !*Tensor {
        const t = c.ggml_new_tensor_2d(@ptrCast(self), @intFromEnum(typ), ne0, ne1);
        if (t == null) return error.GgmlNewTensorFailed;
        return @as(*Tensor, @ptrCast(t));
    }

    /// 创建 3D 张量
    pub fn newTensor3d(self: *Context, typ: Type, ne0: i64, ne1: i64, ne2: i64) !*Tensor {
        const t = c.ggml_new_tensor_3d(@ptrCast(self), @intFromEnum(typ), ne0, ne1, ne2);
        if (t == null) return error.GgmlNewTensorFailed;
        return @as(*Tensor, @ptrCast(t));
    }

    /// 创建 4D 张量
    pub fn newTensor4d(self: *Context, typ: Type, ne0: i64, ne1: i64, ne2: i64, ne3: i64) !*Tensor {
        const t = c.ggml_new_tensor_4d(@ptrCast(self), @intFromEnum(typ), ne0, ne1, ne2, ne3);
        if (t == null) return error.GgmlNewTensorFailed;
        return @as(*Tensor, @ptrCast(t));
    }

    /// 创建 1D 张量的视图
    pub fn view1d(self: *Context, a: *Tensor, ne0: i64, offset: usize) *Tensor {
        return @as(*Tensor, @ptrCast(c.ggml_view_1d(
            @ptrCast(self),
            @ptrCast(@alignCast(a)),
            ne0,
            offset,
        )));
    }

    /// 创建 2D 张量的视图
    pub fn view2d(
        self: *Context,
        a: *Tensor,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) *Tensor {
        return @as(*Tensor, @ptrCast(c.ggml_view_2d(
            @ptrCast(self),
            @ptrCast(@alignCast(a)),
            ne0,
            ne1,
            nb1,
            offset,
        )));
    }

    /// 创建 3D 张量的视图
    pub fn view3d(
        self: *Context,
        a: *Tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) *Tensor {
        return @as(*Tensor, @ptrCast(c.ggml_view_3d(
            @ptrCast(self),
            @ptrCast(@alignCast(a)),
            ne0,
            ne1,
            ne2,
            nb1,
            nb2,
            offset,
        )));
    }

    /// 获取 context 的内存使用详情（docs/MEMMGT.md §4.2）
    /// 返回已用内存和总内存的元组
    pub fn usage(self: *Context) struct { used: usize, total: usize, ratio: f64 } {
        const used = self.usedMem();
        const total = self.totalMem();
        const ratio = if (total > 0)
            @as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total))
        else
            0.0;
        return .{ .used = used, .total = total, .ratio = ratio };
    }

    /// 打印 context 内存使用详情（调试用）
    pub fn printUsage(self: *Context, label: []const u8) void {
        const u = self.usage();
        std.log.debug("{s}: used={d:.1} MB / total={d:.1} MB ({d:.1}%)", .{
            label,
            @as(f64, @floatFromInt(u.used)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(u.total)) / (1024.0 * 1024.0),
            u.ratio * 100.0,
        });
    }

    /// 获取 context 使用的内存大小
    pub fn usedMem(self: *Context) usize {
        return c.ggml_used_mem(@ptrCast(self));
    }

    /// 获取 context 的总内存大小
    pub fn totalMem(self: *Context) usize {
        return @as(usize, @intCast(c.ggml_get_mem_size(@ptrCast(self))));
    }
    /// 创建 4D 张量的视图
    pub fn view4d(
        self: *Context,
        a: *Tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) *Tensor {
        return @as(*Tensor, @ptrCast(c.ggml_view_4d(
            @ptrCast(self),
            @ptrCast(@alignCast(a)),
            ne0,
            ne1,
            ne2,
            ne3,
            nb1,
            nb2,
            nb3,
            offset,
        )));
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "Context init and deinit" {
    var ctx = try Context.init(1024 * 1024);
    defer ctx.deinit();
    try testing.expect(ctx.usedMem() > 0);
}

test "Context initNoAlloc" {
    var ctx = try Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();
    try testing.expectEqual(@as(usize, 0), ctx.usedMem());
}

test "Context newTensor1d" {
    var ctx = try Context.init(1024 * 1024);
    defer ctx.deinit();
    const t = try ctx.newTensor1d(.f32, 100);
    try testing.expectEqual(@as(i64, 100), t.ne()[0]);
}
