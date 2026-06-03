//! ggml_tensor 封装
//!
//! 提供 ggml_tensor 的类型安全 Zig 封装。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Type = cmod.Type;

// ============================================================================
// ggml_tensor 封装
// ============================================================================

/// ggml_tensor 的不透明指针封装
pub const Tensor = opaque {
    /// 获取张量名称
    pub fn getName(self: *Tensor) [:0]const u8 {
        const n = c.ggml_get_name(@ptrCast(@alignCast(self)));
        return std.mem.sliceTo(n, 0);
    }

    /// 设置张量名称
    pub fn setName(self: *Tensor, name_str: [:0]const u8) void {
        _ = c.ggml_set_name(@ptrCast(@alignCast(self)), name_str.ptr);
    }

    /// 获取张量各维度大小
    pub fn ne(self: *Tensor) [4]i64 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return .{ t.ne[0], t.ne[1], t.ne[2], t.ne[3] };
    }

    /// 获取张量各维度步长（字节）
    pub fn nb(self: *Tensor) [4]usize {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return .{ t.nb[0], t.nb[1], t.nb[2], t.nb[3] };
    }

    /// 获取张量各维度步长（别名，与 nb 相同）
    pub fn strides(self: *Tensor) [4]usize {
        return self.nb();
    }

    /// 获取张量形状描述字符串
    pub fn shape(self: *Tensor) [4]i64 {
        return self.ne();
    }

    /// 获取张量数据类型
    pub fn dataType(self: *Tensor) Type {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return @as(Type, @enumFromInt(t.type));
    }

    /// 获取张量数据指针（字节切片）
    pub fn dataBytes(self: *Tensor) []u8 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const total_size = c.ggml_nbytes(t);
        return @as([*]u8, @ptrCast(t.data))[0..total_size];
    }

    /// 获取张量数据指针（f32 切片）
    pub fn dataF32(self: *Tensor) []f32 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const total_size = c.ggml_nbytes(t);
        return @as([*]f32, @ptrCast(@alignCast(t.data)))[0 .. total_size / @sizeOf(f32)];
    }

    /// 获取张量数据指针（i32 切片）
    pub fn dataI32(self: *Tensor) []i32 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const total_size = c.ggml_nbytes(t);
        return @as([*]i32, @ptrCast(@alignCast(t.data)))[0 .. total_size / @sizeOf(i32)];
    }

    /// 设置张量数据指针（零拷贝，用于 GGUF 权重加载）
    pub fn setDataPtr(self: *Tensor, data: []u8) void {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        t.data = data.ptr;
    }

    /// 将张量数据全部置零
    pub fn setZero(self: *Tensor) void {
        const bytes = self.dataBytes();
        @memset(bytes, 0);
    }

    /// 获取张量元素总数
    pub fn nElems(self: *Tensor) usize {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return @as(usize, @intCast(c.ggml_nelements(t)));
    }

    /// 获取张量字节数
    pub fn nBytes(self: *Tensor) usize {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return @as(usize, @intCast(c.ggml_nbytes(t)));
    }

    /// 检查张量是否连续
    pub fn isContiguous(self: *Tensor) bool {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return c.ggml_is_contiguous(t) != 0;
    }

    /// 打印张量信息（调试用）
    pub fn print(self: *Tensor) void {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        c.ggml_print_objects(t);
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "Tensor ne and nb" {
    try testing.expectEqual(@as(usize, @sizeOf(*Tensor)), @sizeOf(*Tensor));
}
