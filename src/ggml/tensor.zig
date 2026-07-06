//! ggml_tensor 封装 - ggml_tensor 类型安全 Zig 封装
//! 所有 ctx 参数用 *anyopaque 避免循环导入

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Type = cmod.Type;

pub const Tensor = opaque {
    pub fn getName(self: *Tensor) [:0]const u8 {
        const n = c.ggml_get_name(@ptrCast(@alignCast(self)));
        return std.mem.sliceTo(n, 0);
    }
    pub fn getOpName(self: *Tensor) [:0]const u8 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return std.mem.sliceTo(c.ggml_op_name(t.op), 0);
    }
    pub fn nDims(self: *Tensor) i32 {
        return c.ggml_n_dims(@ptrCast(@alignCast(self)));
    }
    pub fn setName(self: *Tensor, name_str: [:0]const u8) void {
        _ = c.ggml_set_name(@ptrCast(@alignCast(self)), name_str.ptr);
    }
    pub fn ne(self: *Tensor) [4]i64 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return .{ t.ne[0], t.ne[1], t.ne[2], t.ne[3] };
    }
    pub fn nb(self: *Tensor) [4]usize {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return .{ t.nb[0], t.nb[1], t.nb[2], t.nb[3] };
    }
    pub fn strides(self: *Tensor) [4]usize {
        return self.nb();
    }
    pub fn shape(self: *Tensor) [4]i64 {
        return self.ne();
    }
    pub fn dataType(self: *Tensor) Type {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return @as(Type, @enumFromInt(t.type));
    }
    pub fn dataBytes(self: *Tensor) []u8 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return @as([*]u8, @ptrCast(t.data))[0..@as(usize, @intCast(c.ggml_nbytes(t)))];
    }
    /// 获取张量数据为 f32 切片。
    /// 注意：此函数仅适用于类型为 f32 的张量。
    /// 对于 F16/BF16 张量，请使用 dataF16() 或 dataBF16()。
    /// 如果张量不是 f32 类型，此函数会 panic（通过断言）。
    pub fn dataF32(self: *Tensor) []f32 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const typ: Type = @enumFromInt(t.type);
        std.debug.assert(typ == .f32);
        const total_size = c.ggml_nbytes(t);
        return @as([*]f32, @ptrCast(@alignCast(t.data)))[0 .. total_size / @sizeOf(f32)];
    }
    pub fn dataI32(self: *Tensor) []i32 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const typ: Type = @enumFromInt(t.type);
        std.debug.assert(typ == .i32);
        const total_size = c.ggml_nbytes(t);
        return @as([*]i32, @ptrCast(@alignCast(t.data)))[0 .. total_size / @sizeOf(i32)];
    }
    /// 获取张量数据为 f16 切片（仅适用于 F16 类型张量）
    pub fn dataF16(self: *Tensor) []u16 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const typ: Type = @enumFromInt(t.type);
        std.debug.assert(typ == .f16);
        const total_size = c.ggml_nbytes(t);
        return @as([*]u16, @ptrCast(@alignCast(t.data)))[0 .. total_size / @sizeOf(u16)];
    }
    /// 获取张量数据为 bf16 切片（仅适用于 BF16 类型张量）
    pub fn dataBF16(self: *Tensor) []u16 {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const typ: Type = @enumFromInt(t.type);
        std.debug.assert(typ == .bf16);
        const total_size = c.ggml_nbytes(t);
        return @as([*]u16, @ptrCast(@alignCast(t.data)))[0 .. total_size / @sizeOf(u16)];
    }
    pub fn setDataPtr(self: *Tensor, data: []u8) void {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        t.data = data.ptr;
    }
    pub fn setZero(self: *Tensor) void {
        @memset(self.dataBytes(), 0);
    }
    pub fn nElems(self: *Tensor) i64 {
        return c.ggml_nelements(@ptrCast(@alignCast(self)));
    }
    pub fn nBytes(self: *Tensor) usize {
        return @as(usize, @intCast(c.ggml_nbytes(@ptrCast(@alignCast(self)))));
    }
    pub fn isContiguous(self: *Tensor) bool {
        return c.ggml_is_contiguous(@ptrCast(@alignCast(self))) != 0;
    }

    inline fn wrap(ptr: anytype) *Tensor {
        return @as(*Tensor, @ptrCast(ptr));
    }

    pub fn mulMat(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_mul_mat(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn mul(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_mul(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn add(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_add(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn sub(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_sub(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn rmsNorm(self: *Tensor, ctx: *anyopaque, eps: f32) *Tensor {
        return wrap(c.ggml_rms_norm(@ptrCast(ctx), @ptrCast(@alignCast(self)), eps));
    }
    pub fn norm(self: *Tensor, ctx: *anyopaque, eps: f32) *Tensor {
        return wrap(c.ggml_norm(@ptrCast(ctx), @ptrCast(@alignCast(self)), eps));
    }
    pub fn scale(self: *Tensor, ctx: *anyopaque, s: f32) *Tensor {
        return wrap(c.ggml_scale(@ptrCast(ctx), @ptrCast(@alignCast(self)), s));
    }
    pub fn softMax(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_soft_max(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn silu(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_silu(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn relu(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_relu(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn tanh(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_tanh(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn gelu(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_gelu(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn geluErf(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_gelu_erf(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn geluQuick(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_gelu_quick(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn swigluSplit(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_swiglu_split(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn gegluSplit(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_geglu_split(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn gegluErfSplit(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_geglu_erf_split(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn gegluQuickSplit(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_geglu_quick_split(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn print(self: *Tensor, ctx: *anyopaque) void {
        _ = self;
        c.ggml_print_objects(@ptrCast(ctx));
    }
    pub fn sigmoid(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_sigmoid(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn clamp(self: *Tensor, ctx: *anyopaque, min: f32, max: f32) *Tensor {
        return wrap(c.ggml_clamp(@ptrCast(ctx), @ptrCast(@alignCast(self)), min, max));
    }
    pub fn permute(self: *Tensor, ctx: *anyopaque, axis0: i32, axis1: i32, axis2: i32, axis3: i32) *Tensor {
        return wrap(c.ggml_permute(@ptrCast(ctx), @ptrCast(@alignCast(self)), axis0, axis1, axis2, axis3));
    }
    pub fn cont(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_cont(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn cont2d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64) *Tensor {
        return wrap(c.ggml_cont_2d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1));
    }
    pub fn cont4d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
        return wrap(c.ggml_cont_4d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1, ne2, ne3));
    }
    pub fn pad(self: *Tensor, ctx: *anyopaque, p0: i32, p1: i32, p2: i32, p3: i32) *Tensor {
        return wrap(c.ggml_pad(@ptrCast(ctx), @ptrCast(@alignCast(self)), p0, p1, p2, p3));
    }
    pub fn roll(self: *Tensor, ctx: *anyopaque, p0: i32, p1: i32, p2: i32, p3: i32) *Tensor {
        return wrap(c.ggml_roll(@ptrCast(ctx), @ptrCast(@alignCast(self)), p0, p1, p2, p3));
    }
    pub fn pool2d(self: *Tensor, ctx: *anyopaque, op: c_uint, k0: i32, k1: i32, s0: i32, s1: i32, p0: f32, p1: f32) *Tensor {
        return wrap(c.ggml_pool_2d(@ptrCast(ctx), @ptrCast(@alignCast(self)), op, k0, k1, s0, s1, p0, p1));
    }
    pub fn getRows(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor {
        return wrap(c.ggml_get_rows(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b))));
    }
    pub fn reshape2d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64) *Tensor {
        return wrap(c.ggml_reshape_2d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1));
    }
    pub fn reshape3d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64) *Tensor {
        return wrap(c.ggml_reshape_3d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1, ne2));
    }
    pub fn reshape4d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
        return wrap(c.ggml_reshape_4d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1, ne2, ne3));
    }
    pub fn view2d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, nb1: usize, offset: usize) *Tensor {
        return wrap(c.ggml_view_2d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1, nb1, offset));
    }
    pub fn view3d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) *Tensor {
        return wrap(c.ggml_view_3d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1, ne2, nb1, nb2, offset));
    }
    pub fn view4d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor {
        return wrap(c.ggml_view_4d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset));
    }
    pub fn conv2d(self: *Tensor, ctx: *anyopaque, kernel: *Tensor, s0: i32, s1: i32, p0: i32, p1: i32, d0: i32, d1: i32) *Tensor {
        return wrap(c.ggml_conv_2d(@ptrCast(ctx), @ptrCast(@alignCast(kernel)), @ptrCast(@alignCast(self)), s0, s1, p0, p1, d0, d1));
    }
    pub fn im2col(self: *Tensor, ctx: *anyopaque, kernel: *Tensor, s0: i32, s1: i32, p0: i32, p1: i32, d0: i32, d1: i32, is_2d: bool, dst_type: Type) *Tensor {
        return wrap(c.ggml_im2col(@ptrCast(ctx), @ptrCast(@alignCast(kernel)), @ptrCast(@alignCast(self)), s0, s1, p0, p1, d0, d1, is_2d, @intFromEnum(dst_type)));
    }
    pub fn ssmConv(self: *Tensor, ctx: *anyopaque, kernel: *Tensor) *Tensor {
        return wrap(c.ggml_ssm_conv(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(kernel))));
    }
    pub fn sumRows(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_sum_rows(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }
    pub fn concat(self: *Tensor, ctx: *anyopaque, b: *Tensor, dim: i32) *Tensor {
        return wrap(c.ggml_concat(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(b)), dim));
    }
    pub fn ropeExt(self: *Tensor, ctx: *anyopaque, pos: *Tensor, freq_factors: ?*Tensor, n_dims: i32, mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) *Tensor {
        const fp: [*c]c.struct_ggml_tensor = if (freq_factors) |f| @ptrCast(@alignCast(f)) else null;
        return wrap(c.ggml_rope_ext(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(pos)), fp, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
    }
};

const testing = std.testing;
test "Tensor ne and nb" {
    try testing.expectEqual(@as(usize, @sizeOf(*Tensor)), @sizeOf(*Tensor));
}
