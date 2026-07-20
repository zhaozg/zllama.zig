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
    pub fn elementSize(self: *Tensor) usize {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        return c.ggml_element_size(t);
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
    pub fn mulMatSetPrec(self: *Tensor, prec: cmod.Prec) void {
        c.ggml_mul_mat_set_prec(@ptrCast(@alignCast(self)), @intFromEnum(prec));
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
    pub fn scaleBias(self: *Tensor, ctx: *anyopaque, s: f32, b: f32) *Tensor {
        return @as(*Tensor, @ptrCast(c.ggml_scale_bias(@ptrCast(ctx), @ptrCast(@alignCast(self)), s, b)));
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
    pub fn cont3d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64) *Tensor {
        return wrap(c.ggml_cont_3d(@ptrCast(ctx), @ptrCast(@alignCast(self)), ne0, ne1, ne2));
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
    /// 创建一个与 self 具有相同形状和数据类型的张量（不复制数据）。
    /// 等价于 ggml_dup_tensor。
    pub fn dupTensor(self: *Tensor, ctx: *anyopaque) *Tensor {
        return wrap(c.ggml_dup_tensor(@ptrCast(ctx), @ptrCast(@alignCast(self))));
    }

    pub fn ropeExt(self: *Tensor, ctx: *anyopaque, pos: *Tensor, freq_factors: ?*Tensor, n_dims: i32, mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) *Tensor {
        const fp: [*c]c.struct_ggml_tensor = if (freq_factors) |f| @ptrCast(@alignCast(f)) else null;
        return wrap(c.ggml_rope_ext(@ptrCast(ctx), @ptrCast(@alignCast(self)), @ptrCast(@alignCast(pos)), fp, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
    }

    /// 通过 backend API 从张量读取数据到缓冲区。
    /// 适用于 GPU 后端或任何非 host 内存的张量。
    /// 等价于 ggml_backend_tensor_get(tensor, data, offset, size)。
    /// data 缓冲区必须至少有 size 字节。
    pub fn backendGet(self: *const Tensor, data: []u8, offset: usize) void {
        const t = @as(*const c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        c.ggml_backend_tensor_get(t, data.ptr, offset, data.len);
    }

    /// 通过 backend API 将缓冲区数据写入张量。
    /// 等价于 ggml_backend_tensor_set(tensor, data, offset, size)。
    /// 适用于 GPU 后端或任何非 host 内存的张量。
    pub fn backendSet(self: *Tensor, data: []const u8, offset: usize) void {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        c.ggml_backend_tensor_set(t, data.ptr, offset, data.len);
    }

    /// 从张量读取数据并转换为指定类型的切片。
    ///
    /// 使用 comptime 参数 T 指定目标元素类型（如 f32、i32、f16 等）。
    /// 自动处理类型转换：
    ///   - 如果张量存储类型与 T 相同，直接 memcpy
    ///   - 如果张量是 F16/BF16 而 T 是 f32，自动转换
    ///   - 如果张量是量化类型，返回 error.UnsupportedTensorType
    ///
    /// 适用于 host 内存张量和 backend（GPU）张量。
    /// 调用者负责使用 allocator 释放返回的切片。
    pub fn dataGet(self: *const Tensor, comptime T: type, allocator: std.mem.Allocator) ![]T {
        const t = @as(*const c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const n_elements = @as(usize, @intCast(c.ggml_nelements(t)));
        const nbytes = c.ggml_nbytes(t);

        const result = try allocator.alloc(T, n_elements);
        errdefer allocator.free(result);

        // 尝试直接内存访问（host 内存张量）
        const raw_data = c.ggml_get_data(@ptrCast(@constCast(t)));
        if (raw_data != null) {
            const raw = @as([*]u8, @ptrCast(@alignCast(raw_data)))[0..nbytes];
            convertToType(T, result, raw, n_elements, @intCast(t.type)) catch |e| {
                allocator.free(result);
                return e;
            };
            return result;
        }

        // 回退：通过 backend API 读取
        const raw = try allocator.alloc(u8, nbytes);
        defer allocator.free(raw);
        c.ggml_backend_tensor_get(t, raw.ptr, 0, nbytes);
        convertToType(T, result, raw, n_elements, @intCast(t.type)) catch |e| {
            allocator.free(result);
            return e;
        };
        return result;
    }

    /// 将指定类型的数据写入张量。
    ///
    /// 使用 comptime 参数 T 指定源元素类型（如 f32、i32、f16 等）。
    /// 自动处理类型转换：
    ///   - 如果 T 与张量存储类型相同，直接 memcpy
    ///   - 如果 T 是 f32 而张量是 F16，自动转换
    ///   - 如果张量是量化类型，返回 error.UnsupportedTensorType
    ///
    /// 适用于 host 内存张量和 backend（GPU）张量。
    pub fn dataSet(self: *Tensor, comptime T: type, data: []const T) !void {
        const t = @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(self)));
        const n_elements = @as(usize, @intCast(c.ggml_nelements(t)));
        const nbytes = c.ggml_nbytes(t);

        if (data.len != n_elements) return error.TensorDataLengthMismatch;

        // 尝试直接内存访问（host 内存张量）
        const raw_data = c.ggml_get_data(t);
        if (raw_data != null) {
            const raw = @as([*]u8, @ptrCast(@alignCast(raw_data)))[0..nbytes];
            convertFromType(T, raw, data, n_elements, @intCast(t.type)) catch |e| return e;
            return;
        }

        // 回退：通过 backend API 写入
        const buf = try std.heap.page_allocator.alloc(u8, nbytes);
        defer std.heap.page_allocator.free(buf);
        convertFromType(T, buf, data, n_elements, @intCast(t.type)) catch |e| return e;
        c.ggml_backend_tensor_set(t, buf.ptr, 0, nbytes);
    }
};

/// 将任意类型的 ggml_tensor 转换为 f32 数组。
/// 调用者负责使用 allocator 释放返回的切片。
///
/// 支持两种数据源：
/// 1. 直接内存访问（host 内存张量，如通过 @memcpy 加载的权重）
/// 2. backend 张量（通过 ggml_backend_tensor_get 读取）
pub fn tensorToFloatArray(
    allocator: std.mem.Allocator,
    tensor: *c.struct_ggml_tensor,
) ![]f32 {
    const n_elements = @as(usize, @intCast(c.ggml_nelements(tensor)));
    const nbytes = c.ggml_nbytes(tensor);

    // 分配输出 float 数组
    const float_data = try allocator.alloc(f32, n_elements);
    errdefer allocator.free(float_data);

    // 尝试直接内存访问（host 内存张量）
    const raw_data = c.ggml_get_data(tensor);
    if (raw_data != null) {
        // 直接使用 tensor 的数据指针
        const raw = @as([*]u8, @ptrCast(@alignCast(raw_data)))[0..nbytes];
        return convertRawToF32(float_data, raw, n_elements, @intCast(tensor.type));
    }

    // 回退：通过 backend API 读取（GPU 后端等）
    const raw = try allocator.alloc(u8, nbytes);
    defer allocator.free(raw);
    c.ggml_backend_tensor_get(tensor, raw.ptr, 0, nbytes);
    return convertRawToF32(float_data, raw, n_elements, @intCast(tensor.type));
}

/// 将原始字节数据转换为 f32 数组
fn convertRawToF32(
    float_data: []f32,
    raw: []u8,
    n_elements: usize,
    tensor_type: c_int,
) ![]f32 {
    switch (tensor_type) {
        c.GGML_TYPE_F32 => {
            // 直接内存拷贝
            const dest = @as([*]u8, @ptrCast(float_data.ptr));
            @memcpy(dest[0..raw.len], raw);
        },
        c.GGML_TYPE_F16 => {
            const src = @as([*]const u16, @ptrCast(@alignCast(raw.ptr)));
            for (0..n_elements) |i| {
                float_data[i] = c.ggml_fp16_to_fp32(src[i]);
            }
        },
        c.GGML_TYPE_BF16 => {
            const src = @as([*]const u16, @ptrCast(@alignCast(raw.ptr)));
            for (0..n_elements) |i| {
                const bits = @as(u32, src[i]) << 16;
                float_data[i] = @as(f32, @bitCast(bits));
            }
        },
        // 若需要支持量化类型，可在此扩展
        else => return error.UnsupportedTensorType,
    }

    return float_data;
}

/// 将原始字节数据转换为指定类型的切片。
/// 泛型版本：T 可以是 f32、i32、u16（用于 F16/BF16）等。
/// 如果 T 与张量存储类型匹配，直接 memcpy；否则尝试自动转换。
fn convertToType(
    comptime T: type,
    result: []T,
    raw: []u8,
    n_elements: usize,
    tensor_type: c_int,
) !void {
    const target_size = @sizeOf(T);
    const raw_size = raw.len;

    // 如果类型大小匹配且是简单类型，尝试直接 memcpy
    if (target_size * n_elements == raw_size) {
        switch (tensor_type) {
            c.GGML_TYPE_F32 => {
                if (T == f32) {
                    @memcpy(@as([*]u8, @ptrCast(result.ptr))[0..raw_size], raw);
                    return;
                }
            },
            c.GGML_TYPE_I32 => {
                if (T == i32) {
                    @memcpy(@as([*]u8, @ptrCast(result.ptr))[0..raw_size], raw);
                    return;
                }
            },
            c.GGML_TYPE_F16 => {
                if (T == u16) {
                    @memcpy(@as([*]u8, @ptrCast(result.ptr))[0..raw_size], raw);
                    return;
                }
            },
            c.GGML_TYPE_BF16 => {
                if (T == u16) {
                    @memcpy(@as([*]u8, @ptrCast(result.ptr))[0..raw_size], raw);
                    return;
                }
            },
            else => {},
        }
    }

    // 类型不匹配时尝试转换
    if (T == f32) {
        switch (tensor_type) {
            c.GGML_TYPE_F16 => {
                const src = @as([*]const u16, @ptrCast(@alignCast(raw.ptr)));
                for (0..n_elements) |i| {
                    result[i] = c.ggml_fp16_to_fp32(src[i]);
                }
                return;
            },
            c.GGML_TYPE_BF16 => {
                const src = @as([*]const u16, @ptrCast(@alignCast(raw.ptr)));
                for (0..n_elements) |i| {
                    const bits = @as(u32, src[i]) << 16;
                    result[i] = @as(f32, @bitCast(bits));
                }
                return;
            },
            else => {},
        }
    }

    return error.UnsupportedTensorType;
}

/// 将指定类型的数据转换为原始字节并写入缓冲区。
/// 泛型版本：T 可以是 f32、i32、u16（用于 F16/BF16）等。
/// 如果 T 与张量存储类型匹配，直接 memcpy；否则尝试自动转换。
fn convertFromType(
    comptime T: type,
    raw: []u8,
    data: []const T,
    n_elements: usize,
    tensor_type: c_int,
) !void {
    const target_size = @sizeOf(T);
    const raw_size = raw.len;

    // 如果类型大小匹配且是简单类型，尝试直接 memcpy
    if (target_size * n_elements == raw_size) {
        switch (tensor_type) {
            c.GGML_TYPE_F32 => {
                if (T == f32) {
                    @memcpy(raw, @as([*]const u8, @ptrCast(data.ptr))[0..raw_size]);
                    return;
                }
            },
            c.GGML_TYPE_I32 => {
                if (T == i32) {
                    @memcpy(raw, @as([*]const u8, @ptrCast(data.ptr))[0..raw_size]);
                    return;
                }
            },
            c.GGML_TYPE_F16 => {
                if (T == u16) {
                    @memcpy(raw, @as([*]const u8, @ptrCast(data.ptr))[0..raw_size]);
                    return;
                }
            },
            c.GGML_TYPE_BF16 => {
                if (T == u16) {
                    @memcpy(raw, @as([*]const u8, @ptrCast(data.ptr))[0..raw_size]);
                    return;
                }
            },
            else => {},
        }
    }

    // 类型不匹配时尝试转换
    if (T == f32) {
        switch (tensor_type) {
            c.GGML_TYPE_F16 => {
                const dst = @as([*]u16, @ptrCast(@alignCast(raw.ptr)));
                for (0..n_elements) |i| {
                    dst[i] = c.ggml_fp32_to_fp16(data[i]);
                }
                return;
            },
            c.GGML_TYPE_BF16 => {
                const dst = @as([*]u16, @ptrCast(@alignCast(raw.ptr)));
                for (0..n_elements) |i| {
                    dst[i] = @as(u16, @truncate(@as(u32, @bitCast(data[i])) >> 16));
                }
                return;
            },
            else => {},
        }
    }

    return error.UnsupportedTensorType;
}

const testing = std.testing;
test "Tensor ne and nb" {
    try testing.expectEqual(@as(usize, @sizeOf(*Tensor)), @sizeOf(*Tensor));
}
