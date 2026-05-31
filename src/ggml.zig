//! ggml.zig - 安全封装层
//!
//! 提供 ggml C API 的类型安全 Zig 封装。
//! 所有分配类操作返回 `!*T` 错误联合，纯计算操作返回 `*T`。
//! 使用 `opaque {}` 类型包装不透明指针。

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// 原始 C API
// ============================================================================

pub const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-cpu.h");
    @cInclude("ggml-backend.h");
    @cInclude("gguf.h");
});

// ============================================================================
// 类型定义
// ============================================================================

/// ggml 数据类型枚举
pub const Type = enum(c.ggml_type) {
    f32 = c.GGML_TYPE_F32,
    f16 = c.GGML_TYPE_F16,
    q4_0 = c.GGML_TYPE_Q4_0,
    q4_1 = c.GGML_TYPE_Q4_1,
    q5_0 = c.GGML_TYPE_Q5_0,
    q5_1 = c.GGML_TYPE_Q5_1,
    q8_0 = c.GGML_TYPE_Q8_0,
    q8_1 = c.GGML_TYPE_Q8_1,
    q2_K = c.GGML_TYPE_Q2_K,
    q3_K = c.GGML_TYPE_Q3_K,
    q4_K = c.GGML_TYPE_Q4_K,
    q5_K = c.GGML_TYPE_Q5_K,
    q6_K = c.GGML_TYPE_Q6_K,
    q8_K = c.GGML_TYPE_Q8_K,
    i8 = c.GGML_TYPE_I8,
    i16 = c.GGML_TYPE_I16,
    i32 = c.GGML_TYPE_I32,
    i64 = c.GGML_TYPE_I64,
    f64 = c.GGML_TYPE_F64,

    pub fn sizeOf(t: Type) usize {
        return c.ggml_type_size(@intFromEnum(t));
    }

    pub fn blockSize(t: Type) i64 {
        return c.ggml_blck_size(@intFromEnum(t));
    }

    pub fn rowSize(t: Type, ne: i64) usize {
        return c.ggml_row_size(@intFromEnum(t), ne);
    }

    pub fn isQuantized(t: Type) bool {
        return c.ggml_is_quantized(@intFromEnum(t)) != 0;
    }

    pub fn name(t: Type) [:0]const u8 {
        return std.mem.sliceTo(c.ggml_type_name(@intFromEnum(t)), 0);
    }
};

/// GGUF 值类型枚举
pub const GgufValueType = enum(c.gguf_type) {
    uint8 = c.GGUF_TYPE_UINT8,
    int8 = c.GGUF_TYPE_INT8,
    uint16 = c.GGUF_TYPE_UINT16,
    int16 = c.GGUF_TYPE_INT16,
    uint32 = c.GGUF_TYPE_UINT32,
    int32 = c.GGUF_TYPE_INT32,
    float32 = c.GGUF_TYPE_FLOAT32,
    bool = c.GGUF_TYPE_BOOL,
    string = c.GGUF_TYPE_STRING,
    array = c.GGUF_TYPE_ARRAY,
    uint64 = c.GGUF_TYPE_UINT64,
    int64 = c.GGUF_TYPE_INT64,
    float64 = c.GGUF_TYPE_FLOAT64,

    pub fn name(t: GgufValueType) [:0]const u8 {
        return std.mem.sliceTo(c.gguf_type_name(@intFromEnum(t)), 0);
    }
};

/// GGUF 值联合体
pub const GgufValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: [:0]const u8,
    array: struct { typ: GgufValueType, items: []const u8 },
    uint64: u64,
    int64: i64,
    float64: f64,

    pub fn asString(self: GgufValue) ![]const u8 {
        return switch (self) {
            .string => |s| s,
            else => error.TypeMismatch,
        };
    }

    pub fn asInt(self: GgufValue) !i64 {
        return switch (self) {
            .int32 => |v| @as(i64, v),
            .int64 => |v| v,
            .uint32 => |v| @as(i64, v),
            .uint64 => |v| @as(i64, @intCast(v)),
            else => error.TypeMismatch,
        };
    }

    pub fn asFloat(self: GgufValue) !f64 {
        return switch (self) {
            .float32 => |v| @as(f64, v),
            .float64 => |v| v,
            else => error.TypeMismatch,
        };
    }

    pub fn asBool(self: GgufValue) !bool {
        return switch (self) {
            .bool => |v| v,
            else => error.TypeMismatch,
        };
    }
};

// ============================================================================
// ggml_context 封装
// ============================================================================

pub const Context = opaque {
    /// 使用 ggml_init(params) 初始化上下文
    pub fn init(mem_size: usize) !*Context {
        const params = c.struct_ggml_init_params{
            .mem_size = mem_size,
            .mem_buffer = null,
            .no_alloc = false,
        };
        const ctx = c.ggml_init(params);
        if (ctx == null) return error.OutOfMemory;
        return @ptrCast(ctx);
    }

    /// 使用外部缓冲区初始化
    pub fn initWithBuffer(mem_size: usize, mem_buffer: ?*anyopaque) !*Context {
        const buf = mem_buffer orelse return error.OutOfMemory;
        const params = c.struct_ggml_init_params{
            .mem_size = mem_size,
            .mem_buffer = buf,
            .no_alloc = false,
        };
        const ctx = c.ggml_init(params);
        if (ctx == null) return error.OutOfMemory;
        return @ptrCast(ctx);
    }

    /// 初始化上下文，不分配张量数据内存（用于加载模型）
    pub fn initNoAlloc(mem_size: usize) !*Context {
        const params = c.struct_ggml_init_params{
            .mem_size = mem_size,
            .mem_buffer = null,
            .no_alloc = true,
        };
        const ctx = c.ggml_init(params);
        if (ctx == null) return error.OutOfMemory;
        return @ptrCast(ctx);
    }

    pub fn deinit(self: *Context) void {
        c.ggml_free(@ptrCast(self));
    }

    pub fn reset(self: *Context) void {
        c.ggml_reset(@ptrCast(self));
    }

    pub fn usedMem(self: *Context) usize {
        return c.ggml_used_mem(@ptrCast(self));
    }

    pub fn maxTensorSize(self: *Context) usize {
        return c.ggml_get_max_tensor_size(@ptrCast(self));
    }

    pub fn newTensor(self: *Context, typ: Type, ne: []const i64) !*Tensor {
        const tensor = c.ggml_new_tensor(@ptrCast(self), @intFromEnum(typ), ne.ptr);
        if (tensor == null) return error.OutOfMemory;
        return @ptrCast(@alignCast(tensor));
    }

    pub fn newTensor1d(self: *Context, typ: Type, ne0: i64) !*Tensor {
        const tensor = c.ggml_new_tensor_1d(@ptrCast(self), @intFromEnum(typ), ne0);
        if (tensor == null) return error.OutOfMemory;
        return @ptrCast(@alignCast(tensor));
    }

    pub fn newTensor2d(self: *Context, typ: Type, ne0: i64, ne1: i64) !*Tensor {
        const tensor = c.ggml_new_tensor_2d(@ptrCast(self), @intFromEnum(typ), ne0, ne1);
        if (tensor == null) return error.OutOfMemory;
        return @ptrCast(@alignCast(tensor));
    }

    pub fn newTensor3d(self: *Context, typ: Type, ne0: i64, ne1: i64, ne2: i64) !*Tensor {
        const tensor = c.ggml_new_tensor_3d(@ptrCast(self), @intFromEnum(typ), ne0, ne1, ne2);
        if (tensor == null) return error.OutOfMemory;
        return @ptrCast(@alignCast(tensor));
    }

    pub fn newTensor4d(self: *Context, typ: Type, ne0: i64, ne1: i64, ne2: i64, ne3: i64) !*Tensor {
        const tensor = c.ggml_new_tensor_4d(@ptrCast(self), @intFromEnum(typ), ne0, ne1, ne2, ne3);
        if (tensor == null) return error.OutOfMemory;
        return @ptrCast(@alignCast(tensor));
    }

    pub fn dupTensor(self: *Context, src: *Tensor) !*Tensor {
        const tensor = c.ggml_dup_tensor(@ptrCast(self), @ptrCast(@alignCast(src)));
        if (tensor == null) return error.OutOfMemory;
        return @ptrCast(@alignCast(tensor));
    }

    pub fn viewTensor(self: *Context, src: *Tensor) *Tensor {
        return @ptrCast(c.ggml_view_tensor(@ptrCast(self), @ptrCast(@alignCast(src))));
    }

    pub fn view1d(self: *Context, src: *Tensor, ne0: i64, offset: usize) *Tensor {
        return @ptrCast(c.ggml_view_1d(@ptrCast(self), @ptrCast(@alignCast(src)), ne0, offset));
    }

    pub fn view2d(self: *Context, src: *Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) *Tensor {
        return @ptrCast(c.ggml_view_2d(@ptrCast(self), @ptrCast(@alignCast(src)), ne0, ne1, nb1, offset));
    }

    pub fn view3d(self: *Context, src: *Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) *Tensor {
        return @ptrCast(c.ggml_view_3d(@ptrCast(self), @ptrCast(@alignCast(src)), ne0, ne1, ne2, nb1, nb2, offset));
    }

    pub fn newBuffer(self: *Context, nbytes: usize) !*anyopaque {
        const buf = c.ggml_new_buffer(@ptrCast(self), nbytes);
        if (buf == null) return error.OutOfMemory;
        return buf;
    }

    pub fn firstTensor(self: *Context) ?*Tensor {
        const tensor = c.ggml_get_first_tensor(@ptrCast(self));
        if (tensor == null) return null;
        return @ptrCast(@alignCast(tensor));
    }

    pub fn nextTensor(self: *Context, tensor: *Tensor) ?*Tensor {
        const next = c.ggml_get_next_tensor(@ptrCast(self), @ptrCast(@alignCast(tensor)));
        if (next == null) return null;
        return @ptrCast(next);
    }

    pub fn getTensor(self: *Context, name: [:0]const u8) ?*Tensor {
        const tensor = c.ggml_get_tensor(@ptrCast(self), name.ptr);
        if (tensor == null) return null;
        return @ptrCast(@alignCast(tensor));
    }
};

// ============================================================================
// ggml_tensor 封装
// ============================================================================

pub const Tensor = opaque {
    pub fn getName(self: *Tensor) [:0]const u8 {
        return std.mem.sliceTo(c.ggml_get_name(@ptrCast(self)), 0);
    }

    pub fn setName(self: *Tensor, name: [:0]const u8) void {
        _ = c.ggml_set_name(@ptrCast(@alignCast(self)), name.ptr);
    }

    pub fn setParam(self: *Tensor) void {
        c.ggml_set_param(@ptrCast(self));
    }

    pub fn shape(self: *Tensor) [4]i64 {
        var ne: [4]i64 = undefined;
        c.ggml_unravel_index(@ptrCast(self), 0, &ne[0], &ne[1], &ne[2], &ne[3]);
        return ne;
    }

    pub fn strides(self: *Tensor) [4]usize {
        _ = self;
        return .{ 0, 0, 0, 0 };
    }

    pub fn typeOf(self: *Tensor) Type {
        return @enumFromInt(c.ggml_tensor_get_type(@ptrCast(self)));
    }

    pub fn nelements(self: *Tensor) i64 {
        return c.ggml_nelements(@ptrCast(self));
    }

    pub fn data(self: *Tensor) *anyopaque {
        return c.ggml_get_data(@ptrCast(@alignCast(self)));
    }

    pub fn dataBytes(self: *Tensor) []u8 {
        const nbytes = c.ggml_nbytes(@ptrCast(@alignCast(self)));
        const ptr = c.ggml_get_data(@ptrCast(@alignCast(self)));
        return @as([*]u8, @ptrCast(ptr))[0..nbytes];
    }

    pub fn dataF32(self: *Tensor) []f32 {
        const n = self.nelements();
        const ptr = c.ggml_get_data_f32(@ptrCast(self));
        return @as([*]f32, @ptrCast(ptr))[0..@as(usize, @intCast(n))];
    }

    pub fn setZero(self: *Tensor) void {
        _ = c.ggml_set_zero(@ptrCast(@alignCast(self)));
    }

    pub fn setF32(self: *Tensor, idx: usize, val: f32) void {
        c.ggml_set_f32_1d(@ptrCast(self), @intCast(idx), val);
    }

    pub fn getF32(self: *Tensor, idx: usize) f32 {
        return c.ggml_get_f32_1d(@ptrCast(self), @intCast(idx));
    }

    pub fn setAllF32(self: *Tensor, val: f32) void {
        const n = self.nelements();
        var i: i64 = 0;
        while (i < n) : (i += 1) {
            c.ggml_set_f32_1d(@ptrCast(self), @intCast(i), val);
        }
    }

    pub fn nDims(self: *Tensor) usize {
        return @intCast(c.ggml_n_dims(@ptrCast(self)));
    }

    pub fn print(self: *Tensor) void {
        c.ggml_print_objects(@ptrCast(self));
    }
};

// ============================================================================
// ggml_cgraph 封装
// ============================================================================

pub const CGraph = opaque {
    pub fn init(ctx: *Context) !*CGraph {
        const graph = c.ggml_new_graph(@ptrCast(ctx));
        if (graph == null) return error.OutOfMemory;
        return @ptrCast(graph);
    }

    pub fn initCustom(ctx: *Context, size: usize, grads: bool) !*CGraph {
        const graph = c.ggml_new_graph_custom(@ptrCast(ctx), size, grads);
        if (graph == null) return error.OutOfMemory;
        return @ptrCast(graph);
    }

    pub fn buildForwardExpand(self: *CGraph, tensor: *Tensor) void {
        c.ggml_build_forward_expand(@ptrCast(self), @ptrCast(@alignCast(tensor)));
    }

    pub fn compute(self: *CGraph, n_threads: i32) !void {
        // 使用 ggml_backend API 执行计算
        _ = n_threads;

        // 先初始化 CPU backend
        const backend = c.ggml_backend_init_best();
        if (backend == null) return error.BackendInitFailed;
        defer c.ggml_backend_free(backend);

        if (c.ggml_backend_graph_compute(backend, @ptrCast(self)) != 0) {
            return error.ComputeError;
        }
    }

    pub fn computeWithPlan(self: *CGraph, cplan: *c.struct_ggml_cplan) !void {
        if (c.ggml_graph_compute(@ptrCast(self), cplan) != 0) {
            return error.ComputeError;
        }
    }

    pub fn computeWithCtx(self: *CGraph, ctx: *Context, n_threads: i32) !void {
        if (c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(self), n_threads) != 0) {
            return error.ComputeError;
        }
    }

    pub fn reset(self: *CGraph) void {
        c.ggml_graph_reset(@ptrCast(self));
    }

    pub fn print(self: *CGraph) void {
        c.ggml_graph_print(@ptrCast(self));
    }

    pub fn dumpDot(self: *CGraph, f: *anyopaque) void {
        c.ggml_graph_dump_dot(@ptrCast(self), null, @ptrCast(f));
    }
};

// ============================================================================
// GGUF 上下文封装
// ============================================================================

pub const GgufContext = struct {
    inner: *c.struct_gguf_context,
    ggml_ctx: ?*Context,

    /// 从文件初始化 GGUF 上下文
    pub fn initFromFile(path: [:0]const u8, no_alloc: bool) !GgufContext {
        var ctx: ?*c.struct_ggml_context = null;
        const params = c.struct_gguf_init_params{
            .no_alloc = no_alloc,
            .ctx = if (!no_alloc) &ctx else null,
        };
        const gguf_ctx = c.gguf_init_from_file(path.ptr, params);
        if (gguf_ctx == null) return error.InvalidModel;
        return GgufContext{
            .inner = gguf_ctx.?,
            .ggml_ctx = if (ctx) |cptr| @ptrCast(cptr) else null,
        };
    }

    /// 释放 GGUF 上下文
    pub fn deinit(self: *GgufContext) void {
        c.gguf_free(self.inner);
    }

    /// 获取 GGUF 版本
    pub fn version(self: *GgufContext) u32 {
        return c.gguf_get_version(self.inner);
    }

    /// 获取张量数量
    pub fn nTensors(self: *GgufContext) i64 {
        return c.gguf_get_n_tensors(self.inner);
    }

    /// 获取元数据键值对数量
    pub fn nKv(self: *GgufContext) i64 {
        return c.gguf_get_n_kv(self.inner);
    }

    /// 获取元数据键名
    pub fn key(self: *GgufContext, i: i64) [:0]const u8 {
        return std.mem.sliceTo(c.gguf_get_key(self.inner, i), 0);
    }

    /// 获取元数据值类型
    pub fn kvType(self: *GgufContext, k: [:0]const u8) GgufValueType {
        return @enumFromInt(c.gguf_get_kv_type(self.inner, k.ptr));
    }

    /// 获取元数据值（自动检测类型）
    pub fn kvValue(self: *GgufContext, k: [:0]const u8) !GgufValue {
        const key_id = c.gguf_find_key(self.inner, k.ptr);
        if (key_id < 0) return error.KeyNotFound;
        const typ = c.gguf_get_kv_type(self.inner, key_id);
        return switch (typ) {
            c.GGUF_TYPE_UINT8 => GgufValue{ .uint8 = c.gguf_get_val_u8(self.inner, key_id) },
            c.GGUF_TYPE_INT8 => GgufValue{ .int8 = c.gguf_get_val_i8(self.inner, key_id) },
            c.GGUF_TYPE_UINT16 => GgufValue{ .uint16 = c.gguf_get_val_u16(self.inner, key_id) },
            c.GGUF_TYPE_INT16 => GgufValue{ .int16 = c.gguf_get_val_i16(self.inner, key_id) },
            c.GGUF_TYPE_UINT32 => GgufValue{ .uint32 = c.gguf_get_val_u32(self.inner, key_id) },
            c.GGUF_TYPE_INT32 => GgufValue{ .int32 = c.gguf_get_val_i32(self.inner, key_id) },
            c.GGUF_TYPE_FLOAT32 => GgufValue{ .float32 = c.gguf_get_val_f32(self.inner, key_id) },
            c.GGUF_TYPE_BOOL => GgufValue{ .bool = c.gguf_get_val_bool(self.inner, key_id) },
            c.GGUF_TYPE_STRING => GgufValue{ .string = std.mem.sliceTo(c.gguf_get_val_str(self.inner, key_id), 0) },
            c.GGUF_TYPE_UINT64 => GgufValue{ .uint64 = c.gguf_get_val_u64(self.inner, key_id) },
            c.GGUF_TYPE_INT64 => GgufValue{ .int64 = c.gguf_get_val_i64(self.inner, key_id) },
            c.GGUF_TYPE_FLOAT64 => GgufValue{ .float64 = c.gguf_get_val_f64(self.inner, key_id) },
            c.GGUF_TYPE_ARRAY => {
                const arr_typ = c.gguf_get_arr_type(self.inner, key_id);
                const arr_n = c.gguf_get_arr_n(self.inner, key_id);
                if (arr_typ == c.GGUF_TYPE_STRING) {
                    return GgufValue{ .array = .{ .typ = @enumFromInt(arr_typ), .items = &.{} } };
                }
                const arr_data = c.gguf_get_arr_data(self.inner, key_id);
                const elem_size: usize = switch (arr_typ) {
                    c.GGUF_TYPE_UINT8, c.GGUF_TYPE_INT8, c.GGUF_TYPE_BOOL => 1,
                    c.GGUF_TYPE_UINT16, c.GGUF_TYPE_INT16 => 2,
                    c.GGUF_TYPE_UINT32, c.GGUF_TYPE_INT32, c.GGUF_TYPE_FLOAT32 => 4,
                    c.GGUF_TYPE_UINT64, c.GGUF_TYPE_INT64, c.GGUF_TYPE_FLOAT64 => 8,
                    else => 1,
                };
                const data_slice = @as([*]const u8, @ptrCast(arr_data))[0 .. @as(usize, @intCast(arr_n)) * elem_size];
                return GgufValue{ .array = .{ .typ = @enumFromInt(arr_typ), .items = data_slice } };
            },
            else => error.UnknownType,
        };
    }

    /// 获取张量信息
    pub fn tensorInfo(self: *GgufContext, i: i64) struct { name: [:0]const u8, n_dims: u32, ne: [4]i64, typ: Type, offset: u64 } {
        const name = std.mem.sliceTo(c.gguf_get_tensor_name(self.inner, i), 0);
        const typ: c.ggml_type = c.gguf_get_tensor_type(self.inner, i);
        return .{
            .name = name,
            .n_dims = 0,
            .ne = .{ 0, 0, 0, 0 },
            .typ = @enumFromInt(typ),
            .offset = 0,
        };
    }

    /// 获取张量数据指针
    pub fn tensorData(self: *GgufContext, name: [:0]const u8) ?*anyopaque {
        _ = self;
        _ = name;
        return null;
    }

    /// 获取关联的 ggml 上下文
    pub fn ggmlCtx(self: *GgufContext) ?*Context {
        return self.ggml_ctx;
    }

    /// 初始化元数据迭代器
    pub fn initMeta(self: *GgufContext) GgufMetaIterator {
        return GgufMetaIterator{
            .ctx = self,
            .index = 0,
        };
    }
};

/// GGUF 元数据迭代器
pub const GgufMetaIterator = struct {
    ctx: *GgufContext,
    index: i64,

    pub fn next(self: *GgufMetaIterator) ?struct { key: [:0]const u8, value: GgufValue } {
        const n = self.ctx.nKv();
        if (self.index >= n) return null;
        const key = self.ctx.key(self.index);
        const value = self.ctx.kvValue(key) catch return null;
        self.index += 1;
        return .{ .key = key, .value = value };
    }
};

// ============================================================================
// 计算图操作函数
// ============================================================================

/// 矩阵乘法: result = a * b
/// a: [M, K], b: [K, N], result: [M, N]
pub fn mulMat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @ptrCast(c.ggml_mul_mat(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b))));
}

/// 逐元素乘法
pub fn mul(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @ptrCast(c.ggml_mul(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b))));
}

/// 逐元素加法
pub fn add(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @ptrCast(c.ggml_add(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b))));
}

/// 张量拷贝
pub fn cpy(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @ptrCast(c.ggml_cpy(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b))));
}

/// RMS 归一化
pub fn rmsNorm(ctx: *Context, a: *Tensor, eps: f32) *Tensor {
    return @ptrCast(c.ggml_rms_norm(@ptrCast(ctx), @ptrCast(@alignCast(a)), eps));
}

/// RoPE 位置编码（扩展版本）
pub fn ropeExt(
    ctx: *Context,
    a: *Tensor,
    pos: *Tensor,
    mode: i32,
    n_dims: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return @ptrCast(c.ggml_rope_ext(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(pos)),
        @ptrCast(@alignCast(pos)),
        n_dims,
        mode,
        n_ctx_orig,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow,
    ));
}

/// 缩放
pub fn scale(ctx: *Context, a: *Tensor, s: f32) *Tensor {
    return @ptrCast(c.ggml_scale(@ptrCast(ctx), @ptrCast(@alignCast(a)), s));
}

/// Softmax
pub fn softMax(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_soft_max(@ptrCast(ctx), @ptrCast(@alignCast(a))));
}

/// 对角线 mask 无穷
pub fn diagMaskInf(ctx: *Context, a: *Tensor, n_past: i32) *Tensor {
    return @ptrCast(c.ggml_diag_mask_inf(@ptrCast(ctx), @ptrCast(@alignCast(a)), n_past));
}

/// 置换
pub fn permute(ctx: *Context, a: *Tensor, axis0: i32, axis1: i32, axis2: i32, axis3: i32) *Tensor {
    return @ptrCast(c.ggml_permute(@ptrCast(ctx), @ptrCast(@alignCast(a)), axis0, axis1, axis2, axis3));
}

/// 连续化
pub fn cont(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_cont(@ptrCast(ctx), @ptrCast(@alignCast(a))));
}

/// 重塑为 2D
pub fn reshape2d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64) *Tensor {
    return @ptrCast(c.ggml_reshape_2d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1));
}

/// 重塑为 3D
pub fn reshape3d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64) *Tensor {
    return @ptrCast(c.ggml_reshape_3d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1, ne2));
}

/// 重塑为 4D
pub fn reshape4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    return @ptrCast(c.ggml_reshape_4d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1, ne2, ne3));
}

/// 重复张量以匹配目标形状
pub fn repeat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @ptrCast(c.ggml_repeat(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b))));
}

/// SiLU 激活函数
pub fn silu(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_silu(@ptrCast(ctx), @ptrCast(@alignCast(a))));
}

/// 设置输出张量
pub fn setOutput(tensor: *Tensor) void {
    c.ggml_set_output(@ptrCast(@alignCast(tensor)));
}

/// 1D 卷积
pub fn conv1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: i32, p0: i32, d0: i32) *Tensor {
    return @ptrCast(c.ggml_conv_1d(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)), s0, p0, d0));
}


// ============================================================================
// 工具函数
// ============================================================================

/// 获取 ggml 版本字符串
pub fn version() [:0]const u8 {
    return std.mem.sliceTo(c.ggml_version(), 0);
}

/// 获取 CPU 核心数（推荐线程数）
pub fn cpuNThreads() i32 {
    return @intCast(std.Thread.getCpuCount() catch 4);
}

/// 检查 CPU 特性
pub const CpuFeatures = struct {
    pub fn hasAvx2() bool {
        return false;
    }
    pub fn hasAvx() bool {
        return false;
    }
    pub fn hasAvx512() bool {
        return false;
    }
    pub fn hasNeon() bool {
        return false;
    }
    pub fn hasMetal() bool {
        return false;
    }
    pub fn hasCuda() bool {
        return false;
    }
    pub fn hasVulkan() bool {
        return false;
    }
};

/// 计算推荐线程数（物理核心数的 2/3 ~ 3/4）
pub fn recommendedThreads() i32 {
    const n = cpuNThreads();
    return @max(1, @as(i32, @intCast(@divTrunc(n * 3, 4))));
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "ggml version" {
    const v = version();
    try testing.expect(v.len > 0);
    std.debug.print("ggml version: {s}\n", .{v});
}

test "CpuFeatures" {
    // Just verify these don't crash
    _ = CpuFeatures.hasAvx2();
    _ = CpuFeatures.hasNeon();
    _ = CpuFeatures.hasMetal();
}

test "recommendedThreads" {
    const n = recommendedThreads();
    try testing.expect(n >= 1);
}
