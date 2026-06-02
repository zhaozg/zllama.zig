//! ggml.zig - 安全封装层
//!
//! 提供 ggml C API 的类型安全 Zig 封装。
//! 所有分配类操作返回 `!*T` 错误联合，纯计算操作返回 `*T`。
//! 使用 `opaque {}` 类型包装不透明指针。

const std = @import("std");
const log = std.log.scoped(.ggml);

const builtin = @import("builtin");

// ============================================================================
// 全局状态
// ============================================================================

var backends_loaded: bool = false;

// ============================================================================
// 原始 C API
// ============================================================================

pub const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-cpu.h");
    @cInclude("ggml-backend.h");
    @cInclude("ggml-alloc.h");
    @cInclude("gguf.h");
});

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

    pub fn setNoAlloc(self: *Context, no_alloc: bool) void {
        c.ggml_set_no_alloc(@ptrCast(self), no_alloc);
    }

    pub fn usedMem(self: *Context) usize {
        return c.ggml_used_mem(@ptrCast(self));
    }

    pub fn maxTensorSize(self: *Context) usize {
        return c.ggml_get_max_tensor_size(@ptrCast(self));
    }

    pub fn newTensor(self: *Context, typ: Type, ne: []const i64) !*Tensor {
        // ggml_new_tensor(ctx, type, n_dims, *ne) — 需要 n_dims 参数
        const tensor = c.ggml_new_tensor(@ptrCast(self), @intFromEnum(typ), @intCast(ne.len), ne.ptr);
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

/// 内部辅助：将 opaque Tensor 指针转换为 C struct 指针
fn asCStruct(tensor: *Tensor) *c.struct_ggml_tensor {
    return @ptrCast(@alignCast(tensor));
}

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

    /// 获取张量形状 [ne0, ne1, ne2, ne3]
    pub fn shape(self: *Tensor) [4]i64 {
        const ct = asCStruct(self);
        return ct.ne;
    }

    /// 获取张量步长（字节）[nb0, nb1, nb2, nb3]
    pub fn strides(self: *Tensor) [4]usize {
        const ct = asCStruct(self);
        return ct.nb;
    }

    /// 获取张量数据类型
    pub fn typeOf(self: *Tensor) Type {
        const ct = asCStruct(self);
        return @enumFromInt(ct.type);
    }

    pub fn nelements(self: *Tensor) i64 {
        return c.ggml_nelements(@ptrCast(@alignCast(self)));
    }

    pub fn ne(self: *Tensor) [4]i64 {
        const ct = asCStruct(self);
        return ct.ne;
    }

    // 设置张量的数据指针（用于 initNoAlloc 模式）
    pub fn setDataPtr(self: *Tensor, ptr: *anyopaque) void {
        var ct = asCStruct(self);
        ct.data = ptr;
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
        const ptr = c.ggml_get_data_f32(@ptrCast(@alignCast(self)));
        return @as([*]f32, @ptrCast(ptr))[0..@as(usize, @intCast(n))];
    }

    pub fn setZero(self: *Tensor) void {
        _ = c.ggml_set_zero(@ptrCast(@alignCast(self)));
    }

    /// 通过直接内存访问设置标量值（ggml 0.13.1 无 ggml_set_f32_1d）
    pub fn setF32(self: *Tensor, idx: usize, val: f32) void {
        const ct = asCStruct(self);
        const ptr = @as([*]f32, @ptrCast(ct.data));
        ptr[idx] = val;
    }

    /// 通过直接内存访问获取标量值（ggml 0.13.1 无 ggml_get_f32_1d）
    pub fn getF32(self: *Tensor, idx: usize) f32 {
        const ct = asCStruct(self);
        const ptr = @as([*]f32, @ptrCast(ct.data));
        return ptr[idx];
    }

    pub fn setAllF32(self: *Tensor, val: f32) void {
        const n = self.nelements();
        const ct = asCStruct(self);
        const ptr = @as([*]f32, @ptrCast(ct.data));
        var i: i64 = 0;
        while (i < n) : (i += 1) {
            ptr[@as(usize, @intCast(i))] = val;
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
            // backend 计算失败，回退到 CPU
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

    // computeWithPlan and computeWithCtx are not available in Homebrew ggml
    // (ggml_graph_compute is in libggml-cpu.so, loaded dynamically via ggml_backend)
    // pub fn computeWithPlan(self: *CGraph, cplan: *c.struct_ggml_cplan) !void {
    //     if (c.ggml_graph_compute(@ptrCast(self), cplan) != 0) {
    //         return error.ComputeError;
    //     }
    // }
    //
    // pub fn computeWithCtx(self: *CGraph, ctx: *Context, n_threads: i32) !void {
    //     if (c.ggml_graph_compute_with_ctx(@ptrCast(ctx), @ptrCast(self), n_threads) != 0) {
    //         return error.ComputeError;
    //     }
    // }

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
// Backend 封装
// ============================================================================
pub const Backend = c.struct_ggml_backend;
pub const BackendBufferType = c.struct_ggml_backend_buffer_type;
pub fn backendCpuInit() !*Backend {
    const backend = c.ggml_backend_cpu_init();
    if (backend == null) return error.OutOfMemory;
    return @ptrCast(backend);
}

pub fn backendCpuBufferType() *BackendBufferType {
    return @ptrCast(c.ggml_backend_cpu_buffer_type());
}

/// 获取 backend 的默认 buffer type
pub fn backendGetDefaultBufferType(backend: *Backend) *BackendBufferType {
    return @ptrCast(c.ggml_backend_get_default_buffer_type(@ptrCast(backend)));
}

/// 为 context 中所有未分配的张量分配内存
/// 权重张量（已通过 setDataPtr 设置）不会被重新分配
pub fn backendAllocCtxTensors(ctx: *Context, backend: *Backend) !void {
    const buf = c.ggml_backend_alloc_ctx_tensors(@ptrCast(ctx), @ptrCast(backend));
    if (buf == null) {
        // 可能所有张量都已分配，这不是错误
        log.debug("backendAllocCtxTensors: no tensors needed allocation (or all already allocated)", .{});
    }
}

/// 使用指定的 buffer type 为 context 中所有未分配的张量分配内存
pub fn backendAllocCtxTensorsFromBuft(ctx: *Context, buft: *BackendBufferType) !void {
    const buf = c.ggml_backend_alloc_ctx_tensors_from_buft(@ptrCast(ctx), @ptrCast(buft));
    if (buf == null) {
        log.debug("backendAllocCtxTensorsFromBuft: no tensors needed allocation", .{});
    }
}

/// 释放 backend
pub fn backendFree(backend: *Backend) void {
    c.ggml_backend_free(@ptrCast(backend));
}

// ============================================================================

// Graph Allocator (gallocr) 封装
// ============================================================================

/// Graph allocator 句柄
pub const Gallocr = opaque {
    /// 创建新的 graph allocator
    pub fn init(buft: *BackendBufferType) !*Gallocr {
        const galloc = c.ggml_gallocr_new(buft);
        if (galloc == null) return error.OutOfMemory;
        return @ptrCast(galloc);
    }

    /// 分配计算图的内存
    pub fn allocGraph(self: *Gallocr, graph: *CGraph) bool {
        return c.ggml_gallocr_alloc_graph(@ptrCast(self), @ptrCast(graph));
    }

    /// 释放 graph allocator
    pub fn free(self: *Gallocr) void {
        c.ggml_gallocr_free(@ptrCast(self));
    }
};

/// 设置输入张量（告诉 allocator 此张量不会被覆盖）
pub fn setInput(tensor: *Tensor) void {
    c.ggml_set_input(@ptrCast(@alignCast(tensor)));
}

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


/// 取负
pub fn neg(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_neg(@ptrCast(ctx), @ptrCast(@alignCast(a))));
}

/// 指数运算
pub fn exp(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_exp(@ptrCast(ctx), @ptrCast(@alignCast(a))));
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
        null, // c: second position tensor (NULL = not used)
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
    const s = a.shape();
    std.log.debug("reshape2d: [{d},{d},{d},{d}] -> [{d},{d}] nelem={d} expected={d}", .{ s[0], s[1], s[2], s[3], ne0, ne1, a.nelements(), ne0 * ne1 });
    return @ptrCast(c.ggml_reshape_2d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1));
}

/// 重塑为 3D
pub fn reshape3d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64) *Tensor {
    const s = a.shape();
    std.log.debug("reshape3d: [{d},{d},{d},{d}] -> [{d},{d},{d}] nelem={d} expected={d}", .{ s[0], s[1], s[2], s[3], ne0, ne1, ne2, a.nelements(), ne0 * ne1 * ne2 });
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


/// Sigmoid 激活函数
pub fn sigmoid(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_sigmoid(@ptrCast(ctx), @ptrCast(@alignCast(a))));
}


/// Softplus 激活函数
pub fn softplus(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_softplus(@ptrCast(ctx), @ptrCast(@alignCast(a))));
}

/// 转置张量
pub fn transpose(ctx: *Context, a: *Tensor) *Tensor {
    return @ptrCast(c.ggml_transpose(@ptrCast(ctx), @ptrCast(@alignCast(a))));
}

/// 设置输出张量
pub fn setOutput(tensor: *Tensor) void {
    c.ggml_set_output(@ptrCast(@alignCast(tensor)));
}

/// 1D 卷积
pub fn conv1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: i32, p0: i32, d0: i32) *Tensor {
    return @ptrCast(c.ggml_conv_1d(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)), s0, p0, d0));
}

/// SSM 卷积（Mamba-2 风格）
/// sx: [d_conv-1+n_t, d_inner, n_s] — 3D 输入（包含历史状态）
/// c:  [d_conv, d_inner] — 2D 卷积核
/// 返回: [d_inner, n_t, n_s] — 3D 输出
pub fn ssmConv(ctx: *Context, sx: *Tensor, kernel: *Tensor) *Tensor {
    return @ptrCast(c.ggml_ssm_conv(@ptrCast(ctx), @ptrCast(@alignCast(sx)), @ptrCast(@alignCast(kernel))));
}


/// SSM Scan 操作
pub fn ssmScan(ctx: *Context, sx: *Tensor, B: *Tensor, C: *Tensor, dt: *Tensor, A: *Tensor, state: *Tensor) *Tensor {
    return @ptrCast(c.ggml_ssm_scan(@ptrCast(ctx), @ptrCast(@alignCast(sx)), @ptrCast(@alignCast(B)), @ptrCast(@alignCast(C)), @ptrCast(@alignCast(dt)), @ptrCast(@alignCast(A)), @ptrCast(@alignCast(state))));
}

/// 沿指定轴拼接两个张量
pub fn concat(ctx: *Context, a: *Tensor, b: *Tensor, axis: i32) *Tensor {
    return @ptrCast(c.ggml_concat(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)), axis));
}
/// 行查找（embedding lookup）
/// a: [n_embd, n_rows] — 数据矩阵
/// b: [n_indices] — 行索引（i32 类型）
/// 返回: [n_embd, n_indices]
pub fn getRows(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @ptrCast(c.ggml_get_rows(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b))));
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

test "conv1d basic" {
    // 测试 ggml_conv_1d 的正确调用方式
    // kernel: [kernel_size, n_channels] -> ne[0]=kernel_size, ne[1]=n_channels
    // data:   [n_tokens, n_channels]    -> ne[0]=n_tokens, ne[1]=n_channels
    const kernel_size: i64 = 4;
    const n_channels: i64 = 6;
    const n_tokens: i64 = 10;

    // 估算内存大小
    const mem_size = 1024 * 1024; // 1MB
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    // 创建 kernel: [kernel_size, n_channels]
    const kernel = try ctx.newTensor2d(.f32, kernel_size, n_channels);
    kernel.setName("test_conv_kernel");
    // 初始化 kernel 数据
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    // 创建 data: [n_tokens, n_channels]
    const data_tensor = try ctx.newTensor2d(.f32, n_tokens, n_channels);
    data_tensor.setName("test_conv_data");
    {
        const data = data_tensor.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    // 执行 conv1d: stride=1, padding=kernel_size-1 (因果卷积), dilation=1
    var graph = try CGraph.init(ctx);
    const result = conv1d(ctx, kernel, data_tensor, 1, @as(i32, @intCast(kernel_size - 1)), 1);
    result.setName("test_conv_result");
    setOutput(result);
    graph.buildForwardExpand(result);

    // 计算
    try graph.compute(1);

    // 验证输出形状: [n_tokens, n_channels]
    const shape = result.shape();
    try testing.expectEqual(n_tokens, shape[0]);
    try testing.expectEqual(n_channels, shape[1]);

    std.debug.print("conv1d test passed: output shape [{d}, {d}]\n", .{ shape[0], shape[1] });
}

test "conv1d with transpose" {
    // 测试 model.zig 中的使用模式:
    // 输入是 [n_channels, n_tokens]，需要转置为 [n_tokens, n_channels] 再调用 conv1d
    const kernel_size: i64 = 4;
    const n_channels: i64 = 6;
    const n_tokens: i64 = 10;

    const mem_size = 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    // 创建 kernel: [kernel_size, n_channels]
    const kernel = try ctx.newTensor2d(.f32, kernel_size, n_channels);
    kernel.setName("test_conv_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    // 创建输入: [n_channels, n_tokens] (类似 model.zig 中的 qkv)
    const input = try ctx.newTensor2d(.f32, n_channels, n_tokens);
    input.setName("test_input");
    {
        const data = input.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    // 转置为 [n_tokens, n_channels]
    var graph = try CGraph.init(ctx);
    const transposed = cont(ctx, permute(ctx, input, 1, 0, 2, 3));
    transposed.setName("test_transposed");

    // conv1d
    const result = conv1d(ctx, kernel, transposed, 1, @as(i32, @intCast(kernel_size - 1)), 1);
    result.setName("test_conv_result");

    // 转置回 [n_channels, n_tokens]
    const result_t = cont(ctx, permute(ctx, result, 1, 0, 2, 3));
    result_t.setName("test_result_t");

    setOutput(result_t);
    graph.buildForwardExpand(result_t);

    try graph.compute(1);

    const shape = result_t.shape();
    try testing.expectEqual(n_channels, shape[0]);
    try testing.expectEqual(n_tokens, shape[1]);

    std.debug.print("conv1d transpose test passed: output shape [{d}, {d}]\n", .{ shape[0], shape[1] });
}

test "ssmConv basic" {
    // 测试 ggml_ssm_conv 的正确调用方式
    // ggml_ssm_conv 要求:
    //   sx: 3D 张量 [d_conv-1+n_t, d_inner, n_s]
    //   c:  2D 矩阵 [d_conv, d_inner]
    // 输出: [d_inner, n_t, n_s]
    const d_conv: i64 = 4;
    const d_inner: i64 = 6;
    const n_t: i64 = 10;
    const n_s: i64 = 1;

    const mem_size = 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    // 创建 kernel c: [d_conv, d_inner]
    const kernel = try ctx.newTensor2d(.f32, d_conv, d_inner);
    kernel.setName("test_ssm_conv_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    // 创建输入 sx: [d_conv-1+n_t, d_inner, n_s]
    const sx = try ctx.newTensor3d(.f32, d_conv - 1 + n_t, d_inner, n_s);
    sx.setName("test_ssm_conv_sx");
    {
        const data = sx.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    // 执行 ssmConv
    var graph = try CGraph.init(ctx);
    const result = ssmConv(ctx, sx, kernel);
    result.setName("test_ssm_conv_result");
    setOutput(result);
    graph.buildForwardExpand(result);

    // 计算
    try graph.compute(1);

    // 验证输出形状: [d_inner, n_t, n_s]
    const shape = result.shape();
    try testing.expectEqual(d_inner, shape[0]);
    try testing.expectEqual(n_t, shape[1]);
    try testing.expectEqual(n_s, shape[2]);

    std.debug.print("ssmConv test passed: output shape [{d}, {d}, {d}]\n", .{ shape[0], shape[1], shape[2] });
}

test "ssmConv with concat and view3d" {
    // 测试 model.zig 中的使用模式:
    // 1. 创建卷积状态 [d_conv-1, d_inner]
    // 2. 创建输入数据 [n_t, d_inner]
    // 3. concat 拼接为 [d_conv-1+n_t, d_inner]
    // 4. view3d 创建 3D 视图 [d_conv-1+n_t, d_inner, 1]
    // 5. ssmConv 执行卷积
    const d_conv: i64 = 4;
    const d_inner: i64 = 6;
    const n_t: i64 = 10;

    const mem_size = 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    // 创建 kernel: [d_conv, d_inner]
    const kernel = try ctx.newTensor2d(.f32, d_conv, d_inner);
    kernel.setName("test_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    // 创建卷积状态: [d_conv-1, d_inner]（零填充）
    const conv_state = try ctx.newTensor2d(.f32, d_conv - 1, d_inner);
    conv_state.setName("test_conv_state");
    conv_state.setZero();

    // 创建输入数据: [n_t, d_inner]
    const input = try ctx.newTensor2d(.f32, n_t, d_inner);
    input.setName("test_input");
    {
        const data = input.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    // concat 拼接: [d_conv-1+n_t, d_inner]
    var graph = try CGraph.init(ctx);
    const concat_result = concat(ctx, conv_state, input, 0);
    concat_result.setName("test_concat");

    // 确保连续
    const concat_cont = cont(ctx, concat_result);
    concat_cont.setName("test_concat_cont");

    // view3d 创建 3D 视图: [d_conv-1+n_t, d_inner, 1]
    const sx_3d = ctx.view3d(concat_cont, d_conv - 1 + n_t, d_inner, 1, @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32))), @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32) * d_inner)), 0);
    sx_3d.setName("test_sx_3d");

    // ssmConv
    const result = ssmConv(ctx, sx_3d, kernel);
    result.setName("test_ssm_conv_result");
    setOutput(result);
    graph.buildForwardExpand(result);

    // 计算
    try graph.compute(1);

    // 验证输出形状: [d_inner, n_t, 1]
    const shape = result.shape();
    try testing.expectEqual(d_inner, shape[0]);
    try testing.expectEqual(n_t, shape[1]);
    try testing.expectEqual(@as(i64, 1), shape[2]);

    std.debug.print("ssmConv concat+view3d test passed: output shape [{d}, {d}, {d}]\n", .{ shape[0], shape[1], shape[2] });
}

test "conv1d vs ssmConv equivalence" {
    // 验证 conv1d 和 ssmConv 在相同输入下产生相同形状的输出
    const d_conv: i64 = 4;
    const d_inner: i64 = 6;
    const n_t: i64 = 10;

    const mem_size = 2 * 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    // 创建 kernel: [d_conv, d_inner]
    const kernel = try ctx.newTensor2d(.f32, d_conv, d_inner);
    kernel.setName("test_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    // 创建输入数据: [n_t, d_inner]
    const input = try ctx.newTensor2d(.f32, n_t, d_inner);
    input.setName("test_input");
    {
        const data = input.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    // conv1d: kernel [d_conv, d_inner], data [n_t, d_inner]
    var graph1 = try CGraph.init(ctx);
    const conv1d_result = conv1d(ctx, kernel, input, 1, @as(i32, @intCast(d_conv - 1)), 1);
    conv1d_result.setName("test_conv1d_result");
    setOutput(conv1d_result);
    graph1.buildForwardExpand(conv1d_result);
    try graph1.compute(1);

    const conv1d_shape = conv1d_result.shape();
    try testing.expectEqual(n_t, conv1d_shape[0]);
    try testing.expectEqual(d_inner, conv1d_shape[1]);

    // ssmConv: 需要先创建带历史状态的输入
    const conv_state = try ctx.newTensor2d(.f32, d_conv - 1, d_inner);
    conv_state.setName("test_conv_state");
    conv_state.setZero();

    var graph2 = try CGraph.init(ctx);
    const concat_result = concat(ctx, conv_state, input, 0);
    concat_result.setName("test_concat");

    const concat_cont = cont(ctx, concat_result);
    concat_cont.setName("test_concat_cont");

    const sx_3d = ctx.view3d(concat_cont, d_conv - 1 + n_t, d_inner, 1, @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32))), @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32) * d_inner)), 0);
    sx_3d.setName("test_sx_3d");

    const ssm_conv_result = ssmConv(ctx, sx_3d, kernel);
    ssm_conv_result.setName("test_ssm_conv_result");
    setOutput(ssm_conv_result);
    graph2.buildForwardExpand(ssm_conv_result);
    try graph2.compute(1);

    const ssm_conv_shape = ssm_conv_result.shape();
    try testing.expectEqual(d_inner, ssm_conv_shape[0]);
    try testing.expectEqual(n_t, ssm_conv_shape[1]);
    try testing.expectEqual(@as(i64, 1), ssm_conv_shape[2]);

    std.debug.print("conv1d vs ssmConv equivalence test passed\n", .{});
    std.debug.print("  conv1d output:  [{d}, {d}]\n", .{ conv1d_shape[0], conv1d_shape[1] });
    std.debug.print("  ssmConv output: [{d}, {d}, {d}]\n", .{ ssm_conv_shape[0], ssm_conv_shape[1], ssm_conv_shape[2] });
}
