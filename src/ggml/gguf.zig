//! gguf.h 的 Zig 绑定层
//!
//! 提供 gguf_context 的类型安全 Zig 封装。
//! 所有分配类操作返回 `!*T` 错误联合，纯查询操作返回普通值。
//! 使用 `opaque {}` 类型包装不透明指针。
//!
//! 参考: deps/ggml/include/gguf.h

const std = @import("std");
const c = @import("c.zig").c;
const Type = @import("c.zig").Type;
const GgufValueType = @import("c.zig").GgufValueType;

const log = std.log.scoped(.gguf_bind);

// ============================================================================
// 不透明类型
// ============================================================================

/// gguf_context 的不透明包装
pub const Context = opaque {
    /// 从文件路径初始化 GGUF 上下文
    /// 内部使用 mmap 或文件读取，由 ggml 实现决定
    pub fn initFromFile(fname: [:0]const u8, params: InitParams) !*Context {
        const c_params = c.struct_gguf_init_params{
            .no_alloc = params.no_alloc,
            .ctx = if (params.ctx) |ctx_ptr| @ptrCast(ctx_ptr) else null,
        };
        const result = c.gguf_init_from_file(fname.ptr, c_params);
        if (result == null) return error.GGUFInitFailed;
        return @as(*Context, @ptrCast(result));
    }

    /// 从内存缓冲区初始化 GGUF 上下文
    /// 数据不会被复制，通过回调按需读取
    pub fn initFromBuffer(data: []const u8, params: InitParams) !*Context {
        const c_params = c.struct_gguf_init_params{
            .no_alloc = params.no_alloc,
            .ctx = if (params.ctx) |ctx_ptr| @ptrCast(ctx_ptr) else null,
        };
        const result = c.gguf_init_from_buffer(data.ptr, data.len, c_params);
        if (result == null) return error.GGUFInitFailed;
        return @as(*Context, @ptrCast(result));
    }

    /// 释放 GGUF 上下文
    pub fn deinit(self: *Context) void {
        c.gguf_free(@ptrCast(self));
    }

    // -------------------------------------------------------------------------
    // 元数据查询
    // -------------------------------------------------------------------------

    /// 获取 GGUF 版本
    pub fn getVersion(self: *const Context) u32 {
        return c.gguf_get_version(@ptrCast(self));
    }

    /// 获取对齐字节数
    pub fn getAlignment(self: *const Context) usize {
        return c.gguf_get_alignment(@ptrCast(self));
    }

    /// 获取张量数据在文件中的偏移量（已对齐）
    pub fn getDataOffset(self: *const Context) usize {
        return c.gguf_get_data_offset(@ptrCast(self));
    }

    /// 获取 KV 元数据数量
    pub fn getNKv(self: *const Context) i64 {
        return c.gguf_get_n_kv(@ptrCast(self));
    }

    /// 按 key 查找元数据，返回 key_id（-1 表示未找到）
    pub fn findKey(self: *const Context, key: [:0]const u8) i64 {
        return c.gguf_find_key(@ptrCast(self), key.ptr);
    }

    /// 按 key_id 获取 key 名称
    pub fn getKey(self: *const Context, key_id: i64) [:0]const u8 {
        return std.mem.sliceTo(c.gguf_get_key(@ptrCast(self), key_id), 0);
    }

    /// 按 key_id 获取值类型
    pub fn getKvType(self: *const Context, key_id: i64) GgufValueType {
        return @enumFromInt(c.gguf_get_kv_type(@ptrCast(self), key_id));
    }

    /// 按 key_id 获取数组元素类型（仅对 array 类型有效）
    pub fn getArrType(self: *const Context, key_id: i64) GgufValueType {
        return @enumFromInt(c.gguf_get_arr_type(@ptrCast(self), key_id));
    }

    // -------------------------------------------------------------------------
    // 按 key_id 获取元数据值
    // -------------------------------------------------------------------------

    pub fn getValU8(self: *const Context, key_id: i64) u8 {
        return c.gguf_get_val_u8(@ptrCast(self), key_id);
    }

    pub fn getValI8(self: *const Context, key_id: i64) i8 {
        return c.gguf_get_val_i8(@ptrCast(self), key_id);
    }

    pub fn getValU16(self: *const Context, key_id: i64) u16 {
        return c.gguf_get_val_u16(@ptrCast(self), key_id);
    }

    pub fn getValI16(self: *const Context, key_id: i64) i16 {
        return c.gguf_get_val_i16(@ptrCast(self), key_id);
    }

    pub fn getValU32(self: *const Context, key_id: i64) u32 {
        return c.gguf_get_val_u32(@ptrCast(self), key_id);
    }

    pub fn getValI32(self: *const Context, key_id: i64) i32 {
        return c.gguf_get_val_i32(@ptrCast(self), key_id);
    }

    pub fn getValF32(self: *const Context, key_id: i64) f32 {
        return c.gguf_get_val_f32(@ptrCast(self), key_id);
    }

    pub fn getValU64(self: *const Context, key_id: i64) u64 {
        return c.gguf_get_val_u64(@ptrCast(self), key_id);
    }

    pub fn getValI64(self: *const Context, key_id: i64) i64 {
        return c.gguf_get_val_i64(@ptrCast(self), key_id);
    }

    pub fn getValF64(self: *const Context, key_id: i64) f64 {
        return c.gguf_get_val_f64(@ptrCast(self), key_id);
    }

    pub fn getValBool(self: *const Context, key_id: i64) bool {
        return c.gguf_get_val_bool(@ptrCast(self), key_id);
    }

    pub fn getValStr(self: *const Context, key_id: i64) [:0]const u8 {
        return std.mem.sliceTo(c.gguf_get_val_str(@ptrCast(self), key_id), 0);
    }

    /// 获取值的原始数据指针（用于 array 等复杂类型）
    pub fn getValData(self: *const Context, key_id: i64) ?*const anyopaque {
        return @as(?*const anyopaque, @ptrCast(c.gguf_get_val_data(@ptrCast(self), key_id)));
    }

    /// 获取数组长度
    pub fn getArrN(self: *const Context, key_id: i64) usize {
        return c.gguf_get_arr_n(@ptrCast(self), key_id);
    }

    /// 获取数组数据的原始指针
    pub fn getArrData(self: *const Context, key_id: i64) ?*const anyopaque {
        return @as(?*const anyopaque, @ptrCast(c.gguf_get_arr_data(@ptrCast(self), key_id)));
    }

    /// 获取数组中的第 i 个字符串
    pub fn getArrStr(self: *const Context, key_id: i64, i: usize) [:0]const u8 {
        return std.mem.sliceTo(c.gguf_get_arr_str(@ptrCast(self), key_id, i), 0);
    }

    // -------------------------------------------------------------------------
    // 便捷方法：按 key 直接获取值
    // -------------------------------------------------------------------------

    /// 按 key 获取 u32 值，如果 key 不存在返回 null
    pub fn getU32(self: *const Context, key: [:0]const u8) ?u32 {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValU32(id);
    }

    /// 按 key 获取 i32 值，如果 key 不存在返回 null
    pub fn getI32(self: *const Context, key: [:0]const u8) ?i32 {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValI32(id);
    }

    /// 按 key 获取 f32 值，如果 key 不存在返回 null
    pub fn getF32(self: *const Context, key: [:0]const u8) ?f32 {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValF32(id);
    }

    /// 按 key 获取 bool 值，如果 key 不存在返回 null
    pub fn getBool(self: *const Context, key: [:0]const u8) ?bool {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValBool(id);
    }

    /// 按 key 获取字符串值，如果 key 不存在返回 null
    pub fn getString(self: *const Context, key: [:0]const u8) ?[:0]const u8 {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValStr(id);
    }

    /// 按 key 获取 u64 值，如果 key 不存在返回 null
    pub fn getU64(self: *const Context, key: [:0]const u8) ?u64 {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValU64(id);
    }

    /// 按 key 获取 i64 值，如果 key 不存在返回 null
    pub fn getI64(self: *const Context, key: [:0]const u8) ?i64 {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValI64(id);
    }

    /// 按 key 获取 f64 值，如果 key 不存在返回 null
    pub fn getF64(self: *const Context, key: [:0]const u8) ?f64 {
        const id = self.findKey(key);
        if (id < 0) return null;
        return self.getValF64(id);
    }

    // -------------------------------------------------------------------------
    // 张量信息查询
    // -------------------------------------------------------------------------

    /// 获取张量数量
    pub fn getNTensors(self: *const Context) i64 {
        return c.gguf_get_n_tensors(@ptrCast(self));
    }

    /// 按名称查找张量，返回 tensor_id（-1 表示未找到）
    pub fn findTensor(self: *const Context, name: [:0]const u8) i64 {
        return c.gguf_find_tensor(@ptrCast(self), name.ptr);
    }

    /// 按 tensor_id 获取张量在数据段中的偏移量
    pub fn getTensorOffset(self: *const Context, tensor_id: i64) usize {
        return c.gguf_get_tensor_offset(@ptrCast(self), tensor_id);
    }

    /// 按 tensor_id 获取张量名称
    pub fn getTensorName(self: *const Context, tensor_id: i64) [:0]const u8 {
        return std.mem.sliceTo(c.gguf_get_tensor_name(@ptrCast(self), tensor_id), 0);
    }

    /// 按 tensor_id 获取张量数据类型
    pub fn getTensorType(self: *const Context, tensor_id: i64) Type {
        return @enumFromInt(c.gguf_get_tensor_type(@ptrCast(self), tensor_id));
    }

    /// 按 tensor_id 获取张量数据大小（字节）
    pub fn getTensorSize(self: *const Context, tensor_id: i64) usize {
        return c.gguf_get_tensor_size(@ptrCast(self), tensor_id);
    }

    // -------------------------------------------------------------------------
    // 写入操作（用于创建/修改 GGUF 文件）
    // -------------------------------------------------------------------------

    /// 设置 KV 值（u8）
    pub fn setValU8(self: *Context, key: [:0]const u8, val: u8) void {
        c.gguf_set_val_u8(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（i8）
    pub fn setValI8(self: *Context, key: [:0]const u8, val: i8) void {
        c.gguf_set_val_i8(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（u16）
    pub fn setValU16(self: *Context, key: [:0]const u8, val: u16) void {
        c.gguf_set_val_u16(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（i16）
    pub fn setValI16(self: *Context, key: [:0]const u8, val: i16) void {
        c.gguf_set_val_i16(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（u32）
    pub fn setValU32(self: *Context, key: [:0]const u8, val: u32) void {
        c.gguf_set_val_u32(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（i32）
    pub fn setValI32(self: *Context, key: [:0]const u8, val: i32) void {
        c.gguf_set_val_i32(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（f32）
    pub fn setValF32(self: *Context, key: [:0]const u8, val: f32) void {
        c.gguf_set_val_f32(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（u64）
    pub fn setValU64(self: *Context, key: [:0]const u8, val: u64) void {
        c.gguf_set_val_u64(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（i64）
    pub fn setValI64(self: *Context, key: [:0]const u8, val: i64) void {
        c.gguf_set_val_i64(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（f64）
    pub fn setValF64(self: *Context, key: [:0]const u8, val: f64) void {
        c.gguf_set_val_f64(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（bool）
    pub fn setValBool(self: *Context, key: [:0]const u8, val: bool) void {
        c.gguf_set_val_bool(@ptrCast(self), key.ptr, val);
    }

    /// 设置 KV 值（string）
    pub fn setValStr(self: *Context, key: [:0]const u8, val: [:0]const u8) void {
        c.gguf_set_val_str(@ptrCast(self), key.ptr, val.ptr);
    }

    /// 设置 KV 值（array data）
    pub fn setArrData(self: *Context, key: [:0]const u8, typ: GgufValueType, data: *const anyopaque, n: usize) void {
        c.gguf_set_arr_data(@ptrCast(self), key.ptr, @intFromEnum(typ), @ptrCast(data), n);
    }

    /// 设置 KV 值（array string）
    pub fn setArrStr(self: *Context, key: [:0]const u8, data: []const [:0]const u8) void {
        c.gguf_set_arr_str(@ptrCast(self), key.ptr, @ptrCast(data.ptr), data.len);
    }

    /// 从另一个 context 复制 KV 对
    pub fn setKv(self: *Context, src: *const Context) void {
        c.gguf_set_kv(@ptrCast(self), @ptrCast(src));
    }

    /// 移除指定 key 的 KV 对，返回该 key 之前的 id（-1 表示不存在）
    pub fn removeKey(self: *Context, key: [:0]const u8) i64 {
        return c.gguf_remove_key(@ptrCast(self), key.ptr);
    }

    /// 添加张量
    pub fn addTensor(self: *Context, tensor: *const anyopaque) void {
        c.gguf_add_tensor(@ptrCast(self), @ptrCast(tensor));
    }

    /// 设置张量类型
    pub fn setTensorType(self: *Context, name: [:0]const u8, typ: Type) void {
        c.gguf_set_tensor_type(@ptrCast(self), name.ptr, @intFromEnum(typ));
    }

    /// 设置张量数据
    pub fn setTensorData(self: *Context, name: [:0]const u8, data: *const anyopaque) void {
        c.gguf_set_tensor_data(@ptrCast(self), name.ptr, data);
    }

    /// 写入到文件（通过 FILE*）
    pub fn writeToFilePtr(self: *const Context, file: *anyopaque, only_meta: bool) bool {
        return c.gguf_write_to_file_ptr(@ptrCast(self), @ptrCast(file), only_meta);
    }

    /// 写入到文件路径
    pub fn writeToFile(self: *const Context, fname: [:0]const u8, only_meta: bool) bool {
        return c.gguf_write_to_file(@ptrCast(self), fname.ptr, only_meta);
    }

    /// 获取元数据大小（包括填充）
    pub fn getMetaSize(self: *const Context) usize {
        return c.gguf_get_meta_size(@ptrCast(self));
    }

    /// 写入元数据到指针
    pub fn getMetaData(self: *const Context, data: *anyopaque) void {
        c.gguf_get_meta_data(@ptrCast(self), data);
    }
};

// ============================================================================
// 辅助类型
// ============================================================================

/// gguf_init_params 的 Zig 版本
pub const InitParams = struct {
    /// 如果为 true，不分配张量数据（仅创建空张量）
    no_alloc: bool = false,
    /// 如果非 null，在此 ggml_context 中创建张量
    ctx: ?*?*anyopaque = null,
};

/// 从 GGUF 文件读取的元数据值（便捷包装）
pub const MetadataValue = union(enum) {
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

    pub fn asString(self: MetadataValue) ![]const u8 {
        return switch (self) {
            .string => |s| s,
            else => error.TypeMismatch,
        };
    }

    pub fn asInt(self: MetadataValue) !i64 {
        return switch (self) {
            .int32 => |v| @as(i64, v),
            .int64 => |v| v,
            .uint32 => |v| @as(i64, v),
            .uint64 => |v| @as(i64, @intCast(v)),
            else => error.TypeMismatch,
        };
    }

    pub fn asFloat(self: MetadataValue) !f64 {
        return switch (self) {
            .float32 => |v| @as(f64, v),
            .float64 => |v| v,
            else => error.TypeMismatch,
        };
    }

    pub fn asBool(self: MetadataValue) !bool {
        return switch (self) {
            .bool => |v| v,
            else => error.TypeMismatch,
        };
    }
};

// ============================================================================
// 测试（测试放在 src/tests/test_ggml_gguf.zig 中）
// ============================================================================
