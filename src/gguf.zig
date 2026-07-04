//! GGUF 格式解析器（v2/v3）
//!
//! 支持 GGUF v2 和 v3 格式的解析。
//! v3 使用 64 位字段（tensor_count、metadata_kv_count），无填充，张量数据 32 字节对齐。
//! v2 使用 64 位字段，有填充。
//!
//! 复用 src/ggml/ 中的类型定义：
//! - GgufValueType（来自 ggml.c.zig）替代 MetadataValueType
//! - Type（来自 ggml.c.zig）替代 TensorDataType
//!
//! 参考: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const std = @import("std");
const ggml = @import("ggml");

// ============================================================================
// 类型别名（复用 ggml 模块中的类型，保持向后兼容）
// ============================================================================

/// GGUF 元数据值类型标签（复用 ggml.GgufValueType）
pub const MetadataValueType = ggml.GgufValueType;

/// GGUF 张量数据类型（复用 ggml.Type）
pub const TensorDataType = ggml.Type;

// ============================================================================
// GGUF 容器版本
// ============================================================================

/// GGUF 容器版本
pub const GGUFVersion = enum(u32) {
    v1 = 1,
    v2 = 2,
    v3 = 3,
    _,

    pub fn fromInt(v: u32) ?GGUFVersion {
        return switch (v) {
            1 => .v1,
            2 => .v2,
            3 => .v3,
            else => null,
        };
    }
};

// ============================================================================
// GGUF 元数据值（解析后的形式）
// ============================================================================

/// GGUF 元数据值（解析后的形式）
/// 使用 flat struct 而非 tagged union，便于在解析过程中逐步填充。
pub const MetadataValue = struct {
    value_type: MetadataValueType,

    // 联合存储各种类型的值
    uint8_val: u8 = 0,
    int8_val: i8 = 0,
    uint16_val: u16 = 0,
    int16_val: i16 = 0,
    uint32_val: u32 = 0,
    int32_val: i32 = 0,
    float32_val: f32 = 0,
    uint64_val: u64 = 0,
    int64_val: i64 = 0,
    float64_val: f64 = 0,

    bool_val: bool = false,
    string_val: []const u8 = "",
    array_val: []MetadataValue = &[_]MetadataValue{},

    pub fn asString(self: *const MetadataValue) ?[]const u8 {
        return switch (self.value_type) {
            .string => self.string_val,
            else => null,
        };
    }

    pub fn asU32(self: *const MetadataValue) ?u32 {
        return switch (self.value_type) {
            .uint32 => self.uint32_val,
            .int32 => @as(u32, @intCast(self.int32_val)),
            .uint16 => @as(u32, self.uint16_val),
            .int16 => @as(u32, @intCast(self.int16_val)),
            .uint8 => @as(u32, self.uint8_val),
            .int8 => @as(u32, @intCast(self.int8_val)),
            .uint64 => @as(u32, @truncate(self.uint64_val)),
            .int64 => @as(u32, @truncate(@as(u64, @intCast(self.int64_val)))),
            else => null,
        };
    }

    pub fn asF32(self: *const MetadataValue) ?f32 {
        return switch (self.value_type) {
            .float32 => self.float32_val,
            .float64 => @as(f32, @floatCast(self.float64_val)),
            else => null,
        };
    }

    pub fn asBool(self: *const MetadataValue) ?bool {
        return switch (self.value_type) {
            .bool => self.bool_val,
            .uint8 => self.uint8_val != 0,
            .int8 => self.int8_val != 0,
            else => null,
        };
    }

    pub fn asI32(self: *const MetadataValue) ?i32 {
        return switch (self.value_type) {
            .int32 => self.int32_val,
            .uint32 => @as(i32, @bitCast(self.uint32_val)),
            .int16 => @as(i32, self.int16_val),
            .uint16 => @as(i32, self.uint16_val),
            .int8 => @as(i32, self.int8_val),
            .uint8 => @as(i32, self.uint8_val),
            .int64 => @as(i32, @truncate(self.int64_val)),
            .uint64 => @as(i32, @truncate(@as(i64, @bitCast(self.uint64_val)))),
            else => null,
        };
    }
};

// ============================================================================
// GGUF 张量描述符
// ============================================================================

/// GGUF 张量描述符
pub const TensorInfo = struct {
    /// 张量名称
    name: []const u8,
    /// 维度数量
    n_dims: u64,
    /// 各维度大小（最多 4 维）
    dims: [4]u64,
    /// 数据类型（复用 ggml.Type）
    data_type: TensorDataType,
    /// 在文件中的偏移量（相对于张量数据起始位置）
    offset: u64,

    /// 计算张量数据的总字节数
    /// 使用 ggml_row_size 语义：type_size * ne[0] / blck_size * ne[1] * ne[2] * ne[3]
    /// 参考: ggml_row_size(type, ne) = ggml_type_size(type) * ne / ggml_blck_size(type)
    pub fn sizeBytes(self: *const TensorInfo) usize {
        const type_size = self.data_type.sizeOf();
        const blck_size = @as(u64, @intCast(self.data_type.blockSize()));

        // ne[0] 维度按块计算行大小
        const row_size = type_size * self.dims[0] / blck_size;

        // 乘以其余维度
        var total: u64 = row_size;
        for (1..self.n_dims) |i| {
            total *= self.dims[i];
        }
        return @as(usize, total);
    }
};

// ============================================================================
// 解析后的 GGUF 文件
// ============================================================================

/// 解析后的 GGUF 文件
pub const GGUFFile = struct {
    /// GGUF 容器版本
    version: GGUFVersion,
    /// 张量描述符数量
    tensor_count: u64,
    /// 键值元数据映射
    metadata: std.StringHashMapUnmanaged(MetadataValue),
    /// 元数据键的原始顺序（与 metadata 中的键一一对应）
    metadata_keys: []const []const u8,
    /// 张量描述符列表
    tensors: std.ArrayList(TensorInfo),
    /// 张量数据在文件中的起始偏移
    tensor_data_offset: u64,
    /// 原始 GGUF 文件数据（借用自调用者）
    data: []const u8,
    /// 用于 map/list 控制块的分配器（不用于 strings/arrays）
    allocator: std.mem.Allocator,
    /// 持有所有解析后的字符串和数组的 Arena 分配器
    arena: std.heap.ArenaAllocator,

    /// 释放解析文件拥有的所有内存
    pub fn deinit(self: *GGUFFile) void {
        // metadata 和 tensors 的内存由 arena 管理，不需要单独释放
        self.arena.deinit();
        self.* = undefined;
    }

    /// 按 key 获取元数据字符串值
    pub fn getString(self: *const GGUFFile, key: []const u8) ?[]const u8 {
        const val = self.metadata.get(key) orelse return null;
        return val.asString();
    }

    /// 按 key 获取元数据 u32 值
    pub fn getU32(self: *const GGUFFile, key: []const u8) ?u32 {
        const val = self.metadata.get(key) orelse return null;
        return val.asU32();
    }

    /// 按 key 获取元数据 f32 值
    pub fn getF32(self: *const GGUFFile, key: []const u8) ?f32 {
        const val = self.metadata.get(key) orelse return null;
        return val.asF32();
    }

    /// 按 key 获取元数据 bool 值
    pub fn getBool(self: *const GGUFFile, key: []const u8) ?bool {
        const val = self.metadata.get(key) orelse return null;
        return val.asBool();
    }

    /// 按 key 获取元数据 u32 数组值
    pub fn getU32Array(self: *GGUFFile, key: []const u8) ?[]const u32 {
        const val = self.metadata.get(key) orelse return null;
        if (val.value_type != .array) return null;
        const arr = val.array_val;
        var result = std.ArrayList(u32).initCapacity(self.arena.allocator(), 0) catch return null;
        for (arr) |item| {
            if (item.asU32()) |v| {
                result.append(self.arena.allocator(), v) catch return null;
            } else if (item.asI32()) |v| {
                result.append(self.arena.allocator(), @as(u32, @intCast(v))) catch return null;
            } else {
                return null;
            }
        }
        return result.items;
    }

    /// 按 key 获取元数据 bool 数组值
    pub fn getBoolArray(self: *GGUFFile, key: []const u8) ?[]const bool {
        const val = self.metadata.get(key) orelse return null;
        if (val.value_type != .array) return null;
        const arr = val.array_val;
        var result = std.ArrayList(bool).initCapacity(self.arena.allocator(), 0) catch return null;
        for (arr) |item| {
            if (item.asBool()) |v| {
                result.append(self.arena.allocator(), v) catch return null;
            } else {
                return null;
            }
        }
        return result.items;
    }

    /// 按 key 获取元数据 f32 数组值
    pub fn getF32Array(self: *const GGUFFile, key: []const u8, expected_len: usize) ?[]const f32 {
        const val = self.metadata.get(key) orelse return null;
        if (val.value_type != .array) return null;
        const arr = val.array_val;
        if (arr.len != expected_len) return null;
        // Use the arena allocator (cast away const since ArenaAllocator.allocator() takes *ArenaAllocator)
        const arena_ptr = @as(*std.heap.ArenaAllocator, @constCast(&self.arena));
        var result = std.ArrayList(f32).initCapacity(arena_ptr.allocator(), expected_len) catch return null;
        for (arr) |item| {
            if (item.asF32()) |v| {
                result.append(arena_ptr.allocator(), v) catch return null;
            } else {
                return null;
            }
        }
        return result.items;
    }

    /// 按名称查找张量描述符
    pub fn findTensor(self: *const GGUFFile, name: []const u8) ?*const TensorInfo {
        for (self.tensors.items) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    /// 计算所有张量数据的原始总字节数
    pub fn totalTensorDataSize(self: *const GGUFFile) usize {
        var total: usize = 0;
        for (self.tensors.items) |*t| {
            total += t.sizeBytes();
        }
        return total;
    }

    /// 获取张量数据的切片
    pub fn getTensorData(self: *const GGUFFile, info: *const TensorInfo) []const u8 {
        const offset = self.tensor_data_offset + info.offset;
        const size = info.sizeBytes();
        return self.data[offset .. offset + size];
    }
};

const log = std.log.scoped(.gguf);

// ============================================================================
// 内部解析辅助函数
// ============================================================================

/// 读取 GGUF 文件中的字符串（前 8/4 字节长度 + UTF-8 数据）
pub fn readString(data: []const u8, pos: *usize, version: GGUFVersion, arena: *std.heap.ArenaAllocator) ![]const u8 {
    const len = if (version == .v3 or version == .v2) blk: {
        // v2/v3 都使用 64 位长度
        if (pos.* + 8 > data.len) return error.InvalidGGUFFile;
        const l = std.mem.readInt(u64, data[pos.*..][0..8], .little);
        pos.* += 8;
        break :blk l;
    } else blk: {
        // v1 使用 32 位长度
        if (pos.* + 4 > data.len) return error.InvalidGGUFFile;
        const l = std.mem.readInt(u32, data[pos.*..][0..4], .little);
        pos.* += 4;
        break :blk @as(u64, l);
    };

    if (pos.* + @as(usize, @intCast(len)) > data.len) return error.InvalidGGUFFile;
    const str = data[pos.* .. pos.* + @as(usize, @intCast(len))];
    pos.* += @as(usize, @intCast(len));

    // 复制到 arena 中
    const arena_allocator = arena.allocator();
    const copied = try arena_allocator.alloc(u8, str.len);
    @memcpy(copied, str);
    return copied;
}

/// 读取一个元数据值
pub fn readMetadataValue(data: []const u8, pos: *usize, version: GGUFVersion, arena: *std.heap.ArenaAllocator) !MetadataValue {
    if (pos.* + 4 > data.len) return error.InvalidGGUFFile;
    const value_type_int = std.mem.readInt(u32, data[pos.*..][0..4], .little);
    pos.* += 4;
    const value_type: MetadataValueType = @enumFromInt(value_type_int);

    var val = MetadataValue{ .value_type = value_type };

    switch (value_type) {
        .uint8 => {
            val.uint8_val = data[pos.*];
            pos.* += 1;
        },
        .int8 => {
            val.int8_val = @as(i8, @bitCast(data[pos.*]));
            pos.* += 1;
        },
        .uint16 => {
            val.uint16_val = std.mem.readInt(u16, data[pos.*..][0..2], .little);
            pos.* += 2;
        },
        .int16 => {
            val.int16_val = std.mem.readInt(i16, data[pos.*..][0..2], .little);
            pos.* += 2;
        },
        .uint32 => {
            val.uint32_val = std.mem.readInt(u32, data[pos.*..][0..4], .little);
            pos.* += 4;
        },
        .int32 => {
            val.int32_val = std.mem.readInt(i32, data[pos.*..][0..4], .little);
            pos.* += 4;
        },
        .float32 => {
            val.float32_val = @as(f32, @bitCast(std.mem.readInt(u32, data[pos.*..][0..4], .little)));
            pos.* += 4;
        },
        .bool => {
            val.bool_val = data[pos.*] != 0;
            pos.* += 1;
        },
        .string => {
            val.string_val = try readString(data, pos, version, arena);
        },
        .array => {
            const array_type_int = std.mem.readInt(u32, data[pos.*..][0..4], .little);
            pos.* += 4;
            const array_len = if (version == .v3 or version == .v2) blk: {
                const l = std.mem.readInt(u64, data[pos.*..][0..8], .little);
                pos.* += 8;
                break :blk @as(usize, @intCast(l));
            } else blk: {
                const l = std.mem.readInt(u32, data[pos.*..][0..4], .little);
                pos.* += 4;
                break :blk @as(usize, l);
            };

            const arena_allocator = arena.allocator();
            const arr = try arena_allocator.alloc(MetadataValue, array_len);
            for (0..array_len) |i| {
                // 数组元素类型由 array_type_int 指定
                const elem_type: MetadataValueType = @enumFromInt(array_type_int);
                // 读取元素值
                arr[i] = try readMetadataValueOfType(data, pos, elem_type, version, arena);
            }
            val.array_val = arr;
        },
        .uint64 => {
            val.uint64_val = std.mem.readInt(u64, data[pos.*..][0..8], .little);
            pos.* += 8;
        },
        .int64 => {
            val.int64_val = std.mem.readInt(i64, data[pos.*..][0..8], .little);
            pos.* += 8;
        },
        .float64 => {
            val.float64_val = @as(f64, @bitCast(std.mem.readInt(u64, data[pos.*..][0..8], .little)));
            pos.* += 8;
        },
    }

    return val;
}

/// 读取指定类型的元数据值（用于数组元素）
pub fn readMetadataValueOfType(data: []const u8, pos: *usize, value_type: MetadataValueType, version: GGUFVersion, arena: *std.heap.ArenaAllocator) !MetadataValue {
    var val = MetadataValue{ .value_type = value_type };

    switch (value_type) {
        .uint8 => {
            val.uint8_val = data[pos.*];
            pos.* += 1;
        },
        .int8 => {
            val.int8_val = @as(i8, @bitCast(data[pos.*]));
            pos.* += 1;
        },
        .uint16 => {
            val.uint16_val = std.mem.readInt(u16, data[pos.*..][0..2], .little);
            pos.* += 2;
        },
        .int16 => {
            val.int16_val = std.mem.readInt(i16, data[pos.*..][0..2], .little);
            pos.* += 2;
        },
        .uint32 => {
            val.uint32_val = std.mem.readInt(u32, data[pos.*..][0..4], .little);
            pos.* += 4;
        },
        .int32 => {
            val.int32_val = std.mem.readInt(i32, data[pos.*..][0..4], .little);
            pos.* += 4;
        },
        .float32 => {
            val.float32_val = @as(f32, @bitCast(std.mem.readInt(u32, data[pos.*..][0..4], .little)));
            pos.* += 4;
        },
        .bool => {
            val.bool_val = data[pos.*] != 0;
            pos.* += 1;
        },
        .string => {
            val.string_val = try readString(data, pos, version, arena);
        },
        .uint64 => {
            val.uint64_val = std.mem.readInt(u64, data[pos.*..][0..8], .little);
            pos.* += 8;
        },
        .int64 => {
            val.int64_val = std.mem.readInt(i64, data[pos.*..][0..8], .little);
            pos.* += 8;
        },
        .float64 => {
            val.float64_val = @as(f64, @bitCast(std.mem.readInt(u64, data[pos.*..][0..8], .little)));
            pos.* += 8;
        },
        .array => {
            // 嵌套数组暂不支持
            return error.NestedArrayNotSupported;
        },
    }

    return val;
}


pub const parse = @import("gguf_parse.zig").parse;
// 测试
// ============================================================================

const testing = std.testing;

test "GGUFVersion fromInt" {
    try testing.expectEqual(GGUFVersion.v2, GGUFVersion.fromInt(2).?);
    try testing.expectEqual(GGUFVersion.v3, GGUFVersion.fromInt(3).?);
    try testing.expect(GGUFVersion.fromInt(0) == null);
    try testing.expect(GGUFVersion.fromInt(99) == null);
}

test "TensorDataType typeSize (via ggml.Type)" {
    try testing.expectEqual(@as(usize, 4), TensorDataType.f32.sizeOf());
    try testing.expectEqual(@as(usize, 2), TensorDataType.f16.sizeOf());
}

test "parse invalid magic" {
    const data = [_]u8{0} ** 16;
    const result = parse(&data, std.testing.allocator);
    try testing.expectError(error.InvalidGGUFFile, result);
}

test "MetadataValueType is GgufValueType" {
    try testing.expectEqual(@as(u32, 0), @intFromEnum(MetadataValueType.uint8));
    try testing.expectEqual(@as(u32, 4), @intFromEnum(MetadataValueType.uint32));
    try testing.expectEqual(@as(u32, 6), @intFromEnum(MetadataValueType.float32));
    try testing.expectEqual(@as(u32, 7), @intFromEnum(MetadataValueType.bool));
    try testing.expectEqual(@as(u32, 8), @intFromEnum(MetadataValueType.string));
    try testing.expectEqual(@as(u32, 9), @intFromEnum(MetadataValueType.array));
}
