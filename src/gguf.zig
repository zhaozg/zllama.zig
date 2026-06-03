//! GGUF 格式解析器（v2/v3）
//!
//! 支持 GGUF v2 和 v3 格式的解析。
//! v3 使用 64 位字段（tensor_count、metadata_kv_count），无填充，张量数据 32 字节对齐。
//! v2 使用 64 位字段，有填充。
//!
//! 参考: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const std = @import("std");

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

/// GGUF 元数据值类型标签
pub const MetadataValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

/// GGUF 元数据值（解析后的形式）
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
};

/// GGUF 张量数据类型
pub const TensorDataType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    i8 = 16,
    i16 = 17,
    i32 = 18,
    _,

    /// 返回该数据类型每个元素的字节数（近似值，量化类型为块大小平均）
    pub fn typeSize(self: TensorDataType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .q4_0 => 18, // sizeof(block_q4_0) = sizeof(ggml_half) + QK4_0/2 = 2 + 16
            .q4_1 => 20, // sizeof(block_q4_1) = 2*sizeof(ggml_half) + QK4_1/2 = 4 + 16
            .q5_0 => 22, // sizeof(block_q5_0) = sizeof(ggml_half) + sizeof(uint32_t) + QK5_0/2 = 2 + 4 + 16
            .q5_1 => 24, // sizeof(block_q5_1) = 2*sizeof(ggml_half) + sizeof(uint32_t) + QK5_1/2 = 4 + 4 + 16
            .q8_0 => 34, // sizeof(block_q8_0) = sizeof(ggml_half) + QK8_0 = 2 + 32
            .q8_1 => 36, // sizeof(block_q8_1) = 2*sizeof(ggml_half) + QK8_1 = 4 + 32
            .q2_k => 74, // sizeof(block_q2_K)
            .q3_k => 106, // sizeof(block_q3_K)
            .q4_k => 144, // sizeof(block_q4_K) = 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 = 4 + 12 + 128
            .q5_k => 176, // sizeof(block_q5_K)
            .q6_k => 210, // sizeof(block_q6_K)
            .q8_k => 292, // sizeof(block_q8_K)
            else => 4, // 默认返回 f32 大小
        };
    }

    /// 返回该数据类型的块大小
    pub fn blockSize(self: TensorDataType) usize {
        return switch (self) {
            .f32, .f16, .i8, .i16, .i32 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            else => 1,
        };
    }
};

/// GGUF 张量描述符
pub const TensorInfo = struct {
    /// 张量名称
    name: []const u8,
    /// 维度数量
    n_dims: u64,
    /// 各维度大小（最多 4 维）
    dims: [4]u64,
    /// 数据类型
    data_type: TensorDataType,
    /// 在文件中的偏移量（相对于张量数据起始位置）
    offset: u64,

    /// 计算张量数据的总字节数
    pub fn sizeBytes(self: *const TensorInfo) usize {
        var n_elems: u64 = 1;
        for (0..self.n_dims) |i| {
            n_elems *= self.dims[i];
        }
        const block_size = self.data_type.blockSize();
        const type_size = self.data_type.typeSize();
        // 对于量化类型，需要按块计算
        if (block_size > 1) {
            const n_blocks = (n_elems + @as(u64, @intCast(block_size)) - 1) / @as(u64, @intCast(block_size));
            return @as(usize, @intCast(n_blocks * @as(u64, @intCast(type_size))));
        }
        return @as(usize, @intCast(n_elems * @as(u64, @intCast(type_size))));
    }
};

/// 解析后的 GGUF 文件
pub const GGUFFile = struct {
    /// GGUF 容器版本
    version: GGUFVersion,
    /// 张量描述符数量
    tensor_count: u64,
    /// 键值元数据映射
    metadata: std.StringHashMapUnmanaged(MetadataValue),
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

    /// 按名称查找张量描述符
    pub fn findTensor(self: *const GGUFFile, name: []const u8) ?*const TensorInfo {
        for (self.tensors.items) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    /// 获取张量数据的切片
    pub fn getTensorData(self: *const GGUFFile, info: *const TensorInfo) []const u8 {
        const offset = self.tensor_data_offset + info.offset;
        const size = info.sizeBytes();
        return self.data[offset .. offset + size];
    }
};

const log = std.log.scoped(.gguf);

/// 读取 GGUF 文件中的字符串（前 8/4 字节长度 + UTF-8 数据）
fn readString(data: []const u8, pos: *usize, version: GGUFVersion, arena: *std.heap.ArenaAllocator) ![]const u8 {
    const len = if (version == .v3 or version == .v2) blk: {
        // v2/v3 都使用 64 位长度
        const l = std.mem.readInt(u64, data[pos.*..][0..8], .little);
        pos.* += 8;
        break :blk l;
    } else blk: {
        // v1 使用 32 位长度
        const l = std.mem.readInt(u32, data[pos.*..][0..4], .little);
        pos.* += 4;
        break :blk @as(u64, l);
    };

    const str = data[pos.* .. pos.* + @as(usize, @intCast(len))];
    pos.* += @as(usize, @intCast(len));

    // 复制到 arena 中
    const arena_allocator = arena.allocator();
    const copied = try arena_allocator.alloc(u8, str.len);
    @memcpy(copied, str);
    return copied;
}

/// 读取一个元数据值
fn readMetadataValue(data: []const u8, pos: *usize, version: GGUFVersion, arena: *std.heap.ArenaAllocator) !MetadataValue {
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
        else => unreachable,
    }

    return val;
}

/// 读取指定类型的元数据值（用于数组元素）
fn readMetadataValueOfType(data: []const u8, pos: *usize, value_type: MetadataValueType, version: GGUFVersion, arena: *std.heap.ArenaAllocator) !MetadataValue {
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
        else => unreachable,
    }

    return val;
}

/// 解析 GGUF 文件
/// @param data 完整的 GGUF 文件数据
/// @param allocator 分配器
/// @returns 解析后的 GGUFFile
pub fn parse(data: []const u8, allocator: std.mem.Allocator) !GGUFFile {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    var pos: usize = 0;

    // 读取魔数 "GGUF"
    if (data.len < 4) return error.InvalidGGUFFile;
    const magic = data[0..4];
    if (!std.mem.eql(u8, magic, "GGUF")) {
        return error.InvalidGGUFFile;
    }
    pos += 4;

    // 读取版本号
    if (data.len < pos + 4) return error.InvalidGGUFFile;
    const version_int = std.mem.readInt(u32, data[pos..][0..4], .little);
    pos += 4;

    const version = GGUFVersion.fromInt(version_int) orelse return error.UnsupportedGGUFVersion;

    // 读取 tensor_count 和 metadata_kv_count
    // v3: 64位，v2: 64位，v1: 32位
    const is_64bit = version == .v3 or version == .v2;

    if (is_64bit) {
        if (data.len < pos + 16) return error.InvalidGGUFFile;
    } else {
        if (data.len < pos + 8) return error.InvalidGGUFFile;
    }

    const tensor_count = if (is_64bit) blk: {
        const tc = std.mem.readInt(u64, data[pos..][0..8], .little);
        pos += 8;
        break :blk tc;
    } else blk: {
        const tc = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;
        break :blk @as(u64, tc);
    };

    const metadata_kv_count = if (is_64bit) blk: {
        const mc = std.mem.readInt(u64, data[pos..][0..8], .little);
        pos += 8;
        break :blk mc;
    } else blk: {
        const mc = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;
        break :blk @as(u64, mc);
    };

    // 解析元数据
    var metadata = std.StringHashMapUnmanaged(MetadataValue){};
    try metadata.ensureTotalCapacity(arena_allocator, @as(u32, @intCast(metadata_kv_count)));

    for (0..@as(usize, @intCast(metadata_kv_count))) |_| {
        const key = try readString(data, &pos, version, &arena);
        const val = try readMetadataValue(data, &pos, version, &arena);
        metadata.putAssumeCapacity(key, val);
    }

    // 解析张量信息
    // 解析张量信息（v3 不需要对齐，只有 tensor data 需要对齐）
    var tensors = std.ArrayList(TensorInfo).empty;
    try tensors.ensureTotalCapacity(arena_allocator, @as(usize, @intCast(tensor_count)));

    for (0..@as(usize, @intCast(tensor_count))) |_| {
        const name = try readString(data, &pos, version, &arena);
        const n_dims = if (is_64bit) blk: {
            const nd = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;
            break :blk nd;
        } else blk: {
            const nd = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;
            break :blk nd;
        };

        var dims: [4]u64 = [_]u64{0} ** 4;
        for (0..@as(usize, @intCast(n_dims))) |i| {
            if (is_64bit) {
                dims[i] = std.mem.readInt(u64, data[pos..][0..8], .little);
                pos += 8;
            } else {
                dims[i] = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
            }
        }

        const data_type_int = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;
        const data_type: TensorDataType = @enumFromInt(data_type_int);

        // v2/v3 使用 64 位偏移
        const offset = if (is_64bit) blk: {
            const off = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            break :blk off;
        } else blk: {
            const off = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;
            break :blk @as(u64, off);
        };

        tensors.appendAssumeCapacity(TensorInfo{
            .name = name,
            .n_dims = n_dims,
            .dims = dims,
            .data_type = data_type,
            .offset = offset,
        });
    }

    // v3 张量数据对齐到 32 字节
    if (version == .v3) {
        pos = std.mem.alignForward(usize, pos, 32);
    }

    const tensor_data_offset = pos;

    return GGUFFile{
        .version = version,
        .tensor_count = tensor_count,
        .metadata = metadata,
        .tensors = tensors,
        .tensor_data_offset = tensor_data_offset,
        .data = data,
        .allocator = allocator,
        .arena = arena,
    };
}

const testing = std.testing;

test "GGUFVersion fromInt" {
    try testing.expectEqual(GGUFVersion.v2, GGUFVersion.fromInt(2).?);
    try testing.expectEqual(GGUFVersion.v3, GGUFVersion.fromInt(3).?);
    try testing.expect(GGUFVersion.fromInt(0) == null);
    try testing.expect(GGUFVersion.fromInt(99) == null);
}

test "TensorDataType typeSize" {
    try testing.expectEqual(@as(usize, 4), TensorDataType.f32.typeSize());
    try testing.expectEqual(@as(usize, 2), TensorDataType.f16.typeSize());
}

test "parse invalid magic" {
    const data = [_]u8{0} ** 16;
    const result = parse(&data, std.testing.allocator);
    try testing.expectError(error.InvalidGGUFFile, result);
}
