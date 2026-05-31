//! gguf.zig - GGUF 文件格式解析器
//!
//! 支持 GGUF v2 和 v3 格式。
//! v2: 32 位计数字段
//! v3: 64 位计数字段，32 字节对齐
//!
//! 参考: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const std = @import("std");
const ggml = @import("ggml.zig");

// ============================================================================
// 常量
// ============================================================================

/// GGUF 文件魔数
pub const MAGIC: [4]u8 = .{ 'G', 'G', 'U', 'F' };

/// 默认对齐值
pub const DEFAULT_ALIGNMENT: u32 = 32;

/// 支持的 GGUF 版本
pub const SUPPORTED_VERSIONS = [_]i32{ 2, 3 };

// ============================================================================
// 错误类型
// ============================================================================

pub const Error = error{
    InvalidMagic,
    UnsupportedVersion,
    InvalidHeader,
    InvalidMetadata,
    InvalidTensor,
    OutOfMemory,
    IoError,
    KeyNotFound,
    TypeMismatch,
};

// ============================================================================
// 头部结构
// ============================================================================

/// GGUF 头部（v2 和 v3 通用部分）
pub const Header = struct {
    magic: [4]u8,
    version: i32,
    tensor_count: i64,
    metadata_kv_count: i64,
    alignment: u32,
};

/// 张量信息
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    ne: [4]i64,
    typ: ggml.Type,
    offset: u64,
};

/// 元数据值（与 ggml.GgufValue 相同，但使用自有分配器）
pub const MetaValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: ArrayValue,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub fn deinit(self: *MetaValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .string => |s| allocator.free(s),
            .array => |*a| a.deinit(allocator),
            else => {},
        }
    }
};

pub const ArrayValue = struct {
    typ: ggml.GgufValueType,
    items: []const u8,

    pub fn deinit(self: *ArrayValue, allocator: std.mem.Allocator) void {
        allocator.free(self.items);
    }
};

// ============================================================================
// GGUF 解析器
// ============================================================================

pub const Parser = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    header: Header,
    metadata: std.StringHashMap(MetaValue),
    tensors: std.ArrayList(TensorInfo),
    tensor_data_offset: u64,
    file_size: u64,
    file_offset: u64,
    file: ?std.Io.File,

    pub fn init(io: std.Io, allocator: std.mem.Allocator) Parser {
        return .{
            .io = io,
            .allocator = allocator,
            .header = undefined,
            .metadata = std.StringHashMap(MetaValue).init(allocator),
            .tensors = std.ArrayList(TensorInfo).empty,
            .tensor_data_offset = 0,
            .file_size = 0,
            .file_offset = 0,
            .file = null,
        };
    }

    pub fn deinit(self: *Parser) void {
        var meta_iter = self.metadata.iterator();
        while (meta_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.metadata.deinit();
        for (self.tensors.items) |*t| {
            self.allocator.free(t.name);
        }
        self.tensors.deinit(self.allocator);
        if (self.file) |f| {
            f.close(self.io);
        }
    }

    /// 从文件路径解析 GGUF 文件
    pub fn parseFromFile(self: *Parser, path: [:0]const u8) !void {
        self.file = try std.Io.Dir.cwd().openFile(self.io, path, .{ .mode = .read_only });
        const file = self.file.?;

        const stat = try file.stat(self.io);
        const file_size = @as(u64, @intCast(stat.size));
        self.file_size = file_size;

        // 读取整个文件到内存
        const data = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(data);

        var filereader = file.reader(self.io, data);
        const reader: *std.Io.Reader = &filereader.interface;

        try self.parseReader(reader, file_size);
    }


    /// 使用 peekInt + toss 模拟 readInt 的行为。
    /// 参数：
    ///   reader - 任意实现了 peekInt 和 toss 方法的 Reader
    ///   io     - 来自 std.process.Init.io 的 I/O 实例（Zig 0.16 要求）
    ///   T      - 整型类型（如 u32, i64）
    ///   endian - 字节序（.little 或 .big）
    /// 返回：读取到的 T 类型的整数
    fn readInt(self: *Parser, reader: *std.Io.Reader, comptime T: type, endian: std.builtin.Endian) !T {
        // 1. 偷看整数值
        const value = try reader.peekInt(T, endian);
        // 2. 跳过已偷看的字节数
        reader.toss(@sizeOf(T));
        self.file_offset += @sizeOf(T);
        return value;
    }

    /// 从 reader 解析 GGUF 数据
    pub fn parseReader(self: *Parser, reader: *std.Io.Reader, file_size: u64) !void {
        // 1. 读取魔数
        var magic: [4]u8 = undefined;
        reader.readSliceAll(&magic) catch return Error.IoError;
        if (!std.mem.eql(u8, &magic, &MAGIC)) {
            return Error.InvalidMagic;
        }

        // 2. 读取版本
        const version = try readInt(self, reader, i32, .little);
        const is_v3 = version >= 3;

        // 3. 读取张量数量和元数据键值对数量
        const tensor_count: i64 = if (is_v3)
            try readInt(self, reader, i64, .little)
        else
            @as(i64, try readInt(self, reader, i32, .little));

        const metadata_kv_count: i64 = if (is_v3)
            try readInt(self, reader, i64, .little)
        else
            @as(i64, try readInt(self, reader, i32, .little));

        // 4. 读取对齐值（v3 新增）
        var alignment: u32 = DEFAULT_ALIGNMENT;
        if (is_v3) {
            alignment = try readInt(self, reader, u32, .little);
        }

        self.header = .{
            .magic = magic,
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
            .alignment = alignment,
        };

        // 5. 读取元数据键值对
        var i: i64 = 0;
        while (i < metadata_kv_count) : (i += 1) {
            const key = try self.readString(reader);
            const value = try self.readMetaValue(reader);
            try self.metadata.put(key, value);
        }

        // 6. 记录张量数据偏移（当前 reader 位置）
        // 注意: 由于 reader 是 fixedBufferStream，无法直接获取当前偏移，但我们可以通过
        // 计算已读取的字节数来得到。此处简化：从文件大小减去剩余字节数。
        // 为简单起见，我们之后将使用 GGUF 文件中记录的张量偏移量。
        // 实际上 tensor data 开始位置就是当前 reader 位置。
        // 对于 mmap 方式，可以使用文件偏移。
        // 这里我们用一个占位符，实际需要时可从文件偏移获得。
        self.tensor_data_offset = file_size - self.file_offset;

        // 7. 读取张量信息
        var j: i64 = 0;
        while (j < tensor_count) : (j += 1) {
            const info = try self.readTensorInfo(reader);
            try self.tensors.append(self.allocator, info);
        }
    }

    fn readString(self: *Parser, reader: *std.Io.Reader) ![]const u8 {
        const len = try readInt(self, reader, u64, .little);
        const buf = try self.allocator.alloc(u8, @as(usize, @intCast(len)));
        errdefer self.allocator.free(buf);
        try reader.readSliceAll(buf);
        self.file_offset += len;
        return buf;
    }

    fn readMetaValue(self: *Parser, reader: *std.Io.Reader) !MetaValue {
        const typ_val = try readInt(self, reader, u32, .little);
        const gguf_typ: ggml.GgufValueType = @enumFromInt(typ_val);

        return switch (gguf_typ) {
            .uint8 => MetaValue{ .uint8 = try readInt(self, reader, u8, .little) },
            .int8 => MetaValue{ .int8 = try readInt(self, reader, i8, .little) },
            .uint16 => MetaValue{ .uint16 = try readInt(self, reader,  u16,.little) },
            .int16 => MetaValue{ .int16 = try readInt(self, reader, i16, .little) },
            .uint32 => MetaValue{ .uint32 = try readInt(self, reader, u32, .little) },
            .int32 => MetaValue{ .int32 = try readInt(self, reader, i32, .little) },
            .float32 => MetaValue{ .float32 = @bitCast(try readInt(self, reader, u32, .little)) },
            .bool => MetaValue{ .bool = (try readInt(self, reader, u8, .little)) != 0 },
            .string => MetaValue{ .string = try self.readString(reader) },
            .array => {
                const arr_typ_val = try readInt(self, reader, u32, .little);
                const arr_typ: ggml.GgufValueType = @enumFromInt(arr_typ_val);
                const arr_n = try readInt(self, reader, u64, .little);
                const items = try self.allocator.alloc(u8, @as(usize, @intCast(arr_n)));
                errdefer self.allocator.free(items);
                try reader.readSliceAll(items);
                return MetaValue{ .array = .{ .typ = arr_typ, .items = items } };
            },
            .uint64 => MetaValue{ .uint64 = try readInt(self, reader, u64, .little) },
            .int64 => MetaValue{ .int64 = try readInt(self, reader, i64, .little) },
            .float64 => MetaValue{ .float64 = @bitCast(try readInt(self, reader, u64, .little)) },
        };
    }

    fn readTensorInfo(self: *Parser, reader: *std.Io.Reader) !TensorInfo {
        const name = try self.readString(reader);
        errdefer self.allocator.free(name);

        const n_dims = try readInt(self, reader, u32, .little);
        var ne: [4]i64 = .{ 0, 0, 0, 0 };
        var d: u32 = 0;
        while (d < n_dims) : (d += 1) {
            ne[d] = try readInt(self, reader, i64, .little);
        }

        const typ_val = try readInt(self, reader, u32, .little);
        const typ: ggml.Type = @enumFromInt(typ_val);

        const offset = try readInt(self, reader, u64, .little);

        return TensorInfo{
            .name = name,
            .n_dims = n_dims,
            .ne = ne,
            .typ = typ,
            .offset = offset,
        };
    }

    /// 获取元数据值
    pub fn getMetadata(self: *Parser, key: []const u8) ?MetaValue {
        return self.metadata.get(key);
    }

    /// 获取元数据字符串值
    pub fn getMetadataString(self: *Parser, key: []const u8) ?[]const u8 {
        const val = self.getMetadata(key) orelse return null;
        return switch (val) {
            .string => |s| s,
            else => null,
        };
    }

    /// 获取元数据整数值（转换为 i64）
    pub fn getMetadataInt(self: *Parser, key: []const u8) ?i64 {
        const val = self.getMetadata(key) orelse return null;
        return switch (val) {
            .int32 => |v| @as(i64, v),
            .int64 => |v| v,
            .uint32 => |v| @as(i64, v),
            .uint64 => |v| @as(i64, @intCast(v)),
            else => null,
        };
    }

    /// 获取元数据浮点值（转换为 f64）
    pub fn getMetadataFloat(self: *Parser, key: []const u8) ?f64 {
        const val = self.getMetadata(key) orelse return null;
        return switch (val) {
            .float32 => |v| @as(f64, v),
            .float64 => |v| v,
            else => null,
        };
    }

    /// 打印元数据
    pub fn dumpMetadata(self: *Parser) void {
        std.debug.print("Metadata:\n", .{});
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            const key = entry.key_ptr.*;
            const val = entry.value_ptr.*;
            switch (val) {
                .string => |s| std.debug.print("  {s}: {s}\n", .{ key, s }),
                .int32 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .uint32 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .float32 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .bool => |v| std.debug.print("  {s}: {}\n", .{ key, v }),
                .int64 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .uint64 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                else => std.debug.print("  {s}: (other)\n", .{key}),
            }
        }
    }

    /// 打印张量信息
    pub fn dumpTensors(self: *Parser) void {
        std.debug.print("Tensors ({d}):\n", .{self.tensors.items.len});
        for (self.tensors.items, 0..) |t, i| {
            std.debug.print("  [{d}] {s}: type={s}, shape=[{d},{d},{d},{d}], offset={d}\n", .{
                i,
                t.name,
                @tagName(t.typ),
                t.ne[0],
                t.ne[1],
                t.ne[2],
                t.ne[3],
                t.offset,
            });
        }
    }
};

// ============================================================================
// 测试
// ============================================================================

test "GGUF magic constant" {
    try std.testing.expectEqual(@as(u8, 'G'), MAGIC[0]);
    try std.testing.expectEqual(@as(u8, 'G'), MAGIC[1]);
    try std.testing.expectEqual(@as(u8, 'U'), MAGIC[2]);
    try std.testing.expectEqual(@as(u8, 'F'), MAGIC[3]);
}

test "Parser init and deinit" {
    var io = try std.Io.init(.{ .allocator = std.testing.allocator });
    defer io.deinit();
    var parser = Parser.init(&io, std.testing.allocator);
    defer parser.deinit();
    try std.testing.expectEqual(@as(i64, 0), parser.header.tensor_count);
}
