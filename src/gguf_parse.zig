//! GGUF 文件解析（从原始字节数据解析为 GGUFFile）
const std = @import("std");
const ggml = @import("ggml");

const MetadataValueType = @import("./gguf.zig").MetadataValueType;
const TensorDataType = @import("./gguf.zig").TensorDataType;
const GGUFVersion = @import("./gguf.zig").GGUFVersion;
const MetadataValue = @import("./gguf.zig").MetadataValue;
const TensorInfo = @import("./gguf.zig").TensorInfo;
const GGUFFile = @import("./gguf.zig").GGUFFile;

const readString = @import("./gguf.zig").readString;
const readMetadataValue = @import("./gguf.zig").readMetadataValue;

const log = std.log.scoped(.gguf_parse);

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

    var metadata = std.StringHashMapUnmanaged(MetadataValue){};
    try metadata.ensureTotalCapacity(arena_allocator, @as(u32, @intCast(metadata_kv_count)));

    const metadata_keys = try arena_allocator.alloc([]const u8, @as(usize, @intCast(metadata_kv_count)));

    for (0..@as(usize, @intCast(metadata_kv_count))) |i| {
        const key = try readString(data, &pos, version, &arena);
        const val = try readMetadataValue(data, &pos, version, &arena);
        metadata_keys[i] = key;
        metadata.putAssumeCapacity(key, val);
    }

    // 解析张量信息（v3 不需要对齐，只有 tensor data 需要对齐）
    var tensors = std.ArrayList(TensorInfo).empty;
    try tensors.ensureTotalCapacity(arena_allocator, @as(usize, @intCast(tensor_count)));

    for (0..@as(usize, @intCast(tensor_count))) |_| {
        const name = try readString(data, &pos, version, &arena);
        if (pos + 4 > data.len) return error.InvalidGGUFFile;
        const n_dims = blk: {
            const nd = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;
            break :blk nd;
        };

        var dims: [4]u64 = [_]u64{0} ** 4;
        for (0..@as(usize, @intCast(n_dims))) |i| {
            if (is_64bit) {
                if (pos + 8 > data.len) return error.InvalidGGUFFile;
                dims[i] = std.mem.readInt(u64, data[pos..][0..8], .little);
                pos += 8;
            } else {
                if (pos + 4 > data.len) return error.InvalidGGUFFile;
                dims[i] = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
            }
        }

        if (pos + 4 > data.len) return error.InvalidGGUFFile;
        const data_type_int = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;
        const data_type: TensorDataType = @enumFromInt(data_type_int);

        // v2/v3 使用 64 位偏移
        const offset = if (is_64bit) blk: {
            if (pos + 8 > data.len) return error.InvalidGGUFFile;
            const off = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            break :blk off;
        } else blk: {
            if (pos + 4 > data.len) return error.InvalidGGUFFile;
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
        .metadata_keys = metadata_keys,
        .tensor_data_offset = tensor_data_offset,
        .data = data,
        .allocator = allocator,
        .arena = arena,
    };
}
