//! GGUF 解析测试
//!
//! 验证 GGUF v2/v3 格式的解析正确性。
//! 包括：文件头、元数据 KV、张量信息、对齐等。
//!
//! 测试策略：
//! - 手工构造 v2/v3 格式的二进制数据
//! - 验证解析结果与预期一致
//! - 测试边界条件（空元数据、空张量、对齐等）

const std = @import("std");
const testing = std.testing;
const gguf = @import("gguf");

// ============================================================================
// 辅助函数：构造 GGUF 二进制数据
// ============================================================================

/// 写入 u32 到缓冲区（小端）
fn writeU32(buf: []u8, pos: *usize, val: u32) void {
    std.mem.writeInt(u32, buf[pos.*..][0..4], val, .little);
    pos.* += 4;
}

/// 写入 u64 到缓冲区（小端）
fn writeU64(buf: []u8, pos: *usize, val: u64) void {
    std.mem.writeInt(u64, buf[pos.*..][0..8], val, .little);
    pos.* += 8;
}

/// 写入字符串到缓冲区（v2/v3 格式：u64 长度 + 数据）
fn writeString(buf: []u8, pos: *usize, str: []const u8) void {
    writeU64(buf, pos, @intCast(str.len));
    @memcpy(buf[pos.*..][0..str.len], str);
    pos.* += str.len;
}

/// 写入 u32 元数据值
fn writeMetadataU32(buf: []u8, pos: *usize, key: []const u8, val: u32) void {
    writeString(buf, pos, key);
    // value type: U32 = 4
    writeU32(buf, pos, 4);
    writeU32(buf, pos, val);
}

/// 写入 f32 元数据值
fn writeMetadataF32(buf: []u8, pos: *usize, key: []const u8, val: f32) void {
    writeString(buf, pos, key);
    // value type: F32 = 6
    writeU32(buf, pos, 6);
    std.mem.writeInt(u32, buf[pos.*..][0..4], @bitCast(val), .little);
    pos.* += 4;
}

/// 写入字符串元数据值
fn writeMetadataString(buf: []u8, pos: *usize, key: []const u8, val: []const u8) void {
    writeString(buf, pos, key);
    // value type: STRING = 8
    writeU32(buf, pos, 8);
    writeString(buf, pos, val);
}

/// 写入 bool 元数据值
fn writeMetadataBool(buf: []u8, pos: *usize, key: []const u8, val: bool) void {
    writeString(buf, pos, key);
    // value type: BOOL = 7
    writeU32(buf, pos, 7);
    // BOOL is 1 byte in GGUF spec
    buf[pos.*] = if (val) 1 else 0;
    pos.* += 1;
}

/// 写入数组元数据值（u32 数组）
fn writeMetadataArrayU32(buf: []u8, pos: *usize, key: []const u8, vals: []const u32) void {
    writeString(buf, pos, key);
    // value type: ARRAY = 9
    writeU32(buf, pos, 9);
    // array element type: U32 = 4
    writeU32(buf, pos, 4);
    // array length
    writeU64(buf, pos, @intCast(vals.len));
    for (vals) |v| {
        writeU32(buf, pos, v);
    }
}

/// 写入张量信息（v2/v3 格式）
fn writeTensorInfo(
    buf: []u8,
    pos: *usize,
    name: []const u8,
    n_dims: u32,
    dims: []const u64,
    data_type: u32,
    offset: u64,
) void {
    writeString(buf, pos, name);
    writeU32(buf, pos, n_dims);
    for (dims) |d| {
        writeU64(buf, pos, d);
    }
    writeU32(buf, pos, data_type);
    writeU64(buf, pos, offset);
}

/// 构造一个简单的 v3 GGUF 文件
/// 包含：魔数 + 版本 + tensor_count + metadata_kv_count + 元数据 + 张量信息 + 对齐 + 张量数据
fn buildSimpleV3GGUF(allocator: std.mem.Allocator) ![]u8 {
    // 先计算精确大小
    var pos: usize = 0;
    // header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24
    pos += 24;
    // metadata: 4 KV pairs
    // "general.architecture" = 8+20 + type(4) + "llama" = 8+5 = 45
    pos += 45;
    // "llama.block_count" = 8+17 + type(4) + u32(4) = 33
    pos += 33;
    // "llama.rope.freq_base" = 8+20 + type(4) + f32(4) = 36
    pos += 36;
    // "general.file_type" = 8+17 + type(4) + bool(1) = 30
    pos += 30;
    // tensor info: "token_embd_weight" = 8+18 + n_dims(4) + dims(8+8) + data_type(4) + offset(8) = 58
    pos += 58;
    // v3 alignment to 32
    const aligned_pos = std.mem.alignForward(usize, pos, 32);
    const total_size = aligned_pos + 64; // 64 bytes of fake tensor data
    var buf = try allocator.alloc(u8, total_size);
    @memset(buf, 0);
    pos = 0;

    // 魔数 "GGUF"
    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;

    // 版本号 v3
    writeU32(buf, &pos, 3);

    // tensor_count = 1, metadata_kv_count = 4
    writeU64(buf, &pos, 1);
    writeU64(buf, &pos, 4);

    // 元数据 KV
    writeMetadataString(buf, &pos, "general.architecture", "llama");
    writeMetadataU32(buf, &pos, "llama.block_count", 32);
    writeMetadataF32(buf, &pos, "llama.rope.freq_base", 10000000.0);
    writeMetadataBool(buf, &pos, "general.file_type", true);

    // 张量信息：token_embd_weight [128, 4096] f32
    writeTensorInfo(buf, &pos, "token_embd_weight", 2, &[_]u64{ 128, 4096 }, 0, 0); // f32 = 0

    // v3 对齐到 32 字节
    const aligned = std.mem.alignForward(usize, pos, 32);
    @memset(buf[pos..aligned], 0);
    pos = aligned;

    // 写 64 字节占位张量数据
    @memset(buf[pos..][0..64], 0);
    pos += 64;

    return buf[0..pos];
}

/// 构造一个空的 v3 GGUF 文件（无元数据、无张量）
fn buildEmptyV3GGUF(allocator: std.mem.Allocator) ![]u8 {
    // header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24
    const total_size: usize = 24;
    var buf = try allocator.alloc(u8, total_size);
    @memset(buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;

    writeU32(buf, &pos, 3); // version v3
    writeU64(buf, &pos, 0); // tensor_count = 0
    writeU64(buf, &pos, 0); // metadata_kv_count = 0

    return buf;
}

/// 构造一个 v2 GGUF 文件（与 v3 结构相同，只是版本号不同）
fn buildV2GGUF(allocator: std.mem.Allocator) ![]u8 {
    // header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24
    // metadata: "general.architecture" = 8+20 + type(4) + "qwen2" = 8+5 = 45
    const total_size: usize = 24 + 45;
    var buf = try allocator.alloc(u8, total_size);
    @memset(buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;

    writeU32(buf, &pos, 2); // version v2
    writeU64(buf, &pos, 0); // tensor_count = 0
    writeU64(buf, &pos, 1); // metadata_kv_count = 1

    writeMetadataString(buf, &pos, "general.architecture", "qwen2");

    return buf;
}

// ============================================================================
// 测试用例
// ============================================================================

test "GGUF v3 - basic parsing" {
    const data = try buildSimpleV3GGUF(testing.allocator);
    defer testing.allocator.free(data);

    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    try testing.expectEqual(gguf.GGUFVersion.v3, file.version);
    try testing.expectEqual(@as(u64, 1), file.tensor_count);
}

test "GGUF v3 - metadata reading" {
    const data = try buildSimpleV3GGUF(testing.allocator);
    defer testing.allocator.free(data);

    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    // 读取字符串元数据
    const arch = file.getString("general.architecture");
    try testing.expect(arch != null);
    try testing.expectEqualStrings("llama", arch.?);

    // 读取 u32 元数据
    const block_count = file.getU32("llama.block_count");
    try testing.expect(block_count != null);
    try testing.expectEqual(@as(u32, 32), block_count.?);

    // 读取 f32 元数据
    const freq_base = file.getF32("llama.rope.freq_base");
    try testing.expect(freq_base != null);
    try testing.expectApproxEqAbs(@as(f32, 10000000.0), freq_base.?, 0.001);

    // 读取 bool 元数据
    const file_type = file.getBool("general.file_type");
    try testing.expect(file_type != null);
    try testing.expect(file_type.?);
}

test "GGUF v3 - tensor info" {
    const data = try buildSimpleV3GGUF(testing.allocator);
    defer testing.allocator.free(data);

    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const tensor = file.findTensor("token_embd_weight");
    try testing.expect(tensor != null);
    try testing.expectEqual(@as(u64, 2), tensor.?.n_dims);
    try testing.expectEqual(@as(u64, 128), tensor.?.dims[0]);
    try testing.expectEqual(@as(u64, 4096), tensor.?.dims[1]);
    try testing.expectEqual(gguf.TensorDataType.f32, tensor.?.data_type);
}

test "GGUF v3 - empty file" {
    const data = try buildEmptyV3GGUF(testing.allocator);
    defer testing.allocator.free(data);

    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    try testing.expectEqual(gguf.GGUFVersion.v3, file.version);
    try testing.expectEqual(@as(u64, 0), file.tensor_count);
    try testing.expect(file.getString("nonexistent") == null);
}

test "GGUF v2 - parsing" {
    const data = try buildV2GGUF(testing.allocator);
    defer testing.allocator.free(data);

    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    try testing.expectEqual(gguf.GGUFVersion.v2, file.version);
    const arch = file.getString("general.architecture");
    try testing.expect(arch != null);
    try testing.expectEqualStrings("qwen2", arch.?);
}

test "GGUF - invalid magic" {
    const data = [_]u8{0} ** 16;
    const result = gguf.parse(&data, testing.allocator);
    try testing.expectError(error.InvalidGGUFFile, result);
}

test "GGUF - unsupported version" {
    var buf: [16]u8 = undefined;
    @memcpy(buf[0..4], "GGUF");
    std.mem.writeInt(u32, buf[4..8][0..4], 99, .little); // version 99
    const result = gguf.parse(&buf, testing.allocator);
    try testing.expectError(error.UnsupportedGGUFVersion, result);
}

test "GGUF - nonexistent key" {
    const data = try buildEmptyV3GGUF(testing.allocator);
    defer testing.allocator.free(data);

    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    try testing.expect(file.getString("nonexistent.key") == null);
    try testing.expect(file.getU32("nonexistent.key") == null);
    try testing.expect(file.getF32("nonexistent.key") == null);
    try testing.expect(file.getBool("nonexistent.key") == null);
}

test "GGUF - tensor not found" {
    const data = try buildEmptyV3GGUF(testing.allocator);
    defer testing.allocator.free(data);

    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    try testing.expect(file.findTensor("nonexistent") == null);
}

test "GGUF - metadata with array value" {
    // 构造包含数组元数据的 GGUF
    var buf: [256]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    writeU32(&buf, &pos, 3); // v3
    writeU64(&buf, &pos, 0); // tensor_count = 0
    writeU64(&buf, &pos, 1); // metadata_kv_count = 1

    // 写入 u32 数组元数据
    const values = [_]u32{ 1, 2, 3, 4, 5 };
    writeMetadataArrayU32(&buf, &pos, "test.array", &values);

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    // 验证数组元数据
    // 注意：当前 GGUFFile 没有直接的 getArray 方法
    // 我们通过 metadata HashMap 来验证
    const val = file.metadata.get("test.array");
    try testing.expect(val != null);
    try testing.expectEqual(gguf.MetadataValueType.array, val.?.value_type);
}

test "GGUF - multiple tensors" {
    // 构造包含多个张量的 GGUF
    var buf: [512]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    writeU32(&buf, &pos, 3); // v3
    writeU64(&buf, &pos, 3); // tensor_count = 3
    writeU64(&buf, &pos, 0); // metadata_kv_count = 0

    // 张量 1: [4096, 4096] f32
    writeTensorInfo(&buf, &pos, "layer.0.attention.wq.weight", 2, &[_]u64{ 4096, 4096 }, 0, 0);
    // 张量 2: [4096, 4096] f32
    writeTensorInfo(&buf, &pos, "layer.0.attention.wk.weight", 2, &[_]u64{ 4096, 4096 }, 0, 1);
    // 张量 3: [128, 4096] f16
    writeTensorInfo(&buf, &pos, "token_embd.weight", 2, &[_]u64{ 128, 4096 }, 1, 2); // f16 = 1

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    try testing.expectEqual(@as(u64, 3), file.tensor_count);
    try testing.expect(file.findTensor("layer.0.attention.wq.weight") != null);
    try testing.expect(file.findTensor("layer.0.attention.wk.weight") != null);
    try testing.expect(file.findTensor("token_embd.weight") != null);
    try testing.expect(file.findTensor("nonexistent") == null);

    // 验证张量类型
    const t3 = file.findTensor("token_embd.weight").?;
    try testing.expectEqual(gguf.TensorDataType.f16, t3.data_type);
}

test "GGUF - v3 alignment" {
    // 验证 v3 的张量数据起始位置对齐到 32 字节
    var buf: [256]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    writeU32(&buf, &pos, 3); // v3
    writeU64(&buf, &pos, 1); // tensor_count = 1
    writeU64(&buf, &pos, 1); // metadata_kv_count = 1

    // 写入一个较长的 key 使 pos 不对齐
    writeMetadataString(&buf, &pos, "general.architecture", "llama");

    // 张量信息
    writeTensorInfo(&buf, &pos, "test.weight", 1, &[_]u64{64}, 0, 0);

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    // tensor_data_offset 应该对齐到 32
    try testing.expectEqual(@as(usize, 0), file.tensor_data_offset % 32);
}

test "GGUF - TensorDataType enum" {
    try testing.expectEqual(@as(usize, 4), gguf.TensorDataType.f32.typeSize());
    try testing.expectEqual(@as(usize, 2), gguf.TensorDataType.f16.typeSize());
    try testing.expectEqual(@as(usize, 1), gguf.TensorDataType.f32.blockSize());
    try testing.expectEqual(@as(usize, 1), gguf.TensorDataType.f16.blockSize());
}

test "GGUF - TensorInfo sizeBytes" {
    // f32 张量 [128, 4096]
    const info = gguf.TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ 128, 4096, 0, 0 },
        .data_type = .f32,
        .offset = 0,
    };
    try testing.expectEqual(@as(usize, 128 * 4096 * 4), info.sizeBytes());
}

test "GGUF - MetadataValueType enum" {
    try testing.expectEqual(@as(u32, 7), @intFromEnum(gguf.MetadataValueType.bool));
    try testing.expectEqual(@as(u32, 6), @intFromEnum(gguf.MetadataValueType.float32));
    try testing.expectEqual(@as(u32, 4), @intFromEnum(gguf.MetadataValueType.uint32));
    try testing.expectEqual(@as(u32, 8), @intFromEnum(gguf.MetadataValueType.string));
    try testing.expectEqual(@as(u32, 9), @intFromEnum(gguf.MetadataValueType.array));
}

test "GGUF - MetadataValue asString" {
    // 测试 MetadataValue 的 asString 方法
    const val = gguf.MetadataValue{ .value_type = .string, .string_val = "hello" };
    const s = val.asString();
    try testing.expect(s != null);
    try testing.expectEqualStrings("hello", s.?);
}

test "GGUF - MetadataValue asU32" {
    const val = gguf.MetadataValue{ .value_type = .uint32, .uint32_val = 42 };
    const n = val.asU32();
    try testing.expect(n != null);
    try testing.expectEqual(@as(u32, 42), n.?);
}

test "GGUF - MetadataValue asF32" {
    const val = gguf.MetadataValue{ .value_type = .float32, .float32_val = 3.14 };
    const n = val.asF32();
    try testing.expect(n != null);
    try testing.expectApproxEqAbs(@as(f32, 3.14), n.?, 0.001);
}

test "GGUF - MetadataValue asBool" {
    const val_true = gguf.MetadataValue{ .value_type = .bool, .bool_val = true };
    try testing.expect(val_true.asBool().?);

    const val_false = gguf.MetadataValue{ .value_type = .bool, .bool_val = false };
    try testing.expect(!val_false.asBool().?);
}

test "GGUF - MetadataValue wrong type returns null" {
    const str_val = gguf.MetadataValue{ .value_type = .string, .string_val = "hello" };
    try testing.expect(str_val.asU32() == null);
    try testing.expect(str_val.asF32() == null);
    try testing.expect(str_val.asBool() == null);

    const u32_val = gguf.MetadataValue{ .value_type = .uint32, .uint32_val = 42 };
    try testing.expect(u32_val.asString() == null);
}
