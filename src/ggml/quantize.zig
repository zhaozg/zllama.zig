//! 量化操作函数
//!
//! 提供 ggml 量化 C API 的类型安全 Zig 封装。
//! 封装 ggml_quantize_chunk、ggml_quantize_init、ggml_quantize_free、
//! ggml_quantize_requires_imatrix 等函数。
//!
//! 参考: deps/ggml/include/ggml.h (量化章节)

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Type = cmod.Type;

const log = std.log.scoped(.ggml_quantize);

// ============================================================================
// 量化 API
// ============================================================================

/// 初始化量化查找表。
/// 某些量化类型（IQ2_XXS、IQ2_XS、IQ2_S、IQ1_S、IQ1_M、IQ3_XXS、IQ3_S）
/// 需要预计算的查找表。此函数可多次调用同一类型——首次调用后或
/// ggml_quantizeFree 后才会真正初始化。
/// 线程安全。
pub fn quantizeInit(typ: Type) void {
    c.ggml_quantize_init(@intFromEnum(typ));
}

/// 释放所有量化查找表。
/// 在程序结束时调用以避免内存泄漏。
/// 线程安全。
pub fn quantizeFree() void {
    c.ggml_quantize_free();
}

/// 检查指定的量化类型是否需要重要性矩阵（imatrix）。
/// 需要 imatrix 的类型：IQ2_XXS、IQ2_XS、IQ1_S。
pub fn quantizeRequiresImatrix(typ: Type) bool {
    return c.ggml_quantize_requires_imatrix(@intFromEnum(typ));
}

/// 执行量化：将 f32 源数据量化为目标类型。
///
/// 参数：
///   typ      - 目标量化类型
///   src      - f32 源数据切片
///   dst      - 目标缓冲区（大小必须 >= 量化后数据大小）
///   start    - 起始元素偏移（必须是 blck_size 和 n_per_row 的倍数）
///   nrows    - 行数
///   n_per_row - 每行元素数
///   imatrix  - 重要性矩阵（可为 null，但 IQ2_XXS/IQ2_XS/IQ1_S 必须提供）
///
/// 返回：量化后数据字节数
///
/// 注意：此函数内部会调用 ggml_quantize_init，无需手动调用。
pub fn quantizeChunk(
    typ: Type,
    src: []const f32,
    dst: []u8,
    start: i64,
    nrows: i64,
    n_per_row: i64,
    imatrix: ?[]const f32,
) usize {
    const imatrix_ptr: ?[*]const f32 = if (imatrix) |im| im.ptr else null;
    return c.ggml_quantize_chunk(
        @intFromEnum(typ),
        src.ptr,
        dst.ptr,
        start,
        nrows,
        n_per_row,
        imatrix_ptr,
    );
}

/// 便捷函数：量化整个张量（从 start=0 开始）。
///
/// 参数：
///   typ       - 目标量化类型
///   src       - f32 源数据切片（长度 = nrows * n_per_row）
///   dst       - 目标缓冲区（大小必须 >= 量化后数据大小）
///   nrows     - 行数
///   n_per_row - 每行元素数
///   imatrix   - 重要性矩阵（可为 null）
///
/// 返回：量化后数据字节数
pub fn quantizeTensor(
    typ: Type,
    src: []const f32,
    dst: []u8,
    nrows: i64,
    n_per_row: i64,
    imatrix: ?[]const f32,
) usize {
    return quantizeChunk(typ, src, dst, 0, nrows, n_per_row, imatrix);
}

/// 计算量化后数据所需的最小缓冲区大小（字节）。
/// 等价于 ggml_row_size(type, n_per_row) * nrows。
pub fn quantizedSize(typ: Type, nrows: i64, n_per_row: i64) usize {
    const row_size = c.ggml_row_size(@intFromEnum(typ), n_per_row);
    return @as(usize, @intCast(row_size * nrows));
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "quantizeInit and quantizeFree" {
    // 验证初始化/释放不崩溃
    quantizeInit(.q4_0);
    quantizeInit(.q4_0); // 重复调用应为 noop
    quantizeFree();
    quantizeFree(); // 重复调用应为 noop
}

test "quantizeRequiresImatrix" {
    // 需要 imatrix 的类型
    try testing.expect(quantizeRequiresImatrix(.iq2_xxs));
    try testing.expect(quantizeRequiresImatrix(.iq2_xs));
    try testing.expect(quantizeRequiresImatrix(.iq1_s));

    // 不需要 imatrix 的类型
    try testing.expect(!quantizeRequiresImatrix(.q4_0));
    try testing.expect(!quantizeRequiresImatrix(.q4_K));
    try testing.expect(!quantizeRequiresImatrix(.q8_0));
    try testing.expect(!quantizeRequiresImatrix(.f16));
    try testing.expect(!quantizeRequiresImatrix(.iq1_m));
    try testing.expect(!quantizeRequiresImatrix(.iq3_xxs));
    try testing.expect(!quantizeRequiresImatrix(.iq3_s));
    try testing.expect(!quantizeRequiresImatrix(.iq2_s));
    try testing.expect(!quantizeRequiresImatrix(.iq4_nl));
    try testing.expect(!quantizeRequiresImatrix(.iq4_xs));
}

test "quantizedSize matches ggml_row_size" {
    // Q4_0: block_size=32, type_size=2 (2 bytes per block of 32 elements)
    // row_size = 2 * ne / 32
    const nrows: i64 = 4;
    const n_per_row: i64 = 256;
    const expected = c.ggml_row_size(@intFromEnum(Type.q4_0), n_per_row) * @as(usize, @intCast(nrows));
    try testing.expectEqual(expected, quantizedSize(.q4_0, nrows, n_per_row));

    // Q8_0: block_size=32, type_size=2
    const expected_q8 = c.ggml_row_size(@intFromEnum(Type.q8_0), n_per_row) * @as(usize, @intCast(nrows));
    try testing.expectEqual(expected_q8, quantizedSize(.q8_0, nrows, n_per_row));

    // Q4_K: block_size=256, type_size=?
    const expected_q4k = c.ggml_row_size(@intFromEnum(Type.q4_K), n_per_row) * @as(usize, @intCast(nrows));
    try testing.expectEqual(expected_q4k, quantizedSize(.q4_K, nrows, n_per_row));
}

test "quantizeChunk q4_0 roundtrip" {
    // 创建一个简单的 f32 数据块，量化为 Q4_0，验证输出大小
    const nrows: i64 = 2;
    const n_per_row: i64 = 64; // 2 blocks of 32

    var src: [128]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 128.0;
    }

    const dst_size = quantizedSize(.q4_0, nrows, n_per_row);
    var dst: [256]u8 = undefined;
    try testing.expect(dst.len >= dst_size);

    const result = quantizeChunk(.q4_0, &src, &dst, 0, nrows, n_per_row, null);
    try testing.expectEqual(dst_size, result);

    // 验证输出不为全零（量化后的数据应有变化）
    var all_zero = true;
    for (dst[0..dst_size]) |byte| {
        if (byte != 0) {
            all_zero = false;
            break;
        }
    }
    try testing.expect(!all_zero);
}

test "quantizeChunk with start offset" {
    // 验证 start 偏移量参数
    const nrows: i64 = 4;
    const n_per_row: i64 = 64;

    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 256.0;
    }

    // 分两次量化：先量化前 2 行，再量化后 2 行
    const half_rows: i64 = 2;
    const half_size = quantizedSize(.q4_0, half_rows, n_per_row);

    var dst1: [256]u8 = undefined;
    var dst2: [256]u8 = undefined;

    _ = quantizeChunk(.q4_0, &src, &dst1, 0, half_rows, n_per_row, null);
    _ = quantizeChunk(.q4_0, &src, &dst2, half_rows * n_per_row, half_rows, n_per_row, null);

    // 一次性量化全部
    var dst_full: [256]u8 = undefined;
    _ = quantizeChunk(.q4_0, &src, &dst_full, 0, nrows, n_per_row, null);

    // 分块量化的结果应与一次性量化一致
    const full_size = quantizedSize(.q4_0, nrows, n_per_row);
    try testing.expectEqual(full_size, half_size * 2);

    // 验证 dst1 与 dst_full 的前半部分一致
    try testing.expectEqualSlices(u8, dst1[0..half_size], dst_full[0..half_size]);
    // 验证 dst2 与 dst_full 的后半部分一致
    try testing.expectEqualSlices(u8, dst2[0..half_size], dst_full[half_size..full_size]);
}

test "quantizeChunk q8_0" {
    // Q8_0 量化测试
    const nrows: i64 = 1;
    const n_per_row: i64 = 32;

    var src: [32]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) - 16.0;
    }

    const dst_size = quantizedSize(.q8_0, nrows, n_per_row);
    var dst: [64]u8 = undefined;
    try testing.expect(dst.len >= dst_size);

    const result = quantizeChunk(.q8_0, &src, &dst, 0, nrows, n_per_row, null);
    try testing.expectEqual(dst_size, result);
}

test "quantizeChunk q4_K" {
    // K-Quant 量化测试
    const nrows: i64 = 1;
    const n_per_row: i64 = 256; // K-Quant 超级块大小

    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 256.0;
    }

    const dst_size = quantizedSize(.q4_K, nrows, n_per_row);
    var dst: [512]u8 = undefined;
    try testing.expect(dst.len >= dst_size);

    const result = quantizeChunk(.q4_K, &src, &dst, 0, nrows, n_per_row, null);
    try testing.expectEqual(dst_size, result);
}

test "quantizeChunk f16 (passthrough)" {
    // F16 量化本质上是类型转换
    const nrows: i64 = 1;
    const n_per_row: i64 = 64;

    var src: [64]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i));
    }

    const dst_size = quantizedSize(.f16, nrows, n_per_row);
    var dst: [256]u8 = undefined;
    try testing.expect(dst.len >= dst_size);

    const result = quantizeChunk(.f16, &src, &dst, 0, nrows, n_per_row, null);
    try testing.expectEqual(dst_size, result);
}

test "quantizeChunk bf16 (passthrough)" {
    const nrows: i64 = 1;
    const n_per_row: i64 = 64;

    var src: [64]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.5;
    }

    const dst_size = quantizedSize(.bf16, nrows, n_per_row);
    var dst: [256]u8 = undefined;
    try testing.expect(dst.len >= dst_size);

    const result = quantizeChunk(.bf16, &src, &dst, 0, nrows, n_per_row, null);
    try testing.expectEqual(dst_size, result);
}

test "quantizeChunk f32 (identity)" {
    // F32 量化是恒等复制
    const nrows: i64 = 1;
    const n_per_row: i64 = 16;

    var src: [16]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 1.5;
    }

    const dst_size = quantizedSize(.f32, nrows, n_per_row);
    try testing.expectEqual(@as(usize, 16 * @sizeOf(f32)), dst_size);

    var dst: [64]u8 = undefined;
    const result = quantizeChunk(.f32, &src, &dst, 0, nrows, n_per_row, null);
    try testing.expectEqual(dst_size, result);

    // F32 量化应保持原值
    const dst_f32 = @as([*]f32, @ptrCast(@alignCast(&dst)))[0..16];
    for (src, dst_f32) |s, d| {
        try testing.expectEqual(s, d);
    }
}

test "quantizeTensor convenience" {
    const nrows: i64 = 2;
    const n_per_row: i64 = 64;

    var src: [128]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 128.0;
    }

    const dst_size = quantizedSize(.q4_0, nrows, n_per_row);
    var dst: [256]u8 = undefined;

    const result = quantizeTensor(.q4_0, &src, &dst, nrows, n_per_row, null);
    try testing.expectEqual(dst_size, result);
}

test "quantizeChunk multiple types" {
    // 验证多种量化类型都能正常工作
    const types_to_test = [_]Type{
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0,
        .q2_K, .q3_K, .q4_K, .q5_K, .q6_K,
    };

    const nrows: i64 = 1;
    const n_per_row: i64 = 256;

    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 256.0;
    }

    for (types_to_test) |typ| {
        const dst_size = quantizedSize(typ, nrows, n_per_row);
        var dst: [2048]u8 = undefined;
        if (dst.len < dst_size) {
            // 某些类型可能需要更大缓冲区，跳过
            continue;
        }
        const result = quantizeChunk(typ, &src, &dst, 0, nrows, n_per_row, null);
        try testing.expectEqual(dst_size, result);
    }
}
