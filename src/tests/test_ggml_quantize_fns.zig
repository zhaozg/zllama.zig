//! ggml_quantize_fns 测试
//!
//! 对应 deps/ggml/tests/test-quantize-fns.cpp
//! 测试量化函数的正确性：量化-反量化往返、类型特征表等。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

// ============================================================================
// 量化-反量化往返测试
// ============================================================================

/// 测试指定量化类型的量化-反量化往返精度
fn testQuantizeDequantizeRoundtrip(typ: ggml.Type) !void {
    const n_per_row: i64 = 256;
    const n_rows: i64 = 2;
    const n_elements = @as(usize, @intCast(n_per_row * n_rows));

    // 创建测试数据
    const src = try testing.allocator.alloc(f32, n_elements);
    defer testing.allocator.free(src);

    var idx: usize = 0;
    while (idx < n_elements) : (idx += 1) {
        const i = @as(f32, @floatFromInt(idx));
        src[idx] = switch (idx % 6) {
            0 => i,
            1 => -i,
            2 => @sin(i * 0.1),
            3 => @cos(i * 0.1),
            4 => 0.0,
            5 => 1.0,
            else => 0.0,
        };
    }

    // 量化
    const quant_size = ggml.quantizedSize(typ, n_rows, n_per_row);
    const quant_buf = try testing.allocator.alloc(u8, quant_size);
    defer testing.allocator.free(quant_buf);

    _ = ggml.quantizeChunk(typ, src, quant_buf, 0, n_rows, n_per_row, null);

    // 反量化
    const dequant = try testing.allocator.alloc(f32, n_elements);
    defer testing.allocator.free(dequant);

    ggml.dequantizeTensor(typ, quant_buf, dequant, n_per_row, n_rows) catch |err| {
        if (err == error.DequantizeNotSupported) {
            return error.SkipZigTest;
        }
        return err;
    };

    // 计算 NMSE
    var sum_sq_err: f64 = 0.0;
    var sum_sq_src: f64 = 0.0;

    for (src, dequant) |s, d| {
        const err = @abs(d - s);
        sum_sq_err += @as(f64, @floatCast(err * err));
        sum_sq_src += @as(f64, @floatCast(s * s));
    }

    const nmse = if (sum_sq_src > 0.0)
        @as(f64, @floatCast(sum_sq_err / sum_sq_src))
    else
        sum_sq_err;

    try testing.expect(nmse < 0.1);

    for (dequant) |d| {
        try testing.expect(!std.math.isNan(d));
    }
}

test "quantize-dequantize roundtrip f16" { try testQuantizeDequantizeRoundtrip(.f16); }
test "quantize-dequantize roundtrip bf16" { try testQuantizeDequantizeRoundtrip(.bf16); }
test "quantize-dequantize roundtrip q4_0" { try testQuantizeDequantizeRoundtrip(.q4_0); }
test "quantize-dequantize roundtrip q4_1" { try testQuantizeDequantizeRoundtrip(.q4_1); }
test "quantize-dequantize roundtrip q5_0" { try testQuantizeDequantizeRoundtrip(.q5_0); }
test "quantize-dequantize roundtrip q5_1" { try testQuantizeDequantizeRoundtrip(.q5_1); }
test "quantize-dequantize roundtrip q8_0" { try testQuantizeDequantizeRoundtrip(.q8_0); }
test "quantize-dequantize roundtrip q2_K" { try testQuantizeDequantizeRoundtrip(.q2_K); }
test "quantize-dequantize roundtrip q3_K" { try testQuantizeDequantizeRoundtrip(.q3_K); }
test "quantize-dequantize roundtrip q4_K" { try testQuantizeDequantizeRoundtrip(.q4_K); }
test "quantize-dequantize roundtrip q5_K" { try testQuantizeDequantizeRoundtrip(.q5_K); }
test "quantize-dequantize roundtrip q6_K" { try testQuantizeDequantizeRoundtrip(.q6_K); }

// ============================================================================
// 类型特征表测试
// ============================================================================

test "ggml_type_traits block_size" {
    try testing.expectEqual(@as(i64, 1), ggml.Type.blockSize(.f32));
    try testing.expectEqual(@as(i64, 1), ggml.Type.blockSize(.f16));
    try testing.expectEqual(@as(i64, 1), ggml.Type.blockSize(.i32));
    try testing.expectEqual(@as(i64, 32), ggml.Type.blockSize(.q4_0));
    try testing.expectEqual(@as(i64, 32), ggml.Type.blockSize(.q4_1));
    try testing.expectEqual(@as(i64, 32), ggml.Type.blockSize(.q5_0));
    try testing.expectEqual(@as(i64, 32), ggml.Type.blockSize(.q5_1));
    try testing.expectEqual(@as(i64, 32), ggml.Type.blockSize(.q8_0));
    try testing.expectEqual(@as(i64, 256), ggml.Type.blockSize(.q2_K));
    try testing.expectEqual(@as(i64, 256), ggml.Type.blockSize(.q3_K));
    try testing.expectEqual(@as(i64, 256), ggml.Type.blockSize(.q4_K));
    try testing.expectEqual(@as(i64, 256), ggml.Type.blockSize(.q5_K));
    try testing.expectEqual(@as(i64, 256), ggml.Type.blockSize(.q6_K));
    try testing.expectEqual(@as(i64, 256), ggml.Type.blockSize(.q8_K));
}

test "ggml_type_traits type_size" {
    try testing.expectEqual(@as(usize, 4), ggml.Type.sizeOf(.f32));
    try testing.expectEqual(@as(usize, 2), ggml.Type.sizeOf(.f16));
    try testing.expectEqual(@as(usize, 4), ggml.Type.sizeOf(.i32));
    try testing.expect(ggml.Type.sizeOf(.q4_0) > 0);
    try testing.expect(ggml.Type.sizeOf(.q8_0) > 0);
    try testing.expect(ggml.Type.sizeOf(.q4_K) > 0);
}

test "ggml_type_traits is_quantized" {
    try testing.expect(!ggml.Type.isQuantized(.f32));
    try testing.expect(!ggml.Type.isQuantized(.f16));
    try testing.expect(!ggml.Type.isQuantized(.i32));
    try testing.expect(!ggml.Type.isQuantized(.bf16));
    try testing.expect(ggml.Type.isQuantized(.q4_0));
    try testing.expect(ggml.Type.isQuantized(.q4_1));
    try testing.expect(ggml.Type.isQuantized(.q5_0));
    try testing.expect(ggml.Type.isQuantized(.q5_1));
    try testing.expect(ggml.Type.isQuantized(.q8_0));
    try testing.expect(ggml.Type.isQuantized(.q2_K));
    try testing.expect(ggml.Type.isQuantized(.q3_K));
    try testing.expect(ggml.Type.isQuantized(.q4_K));
    try testing.expect(ggml.Type.isQuantized(.q5_K));
    try testing.expect(ggml.Type.isQuantized(.q6_K));
    try testing.expect(ggml.Type.isQuantized(.q8_K));
    try testing.expect(ggml.Type.isQuantized(.iq2_xxs));
    try testing.expect(ggml.Type.isQuantized(.iq2_xs));
    try testing.expect(ggml.Type.isQuantized(.iq3_xxs));
    try testing.expect(ggml.Type.isQuantized(.iq1_s));
    try testing.expect(ggml.Type.isQuantized(.iq4_nl));
    try testing.expect(ggml.Type.isQuantized(.iq3_s));
    try testing.expect(ggml.Type.isQuantized(.iq2_s));
    try testing.expect(ggml.Type.isQuantized(.iq4_xs));
    try testing.expect(ggml.Type.isQuantized(.iq1_m));
}

test "ggml_type_traits row_size" {
    const ne: i64 = 256;
    try testing.expectEqual(@as(usize, 1024), ggml.Type.rowSize(.f32, ne));
    try testing.expectEqual(@as(usize, 512), ggml.Type.rowSize(.f16, ne));
    try testing.expectEqual(@as(usize, 1024), ggml.Type.rowSize(.i32, ne));
}

test "ggml_type_traits type_name" {
    try testing.expectEqualStrings("f32", ggml.Type.name(.f32));
    try testing.expectEqualStrings("f16", ggml.Type.name(.f16));
    try testing.expectEqualStrings("q4_0", ggml.Type.name(.q4_0));
    try testing.expectEqualStrings("q4_K", ggml.Type.name(.q4_K));
    try testing.expectEqualStrings("q8_0", ggml.Type.name(.q8_0));
    try testing.expectEqualStrings("i32", ggml.Type.name(.i32));
    try testing.expectEqualStrings("bf16", ggml.Type.name(.bf16));
}

// ============================================================================
// 反量化函数测试
// ============================================================================

test "dequantizeRow q4_0" {
    const n_per_row: i64 = 32;
    var src = [_]f32{0.0} ** 32;
    for (&src, 0..) |*v, i| { v.* = @as(f32, @floatFromInt(i)) - 16.0; }

    var quant_buf: [128]u8 = undefined;
    _ = ggml.quantizeChunk(.q4_0, &src, &quant_buf, 0, 1, n_per_row, null);

    var dequant: [32]f32 = undefined;
    try ggml.dequantizeRow(.q4_0, &quant_buf, &dequant, n_per_row);

    var max_err: f32 = 0.0;
    for (src, dequant) |s, d| { const err = @abs(d - s); if (err > max_err) max_err = err; }
    try testing.expect(max_err < 5.0);
}

test "dequantizeRow q8_0" {
    const n_per_row: i64 = 32;
    var src = [_]f32{0.0} ** 32;
    for (&src, 0..) |*v, i| { v.* = @as(f32, @floatFromInt(i)) - 16.0; }

    var quant_buf: [128]u8 = undefined;
    _ = ggml.quantizeChunk(.q8_0, &src, &quant_buf, 0, 1, n_per_row, null);

    var dequant: [32]f32 = undefined;
    try ggml.dequantizeRow(.q8_0, &quant_buf, &dequant, n_per_row);

    var max_err: f32 = 0.0;
    for (src, dequant) |s, d| { const err = @abs(d - s); if (err > max_err) max_err = err; }
    try testing.expect(max_err < 2.0);
}

test "dequantizeTensor full tensor" {
    const n_per_row: i64 = 64;
    const n_rows: i64 = 4;

    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| { v.* = @as(f32, @floatFromInt(i)) / 256.0; }

    var quant_buf: [512]u8 = undefined;
    _ = ggml.quantizeChunk(.q4_0, &src, &quant_buf, 0, n_rows, n_per_row, null);

    var dequant: [256]f32 = undefined;
    try ggml.dequantizeTensor(.q4_0, &quant_buf, &dequant, n_per_row, n_rows);

    var max_err: f32 = 0.0;
    for (src, dequant) |s, d| { const err = @abs(d - s); if (err > max_err) max_err = err; }
    try testing.expect(max_err < 5.0);
}

test "dequantizeRow unsupported type" {
    var buf: [16]f32 = undefined;
    const result = ggml.dequantizeRow(.i32, @as(*const anyopaque, @ptrCast(&buf)), &buf, 16);
    try testing.expectError(error.DequantizeNotSupported, result);
}

// ============================================================================
// 多类型量化测试
// ============================================================================

test "quantize multiple types consistency" {
    const n_per_row: i64 = 256;
    const n_rows: i64 = 1;

    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| { v.* = @as(f32, @floatFromInt(i)) / 256.0; }

    const size_q4_0 = ggml.quantizedSize(.q4_0, n_rows, n_per_row);
    const size_q4_K = ggml.quantizedSize(.q4_K, n_rows, n_per_row);
    const size_q8_0 = ggml.quantizedSize(.q8_0, n_rows, n_per_row);
    const size_f16 = ggml.quantizedSize(.f16, n_rows, n_per_row);

    try testing.expect(size_q4_0 > 0);
    try testing.expect(size_q4_K > 0);
    try testing.expect(size_q8_0 > 0);
    try testing.expect(size_f16 > 0);
    try testing.expect(size_f16 > size_q8_0);
}
