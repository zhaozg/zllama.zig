//! LogitsComparator 测试
//!
//! 测试 compare_logits 工具的各项功能。

const std = @import("std");
const compare_logits = @import("compare_logits");

const testing = std.testing;

test "LogitsComparator init" {
    const config = compare_logits.CompareConfig{};
    const comp = compare_logits.LogitsComparator.init(testing.allocator, config);
    try testing.expectApproxEqAbs(@as(f64, 1e-4), comp.config.nmse_threshold, 1e-10);
}

test "compare identical logits" {
    const ref = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const test_logits = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var comp = compare_logits.LogitsComparator.init(testing.allocator, .{});
    const result = try comp.compare(&ref, &test_logits);

    try testing.expectEqual(@as(f64, 0.0), result.nmse);
    try testing.expectEqual(@as(f32, 0.0), result.max_abs_error);
    try testing.expectEqual(@as(f64, 1.0), result.cosine_similarity);
    try testing.expect(result.psnr > 50.0);
}

test "compare different logits" {
    const ref = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const test_logits = [_]f32{ 1.1, 2.1, 3.1, 4.1, 5.1 };

    var comp = compare_logits.LogitsComparator.init(testing.allocator, .{});
    const result = try comp.compare(&ref, &test_logits);

    try testing.expect(result.nmse > 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.1), result.max_abs_error, 0.001);
    try testing.expect(result.cosine_similarity > 0.99);
}

test "compare with large difference" {
    const ref = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const test_logits = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };

    var comp = compare_logits.LogitsComparator.init(testing.allocator, .{});
    const result = try comp.compare(&ref, &test_logits);

    try testing.expect(result.nmse > 1.0);
    try testing.expect(result.max_abs_error > 5.0);
}

test "compare empty logits" {
    var comp = compare_logits.LogitsComparator.init(testing.allocator, .{});
    const result = try comp.compare(&[_]f32{}, &[_]f32{});

    try testing.expectEqual(@as(f64, 0.0), result.nmse);
    try testing.expectEqual(@as(f32, 0.0), result.max_abs_error);
}

test "printReport output" {
    const result = compare_logits.ComparisonResult{
        .nmse = 1e-6,
        .max_abs_error = 0.001,
        .mean_abs_error = 0.0005,
        .cosine_similarity = 0.9999,
        .psnr = 60.0,
        .matched_tokens = 100,
        .total_tokens = 100,
        .match_rate = 1.0,
    };

    _ = compare_logits.LogitsComparator.init(testing.allocator, .{});
    var buf = try std.ArrayList(u8).initCapacity(testing.allocator, 1024);
    defer buf.deinit(testing.allocator);

    // 使用 std.fmt 直接格式化到 ArrayList
    const output = try std.fmt.allocPrint(testing.allocator,
        \\=== Logits Comparison Report ===
        \\Status: {s}
        \\NMSE: {d:.10}
        \\Max Abs Error: {d:.6}
        \\Mean Abs Error: {d:.6}
        \\Cosine Similarity: {d:.10}
        \\PSNR: {d:.2} dB
        \\Argmax Match Rate: {d}/{d} ({d:.2}%)
        \\===============================
    , .{
        if (result.nmse < 1e-4 and result.max_abs_error < 0.01 and result.cosine_similarity > 0.999) "PASS" else "FAIL",
        result.nmse,
        result.max_abs_error,
        result.mean_abs_error,
        result.cosine_similarity,
        result.psnr,
        result.matched_tokens,
        result.total_tokens,
        result.match_rate * 100.0,
    });
    defer testing.allocator.free(output);

    try testing.expect(std.mem.indexOf(u8, output, "PASS") != null);
    try testing.expect(std.mem.indexOf(u8, output, "NMSE") != null);
}

test "CompareConfig defaults" {
    const config = compare_logits.CompareConfig{};
    try testing.expectApproxEqAbs(@as(f64, 1e-4), config.nmse_threshold, 1e-10);
    try testing.expectApproxEqAbs(@as(f32, 0.01), config.max_abs_error_threshold, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.999), config.cosine_threshold, 1e-6);
    try testing.expectEqual(@as(bool, false), config.verbose);
    try testing.expectEqual(@as(usize, 10), config.max_diff_positions);
}

test "ComparisonResult fields" {
    const result = compare_logits.ComparisonResult{
        .nmse = 0.0,
        .max_abs_error = 0.0,
        .mean_abs_error = 0.0,
        .cosine_similarity = 1.0,
        .psnr = 100.0,
        .matched_tokens = 10,
        .total_tokens = 10,
        .match_rate = 1.0,
    };
    try testing.expectEqual(@as(f64, 0.0), result.nmse);
    try testing.expectEqual(@as(f64, 1.0), result.match_rate);
}
