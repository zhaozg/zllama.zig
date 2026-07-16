//! 共享指标计算函数
//!
//! 提供 NMSE、余弦相似度等工具间共享的指标计算。
//! 被 compare_mtmd_vision, compare_mtmd_audio, compare_with_llamacpp 等工具引用。

const std = @import("std");

/// Argmax 匹配结果
pub const ArgmaxResult = struct {
    ours: usize,
    ref: usize,
    match: bool,
};

/// NMSE（归一化均方误差）
/// NMSE = sum((a - b)^2) / (sum(a^2) + 1e-10)
pub fn calcNMSE(a: []const f32, b: []const f32) f64 {
    var sum_sq_err: f64 = 0.0;
    var sum_sq_ref: f64 = 0.0;
    for (a, b) |av, bv| {
        const err: f64 = @as(f64, @floatCast(av)) - @as(f64, @floatCast(bv));
        sum_sq_err += err * err;
        sum_sq_ref += @as(f64, @floatCast(av)) * @as(f64, @floatCast(av));
    }
    return sum_sq_err / (sum_sq_ref + 1e-10);
}

/// 余弦相似度
pub fn calcCosineSimilarity(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0.0;
    var norm_a: f64 = 0.0;
    var norm_b: f64 = 0.0;
    for (a, b) |av, bv| {
        dot += @as(f64, @floatCast(av)) * @as(f64, @floatCast(bv));
        norm_a += @as(f64, @floatCast(av)) * @as(f64, @floatCast(av));
        norm_b += @as(f64, @floatCast(bv)) * @as(f64, @floatCast(bv));
    }
    return dot / (@sqrt(norm_a) * @sqrt(norm_b) + 1e-10);
}

/// 最大绝对误差
pub fn calcMaxAbsError(a: []const f32, b: []const f32) f32 {
    var max_err: f32 = 0.0;
    for (a, b) |av, bv| {
        const err = @abs(av - bv);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/// Argmax 匹配检查
pub fn calcArgmaxMatch(a: []const f32, b: []const f32) ArgmaxResult {
    var max_ours: f32 = -std.math.inf(f32);
    var max_ref: f32 = -std.math.inf(f32);
    var idx_ours: usize = 0;
    var idx_ref: usize = 0;
    for (a, 0..) |v, i| {
        if (v > max_ours) {
            max_ours = v;
            idx_ours = i;
        }
    }
    for (b, 0..) |v, i| {
        if (v > max_ref) {
            max_ref = v;
            idx_ref = i;
        }
    }
    return .{ .ours = idx_ours, .ref = idx_ref, .match = idx_ours == idx_ref };
}

/// 打印带阈值的指标
pub fn printMetric(io: std.Io, name: []const u8, value: anytype, threshold: anytype, lower_is_better: bool) !void {
    const stdout_file = std.Io.File.stdout();
    var buf: [256]u8 = undefined;
    const pass = if (lower_is_better) value < threshold else value > threshold;
    const status = if (pass) "✅" else "❌";
    const line = try std.fmt.bufPrint(&buf, "  {s} {s}: {e} (threshold: {e})\n", .{ status, name, value, threshold });
    try stdout_file.writeStreamingAll(io, line);
}

/// 打印 Argmax 匹配结果
pub fn printArgmaxResult(io: std.Io, argmax: ArgmaxResult) !void {
    const stdout_file = std.Io.File.stdout();
    var buf: [256]u8 = undefined;
    const status = if (argmax.match) "✅" else "❌";
    const line = try std.fmt.bufPrint(&buf, "  {s} Argmax: ours={d}, ref={d}, match={}\n", .{ status, argmax.ours, argmax.ref, argmax.match });
    try stdout_file.writeStreamingAll(io, line);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "calcNMSE identical" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expectApproxEqAbs(@as(f64, 0.0), calcNMSE(&a, &b), 1e-10);
}

test "calcNMSE different" {
    const a = [_]f32{ 1.0, 1.0, 1.0 };
    const b = [_]f32{ 2.0, 2.0, 2.0 };
    const result = calcNMSE(&a, &b);
    try testing.expect(result > 0.1);
}

test "calcCosineSimilarity identical" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expectApproxEqAbs(@as(f64, 1.0), calcCosineSimilarity(&a, &b), 1e-6);
}

test "calcCosineSimilarity opposite" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ -1.0, -2.0, -3.0 };
    try testing.expectApproxEqAbs(@as(f64, -1.0), calcCosineSimilarity(&a, &b), 1e-6);
}

test "calcMaxAbsError" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.5, 3.0 };
    try testing.expectApproxEqAbs(@as(f32, 0.5), calcMaxAbsError(&a, &b), 1e-6);
}

test "calcArgmaxMatch" {
    const a = [_]f32{ 1.0, 5.0, 3.0 };
    const b = [_]f32{ 1.0, 5.0, 3.0 };
    const result = calcArgmaxMatch(&a, &b);
    try testing.expect(result.match);
    try testing.expectEqual(@as(usize, 1), result.ours);
    try testing.expectEqual(@as(usize, 1), result.ref);
}
