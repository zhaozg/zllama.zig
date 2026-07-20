//! 对齐比较工具 — 核心计算函数
//!
//! 提供 computeMetrics、alignmentVerdict 等纯计算函数。
//! 从 align_cmp.zig 中抽取，以保持主文件精简（<600 行）。
//! 委托给共享 metrics 模块以避免重复实现。

const std = @import("std");
const metrics_mod = @import("metrics");
const config_mod = @import("align_cmp_config.zig");

const AlignMetrics = config_mod.AlignMetrics;
const AlignCmpConfig = config_mod.AlignCmpConfig;
const ArgmaxResult = config_mod.ArgmaxResult;

// ============================================================================
// 指标计算
// ============================================================================

/// 计算综合对齐指标
/// 委托共享的 metrics 模块计算 NMSE、余弦相似度、最大绝对误差；
/// 自行计算 L2/RMSE/MAE、相对 MaxErr、离群点统计与比值统计。
pub fn computeMetrics(v1: []const f32, v2: []const f32) AlignMetrics {
    std.debug.assert(v1.len == v2.len);

    const n = v1.len;
    if (n == 0) {
        return AlignMetrics{
            .cosine = 1.0,
            .nmse = 0.0,
            .l2_distance = 0.0,
            .rmse = 0.0,
            .mae = 0.0,
            .max_abs_err = 0.0,
            .rel_max_err = 0.0,
            .ref_max_abs = 0.0,
            .outlier_count = 0,
            .outlier_ratio = 0.0,
            .avg_ratio = 1.0,
            .ratio_std = 0.0,
            .is_scaled = false,
            .dim = 0,
        };
    }

    // 委托共享模块计算核心指标
    const nmse: f64 = metrics_mod.calcNMSE(v1, v2);
    const cosine: f64 = metrics_mod.calcCosineSimilarity(v1, v2);
    const max_abs_err: f64 = @floatCast(metrics_mod.calcMaxAbsError(v1, v2));

    // 自行计算 L2 / MAE / 比值统计 / 离群点统计（一次遍历）
    var sum_sq_diff: f64 = 0.0;
    var sum_abs_err: f64 = 0.0;
    var ratio_sum: f64 = 0.0;
    var ratio_sq_sum: f64 = 0.0;
    var valid_ratio_count: usize = 0;
    var ref_max_abs: f64 = 0.0;

    for (v1, v2) |a, b| {
        const af: f64 = @floatCast(a);
        const bf: f64 = @floatCast(b);

        const diff = af - bf;
        sum_sq_diff += diff * diff;
        sum_abs_err += @abs(diff);

        // 跟踪参考向量最大绝对值（用于相对误差计算）
        const abs_bf = @abs(bf);
        if (abs_bf > ref_max_abs) {
            ref_max_abs = abs_bf;
        }

        // 比值统计（排除零值，避免除零）
        if (@abs(bf) > 1e-20 and @abs(af) > 1e-20) {
            const ratio = af / bf;
            ratio_sum += ratio;
            ratio_sq_sum += ratio * ratio;
            valid_ratio_count += 1;
        }
    }

    const n_f: f64 = @floatFromInt(n);
    const l2_distance: f64 = @sqrt(sum_sq_diff);
    const rmse: f64 = @sqrt(sum_sq_diff / n_f);
    const mae: f64 = sum_abs_err / n_f;

    // 相对最大绝对误差（尺度自适应）
    const eps: f64 = 1e-20;
    const rel_max_err: f64 = max_abs_err / (ref_max_abs + eps);

    // 离群点统计：abs(error) > outlier_sigma * RMSE
    const outlier_threshold: f64 = 3.0 * rmse;
    var outlier_count: usize = 0;
    for (v1, v2) |a, b| {
        const af: f64 = @floatCast(a);
        const bf: f64 = @floatCast(b);
        const diff = af - bf;
        if (@abs(diff) > outlier_threshold) {
            outlier_count += 1;
        }
    }
    const outlier_ratio: f64 = if (n > 0)
        @as(f64, @floatFromInt(outlier_count)) / n_f
    else
        0.0;

    // 平均比值
    const avg_ratio: f64 = if (valid_ratio_count > 0)
        ratio_sum / @as(f64, @floatFromInt(valid_ratio_count))
    else
        1.0;

    // 比值标准差（无偏估计）
    const ratio_std: f64 = if (valid_ratio_count > 2) blk: {
        const count_f: f64 = @floatFromInt(valid_ratio_count);
        const variance: f64 = (ratio_sq_sum - (ratio_sum * ratio_sum) / count_f) / (count_f - 1.0);
        break :blk @sqrt(variance);
    } else 0.0;

    // 缩放检测：变异系数 CV = std/mean < 5% 且均值偏离 > 0.1%
    const ratio_deviation: f64 = @abs(avg_ratio - 1.0);
    const is_scaled: bool = (ratio_std / avg_ratio) < 0.05 and ratio_deviation > 0.001;

    return AlignMetrics{
        .cosine = cosine,
        .nmse = nmse,
        .l2_distance = l2_distance,
        .rmse = rmse,
        .mae = mae,
        .max_abs_err = max_abs_err,
        .rel_max_err = rel_max_err,
        .ref_max_abs = ref_max_abs,
        .outlier_count = outlier_count,
        .outlier_ratio = outlier_ratio,
        .avg_ratio = avg_ratio,
        .ratio_std = ratio_std,
        .is_scaled = is_scaled,
        .dim = n,
    };
}

/// 计算 Argmax 匹配 — 委托给共享 metrics 模块
pub fn calcArgmaxMatch(a: []const f32, b: []const f32) ArgmaxResult {
    return metrics_mod.calcArgmaxMatch(a, b);
}

// ============================================================================
// 判决
// ============================================================================

/// 输出算法对齐的判决结果
///
/// 判决逻辑（按优先级从高到低）:
///   1. 余弦相似度 > tol_cosine   — 方向一致性（首要指标）
///   2. NMSE < tol_nmse            — 归一化均方误差
///   3. 相对 MaxErr < tol_rel_max_err — 尺度自适应最大误差
///   4. RMSE < tol_rmse            — 每维度平均误差
///   5. Argmax 必须匹配            — 最终预测一致性
///   6. 离群点占比检查             — 若占比极小，MaxErr 超标可降级为警告
///   7. 平均幅值比值偏离检查       — 检测系统性缩放
///
/// 注意: is_scaled 仅输出信息性警告，不阻断通过判决。
/// 在 FP16/BF16 CUDA 内核中，非结合律加法可能导致 ~1e-4 NMSE，
/// 这是正常数值误差，不应视为对齐失败。
pub fn alignmentVerdict(al_metrics: AlignMetrics, argmax: ArgmaxResult, config: AlignCmpConfig) []const u8 {
    // ── 硬性失败条件（❌）──

    // 余弦相似度是主要判决指标
    if (al_metrics.cosine < config.tol_cosine) {
        return "❌ 对齐失败: 余弦相似度不满足要求";
    }
    if (al_metrics.nmse > config.tol_nmse) {
        return "❌ 对齐失败: 归一化均方误差 (NMSE) 超出允许范围";
    }
    if (al_metrics.rel_max_err > config.tol_rel_max_err) {
        // 相对 MaxErr 超标时，检查离群点占比
        // 若离群点极少，说明是单个野点导致，降级为警告而非直接失败
        if (al_metrics.outlier_ratio > config.tol_outlier_ratio) {
            return "❌ 对齐失败: 相对最大误差超出允许范围，且离群点占比过高";
        }
        // 离群点极少，降级为警告
        return "⚠️ 对齐通过（需核查）: 相对最大误差超标，但离群点占比极小，可能为个别野点";
    }
    if (al_metrics.rmse > config.tol_rmse) {
        return "❌ 对齐失败: 均方根误差 (RMSE) 超出允许误差";
    }
    if (!argmax.match) {
        return "❌ 对齐失败: Argmax 索引不一致，最终预测结果不同";
    }

    // ── 信息性警告（⚠️，不阻断通过）──
    const ratio_deviation: f64 = @abs(al_metrics.avg_ratio - 1.0);
    if (ratio_deviation > config.tol_ratio_deviation) {
        return "⚠️ 算法对齐验证通过，但存在系统性幅值偏差（平均比值偏离 1.0），建议检查 scale 因子";
    }

    // 综合检查：绝对 MaxErr 超标但相对 MaxErr 正常（数据尺度大导致）
    if (al_metrics.max_abs_err > config.tol_max_abs_err and al_metrics.rel_max_err <= config.tol_rel_max_err) {
        return "✅ 算法对齐验证通过（注意: 绝对 MaxErr 较大但相对误差正常，由数据尺度引起）";
    }

    return "✅ 算法对齐验证通过！";
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "computeMetrics identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const m = computeMetrics(&a, &b);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.nmse, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.cosine, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.l2_distance, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.rmse, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.mae, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.max_abs_err, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.rel_max_err, 1e-10);
    try testing.expectEqual(@as(usize, 5), m.dim);
    try testing.expect(!m.is_scaled);
    try testing.expectEqual(@as(usize, 0), m.outlier_count);
}

test "computeMetrics scaled vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 2.0, 4.0, 6.0 };
    const m = computeMetrics(&a, &b);
    // NMSE = (1+4+9)/(1+4+9) = 1.0 for exact 2x scaling
    try testing.expect(m.nmse > 0.9);
    // Cosine should be ~1.0 (direction is same)
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.cosine, 1e-6);
}

test "computeMetrics empty" {
    const a: [0]f32 = .{};
    const b: [0]f32 = .{};
    const m = computeMetrics(&a, &b);
    try testing.expectEqual(@as(usize, 0), m.dim);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.cosine, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.nmse, 1e-10);
}

test "computeMetrics rel_max_err" {
    // 大尺度数据：幅值 ~1000，误差 ~1，相对误差应很小
    const a = [_]f32{ 1000.0, 2000.0, 3000.0 };
    const b = [_]f32{ 1001.0, 2000.0, 3000.0 };
    const m = computeMetrics(&a, &b);
    // 绝对 MaxErr = 1.0，但相对 MaxErr ≈ 1/3000 ≈ 3.3e-4
    try testing.expect(m.max_abs_err > 0.9);
    try testing.expect(m.rel_max_err < 0.001);
    try testing.expect(m.ref_max_abs > 2999.0);
}

test "computeMetrics outlier detection" {
    // 一个野点，其余完全一致
    var a: [100]f32 = undefined;
    var b: [100]f32 = undefined;
    for (&a, &b, 0..) |*pa, *pb, i| {
        pa.* = @floatFromInt(i);
        pb.* = @floatFromInt(i);
    }
    b[50] = 1000.0; // 野点
    const m = computeMetrics(&a, &b);
    try testing.expect(m.outlier_count >= 1);
    try testing.expect(m.outlier_ratio > 0.0);
}

test "alignmentVerdict pass" {
    const m = AlignMetrics{
        .cosine = 0.99999,
        .nmse = 1e-6,
        .l2_distance = 0.001,
        .rmse = 0.0001,
        .mae = 0.00005,
        .max_abs_err = 0.001,
        .rel_max_err = 1e-6,
        .ref_max_abs = 100.0,
        .outlier_count = 0,
        .outlier_ratio = 0.0,
        .avg_ratio = 1.0001,
        .ratio_std = 0.00001,
        .is_scaled = false,
        .dim = 4096,
    };
    const argmax = ArgmaxResult{ .ours = 42, .ref = 42, .match = true };
    const config = AlignCmpConfig{};
    const verdict = alignmentVerdict(m, argmax, config);
    try testing.expect(std.mem.startsWith(u8, verdict, "✅"));
}

test "alignmentVerdict fail nmse" {
    const m = AlignMetrics{
        .cosine = 0.99999,
        .nmse = 0.1,
        .l2_distance = 10.0,
        .rmse = 0.1,
        .mae = 0.05,
        .max_abs_err = 0.001,
        .rel_max_err = 1e-6,
        .ref_max_abs = 100.0,
        .outlier_count = 0,
        .outlier_ratio = 0.0,
        .avg_ratio = 1.0,
        .ratio_std = 0.0,
        .is_scaled = false,
        .dim = 4096,
    };
    const argmax = ArgmaxResult{ .ours = 42, .ref = 42, .match = true };
    const config = AlignCmpConfig{};
    const verdict = alignmentVerdict(m, argmax, config);
    try testing.expect(std.mem.startsWith(u8, verdict, "❌"));
}

test "alignmentVerdict fail argmax" {
    const m = AlignMetrics{
        .cosine = 0.99999,
        .nmse = 1e-6,
        .l2_distance = 0.001,
        .rmse = 0.0001,
        .mae = 0.00005,
        .max_abs_err = 0.001,
        .rel_max_err = 1e-6,
        .ref_max_abs = 100.0,
        .outlier_count = 0,
        .outlier_ratio = 0.0,
        .avg_ratio = 1.0,
        .ratio_std = 0.0,
        .is_scaled = false,
        .dim = 4096,
    };
    const argmax = ArgmaxResult{ .ours = 42, .ref = 99, .match = false };
    const config = AlignCmpConfig{};
    const verdict = alignmentVerdict(m, argmax, config);
    try testing.expect(std.mem.startsWith(u8, verdict, "❌"));
}

test "alignmentVerdict pass with large scale data" {
    // 模拟池化后的情况：数值范围大，绝对 MaxErr 大，但相对误差小
    const m = AlignMetrics{
        .cosine = 0.999998,
        .nmse = 1e-5,
        .l2_distance = 100.0,
        .rmse = 2.0,
        .mae = 1.0,
        .max_abs_err = 30.0,
        .rel_max_err = 1e-5, // 相对误差很小
        .ref_max_abs = 3000.0,
        .outlier_count = 0,
        .outlier_ratio = 0.0,
        .avg_ratio = 1.0001,
        .ratio_std = 0.00001,
        .is_scaled = false,
        .dim = 4096,
    };
    const argmax = ArgmaxResult{ .ours = 42, .ref = 42, .match = true };
    const config = AlignCmpConfig{};
    const verdict = alignmentVerdict(m, argmax, config);
    // 应该通过，因为相对误差小，且余弦相似度极高
    try testing.expect(std.mem.startsWith(u8, verdict, "✅"));
}

test "alignmentVerdict warn with single outlier" {
    // 单个野点导致 MaxErr 大，但离群点占比极小
    const m = AlignMetrics{
        .cosine = 0.99999,
        .nmse = 1e-5,
        .l2_distance = 1.0,
        .rmse = 0.01,
        .mae = 0.001,
        .max_abs_err = 10.0,
        .rel_max_err = 0.01, // 相对误差超标
        .ref_max_abs = 100.0,
        .outlier_count = 1,
        .outlier_ratio = 0.0001, // 占比极小（0.01%）
        .avg_ratio = 1.0,
        .ratio_std = 0.0,
        .is_scaled = false,
        .dim = 10000,
    };
    const argmax = ArgmaxResult{ .ours = 42, .ref = 42, .match = true };
    const config = AlignCmpConfig{};
    const verdict = alignmentVerdict(m, argmax, config);
    // 应该降级为警告而非失败
    try testing.expect(std.mem.startsWith(u8, verdict, "⚠️"));
}

test "alignmentVerdict fail with many outliers" {
    // 多个野点导致 MaxErr 大，且离群点占比高
    const m = AlignMetrics{
        .cosine = 0.999,
        .nmse = 1e-3,
        .l2_distance = 10.0,
        .rmse = 0.1,
        .mae = 0.05,
        .max_abs_err = 10.0,
        .rel_max_err = 0.1, // 相对误差超标
        .ref_max_abs = 100.0,
        .outlier_count = 100,
        .outlier_ratio = 0.01, // 占比 1%，超标
        .avg_ratio = 1.0,
        .ratio_std = 0.0,
        .is_scaled = false,
        .dim = 10000,
    };
    const argmax = ArgmaxResult{ .ours = 42, .ref = 42, .match = true };
    const config = AlignCmpConfig{};
    const verdict = alignmentVerdict(m, argmax, config);
    // 应该失败
    try testing.expect(std.mem.startsWith(u8, verdict, "❌"));
}
