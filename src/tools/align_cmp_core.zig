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
/// 自行计算 L2/RMSE/MAE 与比值统计。
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
            .avg_ratio = 1.0,
            .ratio_std = 0.0,
            .is_scaled = false,
            .dim = 0,
        };
    }

    // 委托共享模块计算核心指标（一次遍历各自计算，避免重复遍历）
    const nmse: f64 = metrics_mod.calcNMSE(v1, v2);
    const cosine: f64 = metrics_mod.calcCosineSimilarity(v1, v2);
    const max_abs_err: f64 = @floatCast(metrics_mod.calcMaxAbsError(v1, v2));

    // 自行计算 L2 / MAE / 比值统计（一次遍历）
    var sum_sq_diff: f64 = 0.0;
    var sum_abs_err: f64 = 0.0;
    var ratio_sum: f64 = 0.0;
    var ratio_sq_sum: f64 = 0.0;
    var valid_ratio_count: usize = 0;
    var max_abs_val: f64 = 0.0;

    for (v1, v2) |a, b| {
        const af: f64 = @floatCast(a);
        const bf: f64 = @floatCast(b);

        const diff = af - bf;
        sum_sq_diff += diff * diff;
        sum_abs_err += @abs(diff);

        // 跟踪最大绝对值（用于相对误差计算）
        const abs_val = @max(@abs(af), @abs(bf));
        if (abs_val > max_abs_val) {
            max_abs_val = abs_val;
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
///   1. NMSE < tol_nmse          — 归一化均方误差
///   2. 余弦相似度 > tol_cosine   — 方向一致性
///   3. RMSE < tol_rmse           — 每维度平均误差
///   4. 最大绝对误差 < tol_max_abs_err — 防止 argmax 翻转
///   5. Argmax 必须匹配           — 最终预测一致性
///   6. 平均幅值比值偏离 < tol_ratio_deviation — 检测系统性缩放
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
    if (al_metrics.rmse > config.tol_rmse) {
        return "❌ 对齐失败: 均方根误差 (RMSE) 超出允许误差";
    }
    if (al_metrics.max_abs_err > config.tol_max_abs_err) {
        return "❌ 对齐失败: 最大绝对误差超出允许范围，存在 argmax 翻转风险";
    }
    if (!argmax.match) {
        return "❌ 对齐失败: Argmax 索引不一致，最终预测结果不同";
    }

    // ── 信息性警告（⚠️，不阻断通过）──
    const ratio_deviation: f64 = @abs(al_metrics.avg_ratio - 1.0);
    if (ratio_deviation > config.tol_ratio_deviation) {
        return "⚠️ 算法对齐验证通过，但存在系统性幅值偏差（平均比值偏离 1.0），建议检查 scale 因子";
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
    try testing.expectEqual(@as(usize, 5), m.dim);
    try testing.expect(!m.is_scaled);
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

test "alignmentVerdict pass" {
    const m = AlignMetrics{
        .cosine = 0.99999,
        .nmse = 1e-6,
        .l2_distance = 0.001,
        .rmse = 0.0001,
        .mae = 0.00005,
        .max_abs_err = 0.001,
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

test "alignmentVerdict pass with large rmse but high cosine" {
    // 模拟池化后的情况：数值范围大，RMSE 偏大，但余弦相似度极高
    const m = AlignMetrics{
        .cosine = 0.999998,
        .nmse = 1e-5,
        .l2_distance = 100.0,
        .rmse = 2.0,
        .mae = 1.0,
        .max_abs_err = 30.0,
        .avg_ratio = 1.0001,
        .ratio_std = 0.00001,
        .is_scaled = false,
        .dim = 4096,
    };
    const argmax = ArgmaxResult{ .ours = 42, .ref = 42, .match = true };
    const config = AlignCmpConfig{};
    const verdict = alignmentVerdict(m, argmax, config);
    // 应该通过，因为余弦相似度极高，且 RMSE 和 MaxErr 在放宽后的阈值内
    try testing.expect(std.mem.startsWith(u8, verdict, "✅"));
}
