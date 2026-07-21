//! 对齐比较工具 — 配置类型与指标结构体
//!
//! 定义比较模式、阈值配置、综合指标等纯数据类型。
//! 从 align_cmp.zig 中抽取，以保持主文件精简（<600 行）。

const std = @import("std");
const metrics_mod = @import("metrics");

// ============================================================================
// 比较模式
// ============================================================================

/// 比较模式
pub const CompareMode = enum {
    /// 普通模式：输出相似度等级
    general,
    /// 严格对齐模式：输出通过/失败判决
    alignment,
};

/// 输出格式
pub const OutputFormat = enum {
    /// 人类可读：带装饰线、emoji、中文标签
    human,
    /// AI/机器可读：紧凑的 KEY=VALUE 格式，无 emoji，适合脚本解析
    ai,
};

// ============================================================================
// 配置
// ============================================================================

/// 对齐比较配置
///
/// 默认值采用工业级严格标准，适合 FP16/BF16 CUDA 内核验证。
/// 在 FP16/BF16 下，非结合律浮点加法会导致 ~1e-4 量级的 NMSE，
/// 这是正常的数值误差范围，不应视为对齐失败。
/// 所有指标平等参与判决，任一不满足即判定为失败。
pub const AlignCmpConfig = struct {
    /// 参考文件路径
    ref_path: []const u8 = "",
    /// 测试文件路径
    test_path: []const u8 = "",
    /// JSON 向量路径（点分隔，如 "data.emb"）
    key: ?[]const u8 = null,
    /// 比较模式
    mode: CompareMode = .alignment,
    /// 输出格式
    output_format: OutputFormat = .human,

    // ── 指标阈值（工业级严格） ──

    /// NMSE 最大容忍度
    /// NMSE = sum((a-b)^2) / sum(a^2)
    /// 推荐: < 1e-4（FP16/BF16），< 1e-5（FP32 完全确定性）
    tol_nmse: f64 = 1e-4,

    /// 余弦相似度最低要求
    /// 推荐: > 0.9999（高维嵌入下 FP16 误差典型值）
    tol_cosine: f64 = 0.9999,

    /// RMSE 最大容忍度（每维度平均误差）
    /// 推荐: < 0.001（4096 维下对应 L2 ≈ 0.064）
    tol_rmse: f64 = 0.001,

    /// 最大绝对误差容忍度
    /// 用于检测单个维度上的异常偏差，防止 argmax 翻转
    tol_max_abs_err: f64 = 0.01,

    /// 相对最大绝对误差容忍度（尺度自适应）
    /// rel_max_err = max_abs_err / max(abs(ref), eps)
    /// 推荐: < 1e-4（适用于任意数据尺度）
    tol_rel_max_err: f64 = 1e-4,

    /// 平均幅值比值允许偏离 1.0 的容忍度
    /// 用于检测系统性缩放（如缺失的 scale 因子）
    tol_ratio_deviation: f64 = 0.001,

    /// 离群点占比容忍度（异常点数量 / 总维度）
    /// 用于判断 MaxErr 高是否由个别野点引起
    /// 推荐: < 0.001（0.1%）
    tol_outlier_ratio: f64 = 0.001,

    /// 离群点判定倍数：abs(error) > outlier_sigma * RMSE 视为离群点
    outlier_sigma: f64 = 3.0,
};

// ============================================================================
// 指标结果
// ============================================================================

/// 综合对齐指标
pub const AlignMetrics = struct {
    /// 余弦相似度
    cosine: f64,
    /// 归一化均方误差 (NMSE)
    nmse: f64,
    /// 欧几里得距离 (L2) — 仅参考，不用于判决
    l2_distance: f64,
    /// 均方根误差 (RMSE) = L2 / sqrt(dim)
    rmse: f64,
    /// 平均绝对误差 (MAE)
    mae: f64,
    /// 最大绝对误差 (MaxAbsErr) — 检测异常离群点
    max_abs_err: f64,
    /// 相对最大绝对误差（尺度自适应）
    /// rel_max_err = max_abs_err / max(abs(ref), eps)
    rel_max_err: f64,
    /// 参考向量最大绝对值（用于相对误差计算）
    ref_max_abs: f64,
    /// 离群点数量（abs(error) > outlier_sigma * RMSE）
    outlier_count: usize,
    /// 离群点占比（outlier_count / dim）
    outlier_ratio: f64,
    /// 平均幅值比值 (A/B)
    avg_ratio: f64,
    /// 幅值比值的标准差
    ratio_std: f64,
    /// 是否疑似存在线性缩放
    is_scaled: bool,
    /// 向量维度
    dim: usize,
};

/// Argmax 匹配结果（重新导出共享类型以保证 API 兼容）
pub const ArgmaxResult = metrics_mod.ArgmaxResult;
