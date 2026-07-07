#!/usr/bin/env python3
"""
通用浮点数数组对比工具（增强版）
用法:
  python compare-float.py [--file1 FILE1] [--file2 FILE2] [--verbose]
                         [--cos-threshold COS] [--rel-threshold REL]
"""

import json
import numpy as np
import argparse
import sys

def load_json(filename):
    """从 JSON 文件加载数据并返回扁平化 float32 数组"""
    with open(filename, 'r') as f:
        data = json.load(f)
    arr = np.array(data, dtype=np.float32)
    return arr.flatten() if arr.ndim > 1 else arr

def compare_arrays(a, b, file1, file2, verbose=False, cos_threshold=0.9999, rel_threshold=0.01):
    """
    对比两个数组并输出统计信息
    返回 True 表示视为一致，False 表示存在显著差异
    """
    if a.shape != b.shape:
        print(f"❌ 形状不匹配: {a.shape} vs {b.shape}")
        return False

    # 计算基本差异
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    # 基础统计
    stats = {
        '长度': len(a),
        '均值 (a)': np.mean(a),
        '均值 (b)': np.mean(b),
        '标准差 (a)': np.std(a),
        '标准差 (b)': np.std(b),
        '最小值 (a)': np.min(a),
        '最大值 (a)': np.max(a),
        '最小值 (b)': np.min(b),
        '最大值 (b)': np.max(b),
    }

    diff_stats = {
        '绝对差均值': mean_abs,
        '绝对差最大值': max_abs,
        '绝对差中位数': np.median(abs_diff),
        '均方误差 (MSE)': mse,
        '均方根误差 (RMSE)': rmse,
    }

    # 余弦相似度
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_sim = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

    # 相对误差
    eps = 1e-8
    max_abs_a = np.max(np.abs(a))
    max_abs_b = np.max(np.abs(b))
    max_abs_combined = max(max_abs_a, max_abs_b)
    rel_max_abs = max_abs / (max_abs_combined + eps)

    mean_abs_a = np.mean(np.abs(a))
    mean_abs_b = np.mean(np.abs(b))
    mean_abs_combined = max(mean_abs_a, mean_abs_b)
    rel_mean_abs = mean_abs / (mean_abs_combined + eps)

    # ===== 差异等级判定（新逻辑）=====
    # 优先检查余弦相似度
    if cos_sim < cos_threshold:
        level = "🔴 显著差异"
        detail = f"余弦相似度 {cos_sim:.4f} 低于阈值 {cos_threshold}"
    elif rel_max_abs > rel_threshold:
        level = "🔴 显著差异"
        detail = f"最大相对误差 {rel_max_abs:.2%} 超过 {rel_threshold:.0%}"
    elif rel_mean_abs > 0.001:
        level = "🟡 明显差异"
        detail = f"平均相对误差 {rel_mean_abs:.2%} 超过 0.1%"
    elif max_abs > 0.012 or mean_abs > 0.001:
        level = "🟡 明显差异"
        detail = "绝对误差超出经验阈值"
    elif max_abs > 0.001 or mean_abs > 0.0001:
        level = "🟢 微小差异"
        detail = "偏差微小，通常无影响"
    else:
        level = "✅ 几乎一致"
        detail = "在浮点误差范围内"

    # 特殊情况：余弦相似度很高但绝对误差大（数据量级大）
    if cos_sim >= cos_threshold and max_abs > 1.0 and mean_abs > 0.1:
        level = "🟢 微小差异"
        detail = "余弦相似度很高，但绝对差异较大，可能源于量级"

    # ===== 输出 =====
    print("=" * 60)
    print(f"对比文件: {file1} vs {file2}")
    print("=" * 60)

    if verbose:
        print("\n📊 基本统计:")
        for key, val in stats.items():
            print(f"  {key:12} : {val:.6f}")

    print("\n📉 差异统计:")
    for key, val in diff_stats.items():
        print(f"  {key:16} : {val:.6e}")

    print(f"\n🔗 余弦相似度: {cos_sim:.10f}")
    print(f"📊 最大相对误差: {rel_max_abs:.2%}")
    print(f"📊 平均相对误差: {rel_mean_abs:.2%}")

    print(f"\n📌 差异等级: {level}")
    print(f"   {detail}")
    print(f"   最大绝对差: {max_abs:.6f}, 平均绝对差: {mean_abs:.6f}")

    # 若存在差异，显示最大偏差位置（如果verbose）
    if verbose and max_abs > 0.0001:
        max_idx = np.argmax(abs_diff)
        print(f"   最大差异位置: {max_idx}, 值: a={a.flat[max_idx]:.6f}, b={b.flat[max_idx]:.6f}, diff={diff.flat[max_idx]:.6f}")
        # 显示前5个大差异位置
        top5_idx = np.argsort(abs_diff)[-5:][::-1]
        print("\n   Top 5 差异位置:")
        for idx in top5_idx:
            print(f"    [{idx:6d}] a={a[idx]:.6f}  b={b[idx]:.6f}  diff={diff[idx]:.6f}")

    # 返回是否一致（用于退出码）
    is_identical = (cos_sim >= cos_threshold and rel_max_abs < rel_threshold and rel_mean_abs < 0.001)
    return is_identical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比两个 JSON 浮点数组文件（增强判定）")
    parser.add_argument("--file1", default="llama_audio_mel.json", help="第一个文件路径")
    parser.add_argument("--file2", default="zllama_audio_mel.json", help="第二个文件路径")
    parser.add_argument("--verbose", action="store_true", help="输出详细统计信息及 Top 5 差异")
    parser.add_argument("--cos-threshold", type=float, default=0.9999, help="余弦相似度阈值（默认0.9999）")
    parser.add_argument("--rel-threshold", type=float, default=0.01, help="最大相对误差阈值（默认0.01）")
    args = parser.parse_args()

    try:
        a = load_json(args.file1)
        b = load_json(args.file2)
        ok = compare_arrays(
            a, b,
            file1=args.file1,
            file2=args.file2,
            verbose=args.verbose,
            cos_threshold=args.cos_threshold,
            rel_threshold=args.rel_threshold
        )
        sys.exit(0 if ok else 1)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"未预期错误: {e}")
        sys.exit(1)
