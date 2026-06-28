#!/usr/bin/env python3
"""
通用浮点数数组对比工具（增强版）
用法:
  python compare-float.py [--file1 FILE1] [--file2 FILE2] [--verbose]
"""

import json
import numpy as np
import argparse
import sys

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    arr = np.array(data, dtype=np.float32)
    return arr.flatten() if arr.ndim > 1 else arr

def compare_arrays(a, b, file1, file2, verbose=False):
    if a.shape != b.shape:
        print(f"❌ 形状不匹配: {a.shape} vs {b.shape}")
        return False

    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)

    # 基本统计
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
        '均方误差 (MSE)': np.mean(diff ** 2),
        '均方根误差 (RMSE)': np.sqrt(np.mean(diff ** 2)),
    }

    # 余弦相似度
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_sim = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

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

    # --- 差异程度判定 ---
    # 等级阈值定义（可根据需要调整）
    if max_abs > 0.1 or mean_abs > 0.01:
        level = "🔴 显著差异"
        detail = "数值存在实质差异，可能严重影响下游结果"
    elif max_abs > 0.01 or mean_abs > 0.001:
        level = "🟡 明显差异"
        detail = "存在不可忽略的偏差，建议排查原因"
    elif max_abs > 0.001 or mean_abs > 0.0001:
        level = "🟢 微小差异"
        detail = "偏差在可接受范围内，通常不影响最终结论"
    else:
        level = "✅ 几乎一致"
        detail = "在浮点误差范围内，可视为完全相同"

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
    is_identical = (max_abs <= 0.001 and mean_abs <= 0.0001)
    return is_identical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比两个 JSON 浮点数组文件")
    parser.add_argument("--file1", default="llama_audio_mel.json", help="第一个文件路径")
    parser.add_argument("--file2", default="zllama_audio_mel.json", help="第二个文件路径")
    parser.add_argument("--verbose", action="store_true", help="输出详细统计信息及 Top 5 差异")
    args = parser.parse_args()

    file1, file2 = args.file1, args.file2

    try:
        a = load_json(file1)
        b = load_json(file2)
        ok = compare_arrays(a, b, file1, file2, verbose=args.verbose)
        sys.exit(0 if ok else 1)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
