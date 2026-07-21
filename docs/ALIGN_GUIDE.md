# 算法对齐验证指南 (Alignment Guide)

> **项目**: zllama.zig — 纯 Zig 实现的多模型本地推理引擎  
> **参考**: llama.cpp (deps/llama.cpp)

---

## 目录

1. [核心概念](#1-核心概念)
2. [工具清单](#2-工具清单)
3. [文本 LLM 对齐流水线](#3-文本-llm-对齐流水线)
4. [多模态视觉对齐流水线](#4-多模态视觉对齐流水线)
5. [指标详解](#5-指标详解)
6. [阈值配置](#6-阈值配置)
7. [判决逻辑](#7-判决逻辑)
8. [调试错位层](#8-调试错位层)
9. [CI 自动化](#9-ci-自动化)
10. [扩展指南](#10-扩展指南)

---

## 1. 核心概念

### 1.1 为什么需要对齐验证

zllama.zig 是 llama.cpp 的 Zig 移植。两者使用完全相同的计算图拓扑（ggml 算子序列），因此**在相同输入、相同权重下，中间张量和最终 logits 必须数值一致**。

对齐验证分两层：

| 层级 | 粒度 | 用途 |
|------|------|------|
| **端到端** | 最终 logits | 确认推理结果一致，冒烟测试 |
| **逐层** | 每层 ViT / Attention / FFN 输出 | 定位第一个出现偏差的计算步骤 |

### 1.2 数据流模型

```
    参考 (llama.cpp)              测试 (zllama.zig)
    ────────────────              ────────────────
    模型.gguf + 图像.png ────→    相同模型 + 相同图像
           │                            │
    按层 dump 张量 ─── JSON ───→  按层 dump 张量 ─── JSON
           │                            │
           └──── align-cmp 逐对比较 ────┘
                        │
                   PASS / FAIL
```

### 1.3 命名约定

**参考文件**（llama.cpp 输出）：

```
debug_vision/llama_vision_{stage}.json
```

**测试文件**（zllama.zig 输出）：

```
debug_vision/zllama_vision_{stage}.json
```

`stage` 使用两字符序号（`00`-`07`）+ 可选的层级后缀标识流水线中的位置。

---

## 2. 工具清单

### 2.1 对齐比较器 `zllama-align-cmp`

**主入口**，用于比较两个 JSON 或二进制格式的向量文件。

```
zllama-align-cmp --ref <ref.json> --test <test.json> [options]
```

**关键参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ref`, `--test` | 参考 / 测试文件路径 | 必需 |
| `--key` | JSON 中的向量路径（如 `data.emb`） | 自动检测 |
| `--output` | 输出格式：`human` / `ai` | `human` |
| `--tol-nmse` | NMSE 上限 | `1e-4` |
| `--tol-cosine` | 余弦相似度下限 | `0.9999` |
| `--tol-rmse` | RMSE 上限 | `0.001` |
| `--tol-rel-max-err` | 相对 MaxErr 上限 | `1e-4` |
| `--tol-outlier-ratio` | 离群点占比上限 | `0.001` |
| `--tol-ratio-dev` | 幅值比偏离上限 | `0.001` |

**输出格式**：

- `human`：带装饰线、emoji、中文标签，适合人工审阅
- `ai`：紧凑的 `KEY=VALUE` 格式，适合脚本解析（`grep PASS=`, `awk` 等）

**源码结构**：

```
src/tools/align_cmp.zig           # 主入口（CLI + I/O）
src/tools/align_cmp_config.zig    # 配置类型 + 指标结构体
src/tools/align_cmp_core.zig      # 核心计算（computeMetrics / alignmentVerdict）
src/tools/metrics.zig             # 共享指标函数（NMSE / 余弦 / MaxAbs / Argmax）
```

### 2.2 端到端对比 `zllama-compare-llamacpp`

加载模型 → 推理 → 直接与 llama.cpp 参考 logits 对比。

```bash
zllama-compare-llamacpp --model model.gguf --prompt "Hello" --ref-logits ref.bin
```

### 2.3 Logits 对比 `zllama-compare-logits`

比较两组已保存的 logits 二进制文件。

```bash
zllama-compare-logits --ref ref_logits.bin --test test_logits.bin
```

### 2.4 参考生成 `zllama-gen-ref`

运行推理并将 logits 保存为金标准（Golden Master）。

```bash
zllama-gen-ref --model model.gguf --prompt "Hello" --output ref.bin
```

### 2.5 多模态视觉验证 `zllama-compare-mtmd-vision`

完整多模态 pipeline 验证（图像编码 → 投影 → 文本 LLM）。

```bash
zllama-compare-mtmd-vision \
    --model model.gguf --mmproj mmproj.gguf --image hello.png \
    --prompt "Describe this image" --ref-logits ref.bin
```

### 2.6 多模态音频验证 `zllama-compare-mtmd-audio`

与视觉类似，针对音频编码 pipeline。

```bash
zllama-compare-mtmd-audio \
    --model model.gguf --mmproj mmproj.gguf --audio hello.wav \
    --ref-logits ref.bin
```

---

## 3. 文本 LLM 对齐流水线

### 3.1 工作流

```
Step 1: llama.cpp 生成参考 logits
───────────────────────────────────
echo /exit | llama-cli -m model.gguf -p "Hello" --logit-binary ref.bin

Step 2: zllama.zig 推理 + 对比
───────────────────────────────────
zllama-compare-llamacpp --model model.gguf --prompt "Hello" --ref-logits ref.bin

Step 3: 解读结果
───────────────────────────────────
✅ NMSE: 1.234e-06 (threshold: 1e-4)
✅ Cosine: 0.999987 (threshold: 0.999)
✅ Max Abs Error: 0.0012
✅ Argmax: ours=1234, ref=1234, match=true
→ PASS
```

### 3.2 基准阈值

| 指标 | FP32 (确定性) | Q4_K_M (量化) |
|------|--------------|---------------|
| NMSE | `< 1e-5` | `< 1e-4` |
| 余弦相似度 | `> 0.99999` | `> 0.9999` |
| Argmax | 必须一致 | 必须一致 |

### 3.3 常见失败原因

| 症状 | 可能原因 |
|------|----------|
| NMSE > 0.1 | 权重加载偏移，或 Attention Mask 错误 |
| Argmax 不一致 | RoPE 频率计算错误，或最后一层 norm 缺失 |
| 余弦 ~0.7-0.9 | 系统性缩放，检查 Embedding / lm_head 的 scale 因子 |
| 所有指标完美但输出不同 | 采样器差异（非模型问题） |

---

## 4. 多模态视觉对齐流水线

### 4.1 debug_vision 目录结构

```
debug_vision/
├── llama_vision_{stage}.json    # llama.cpp 参考（32 个文件）
└── zllama_vision_{stage}.json   # zllama.zig 测试（34 个文件，含权重）
```

### 4.2 流水线阶段

视觉编码器经过以下阶段，每阶段都可以导出中间张量进行对比：

| Stage | 文件后缀 | 含义 |
|-------|----------|------|
| **00** | `00_images` | 原始图像数据（RGB 归一化） |
| **00** | `00_inp_raw` | 输入张量原始值 |
| **01** | `01_inp_raw_scaled` | 缩放后的输入 |
| **02** | `02z_inp_final` | 最终预处理输入 |
| **03** | `03_pos_embd` | 位置嵌入 |
| **04** | `04d_layer_0_out` ~ `04d_layer_e_out` | ViT 层 0–14 输出（16 层，hex 序号） |
| **04** | `04d_layer_3_*` | 第 3 层内部子步骤（详见 8.2） |
| **04** | `04g_vit_out_scaled` | ViT 输出缩放 |
| **04** | `04z_vit_output` | ViT 最终输出 |
| **05** | `05a_pooled-*` ~ `05z_pooled` | 池化子步骤 |
| **07** | `07_mm_output` | 多模态最终输出（嵌入 LLM） |

> **hex 层号**: `04d_layer_0` 到 `04d_layer_9`，然后 `a`(10), `b`(11), `c`(12), `d`(13), `e`(14)。

### 4.3 运行全量对齐

```bash
# ReleaseFast 构建 + 全量 32 项对齐
zig build -Doptimize=ReleaseFast

cd debug_vision
pass=0 fail=0
for f in $(ls llama_*.json | sort); do
    result=$(../zig-out/bin/zllama-align-cmp \
        --ref "$f" --test "z$f" --output ai \
        --tol-rel-max-err 5e-4 --tol-outlier-ratio 0.02 --tol-rmse 0.02 \
        2>&1)
    if echo "$result" | grep -q "PASS=1"; then
        pass=$((pass+1))
    else
        fail=$((fail+1))
    fi
done
echo "PASS=$pass FAIL=$fail TOTAL=$((pass+fail))"
```

### 4.4 视觉编码器的特殊阈值

视觉编码器内部使用 FP32 计算，但 ggml 的某些操作（如 softmax、grouped attention）在 zig 端可能与 C 参考有微小差异，因此阈值需适当放宽：

```bash
# 视觉编码器专用阈值
--tol-rel-max-err 5e-4     # 放宽相对最大误差
--tol-outlier-ratio 0.02   # 允许 2% 离群点
--tol-rmse 0.02            # 放宽 RMSE
```

> **注意**：这些宽松阈值**仅适用于中间层**。最终 `07_mm_output` 和端到端 logits 应使用严格标准。

### 4.5 生成新的参考数据

在 llama.cpp 端修改 `llama_mtmd_cli` 或调试 hook 添加 tensor dump 后，运行：

```bash
# llama.cpp 端生成参考 JSON
llama-mtmd-cli --dump-tensors debug_vision -m model.gguf --mmproj mmproj.gguf ...

# zllama.zig 端生成测试 JSON
zllama-compare-mtmd-vision --dump-tensors debug_vision -m model.gguf ...
```

---

## 5. 指标详解

### 5.1 归一化均方误差 (NMSE)

```
NMSE = Σ(a_i - b_i)² / (Σ a_i² + ε)
```

- **范围**: [0, ∞)，0 表示完全一致
- **优点**: 尺度归一化，不受向量幅值影响
- **适用**: 所有场景

### 5.2 余弦相似度 (Cosine Similarity)

```
cos = (a·b) / (‖a‖·‖b‖ + ε)
```

- **范围**: [-1, 1]，1 表示方向完全一致
- **优点**: 对幅值缩放不敏感
- **适用**: 高维嵌入（如 n_embd=4096）

### 5.3 均方根误差 (RMSE)

```
RMSE = √( Σ(a_i - b_i)² / n )
```

- **范围**: [0, ∞)
- **优点**: 直观的"每维度平均误差"
- **注意**: 对数据尺度敏感，大数值张量（如池化后）绝对 RMSE 会偏大

### 5.4 相对最大绝对误差 (RelMaxErr)

```
RelMaxErr = max(|a_i - b_i|) / max(|ref_i|, ε)
```

- **范围**: [0, ∞)
- **优点**: 尺度自适应，大数值和接近零的区域都能正确判断
- **适用**: 检测单个维度上的异常偏差（可能导致 argmax 翻转）

### 5.5 离群点占比 (Outlier Ratio)

```
outlier_ratio = count(abs(error_i) > 3×RMSE) / n
```

- **范围**: [0, 1]
- **用途**: 区分"整体轻微偏差"与"少数野点"
- **判决**: 若 RelMaxErr 超标但 outlier_ratio 极低，降级为警告而非失败

### 5.6 平均幅值比 (Avg Ratio)

```
avg_ratio = mean(a_i / b_i)  （排除零值维度）
```

- **范围**: 接近 1.0 表示幅值一致
- **用途**: 检测系统性缩放（如缺失 `scale` 因子或 `sqrt(d_k)` 因子）

---

## 6. 阈值配置

### 6.1 严格模式（文本 LLM + 最终 logits）

用于端到端对齐，FP32 确定性计算：

```
NMSE         < 1e-4
余弦相似度   > 0.9999
RMSE         < 0.001
相对 MaxErr  < 1e-4
离群点占比   < 0.001
Argmax       必须匹配
```

### 6.2 宽松模式（多模态中间层）

用于 ViT 内部层、池化、投影等中间张量：

```
NMSE         < 1e-4（不变）
余弦相似度   > 0.9999（不变）
RMSE         < 0.02  （放宽 20x）
相对 MaxErr  < 5e-4  （放宽 5x）
离群点占比   < 0.02  （放宽 20x）
Argmax       必须匹配
```

### 6.3 阈值选择决策树

```
是最终 logits 吗？
├─ 是 → 严格模式（所有默认值）
└─ 否 → 是池化/投影输出吗？
    ├─ 是 → 值域大（~1000+），用 --tol-rel-max-err 5e-4
    └─ 否 → 是 ViT 中间层吗？
        ├─ 是 → 宽松模式
        └─ 否 → 严格模式
```

---

## 7. 判决逻辑

`align_cmp_core.zig` 中的 `alignmentVerdict()` 按以下优先级判决：

```
1. 余弦相似度 < tol_cosine           → ❌ 对齐失败
2. NMSE  > tol_nmse                  → ❌ 对齐失败
3. RelMaxErr > tol_rel_max_err       → 检查离群点占比
   ├─ outlier_ratio > tol_outlier    → ❌ 对齐失败（多处偏差）
   └─ outlier_ratio ≤ tol_outlier    → ⚠️ 警告（个别野点）
4. RMSE > tol_rmse                   → ❌ 对齐失败
5. Argmax 不匹配                     → ❌ 对齐失败
6. 幅值比偏离 > tol_ratio_deviation  → ⚠️ 警告（可能系统性缩放）
7. 绝对 MaxErr 大但相对正常          → ✅ 通过（由数据尺度引起）
8. 以上全部通过                      → ✅ 对齐验证通过
```

**判决符号说明**：

| 符号 | 含义 |
|------|------|
| ✅ | 通过 |
| ⚠️ | 通过但存在可疑偏差，建议复核 |
| ❌ | 失败，必须修复 |

---

## 8. 调试错位层

### 8.1 二分定位法

当逐层对齐发现某一层开始出现偏差：

```
层 0-2  ✅ 通过
层 3    ❌ 失败（余弦 < 0.99）  ← 问题出现在这里
层 4-14 ❌ 失败（逐层累积）

→ 重点检查层 3 的实现
```

### 8.2 细粒度子步骤

对于出问题的层，需要进一步分解。例如 `04d_layer_3` 拆分为：

| 文件 | 含义 |
|------|------|
| `04d_layer_3_a_layer_inp_normed` | 层输入 Pre-Norm |
| `04d_layer_3_g_attn_out` | Attention 输出 |
| `04d_layer_3_k_attn_post_normed` | Attention Post-Norm |
| `04d_layer_3_n_ffn_post_normed` | FFN Post-Norm |
| `04d_layer_3_out` | 层最终输出（残差连接后） |

定位流程：

```
Pre-Norm  ✅ → Norm 实现正确
Attn Out  ❌ → 问题在 Attention 内部：
              ├─ Q/K/V 投影 ✓
              ├─ RoPE / PosEmb  ← 最可能的罪魁祸首
              ├─ softmax(QK^T/√d_k) ✓
              └─ 输出投影 ✓
```

### 8.3 常见根因速查表

| 子步骤 | 常见错误 | llama.cpp 参考位置 |
|--------|----------|-------------------|
| Pre-Norm | RMSNorm 的 epsilon 值 | `llama.cpp/ggml.cpp:ggml_norm` |
| RoPE | `freq_base` 读取错误，`theta` 计算 | `llama.cpp:llama_rope` |
| Attention Mask | 因果 mask 下三角填充 | `ggml_diag_mask_inf` |
| Q/K 投影 | `n_head` vs `n_kv_head` 维度混淆 | 模型 `build_llama` |
| Softmax | `scale` 因子（`1/√d_k`） | Attention 层 |
| FFN | SwiGLU gate 激活 | `ggml_silu` + `ggml_mul` |
| 残差连接 | 加法顺序错误 | 层末尾 |

### 8.4 排查清单

```
□ 权重 shape 与 GGUF 元数据一致？
□ n_head / n_kv_head / n_head_dim 与 llama.cpp 相同？
□ RoPE freq_base 值正确（GGUF 中 "rope.theta"）？
□ eps 值（RMSNorm / LayerNorm）与 GGUF 一致？
□ 残差连接顺序与参考一致？
□ 图构建中 scale 因子（attention / FFN）不缺不漏？
```

---

## 9. CI 自动化

### 9.1 快速冒烟测试

```bash
# 全量单元测试
zig build test -Doptimize=ReleaseSafe --summary all

# 仅对齐相关测试
zig build test -- --test-filter "align"
```

### 9.2 全量多模态对齐

```bash
# ReleaseFast 构建（生产级优化）
zig build -Doptimize=ReleaseFast

# 运行全量 32 项对齐（PASS/FAIL 统计）
./tools/run_align.sh
```

### 9.3 回归检查脚本

```bash
#!/bin/bash
# tools/run_align.sh — 多模态全量对齐回归检查
set -euo pipefail

BUILD_DIR="zig-out/bin"
REF_DIR="debug_vision"
TOL="--tol-rel-max-err 5e-4 --tol-outlier-ratio 0.02 --tol-rmse 0.02"

pass=0 fail=0

cd "$REF_DIR"
for ref in llama_*.json; do
    test="z${ref}"
    if [ ! -f "$test" ]; then continue; fi

    result=$("../$BUILD_DIR/zllama-align-cmp" \
        --ref "$ref" --test "$test" --output ai $TOL 2>&1)

    if echo "$result" | grep -q "PASS=1"; then
        ((pass++))
    else
        ((fail++))
        echo "FAIL: $ref"
        echo "$result" | grep -E "NMSE|COSINE|VERDICT"
    fi
done

echo "PASS=$pass FAIL=$fail TOTAL=$((pass+fail))"
[ $fail -eq 0 ] && echo "✅ 全量对齐通过" || (echo "❌ 对齐失败" && exit 1)
```

---

## 10. 扩展指南

### 10.1 添加新的调试阶段

在 llama.cpp 和 zllama.zig 两端的 encoder 中添加 tensor dump hook：

1. 在 llama.cpp 端修改 `llama_mtmd_cli`，在目标操作后调用 dump 函数
2. 在 zllama.zig 端使用 `DebugTensorRegistry` 注册同名 hook
3. 两端的文件命名保持一致：`{prefix}_vision_{stage}.json`

### 10.2 添加新的指标

1. 在 `src/tools/metrics.zig` 中添加计算函数（如 `calcKLDivergence`）
2. 在 `src/tools/align_cmp_config.zig` 的 `AlignMetrics` 中添加字段
3. 在 `src/tools/align_cmp_core.zig` 的 `computeMetrics()` 中调用新函数
4. 在 `alignmentVerdict()` 中添加对应的判决条件
5. 在 `src/tools/align_cmp_help.txt` 中添加对应的 `--tol-*` 参数

### 10.3 添加音频对齐

音频编码器的对齐流水线与视觉类似：

1. 将参考/测试 tensor dump 放入 `debug_audio/`
2. 使用相同的 `zllama-align-cmp` 工具
3. 音频中间层阈值可比视觉更宽松（音频特征维度更高）

---

## 附录 A: 工具快速参考

| 命令 | 用途 |
|------|------|
| `zllama-align-cmp --ref a.json --test b.json` | 比较两个向量文件 |
| `zllama-compare-llamacpp --model m.gguf --prompt "Hi" --ref-logits r.bin` | 端到端 logits 对齐 |
| `zllama-compare-logits --ref r.bin --test t.bin` | 已保存 logits 对比 |
| `zllama-gen-ref --model m.gguf -o r.bin` | 生成金标准 logits |
| `zllama-compare-mtmd-vision ...` | 多模态视觉完整验证 |
| `zllama-compare-mtmd-audio ...` | 多模态音频完整验证 |
| `zig build test -- --test-filter "align"` | 运行对齐相关单元测试 |
| `./tools/run_align.sh` | 全量多模态对齐回归 |

## 附录 B: 文件依赖图

```
src/tools/
├── metrics.zig                ← 共享指标计算（NMSE、余弦、Argmax）
├── align_cmp_config.zig       ← 配置、阈值、AlignMetrics 结构体
├── align_cmp_core.zig         ← computeMetrics() + alignmentVerdict()
├── align_cmp.zig              ← CLI 入口（参数解析 + JSON 加载 + 输出）
├── align_cmp_help.txt         ← 帮助文本
├── compare_logits.zig         ← 独立 logits 比较器
├── compare_with_llamacpp.zig  ← 端到端 llama.cpp 集成对比
├── compare_mtmd_vision.zig    ← 多模态视觉验证
├── compare_mtmd_audio.zig     ← 多模态音频验证
├── generate_reference.zig     ← 金标准生成器
├── dump_graph.zig             ← 计算图导出
└── dump_tensors.zig           ← 张量导出
```

---

> **相关文档**: `AGENTS.md`（AI 编程入口）、`docs/TEST.md`（测试体系）、`docs/TOOLS.md`（工具说明）
