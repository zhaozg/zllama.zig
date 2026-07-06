# 量化能力设计

> 本文档描述 `zllama.zig` 的量化支持现状、架构设计及未来路线图。

---

## 📊 当前状态（2025-07）

| 能力 | 状态 | 说明 |
|------|------|------|
| **量化模型推理** | ✅ **已完成** | 通过 ggml C 库直接支持所有量化类型推理 |
| **量化类型定义** | ✅ **已完成** | `src/ggml/c.zig` 中 `Type` 枚举覆盖全部 30+ 量化类型 |
| **反量化内核** | ✅ **已完成** | ggml 的 `quants.c` / `ggml-quants.c` 编译进静态库 |
| **混合精度加载** | ✅ **已完成** | `weight_loader.zig` 按 GGUF 元数据读取各张量类型 |
| **GGUF 写入 API** | ✅ **已完成** | `src/ggml/gguf.zig` 有 `setVal*`、`addTensor`、`writeToFile` |
| **量化 API 绑定** | ❌ **缺失** | `ggml_quantize_chunk` 等 C API 未在 Zig 绑定层封装 |
| **量化工具** | ❌ **缺失** | 无 `llama-quantize` 等价工具 |
| **重要性矩阵工具** | ❌ **缺失** | 无 `llama-imatrix` 等价工具 |
| **校准支持** | ❌ **缺失** | 无校准数据集处理流程 |

---

## 🧱 架构说明：为什么量化推理"开箱即用"

`zllama.zig` 的量化支持策略与 `llama.cpp` 不同：

### 传统方案（llama.cpp）

```
量化权重 → Zig 重新实现反量化内核 → Zig 实现 GEMM
```

需要在 Zig 中完整移植 `quants.c` 的所有量化/反量化函数。

### zllama.zig 方案（直接服用 ggml）

```
量化权重 → ggml C 库（quants.c 反量化内核）→ ggml_mul_mat（含量化感知计算）
```

本项目**直接编译 ggml C 源码**为静态库（`buildGgmlFromSource`），所有量化类型的：
- **类型定义**：`ggml_type` 枚举（`src/ggml/c.zig` 中 `Type` 枚举已完整映射）
- **行大小计算**：`ggml_type_size()` / `ggml_blck_size()` / `ggml_row_size()`
- **反量化内核**：`ggml-quants.c` 中的 `dequantize_row_q4_0()` 等
- **量化感知计算**：`ggml_mul_mat` 内部自动处理量化输入的实时反量化

全部通过 C 编译直接可用，**无需在 Zig 中重新实现任何量化内核**。

### 关键绑定层

`src/ggml/c.zig` 中 `Type` 枚举已覆盖所有 ggml 量化类型：

```zig
pub const Type = enum(c.ggml_type) {
    f32, f16, bf16,
    q4_0, q4_1, q5_0, q5_1, q8_0, q8_1,
    q2_K, q3_K, q4_K, q5_K, q6_K, q8_K,
    iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s,
    iq3_xxs, iq3_s, iq4_nl, iq4_xs,
    tq1_0, tq2_0,
    mxfp4, nvfp4,
    // ...
};
```

每个类型提供 `sizeOf()`、`blockSize()`、`rowSize()`、`isQuantized()`、`name()` 方法。

---

## 🎯 未来目标

### 阶段一：量化 API 绑定（短期）

在 `src/ggml/` 中封装 ggml 提供的量化 C API，当前**完全缺失**：

| C API | Zig 绑定状态 | 用途 |
|-------|-------------|------|
| `ggml_quantize_chunk` | ❌ 未绑定 | 将 f32 数据量化为指定类型 |
| `ggml_quantize_init` | ❌ 未绑定 | 初始化量化查找表（IQx 系列需要） |
| `ggml_quantize_free` | ❌ 未绑定 | 释放量化查找表 |
| `ggml_quantize_requires_imatrix` | ❌ 未绑定 | 检查量化类型是否需要重要性矩阵 |

**实现位置**：`src/ggml/ops.zig` 或新建 `src/ggml/quantize.zig`

```zig
// 预期 API 设计
pub fn quantizeChunk(typ: Type, src: []const f32, dst: []u8, start: i64, nrows: i64, n_per_row: i64, imatrix: ?[]const f32) usize;
pub fn quantizeInit(typ: Type) void;
pub fn quantizeFree() void;
pub fn quantizeRequiresImatrix(typ: Type) bool;
```

### 阶段二：量化工具（中期）

实现类似 `llama-quantize` 的命令行工具，将 FP16/BF16/FP32 模型转换为各种 GGUF 量化格式。

**设计要点**：
- 读取源 GGUF 文件，解析所有张量的元数据
- 对每个张量，读取 f32 数据 → 调用 `ggml_quantize_chunk` → 写入新 GGUF 文件
- 支持混合精度配方（如 Q4_K_M：关键层 Q6_K，其余 Q4_K）
- 输出 GGUF v3 格式

**实现位置**：`src/tools/quantize.zig`

**使用方式**：
```bash
zig-out/bin/zllama-quantize --model model.gguf --type q4_K_M --output model-Q4_K_M.gguf
```

### 阶段三：重要性矩阵工具（中期）

实现类似 `llama-imatrix` 的工具，通过校准数据集生成重要性矩阵，用于高质量量化。

**设计要点**：
- 加载模型，在校准数据上执行前向传播
- 收集每个权重的梯度/重要性信息
- 输出重要性矩阵文件（与 GGUF 格式兼容）
- IQx 系列量化（IQ2_XXS、IQ2_XS、IQ1_S）需要 imatrix

**实现位置**：`src/tools/imatrix.zig`

### 阶段四：校准支持（长期）

支持校准数据集处理流程，实现更高质量的量化：

- 支持多种校准数据格式（文本、JSONL）
- 自动选择代表性校准样本
- 与重要性矩阵工具集成

---

## 📋 任务清单

### P0 — 量化 API 绑定（预计 1-2 天）

- [ ] 在 `src/ggml/` 中新建 `quantize.zig`，封装 `ggml_quantize_chunk`、`ggml_quantize_init`、`ggml_quantize_free`、`ggml_quantize_requires_imatrix`
- [ ] 在 `src/ggml/mod.zig` 中重新导出新模块
- [ ] 编写单元测试，验证各量化类型的 `quantize_chunk` 输出大小与 `ggml_row_size` 一致
- [ ] 验证 `ggml_quantize_init`/`free` 的线程安全性

### P1 — 量化工具（预计 3-5 天）

- [ ] 实现 `src/tools/quantize.zig`：读取 GGUF → 逐张量量化 → 写入新 GGUF
- [ ] 支持 `--type` 参数选择量化类型
- [ ] 支持 `--recipe` 参数选择混合精度配方（Q4_K_M、Q3_K_M 等）
- [ ] 支持 `--output` 指定输出路径
- [ ] 在 `build.zig` 中注册为独立可执行文件
- [ ] 端到端测试：FP16 模型 → 量化 → 推理验证 logits 一致性

### P2 — 重要性矩阵工具（预计 3-5 天）

- [ ] 实现 `src/tools/imatrix.zig`：加载模型 + 校准数据 → 生成重要性矩阵
- [ ] 支持 `--model`、`--data`（校准集路径）、`--output` 参数
- [ ] 支持 IQ2_XXS、IQ2_XS、IQ1_S 等需要 imatrix 的量化类型
- [ ] 与量化工具集成：`--imatrix` 参数传入重要性矩阵

### P3 — 校准与优化（长期）

- [ ] 支持多种校准数据格式（JSONL、纯文本）
- [ ] 自动样本选择策略
- [ ] 量化质量评估工具（perplexity 对比）
- [ ] 支持 `ggml_quantize_chunk` 的 `start`/`nrows` 分块处理大张量

---

## 🔧 技术参考

### ggml 量化 API 签名

```c
// 初始化量化查找表（IQx 系列需要，线程安全）
void ggml_quantize_init(enum ggml_type type);

// 释放量化查找表
void ggml_quantize_free(void);

// 检查量化类型是否需要重要性矩阵
bool ggml_quantize_requires_imatrix(enum ggml_type type);

// 执行量化：将 src（f32）量化为 dst（目标类型）
// start: 起始元素偏移（必须是 blck_size 和 n_per_row 的倍数）
// nrows: 行数
// n_per_row: 每行元素数
// imatrix: 重要性矩阵（可为 NULL，但 IQ2_XXS/IQ2_XS/IQ1_S 必须提供）
// 返回: 量化后数据字节数
size_t ggml_quantize_chunk(
    enum ggml_type   type,
       const float * src,
              void * dst,
           int64_t   start,
           int64_t   nrows,
           int64_t   n_per_row,
       const float * imatrix);
```

### 混合精度配方参考（llama.cpp 预设）

| 预设名 | 主要类型 | 关键张量例外 |
|--------|---------|-------------|
| `q4_0` | Q4_0 | 无 |
| `q4_1` | Q4_1 | 无 |
| `q4_K_M` | Q4_K | attention.wv、ffn.w2 → Q6_K |
| `q4_K_S` | Q4_K | 无 |
| `q3_K_M` | Q3_K | attention.wv、ffn.w2 → Q6_K |
| `q5_K_M` | Q5_K | attention.wv、ffn.w2 → Q6_K |
| `q6_K` | Q6_K | 无 |
| `q8_0` | Q8_0 | 无 |

### 需要重要性矩阵的量化类型

| 类型 | 需要 imatrix |
|------|-------------|
| IQ2_XXS | ✅ 是 |
| IQ2_XS | ✅ 是 |
| IQ1_S | ✅ 是 |
| IQ1_M | ❌ 否（当前实现） |
| IQ3_XXS | ❌ 否 |
| IQ3_S | ❌ 否 |
| IQ2_S | ❌ 否 |
| IQ4_NL | ❌ 否 |
| IQ4_XS | ❌ 否 |

---

## 💎 总结

`zllama.zig` 的量化推理能力已通过**直接服用 ggml C 库**完整实现——所有量化类型的加载、反量化、计算均由 ggml 的 `quants.c` / `ggml-quants.c` 处理，Zig 层无需重新实现任何量化内核。

当前缺失的是**量化工具链**（将 FP16 模型转换为量化 GGUF 的工具）和**底层量化 API 的 Zig 绑定**。这些是纯工具性缺失，不影响已量化模型的推理能力。

路线图优先级：
1. **P0**：封装 `ggml_quantize_chunk` 等 C API 到 Zig 绑定层
2. **P1**：实现 `zllama-quantize` 量化工具
3. **P2**：实现 `zllama-imatrix` 重要性矩阵工具
4. **P3**：校准支持与质量评估
