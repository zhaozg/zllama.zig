# 嵌入式模型向量生成开发指南

> **项目：** zllama.zig — Embedding（嵌入向量）生成功能
> **测试模型：** `~/.cache/models/Qwen3-Embedding-0.6B-Q8_0.gguf`
> **参考：** llama.cpp `llama-embedding` 示例、`examples/embedding/embedding.cpp`

---

## 1. 背景与概念

### 1.1 生成式模型 vs 嵌入模型

| 维度 | 生成式模型 (Generative) | 嵌入模型 (Embedding) |
|------|------------------------|---------------------|
| 目标 | 逐 token 生成文本序列 | 输出固定维度稠密向量 |
| 计算图 | 因果注意力（causal mask） | 双向注意力（无 causal mask） |
| KV Cache | 需要，支持增量解码 | **不需要**（单次前向） |
| 输出 | [n_vocab, n_tokens] logits | [n_embd, n_tokens] → pooling → [n_embd] |
| 典型架构 | Qwen2.5, LLaMA 3, Gemma | Qwen3-Embedding, BGE, GTE |
| GGUF 特征 | 有 `output.weight` / `output_norm.weight` | 只到 `output_norm.weight`，无 `output.weight` |

### 1.2 Qwen3-Embedding 架构

Qwen3-Embedding 基于 **Qwen2 架构**，GGUF 元数据中 `general.architecture` = `"qwen2"`。

关键区别：
- **无 `output.weight`**（不需要 lm_head）
- **双向注意力**（去掉因果 mask）
- **输出处理**：hidden states → 池化（pooling）→ L2 归一化
- 分词器、嵌入表、layer 结构完全一致

---

## 2. 设计决策

### 2.1 架构检测

通过 `output_norm.weight` 存在但 `output.weight` 缺失来检测嵌入模型。

### 2.2 双向注意力

跳过 `ggml.diagMaskInf()`，利用 ggml `mulMat` 的隐式 GQA 广播（ne[2] 维度不匹配时自动广播）实现双向注意力。

### 2.3 池化策略

| 策略 | 说明 |
|------|------|
| `mean` | 所有 token 取均值（默认） |
| `cls` | 取第一个 token（BOS/CLS） |
| `last` | 取最后一个 token |

### 2.4 L2 归一化

输出向量 L2 归一化，直接使用 `ggml.l2Norm(ctx, a, eps)`。

---

## 3. 崩溃修复记录

### 根因：`n_head_dim` 计算错误

`parseParams()` 中 `n_head_dim = n_embd / n_head = 1024/16 = 64`，但 Qwen3-Embedding 的 Q 权重实际形状 `[2048, 1024]`，真实 head_dim = `key_length = 128`。`reshape3d([64,16,n])` 与 Q 张量 nelements=2048 不匹配。

### 修复（2026-06-10）

**`src/models/qwen2.zig`** — 两处关键修改：

1. **qwen3.*/qwen2.* 参数回退读取**：优先读 `qwen3.attention.head_count`、`qwen3.embedding_length` 等，失败时回退到 `qwen2.*` 和 `llama.*` 前缀。

2. **n_head_dim 覆盖**：当 `n_head_dim_k`（从 `key_length` 读取）> 0 且 ≠ `n_embd/n_head` 时，用 `n_head_dim_k` 覆盖 `n_head_dim`：
```zig
if (p.base.n_head_dim_k > 0 and p.base.n_head_dim_k != p.base.n_head_dim) {
    p.base.n_head_dim = p.base.n_head_dim_k;
}
```

**`src/models/embedding.zig`** — 使用 ggml `mulMat` 隐式 GQA 广播实现双向注意力（无 repeat_kv）。

### 内存泄漏修复

详见 `MEM.md`：
1. `EmbeddingModel.deinitAdapter` 缺少 `allocator.destroy(self)`
2. `generateEmbedding()` 返回的 `[]f32` 向量在 main 中未释放

---

## 4. 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `src/layers/pooling.zig` | 池化 + L2 归一化层 | ✅ 完成 |
| `src/models/embedding.zig` | 嵌入模型实现（双向注意力 + 内存修复） | ✅ 完成 |
| `src/models/qwen2.zig` | qwen3.*/qwen2.* 回退 + n_head_dim 覆盖 | ✅ 完成 |
| `src/model.zig` | `embedding_qwen2` 枚举 + qwen3 别名 | ✅ 完成 |
| `src/models/registry.zig` | 嵌入模型检测 + 工厂 | ✅ 完成 |
| `src/main.zig` | `--embed` CLI + generateEmbedding() + 内存修复 | ✅ 完成 |
| `src/tests/test_embed.zig` | 嵌入模型测试 | ✅ 完成 |
| `MEM.md` | 内存泄漏追踪 | ✅ 完成 |

### 已知限制

- **Qwen3-Embedding 输出精度**：当前利用 ggml `mulMat` 的隐式 GQA 广播实现双向注意力，语义上与参考实现的显式 repeat 等价。但由于未应用 Q/K normalization，输出与 llama.cpp 参考有偏差。完整的 Qwen3-Embedding 支持标记为 TODO。

---

## 5. 验证清单

```bash
# 编译
zig build -Doptimize=ReleaseFast

# 嵌入向量生成
./zig-out/bin/zllama --embed --model Qwen3-Embedding-0.6B-Q8_0.gguf -p "hello world"

# 无 DebugAllocator 泄漏
zig build  # Debug 模式自动检测泄漏
./zig-out/bin/zllama --embed --model Qwen3-Embedding-0.6B-Q8_0.gguf -p "h"

# 单元测试
zig build test
```
