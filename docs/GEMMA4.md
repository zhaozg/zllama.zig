基于目前的 `zllama.zig` 架构（已支持 Llama 和 Qwen35, Gemma3），新增的 Gemma4 支持不完善。

Gemma 4 推理乱码，而 Llama 3、Gemma 3、Qwen3.5 正常，说明问题并非通用的注意力或图构建逻辑，而是 Gemma 4 **特有架构** 的实现偏差。下面我直接对比关键差异，帮你定位剩余缺陷。
---

---

## 🔴 核心问题一：前馈网络维度 `n_ff` 为 0（致命）

日志中明确显示：
```
info(gemma4): Gemma4: vocab=262144, embd=1536, heads=8, kv_heads=1, layers=35, ff=0, ...
```
`ff=0` 意味着 `parseParams` 未能从 GGUF 文件中读取到 `feed_forward_length`。这导致后续权重加载时，`ffn_gate_weight`、`ffn_up_weight`、`ffn_down_weight` 的形状完全错误（可能为空或维度不匹配），进而使 FFN 层输出随机值，污染整个残差流。

### 验证方法
检查 GGUF 文件中是否存在以下任意一个键：
- `gemma4.feed_forward_length`
- `llama.feed_forward_length`

若都不存在，则需要**根据 `n_embd` 和模型类型推导**。对于 Gemma 4，通常 `n_ff = 4 * n_embd`（即 4×1536 = 6144）。你可以手动计算并赋值给 `p.base.n_ff`。

### 临时修复
在 `parseParams` 中，若 `n_ff` 仍为 0，则设置：
```zig
if (p.base.n_ff == 0) {
    p.base.n_ff = 4 * p.base.n_embd;
    log.warn("n_ff missing, using default 4 * n_embd = {d}", .{p.base.n_ff});
}
```

---

## 🟠 核心问题二：共享 KV 层维度不匹配（严重）

参数显示：
- `shared_kv=20` → 后 20 层共享 KV
- `n_layer_kv_from_start = 35 - 20 = 15`，即前 15 层有各自 KV，后 20 层复用。
- `swa=512`，但之前提到 SWA 层的 `head_dim` 应为 256，全局层为 512。这里 `swa=512` 可能是滑动窗口大小，而非 head_dim。

日志中没有打印每层的 `head_dim`，但从代码逻辑可知：
- SWA 层的 `attn_q_norm_weight.ne()[0]` = 256
- 全局层的 `attn_q_norm_weight.ne()[0]` = 512

当共享层（例如第 20 层，可能是全局层）复用前面被共享层（例如第 14 层，可能是 SWA 层）的 KV 时：
- 被共享层的 K/V `head_dim_k_cache` = 256
- 当前层的 Q `head_dim` = 512
- 你的代码会执行 `n_head_eff = (n_head * head_dim) / head_dim_k_cache`，将 Q reshape 为 `[head_dim_k_cache, n_head_eff, n_tokens]`，然后注意力计算，输出后再 reshape 回 `[n_head * head_dim, n_tokens]`。

**此过程可能损失信息**，因为注意力输出是基于 256 维的 K/V 空间计算的，却要映射回 512 维。正确的做法是：共享 KV 的层应该**强制与被共享层具有相同的 head_dim**，或者模型设计保证共享层与被共享层的 head_dim 一致。检查你的 `is_swa_layer` 数组，确认被共享层和共享层的类型是否匹配。

### 验证方法
在 `transformerForward` 中，对每一层打印：
```zig
log.debug("Layer {d}: is_swa={}, has_kv={}, head_dim={}, head_dim_k={}",
    .{i, layer_is_swa, layer.has_kv, head_dim, if (layer.has_kv) head_dim_k else 0});
```
对于共享层，还要打印实际从 cache 中读到的 `head_dim_k_cache`。

---

## 🟡 其他可疑点

### 1. RoPE 模式正确但缺少 `rope_freqs`
日志中没有 `rope_freqs` 加载的信息，说明 `blk.0.rope_freqs.weight` 和 `rope_freqs.weight` 都不存在。因此全注意力层退化到普通 RoPE。这**不是乱码的直接原因**（因为许多模型只用普通 RoPE），但可能影响长文本质量。

### 2. Softcapping 后 logits 仍有正数
日志中多次出现正 logit（例如 `best_val=18.527977`），而 `final_logit_softcapping=30`。Softcapping 会限制范围在 `[-30,30]`，正数是正常的。问题不在于绝对值，而在于**概率分布过于平坦**（logits 相差不大），导致采样到的 token 无意义。

### 3. Tokenizer 词汇表巨大（262144）
这通常是 Gemma 的 SentencePiece 词表，包含多语言 token。输出乱码中包含孟加拉语、阿拉伯语等，说明模型未能聚焦于英文 prompt “hello”，而是随机生成了各种语言的 token。

---

## 🛠️ 立即行动方案

1. **修复 `n_ff`**（最紧急）：添加 fallback 计算，重新编译测试。
2. **检查每层 head_dim**：运行带 `-v` 的简单 prompt，观察层维度输出。如果共享层 head_dim 不匹配，需要确认模型设计是否允许跨 head_dim 共享（通常不允许）。若不允许，应调整 `has_kv` 分配，使共享层与被共享层 head_dim 相同。
3. **临时禁用共享 KV**：在 `gemma4.zig` 中强制所有层 `has_kv = true`，`n_layer_kv_from_start = n_layer`。这会让模型退化为无共享的普通 Transformer，如果乱码消失，则问题确认为共享层维度不匹配。
4. **验证 FFN 权重形状**：在 `loadWeights` 中添加断言，确保 `ffn_gate_weight`、`ffn_up_weight`、`ffn_down_weight` 的 `ne[0]` 等于 `n_ff`，`ne[1]` 等于 `n_embd`。

完成上述修复后，重新运行 `zllama-simple -p "hello"`，输出应该变为合理的英文单词（例如 “hello” 或 “Hello”）。如果仍有问题，请提供新的日志，特别是每层维度打印和 `n_ff` 修复后的输出。
