# Gemma 4 文本推理修复 & 语音识别

## 已修复 Bug

### 1. RoPE mode: NORMAL→NEOX
- `ggml.ropeExt(ctx, ..., 0, 0, ...)` → `ggml.ropeExt(ctx, ..., 2, 0, ...)`
- GGML_ROPE_TYPE_NEOX=2, GPT-NeoX 风格: [cos...cos, sin...sin]

### 2. rope_freqs 全局共享
- 原来只在 layer 0 加载，其他全注意力层用 null
- 修复：加载一次 `rope_freqs.weight`，所有 `!is_swa_layer` 层共享

### 3. attention scale_factor: 1.0 → 1/sqrt(head_dim_k)
- 原来 scale_factor=1.0，导致 attention scores 过大，softmax 饱和 → 随机输出
- 修复：与 Qwen/LLaMA 一致，使用 1/sqrt(head_dim_k)

### 4. KV cache head_dim: 256 → max(512, 256)
- 原来 KV cache 用 `n_head_dim_k=256`（SWA 维度）
- 全注意力层 K/V head_dim=512，view3d 创建 view 时 strides 错乱
- 修复: main.zig & simple_main.zig 中用 `@max(n_head_dim, n_head_dim_k)`

### 5. Tokenizer: gemma4 → GPT-2 预分词
- 原来 gemma4 fallthrough 到 default（整段文本当一个 token）
- 修复: tokenizer/encode.zig 添加 `.gemma4` 到 GPT2 预分词 case

### 6. 注意力 logit softcapping
- 在 attention.zig 添加了 `attn_logit_softcap` 参数（默认 0.0=禁用）
- Gemma 4 中暂时未启用（llama.cpp 也没有用）

## 已验证的模型参数（q_norm/k_norm 维度调试）

| 层 | Q norm | K norm | 备注 |
|----|--------|--------|------|
| 0-3 | 256 | 256 | SWA |
| 4 | 512 | 512 | 全注意力 |
| 5-8 | 256 | 256 | SWA |
| 9 | 512 | 512 | 全注意力 |
| 10-13 | 256 | 256 | SWA |
| 14 | 512 | 512 | 全注意力 (最后一个有 KV 的层) |
| 15-18 | 256 | 0 | 共享 KV |
| 19 | 512 | 0 | 共享 KV |
| 20-23 | 256 | 0 | 共享 KV |
| 24 | 512 | 0 | 共享 KV |
| 25-28 | 256 | 0 | 共享 KV |
| 29 | 512 | 0 | 共享 KV |
| 30-33 | 256 | 0 | 共享 KV |
| 34 | 512 | 0 | 共享 KV |

结论: Q 和 K head_dim 始终相同（256 或 512），之前认为 Q head_dim ≠ K head_dim 的假设不成立。

## 当前状态：仍然输出乱码 ❌

```
$ ./zig-out/bin/zllama --model gemma-4-E2B-it-Q4_K_M.gguf -n 15 -p "The capital of France is"
The capital of France is RChangedEventArgs Dora S working ute ute puterea tầm登 Argon Scarborough rock SRC tầm
```

## 待调查

- llama.cpp 中 Gemma 4 使用 `n_embd_head = hparams.n_embd_head_k(il)` 同时作为 Q 和 K 的 head_dim
- 我们的实现已与此一致（head_dim = q_norm.ne[0] = k_norm.ne[0]）
- 需要对比 logits 与参考实现定位剩余差异
- 可能差异点：attention 内部实现细节（permute 顺序、GQA 广播等）
