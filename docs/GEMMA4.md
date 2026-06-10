基于目前的 `zllama.zig` 架构（已支持 Llama 和 Qwen35, Gemma3），新增的 Gemma4 支持不完善。

Gemma 4 推理乱码，而 Llama 3、Gemma 3、Qwen3.5 正常，说明问题并非通用的注意力或图构建逻辑，而是 Gemma 4 **特有架构** 的实现偏差。下面我直接对比关键差异，帮你定位剩余缺陷。
---

## 0. 任务与参考

实现对 gemma 4 模型的支持,

模型路径:

- ~/.cache/models/gemma-4-12b-it-Q4_K_M.gguf

参考 [llama.cpp](deps/llama.cpp) 目录下的

- src/models/gemma4.cpp

## 一、Gemma 4 与正常模型的架构差异

| 特性 | Llama 3 | Gemma 3 | Qwen3.5 | **Gemma 4** (E2B) |
|------|---------|---------|---------|-------------------|
| 注意力模式 | 全部全局因果 | 全部全局因果 | 全部全局因果 | **混合**：交替 SWA（滑动窗口 4096）和全局注意力 |
| 共享 KV 层 | 无 | 无 | 无 | **有**（后 1/3 层不计算 K/V，复用前面层） |
| GQA 比率 | 1:8（常见）| 1:1 | 1:1 或 1:8 | **1:1**（所有头 KV 头数 = Q 头数） |
| Q/K 头维度 | 一致 | 一致 | 一致 | **分层变化**：SWA 层 256，全局层 512 |
| 位置编码 | RoPE (NEOX) | RoPE | RoPE | **RoPE + 每层独立的 `rope_freqs`（全局层用）** |
| 前 SWA 层是否有特殊 qk_norm？ | 否 | 否 | 否 | **有** `q_norm` / `k_norm`，且 SWA 层头维度为 256 |
| 注意力 logit 软上限 | 无 | 无 | 无 | llama.cpp 中默认启用 `attn_logit_softcap = 50.0` |

**致命点**：即便你修复了 RoPE mode、freqs 加载和 scale_factor，若 **SWA 掩码** 和 **共享 KV 的引用逻辑** 未正确实现，全局层会读到全零或错乱的历史，导致输出完全无意义。

---

## 🔴 关键缺陷与修正建议

### 1. SWA 掩码很可能未正确实现
**现象**：`attention.scaledDotProductAttention` 虽然接收 `window_size` 参数，但需要确认其内部是否真的构造了滑动窗口掩码（只允许当前 token 向前 `window_size` 个 token 的注意力）。如果仅仅是因果掩码，模型会关注到过远的 token，破坏 SWA 层的局部性。

**修正**：检查 `attention.zig` 中的 `scaledDotProductAttention` 函数。
```zig
// 伪代码：当 window_size != null 时，应生成形如
// mask[i,j] = 0 if j <= i and (i - j) < window_size else -inf
```
如果未实现，需要补全。

### 2. 共享 KV 的层索引计算错误
当前 `findKVLayer` 函数：
```zig
fn findKVLayer(p: *const Gemma4Params, layer_idx: usize) usize {
    if (p.n_layer_kv_from_start > 1 and layer_idx >= p.n_layer_kv_from_start) {
        const is_swa = p.is_swa_layer.items[layer_idx];
        if (is_swa) return p.n_layer_kv_from_start - 2;
        else return p.n_layer_kv_from_start - 1;
    }
    return 0;
}
```
**问题**：
- 当 `n_layer_kv_from_start = 1` 时，`-2` 会下溢（usize 下溢成巨大值）。
- 总是回退到第 0 层不合理，应该复用**最近的非共享层**。
- llama.cpp 的实际逻辑是：共享层复用 `n_layer_kv_from_start - 1` 层（对全注意力层）或 `n_layer_kv_from_start - 2`（对 SWA 层），但前提是这些索引有效。

**修正**：参考 llama.cpp `gemma4.cpp` 中的 `llama_build_gemma4` 和 `reuse` 函数：
```zig
fn findKVLayer(p: *const Gemma4Params, layer_idx: usize) usize {
    if (layer_idx >= p.n_layer_kv_from_start) {
        const is_swa = p.is_swa_layer.items[layer_idx];
        // 确保减后索引不越界
        if (is_swa and p.n_layer_kv_from_start >= 2) {
            return p.n_layer_kv_from_start - 2;
        } else if (p.n_layer_kv_from_start >= 1) {
            return p.n_layer_kv_from_start - 1;
        }
    }
    // 如果层本身有 KV，返回自身
    return layer_idx;
}
```
然后在 `forward` 中，对于非 KV 层应使用 `kv_layer_idx` 从 cache 中读取 K/V。

### 3. Proportional RoPE 可能未生效
代码中全注意力层使用 `global_rope_freqs` 张量，但 `ggml.ropeExt` 是否正确处理了该张量？如果 `ropeExt` 只是将 `freqs` 作为普通输入（如加在位置编码上），而不是按 proportional RoPE 公式进行插值，则位置编码完全错误。

**验证**：检查 `rope.zig` 中 `ropeExt` 的实现。若未实现 `freqs` 的语义，需按 llama.cpp 的 `ggml_rope_ext` 实现：
```c
// 伪逻辑：如果 freqs 不为空，则使用 freqs 中的频率值代替预计算的基频
// 具体参考 llama.cpp 中 gemma4 的 rope_freqs 使用方式
```

### 4. 注意力 logit softcap 未实际应用
`p.attn_logit_softcapping` 默认为 50.0，但需要确认 `scaledDotProductAttention` 内部是否在 softmax 前对 logits 做了 `tanh` 缩放：
```python
logits = logits / cap
logits = tanh(logits)
logits = logits * cap
```
若未实现，会导致注意力分布过于极端，影响生成质量。

### 5. 内存分配方式可能有隐患
`ctx_weights` 使用 `ggml.Context.initNoAlloc(mem_size_estimate)` 但未提供实际的内存缓冲区。查看 `ggml` 的 API，`initNoAlloc` 通常需要一个外部预分配的缓冲区。当前写法可能导致后续 `newTensor` 时在未初始化的内存上操作，引发随机崩溃或数据损坏。

**修正**：正确做法是先分配内存，再传入：
```zig
const mem_buf = try allocator.alignedAlloc(u8, 64, mem_size_estimate);
var ctx = try ggml.Context.init(mem_buf, mem_size_estimate);
```
或使用 `init` 自动分配。

### 6. 缺少对 per‑layer embedding 的支持
虽然文档标记为“暂不支持”，但若模型确实使用了 per‑layer token embedding（例如多模态场景），缺失会导致输入嵌入错误。检查 GGUF 中是否存在 `token_embd_per_layer.weight` 等张量，若有则必须实现。

## 🛠️ 建议的调试步骤

1. **临时禁用 SWA 和共享 KV**：在 `forward` 中强制所有层 `layer_is_swa = false`，并让所有层 `has_kv = true`。如果乱码消失，则问题出在 SWA 或共享逻辑。
2. **打印关键张量**：在注意力计算前打印 Q 和 K 的前几个值，对比 llama.cpp 的输出。
3. **验证 rope_freqs**：确认 `global_rope_freqs` 是否成功加载（不为 null），并打印其数值。
4. **修复内存分配**：按正确方式初始化 `ctx_weights`，避免内存损坏。

## 📌 总结

你的 `gemma4.zig` 框架是完整的，但几个关键细节偏离了 llama.cpp 的实现，尤其是 **SWA 掩码** 和 **共享 KV 索引**。优先修复这两点，乱码问题大概率会解决。如果你能提供 `attention.zig` 和 `rope.zig` 中相关函数的实现，我可以给出更精确的修正代码。

