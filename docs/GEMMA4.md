基于目前的 `zllama.zig` 架构（已支持 Llama 和 Qwen35, Gemma3），新增的 Gemma4 支持不完善。

---

## 0. 任务与参考

实现对 gemma 4 模型的支持,

模型路径:

- ~/.cache/models/gemma-4-12b-it-Q4_K_M.gguf

参考 [llama.cpp](deps/llama.cpp) 目录下的

- src/models/gemma4.cpp

## 1. ✅ 已修复：RoPE 崩溃问题（2026-06-07）

### 崩溃原因

Gemma 4 使用**混合注意力架构**：部分层是全注意力（full attention），部分层是滑动窗口注意力（SWA）。
这两类层的 `head_dim` 不同：

- **全注意力层**：使用 `key_length`（较大的维度，如 256 或 512）
- **SWA 层**：使用 `key_length_swa`（较小的维度，如 128 或 256）

但代码中 `rope_dim` 是**全局单一值**（从 `gemma4.rope.dimension_count` 读取，对应全注意力层的维度）。
当 SWA 层的 `head_dim` < `rope_dim` 时，`ggml_ropeExt` 会断言 `n_dims <= ne0` 失败并崩溃。

### 修复内容

**文件：`src/models/gemma4.zig`**

1. **`Gemma4Params` 结构体**：新增 `rope_dim_swa: u32 = 0` 字段，存储 SWA 专用的 RoPE 维度

2. **`parseParams` 函数**：读取 GGUF 元数据键 `gemma4.rope.dimension_count_swa`（对应 llama.cpp 的 `LLM_KV_ROPE_DIMENSION_COUNT_SWA`），默认回退到 `base.rope_dim`

3. **`forward` 函数**：根据 `is_swa_layer[i]` 标志动态选择 RoPE 维度：
   - SWA 层：使用 `p.rope_dim_swa`
   - 全注意力层：使用 `p.base.rope_dim`
   - 安全夹持：`@min(rope_dim, head_dim)` 防止超限

4. **图节点容量**：`ggml_new_graph` 默认支持 2048 个节点，Gemma 4（48 层）需要更多。
   - `src/main.zig`：提示处理使用 `CGraph.initReserved(ctx, 16384)`
   - `src/core/graph_context.zig`：增量解码使用 `CGraph.initReserved(ctx, 16384)`

### 验证结果

- ✅ 模型加载成功，不再触发 `n_dims <= ne0` 断言
- ✅ 模型加载成功，不再触发 `cgraph->n_nodes < cgraph->size` 断言
- ⚠️ 生成质量不佳（输出乱码），说明架构实现仍有其他问题（KV Cache、注意力实现等需要进一步对齐 llama.cpp）

---

## 2. 参考：llama.cpp 的 Gemma 4 RoPE 处理

在 llama.cpp 中，Gemma 4 使用 `n_rot(il)` 方法**按层返回**不同的 RoPE 维度：

```cpp
uint32_t n_rot(uint32_t il) const {
    return is_swa(il) ? n_rot_swa : n_rot_full;
}
```

其中：
- `n_rot_full` 从 `{arch}.rope.dimension_count` 读取（默认 = `n_embd_head_k_full`）
- `n_rot_swa` 从 `{arch}.rope.dimension_count_swa` 读取（默认 = `n_rot_full`）
- 全注意力层还使用 `rope_freqs` 进行 proportional RoPE

---

## 3. 待解决：后续问题

Gemma 4 架构实现仍有以下问题需要解决：

1. **KV Cache**：当前 Gemma 4 的 KV Cache 被禁用（待实现 per-layer n_kv_head 支持）
2. **共享 KV 层**：后面几层复用前面层的 KV（`gemma4.attention.shared_kv_layers`）
3. **Per-layer embeddings**：支持 `gemma4.embedding_length_per_layer`
4. **MoE（Mixture of Experts）**：部分 Gemma 4 模型使用 MoE FFN
5. **Q/K 维度不匹配的处理**：当 `head_dim_q != head_dim_k` 时的 reshape 逻辑
6. **输出乱码**：当前生成结果不正确，需要对照 llama.cpp 逐层调试

---

## 4. 总结

- **直接崩溃点（已修复）**：RoPE 的 `n_dims` 大于输入向量长度。
- **根本原因**：Gemma 4 的 SWA 层 `head_dim` 小于全注意力层，但 `rope_dim` 使用全局单一值。
- **修复路径**：
  1. 从 GGUF 正确读取 `rope.dimension_count_swa` 超参数 ✅
  2. 在 `forward` 中按层动态选择 `rope_dim` ✅
  3. 安全夹持 `rope_dim ≤ head_dim` ✅
  4. 增大计算图节点容量 ✅
- **下一步**：修复 KV Cache、共享 KV 层、per-layer embeddings 等，对齐 llama.cpp 实现以提高生成质量。
