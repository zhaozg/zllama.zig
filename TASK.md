# 修复状态

## ✅ 已完成修复

### 1. Gemma 3 崩溃 → FIXED
**原因**: K/V 张量在 permute(0,2,1,3) 之后（布局变为 `[head_dim, n_tokens, n_kv_head]`）才存入 KV Cache，
但 cache 期望 `[head_dim, n_kv_head, seq_len]` 布局，导致 `ggml_cpy` 元素数不匹配。

**修复**: 在 gemma3.zig 中将 `setKv` 调用移到 permute 之前，保持 `[head_dim, n_kv_head, n_tokens]` 布局。
attention 层内部已自行执行 permute(0,2,1,3)。

### 2. 共享 KV 层 → FIXED
**原因**: 非 KV 层（Gemma 4 中 `has_kv=false`）的 `getKView`/`getVView` 使用该层自己的 `current_len`（为 0），
返回空视图。

**修复**: `getKView`/`getVView` 现在使用全局最大长度 `self.currentLen()`，而非层自己的 `current_len`。

### 3. GQA 广播 → FIXED
**原因**: 之前依赖 ggml_mul_mat 自动 GQA 广播，但该方法在跨维度 reshape 场景下不可靠。

**修复**: 在 attention.zig 中添加显式 GQA repeat：当 `n_head != n_kv_head` 时，
用 `reshape4d` + `repeat4d` 将 K/V 从 `[head_dim, n_kv_head, cache_len]` 扩展为 `[head_dim, n_head, cache_len]`。

### 4. KV Cache stride 计算 → FIXED
**原因**: `setKv` / `getKView` / `getVView` 中 view3d 的 stride（nb1, nb2）基于视图维度手动计算，
当 per-layer head_dim 与全局不同时，stride 计算错误。

**修复**: 使用父张量的实际 stride `parent_nb[1]` 和 `parent_nb[2]`。

### 5. SWA 掩码支持 → ADDED
- 在 `ggml/ops.zig` 添加 `softMaxExt` 绑定
- 在 `attention.zig` 添加 `buildAttentionMask` 函数（构建因果+滑动窗口掩码）
- `scaledDotProductAttention` 新增 `swa_window: ?i64` 参数
  - `null`: 使用原有 `diagMaskInf` + `softMax`（全注意力）
  - `Some(window)`: 使用 SWA 掩码 + `softMaxExt`（滑动窗口注意力）
- Gemma 3: 传 `null`（SWA 通过 RoPE 频率变化实现）
- Gemma 4: 对 SWA 层传 `p.n_swa`，全注意力层传 `null`
- LLaMA/Qwen: 传 `null`

## 待验证
- [ ] 实际运行 Gemma 3 270m 模型确认不再崩溃
- [ ] 运行 Llama 3.2 3B 确认不再重复输出
- [ ] 运行 Gemma 4 模型确认输出质量
- [ ] 对比 logits 与参考实现定位剩余差异

## 修改文件清单
- `src/kv_cache.zig`: getKView/getVView 使用全局 currentLen；setKv/getKView/getVView 使用父张量 stride
- `src/models/gemma3.zig`: KV cache 存入时机移到 permute 之前
- `src/models/gemma4.zig`: SWA 层传窗口大小给 attention
- `src/models/llama.zig`: 传 null swa_window
- `src/models/qwen35.zig`: 传 null swa_window
- `src/models/qwen2.zig`: 传 null swa_window
- `src/layers/attention.zig`: 显式 GQA repeat；SWA 掩码构建；新增 swa_window 参数
- `src/ggml/ops.zig`: 新增 softMaxExt 绑定
- `src/ggml.zig`: 重新导出 softMaxExt
