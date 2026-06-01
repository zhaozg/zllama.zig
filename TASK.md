## 当前代码状态与待修复问题

### 当前状态

代码已实现：
- ✅ GGUF 文件解析（v2/v3）
- ✅ 模型参数加载
- ✅ 权重加载（全注意力层 + SSM 层）
- ✅ 分词器初始化
- ✅ KV Cache 初始化
- ✅ 首 token 完整前向计算图构建
- ✅ 增量解码循环

### 已修复问题

#### ✅ 1. `setKv()` 中的 `ggml_cpy` 未加入计算图

**修复**：`kv_cache.zig` 中的 `setKv()` 现在接受 `*ggml.CGraph` 参数，并将 `ggml_cpy` 的结果通过 `graph.buildForwardExpand()` 注册到计算图中。

**修改文件**：`src/kv_cache.zig`
- `setKv()` 签名新增 `graph: *ggml.CGraph` 参数
- `ggml.cpy()` 的结果调用 `graph.buildForwardExpand()` 加入计算图

#### ✅ 2. KV Cache 在第一个全注意力层执行时长度为 0

**修复**：在 `model.zig` 的 `buildForwardGraph` 中，当 `cache_len == 0` 时（SSM 层未写入 Cache），使用 `n_tokens` 作为 cache_len。

**修改文件**：`src/model.zig`
- 新增 `raw_cache_len` 变量获取原始 cache 长度
- 新增 `cache_len` 变量，当 `raw_cache_len > 0` 时使用 `raw_cache_len`，否则使用 `n_tokens_i64`

#### ✅ 3. `setKv()` 调用更新

**修复**：`model.zig` 中调用 `cache.setKv()` 时传入 `graph` 参数。

**修改文件**：`src/model.zig`
- `cache.setKv(ctx, i, k, v, ...)` → `cache.setKv(ctx, graph, i, k, v, ...)`

### 待修复问题

#### ⏳ 4. `ggml.Context.reset()` 导致 KV Cache 数据丢失

**现象**：增量解码时，`self.ggml_ctx.reset()` 会释放 KV Cache 张量的内存。

**原因**：
- `main.zig` 中 `generate()` 函数在增量步骤后调用 `self.ggml_ctx.reset()` 以释放中间张量。
- 但 KV Cache 张量也是在同一个 ggml context 中分配的，`reset()` 会释放所有后续分配的内存。

**修复方案**：
- 使用独立的 ggml context 管理 KV Cache，或者在 `reset()` 后重新初始化 Cache 视图。

#### ⏳ 5. SSM 层不写入 KV Cache

**现象**：SSM 层不调用 `setKv()`，导致 Cache 长度不更新。

**说明**：SSM 层的 K/V 维度（`ssm_inner`）与全注意力层的 K/V 维度（`head_dim * n_kv_head`）不同，不能直接写入同一个 Cache。当前修复方案（问题 2）通过 `cache_len` 回退机制解决了这个问题。

#### ⏳ 6. 计算图构建中的张量维度不匹配

**现象**：在注意力计算中，`reshape2d` 和 `mulMat` 操作可能因维度不匹配而失败。

**修复方案**：
- 在 `buildForwardGraph` 中，当 `cache_len == 0` 时，跳过注意力计算，直接使用当前输入的 K/V。

### 修复优先级

1. ✅ **已完成**：修复问题 1（ggml_cpy 未加入计算图）和问题 2（cache_len=0 导致崩溃）
2. **中优先级**：修复问题 4（reset 导致 KV Cache 丢失）
3. **低优先级**：修复问题 5（SSM 层不写入 Cache）和问题 6（维度不匹配的健壮性）

### 验证方法

1. 运行 `zig build` 确保编译通过 ✅
2. 运行 `zig build test` 确保测试通过 ✅
3. 运行 `zig build run -- --model <model_path> --prompt "Hello" --max-tokens 10` 测试推理
4. 检查输出是否合理（非乱码）
5. 检查 KV Cache 长度是否正确增长
