# gemma-3
```
❯ zig-out/bin/zllama -m ~/.cache/models/gemma-3-270m-it-Q8_0.gguf -p "say hello"
/private/tmp/ggml-20260608-5212-y1qoui/ggml-0.14.0/src/ggml.c:3485: GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b)) failed
WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.
WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.
See: https://github.com/ggml-org/llama.cpp/pull/17869
0   libggml-base.0.14.0.dylib           0x00000001019f48e5 ggml_print_backtrace + 273
1   libggml-base.0.14.0.dylib           0x00000001019f4b27 ggml_abort + 250
2   libggml-base.0.14.0.dylib           0x00000001019f84bf ggml_cast + 0
3   zllama                              0x000000010177cc7c ggml.ops.cpy + 92
4   zllama                              0x000000010177c971 kv_cache.KVCache.setKv + 1425
5   zllama                              0x00000001017c6c4e models.gemma3.Gemma3Model.forward + 3534
6   zllama                              0x00000001017c77a4 models.gemma3.Gemma3Model.buildGraph + 212
7   zllama                              0x00000001017c5e03 models.gemma3.Gemma3Model.buildGraphAdapter + 163
8   zllama                              0x00000001017497a0 model.ModelInstance.buildGraph + 96
9   zllama                              0x000000010174d36a main.InferenceEngine.generate + 1034
10  zllama                              0x00000001017660ae main.main + 2302
11  zllama                              0x0000000101766851 main + 1649
12  dyld                                0x00007ff8043c6b28 start + 3240
[1]    76699 abort      zig-out/bin/zllama -m ~/.cache/models/gemma-3-270m-it-Q8_0.gguf -p "say hello
```

## llama-3.2

```
❯ zig-out/bin/zllama -m ~/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p "1+1=?"
[ggml] [INFO] load_backend: loaded BLAS backend from /usr/local/Cellar/ggml/0.14.0/libexec/libggml-blas.so
[ggml] [INFO] load_backend: loaded CPU backend from /usr/local/Cellar/ggml/0.14.0/libexec/libggml-cpu-icelake.so
1+1=? (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (
```

---

## 待调查

- llama.cpp 中 Gemma 4 使用 `n_embd_head = hparams.n_embd_head_k(il)` 同时作为 Q 和 K 的 head_dim
- 我们的实现已与此一致（head_dim = q_norm.ne[0] = k_norm.ne[0]）
- 需要对比 logits 与参考实现定位剩余差异
- 可能差异点：attention 内部实现细节（permute 顺序、GQA 广播等）

## 一、三个问题模型的共同根源

### 1. Gemma 3 崩溃：`ggml_cpy` 形状不匹配
崩溃栈指向 `kv_cache.setKv` → `ggml_cpy` 断言元素数不等。这直接说明 **为某层分配的 cache 视图与实际写入的 K/V 张量维度不一致**。
原因可能是：

- **cache 初始化时使用了统一的 `head_dim`**（例如全局取 `max(head_dim_q, head_dim_k)`），但 Gemma 3 内部不同层的 `head_dim` 可能不同（类似 Gemma 4），导致某些层的 cache 视图过大或过小。
- **共享 KV 层**：如果某一层不计算自己的 K/V 而是复用之前层的，它的 cache 不应该分配新空间，否则会产生无效写入或形状不对。

### 2. Gemma 4 输出乱码（已修复多个 Bug 但仍乱）
虽然你已对齐了 RoPE 模式、频率加载、scale_factor 等，但乱码表明 **注意力计算或 masking 仍有差异**。可能原因：

- **SWA（滑动窗口注意力）掩码未正确实现**：Gemma 4 的部分层使用局部窗口（如 4096），其余层为全局因果注意力。如果窗口掩码或因果掩码的构造有误，会导致模型无法正确关注 context。
- **共享 KV 层的 cache 引用错误**：如果共享层没有正确指向被共享层的 cache，其注意力会读取到无效或全零的 KV，导致输出乱码。
- **GQA 广播未正确执行**：虽然之前认为 Q/K head_dim 相同，但 Gemma 4 可能存在 GQA（Q 头数 > KV 头数）。如果 `repeat_kv` 没有正确实现，注意力计算会维度不匹配或信息丢失。

### 3. Llama 3.2 输出重复 `(1)`
输出 "1+1=? (1) (1)..." 说明 logits 在第一个生成的 token `(1` 上极大概率为最高，且后续推理进入循环。可能原因：

- **K/V cache 序列位置错乱**：如果 cache 写入或读取时的位置索引（`pos`）错误，导致每步生成的 token 都看到同样的历史，模型就会反复输出相同 token。
- **GQA 的 cache 视图错误**：Llama 3.2 3B 采用 GQA（Q 头数 24，KV 头数 8）。若 cache 中 KV 的 `head_dim` 或 `n_kv_head` 不正确，可能会复制出错误的信息，导致模型输出退化。
- **采样参数**：如果温度等参数异常也可能导致，但更可能是 cache 问题。

---

## 二、根本原因总结

| 模型 | 表现 | 直接原因 | 深层原因 |
|------|------|---------|---------|
| Gemma 3 | 崩溃于 setKv | 某层 cache 视图与实际 K/V 张量元素数不同 | cache 未按层特化 head_dim / n_kv_head |
| Gemma 4 | 输出乱码 | SWA mask、共享 KV、GQA 广播未对齐 | 注意力计算 / cache 引用未完全遵循 llama.cpp 逻辑 |
| Llama 3.2 | 重复 token | cache 序列位置或 GQA 视图错误 | GQA 的 kv repeat 或 pos 传入有误 |

所有这些问题的交集都在 **`kv_cache.zig` 和 `attention.zig`**，尤其是：

- 没有 **per-layer** 的 cache 维度配置（每层可能有不同的 `head_dim`、`n_kv_head`）。
- 没有正确处理 **共享 KV 层**（需要引用而非复制）。
- 没有实现 **GQA 的 KV 重复**（将 `[head_dim_kv, n_kv_head, seq]` 扩展为 `[head_dim_kv, n_head, seq]`）。
- **位置编码索引（RoPE pos）** 可能未正确区分 SWA 和全局层。

---

## 三、解决方案路线图

### 1. 重构 KV Cache 为 per-layer 配置
在 `kv_cache.zig` 中，不要使用全局的 `head_dim` 和 `n_kv_head`，而是从模型传入每层的具体参数。初始化时对每一层分配恰好大小的缓冲区。

**关键数据结构**：
```zig
pub const KVCache = struct {
    per_layer: []LayerCache,
    // ...
};

const LayerCache = struct {
    k: *ggml.Tensor,   // [head_dim_k, n_kv_head, max_seqlen]
    v: *ggml.Tensor,
    // 对于共享层，可存储指向另一层的指针
};
```

### 2. 支持共享 KV 层
在模型定义中标记哪些层是共享的，并在 `setKv` 时跳过分配，直接将被共享层的 cache 视图传给 attention。

### 3. 实现正确的 GQA 广播（`repeat_kv`）
在 `attention.zig` 中，当 `n_head != n_kv_head` 时，需要将 K、V 在头维度上复制 `n_head / n_kv_head` 次。ggml 方式：
```zig
// K 形状: [head_dim_kv, n_kv_head, seq]
K = K.reshape4d(ctx, head_dim_kv, n_kv_head, 1, seq);
K = K.repeat(ctx, 1, 1, n_head / n_kv_head, 1); // 重复头维度
K = K.reshape3d(ctx, head_dim_kv, n_head, seq);
```

### 4. 对齐 SWA 掩码
对于 Gemma 4 的 SWA 层，需要构建局部窗口掩码（如 `[seq, seq]` 的上三角掩码但允许窗口内的元素）。参考 llama.cpp 的 `build_attn_mask_swa` 逻辑。

### 5. 检查位置编码传递
确保每个 token 的 `pos` 参数在 prefill 和解码阶段正确递增，且 RoPE 应用时的 offset 一致。对于共享 KV 的层，应使用被共享层的 pos。

### 6. 增加调试日志与对比测试
- 在 `setKv` 前打印每层 K/V 的期望形状和 cache 视图形状。
- 用 Python (llama-cpp-python) 生成相同模型的 logits，与 zig 版本逐层对比。
- 先修复 Gemma 3 崩溃，因为它是最直接的形状错误，修复后很可能同时解决 Gemma 4 的乱码。

---

## 四、预期效果

完成以上重构后：
- Gemma 3 不再崩溃，因为每层 cache 维度正确。
- Gemma 4 输出应恢复正常，因为 SWA mask、共享 KV、GQA 均已对齐。
- Llama 3.2 不再重复输出，因为 cache 视图和 GQA 广播正确，生成逻辑恢复正常。

建议优先修复 **KV cache per-layer 配置** 和 **GQA 广播**，这两个改动能解决大部分模型的共同问题。
