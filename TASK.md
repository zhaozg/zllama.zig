# 修正计算错误导致的维度不匹配崩溃

这个崩溃是由于推理引擎（Qwen Engine / ggml）在解析模型维度时出现了**计算错误**，导致张量重塑（reshape）时元素数量不匹配而触发了断言失败（Assertion Failed）。

以下是详细的问题分析和解决方案：

### 1. 崩溃核心定位
日志中的关键报错信息如下：
```text
debug: reshape3d: [512,10,1,1] -> [128,10,2] nelem=5120 expected=2560
/private/tmp/ggml-20260529-5333-fpdyat/ggml-0.13.1/src/ggml.c:3648: GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed
```
- **实际张量大小**：`[512, 10, 1, 1]`，总元素数为 `512 * 10 = 5120`。（其中 `10` 是你的 prompt 长度，`512` 是 KV 的特征维度）。
- **引擎期望大小**：`[128, 10, 2]`，总元素数为 `128 * 10 * 2 = 2560`。
- **结果**：`5120 != 2560`，触发 `GGML_ASSERT` 崩溃。

### 2. 根本原因分析
引擎在计算注意力机制的维度时出现了逻辑错误：
1. **引擎的错误计算**：从日志 `info(model): embd=1024, heads=8, kv_heads=2, head_dim=128` 可以看出，引擎通过公式 `head_dim = hidden_size / num_attention_heads`（即 `1024 / 8 = 128`）推导出了 `head_dim`。因此，它期望 KV 的特征维度为 `kv_heads * head_dim = 2 * 128 = 256`。
2. **模型的真实配置**：根据 Qwen3.5-0.8B 的官方 `config.json`，该模型采用了混合架构（Hybrid SSM + Full Attention），其显式配置为：
   - `hidden_size`: 1024
   - `num_attention_heads`: 8
   - `num_key_value_heads`: 2
   - **`head_dim`: 256** （这是关键，Qwen3.5 的 head_dim 并不等于 hidden_size / heads）

   因此，真实的 KV 特征维度应为 `2 * 256 = 512`，这与 GGUF 文件中实际加载的张量形状 `[512, ...]` 完全吻合。

**结论**：推理引擎未能正确读取 GGUF 元数据中显式声明的 `head_dim`（或 `attention.key_length`），而是退化使用了错误的默认公式进行计算，从而导致维度不匹配。

### 3. 修复方案

#### 修复：优先从 GGUF 元数据读取显式 head_dim

在 `src/model.zig` 的 `parseParams()` 函数中，修改 `n_head_dim` 的计算逻辑：

**修改前**：
```zig
// 计算 head_dim: 始终使用 n_embd / n_head
params.n_head_dim = if (params.n_head > 0 and params.n_embd > 0)
    params.n_embd / params.n_head
else
    0;
```

**修改后**：
```zig
// 读取 Qwen 3.5 全注意力层的 K/V 维度（优先使用显式声明的 head_dim）
params.attn_key_length = gguf_file.getU32("qwen35.attention.key_length") orelse
    gguf_file.getU32("llama.attention.key_length") orelse 0;
params.attn_value_length = gguf_file.getU32("qwen35.attention.value_length") orelse
    gguf_file.getU32("llama.attention.value_length") orelse 0;

// 计算 head_dim: 优先使用显式声明的 key_length，否则回退到 n_embd / n_head
if (params.attn_key_length > 0) {
    params.n_head_dim = params.attn_key_length;
} else if (params.n_head > 0 and params.n_embd > 0) {
    params.n_head_dim = params.n_embd / params.n_head;
} else {
    params.n_head_dim = 0;
}
```

这个修复确保：
1. 如果 GGUF 元数据中包含 `qwen35.attention.key_length`（或 `llama.attention.key_length`），则优先使用它作为 `head_dim`。
2. 否则回退到原来的 `n_embd / n_head` 公式。
3. KV Cache 初始化（`src/main.zig` 第 230 行）使用 `params.n_head_dim`，因此也会自动使用正确的维度。

### 4. 验证

修复后，对于 Qwen3.5-0.8B 模型：
- `n_head_dim = 256`（从 `qwen35.attention.key_length` 读取）
- K 投影输出：`[n_kv_head * head_dim, n_tokens] = [2 * 256, 10] = [512, 10]`
- K reshape：`[head_dim, n_tokens, n_kv_head] = [256, 10, 2]`
- 元素数：`512 * 10 = 5120 == 256 * 10 * 2 = 5120` ✓

### 总结
这是一个典型的**模型配置元数据与推理引擎解析逻辑不匹配**导致的 Bug。通过优先读取 GGUF 元数据中显式声明的 `key_length`，引擎可以正确识别 `head_dim=256`，从而消除该断言错误。
