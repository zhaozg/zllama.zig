# Qwen 3.5 混合架构分析与实现指南

> **项目：** `zllama.zig` — 纯 Zig 实现的多模型本地推理引擎
> **参考实现：** `deps/llama.cpp/src/models/qwen35.cpp` (649 行)
> **当前实现：** `src/models/qwen.zig` (约 664 行)
> **基类参考：** `deps/llama.cpp/src/models/delta-net-base.cpp` (GDN 算子基类)

---

## 一、架构概述

Qwen 3.5 采用 **混合注意力架构**（Hybrid Attention），在标准 Transformer 中交替使用：

1. **全注意力层（Full Attention）** — 标准 GQA + MRoPE + Q‑gate
2. **线性注意力层（Linear Attention / Gated Delta Net）** — SSM 风格的状态空间模型

这种设计在保持长上下文能力的同时，显著降低计算复杂度。

### 1.1 层分布策略

```
n_main = n_layer - nextn_predict_layers
full_attn_interval = 4  (默认)
recurrent_layer_arr[i] = (i < n_main) && ((i + 1) % full_attn_interval != 0)
```

即：每 4 层中，第 4 层（索引 3, 7, 11, ...）为全注意力层，其余为线性注意力层。

### 1.2 模型规模识别

```cpp
switch (n_layer - nextn_predict_layers) {
    case 24: type = (n_embd == 1024) ? 0.8B : 2B; break;
    case 32: type = (n_embd == 2560) ? 4B  : 9B; break;
    case 64: type = 27B; break;
}
```

---

## 二、核心方法分析（基于 llama.cpp 参考实现）

### 2.1 `load_arch_hparams` — 超参数加载

**参考行：** qwen35.cpp:3-42

从 GGUF 元数据读取 Qwen 3.5 特有参数：

| 参数 | GGUF Key | 说明 |
|------|----------|------|
| `f_norm_rms_eps` | `attention.layer_norm_rms_epsilon` | RMSNorm epsilon |
| `rope_sections` | `rope.dimension_sections` | MRoPE 分段 [4] |
| `ssm_d_conv` | `ssm.conv_kernel` | 卷积核大小 (默认 4) |
| `ssm_d_inner` | `ssm.inner_size` | SSM 内部维度 |
| `ssm_d_state` | `ssm.state_size` | SSM 状态维度 (默认 128) |
| `ssm_dt_rank` | `ssm.time_step_rank` | 时间步长秩 (默认 16) |
| `ssm_n_group` | `ssm.group_count` | K‑head 组数 (默认 16) |
| `nextn_predict_layers` | `nextn.predict_layers` | MTP 层数 |
| `full_attention_interval` | `full_attention_interval` | 全注意力间隔 (默认 4) |

**当前实现状态：** ✅ 已实现 `parseParams()`，读取所有关键参数。

### 2.2 `load_arch_tensors` — 张量加载

**参考行：** qwen35.cpp:44-148

#### 全注意力层张量

| 张量 | 形状 | 说明 |
|------|------|------|
| `attn_norm.weight` | `[n_embd]` | Pre‑attention RMSNorm |
| `attn_post_norm.weight` | `[n_embd]` | Post‑attention RMSNorm |
| `attn_q.weight` | `[n_embd_head*2*n_head, n_embd]` | Q 投影（含 gate） |
| `attn_k.weight` | `[n_embd_head*n_head_kv, n_embd]` | K 投影 |
| `attn_v.weight` | `[n_embd_head*n_head_kv, n_embd]` | V 投影 |
| `attn_output.weight` | `[n_embd_head*n_head, n_embd]` | 输出投影 |
| `attn_q_norm.weight` | `[n_embd_head]` | Q 归一化 |
| `attn_k_norm.weight` | `[n_embd_head]` | K 归一化 |

#### 线性注意力层张量

| 张量 | 形状 | 说明 |
|------|------|------|
| `attn_norm.weight` | `[n_embd]` | Pre‑attention RMSNorm |
| `attn_post_norm.weight` | `[n_embd]` | Post‑attention RMSNorm |
| `attn_qkv.weight` | `[n_embd, key_dim*2+value_dim]` | QKV 混合投影 |
| `attn_gate.weight` | `[n_embd, value_dim]` | Z‑gate 投影 |
| `ssm_conv1d.weight` | `[d_conv, conv_dim]` | 1D 因果卷积 |
| `ssm_dt.bias` | `[dt_rank]` | 时间步长偏置 |
| `ssm_a` | `[dt_rank]` | A 矩阵（无扫描） |
| `ssm_beta.weight` | `[n_embd, n_v_heads]` | Beta 投影 |
| `ssm_alpha.weight` | `[n_embd, n_v_heads]` | Alpha 投影 |
| `ssm_norm.weight` | `[head_v_dim]` | 输出归一化 |
| `ssm_out.weight` | `[value_dim, n_embd]` | 输出投影 |

#### MTP 层额外张量

| 张量 | 形状 | 说明 |
|------|------|------|
| `nextn.eh_proj` | `[2*n_embd, n_embd]` | Embedding‑Hidden 投影 |
| `nextn.enorm` | `[n_embd]` | Embedding norm |
| `nextn.hnorm` | `[n_embd]` | Hidden norm |
| `nextn.embed_tokens` | `[n_embd, n_vocab]` | 可选嵌入 |
| `nextn.shared_head_head` | `[n_embd, n_vocab]` | 可选 LM head |
| `nextn.shared_head_norm` | `[n_embd]` | 可选输出 norm |

**当前实现状态：** ✅ 已实现 `loadWeights()`，支持全注意力和 SSM 层张量加载。MTP 层暂未实现。

### 2.3 `build_arch_graph` — 图构建入口

**参考行：** qwen35.cpp:150-156

```cpp
std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}
```

**当前实现状态：** ⬜ 当前 `forward()` 直接实现推理逻辑，未分离为独立的 graph 类。

### 2.4 `graph` 构造函数 — 主推理图

**参考行：** qwen35.cpp:158-260

核心流程：
```
inpL = build_inp_embd(tok_embd)          // Token 嵌入
inp  = build_inp_mem_hybrid()            // 混合内存输入（KV Cache + SSM State）

for il in 0..n_transformer_layers:
    cur = rms_norm(inpL, attn_norm)

    if is_recurrent(il):
        cur = build_layer_attn_linear(inp->get_recr(), cur, il)  // GDN
    else:
        cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il)  // Full Attn

    cur = cur + inpSA  // Residual

    ffn_residual = cur
    cur = rms_norm(cur, attn_post_norm)
    cur = build_layer_ffn(cur, il)  // SwiGLU
    cur = cur + ffn_residual

    inpL = cur

cur = rms_norm(cur, output_norm)
cur = mm(output, cur)  // LM Head
```

**当前实现状态：** ✅ 已实现 `forward()` 函数，包含完整的层循环。

### 2.5 `build_layer_attn` — 全注意力层

**参考行：** qwen35.cpp:262-370

关键特性：
1. **Q 投影输出 [Q, gate] 联合**：`wq` 输出维度为 `n_embd_head * 2 * n_head`
2. **Q/K 归一化**：使用 `attn_q_norm` 和 `attn_k_norm`
3. **MRoPE**：多维 RoPE，使用 `rope_sections` 分段
4. **Gate 机制**：`sigmoid(gate) * attn_out`
5. **标准 GQA**：通过 `build_attn()` 实现

**当前实现状态：** ✅ 已实现 `forwardFullAttention()`，支持 Q‑gate、Q/K norm、MRoPE。

### 2.6 `build_layer_attn_linear` — 线性注意力层 (GDN)

**参考行：** qwen35.cpp:372-530

核心流程：
```
1. Input projections:
   qkv_mixed = mm(wqkv, input)   // [key_dim*2 + value_dim, n_tokens]
   z         = mm(wqkv_gate, input)  // [value_dim, n_tokens]

2. Beta: beta = sigmoid(mm(ssm_beta, input))

3. Alpha -> Gate:
   alpha = mm(ssm_alpha, input)
   gate = softplus(alpha + dt_bias) * ssm_a

4. Conv1d:
   conv_input = [conv_state | qkv_mixed]
   conv_output = silu(ssm_conv(conv_input, conv_kernel))

5. Extract Q, K, V from conv_output:
   q_conv = conv_output[0:key_dim]
   k_conv = conv_output[key_dim:key_dim*2]
   v_conv = conv_output[key_dim*2:]

6. L2 normalize Q, K

7. Gated Delta Net:
   output = build_recurrent_attn(inp, ssm_states, q_conv, k_conv, v_conv, gate, beta, state)

8. Gated normalization:
   attn_out = norm(output) * silu(z)

9. Output projection:
   cur = mm(ssm_out, attn_out)
```

**当前实现状态：** ✅ 已实现 `forwardSSM()`，包含完整的 GDN 流程，并使用 `ggml.gatedDeltaNet()` 融合算子。

### 2.7 `build_qkvz` — QKV+Z 投影

**参考行：** qwen35.cpp:232-248

```cpp
std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(ggml_tensor * input, int il) {
    ggml_tensor * qkv_mixed = mm(wqkv, input);  // [key_dim*2+value_dim, n_tokens]
    ggml_tensor * z = mm(wqkv_gate, input);      // [value_dim, n_tokens]
    return {qkv_mixed, z};
}
```

**当前实现状态：** ✅ 已实现。

### 2.8 `build_norm_gated` — 门控归一化

**参考行：** qwen35.cpp:250-260

```cpp
ggml_tensor * build_norm_gated(input, weights, gate, layer) {
    normalized = rms_norm(input, weights)
    gated_silu = silu(gate)
    return normalized * gated_silu
}
```

**当前实现状态：** ✅ 已实现。

### 2.9 `build_layer_ffn` — FFN 层

**参考行：** qwen35.cpp:532-542

Qwen 3.5 使用标准 SwiGLU FFN（无 MoE）：
```
cur = silu(mm(gate, x)) * mm(up, x)
cur = mm(down, cur)
```

**当前实现状态：** ✅ 已实现。

### 2.10 `graph_mtp` — MTP 解码头

**参考行：** qwen35.cpp:544-649

MTP（Multi‑Token Prediction）是 Qwen 3.5 的额外解码块，用于预测多个未来 token。当前实现暂不涉及。

**当前实现状态：** ⬜ 未实现（MTP 为可选功能）。

---

## 三、GDN（Gated Delta Net）算子深入分析

### 3.1 参考实现

GDN 算子在 `deps/llama.cpp/src/models/delta-net-base.cpp` 中实现，提供三种模式：

| 模式 | 函数 | 适用场景 |
|------|------|---------|
| Chunking | `build_delta_net_chunking` | 多 token 并行（prefill） |
| Autoregressive | `build_delta_net_autoregressive` | 单 token 自回归（decode） |
| Fused | `build_delta_net_fused` | 使用 `ggml_gated_delta_net` 融合算子 |

### 3.2 数学原理

GDN 的核心是维护一个状态矩阵 S（shape: `[S_v, S_v, H_v, n_seqs]`），通过以下方式更新：

```
S_new = beta * S + (1 - beta) * (k * v^T)   // 状态更新
output = S * q                                 // 状态读取
output = output * gate                         // 门控
```

其中：
- `q, k` 是 L2 归一化后的查询和键
- `v` 是值
- `beta = sigmoid(projection)` 是遗忘门
- `gate = softplus(alpha + dt) * A` 是输入门

### 3.3 当前实现状态

当前 `forwardSSM()` 已使用 `ggml.gatedDeltaNet()` 融合算子（`src/models/qwen.zig:416`），通过 `ggml.zig` 安全封装层调用。状态管理通过 `LayerSSMState` 结构体维护每层的 `conv_state` 和 `ssm_state`，在 `ctx_graph.reset()` 后通过 `resetSSMStates()` 重置。

---

## 四、与 llama.cpp 参考实现的差异分析

| 特性 | llama.cpp (qwen35.cpp) | zllama.zig (qwen.zig) | 差异说明 |
|------|----------------------|----------------------|---------|
| 架构检测 | `LLM_ARCH_QWEN35` | `Architecture.qwen35` | 等效 |
| 超参数加载 | `load_arch_hparams()` | `parseParams()` | 等效 |
| 张量加载 | `load_arch_tensors()` | `loadWeights()` | 等效 |
| 图构建 | `build_arch_graph()` → `graph` 类 | `forward()` 直接实现 | 架构差异 |
| 全注意力 | `build_layer_attn()` | `forwardFullAttention()` | 等效 |
| 线性注意力 | `build_layer_attn_linear()` | `forwardSSM()` | 等效 |
| GDN 实现 | `build_delta_net_*()` (基类) | `ggml.gatedDeltaNet()` 融合算子 | ✅ 已使用融合算子 |
| MTP 支持 | `graph_mtp` 类 | 未实现 | 待完成 |
| 混合内存 | `build_inp_mem_hybrid()` | 手动管理 KV Cache + SSM State | 需改进 |
| MRoPE | `ggml_rope_multi()` | `ggml.ropeMulti()` | 等效 |
| Q‑gate | `sigmoid(gate) * attn_out` | 相同 | 等效 |
| Q/K norm | `rms_norm(Q)`, `rms_norm(K)` | 相同 | 等效 |
| L2 norm (SSM) | `ggml_l2_norm()` | `ggml.l2Norm()` | 等效 |
| SSM 状态重置 | 自动管理 | `resetSSMStates()` | `main.zig` 中缺失调用 |

---

## 五、当前实现状态总结

### ✅ 已完成

1. **GGUF 元数据解析** — 读取所有 Qwen 3.5 特定参数
2. **张量加载** — 支持全注意力和 SSM 层的所有张量
3. **全注意力层前向** — 含 Q‑gate、Q/K norm、MRoPE
4. **SSM 层前向** — 含 Conv1d、GDN（使用 `ggml.gatedDeltaNet()` 融合算子）、门控归一化
5. **层类型判断** — 基于 `full_attention_interval`
6. **输出投影** — 支持 `output.weight` 或 `token_embd.weight` 回退
7. **KV Cache 集成** — 标准 GQA KV Cache
8. **SSM 状态管理** — 每层 conv_state 和 ssm_state，支持 `resetSSMStates()`
9. **`ggml.gatedDeltaNet()` 封装** — 在 `ggml.zig` 中导出，`ops.zig` 中实现
10. **`simple_main.zig` 支持** — 增量解码循环中正确调用 `resetModelSSMStates()`

### ⬜ 待完成

1. **MTP 解码头** — `graph_mtp` 类
2. **混合内存输入** — `build_inp_mem_hybrid` 模式
3. **与 llama.cpp 对比测试** — 确保数值正确性
4. **graph 类分离** — 参考 llama.cpp 的 graph 设计模式

### ✅ 已修复

1. **`main.zig` 缺少 `resetModelSSMStates` 调用** — 在增量解码循环中（`src/main.zig:307`），`ctx_graph.reset()` 后未调用 `registry.resetModelSSMStates()`，导致 Qwen35 的 SSM 状态在第二次迭代时指向已释放的内存。

**修复：** 已在 `src/main.zig:309` 添加 `registry.resetModelSSMStates(self.model_ptr, self.arch)` 调用。

## 六、优化路径

### 阶段 1：正确性验证与修复（当前阶段）

| 任务 | 状态 | 说明 |
|------|------|------|
| 1.1 验证 GGUF 元数据读取 | ✅ | `parseParams()` 已实现 |
| 1.2 验证张量名称和形状 | ✅ | `loadWeights()` 已实现 |
| 1.3 验证全注意力层前向 | ✅ | `forwardFullAttention()` 已实现 |
| 1.4 验证 SSM 层前向 | ✅ | `forwardSSM()` 已实现，使用 `ggml.gatedDeltaNet()` |
| 1.5 验证层类型判断逻辑 | ✅ | `isFullAttentionLayer()` 已实现 |
| 1.6 验证输出投影 | ✅ | 支持 output.weight 或 token_embd.weight |
| 1.7 端到端推理测试 | ⬜ | 需要与 llama.cpp 输出对比 |
| 1.8 `main.zig` 缺少 `resetModelSSMStates` | ✅ | 已在 `src/main.zig:309` 添加调用 |

### 阶段 2：性能优化

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 2.1 使用 `ggml_gated_delta_net` 融合算子 | ✅ | 已完成，`forwardSSM()` 使用 `ggml.gatedDeltaNet()` |
| 2.2 优化 Conv1d 状态管理 | 高 | 使用 ggml 的 SSM conv 操作 |
| 2.3 实现 `ggml_graph_plan` + 线程池 | 中 | 多线程调度优化 |
| 2.4 减少不必要的 `ggml_cont` 调用 | 中 | 避免内存重排 |
| 2.5 预分配 SSM 状态张量 | 低 | 避免运行时分配 |

#### 其他针对性优化建议

- **简化线性注意力实现**：已使用 fused 算子 (`ggml_gated_delta_net`)，消除冗余 reshape。
- **全注意力层 QKV 融合**：对于不支持 Q+Gate 联合投影的简化版本，可退化为标准 fused QKV。
- **MRoPE 预计算**：`rope_sections`、`freq_base`/`freq_scale` 等参数在推理中不变，可提前缓存。
- **MTP 与主图分离**：MTP 可共享主图的部分计算，避免重复。
- **混合缓存预分配**：KV Cache 与 SSM 状态均预分配最大序列长度，使用视图操作避免数据复制。

### 阶段 3：功能完善

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 3.1 MTP 解码头支持 | 低 | 多 token 预测 |
| 3.2 与 llama.cpp 输出对比测试 | 高 | 确保数值正确性 |
| 3.3 支持更多量化格式 | 中 | Q5_K_M, Q6_K, Q8_0 |
| 3.4 支持 `nextn_predict_layers` | 低 | MTP 层数配置 |

### 阶段 4：架构改进

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 4.1 分离 graph 构建与执行 | 中 | 参考 llama.cpp 的 graph 类设计 |
| 4.2 实现 `build_inp_mem_hybrid` | 中 | 统一管理 KV Cache + SSM State |
| 4.3 支持 `llm_graph_input_rs` 接口 | 中 | 递归状态输入 |
| 4.4 支持 `llm_graph_input_attn_kv` 接口 | 中 | 注意力 KV 输入 |

### Zig 特定适配

针对 Zig 0.16.0：
- 将 C++ 的 `std::pair` 替换为 Zig 元组或结构体
- `GGML_ASSERT` → Zig 错误联合 `!void`
- `std::unique_ptr` → `allocator` 模式
- C++ 虚函数 → Zig 的 `switch` + tagged union

---

## 七、推荐实现顺序

```
Phase 1: 基础架构
  ├── GGUF 解析器（读取所有 qwen35 特有参数）
  ├── 模型注册（registry.zig 添加 qwen35）
  └── 张量加载（load_arch_tensors）

Phase 2: 全注意力层
  ├── Q+Gate 联合投影
  ├── Q/K Norm
  ├── MRoPE
  └── Gated Attention

Phase 3: 线性注意力层
  ├── wqkv 投影 + 1D 卷积
  ├── Delta Net 循环注意力（使用 ggml.gatedDeltaNet 融合算子）
  ├── Gated Norm
  └── 状态管理（conv + ssm states）

Phase 4: 混合调度
  ├── 层类型判断（is_recurrent）
  ├── 混合缓存管理
  └── 主推理循环

Phase 5: MTP（可选）
  ├── MTP 图构建
  └── 共享 head 权重
```

---

## 八、参考文件索引

| 文件 | 用途 |
|------|------|
| `deps/llama.cpp/src/models/qwen35.cpp` | Qwen 3.5 主实现（649 行） |
| `deps/llama.cpp/src/models/delta-net-base.cpp` | GDN 基类实现 |
| `deps/llama.cpp/src/models/models.h` (L1753-1798) | Qwen 3.5 结构定义 |
| `deps/llama.cpp/src/llama-graph.h` (L476-560) | 混合内存输入类 |
| `deps/llama.cpp/src/llama-graph.h` (L777-1110) | `llm_graph_context` 基类 |
| `deps/llama.cpp/src/llama-model.h` (L652-750) | `llama_model_base` 基类 |
| `deps/llama.cpp/src/llama-arch.h` (L46-47) | 架构枚举 |
| `deps/llama.cpp/src/llama-arch.cpp` (L864-868) | 架构特性标记 |
