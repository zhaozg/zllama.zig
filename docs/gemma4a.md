# Gemma4 语言推理过程分析

> [llama.cpp](deps/llama.cpp) 中 Gemma4 (`LLM_ARCH_GEMMA4`) 模型推理全流程分析，涵盖模型加载、计算图构建、前向推理与 KV Cache 机制。

---

## 一、架构概览

### 1.1 模型家族

llama.cpp 中 Gemma 系列包含以下架构变体：

| 架构常量 | 字符串标识 | 说明 |
|---|---|---|
| `LLM_ARCH_GEMMA` | `"gemma"` | Gemma v1 |
| `LLM_ARCH_GEMMA2` | `"gemma2"` | Gemma v2 |
| `LLM_ARCH_GEMMA3` | `"gemma3"` | Gemma v3（支持 5:1 SWA） |
| `LLM_ARCH_GEMMA3N` | `"gemma3n"` | Gemma3n（AltUp + Laurel + per-layer） |
| `LLM_ARCH_GEMMA4` | `"gemma4"` | **Gemma4（SWA + MoE + per-layer）** |
| `LLM_ARCH_GEMMA4_ASSISTANT` | `"gemma4-assistant"` | Gemma4 投机解码草稿模型 |

### 1.2 Gemma4 核心特性

- **SWA (Sliding Window Attention)**：交替使用全局 attention 和滑动窗口 attention 层
- **KV 共享层**：后 N 层复用前面层的 KV Cache，节省显存
- **MoE (Mixture of Experts)**：部分层使用 MoE FFN + 共享 Dense FFN
- **Per-Layer Embeddings**：每层注入层特定的嵌入信息
- **Double Wide MLP**：KV 共享层可使用 2x FFN 宽度
- **QK Norm + Post Norm**：Attention 的 Q/K 各自 RMSNorm，attention 和 FFN 后均有 post-norm
- **Proportional RoPE**：全局 attention 层使用 proportional RoPE（部分维度不旋转）
- **Final Logit Softcapping**：logits 输出经 tanh 软截断

### 1.3 模型规模

| `n_layer` | 类型常量 | 说明 |
|---|---|---|
| 30 | `LLM_TYPE_26B_A4B` | 26B 参数，4B 活跃 |
| 35 | `LLM_TYPE_E2B` | ~2B |
| 42 | `LLM_TYPE_E4B` | ~4B |
| 60 | `LLM_TYPE_31B` | 31B |

---

## 二、Python 端模型转换

### 2.1 类层次

```
TextModel
  └── Gemma3Model
        └── Gemma4Model          (Gemma4ForCausalLM)
              ├── Gemma4UnifiedModel  (Gemma4UnifiedForConditionalGeneration)
              └── Gemma4AssistantModel (Gemma4AssistantForCausalLM)
```

### 2.2 关键转换函数

**`Gemma4Model.set_gguf_parameters()`** (`conversion/gemma.py:655`)：
- 写入 `num_kv_shared_layers`（KV 共享层数）
- 写入 `hidden_size_per_layer_input`（per-layer 嵌入维度）
- 写入 `sliding_window_pattern`（每层是否为 SWA 的 bool 数组）
- 分别写入全局/滑动窗口的 `head_dim`、`n_rot`、`n_head_kv`
- 写入 `expert_intermediate_size`（MoE 专家 FFN 尺寸）
- 处理 `use_double_wide_mlp`：KV 共享层 FFN 宽度 *2

**`Gemma4Model.generate_extra_tensors()`** (`conversion/gemma.py:702`)：
- 为全局 attention 层生成 `ROPE_FREQS`：前 `n_rot_full` 个值=1.0，后 `n_unrot_full` 个值=1e30（使 RoPE 在这些维度上频率→0）

**`Gemma4Model.modify_tensors()`** (`conversion/gemma.py:754`)：
- `router.scale` → `ffn_gate_inp.scale`（路由器的 scale 参数）
- `per_expert_scale` → `ffn_down_exp.scale`（每个专家的缩放因子）

---

## 三、C++ 端模型加载

### 3.1 模型注册

**文件**: `src/llama-model.cpp:141-144`

```cpp
case LLM_ARCH_GEMMA4:
    return new llama_model_gemma4(params);
case LLM_ARCH_GEMMA4_ASSISTANT:
    return new llama_model_gemma4_assistant(params);
```

### 3.2 类定义

**文件**: `src/models/models.h:804-822`

```cpp
struct llama_model_gemma4 : public llama_model_base {
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const llama_model & model;
        const int64_t n_embd_per_layer;

        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

### 3.3 超参数加载 (`load_arch_hparams`)

**文件**: `src/models/gemma4.cpp:3-29`

调用链：`llama_model_load()` → `llama_model::load_hparams()` → `load_arch_hparams(ml)`

关键参数：

| 参数 | GGUF Key | 说明 |
|---|---|---|
| `swa_type` | — | 固定为 `LLAMA_SWA_TYPE_STANDARD` |
| `is_swa_impl` | `attention.sliding_window_pattern` | 每层是否为 SWA |
| `n_layer_kv_from_start` | — | = `n_layer_all - num_kv_shared_layers`，非共享 KV 的层数 |
| `n_swa` | `attention.sliding_window` | SWA 窗口大小 |
| `n_embd_per_layer` | `embedding_length_per_layer_input` | per-layer 嵌入维度 |
| `n_embd_head_k_swa` | `attention.key_length_swa` | SWA 层 head dim |
| `n_embd_head_v_swa` | `attention.value_length_swa` | SWA 层 head dim |
| `rope_freq_base_train_swa` | `rope.freq_base_swa` | SWA 层 RoPE base |
| `n_ff_exp` | `expert_feed_forward_length` | MoE 专家 FFN 尺寸 |
| `f_final_logit_softcapping` | `final_logit_softcapping` | logits 软截断系数 |
| `f_attention_scale` | — | 固定为 1.0（不做 pre-attn scaling） |

### 3.4 张量加载 (`load_arch_tensors`)

**文件**: `src/models/gemma4.cpp:31-131`

**Layer 结构**（每层）：

| 张量组 | 张量 | 说明 |
|---|---|---|
| **Attention** | `attn_norm` | 输入 RMSNorm |
| | `wq`, `wk`, `wv` | Q/K/V 投影（wv 可选，不存在时复用 wk） |
| | `wo` | 输出投影 |
| | `attn_q_norm`, `attn_k_norm` | Q/K 各自的 RMSNorm |
| | `attn_post_norm` | Attention 输出后 RMSNorm |
| | `out_scale` | 层输出缩放（per-layer scalar） |
| | `rope_freqs` | 全局 attention 层的 proportional RoPE 频率因子 |
| **FFN (共享)** | `ffn_norm` | FFN 输入 RMSNorm |
| | `ffn_gate`, `ffn_up`, `ffn_down` | GELU-gated FFN |
| | `ffn_post_norm` | FFN 输出后 RMSNorm |
| **MoE** | `ffn_gate_inp` | MoE 路由器权重 |
| | `ffn_gate_inp_s` | 路由器输入 scale |
| | `ffn_pre_norm_2` | MoE 输入 RMSNorm |
| | `ffn_post_norm_1` | 共享 FFN 输出后 RMSNorm |
| | `ffn_post_norm_2` | MoE 输出后 RMSNorm |
| | `ffn_gate_up_exps` | 合并的 gate+up 专家权重 (或分开的 `ffn_gate_exps` + `ffn_up_exps`) |
| | `ffn_down_exps` | 专家 down 投影 |
| **Per-Layer** | `per_layer_inp_gate` | 门控投影 [n_embd, n_embd_per_layer] |
| | `per_layer_proj` | 投影 [n_embd_per_layer, n_embd] |
| | `per_layer_post_norm` | 投影后 RMSNorm |

**全局张量**：
- `output`（lm_head，不存在时与 `tok_embd` 共享权重）
- `tok_embd`（token embedding）
- `output_norm`（最终 RMSNorm）
- `per_layer_tok_embd` / `per_layer_model_proj` / `per_layer_proj_norm`（per-layer 系统）

---

## 四、计算图构建 (`graph::graph`)

### 4.1 入口

**文件**: `src/models/gemma4.cpp:134-136`

```cpp
std::unique_ptr<llm_graph_context> llama_model_gemma4::build_arch_graph(
    const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
```

### 4.2 图构建主流程

**文件**: `src/models/gemma4.cpp:172-448`

#### 阶段 1: 输入准备

```
inpL = build_inp_embd(model.tok_embd)      // [n_embd, n_tokens]
inpL = ggml_scale(inpL, sqrt(n_embd))       // 仅 token 输入时缩放；多模态嵌入不缩放
inp_pos = build_inp_pos()                   // 位置编码
inp_attn = build_attn_inp_kv_iswa()         // ISWA KV cache 输入
inp_per_layer = build_inp_per_layer()        // [n_embd_per_layer, n_tokens, n_layer]
inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer)
```

#### 阶段 2: 逐层循环 (for il in 0..n_layer)

**2a. Q 投影（所有层共享）**

```
cur = RMSNorm(inpL, attn_norm)
Qcur = wq * cur
Qcur = reshape_3d(Qcur, n_embd_head, n_head, n_tokens)
Qcur = RMSNorm(Qcur, attn_q_norm)
Qcur = RoPE(Qcur, inp_pos, rope_freqs?...)   // 全局层用 rope_freqs，SWA 层不用
```

**2b. KV 处理**

有 KV 的层：
```
Kcur = wk * cur
Vcur = wv * cur  (或 Kcur，如果 wv 不存在)
Kcur = reshape_3d(Kcur, n_embd_head, n_head_kv, n_tokens)
Vcur = reshape_3d(Vcur, n_embd_head, n_head_kv, n_tokens)
Kcur = RMSNorm(Kcur, attn_k_norm)
Vcur = RMSNorm(Vcur, eps)          // 无权重 RMSNorm
Kcur = RoPE(Kcur, ...)
cur = build_attn(inp_attn, wo, Qcur, Kcur, Vcur)
```

无 KV 的层（KV 共享层）：
```
cur = build_attn(inp_attn, wo, Qcur, nullptr, nullptr, nullptr)
// 实际 K/V 从 inp_attn 中复用前面层的 KV Cache
```

**2c. Attention 后处理**

```
// 最后一层可选：仅保留输出 token
if (il == n_layer-1 && inp_out_ids && embeddings_nextn_masked) {
    cur  = ggml_get_rows(cur,  inp_out_ids)
    inpL = ggml_get_rows(inpL, inp_out_ids)
}
cur = RMSNorm(cur, attn_post_norm)
attn_out = cur + inpL     // 残差连接
```

**2d. FFN（MoE 层）**

MoE 层同时有共享 FFN 和专家 FFN：

```
// 共享 FFN
cur_mlp = RMSNorm(attn_out, ffn_norm)
cur_mlp = GELU_Gated_FFN(cur_mlp, ffn_gate, ffn_up, ffn_down)
cur_mlp = RMSNorm(cur_mlp, ffn_post_norm_1)

// MoE 路由
tmp = RMSNorm(attn_out, eps)
tmp = scale(tmp, 1/sqrt(n_embd))
tmp = tmp * ffn_gate_inp_s
logits = ffn_gate_inp * tmp          // [n_expert, n_tokens]

// 专家 FFN
cur_moe = RMSNorm(attn_out, ffn_pre_norm_2)
cur_moe = build_moe_ffn(cur_moe, logits, experts..., SOFTMAX, top-k)
cur_moe = RMSNorm(cur_moe, ffn_post_norm_2)

cur = cur_mlp + cur_moe              // 合并共享 + 专家
```

**2e. FFN（纯 Dense 层）**

```
cur = RMSNorm(attn_out, ffn_norm)
cur = GELU_Gated_FFN(cur, ffn_gate, ffn_up, ffn_down)
```

**2f. FFN 后处理**

```
cur = RMSNorm(cur, ffn_post_norm)
cur = cur + attn_out                 // 残差连接
```

**2g. Per-Layer 嵌入注入**

```
pe_in = cur
cur = per_layer_inp_gate * cur      // [n_embd_per_layer, n_tokens]
cur = GELU(cur)
cur = cur * inp_per_layer[il]       // 逐元素乘
cur = per_layer_proj * cur          // [n_embd, n_tokens]
cur = RMSNorm(cur, per_layer_post_norm)
cur = pe_in + cur                   // 残差连接
```

**2h. 层输出**

```
if (out_scale) cur = cur * out_scale
cur = build_cvec(cur, il)           // 用于调试/监控
inpL = cur
```

#### 阶段 3: 输出

```
cur = RMSNorm(inpL, output_norm)
t_h_nextn = cur                     // 暴露给 MTP 草稿模型
res->t_embd = cur (after get_rows if needed)

cur = output * cur                  // lm_head
if (final_logit_softcapping) {
    cur = tanh(cur / cap) * cap
}

// suppress_tokens 抑制
if (!vocab.get_suppress_tokens().empty()) {
    cur = cur + logits_bias          // 将抑制 token 的 logits 设为 -inf
}

res->t_logits = cur
ggml_build_forward_expand(gf, cur)
```

### 4.3 Per-Layer 输入计算

**`build_inp_per_layer()`** (`gemma4.cpp:450-470`)：

```
// Token 路径
inp_per_layer = per_layer_tok_embd[tokens]  // [n_embd_per_layer, n_layer, n_tokens]
inp_per_layer = scale(inp_per_layer, sqrt(n_embd_per_layer))

// 多模态路径（无 token 输入时）
inp_per_layer = per_layer_tok_embd[0]       // padding token
inp_per_layer = reshape(inp_per_layer, n_embd_per_layer, n_layer, 1)
```

**`project_per_layer_inputs()`** (`gemma4.cpp:488-509`)：

```
per_layer_proj = per_layer_model_proj * inp_batch     // [n_embd_per_layer*n_layer, n_tokens]
per_layer_proj = scale(per_layer_proj, 1/sqrt(n_embd))
per_layer_proj = reshape(per_layer_proj, n_embd_per_layer, n_layer, n_tokens)
per_layer_proj = RMSNorm(per_layer_proj, per_layer_proj_norm)

inp_per_layer = inp_per_layer + per_layer_proj
inp_per_layer = scale(inp_per_layer, 1/sqrt(2))

// 最终 permute 为 [n_embd_per_layer, n_tokens, n_layer]
inp_per_layer = permute(inp_per_layer, 0, 2, 1, 3)
```

---

## 五、KV Cache 机制

### 5.1 ISWA (Interleaved Sliding Window Attention)

Gemma4 使用 `llama_kv_cache_iswa`，支持同时存储全局 attention 和 SWA KV。

**初始化** (`src/llama-model.cpp:2198-2230`)：

```cpp
if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
    res = new llama_kv_cache_iswa(...);
}
```

### 5.2 KV 层共享（Layer Reuse）

**文件**: `src/llama-model.cpp:2157-2166`

```cpp
if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
    reuse = [&](uint32_t il) {
        if (il >= (uint32_t)hparams.n_layer_kv_from_start) {
            // 共享层复用前面层的 KV
            return hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
        }
        return -1;  // 不共享
    };
}
```

**KV 共享逻辑**：
- 前 `n_layer_kv_from_start` 层各自持有独立 KV
- 后 `num_kv_shared_layers` 层按 SWA/Global 类型复用前面层的 KV Cache
- SWA 共享层复用 `n_layer_kv_from_start - 2` 层的 KV
- Global 共享层复用 `n_layer_kv_from_start - 1` 层的 KV

### 5.3 Gemma4 Assistant 的 KV 共享

Assistant 模型 (`gemma4-assistant`) 与 target 模型共享 KV Cache：

```cpp
share = [&](int32_t il) {
    if (hparams.is_swa(il)) {
        return llama_model_n_layer(model_other) - 2;
    }
    return llama_model_n_layer(model_other) - 1;
};
```

---

## 六、Gemma4 Assistant（草稿模型）

**文件**: `src/models/gemma4-assistant.cpp`

### 6.1 架构

Gemma4 Assistant 是用于投机解码（speculative decoding）的轻量草稿模型：
- 与 target 共享 KV Cache
- 输入：target embedding **x** + target 隐藏状态 **h**（通过 `nextn_proj_post` 获得）
- 输出：logits + `h_nextn`（回传给 target 作为下一轮的 h）

### 6.2 图构建关键流程

```
// 输入
inp_tokens = token_ids
inp_h = h_nextn_from_target

x  = target_tok_embd[inp_tokens]  // 从 target 模型获取
x  = scale(x, sqrt(n_embd_backbone))
xh = concat(x, inp_h, dim=0)
cur = nextn_proj_pre * xh          // 融合 target embd + h

// 逐层（结构与 Gemma4 类似但简化：无 MoE、无 per-layer embd）
for il in 0..n_layer_nextn:
    Qcur = wq * RMSNorm(inpL, attn_norm)
    Qcur = reshape_3d + RMSNorm(Q_norm) + RoPE
    cur = build_attn(inp_attn, wo, Qcur, nullptr, nullptr)  // 复用 target KV
    cur = RMSNorm(cur + inpL, attn_post_norm)
    cur = GELU_Gated_FFN(RMSNorm(cur, ffn_norm), ...)
    cur = RMSNorm(cur, ffn_post_norm)
    cur = cur + attn_out
    cur = cur * out_scale
    inpL = cur

// 输出
cur = RMSNorm(cur, output_norm)
logits = output * cur
h_nextn = nextn_proj_post * cur    // 传回 target
```

---

## 七、推理执行流程

### 7.1 顶层调用

```
llama_decode(ctx, batch)
  └── llama_context::decode(batch)
        ├── balloc->init()               // 初始化批次分配器
        ├── memory_update(false)          // 处理 KV Cache 移位/复制
        ├── memory->init_batch()          // 获取 memory context
        ├── while (mctx->next()):
        │     └── process_ubatch(ubatch, gtype, mctx, status)
        │           ├── graph_params()    // 计算图参数（决定复用）
        │           ├── model.build_graph(gparams)  // 构建计算图
        │           │     └── llama_model_gemma4::build_arch_graph()
        │           │           └── new graph(model, params)  // gemma4.cpp graph::graph
        │           ├── ggml_backend_sched_alloc_graph()
        │           ├── res->set_inputs(&ubatch)  // 填入输入数据
        │           └── graph_compute()           // 执行计算
        └── 提取 logits / embeddings / h_nextn
```

### 7.2 图复用

`process_ubatch` 中通过 `res->can_reuse(gparams)` 判断是否可以复用上一次的计算图：
- 如果 ubatch 结构相同（n_tokens、n_seq 等），直接 `set_inputs` 后重新计算
- 否则重建计算图并重新分配后端内存

---

## 八、Tokenizer

### 8.1 类型

Gemma4 使用 BPE tokenizer，`tokenizer_model = "gemma4"` (`llama-vocab.cpp:2064`)。

### 8.2 预分词器

`pre_type = LLAMA_VOCAB_PRE_TYPE_GEMMA4` (`llama-vocab.cpp:2180`)：
- `escape_whitespaces = true`（空格用 `▁` U+2581 代替）
- 采用 SPM 风格的 BPE 预处理

### 8.3 特殊处理

**Visible tokens**: `<|channel>`, `<|tool_call>`, `<|tool_response>`, `<|"|>` 等被标记为 `USER_DEFINED` 以确保 chat parser 可见。

---

## 九、关键函数速查表

| 函数 | 文件 | 行号 | 职责 |
|---|---|---|---|
| `llama_model_gemma4::load_arch_hparams` | `gemma4.cpp` | 3 | 加载架构超参数 |
| `llama_model_gemma4::load_arch_tensors` | `gemma4.cpp` | 31 | 加载模型权重张量 |
| `llama_model_gemma4::build_arch_graph` | `gemma4.cpp` | 134 | 图构建入口 |
| `llama_model_gemma4::graph::graph` | `gemma4.cpp` | 172 | **核心：构建完整前向图** |
| `build_inp_per_layer` | `gemma4.cpp` | 450 | 构建 per-layer 输入 |
| `project_per_layer_inputs` | `gemma4.cpp` | 488 | 融合 batch + per-layer 输入 |
| `llama_model_gemma4_assistant::graph::graph` | `gemma4-assistant.cpp` | 95 | 草稿模型图构建 |
| `llama_context::decode` | `llama-context.cpp` | 1647 | 解码主循环 |
| `llama_context::process_ubatch` | `llama-context.cpp` | 1271 | 单 ubatch 处理 |
| `build_attn_inp_kv_iswa` | `llama-graph.cpp` | 3044 | 创建 ISWA KV 输入 |
| `llm_graph_input_logits_bias` | `gemma4.cpp` | 146 | suppress_tokens 抑制逻辑 |
| `Gemma4Model.set_gguf_parameters` | `conversion/gemma.py` | 655 | Python 端参数写入 |
| `Gemma4Model.generate_extra_tensors` | `conversion/gemma.py` | 702 | 生成 ROPE_FREQS |

---

## 十、与 Gemma3 的关键差异

| 特性 | Gemma3 | Gemma3N | Gemma4 |
|---|---|---|---|
| SWA 模式 | 5:1 交替 | 4:1 交替 | 由 `layer_types` 定义 |
| KV 共享 | 无 | 后 N 层复用 | 后 N 层复用 |
| MoE | 无 | 无 | 有（共享 FFN + 专家 FFN） |
| AltUp/Laurel | 无 | 有 | 无 |
| Per-layer embd | 无 | 有（与 AltUp 耦合） | 有（独立） |
| QK Norm | 有 | 有 | 有 |
| V Norm | 无 | 有 | 有 |
| Proportional RoPE | 无 | 无 | 全局层使用 |
| Attention scale | `1/sqrt(head_dim)` | 1.0 | 1.0 |
| Double Wide MLP | 无 | 无 | KV 共享层 FFN 加倍 |
| `suppress_tokens` | 无 | 无 | Gemma4Unified 有 |
| Final softcapping | 可选 | 总是有 | 可选 |


## Gemma4 独有特性

1. **SWA + KV 共享**：通过 `llama_kv_cache_iswa` 实现，后 N 层复用前面层的 KV Cache（`n_layer_kv_from_start = n_layer - num_kv_shared_layers`）

2. **MoE + 共享 FFN 双路径**：MoE 层同时计算共享 Dense FFN（`ffn_norm → ffn_gate/up/down → ffn_post_norm_1`）和专家 FFN（`ffn_pre_norm_2 → MoE routing → experts → ffn_post_norm_2`），最后 `cur = cur_mlp + cur_moe`

3. **Per-Layer Embeddings**：`project_per_layer_inputs()` 将 batch 隐藏状态投影到 per-layer 空间，与 token per-layer embd 相加后注入每层

4. **Proportional RoPE**：全局 attention 层使用 `rope_freqs`（部分维度频率因子=1e30 使其不旋转），SWA 层使用标准 RoPE

5. **Q/K/V Norm**：Q/K 使用有权重 RMSNorm，V 使用无权重 RMSNorm

6. **Suppress Tokens**：Gemma4Unified 通过 `llm_graph_input_logits_bias` 在 logits 上对 `<image|>`、`<audio|>` 等 token 施加 `-inf` 偏置


