# Gemma 4 技术全解

> 本文档涵盖 zllama.zig 中 Gemma 4 全模态（文本 + 视觉 + 音频）的实现细节，
> 与 llama.cpp 参考实现对齐，阐述架构设计、图构建、KV Cache 机制及多媒体集成。

## 相关文档索引

| 文档 | 内容 |
|------|------|
| [gemma4a.md](gemma4a.md) | llama.cpp Gemma4 语言模型推理全流程（文本侧） |
| [gemma4v.md](gemma4v.md) | llama.cpp Gemma4V 视觉编码器图形推理过程 |
| 本文档 | **zllama.zig 中 Gemma 4 完整实现（文本 + 视觉 + 音频 + 多模态编排）** |

---

## 一、模型家族概览

Gemma 4 是 Google 的多模态大语言模型系列，在 zllama.zig 中支持以下子架构：

| GGUF 标识 | 说明 |
|-----------|------|
| `gemma4` | 纯文本 LLM（支持 SWA + MoE + per-layer embedding） |
| `gemma4v` | 视觉模型（文本 LLM + ViT 编码器） |
| `gemma4uv` | Unified Vision（简化视觉，无 ViT 编码器，patch_size 增大补偿） |
| `gemma4a` | 音频模型（文本 LLM + 音频编码器，Mel 频谱输入） |
| `gemma4ua` | Unified Audio（简化音频，直接 PCM 分帧，无 FFT） |

---

## 二、文本模型实现

### 2.1 源文件组织

| 文件 | 职责 | 行数 |
|------|------|------|
| `src/models/gemma4.zig` | 模型入口、参数/权重结构体、vtable 导出 | ~380 |
| `src/models/gemma4_graph.zig` | 计算图构建（核心！） | ~612 |
| `src/models/gemma4_loader.zig` | 超参数解析与权重加载 | ~290 |

### 2.2 超参数 (`Gemma4Params`)

```zig
pub const Gemma4Params = struct {
    base: model.ModelParams,          // 标准参数：n_vocab, n_embd, n_head, n_layer ...
    n_swa: u32,                       // 滑动窗口大小
    n_kv_shared_layers: u32,          // 共享 KV 的层数
    n_layer_kv_from_start: u32,       // 前面独立 KV 的层数
    is_swa_layer: std.ArrayList(bool), // 每层是否为 SWA（bool 数组）
    rope_freq_base_swa: f32,          // SWA 层 RoPE 基础频率
    rope_dim_swa: u32,                // SWA 层 RoPE 维度
    f_attention_scale: f32,           // 固定 1.0（Gemma 4 不做 pre-attn scale）
    f_final_logit_softcapping: f32,   // logits 软截断系数
    attn_logit_softcapping: f32,      // attention logit 软截断系数（默认 50.0）
    n_embd_per_layer: u32,            // per-layer embedding 维度
};
```

**关键设计差异（vs llama.cpp）**：
- `head_dim` 分层：全文注意力层和 SWA 层可能使用不同的 `key_length`/`value_length`
- `rope_freqs`：全文注意力层使用 proportional RoPE（部分维度冻结），SWA 层无此特性
- `n_layer_kv_from_start = n_layer - n_kv_shared_layers`：前 N 层独立 KV，后 N 层共享

### 2.3 权重结构体

```zig
pub const LayerWeights = struct {
    // Attention
    attn_norm_weight: *ggml.Tensor,        // pre-attn RMSNorm
    attn_q_weight: *ggml.Tensor,           // Q 投影
    attn_k_weight: ?*ggml.Tensor,          // K 投影（共享层为 null）
    attn_v_weight: ?*ggml.Tensor,          // V 投影（可选，不存在时复用 K）
    attn_output_weight: *ggml.Tensor,      // 输出投影
    attn_q_norm_weight: *ggml.Tensor,      // Q RMSNorm 权重
    attn_k_norm_weight: *ggml.Tensor,      // K RMSNorm 权重
    attn_post_norm_weight: *ggml.Tensor,   // Attn 后 RMSNorm
    out_scale: ?*ggml.Tensor,             // 层输出缩放
    rope_freqs: ?*ggml.Tensor,            // proportional RoPE 频率因子

    // FFN
    ffn_norm_weight: *ggml.Tensor,        // FFN 输入 RMSNorm
    ffn_gate_weight: *ggml.Tensor,        // GELU gate
    ffn_up_weight: *ggml.Tensor,          // up 投影
    ffn_down_weight: *ggml.Tensor,        // down 投影
    ffn_post_norm_weight: *ggml.Tensor,   // FFN 后 RMSNorm

    // Per-layer embedding
    per_layer_inp_gate: ?*ggml.Tensor,
    per_layer_proj: ?*ggml.Tensor,
    per_layer_post_norm: ?*ggml.Tensor,

    has_kv: bool,                         // 该层是否有独立 KV
};
```

### 2.4 计算图构建 (`Gemma4Graph`)

图构建入口在 `gemma4.zig` 的 `buildGraph`，实际调度到 `gemma4_graph.zig`。支持三种路径：

| 路径 | 函数 | 场景 |
|------|------|------|
| 纯文本 | `Gemma4Graph.build()` | 常规 text-only 预填充/解码 |
| Embedding 覆盖 | `Gemma4Graph.buildWithEmbd()` | 多模态三阶段 prefill（prefix → media → suffix） |
| 纯媒体 | `Gemma4Graph.buildMediaOnly()` | 多模态 media pass（非因果注意力） |

#### 核心流程 (`Gemma4Graph.build`)

```
1. 输入嵌入
   cur = tok_embd[tokens]           // [n_embd, n_tokens]
   cur = scale(cur, sqrt(n_embd))   // 仅 token 输入时缩放（多模态不缩放）

2. Per-Layer 输入 (如果 n_embd_per_layer > 0)
   inp_per_layer = buildPerLayerInputs(cur)
   // 多模态路径: 使用 padding token (ID=0) 的 per-layer embedding

3. 逐层循环 (for il in 0..n_layer)
   ├── RMSNorm(attn_norm) → Q 投影
   ├── Q = reshape_3d(Q, head_dim, n_head, n_tokens)
   ├── Q = Q RMSNorm(attn_q_norm)
   ├── Q = RoPE(Q, pos, rope_freqs?)  // 全文注意力层用 rope_freqs
   │
   ├── [KV 层] K = wk * cur, V = wv * cur (或 K)
   │   ├── K/V = reshape_3d → K RMSNorm(attn_k_norm) → V 无权重 RMSNorm
   │   ├── K = RoPE(K, pos, rope_freqs?)
   │   ├── Cache.setKv → Cache.getKView/VView
   │   └── Attention(Q, K_cache, V_cache, causal/swa)
   │
   ├── [共享层] 复用前面 KV 层的缓存
   │   ├── K/V = Cache.getKView/VView(kv_layer_idx)
   │   └── Attention(Q, K_cache, V_cache, causal/swa)
   │
   ├── Attn 后处理
   │   ├── RMSNorm(attn_post_norm)
   │   └── cur = cur + inpL  (残差)
   │
   ├── FFN
   │   ├── RMSNorm(ffn_norm) → GeGLU(ffn_gate, ffn_up, ffn_down)
   │   ├── RMSNorm(ffn_post_norm)
   │   └── cur = cur + attn_out  (残差)
   │
   ├── [per-layer 注入] (如果 n_embd_per_layer > 0)
   │   ├── gate = GeLU(per_layer_inp_gate * cur)
   │   ├── cur = gate * inp_per_layer[il]  (element-wise)
   │   ├── cur = per_layer_proj * cur
   │   ├── RMSNorm(per_layer_post_norm)
   │   └── cur = pe_in + cur  (残差)
   │
   ├── [out_scale] cur = cur * out_scale (如果存在)
   └── inpL = cur

4. 输出
   cur = RMSNorm(output_norm)
   logits = output * cur
   [softcapping] tanh(cur/cap) * cap
```

---

## 三、视觉编码器 (Gemma4V)

### 3.1 源文件组织

| 文件 | 职责 |
|------|------|
| `src/mtmd/graph/models/gemma4v.zig` | ViT 计算图构建（2D RoPE + VisionPooler + Embedder） |
| `src/mtmd/graph/models/gemma4uv.zig` | Unified Vision（简化版，无 ViT） |
| `src/mtmd/vision/encoder.zig` | VisionEncoder 门面，通过 Backend 分发 |
| `src/mtmd/vision/loader.zig` | 视觉编码器权重加载 |
| `src/mtmd/vision/config.zig` | VisionEncoderParams |

### 3.2 数据流

```
原始图像 [W, H, 3]
  ↓ preprocess.resizeAndNormalize
  ↓ 归一化: (pixels/255 - mean)/std  →  [-1, 1]
  ↓ ggml_conv_2d(patch_embeddings_0, kernel=patch_size, stride=patch_size)
[n_embd, n_patches_w, n_patches_h, B]
  ↓ reshape + transpose
[n_embd, n_patches, B]
  ↓ + learned pos_embd_x (列坐标) + pos_embd_y (行坐标)
  ↓
  ↓ build_vit (N layers)
  │   每层: RMSNorm → QKV → 2D RoPE(Q,K) → V Norm → SDPA → FFN
  │   2D RoPE: Q/K 前半维度用 pos_x, 后半用 pos_y (NEOX 排序)
  ↓
[n_embd, n_patches, B]
  ↓ Gemma4VisionPooler: ggml_pool_2d(kernel=3, stride=3, AVG)
[n_embd, n_patches/9, B]
  ↓ * sqrt(n_embd)
  ↓ (x - std_bias) * std_scale  (如果存在)
  ↓ RMSNorm  →  mm_input_proj_w (线性投影)
[n_mmproj_embd, n_patches/9, B]  ← 最终图像嵌入
```

### 3.3 关键参数

| 参数 | 来源 | 默认值 | 说明 |
|------|------|--------|------|
| `patch_size` | GGUF `patch_size` | 14 | 卷积核/步长 |
| `n_merge` | GGUF `n_merge` | 3 | 池化核大小 |
| `rope_theta` | GGUF 或默认 | 100.0 | 2D RoPE 基础频率 |
| `image_min_pixels` | GGUF | 256*256 | 最小分辨率 |
| `image_max_pixels` | GGUF | 依赖模型 | 最大分辨率 |

### 3.4 Gemma4V vs Gemma4UV

| 特性 | Gemma4V | Gemma4UV |
|------|---------|----------|
| Patch 提取 | `ggml_conv_2d` | `ggml_im2col` + LN + `mul_mat` + LN |
| ViT Transformer | 有 (完整 Transformer) | 无 (跳过) |
| VisionPooler | 有 (avg pool, kernel=3) | 无（patch_size 已扩大 3x） |
| 2D RoPE | 有 (NEOX 排序) | 无 |
| std_bias/scale | 支持 | 不支持 |
| Clamp 线性层 | 支持 | 不支持 |

---

## 四、音频编码器 (Gemma4A)

### 4.1 源文件组织

| 文件 | 职责 |
|------|------|
| `src/mtmd/graph/models/gemma4a.zig` | 音频编码器图构建（Mel 频谱 → Transformer → 投影） |
| `src/mtmd/graph/models/gemma4ua.zig` | Unified Audio（PCM 分帧，无 FFT） |
| `src/mtmd/audio/encoder.zig` | AudioEncoder 门面 |
| `src/mtmd/audio/pipeline.zig` | 音频预处理 pipeline（Mel 频谱计算） |

### 4.2 数据流

```
音频文件 [WAV]
  ↓ mtmd.audio_mod.loadWav → PCM 样本
  ↓
  ├── [Gemma4A]  computeMelSpectrogram(samples, sr)
  │    → [n_mel_bins, n_frames]
  │    → melToTensor (转 ggml)
  │    → buildGraph: mel → encoder → projection
  │    → [n_mmproj_embd, n_audio_tokens]
  │
  └── [Gemma4UA] processRawWaveform(samples, frame_size)
       → 直接分帧（无 FFT）
       → buildGraph: frames → encoder → projection
       → [n_mmproj_embd, n_audio_tokens]
```

### 4.3 关键参数

| 参数 | 说明 |
|------|------|
| `n_mel_bins` | Mel 频谱 bin 数（默认 80） |
| `sample_rate` | 目标采样率（默认 16000 Hz） |
| `fft_n` | FFT 点数（默认 400） |
| `hop_length` | 帧移（默认 160） |
| `frame_size` (UA) | PCM 分帧大小 = `mm_input_proj_w.ne[0]` |

---

## 五、多模态集成与三阶段 Prefill

### 5.1 编排入口

```
main.zig / CLI
  ↓
core/multimodal.zig
  ├── generateWithImage()     // 图像推理入口
  └── generateWithAudio()     // 音频推理入口
       ↓
mtmd/mod.zig (MultiModalManager)
  ├── encodeMedia()           // 编码媒体 → 嵌入张量
  └── MtmdContext             // 媒体标记管理
       ↓
core/prefill.zig (threeStagePrefill)
  ├── Pass 1: textPass()      // 前缀文本（因果注意力）
  ├── Pass 2: mediaPass()     // 媒体 token（非因果注意力）
  └── Pass 3: textPass()      // 后缀文本（因果注意力）
       ↓
core/decode.zig (runDecodeLoop)
  └── 增量解码循环
```

### 5.2 三阶段 Prefill 详解

参考 llama.cpp `mtmd-helper.cpp` 的 `mtmd_helper_decode_image_chunk`：

```
初始状态: KV Cache 为空

Pass 1 — 文本前缀 (causal=true)
  位置: [0, prefix_len)
  操作: llama_decode(prefix_tokens)
  结果: KV Cache 中保存了前缀的 K/V

Pass 2 — 媒体 token (causal=false)
  位置: [prefix_len, prefix_len + media_n_pos)
  操作: 以 chunks (默认 256 token/chunk) 处理
  每个 chunk:
    - buildMediaOnly: 用视觉/音频嵌入替代 token embedding
    - non-causal attention: 让所有媒体 token 互相可见
    - setKv + getKView: 写入缓存后读取全部历史
  结果: KV Cache 中追加了媒体 token 的 K/V

Pass 3 — 文本后缀 (causal=true)
  位置: [prefix_len + media_n_pos, prefix_len + media_n_pos + suffix_len)
  操作: llama_decode(suffix_tokens)
  结果: 采样 logits 用于生成第一个 token

Decode Loop
  从 Pass 3 的 logits 采样 → 增量解码
```

### 5.3 非因果注意力修复

**关键修复（P0）**：在 `mediaPass` 分块处理中，KV 层的非因果路径需要：

1. `setKv(cache)` — 写入当前 chunk 的 K/V
2. `getKView(cache)` / `getVView(cache)` — **读取完整缓存视图**（包含之前所有 chunk 的 token）

这确保每个 chunk 中的 token 可以关注到之前 chunk 中已处理的媒体 token，与 llama.cpp 的 `build_attn` 行为一致（llama.cpp 总是通过缓存视图进行注意力计算，无论因果模式）。

### 5.4 位置编码规则

- **文本 Pass (causal=true)**：使用标准 RoPE，`start_pos` 为当前 KV Cache 长度
- **媒体 Pass (causal=false)**：使用标准 RoPE，`start_pos = prefix_len + chunk_offset`
- **M-RoPE**（Qwen 3 VL 等）：使用 `buildMultiPositionTensor`，每 token 有 4 个位置维度

---

## 六、KV Cache 机制

### 6.1 内存布局

```
K Cache: [per_layer_max_seq_len, n_embd_head_k, n_kv_layers]
V Cache: [per_layer_max_seq_len, n_embd_head_v, n_kv_layers]
```

- 预分配固定大小（`max_seq_len * n_layer`）
- 增量写入使用 `ggml_view_*` 切片，**禁止物理复制**
- SWA 层使用单独的 `max_seq_len_swa`（= `n_swa`），通过 `view2d` 实现滚动窗口

### 6.2 层间 KV 共享

```
前 n_layer_kv_from_start 层: 独立 KV Cache
后 n_kv_shared_layers 层:   复用前面层的 KV

共享规则 (findKVLayer):
  当前层是 SWA 且不是最后一层 → 复用 n_layer_kv_from_start - 2
  当前层是 SWA 且是最后一层     → 复用 n_layer_kv_from_start - 1
  当前层是非 SWA 且不是最后一层 → 复用 n_layer_kv_from_start - 1
  当前层是非 SWA 且是最后一层   → 复用上一层
```

### 6.3 SWA 缓存截断

SWA 层使用滑动窗口注意力，缓存视图被截断为 window_size：

```
缓存总长度 = currentLen()
SWA 视图长度 = min(currentLen, n_swa)
cache_start_abs = currentLen - SWA 视图长度  (当 currentLen > n_swa 时)
```

`attention.scaledDotProductAttention` 通过 `cache_start_abs` 参数构造正确的滑动窗口掩码。

### 6.4 Per-Layer Max SeqLen

通过 `getPerLayerMaxSeqLenAdapter` 为 Inferless (Gallocr) 提供每层的最大序列长度：

| 层类型 | max_seq_len |
|--------|-------------|
| SWA 层 + 独立 KV | `n_swa` |
| 全文注意力层 + 独立 KV | `max_seq_len` |
| 共享层（无 KV） | 1 |

---

## 七、MoE (Mixture of Experts)

**当前状态**：MoE 支持在 Zig 代码中**暂未实现**。

llama.cpp 中 MoE 层的结构：

```
MoE 层 = 共享 Dense FFN + 专家 FFN

共享 FFN:
  RMSNorm(ffn_norm) → GeGLU(ffn_gate, ffn_up, ffn_down) → RMSNorm(ffn_post_norm_1)

路由计算:
  tmp = RMSNorm(attn_out, eps)
  tmp = scale(tmp, 1/sqrt(n_embd))
  tmp = tmp * ffn_gate_inp_s
  logits = ffn_gate_inp * tmp  → [n_expert, n_tokens]

专家 FFN:
  RMSNorm(ffn_pre_norm_2)
  → build_moe_ffn(cur, logits, experts..., SOFTMAX, top-k)
  → RMSNorm(ffn_post_norm_2)

最终: cur = cur_mlp + cur_moe
```

---

## 八、Per-Layer Embedding 注入

Per-layer embedding 是 Gemma 4 的核心特性，为每层注入层特定信息。

### 8.1 计算流程

```
buildPerLayerInputs():
  ┌────────────────────────────────────────────────────────┐
  │ Token 路径 (ubatch.token != null):                      │
  │   inp = per_layer_token_embd[tokens]                   │
  │   inp = reshape(inp, n_embd_pl, n_layer, n_tokens)     │
  │   inp = scale(inp, sqrt(n_embd_pl))                    │
  │                                                        │
  │ 多模态路径 (ubatch.token == null):                      │
  │   padding = per_layer_token_embd[0]  (ID=0 的嵌入)     │
  │   inp = reshape(padding, n_embd_pl, n_layer, 1)        │
  │   inp = scale(inp, sqrt(n_embd_pl))                    │
  └────────────────────────────────────────────────────────┘

  // 与 batch 隐藏状态融合
  proj = per_layer_model_proj * inpL
  proj = scale(proj, 1/sqrt(n_embd))
  proj = reshape(proj, n_embd_pl, n_layer, n_tokens)
  proj = RMSNorm(proj, per_layer_proj_norm)

  inp_per_layer = inp + proj
  inp_per_layer = scale(inp_per_layer, 1/sqrt(2))

  // 最终布局: [n_embd_pl, n_tokens, n_layer]
  inp_per_layer = permute(inp_per_layer, 0, 2, 1, 3)
```

### 8.2 层内注入

```zig
fn buildPerLayerInjection(ctx, p, layer, cur, inp_pl, il, n_tokens_i64) {
    pe_in = cur
    // gate: [n_embd, n_embd_pl] * cur → [n_embd_pl, n_tokens]
    cur = gelu(layer.per_layer_inp_gate * cur)
    // element-wise multiply with this layer's input
    cur = cur * inp_pl[il]  // [n_embd_pl, n_tokens]
    // project back
    cur = layer.per_layer_proj * cur  // [n_embd, n_tokens]
    cur = RMSNorm(cur, layer.per_layer_post_norm)
    cur = pe_in + cur  // residual
    return cur
}
```

---

## 九、与 llama.cpp 的关键差异

| 方面 | llama.cpp | zllama.zig |
|------|-----------|------------|
| 语言 | C++ | Zig 0.16.0 |
| 内存管理 | shared_ptr / unique_ptr | 显式 Allocator + defer |
| ggml 调用 | 直接 C API | 通过 `src/ggml/` 模块封装 |
| 计算图 | `ggml_cgraph` + `ggml_gallocr` | 同，但 Gallocr 跨 Pass 复用 |
| 多态 | 虚函数继承 | VTable 结构体 (ModelVTable / VisionEncoderBackend) |
| MoE | 已实现 | 待实现 |
| Gemma4 Assistant | 已实现 | 待实现 |
| KV Cache 类型 | `llama_kv_cache_iswa` | `kv_cache.KVCache`（通过 per-layer max_seq_len 区分 SWA） |
| 位置编码 | `build_inp_pos()` C++ 重载 | `rope.buildPositionTensor()` |
| Per-layer embedding | `project_per_layer_inputs` | `buildPerLayerInputs` |

---

## 十、关键函数速查

### 文本侧

| 函数 | 文件 | 职责 |
|------|------|------|
| `gemma4_loader.parseParams()` | `gemma4_loader.zig` | 解析 GGUF 超参数 |
| `gemma4_loader.loadWeights()` | `gemma4_loader.zig` | 加载所有权重张量 |
| `Gemma4Graph.build()` | `gemma4_graph.zig` | 纯文本计算图构建 |
| `Gemma4Graph.buildWithEmbd()` | `gemma4_graph.zig` | 带 embedding 覆盖的计算图 |
| `Gemma4Graph.buildMediaOnly()` | `gemma4_graph.zig` | 纯媒体非因果计算图 |
| `Gemma4Graph.buildLayer()` | `gemma4_graph.zig` | 单层 Transformer |
| `Gemma4Graph.buildAttention()` | `gemma4_graph.zig` | 注意力子图（KV/共享层/因果/非因果） |
| `Gemma4Graph.buildPerLayerInputs()` | `gemma4_graph.zig` | per-layer 输入计算 |
| `buildPerLayerInjection()` | `gemma4_graph.zig` | per-layer 嵌入注入 |
| `gegluFFN()` | `gemma4_graph.zig` | GeGLU 前馈网络 |
| `findKVLayer()` | `gemma4_graph.zig` | KV 共享层映射 |

### 多模态侧

| 函数 | 文件 | 职责 |
|------|------|------|
| `generateWithImage()` | `core/multimodal.zig` | 图像推理入口 |
| `generateWithAudio()` | `core/multimodal.zig` | 音频推理入口 |
| `threeStagePrefill()` | `core/prefill.zig` | 三阶段预填充编排 |
| `textPass()` | `core/prefill.zig` | 文本 Pass（因果） |
| `mediaPass()` | `core/prefill.zig` | 媒体 Pass（非因果） |
| `MultiModalManager.encodeMedia()` | `mtmd/mod.zig` | 媒体编码门面 |
| `VisionEncoder.encode()` | `mtmd/vision/encoder.zig` | 视觉编码 |
| `AudioEncoder.encode()` | `mtmd/audio/encoder.zig` | 音频编码 |
| `clip_graph_gemma4v.build()` | `mtmd/graph/models/gemma4v.zig` | ViT 图构建 |

---

## 十一、扩展指南

### 新增 Gemma4 视觉变体

1. 在 `src/mtmd/graph/models/` 下创建新文件（如 `gemma4_foveated.zig`）
2. 实现 `pub fn buildGraph(ctx, gf, w, p, image_tensor)`
3. 导出 `pub const backend = VisionEncoderBackend{ .name = "...", .buildGraph = buildGraph, ... }`
4. 在 `src/mtmd/graph/mod.zig` 的 `model_graphs` 中注册
5. 在 `src/mtmd/vision/mod.zig` 的 `getBackend()` 中注册
6. 在 `src/mtmd/mod.zig` 的 `detectFromGGUF()` 中添加 GGUF key 匹配

### 新增 Gemma4 音频变体

流程同上，但使用 `AudioEncoderBackend`，注册到 `src/mtmd/audio/mod.zig`。

### 实现 MoE

1. 在 `gemma4_loader.zig` 中加载 MoE 相关权重（`ffn_gate_inp`, `ffn_gate_inp_s`, `ffn_pre_norm_2`, `ffn_post_norm_1/2`, 专家权重）
2. 在 `gemma4_graph.zig` 的 `buildLayer` 中检测 `is_moe_layer` 并分派到 MoE 路径
3. 实现 `buildMoeFFN` 函数（路由计算 + 专家前向 + top-k 选择）

---

## 十二、调试指南

### 对比 logits（文本）

```bash
# 使用 compare_logits 工具对比第一个 token logits
zig build compare_logits -- --model tinyllama.gguf
# 期望: NMSE < 1e-5 或余弦相似度 > 0.999
```

### 对比视觉嵌入

```bash
zig build compare_mtmd_vision -- --model gemma4v.gguf --image test.png
```

### 关键验证点

1. **无 KV 缓存测试**：`cache == null` 时，模型应直接使用 K/V 张量（不通过缓存视图）
2. **非因果注意力**：`causal == false` 时，attention mask 应为全零（不使用 diag_mask_inf）
3. **Per-layer embedding 空值**：当 `n_embd_per_layer == 0` 时，跳過 per-layer 注入
4. **Embedding 维度匹配**：视觉/音频编码器输出维度必须与 LLM `n_embd` 一致（Gemma4V 除外，允许适配）
