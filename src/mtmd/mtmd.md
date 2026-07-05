# mtmd 多模态处理 — 差距分析与对齐指南

> 本文档分析 `src/mtmd/` 与参考实现 `deps/llama.cpp/tools/mtmd/` 之间的架构差异，
> 聚焦 **qwen3vl、gemma4a、gemma4v** 三个模型，涵盖架构、内存管理、推理流程。
> 目标：指导后续的功能对齐、正确实现与代码完善。

---

## 目录

1. [参考实现架构总览](#1-参考实现架构总览)
2. [当前 Zig 实现状态](#2-当前-zig-实现状态)
3. [差距矩阵（按功能域）](#3-差距矩阵按功能域)
4. [模型专项分析](#4-模型专项分析)
   - [4.1 qwen3vl](#41-qwen3vl)
   - [4.2 gemma4a](#42-gemma4a)
   - [4.3 gemma4v](#43-gemma4v)
5. [内存管理差距](#5-内存管理差距)
6. [推理流程差距](#6-推理流程差距)
7. [优先级路线图](#7-优先级路线图)

---

## 1. 参考实现架构总览

参考实现 (C++) 整体分为 **四层**：

```
┌─────────────────────────────────────────────────────────────┐
│  mtmd.h / mtmd.cpp          ← 公开 C API + 核心编排          │
│  ├─ mtmd_context             生命周期、编码器引用、标记管理    │
│  ├─ mtmd_tokenizer           文本+媒体标记分片、tokenize     │
│  ├─ mtmd_encode_chunk        分发到 clip_image_batch_encode  │
│  └─ mtmd_batch              批量编码（合并 batch_f32）       │
├─────────────────────────────────────────────────────────────┤
│  clip.h / clip.cpp           ← 编码器核心（设备无关）         │
│  ├─ clip_ctx                 后端、scheduler、graph          │
│  ├─ clip_image_batch_encode  预处理→图构建→compute→后处理    │
│  └─ clip_graph::build()      虚函数 → 模型派发               │
├─────────────────────────────────────────────────────────────┤
│  clip-graph.h (clip_graph)   ← ViT 图构建基类                │
│  clip-model.h (clip_model)   ← 权重/超参数结构               │
│  clip-impl.h                 ← 张量名常量/工具函数            │
├─────────────────────────────────────────────────────────────┤
│  models/models.h + *.cpp     ← 模型特定图构建                 │
│  │  clip_graph_qwen3vl        继承 clip_graph_qwen2vl        │
│  │  clip_graph_gemma4a        Conformer + chunked attn       │
│  │  clip_graph_gemma4v        ViT + 2D RoPE + pool           │
│  └─ clip_graph_*              其他模型...                    │
├─────────────────────────────────────────────────────────────┤
│  mtmd-image.h/cpp            ← 图像预处理                    │
│  mtmd-audio.h/cpp            ← 音频预处理 (Mel)              │
│  mtmd-helper.h/cpp           ← eval 辅助、解码器位置          │
└─────────────────────────────────────────────────────────────┘
```

**关键设计原则**：

1. **虚表多态** (`clip_graph` 虚基类) — 每个模型继承并重写 `build()`
2. **batch_f32 容器** — `clip_image_f32_batch` 承载已预处理图像/音频，在 encode 时传递到图构建
3. **统一的 `clip_image_batch_encode`** — 所有模态共用同一个 encode 入口，内部 dispatch 到对应 `clip_graph` 子类
4. **ggml backend + scheduler** — 管理 GPU/CPU 后端、计算图执行、张量分配
5. **`build_mm()` 虚函数** — 允许模型 hook 矩阵乘法（如 gemma4 的 clamp）

---

## 2. 当前 Zig 实现状态

### 2.1 已完成部分 ✅

| 层次 | 文件 | 状态 |
|------|------|------|
| **计算图构建块** | `src/mtmd/graph/norm.zig`, `ffn.zig`, `attn.zig`, `patch.zig`, `rope.zig`, `merge.zig`, `stack.zig`, `mm.zig`, `vit.zig` | ✅ 已完成（含测试） |
| **模型图构建** | `graph/models/gemma4v.zig`, `gemma4a.zig`, `gemma4uv.zig`, `qwen2vl.zig`, `qwen3vl.zig` | ✅ 图结构完成 |
| **类型系统** | `graph/types.zig` — `VisionEncoderWeights`, `ViTLayerWeights`, `VisionHParams` 等 | ✅ 与 clip-model.h 对齐 |
| **Backend 接口** | `graph/mod.zig` — `VisionEncoderBackend`, `AudioEncoderBackend` | ✅ 已定义，模型注册完毕 |
| **编码器框架** | `vision/encoder.zig`, `audio/encoder.zig` | ✅ 基本框架完成 |
| **检测分发** | `mod.zig` — `detectFromGGUF()`, `MultiModalManager` | ✅ 实现 |
| **预处理** | `preprocess.zig` — `calcSizePreservedRatio`, resize, normalize, `ImageNormalize` | ✅ 基础实现 |
| **音频管线** | `audio/pipeline.zig`, `framing.zig`, `mel.zig`, `fft.zig` | ✅ 完成 |
| **Clamp 支持** | `graph/clamp.zig` — `loadClampInfoFromWeightNames` | ✅ 完成 |
| **Token 化** | `tokenize.zig` — 文本分片 + 图片预处理串联（见图 3.2） | 🟡 图片预处理已集成，音频 Mel 延后至 encode |

### 2.2 缺失 / 不完整 ❌

| 功能域 | 现状 | 严重程度 |
|--------|------|----------|
| **端到端推理管线** | 图能构建，但缺少完整的 encode→decode 流程 | 🔴 致命 |
| **ggml backend/scheduler** | Vision/Audio encoder 无独立 backend、无 scheduler | 🔴 致命 |
| **clip_image_batch_encode 等价** | 无统一的多模态 encode 入口 | 🔴 致命 |
| **音频 Mel→Tensor→encode 集成** | Audio pipeline 与 tokenize 解耦，Mel 计算延后到 encode 步 | 🟠 高 |
| **evalChunks（helper.zig）** | 仅 stub，图片/音频 chunk 编码未实现 | 🔴 致命 |
| **M-RoPE 解码器位置 (Qwen3VL)** | `imageGetDecoderPos` 基础实现，但未与 decode 环集成 | 🟠 高 |
| **非因果注意力 (Gemma4V/A/UV)** | `mtmd_decode_use_non_causal` 未被调用 | 🟠 高 |
| **Token 化 — 音频预处理** | `addAudioChunk` 仅估算 token 数，Mel 计算待 encode 阶段 | 🟡 设计中 |

---

## 3. 差距矩阵（按功能域）

### 3.1 A. 初始化与生命周期

| 参考 (C++) | Zig 现状 | 差距 |
|------------|----------|------|
| `mtmd_init_from_file` 创建独立的 `clip_ctx`(vision) + `clip_ctx`(audio)，各自拥有 backend/scheduler | `MultiModalManager.init` 创建 `VisionEncoder` + `AudioEncoder`，但**无独立 backend/scheduler** | 🔴 无 backend，compute 无法执行 |
| `clip_ctx` 包含 `ggml_backend_sched`，管理 graph compute | encoder 仅存储 `ctx_weights`（权重上下文），compute 时需外部提供 graph + backend | 🔴 缺少 scheduler |
| `mtmd_context` 持有 `image_preproc` 智能指针 | `MultiModalManager` 通过 `preprocess` 模块提供独立函数（无状态 preprocessor 对象） | 🟡 函数式风格，功能等价 |

### 3.2 B. Tokenize 流程

| 参考 (C++) | Zig 现状 | 差距 |
|------------|----------|------|
| `mtmd_tokenizer` 扫描 text→按 marker 分片→对每个 bitmap 调用 preprocessor | `tokenize.zig` 分片逻辑完成 ✅ | — |
| 对 image: `image_preproc->preprocess(img_u8)` → `clip_image_f32_batch` | `addImageChunk` 调用 `preprocess.resizeAndNormalize()` ✅，结果写入 `raw_pixels` | 🟡 中间缓冲为 `[]u8`，非 `clip_image_f32` |
| 对 audio: `audio_preproc->preprocess(samples)` → mel chunks | `addAudioChunk` 仅估算 token 数，Mel 计算延后 | 🟡 设计选择：tokenize 阶段不做重计算 |
| 结果：`mtmd_input_chunk.tokens_image.batch_f32` 已填充 | `ImageTokens.raw_pixels` 已填充处理后的像素 | 🟡 需在 encode 阶段转换为 ggml tensor |

### 3.3 C. Encode 流程

| 参考 (C++) | Zig 现状 | 差距 |
|------------|----------|------|
| `clip_image_batch_encode(ctx, threads, batch_f32, out_embd)` | 无统一入口 | 🔴 |
| 内部: 对 batch 中每个 entry 创建 `clip_graph` 子类 → `build()` → `ggml_backend_sched_alloc_graph` → `ggml_backend_sched_graph_compute` | 图构建完成 (`backend.buildGraph`)，但**无 compute 步骤** | 🔴 |
| 结果写入 `out_embd` vector | 结果需手动通过 `ggml_graph_get_tensor` 查找并拷贝 | 🟠 |
| `build_mm()` 虚函数 hook（gemma4 clamp）| `buildMMWithClamp` 辅助函数实现，但需手动传递 `clamp_map` | 🟡 功能可用，API 略繁琐 |

### 3.4 D. 解码器集成

| 参考 (C++) | Zig 现状 | 差距 |
|------------|----------|------|
| `mtmd_decode_use_non_causal(ctx, chunk)` 查询是否需要非因果 mask | 未实现调用 | 🟠 |
| `mtmd_decode_use_mrope(ctx)` 查询是否使用 M-RoPE | 未实现调用 | 🟠 |
| `mtmd_image_tokens_get_decoder_pos` 提供 M-RoPE 位置 | `imageGetDecoderPos` 基础实现，未集成 | 🟠 |
| `evalChunks` 遍历 chunks：text→decode, image→encode→set embd→decode, audio→encode→set embd→decode | `helper.zig` 仅有 stub | 🔴 |

### 3.5 E. 批量编码

| 参考 (C++) | Zig 现状 | 差距 |
|------------|----------|------|
| `mtmd_batch` 合并多个同类型 chunk 到一个 batch_f32 | 未实现 | 🟡 低优先级 |
| `clip_support_batch()` 查询模型是否支持批处理 | 未实现 | 🟡 |

---

## 4. 模型专项分析

### 4.1 qwen3vl

#### 参考实现关键点

```
文件: deps/llama.cpp/tools/mtmd/models/qwen3vl.cpp
继承: clip_graph_qwen2vl (复用 build_inp_with_temporal_merge)
```

**图构建流程**：
1. `build_inp_with_temporal_merge()` — 两个 Conv2D 相加（temporal merge）
2. Spatial merge — permute+reshape 将 [c,w,h] 合并为 [n_embd, n_patches]
3. `patch_bias` 添加（**Qwen3VL 特有，Qwen2VL 无**）
4. `resize_position_embeddings()` — 可学习位置嵌入插值 + 相同 spatial merge
5. **M-RoPE** — `ggml_rope_multi` with `GGML_ROPE_TYPE_VISION`, `mrope_sections = {d/4,d/4,d/4,d/4}`
6. ViT blocks — **fused QKV** (`attn_qkv.weight`), **无 gate FFN** (`ffn_gate` is null)
7. **Deepstack features** — 特定层输出经过独立的 norm+fc1+gelu+fc2, 最后 concat 到 projector 输出

#### Zig 实现状态

| 项目 | 状态 | 备注 |
|------|------|------|
| Temporal merge | ✅ | `gemma4a.zig` buildGraph 实现正确 |
| Spatial merge | ✅ | permute + reshape 序列匹配 |
| Patch bias | ✅ | 必需断言 + add |
| Learned pos embd | ✅ | `resizePositionEmbeddings` + 相同 spatial merge |
| M-RoPE positions | ✅ | `ggml.ropeMulti` with correct params |
| Fused QKV | ✅ | `layer.qkv_w` + view3d split |
| No-gate FFN | ✅ | `ff_gate_w` = null, buildFFN 正确处理 |
| Deepstack | ✅ | `hasDeepstack()` 检测 + concat |
| MM projector | ✅ | reshape + buildFFN (mm_0/mm_1) + deepstack concat |
| **端到端验证** | ❌ | 未与 llama.cpp 输出对比 |

**差距**：

1. **Deepstack 层检测**：当前从 `clip.vision.is_deepstack_layers` 读取 bool 数组。参考实现从 **weight 存在性** 自动检测 (`layer.has_deepstack()`)。两者等价但需确认 GGUF 中是否都有此 key。
2. **merge_factor 默认值**：参考: `n_merge > 0 ? n_merge * n_merge : 4` — 我们的实现一致。
3. **rope_type_vision = 24**：硬编码为 `24`，需确认 ggml 中 `GGML_ROPE_TYPE_VISION` 的值一致。
4. **无独立 backend/scheduler**：图能构建但无法 compute。

### 4.2 gemma4a

#### 参考实现关键点

```
文件: deps/llama.cpp/tools/mtmd/models/gemma4a.cpp
继承: clip_graph (独立实现)
```

**图构建流程**：
1. **输入**: Mel spectrogram `[n_frames, n_mel_bins]` → reshape 为 4D → transpose
2. **子采样 Conv2D ×2**: stride=2, padding=1, LayerNorm + ReLU
3. **Flatten** → **输入投影** (sscp_inp_proj) → 进入 Conformer 维度
4. **Conformer Blocks** (循环 n_layer 次):
   - **FFN 1 (half-step)**: RMSNorm → ff_up → SiLU → ff_down → scale(0.5) → residual
   - **Chunked Self-Attn + RPE**:
     - Q/K/V 投影（带 per_dim_scale）
     - Q blocking: pad → reshape → permute `[D,C,B,H]`
     - K/V overlapping block extraction: pad → roll → view4d (stride=C)
     - Content attention: `Kblk @ Qcur` → `[S,C,B,H]`
     - RPE: `attn_k_rel @ pos_emb` → project → blocked relative shift
     - Softcap: `tanh(scores/50) * 50`
     - Mask: `kq_mask` (预计算，causal + chunked)
     - Softmax → output: `Vblk @ attn` → reshape back
   - **Conv Module**: RMSNorm → pw1 → GLU(sigmoid gate) → transpose → pad+roll → ssm_conv(dw) → norm → SiLU → pw2
   - **FFN 2 (half-step)**: 同上 FFN 1，使用 ff_norm_1/ff_up_1/ff_down_1
   - **Layer output norm**: RMSNorm (ln_2)
5. **输出投影** (audio_out_proj)
6. **多模态嵌入器**: RMSNorm → soft_emb_norm (mul) → input_proj

**所有矩阵乘法都通过 `build_mm()` hook → Gemma4ClippableLinear（clamp 输入/输出）**

#### Zig 实现状态

| 项目 | 状态 | 备注 |
|------|------|------|
| 子采样 Conv2D | ✅ | stride=2, padding=1, LayerNorm, ReLU |
| Flatten + 输入投影 | ✅ | reshape2d + buildMMWithClamp |
| Chunked attn 参数 (C=12,P=12,S=24,R=13) | ✅ | 硬编码匹配 |
| Q/K/V 投影 + per_dim_scale | ✅ | q_scale/k_scale/softcap 正确 |
| Q blocking (pad→reshape→permute) | ✅ | [D,C,B,H] |
| K/V overlapping block extraction | ✅ | extractBlocks() 函数 |
| Content attention | ✅ | Kblk.mulMat(Qcur) |
| RPE (attn_k_rel @ pos_emb) | ✅ | fillSinusoidalPosEmb + blocked relative shift |
| Softcap + Mask + Softmax | ✅ | tanh(score/50)*50 + kq_mask |
| Conv Module (GLU + ssm_conv) | ✅ | sigmoid gate + pad(4)+roll(4)+ssmConv |
| FFN 1/2 half-step (res_weight=0.5) | ✅ | buildFFNWithClamp |
| 输出投影 + 多模态嵌入器 | ✅ | rmsNorm + soft_emb_norm + input_proj |
| Clamp-aware matmul | ✅ | buildMMWithClamp (clamp_info_map) |
| Sinusoidal RPE 填充 | ✅ | fillSinusoidalPosEmb |
| Chunked attention mask 填充 | ✅ | fillChunkedAttentionMask |
| **端到端验证** | ❌ | 未与 llama.cpp 输出对比 |
| **独立 backend/scheduler** | ❌ | 无法 compute |

**差距**：

1. **SSCP 权重命名**: 参考使用 `a.conv1d.{i}.weight`（1D 卷积），我们使用相同的名称加载。需确认 GGUF 中的实际名称。
2. **GEGLU vs SiLU**: Conformer FFN 使用 SiLU（无 gate），我们的 `buildFFNWithClamp` 正确处理 `gate=null` 时直接激活。
3. **Debug 命名**: 实现中保留了大量 `setName` + `setOutput` 调试代码。**生产环境应移除或条件编译**。

### 4.3 gemma4v

#### 参考实现关键点

```
文件: deps/llama.cpp/tools/mtmd/models/gemma4v.cpp
继承: clip_graph, support_batch() = true
```

**图构建流程**：
1. **输入归一化**: `scale_bias(inp_raw, 2.0, -1.0)` — 将 [0,1] 映射到 [-1,1]
2. **Conv2D patch embedding**: `ggml_conv_2d` → reshape → transpose
3. **2D 位置编码**: `position_embeddings` 作为 X/Y 查找表，`ggml_get_rows`
4. **ViT blocks** (via `build_vit()`):
   - **Norm**: RMSNorm
   - **Attention**: Q/K/V 分离（无 fused QKV），**2D RoPE**:
     - 第一半 dim 用 `pos_x` 做 `ggml_rope_ext(NEOX)`
     - 第二半 dim 用 `pos_y` 做 `ggml_rope_ext(NEOX)`
     - concat 两半
   - **V norm**: 在 attention 前对 V 做 `ggml_rms_norm`（**gemma4v 特有**）
   - **kq_scale = 1.0**（不使用 `1/sqrt(d_head)`）
5. **Gemma4VisionPooler**: transpose → `ggml_pool_2d(AVG, kernel_size)` → reshape → transpose → `scale(sqrt(n_embd))`
6. **标准化**: `(hidden - std_bias) * std_scale`
7. **Gemma4MultimodalEmbedder**: RMSNorm → `mm_input_proj`

**所有矩阵乘法通过 `build_mm()` → Gemma4ClippableLinear**

#### Zig 实现状态

| 项目 | 状态 | 备注 |
|------|------|------|
| 输入归一化 (scale=2, bias=-1) | ✅ | scale + add bias tensor |
| Conv2D patch embedding | ✅ | reshape3d + transpose |
| 2D 位置编码 (X/Y lookup) | ✅ | pos_embd.view2d + getRows |
| ViT blocks (buildVit) | ✅ | 使用通用 `buildVit` + `AddPosContext.callback` |
| 2D RoPE (NEOX, 分半) | ✅ | view4d 拆分 + ropeExt(NEOX) + concat |
| V RMSNorm | ✅ | `v_norm=true` via BuildVitOpts |
| kq_scale = 1.0 | ✅ | BuildVitOpts.kq_scale = 1.0 |
| Pooling (avg pool 2d) | ✅ | pool2d(AVG, kernel_size) |
| 标准化 (std_bias + std_scale) | ✅ | sub + mul |
| 多模态嵌入器 (rmsNorm + proj) | ✅ | rmsNorm + buildMMWithClamp |
| Clamp-aware matmul | ✅ | buildMMWithClamp |
| **端到端验证** | ❌ | 未与 llama.cpp 输出对比 |
| **独立 backend/scheduler** | ❌ | 无法 compute |

**差距**：

1. **`scale_bias` vs `scale + add`**: 参考用 `ggml_scale_bias(ctx, t, 2.0, -1.0)`，我们用 `scale(ctx, 2.0)` + `add(ctx, bias_tensor)`。语义等价但多一次操作。
2. **support_batch**: 参考 `support_batch() = true`，我们在 `GraphBuilder` 中未实现此查询。
3. **`ggml_pool_2d` 在 ggml.zig 中的绑定**: 需确认 `pool2d` 方法正确传递 `GGML_OP_POOL_AVG` 枚举值。

---

## 5. 内存管理差距

### 5.1 参考实现的 Backend/Scheduler 模型

```
clip_ctx {
    ggml_backend_t backend;       // GPU (Metal/CUDA) 或 CPU
    ggml_backend_t backend_cpu;   // 始终存在的 CPU fallback
    ggml_backend_sched_t sched;   // 多后端调度器
    ggml_gallocr_t allocr;        // 图内存分配器
    buf_compute_meta              // 元数据缓冲（图构建用）
}
```

**执行流程**：
```
clip_image_batch_encode:
  for each entry in batch_f32:
    1. new clip_graph_xxx(ctx, entry)  → 创建模型特定图构建器
    2. graph->build()                  → 构建 ggml_cgraph
    3. ggml_backend_sched_reserve(sched, allocr)
    4. ggml_backend_sched_alloc_graph(sched, gf)
    5. ggml_backend_sched_graph_compute(sched, gf)
    6. 从 gf 中提取输出张量 → 拷贝到 out_embd
```

### 5.2 当前 Zig 实现

```zig
VisionEncoder {
    params: VisionEncoderParams,
    weights: VisionEncoderWeights,
    ctx_weights: *ggml.Context,    // 仅权重上下文
    backend: *const VisionEncoderBackend,
}

pub fn encode(...) {
    1. preprocess.normalizeToTensor()  → 创建输入张量
    2. backend.buildGraph()            → 构建 ggml_cgraph
    3. ggml_graph_get_tensor()         → 查找输出
    // ❌ 无 compute 步骤！
}
```

### 5.3 缺失的关键组件

| 组件 | 参考 | 当前 | 影响 |
|------|------|------|------|
| GPU backend | `ggml_backend_init_by_name` (Metal/CUDA) | 无 | 纯 CPU 执行 |
| Scheduler | `ggml_backend_sched` | 无 | 无法调度多后端 |
| Graph allocator | `ggml_gallocr` | 无 | 无法分配图中间张量 |
| Compute context | 独立 `ggml_init` (no_alloc=true) | 复用权重 context | 权重与计算张量混用 |
| `ggml_backend_sched_graph_compute` | ✅ | ❌ 未绑定/未调用 | **无法执行任何计算** |

---

## 6. 推理流程差距

### 6.1 完整推理流程对比

**参考实现（端到端）**：
```
1. mtmd_init_from_file(mmproj, text_model)
   ├─ clip_init (vision) → clip_ctx_v (含 backend + scheduler)
   ├─ clip_init (audio)  → clip_ctx_a
   └─ 选择 preprocessor (fixed_size / dyn_size / llava_uhd / ...)

2. mtmd_tokenize(ctx, text, bitmaps)
   ├─ split by media_marker → parts
   ├─ for each bitmap:
   │   ├─ image: preprocessor.preprocess(u8) → clip_image_f32_batch
   │   │         wrapped in mtmd_image_tokens.batch_f32
   │   └─ audio: audio_preproc.preprocess(samples) → mel chunks
   │             wrapped in mtmd_audio_tokens.batch_f32
   └─ output: mtmd_input_chunks (text/image/audio 交替)

3. for each chunk in chunks:
   ├─ text:   llama_decode(tokens)
   ├─ image:  mtmd_encode_chunk(ctx, chunk)
   │          └─ clip_image_batch_encode(ctx_v, batch_f32, out_embd)
   │          → 将 out_embd 设为 llama 输入嵌入
   │          → llama_decode (非因果 mask 如果需要)
   └─ audio:  同上，使用 ctx_a
```

**当前 Zig 实现**：
```
1. MultiModalManager.init() → VisionEncoder + AudioEncoder
   ❌ 无 backend/scheduler
   ❌ 无 preprocessor 关联

2. tokenize.zig → 仅估算 token 数
   ❌ 不做预处理
   ❌ 不创建 clip_image_f32

3. helper.zig evalChunks → stub
   ❌ "Image chunk evaluation not fully implemented"
   ❌ "Audio chunk evaluation not fully implemented"
```

### 6.2 具体缺失的步骤

#### 步骤 1：初始化

- [ ] 创建独立的 `ggml_backend` (CPU 默认，可选 GPU)
- [ ] 创建 `ggml_backend_sched`
- [ ] 创建计算用的 `ggml.Context`（与权重 context 分离）
- [ ] 初始化图像 preprocessor（根据 `proj_type` 选择 fixed_size / dyn_size / llava_uhd）
- [ ] 初始化音频 preprocessor（Whisper Mel 或 Conformer）

#### 步骤 2：Tokenize 中的预处理

- [ ] 图片: `image_preproc.preprocess(u8_bitmap)` → resize + normalize → `clip_image_f32`
- [ ] 音频: `audio_preproc.preprocess(f32_samples)` → FFT → Mel → `clip_image_f32`
- [ ] 将结果包裹在 `ImageTokens` / chunk 中

#### 步骤 3：Encode

- [ ] 实现 `encodeChunk(ctx, chunk, out_embd)`:
  1. 从 chunk 中提取 `batch_f32`
  2. 对每个 entry 创建图（`backend.buildGraph`）
  3. `sched_reserve` → `sched_alloc_graph` → `sched_graph_compute`
  4. 提取输出嵌入

#### 步骤 4：LLM Decode 集成

- [ ] 实现 `mtmd_decode_use_non_causal()` — 查询 gemma4 系列
- [ ] 实现 `mtmd_decode_use_mrope()` — 查询 qwen 系列
- [ ] 实现 `imageGetDecoderPos()` 的完整 M-RoPE 位置计算
- [ ] 实现 `evalChunks` 完整流程:
  - text chunk → `llama_decode(tokens, n_past)`
  - image/audio chunk → encode → 替换嵌入 → `llama_decode`

---

## 7. 优先级路线图

### P0 — 致命（阻止任何模型运行）

| # | 任务 | 涉及文件 | 状态 |
|---|------|----------|------|
| 1 | **绑定 ggml backend/scheduler API** | `src/ggml/` — 添加 `ggml_backend_*`, `ggml_backend_sched_*`, `ggml_gallocr_*` | ⬜ 未开始 |
| 2 | **实现 encode 计算执行** | `vision/encoder.zig`, `audio/encoder.zig` — 在 `encode()` 中添加 compute 步骤 | ⬜ 依赖 #1 |
| 3 | **实现端到端 evalChunks** | `helper.zig` — 完整实现图片/音频 chunk 的 encode→decode | ⬜ 依赖 #2 |

### P1 — 高（功能正确性）

| # | 任务 | 涉及文件 | 状态 |
|---|------|----------|------|
| 4 | **Tokenize 中集成图像预处理** | `tokenize.zig` — 调用 preprocessor，填充 `batch_f32` | ✅ `addImageChunk` 已调用 `resizeAndNormalize` |
| 5 | **Tokenize 中集成音频预处理** | `tokenize.zig` + `audio/pipeline.zig` | 🟡 Mel 延后到 encode 阶段（设计选择） |
| 6 | **实现 non-causal / M-RoPE 查询** | `mod.zig` — `decodeUseNonCausal()`, `decodeUseMRope()` | ⬜ 未开始 |
| 7 | **M-RoPE 解码器位置计算** | `helper.zig` — 完善 `imageGetDecoderPos` | ⬜ 未开始 |
| 8 | **对比验证 (compare_logits)** | `tools/` — 与 llama.cpp 输出对比，NMSE < 1e-5 | ⬜ 未开始 |

### P2 — 中（完善与优化）

| # | 任务 | 涉及文件 |
|---|------|----------|
| 9 | **动态分辨率预处理** | `preprocess.zig` — 完善 `calcSizePreservedRatio` 调用链 |
| 10 | **批量编码 (batch)** | `mod.zig` — `mtmd_batch` 接口 |
| 11 | **GPU backend 支持** | 基于 P0#1，添加 Metal/CUDA 初始化 |
| 12 | **调试代码清理** | 移除 graph 中的 `setOutput` 调试代码（条件编译） |

### P3 — 低（长期改进）

| # | 任务 |
|---|------|
| 13 | GraphBuilder 统一 `build_mm()` 虚函数模式（避免手动传 clamp_map） |
| 14 | 多模态 streaming（音频流式输入） |
| 15 | 视频支持（Qwen-VL temporal merge 多帧） |

---

## 附录 A：关键 GGUF Key 参考

### qwen3vl

| GGUF Key | 用途 |
|----------|------|
| `clip.vision.image_size` | ViT 输入尺寸 |
| `clip.vision.patch_size` | Patch 大小 |
| `clip.vision.image_min_pixels` / `image_max_pixels` | 动态分辨率范围 |
| `clip.vision.spatial_merge_size` | spatial merge 因子 (n_merge) |
| `clip.vision.is_deepstack_layers` | bool[] 标记哪些层有 deepstack |

### gemma4a

| GGUF Key | 用途 |
|----------|------|
| `clip.audio.embedding_length` | Conformer 嵌入维度 |
| `clip.audio.attention.head_count` | 注意力头数 |
| `clip.audio.block_count` | Conformer 层数 |
| `clip.audio.num_mel_bins` | Mel 滤波器数量 |
| 张量: `a.conv1d.{i}.weight/bias` | 子采样卷积 |
| 张量: `a.input_projection.weight` | 输入投影 |
| 张量: `a.blk.{d}.attn_q/k/v.weight` | Conformer 注意力 |
| 张量: `a.blk.{d}.attn_k_rel.weight` | RPE 投影 |
| 张量: `mm.a.soft_emb_norm.weight` | 多模态嵌入器归一化 |
| 张量: `mm.a.input_projection.weight` | 多模态嵌入器投影 |

### gemma4v

| GGUF Key | 用途 |
|----------|------|
| `clip.vision.*` (标准 ViT) | 超参数 |
| 张量: `v.position_embd.weight` | 2D 位置嵌入表 |
| 张量: `v.std_bias` / `v.std_scale` | Pool 后标准化 |
| 张量: `mm.input_projection.weight` | 多模态投影 |
| 张量: `mm.soft_emb_norm.weight` | 嵌入器归一化 |

---

## 附录 B：与参考实现的故意差异

| 差异 | 原因 |
|------|------|
| Zig 使用 `VisionEncoderBackend` / `AudioEncoderBackend` 静态分发 | Zig 无虚表继承，用函数指针表替代 |
| `GraphBuilder` 不存储 `clip_ctx` 引用 | 解耦图构建与执行 |
| 权重使用 `VisionEncoderWeights` 统一容器 | 简化多态，用 optional 字段区分模型 |
| `buildMMWithClamp` 需手动传 `clamp_map` | 非虚函数，无法自动 hook |

---

*最后更新: 2026-07-05 (tokenize.zig 图片预处理已集成，ImageNormalize 枚举添加到 preprocess.zig)*
