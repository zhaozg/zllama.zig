# 架构设计文档

> **说明：** 本文档描述 **zllama.zig** 当前实现的系统架构。以"编译期强类型 + 显式分配 + VTable 接口"为基石构建白盒插件系统，支撑多模型文本推理与多模态（视觉/音频）编码。

---

## ⛓ 七层 DAG 架构总览

系统按依赖方向自底向上分为七层。每一层只能依赖比它低编号的层——这是一条**硬性约束**。

```
L7  应用入口      main.zig  cli_args.zig  InferenceEngine (engine.zig)
                  │ 持有一切，无依赖限制
                  ▼
L6  执行引擎      engine_common.zig  graph_context.zig  decode.zig  prefill.zig  multimodal.zig
                  │ computeGraph / Gallocr / IncContext / 三阶段 prefill
                  ▼
L5  多模态门面    mtmd/mod.zig (MultiModalManager)  mtmd/helper.zig (evalChunks)
                  │ 根据 GGUF key 选择 Backend，编排视觉/音频编码
                  ▼
L4  编码器&预处理 mtmd/vision/encoder.zig  mtmd/audio/encoder.zig  mtmd/preprocess.zig
                  │ 图像归一化、Mel 谱、各 Backend 分发
                  ▼
L3  模型实现      ┌─ L3a 文本: models/{llama,qwen2,qwen35,gemma3,gemma4,qwen3vl}.zig + registry.zig
                  │  L3b 多模态图: mtmd/graph/models/{gemma4v,gemma4a,gemma4uv,qwen2vl,qwen3vl}.zig
                  │  ★ L3a 与 L3b 互不 import，仅在 L6/L5 处交汇
                  ▼
L2  共享算子&基础设施
                  ├─ L2a 文本构建块: layers/{attention,rope,rms_norm,linear,swiglu,pooling,embed}.zig
                  ├─ L2b 多模态构建块: mtmd/graph/{attn,ffn,norm,patch,rope,merge,stack,mm,clamp,vit,builder}.zig
                  ├─ L2c 基础设施: core/graph_builder.zig  kv_cache.zig  sampler.zig  memory.zig
                  └─ L2d 分词&模板: tokenizer/  chat_template/
                  │ 仅操作 ggml 张量，不感知任何具体模型
                  ▼
L1  数据格式&接口  gguf.zig  model.zig (Architecture, ModelVTable, ModelInstance)  core/memory.zig
                  │ 定义跨层契约，仅依赖 L0
                  ▼
L0  平台抽象&FFI  std  ggml/ (c.zig, context, tensor, graph, backend, ops)
                  │ 纯绑定，无业务逻辑
```

### 三条铁律

| # | 铁律 | 含义 |
|---|------|------|
| **L3 不碰 L6** | 模型图构建（文本 + 多模态）永远不能调用 `computeGraph` / `Gallocr` / `IncContext`。图构建只负责产出 `ggml_cgraph`。 |
| **文本 ↔ 多模态隔离** | `src/models/`（L3a）与 `src/mtmd/graph/models/`（L3b）互不 import。仅交汇于 `core/multimodal.zig`（三阶段 prefill）和 `mtmd/mod.zig`（MultiModalManager）。 |
| **L2 不感知模型** | `layers/` 和 `mtmd/graph/{attn,ffn,…}` 仅操作 `ggml.Tensor`，不 import 任何具体模型 `.zig` 文件。 |

---

## 🧱 两大 VTable 家族

系统不使用 RTTI 或类继承。多态通过 **编译期确定的函数指针结构体（VTable）** 实现。

### 家族 A：文本 LLM — `ModelVTable`（`src/model.zig`）

```
ModelVTable {
    getParams:           fn(ptr) → *const ModelParams
    buildGraph:          fn(ptr, builder, input, n_tokens, cache, pos) → *Tensor
    resetSSMStates:      ?fn(ptr)
    setKVCacheContext:   ?fn(ptr, ctx)
    deinit:              fn(ptr, allocator)
    getPerLayerMaxSeqLen:?fn(ptr, allocator) → ?[]u32
    buildMM:             ?fn(ptr, ctx, graph, input, n_tokens, cache, pos, embd_override, offset, causal) → *Tensor
}
```

- 包装为 `ModelInstance { ptr: *anyopaque, vtable: *const ModelVTable }`。
- 每个模型（llama, qwen35, gemma4, ...）在 `src/models/` 下导出 `pub const vtable = ModelVTable{...}`。
- `registry.createModel()` 根据 `Architecture` 枚举返回 `ModelInstance`。
- 可选函数指针（`?fn`）允许模型按需声明能力：`buildMM` 表示支持多模态 embedding override，`getPerLayerMaxSeqLen` 表示混合注意力层的差异化 seq_len。

### 家族 B：多模态编码器 — `VisionEncoderBackend` / `AudioEncoderBackend`（`src/mtmd/graph/mod.zig`）

```
VisionEncoderBackend {
    name:               []const u8
    supportBatch:       bool          // 默认 false
    loadParams:         fn(gguf_file, params)
    loadWeights:        fn(allocator, gguf_file, ctx, weights)
    loadClampInfo:      fn(allocator, gguf_file, weights)
    buildGraph:         fn(ctx, gf, weights, hparams, image_tensor) → *CGraph
    estimateOutputTokens: fn(img_width, img_height, patch_size, n_merge) → u32
}

AudioEncoderBackend {
    name:               []const u8
    loadParams:         fn(gguf_file, params)
    loadWeights:        fn(allocator, gguf_file, ctx, weights)
    loadClampInfo:      fn(allocator, gguf_file, weights)
    buildGraph:         fn(ctx, gf, weights, hparams, mel_tensor, clamp_map) → *CGraph
    estimateOutputTokens: fn(n_frames) → u32
}
```

- 每个多模态模型在 `src/mtmd/graph/models/` 下导出 `.backend` 单例。
- `src/mtmd/vision/mod.zig` 和 `src/mtmd/audio/mod.zig` 的 `getBackend(name)` 按名称查找。
- `MultiModalManager.detectFromGGUF()` 根据 GGUF 元数据 key 自动选择 Backend。

---

## 🧩 核心模块详解

### L0：ggml FFI 绑定（`src/ggml/`）

| 文件 | 职责 |
|------|------|
| `c.zig` | 裸 C 函数声明（`@cImport`），`c` 命名空间供内部使用 |
| `context.zig` | `Context` 封装：`init`/`initNoAlloc`/`newTensor*`/`view*`/`setNoAlloc`/`reset`/`usedMem` |
| `tensor.zig` | `Tensor` 封装：`ne()`/`nb()`/`dataF32()`/`dataBytes()`/`setName()` |
| `graph.zig` | `CGraph` 封装：`initReserved`/`compute` |
| `backend.zig` | `Backend`/`Gallocr`/`BackendBufferType`/`Scheduler` + `backendCpuInit`/`detectBestBackend` |
| `ops.zig` | 所有 ggml 算子的类型安全封装（`mulMat`/`rmsNorm`/`ropeExt`/`silu`/`softMax`…） |
| `threadpool.zig` | `ThreadPool` 封装 |
| `gguf.zig` | GGUF 格式的 ggml 层解析（与 `src/gguf.zig` 业务层 GGUF 解析互补） |

### L1：跨层契约

| 文件 | 核心类型 | 说明 |
|------|----------|------|
| `src/gguf.zig` | `GGUFFile` | 业务层 GGUF 读取：`getU32`/`getF32`/`getString`/`getTensor`/`getF32Array` |
| `src/model.zig` | `Architecture`、`ModelVTable`、`ModelInstance`、`ModelParams`、`ModelWeights`、`ModelCapabilities`、`SpecialTokens` | 文本 LLM 的接口契约 + 架构特殊 Token 配置 |
| `src/core/memory.zig` | `MemoryContext`、`MemoryVTable`、`KVCacheMemory`、`HybridMemory` | 统一内存接口（KV Cache + SSM 状态 + Conv 状态） |

### L2：共享算子与基础设施

#### L2a 文本构建块（`src/layers/`）

| 文件 | 关键函数 | 说明 |
|------|----------|------|
| `rms_norm.zig` | `rmsNorm(ctx, x, weight, eps)` | RMS 归一化 |
| `rope.zig` | `applyRope(ctx, q, k, pos, config)` | RoPE / M-RoPE 位置编码 |
| `swiglu.zig` | `swiglu(ctx, gate, up, x)` | SwiGLU 前馈 |
| `attention.zig` | `scaledDotProductAttention(...)` | 缩放点积注意力（含 GQA/MQA） |
| `linear.zig` | `linear(ctx, x, weight, bias?)` | 线性投影 |
| `embed.zig` | `tokenEmbed(ctx, tokens, weight)` | Token 嵌入 |
| `pooling.zig` | `attentionPooling(...)` / `meanPooling(...)` | 嵌入模型的池化 |

#### L2b 多模态构建块（`src/mtmd/graph/`）

| 文件 | 关键函数 | 说明 |
|------|----------|------|
| `builder.zig` | `GraphBuilder` 结构体 | 持有 weights/hparams/ctx/gf，提供 `buildNorm`/`buildAttn`/`buildFFN`/`buildVit` 等方法 |
| `vit.zig` | `buildVit()` | 通用 ViT 编码器（多层 attention + FFN + norm） |
| `attn.zig` | `buildAttn()` | 多模态注意力（含 flash attention） |
| `ffn.zig` | `buildFFN()` | 多模态 FFN（GELU/SiLU/ReLU²） |
| `norm.zig` | `buildNorm()` | Layer Norm / RMS Norm |
| `patch.zig` | `buildInp()` / `buildInpRaw()` | 图像→patch 嵌入 |
| `rope.zig` | `buildRope2D()` | 2D RoPE（视觉位置编码） |
| `merge.zig` | `buildPatchMergePermute()` | Patch merge + permute |
| `mm.zig` | `buildMM()` / `buildMLPProjector()` / `buildGemma3Projector()` | 投影器（视觉/音频 embedding→LLM 空间） |
| `types.zig` | `VisionHParams` / `VisionEncoderWeights` / `BuildVitOpts` / `ProjectorType` 等 | 类型定义 |
| `debug.zig` | `DebugTensorRegistry` | 中间层张量钩子（用于 compare 工具） |

#### L2c 文本基础设施

| 文件 | 核心结构 | 说明 |
|------|----------|------|
| `core/graph_builder.zig` | `GraphBuilder` | 持有 ctx/graph/params/allocator，提供 `buildRmsNorm`/`buildRope`/`buildAttention`/`buildSwiGLU` |
| `kv_cache.zig` | `KVCache` | 预分配 KV Cache 张量，通过 `view3d` + `ggml.cpy` 增量写入，支持每层差异化 max_seq_len（ISWA） |
| `sampler.zig` | `Sampler` | 贪心采样 / Top-K / Top-P / Mirostat |
| `core/memory.zig` | `MemoryContext` | 统一内存接口：`getK`/`getV`/`setKv`/`getConvState`/`getSSMState` |

### L3：模型实现

#### L3a 文本模型（`src/models/`）

| 文件 | 架构 | 关键特征 |
|------|------|----------|
| `llama.zig` | LLaMA 2/3 | 标准 transformer，SwiGLU + RoPE + RMSNorm |
| `qwen2.zig` | Qwen 2 / 2.5 / 3 | GQA，与 llama 高度相似 |
| `qwen35.zig` | Qwen 3.5 | 稀疏 MoE + 全注意力混合层 |
| `gemma3.zig` | Gemma 3 | 滑动窗口 + 全局注意力混合，独立 key_length/value_length |
| `gemma4.zig` | Gemma 4 | SWA + ISWA + 文本-only 前向 |
| `gemma4_graph.zig` | Gemma 4 图构建 | `buildMM` 实现：embedding override + 可选非因果注意力 |
| `qwen3vl.zig` | Qwen3-VL 文本侧 | M-RoPE 感知的文本前向 |
| `embedding.zig` | BERT / Nomic | 嵌入模型（mean pooling / attention pooling） |
| `registry.zig` | — | `detectArchitecture()` + `createModel()` + `detectCapabilities()` |

#### L3b 多模态图（`src/mtmd/graph/models/`）

| 文件 | Backend | 说明 |
|------|---------|------|
| `gemma4v.zig` | `VisionEncoderBackend` | Gemma 4 Vision（ViT + MLP 投影器） |
| `gemma4a.zig` | `AudioEncoderBackend` | Gemma 4 Audio（Conformer 编码器） |
| `gemma4uv.zig` | `VisionEncoderBackend` | Gemma 4 Ultra-Vision |
| `qwen2vl.zig` | `VisionEncoderBackend` | Qwen2-VL 视觉编码器 |
| `qwen3vl.zig` | `VisionEncoderBackend` | Qwen3-VL 视觉编码器（含 M-RoPE） |

### L4：编码器与预处理

| 模块 | 关键文件 | 说明 |
|------|----------|------|
| 视觉编码器 | `vision/encoder.zig` | `VisionEncoder`：生命周期管理，通过 `backend.buildGraph()` 分发 |
| 视觉预处理 | `vision/preprocess.zig` | `normalizeToTensor()`：RGB → 归一化 → ggml Tensor |
| 视觉配置 | `vision/config.zig` | `VisionEncoderParams` |
| 视觉权重 | `vision/loader.zig` | 从 GGUF 加载视觉权重张量 |
| 视觉后处理 | `vision/postprocess.zig` | 标准化 + 投影到 LLM 空间 |
| 音频编码器 | `audio/encoder.zig` | `AudioEncoder`：WAV 输入 → Mel 谱 → Conformer 编码 |
| 音频预处理 | `audio/framing.zig` / `audio/mel.zig` / `audio/log_transform.zig` | 分帧 → Mel 滤波器组 → log |
| 音频 Mel 频谱 | `audio/mel_spectrogram.zig` | `processPcmSamples()`：PCM → Mel 谱 |
| 通用预处理 | `preprocess.zig` | 图像 resize / 归一化模式 |
| Token 化 | `tokenize.zig` | 文本+媒体标记混合 tokenize |

### L5：多模态门面

| 文件 | 核心结构 | 说明 |
|------|----------|------|
| `mtmd/mod.zig` | `MultiModalManager` | 持有 `VisionEncoder`/`AudioEncoder`，`detectFromGGUF()` 自动选择 Backend，`encodeMedia()` 编码媒体。同时定义 `InputChunk`/`InputChunks`/`MtmdContext`/`PosType` 等 |
| `mtmd/helper.zig` | `evalChunks()` / `imageGetDecoderPos()` | Chunk 执行编排（文本→视觉→音频顺序），M-RoPE 位置计算 |

### L6：执行引擎

| 文件 | 核心函数/结构 | 说明 |
|------|---------------|------|
| `engine_common.zig` | `computeGraph()` / `WallTimer` / `mmapFile()` | CPU 图计算（Gallocr 所有权上移）、时间测量、mmap 文件读取 |
| `graph_context.zig` | `IncContext` / `DecodeStep` | 增量解码上下文：ctx 复用、Gallocr 预分配、>80% 回收触发 reset |
| `decode.zig` | `runDecodeLoop()` / `DecodeCallbacks` | 共享解码循环（文本/多模态通用），回调驱动 |
| `prefill.zig` | `threeStagePrefill()` | 三阶段多模态 prefill：prefix → media → suffix |
| `multimodal.zig` | `generateWithImage()` / `generateWithAudio()` / `multimodalPrefillUnified()` | 多模态入口，编排 prefill + decode |

### L7：应用入口

| 文件 | 说明 |
|------|------|
| `main.zig` | Juicy Main，CLI 分发（text / vision / audio / benchmark / embedding） |
| `cli_args.zig` | CLI 参数定义 |
| `core/engine.zig` | `InferenceEngine`：统一生命周期管理，持有 model/kv_cache/tokenizer/sampler/mtmd |

---

## 🔄 数据流：从 prompt 到 token

### 纯文本推理

```
main.zig
  → InferenceEngine.init(model_path)
      → registry.detectArchitecture(gguf)
      → registry.createModel(arch)      // 返回 ModelInstance
      → KVCache.init()
      → Tokenizer.init()
  → engine.generate(prompt, max_tokens)
      → applyChatTemplate(prompt)
      → tokenizer.encode(formatted)
      → textPrefill(tokens)             // 构建图 → Gallocr alloc → compute
      → sampler.sample(logits)           // 第一个 token
      → runDecodeLoop(...)               // 增量解码循环
          → IncContext.beginStep()       // 复用 ctx_inc，必要时 reset
          → model.buildGraph(...)        // 单 token 前向
          → gallocr.allocGraph() → compute  (gallocr 由 InferenceEngine 管理)
          → sampler.sample() → output
```

### 多模态推理（图像为例）

```
main.zig
  → engine.generateWithImage(prompt, image_path)
      → mm_manager.detectFromGGUF(mmproj)   // 选择 VisionEncoderBackend
      → mm_manager.init(gguf, ctx)           // 加载视觉权重
      → applyChatTemplateWithMedia(prompt)    // 插入 <__media__> 标记
      → tokenizeWithMediaPlaceholders(...)    // 文本 token + 媒体占位符
      → vision_encoder.encode(image_data)    // ViT forward → embedding
      → threeStagePrefill(...)                // Pass1(prefix) → Pass2(media, non-causal) → Pass3(suffix)
      → runDecodeLoop(...)                    // 增量解码
```

---

## 💾 内存生命周期

```
时间线:
  init                    推理循环                              exit
  
  ctx_weights ──────────── 权重驻留（所有权永不转移）────────────→ free
  ctx_kv_cache ─────────── KV Cache 驻留 ──────────────────────→ free
  ctx_graph ─────┐        ctx 在 prefill 中 reset/复用 ────────→ free
                 │
  Gallocr (prefill) ───── compute → ★故意泄漏★ ────────────────→ OS 回收
  Gallocr (decode) ────── IncContext 管理，复用 ────────────────→ free
```
时间线:
  init                    推理循环                              exit
  
  ctx_weights ──────────── 权重驻留（所有权永不转移）────────────→ free
  ctx_kv_cache ─────────── KV Cache 驻留 ──────────────────────→ free
  ctx_graph ─────┐        ctx 在 prefill 中 reset/复用 ────────→ free
                 │
  Gallocr ──────────────── InferenceEngine 持有，所有权上移 ────→ free (deinit)
  IncContext.galloc ────── IncContext 管理，resetFull 时释放重建 ─→ free
```

关键策略：
- **权重**：`VisionEncoder.init()` 加载后借出，永不转移所有权。
- **Gallocr 所有权上移**：`computeGraph()` 不再内部创建并泄漏 Gallocr，而是由 `InferenceEngine` 在 `init()` 时创建，`deinit()` 时释放。`IncContext.resetFull()` 可触发 Gallocr 释放与重建，回收计算图内存。
- **IncContext**：增量解码的 `ctx_inc` 在占用率超过 80% 时自动 `resetFull()` 回收（含 Gallocr 内存）。
- **三阶段 prefill**：每个 pass 之间调用 `graph_ctx.reset()` 释放上一 pass 的临时张量和图节点；使用传入的 gallocr（非局部创建）。

---

## 🔌 扩展指南


### 新增文本 LLM 架构

1. `src/model.zig`：在 `Architecture` 枚举添加成员，`fromString()` 添加 GGUF 名称映射。
2. `src/models/<new>.zig`：实现 `ModelVTable` 的函数指针，导出 `pub const vtable = ModelVTable{...}`。
3. `src/models/registry.zig`：在 `createModel()` 添加分支返回 `ModelInstance`。

### 新增视觉/音频编码器

1. `src/mtmd/graph/models/<new>.zig`：实现 `buildGraph()` 等函数，导出 `pub const backend = VisionEncoderBackend{...}`。
2. `src/mtmd/graph/mod.zig`：在 `model_graphs` 中注册。
3. `src/mtmd/vision/mod.zig`（或 `audio/mod.zig`）：在 `registered_backends` 和 `getBackend()` 注册。
4. `src/mtmd/mod.zig`：在 `detectFromGGUF()` 添加 GGUF key 匹配。

### 新增 ggml 算子

1. `src/ggml/ops.zig`：封装 C 函数为类型安全函数。
2. `src/ggml/mod.zig`：`pub const` 重新导出。

---

## 📚 相关文档

| 文档 | 内容 |
|------|------|
| `AGENTS.md` | AI 编程入口、设计哲学、决策矩阵 |
| `docs/GGML_BINDING.md` | ggml C API 绑定设计规范 |
| `docs/MTMD_ARCHITECTURE.md` | 多模态模块详细设计 |
| `docs/MULTIMODAL.md` | 多模态使用指南 |
| `docs/HOW_TO_ADD_NEW_MODEL.md` | 新增模型详细指南 |
| `docs/MEMMGT.md` | 内存管理详解 |
| `docs/TEST.md` | 测试体系 |
