# 多模态推理模块 — MULTIMODAL

> 参考：[llama.cpp tools/mtmd](deps/llama.cpp/tools/mtmd/)、Gemma 4 多模态实现
> 当前状态：视觉/音频端到端基础联调已完成（三阶段 prefill + 注意力修复），待质量验证与功能扩展。

---

## 目录

1. [概述与架构](#概述与架构)
2. [公共基础设施](#公共基础设施)
3. [视觉编码器](#视觉编码器)
4. [音频编码器](#音频编码器)
5. [端到端推理流程](#端到端推理流程)
1. [参考资源](#参考资源)

---

## 概述与架构

zllama.zig 通过 `src/mm/` 目录下的模块实现多模态（图像+音频）推理，当前主要支持 **Gemma 4 E2B** 模型的内建编码器。

```
┌───────────────────────────────────────────────────────────────────┐
│  CLI                                                              │
│  --model gguf --mmproj mmproj.gguf --image x.png --audio y.wav    │
└─────────────┬─────────────────────────┬───────────────────────────┘
              │                         │
    ┌─────────▼─────────┐       ┌───────▼────────┐
    │ Image Pipeline    │       │ Audio Pipeline │
    │ preprocess.zig    │       │ preprocess.zig │
    │ vision.zig        │       │ audio.zig      │
    └─────────┬─────────┘       └───────┬────────┘
              │                         │
              └─────────────┬───────────┘
                            │
                  ┌─────────▼─────────┐
                  │ manager.zig       │
                  │ MultiModalManager │
                  │ 编码调度/能力检测 │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ mtmd.zig          │
                  │ Chunk/Bitmap/MtmdContext │
                  │ tokenize() 分词   │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ LLM (gemma4.zig)  │
                  │ forwardWithEmbdOverride │
                  │ 三阶段 prefill    │
                  └───────────────────┘
```

---

## 公共基础设施

### Chunk 模型 — 输入的多态表示

`tokenize.zig` 按 `media_marker`（默认 `<__media__>`）将输入分割为有序的 `InputChunk` 列表：

```
"描述这张图<__media__>和这段音频<__media__>"
→ [text:"描述这张图"] [image:bm_0] [text:"和这段音频"] [audio:bm_1]
```

```zig
pub const ChunkType = enum(u8) { text, image, audio };

pub const InputChunk = struct {
    chunk_type: ChunkType,
    tokens_text: ?[]const i32 = null,
    tokens_image: ?ImageTokens = null,
    tokens_audio_n: u32 = 0,
    id: ?[]const u8 = null,
};

pub const InputChunks = struct {
    // append/clear/deinit, totalTokens(), totalPos()
};
```

### Bitmap — 原始媒体数据

```zig
pub const Bitmap = struct {
    nx: u32, ny: u32,           // 宽/高（音频时 ny=1）
    is_audio: bool = false,
    id: ?[]const u8 = null,
    data: ?[]const u8 = null,   // 像素/音频样本
    allocator: ?std.mem.Allocator = null,
};
```

### MtmdContext — 多模态会话上下文

```zig
pub const MtmdContext = struct {
    mm_manager: *mm.MultiModalManager,
    caps: model.ModelCapabilities,
    n_embd_text: i32,
    tok: ?*tokenizer.Tokenizer,
    media_marker: []const u8,   // "<__media__>"
    img_beg: []const u8,        // "<|image|>"
    img_end: []const u8,
    aud_beg: []const u8,
    aud_end: []const u8,
    // ...
};
```

### 能力检测（自动）

扫描 GGUF tensor 名称识别：

| Tensor 特征 | 推断 |
|------------|------|
| `v.patch_embd.weight` / `mm.input_projection.weight` | 视觉编码器 |
| `patch_norm_1.weight` | `gemma4uv` |
| `a.conv1d.0.weight` / `a.input_projection.weight` | 音频编码器 |

---

## 视觉编码器

### 架构

- **Gemma4V / Gemma4UV**：ViT + SigLIP，2D RoPE + im2col + pooling
- 输入尺寸：896×896（默认）
- Patch size：14×14 → 64×64 patches
- Merge (pooling) 2×2 → 最终 **1024 tokens**（32×32）
- 输出维度：与 LLM 嵌入维度一致（如 1536）

### 预处理管线

```
图像文件 (JPEG/PNG/PPM)
  → stb_image 解码
  → 双线性 Resize 到 896×896
  → 转换为 F32 张量
  → 归一化 (mean/std 从 GGUF 读取)
  → vision.encode()
```

### 当前实现状态

| 组件 | 状态 |
|------|------|
| Vision Encoder 前向 | ✅ 完整 |
| 图像预处理 | ✅ 完整 |
| stb_image 集成 | ✅ 完整 |
| 占位符 token `<|image|>` 处理 | ✅ 已实现（预分词拆分） |
| 三阶段 prefill | ✅ 已实现 |
| 输出质量对比 | 🟡 待完成 |
| 多图像支持 | 🟡 待完成 |
| 动态分辨率 | 🟡 待完成 |

---

## 音频编码器

### 架构

- **Conformer**：ChunkedAttention + Relative Position Encoding + SSM Conv + GLU
- 输入：Mel 频谱 [128, n_frames]
- 子采样：Conv2D×2 (stride=2) → 时间维度 /4
- Conformer 层数：12 层
- 输出投影：1024 → LLM 嵌入维度

### 预处理管线

```
WAV 文件 (16-bit PCM)
  → loadWav() 提取 F32 单声道样本
  → 预加重 (coeff=0.97)
  → STFT: 帧长 320 (20ms@16kHz), 帧移 160 (10ms)
  → Hann 窗 + FFT (n_fft=512)
  → Mel 滤波器组 (128 bins, 80-7600Hz, HTK 尺度)
  → log10 压缩
  → Mel 频谱 [128, n_frames]
  → audio.encode()
```

### 与 llama.cpp 的关键差异

| 差异点 | llama.cpp | zllama.zig | 影响 |
|--------|-----------|------------|------|
| Q/K/V permute 布局 | (0,3,1,2) | (0,2,1,3) | ✅ 数学等价（H/B 轴交换） |
| k_scale 计算 | `logf(1+expf(1))/logf(2)` | `@log2(1+@exp(1))` | ✅ 等价 |
| norm_eps | 硬编码 1e-6 | 从 GGUF 读取 | ✅ 通常相同 |
| 输入填充 | `ggml_set_input` | 直接填充 | ✅ 功能等价 |
| clamp 操作 | 支持（可选） | 未实现 | ✅ 通常为空 |
| FFT | CPU/kissfft | macOS Accelerate | 🟡 需跨平台后备 |

### 当前实现状态

| 组件 | 状态 |
|------|------|
| Audio Encoder 前向 | ✅ 完整 |
| Mel 频谱预处理 | ✅ 完整 |
| 占位符 token `<|audio|>` 处理 | ✅ 已实现 |
| 三阶段 prefill | ✅ 已实现 |
| 端到端输出质量验证 | 🟡 待完成（需 `hello.wav` 测试） |
| 跨平台 FFT 后备 | 🟡 待完成 |

---

## 端到端推理流程

### 三阶段 Prefill（关键设计）

由于图像/音频 token 需要 **bidirectional attention**，而文本 token 需要 causal attention，因此 prefill 分三个阶段执行：

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Text prefix (causal)                              │
│   - 输入："<|turn|>user\n"                                  │
│   - KV cache 写入，使用 causal mask                         │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: Media tokens (non-causal / bidirectional)         │
│   - 视觉/音频嵌入注入                                        │
│   - 所有 media token 相互可见（无 mask）                     │
│   - KV cache 追加                                           │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: Text suffix (causal)                              │
│   - 剩余提示词（如 "Describe this image<turn|>\nmodel\n"）  │
│   - causal mask，只能看到 prefix + media + 自身左侧         │
│   - logits 从最后一个 token 采样                            │
└─────────────────────────────────────────────────────────────┘
```

每阶段使用独立的 `ggml_context` + Gallocr 复用，`ctx_kv_cache` 持久化跨阶段。该设计参考 llama.cpp 的 `llama_set_causal_attn()` per-chunk 模式。

### 关键函数

- `forwardWithEmbdOverride()`：跳过 token embedding 查表，直接注入外部嵌入
- `generateWithImage()` / `generateWithAudio()`：完整的三阶段 prefill 管线

---

## 参考资源

- [llama.cpp mtmd 实现](deps/llama.cpp/tools/mtmd/)
- [Gemma 4 技术报告](https://ai.google.dev/gemma)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [本项目的待办事项](TODO.md)
```

这份合并后的文档可以用 `docs/MULTIMODAL.md` 保存，替代原有的 MTMD.md、IMAGE.md、AUDIO.md。需要我帮你直接生成文件内容以便保存吗？
