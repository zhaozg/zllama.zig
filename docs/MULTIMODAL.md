# 多模态推理模块 — MULTIMODAL

> 参考：[llama.cpp tools/mtmd](deps/llama.cpp/tools/mtmd/)、Gemma 4 多模态实现
> 当前状态：视觉/音频编码器实现完整，三阶段 prefill 已实现，端到端推理已验证。
> **核心调用**：`src/main.zig` → `InferenceEngine.generateWithImage()` / `generateWithAudio()`。

---

## 目录

1. [概述与架构](#概述与架构)
2. [add_media：媒体处理核心流程](#add_media媒体处理核心流程)
3. [公共基础设施](#公共基础设施)
4. [视觉编码器](#视觉编码器)
5. [音频编码器](#音频编码器)
6. [对话模板与媒体嵌入](#对话模板与媒体嵌入)
7. [Token 化管线](#token-化管线)
8. [三阶段 Prefill](#三阶段-prefill)
9. [端到端推理流程](#端到端推理流程)
10. [质量对比验证方法](#质量对比验证方法)
11. [附录：llama.cpp mtmd 关键设计决策](#附录llamacpp-mtmd-关键设计决策)

---

## 概述与架构

zllama.zig 通过 `src/mtmd/` 目录下的模块实现多模态（图像+音频）推理，当前主要支持 **Gemma 4 E2B** 模型的内建编码器。入口点通过 `src/main.zig` 的 CLI 参数调度，核心逻辑集中在 `src/core/engine.zig` 的 `InferenceEngine` 中。

### 系统架构图

```
┌───────────────────────────────────────────────────────────────────────────┐
│  CLI (main.zig)                                                           │
│  --model gguf --mmproj mmproj.gguf --image x.png --audio y.wav -p "..."   │
└──────┬──────────────────────┬──────────────────────┬──────────────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Text-only    │    │ Image Pipeline   │    │ Audio Pipeline   │
│ generate()   │    │ generateWithImage│    │ generateWithAudio│
└──────────────┘    └────────┬─────────┘    └────────┬─────────┘
                             │                       │
                    ┌────────▼─────────┐    ┌────────▼─────────┐
                    │ preprocess.zig   │    │ preprocess.zig   │
                    │ loadImage()      │    │ loadWav()        │
                    │ bilinearResize   │    │ computeMelSpectro│
                    │ imageToTensor    │    │ melToTensor      │
                    └────────┬─────────┘    └────────┬─────────┘
                             │                       │
                    ┌────────▼─────────┐    ┌────────▼─────────┐
                    │ vision.zig       │    │ audio.zig        │
                    │ VisionEncoder    │    │ AudioEncoder     │
                    │ encode()         │    │ encode()         │
                    │ → embeddings     │    │ → embeddings     │
                    └────────┬─────────┘    └────────┬─────────┘
                             │                       │
                    ┌────────▼───────────────────────▼─────────┐
                    │ manager.zig                              │
                    │ MultiModalManager                        │
                    │ encodeMedia() / estimateTokenCount()     │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │ engine.zig                               │
                    │ tokenizeWithMediaPlaceholders()          │
                    │   → TokenizedSegments{prefix, media, suffix} │
                    │ multimodalPrefill() / threeStagePrefill()│
                    │   → Pass 1: text prefix (causal)         │
                    │   → Pass 2: media tokens (non-causal)    │
                    │   → Pass 3: text suffix (causal)         │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │ 自回归解码循环 (runDecodeLoop variant)   │
                    └─────────────────────────────────────────┘
```

### 核心数据流（简化）

```
用户输入 (文本 + 媒体文件)
  │
  ├─[1] 媒体预处理
  │    图像: loadImage → resize → normalize → tensor
  │    音频: loadWav → Mel spectrogram → tensor
  │
  ├─[2] 编码器前向
  │    mm_manager.encodeMedia() → 视觉/音频 embeddings
  │
  ├─[3] 对话模板渲染 (chat_template/gemma4.zig)
  │    → 含 <|image|> 或 <|audio|> 特殊 token 的格式化 prompt
  │
  ├─[4] Token 化 (tokenizeWithMediaPlaceholders)
  │    → 在 token 序列中找到特殊 token，展开为 N 个 token
  │    → 构建 prefix / media / suffix 三段结构
  │
  ├─[5] 三阶段 Prefill (threeStagePrefill)
  │    Pass 1: 文本前缀 (causal)
  │    Pass 2: 媒体 token，注入编码器输出 (non-causal)
  │    Pass 3: 文本后缀 (causal)，采样第一个生成 token
  │
  └─[6] 自回归解码
```

---

## add_media：媒体处理核心流程

### 概念来源

llama.cpp 的 `mtmd.cpp` 中定义了 `add_media()` 函数（约第 1038 行），它是多模态推理的**核心入口**。该函数做以下事情：

1. 加载媒体文件（图像/音频）
2. 预处理媒体数据（缩放、归一化、频谱提取）
3. 通过编码器前向得到 embeddings
4. 构建 token 序列，注入边界标记（`<|image|>`/`<image|>`/`<|audio|>`/`<audio|>`）
5. 调用 LLM 解码

### zllama.zig 中的对应实现

zllama.zig 将 `add_media` 的功能统一到 `src/core/engine.zig` 的 `InferenceEngine` 中：

| llama.cpp mtmd | zllama.zig | 说明 |
|---|---|---|
| `mtmd_helper_bitmap_init_from_file()` | `preprocess.loadImage()` / `preprocess.loadWav()` | 加载媒体文件 |
| 图像/音频预处理器 | `preprocess.computeMelSpectrogram()` / `bilinearResizeRGB()` | 预处理媒体数据 |
| 视觉/音频编码器 | `mm.MultiModalManager.encodeMedia()` | 编码得到 embeddings |
| `add_media()` → `add_text(img_beg)` + chunk + `add_text(img_end)` | `applyChatTemplateWithMedia()` + `tokenizeWithMediaPlaceholders()` | 在 prompt 中插入媒体标记并展开 |
| `llama_decode()` per chunk (causal/non-causal toggle) | `threeStagePrefill()` | 三阶段 prefill（prefix causal → media non-causal → suffix causal） |

### 关键差异

llama.cpp 的 `add_media()` (位于 `deps/llama.cpp/tools/mtmd/mtmd.cpp:1038`) 使用以下流程：

1. **图像**：`add_text(ctx->img_beg)` → 预处理 + 编码 → 创建 image chunk → `add_text(ctx->img_end)`
2. **音频**：`add_text(ctx->aud_beg)` → 预处理 + Mel + 编码 → 创建 audio chunk → `add_text(ctx->aud_end)`

边界标记 `img_beg`/`img_end` (如 `<|image>` / `<image|>`) 和 `aud_beg`/`aud_end` (如 `<|audio>` / `<audio|>`) 是**真实 token**，通过 `parse_special=true` 编码。它们标注了媒体嵌入在 token 序列中的开始和结束位置。

zllama.zig 当前实现使用**不同的路径**：
- 对话模板（Jinja 或预设模板）负责插入 `<|image|>` 或 `<|audio|>` 特殊 token
- `tokenizeWithMediaPlaceholders()` 在已 token 化的序列中找到这些特殊 token
- 将单个特殊 token 展开为 N 个相同的 token（N = 编码器输出 token 数）
- 三阶段 prefill 时，Pass 2 使用 `embed_override` 将占位 token 的嵌入替换为编码器输出

> **设计对等性说明**：虽然 zllama.zig 和 llama.cpp 的实现路径不同，但功能上是等价的：
> - llama.cpp 通过 `llama_batch` 的 `embd` 字段注入外部嵌入
> - zllama.zig 通过三阶段 prefill 的 `embed_override` 参数注入
> - 两者都在媒体 token 位置使用 non-causal attention
> - 两者都保持 KV cache 跨阶段持久化

---

## 公共基础设施

### Chunk 模型 — 输入的多态表示

`src/mtmd/tokenize.zig` 按 `media_marker`（默认 `<__media__>`）将输入分割为有序的 `InputChunk` 列表：

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

- 图像：`nx=width, ny=height, data=RGB字节 [w*h*3]`
- 音频：`nx=num_samples, ny=1, data=F32样本`

### MtmdContext — 多模态会话上下文

```zig
pub const MtmdContext = struct {
    mm_manager: *mm.MultiModalManager,
    caps: model.ModelCapabilities,
    n_embd_text: i32,
    tok: ?*tokenizer.Tokenizer,
    media_marker: []const u8,   // "<__media__>"
    img_beg: []const u8,        // "<|image>" (Gemma4)
    img_end: []const u8,        // "<image|>" (Gemma4)
    aud_beg: []const u8,        // "<|audio>" (Gemma4)
    aud_end: []const u8,        // "<audio|>" (Gemma4)
};
```

### 能力检测（自动）

扫描 GGUF tensor 名称识别：

| Tensor 特征 | 推断 |
|------------|------|
| `v.patch_embd.weight` / `mm.input_projection.weight` | 视觉编码器 |
| `patch_norm_1.weight` | `gemma4uv`（否则 `gemma4v`） |
| `a.conv1d.0.weight` / `a.input_projection.weight` | 音频编码器 (`gemma4a`) |

### 媒体类型系统 (chat_template/types.zig)

```zig
pub const MediaType = enum { none, image, audio };

pub const Media = struct {
    type: MediaType,
    data: union {
        image: struct { data: []const u8, width: u32, height: u32 },
        audio: struct { samples: []const f32, sample_rate: u32 },
    },
};

pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
    media: ?Media = null,
    // ...
};
```

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
图像文件 (JPEG/PNG/PPM/BMP/GIF)
  → stb_image 解码 (helper.zig / preprocess.zig)
  → 双线性 Resize 到 896×896 (bilinearResizeRGB)
  → 转换为 F32 张量 (imageToTensor)
  → 归一化 (mean/std 从 GGUF 读取)
  → vision.encode()
```

### 当前实现状态

| 组件 | 状态 | 文件 |
|------|------|------|
| Vision Encoder 前向 | ✅ 完整 | `src/mtmd/vision.zig` |
| 图像预处理 | ✅ 完整 | `src/mtmd/preprocess.zig` |
| stb_image 集成 | ✅ 完整 | `src/mtmd/preprocess.zig` |
| 占位符 token 处理 | ✅ 已实现 | `src/core/engine.zig` → `generateWithImage()` |
| 三阶段 prefill | ✅ 已实现 | `src/core/prefill.zig` |
| 端到端文本推理 | ✅ 已验证 | Gemma 4 text-only |
| 视觉测试覆盖 | ✅ 已实现 | `src/tests/test_vision.zig` (16 tests) |
| 多图像支持 | 🟡 待完成 | — |
| 动态分辨率 | 🟡 待完成 | — |

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
  → STFT: 半因果左填充 pad_left=frame_length/2
  → 帧长 400 (25ms@16kHz), 帧移 160 (10ms)
  → Hann 窗 + FFT (n_fft=512)
  → Mel 滤波器组 (128 bins, 0-8000Hz, HTK 尺度)
  → 自然对数 (ln) 压缩
  → Mel 频谱 [n_mel_bins, n_frames]
  → audio.encode()
```

### 预处理参数配置

| 参数 | 默认值 | 可从 GGUF 读取 |
|------|--------|---------------|
| `AUDIO_SAMPLE_RATE` | 16000 | ✅ `clip.audio.sample_rate` |
| `AUDIO_FRAME_LENGTH` | 400 | ✅ `clip.audio.window_length` |
| `AUDIO_HOP_LENGTH` | 160 | ✅ `clip.audio.hop_length` |
| `AUDIO_N_FFT` | 512 | ✅ `clip.audio.n_fft` |
| `AUDIO_N_MEL_BINS` | 128 | ✅ `clip.audio.num_mel_bins` |
| `AUDIO_MEL_F_MIN` | 0.0 | ✅ |
| `AUDIO_MEL_F_MAX` | 8000.0 | ✅ |
| `AUDIO_PRE_EMPHASIS` | 0.0 | ✅ (gemma4a disables pre-emphasis) |
| `AUDIO_LOG_OFFSET` | 0.001 | ✅ |

### 与 llama.cpp 的关键差异

| 差异点 | llama.cpp | zllama.zig | 影响 |
|--------|-----------|------------|------|
| Q/K/V permute 布局 | (0,3,1,2) | (0,2,1,3) | ✅ 数学等价（H/B 轴交换） |
| k_scale 计算 | `logf(1+expf(1))/logf(2)` | `@log2(1+@exp(1))` | ✅ 等价 |
| norm_eps | 硬编码 1e-6 | 从 GGUF 读取 | ✅ 通常相同 |
| 输入填充 | `ggml_set_input` | `ggml_set_input` | ✅ 一致（均采用 ggml_set_input 方式） |
| FFT | CPU/kissfft | macOS Accelerate | 🟡 需跨平台后备 |
| pre_emphasis 系数 | 0.0 (gemma4a) | 0.0 | ✅ 一致 |

### 当前实现状态

| 组件 | 状态 | 文件 |
|------|------|------|
| Audio Encoder 前向 | ✅ 完整 | `src/mtmd/audio.zig` |
| Mel 频谱预处理 | ✅ 完整 | `src/mtmd/preprocess.zig` |
| WAV 加载 | ✅ 完整 | `src/mtmd/preprocess.zig` |
| 占位符 token 处理 | ✅ 已实现 | `src/core/engine.zig` → `generateWithAudio()` |
| 三阶段 prefill | ✅ 已实现 | `src/core/prefill.zig` |
| 音频测试覆盖 | ✅ 已实现 | `src/tests/test_audio.zig` (12 tests) |
| 跨平台 FFT 后备 (非 macOS) | 🟡 待完成 | 当前仅支持 macOS Accelerate |

---

## 对话模板与媒体嵌入

### 当前实现路径

zllama.zig 使用**统一路径**处理所有类型的对话模板：

```
┌─────────────────────────────────────────────────────────────┐
│ engine.zig: applyChatTemplateWithMedia()                    │
│                                                             │
│ 1. 获取模板源 (GGUF内置 / CLI指定 / 架构默认)               │
│ 2. 创建 ChatMessage.withMedia("user", prompt, media)        │
│ 3. 模板渲染 → 含 <|image|> 或 <|audio|> 特殊 token 的字符串 │
└─────────────────────────────────────────────────────────────┘
```

模板中的媒体占位符由 `chat_template/gemma4.zig` 的 `appendMediaContent()` 函数处理，该函数将在用户消息中插入媒体标记。

### 两种标记系统

zllama.zig 中使用**统一标记系统**：

| 标记 | 用途 | 示例 | 在词表中？ | 处理方式 |
|------|------|------|-----------|---------|
| 媒体特殊 token | 对话模板中的媒体占位符 | `<\|image\|>` `<\|audio\|>` | ✅ 是 | `parse_special=true` 编码为单个 token |
| 边界标记 | 隐式定义（模板 + 展开） | `<\|image\|>` 前 ← N个token → `<image\|>` 后 | ✅ 是 | 三阶段 prefill 自动划分 prefix/media/suffix |

### Gemma 4 的边界标记

根据 llama.cpp mtmd.cpp 的 `init_vision()` 和 `init_audio()`：

| 编码器类型 | img_beg | img_end | aud_beg | aud_end |
|-----------|---------|---------|---------|---------|
| GEMMA4V / GEMMA4UV | `<\|image>` | `<image\|>` | - | - |
| GEMMA4A / GEMMA4UA | - | - | `<\|audio>` | `<audio\|>` |
| GEMMA3 / GEMMA3NV | `<start_of_image>` | `<end_of_image>` | - | - |

这些边界标记是**真实 token**（通过 tokenizer 的 `parse_special=true` 编码），在对话模板渲染时由 Jinja 模板或预设模板自动插入。

### zllama.zig 与 llama.cpp 的设计对等性

| 检查项 | llama.cpp | zllama.zig | 影响 |
|--------|-----------|------------|------|
| `add_media` 边界标记注入 | ✅ `img_beg` + chunk + `img_end` | ✅ 模板 + 展开（三等分 prefix/media/suffix） | ✅ 对等 |
| Embedding 注入 | `llama_batch.embd` | `embed_override` → `mediaForward()` | ✅ 对等 |
| Non-causal attention | `llama_set_causal_attn(false)` | Pass 2 `causal=false` | ✅ 对等 |
| Token 化方式 | `mtmd_tokenize` 分段处理 | `tokenizeWithMediaPlaceholders` 后处理展开 | 🟡 路径不同，功能对等 |
| pre_emphasis 系数 | 0.0 (gemma4a) | 0.0 | ✅ 一致 |
| FFT 跨平台 | kissfft | macOS Accelerate only | 🟡 中（仅影响非 macOS 音频） |
| 编码器数值精度 | f32 | f32 | ✅ 一致 |
| 三阶段 prefill | per-chunk causal toggle | 三阶段 context reset | ✅ 设计等价 |

---

## Token 化管线

### 核心函数：`tokenizeWithMediaPlaceholders()`

位于 `src/core/engine.zig`，是三阶段 prefill 的**关键前置步骤**：

```
输入: formatted_prompt = "<bos><|turn|>user\n<|image|>Describe this image<turn|>\n<|turn|>model\n"
      media_token_id   = 258880  (<|image|> 的 token ID)
      media_token_count = 1024    (视觉编码器输出的 token 数)

过程:
  1. tokenize(formatted_prompt, add_special=false, parse_special=true)
     → tokens = [<bos>, <|turn|>, user, \n, <|image|>, Describe, ..., <turn|>, \n, <|turn|>, model, \n]
                                          ↑ pos=4
  2. 扫描找到 media_token_id 的位置 → [4]
  3. 将位置 4 的单个 token 展开为 1024 个相同的 token
     → tokens = [<bos>, <|turn|>, user, \n, (258880)×1024, Describe, ..., <turn|>, \n, ...]

输出: TokenizedSegments {
    tokens: [u32],     // 展开后的完整 token 序列
    offsets: [{        // 媒体占位符信息
        token_offset: 4,      // 媒体 token 起始位置
        token_count: 1024,    // 媒体 token 数量
        media_type: .image,
    }],
}
```

### 为什么这样设计？

1. **`parse_special=true`** 确保 `<|image|>` 被 tokenizer 正确编码为单个特殊 token，而不是拆分为多个普通 token
2. **展开为 N 个 token** 是因为视觉编码器输出 1024 个 embedding 向量，每个都需要在 LLM 的序列中占一个位置
3. **保留原始 token ID** 允许 LLM 在 Pass 2 中仍然可以查找 per-layer embedding（用于某些需要位置嵌入的模型）

### Token 化对比：zllama.zig vs llama.cpp mtmd

| 方面 | zllama.zig | llama.cpp mtmd |
|------|-----------|---------------|
| 输入分割 | `<\|image\|>` 作为模板占位符 | `<__media__>` 作为分割标记 |
| Token 化方式 | 全量 tokenize + 后处理展开 | 分段 tokenize，按 chunk 类型 |
| 边界标记 | **无**（只有占位符 token） | `img_beg` + chunk + `img_end` |
| 媒体 chunk 表示 | 连续的 N 个相同 token ID | `InputChunk` 类型标记为 image/audio |

---

## 三阶段 Prefill

### 设计动机

由于图像/音频 token 需要 **bidirectional attention**，而文本 token 需要 causal attention，因此 prefill 分三个阶段执行。每个阶段使用独立的 `ggml_context`（通过 `context.reset()` 回收），KV cache 持久化跨阶段。

### Prefill 流程

```
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: Text prefix (causal)                                │
│   - 输入：文本前缀 token（如 "<bos><|turn|>user\n"）         │
│   - Positions: 0 .. prefix_len-1                            │
│   - causal=true，标准 causal mask                           │
│   - KV cache 写入                                           │
├─────────────────────────────────────────────────────────────┤
│ Pass 2: Media tokens (non-causal / bidirectional)           │
│   - 输入：媒体占位符 token × N（如 258880 × 1024）           │
│   - Positions: prefix_len .. prefix_len + n_media - 1       │
│   - causal=false，所有 media token 相互可见                  │
│   - 使用 embed_override 注入编码器输出（替换 token embedding）│
│   - mediaForward() — 跳过 embedding 查表，直接使用外部嵌入    │
│   - KV cache 追加                                           │
├─────────────────────────────────────────────────────────────┤
│ Pass 3: Text suffix (causal)                                │
│   - 输入：文本后缀 token（如 "Describe this image\n..."）     │
│   - Positions: prefix_len + n_media .. total - 1            │
│   - causal=true，可以看到 prefix + media + 自身左侧         │
│   - logits 从最后一个 token 采样 → 第一个生成 token          │
│   - KV cache 追加                                           │
└─────────────────────────────────────────────────────────────┘
```

### 实现细节

每阶段执行：
1. `context.reset()` — 释放所有之前的 tensor/节点
2. `setNoAlloc(false)` — 允许 tensor 分配
3. 构建计算图（tensor、graph nodes）
4. `setNoAlloc(true)` — 锁定 context
5. Gallocr 分配 + `graph.compute()`

此设计参考 llama.cpp 的 `llama_set_causal_attn()` per-chunk 模式。

### 关键函数

- **`threeStagePrefill()`** (`src/core/prefill.zig`): 可复用的三阶段 prefill 辅助函数
- **`mediaForward()`** (`src/models/gemma4.zig`): 媒体 token 的前向传播（non-causal，embed_override）
- **`forwardWithEmbdOverride()`**: 跳过 token embedding 查表，直接注入外部嵌入

### 非因果注意力判定

根据 llama.cpp `mtmd_decode_use_non_causal()`：

```cpp
bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    auto proj_type = ctx->proj_type_v();
    if (chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        proj_type = ctx->proj_type_a();
    }
    switch (proj_type) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_GEMMA4UV:
            return true;
        default:
            return false;
    }
}
```

Gemma 4 的视觉和音频编码器输出都需要 non-causal 注意力。

### 位置编码

| 位置类型 | 描述 | 适用模型 |
|---------|------|---------|
| `MTMD_POS_TYPE_NORMAL` | 顺序位置：pos = pos_0 + i | Gemma 4, LLaMA 等 |
| `MTMD_POS_TYPE_MROPE` | M-RoPE：t=pos_0, x=pos_0+col, y=pos_0+row | Qwen2-VL |
| `MTMD_POS_TYPE_HUNYUANVL` | HunyuanVL 布局：BOI/EOI + 行新行 | HunyuanVL |

Gemma 4 使用 `MTMD_POS_TYPE_NORMAL`，所有维度（t, x, y, z）都使用相同的顺序位置。

---

## 端到端推理流程

### generateWithImage() 完整流程

位于 `src/core/engine.zig:634`：

```
1. 验证 mm_manager 和 capabilities.has_vision
2. 加载图像: preprocess.loadImage() → ProcessedImage
3. 编码: mm_manager.encodeMedia() → vision_embeddings [n_embd × n_vision_tokens]
4. 查找媒体 token ID: tok.textToToken("<|image|>")
5. 对话模板: applyChatTemplateWithMedia(prompt, Media{.image})
   → formatted_prompt = "<bos><|turn|>user\n<|image|>Describe this image<turn|>\n..."
6. Token 化: tokenizeWithMediaPlaceholders(formatted_prompt, image_token_id, n_vision_tokens)
   → 展开 <|image|> 为 n_vision_tokens 个 token
7. 三阶段 Prefill: threeStagePrefill()
8. 自回归解码: runDecodeLoop 变体
```

### generateWithAudio() 完整流程

位于 `src/core/engine.zig:702`：

```
1. 验证 mm_manager 和 capabilities.has_audio
2. 加载 WAV: preprocess.loadWav() → WavInfo + F32 samples
3. Mel 频谱: preprocess.computeMelSpectrogram() → ProcessedAudio
4. 编码: mm_manager.encodeMedia() → audio_embeddings [n_embd × n_audio_tokens]
5. 查找媒体 token ID: tok.textToToken("<|audio|>")
6. 对话模板: applyChatTemplateWithMedia(prompt, Media{.audio})
7. Token 化: tokenizeWithMediaPlaceholders(formatted_prompt, audio_token_id, n_audio_tokens)
8. 三阶段 Prefill: threeStagePrefill()
9. 自回归解码
```

### generate() 文本流程（对比）

位于 `src/core/engine.zig:439`：

```
1. 对话模板: applyChatTemplate(prompt)
2. Token 化: tok.encode() — 标准编码，无需占位符展开
3. 文本 Prefill: textPrefill() — 单次 causal forward
4. 自回归解码: runDecodeLoop
```

---

## 质量对比验证方法

### 注意力掩码验证 ✅ 已确认

三阶段 prefill（`src/core/prefill.zig`）正确处理不同阶段的注意力掩码：

| 阶段 | 注意力类型 | causal | 实现位置 |
|------|-----------|--------|---------|
| Pass 1: 文本前缀 | Causal | `true` | `model.buildGraph()` → `transformerForward(..., true)` |
| Pass 2: 媒体 tokens | Bidirectional | `false` | `mediaForward()` → `transformerForward(..., false)` |
| Pass 3: 文本后缀 | Causal + 历史可见 | `true` | `model.buildGraph()` → `transformerForward(..., true)` |

### 生成参考 logits

使用 `tools/mtmd_ref_logits.cpp` 从 llama.cpp 生成参考 logits：

```bash
# 视觉
mtmd-ref-logits -m model.gguf --mmproj mmproj.gguf --image hello.png -p ":" -o ref_vision.bin

# 音频
mtmd-ref-logits -m model.gguf --mmproj mmproj.gguf --audio hello.wav -p ":" -o ref_audio.bin
```

参考工具内部流程（`tools/mtmd_ref_logits.cpp`）：
1. `llama_model_load_from_file()` + `llama_init_from_model()`
2. `mtmd_init_from_file()` — 加载 mmproj
3. `mtmd_helper_bitmap_init_from_file()` — 加载媒体
4. `mtmd_tokenize()` — 按 chunk tokenize
5. `mtmd_helper_eval_chunks()` — 自动编码 + llama_decode
6. `llama_get_logits_ith(lctx, -1)` — 获取最后一个 token 的 logits
7. 保存为 f32 二进制文件

### zllama.zig 端到端验证

```bash
# 视觉
zig-out/bin/zllama -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/.cache/models/mmproj-BF16.gguf \
  --image ~/.cache/models/hello.png \
  -p "Describe this image" -n 5

# 音频
zig-out/bin/zllama -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/.cache/models/mmproj-BF16.gguf \
  --audio ~/.cache/models/hello.wav \
  -p "Transcribe this audio to English text. Output the full transcription." -n 20
```

### 已知调试方法

使用 `-d` 启用 debug 日志可以看到：
- 嵌入维度检查 (encoder output vs model n_embd)
- 嵌入数值验证 (all_zero? has_nan?)
- Token 序列验证 (prefix/media/suffix 的 token ID 和位置)
- 每阶段的 KV cache 长度
- 每阶段的执行时间

---

## 附录：llama.cpp mtmd 关键设计决策

### 1. `add_media` 的核心模式

```
add_media(bitmaps):
  for each bitmap:
    if audio: preprocess → Mel → encode → embeddings
    if image: preprocess → resize → encode → embeddings

  build token sequence:
    for each segment:
      if text:  tokenize text → add_text(text)
      if image: add_text(img_beg) + add_image_tokens(n) + add_text(img_end)
      if audio: add_text(aud_beg) + add_audio_tokens(n) + add_text(aud_end)

  eval chunks (per chunk causal/non-causal):
    for each chunk:
      llama_set_causal_attn(causal)
      llama_decode(chunk_tokens)
```

### 2. 媒体标记 vs 占位符

| 标记 | 用途 | 示例 | 在词表中？ | 处理方式 |
|------|------|------|-----------|---------|
| `media_marker` | 输入分割标记 | `<__media__>` | ❌ 否 | 在 `mtmd_tokenizer` 中分割输入 |
| `img_beg`/`img_end` | 图像边界标记 | `<\|image>` / `<image\|>` | ✅ 是 | 作为特殊 token 编码 |
| `aud_beg`/`aud_end` | 音频边界标记 | `<\|audio>` / `<audio\|>` | ✅ 是 | 作为特殊 token 编码 |

**关键设计**：
- `media_marker` 是**分割标记**，不在 tokenizer 词表中，仅用于分割输入
- `img_beg`/`img_end` 是**特殊 token**，在 tokenizer 词表中，通过 `parse_special=true` 编码
- 边界标记确保媒体嵌入在 token 序列中有明确的开始和结束位置

### 3. 编码器类型检测

llama.cpp 通过 `clip_get_projector_type()` 检测编码器类型：

```cpp
case PROJECTOR_TYPE_GEMMA4V:
case PROJECTOR_TYPE_GEMMA4UV:
    img_beg = "<|image>";
    img_end = "<image|>";
    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
    break;
```

### 4. 音频预处理器类型

| 类型 | 边界标记 | 预处理器 |
|------|---------|---------|
| GEMMA4A | `<\|audio>` / `<audio\|>` | `mtmd_audio_preprocessor_gemma4a` |
| GEMMA4UA | `<\|audio>` / `<audio\|>` | `mtmd_audio_preprocessor_gemma4ua` |
| QWEN2A | `<\|audio_bos\|>` / `<\|audio_eos\|>` | `mtmd_audio_preprocessor_whisper` |

### 5. 非因果注意力判定

Gemma 4 的视觉和音频编码器输出都需要 non-causal 注意力。这是通过 `mtmd_decode_use_non_causal()` 函数判定的，在构建计算图时根据 chunk 类型设置 causal 标志。

### 6. 位置编码

Gemma 4 使用 `MTMD_POS_TYPE_NORMAL`，所有维度（t, x, y, z）都使用相同的顺序位置。这与 Qwen2-VL 的 M-RoPE 不同，后者需要为每个 token 计算 (t, x, y) 三维位置。

### 7. zllama.zig 与 llama.cpp 的当前差距分析

| 检查项 | llama.cpp | zllama.zig | 影响 |
|--------|-----------|------------|------|
| `add_media` 边界标记注入 | ✅ `img_beg` + chunk + `img_end` | ✅ 模板渲染 + 三等分 prefix/media/suffix | ✅ 设计对等 |
| 预加重系数 | 0.0 (gemma4a) | 0.0 | ✅ 一致 |
| FFT 跨平台 | kissfft | macOS Accelerate only | 🟡 中（仅影响非 macOS 音频） |
| Token 化方式 | `mtmd_tokenize` 分段处理 | `tokenizeWithMediaPlaceholders` 后处理展开 | 🟡 路径不同，功能对等 |
| 编码器数值精度 | f32 | f32 | ✅ 一致 |
| 三阶段 prefill | per-chunk causal toggle | 三阶段 context reset | ✅ 设计等价 |

### 8. 独立工具已移除

音频/视觉推理通过统一的 `zllama` CLI 入口使用：
其功能完全由 `src/core/engine.zig` 中的 `InferenceEngine.generateWithAudio()` 和 `InferenceEngine.generateWithImage()` 提供。

```bash
# 视觉推理
zllama -m model.gguf --mmproj mmproj.gguf --image hello.png -p "Describe this image" -n 100

# 音频推理
zllama -m model.gguf --mmproj mmproj.gguf --audio hello.wav -p "Transcribe this audio" -n 100
```

### 9. 测试覆盖

多模态处理现在有专门的单元测试：

| 测试文件 | 测试数 | 覆盖范围 |
|----------|--------|---------|
| `src/tests/test_audio.zig` | 12 | 占位符扫描、展开、音频 token 处理、Mel 频谱计算 |
| `src/tests/test_vision.zig` | 16 | 占位符扫描、展开、图像 token 处理、双线性缩放、图像归一化 |
| `src/tests/test_mtmd.zig` | 10 | Bitmap、InputChunk、tokenize、Caps |

运行方式：
```bash
zig build test           # 全部测试 (含音频/视觉)
zig build test-audio     # 仅音频测试
zig build test-vision    # 仅视觉测试
```

---

## 参考资源

- [llama.cpp mtmd 实现](deps/llama.cpp/tools/mtmd/)
- [Gemma 4 技术报告](https://ai.google.dev/gemma)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [mtmd 参考 logits 生成工具](tools/mtmd_ref_logits.cpp)
- [本项目的待办事项](TODO.md)
