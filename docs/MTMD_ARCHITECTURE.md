# MTMD 多模态解码模块架构

> **MTMD** = Multi-Modal Decoding（多模态解码）
>
> 参考实现：llama.cpp `tools/mtmd/`（mtmd.h, mtmd.cpp, mtmd-audio.cpp, clip.cpp, gemma4v.cpp, gemma4a.cpp）
> 当前状态：视觉/音频编码器实现完整，三阶段 prefill 已实现，端到端推理已验证（audio ✅, vision ✅）。

---

## 目录

1. [设计目标与原则](#1-设计目标与原则)
2. [系统架构总览](#2-系统架构总览)
3. [模块层次结构](#3-模块层次结构)
4. [核心数据结构](#4-核心数据结构)
5. [音频处理流水线](#5-音频处理流水线)
6. [视觉处理流水线](#6-视觉处理流水线)
7. [Token 化与 Chunk 管理](#7-token-化与-chunk-管理)
8. [三阶段 Prefill 机制](#8-三阶段-prefill-机制)
9. [媒体标记系统](#9-媒体标记系统)
10. [与 llama.cpp 的对齐策略](#10-与-llamacpp-的对齐策略)
11. [调试与验证工具](#11-调试与验证工具)
12. [已知问题与待办](#12-已知问题与待办)

---

## 1. 设计目标与原则

### 1.1 核心目标

为 `zllama.zig` 提供**生产级、高性能、跨平台**的多模态推理能力，支持 **Gemma 4 E2B** 模型的内建视觉和音频编码器。

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **模块化流水线** | 每个媒体类型（音频/视觉）拆分为独立子模块，各阶段职责清晰 |
| **与 llama.cpp 对齐** | 每个模块标注参考的 llama.cpp 源文件和行号，便于交叉验证 |
| **零成本抽象** | 使用 Zig 的 `comptime` 和泛型回调，避免运行时开销 |
| **内存安全** | 使用 Arena 分配器管理临时内存，`defer` 确保资源释放 |
| **可调试性** | 每个阶段可输出中间结果，支持与 llama.cpp 逐阶段对比 |

### 1.3 与 llama.cpp 的架构差异

| 方面 | llama.cpp | zllama.zig |
|------|-----------|------------|
| 语言 | C++ | Zig |
| 编码器管理 | `mtmd_context` + `clip` 分离 | `MultiModalManager` 统一管理 |
| 音频预处理 | `mtmd_audio_preprocessor_gemma4a` 类 | `pipeline.zig` 编排器 + 回调 |
| FFT 实现 | 自定义 C++ FFT | Apple Accelerate vDSP（macOS） |
| 视觉编码器 | `clip.cpp` + `gemma4v.cpp` | `vision/encoder.zig` 统一封装 |
| 内存管理 | 手动 new/delete | Arena + defer |

---

## 2. 系统架构总览

### 2.1 高层架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLI (main.zig)                                                          │
│  --model gguf --mmproj mmproj.gguf --image x.png --audio y.wav -p "..."  │
└──────┬──────────────────────┬──────────────────────┬─────────────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Text-only    │    │ Image Pipeline   │    │ Audio Pipeline   │
│ generate()   │    │ generateWithImage│    │ generateWithAudio│
└──────────────┘    └────────┬─────────┘    └────────┬─────────┘
                             │                       │
                    ┌────────▼─────────┐    ┌────────▼─────────┐
                    │ mtmd/mod.zig     │    │ mtmd/mod.zig     │
                    │ MultiModalManager│    │ MultiModalManager│
                    │ encodeMedia()    │    │ encodeMedia()    │
                    └────────┬─────────┘    └────────┬─────────┘
                             │                       │
                    ┌────────▼─────────┐    ┌────────▼─────────┐
                    │ vision/          │    │ audio/           │
                    │ pipeline.zig     │    │ pipeline.zig     │
                    │ → encoder.zig    │    │ → framing.zig    │
                    │   (ViT forward)  │    │ → mel.zig        │
                    │ → postprocess    │    │ → encoder.zig    │
                    │   (projection)   │    │   (Conformer)    │
                    └────────┬─────────┘    └────────┬─────────┘
                             │                       │
                    ┌────────▼───────────────────────▼─────────┐
                    │ engine.zig                               │
                    │ tokenizeWithMediaPlaceholders()          │
                    │   → TokenizedSegments{prefix, media, suffix} │
                    │ threeStagePrefill()                       │
                    │   → Pass 1: text prefix (causal)         │
                    │   → Pass 2: media tokens (non-causal)    │
                    │   → Pass 3: text suffix (causal)         │
                    └────────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │ 自回归解码循环 (runDecodeLoop variant)    │
                    └──────────────────────────────────────────┘
```

### 2.2 核心数据流

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
       → 标准 token 生成循环
```

---

## 3. 模块层次结构

### 3.1 文件结构

```
src/mtmd/
├── mod.zig              # 模块根：MtmdContext、MultiModalManager、基础类型
├── helper.zig           # 工具函数：chunk 评估、文件加载、调试输出
├── tokenize.zig         # 多模态 token 化：将文本+媒体标记拆分为 chunks
├── preprocess.zig       # 图像预处理（resize + 归一化）
├── fft.zig              # Apple Accelerate vDSP FFT 封装
│
├── audio/               # 音频处理子模块
│   ├── mod.zig          # 模块入口，重新导出所有公开 API
│   ├── config.zig       # 配置参数（预处理 + 编码器 + ChunkedAttention）
│   ├── types.zig        # 阶段间传递的数据结构
│   ├── loader.zig       # WAV 文件加载与重采样
│   ├── framing.zig      # 分帧 + Hann 窗口
│   ├── mel.zig          # Mel 滤波器组（HTK scale）
│   ├── log_transform.zig # 自然对数变换
│   ├── encoder.zig      # Conformer 编码器（12 层 USM 风格）
│   ├── postprocess.zig  # 后处理（softcapping、melToTensor）
│   └── pipeline.zig     # 流水线编排器
│
└── vision/              # 视觉处理子模块
    ├── mod.zig          # 模块入口，重新导出所有公开 API
    ├── config.zig       # 配置参数（ViT 超参数）
    ├── types.zig        # ViT 层权重和编码器权重类型
    ├── loader.zig       # 从 GGUF 加载视觉编码器权重
    ├── preprocess.zig   # 图像归一化（standard/siglip/passthrough）
    ├── encoder.zig      # ViT 编码器（Gemma4V / Gemma4UV）
    ├── postprocess.zig  # 后处理（标准化 + 投影到 LLM 空间）
    └── pipeline.zig     # 流水线编排器
```

### 3.2 模块依赖关系

```
mtmd/mod.zig
  ├── audio/     (通过 @import("audio"))
  │   ├── config.zig
  │   ├── types.zig
  │   ├── loader.zig
  │   ├── framing.zig
  │   ├── mel.zig
  │   ├── log_transform.zig
  │   ├── encoder.zig
  │   ├── postprocess.zig
  │   └── pipeline.zig
  ├── vision/    (通过 @import("vision"))
  │   ├── config.zig
  │   ├── types.zig
  │   ├── loader.zig
  │   ├── preprocess.zig
  │   ├── encoder.zig
  │   ├── postprocess.zig
  │   └── pipeline.zig
  ├── helper.zig
  ├── tokenize.zig
  ├── preprocess.zig
  └── fft.zig
```

---

## 4. 核心数据结构

### 4.1 `MultiModalManager` — 多模态管理器

**位置**: `src/mtmd/mod.zig:205`

协调音频编码器和视觉编码器的加载与推理，提供统一的多模态输入处理接口。

```zig
pub const MultiModalManager = struct {
    allocator: std.mem.Allocator,
    capabilities: model.ModelCapabilities,
    audio_encoder: ?audio.AudioEncoder = null,
    vision_encoder: ?vision.VisionEncoder = null,
    // ...
};
```

**关键方法**:
- `detectFromGGUF()`: 从 GGUF 元数据检测多模态能力（检查 `v.patch_embd.weight`、`a.conv1d.0.weight` 等张量），同时根据编码器类型填充 `caps.special_tokens`
- `init()`: 初始化编码器（从 mmproj 文件加载权重）
- `encodeMedia()`: 编码单个多模态输入，返回嵌入 tokens
- `estimateTokenCount()`: 估算多模态输入的 token 数量
- `resolveImageMarkers()`: 从 `caps.special_tokens` 解析图像开始/结束标记（优先 dynamic，fallback legacy）
- `resolveAudioMarkers()`: 从 `caps.special_tokens` 解析音频开始/结束标记（优先 dynamic，fallback legacy）

### 4.2 `MtmdContext` — 多模态上下文

**位置**: `src/mtmd/mod.zig:384`

高层多模态上下文，包装 `MultiModalManager` 并提供媒体标记管理、Chunk 化输入处理、能力检测与查询。

```zig
pub const MtmdContext = struct {
    allocator: std.mem.Allocator,
    mm_manager: *MultiModalManager,
    caps: model.ModelCapabilities,
    params: ContextParams,
    n_embd_text: i32,
    tok: ?*tokenizer.Tokenizer = null,
    media_marker: []const u8,       // 默认 "<__media__>"
    img_beg: []const u8,            // 图像开始标记（由 caps.special_tokens 动态解析）
    img_end: []const u8,            // 图像结束标记（由 caps.special_tokens 动态解析）
    aud_beg: []const u8,            // 音频开始标记（由 caps.special_tokens 动态解析）
    aud_end: []const u8,            // 音频结束标记（由 caps.special_tokens 动态解析）
    pos_type: PosType = .normal,    // 位置编码类型
    output_embd: ?[]f32 = null,     // 编码器输出缓存
};
```

### 4.3 `InputChunks` — 输入 Chunk 序列

**位置**: `src/mtmd/mod.zig:136`

将混合输入（文本 + 图像 + 音频）拆分为有序的 Chunk 序列，每个 Chunk 包含类型和对应的 token 数据。

```zig
pub const InputChunk = struct {
    chunk_type: ChunkType,          // .text | .image | .audio
    tokens_text: ?[]const i32 = null,
    tokens_image: ?ImageTokens = null,
    tokens_audio_n: u32 = 0,
    id: ?[]const u8 = null,
};
```

### 4.4 `Bitmap` — 媒体数据载体

**位置**: `src/mtmd/mod.zig:52`

统一表示图像和音频的原始数据。图像使用 `nx/ny` 表示宽高，音频使用 `nx` 表示样本数、`ny=1`、`is_audio=true`。

```zig
pub const Bitmap = struct {
    nx: u32,
    ny: u32,
    is_audio: bool = false,
    id: ?[]const u8 = null,
    data: ?[]const u8 = null,
    allocator: ?std.mem.Allocator = null,
};
```

### 4.5 `MediaInput` — 统一媒体输入

**位置**: `src/mtmd/mod.zig:182`

`encodeMedia()` 的统一输入类型，支持 text/image/audio 三种媒体类型。

```zig
pub const MediaInput = struct {
    media_type: MediaType,
    text: ?[]const u8 = null,
    image_data: ?[]const u8 = null,
    image_width: u32 = 0,
    image_height: u32 = 0,
    mel_data: ?[]const f32 = null,
    mel_bins: u32 = 0,
    mel_frames: u32 = 0,
    audio_length_sec: f32 = 0,
};
```

---

## 5. 音频处理流水线

### 5.1 流水线阶段

参考: llama.cpp `mtmd-audio.cpp` (`mtmd_audio_preprocessor_gemma4a`)

```
WAV 文件
  │
  ├─[1] loader.zig: loadWav()
  │     解析 WAV header，提取 16-bit PCM 数据
  │     转换为 F32 单声道（stereo 取平均）
  │     输出: samples (f32), sample_rate
  │
  ├─[2] loader.zig: resample() [可选]
  │     线性插值重采样到 16kHz
  │     输出: resampled (f32)
  │
  ├─[3] pipeline.zig: processPcmSamples()
  │     ├─ mel.zig: computeFilterbank()
  │     │   预计算 HTK Mel 滤波器组 [128, 257]
  │     │   参数: n_mel_bins=128, n_fft=512, f_min=0, f_max=8000
  │     │   slaney_area_norm=false
  │     │
  │     ├─ framing.zig: computeHannWindow()
  │     │   预计算周期性 Hann 窗口 [512]（零填充到 FFT 大小）
  │     │   window[i] = 0.5 - 0.5*cos(2π*i/320)
  │     │
  │     ├─ fft.zig: AccelFFT.init()
  │     │   初始化 Apple vDSP FFT 引擎 (n=512, log2n=9)
  │     │
  │     ├─ framing.zig: frameAudioWithCallback()
  │     │   半因果填充（左填充 160 个零）
  │     │   30 秒分块处理
  │     │   对每帧: 加窗 → 回调
  │     │
  │     └─ 回调内:
  │        ├─ fft.zig: powerSpectrum()
  │        │   vDSP_fft_zrip → 幅度谱 |X| (use_magnitude=true)
  │        │   缩放因子 0.5（匹配 vDSP 前向 FFT 约定）
  │        │
  │        ├─ mel.zig: applyFilterbankSingle()
  │        │   频谱 → Mel 能量 [128 bins]
  │        │
  │        └─ log_transform.zig: applyLogTransform()
  │           自然对数: log(max(mel, 0.001))
  │           输出: log-mel 谱 [128, n_frames] (mel-major)
  │
  ├─[4] postprocess.zig: melToTensor()
  │     将 log-mel 数据复制到 ggml F32 张量 [n_frames, 128]
  │
  ├─[5] encoder.zig: AudioEncoder.encode()
  │     ├─ Conv1D x2 (子采样, 通过 ggml_conv_2d 实现)
  │     │   Conv2D_0: [3,3,1,128] stride=2 → [64, T/2, 128]
  │     │   Conv2D_1: [3,3,128,32] stride=2 → [32, T/4, 32]
  │     │   每层后接 LayerNorm + ReLU
  │     ├─ Flatten
  │     ├─ Input Projection (线性层): 1024 → n_embd
  │     ├─ 12 层 Conformer:
  │     │   FFN → Self-Attention (chunked, causal) → Causal Conv1D → FFN → Norm
  │     │   注意力: 全自注意力 + 正弦 RPE + 滑动窗口掩码 (24)
  │     │   Softcap: 50.0
  │     └─ Output Projection: n_embd → n_output_embd → RMSNorm
  │
  └─[6] 嵌入后处理
       应用 logit softcapping (50.0)
       输出: [T', n_output_embd] 音频嵌入
```

### 5.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `sample_rate` | 16000 Hz | 目标采样率 |
| `frame_length` | 320 | 窗口长度（20ms @ 16kHz） |
| `hop_length` | 160 | 帧移（10ms @ 16kHz） |
| `n_fft` | 512 | FFT 点数（2 的幂） |
| `n_mel_bins` | 128 | Mel 滤波器组数量 |
| `mel_f_min` | 0.0 Hz | Mel 最低频率 |
| `mel_f_max` | 8000 Hz | Mel 最高频率（sample_rate/2） |
| `mel_floor` | 0.001 | 对数偏移（防 log(0)） |
| `pre_emphasis` | 0.0 | 预加重系数（gemma4a 不使用） |
| `use_magnitude` | true | 使用幅度谱（非功率谱） |
| `use_natural_log` | true | 使用自然对数 |
| `mel_scale` | HTK | HTK Mel 刻度 |
| `slaney_area_norm` | false | 不进行面积归一化 |

### 5.3 Conformer 编码器架构

**位置**: `src/mtmd/audio/encoder.zig`

参考: llama.cpp `tools/mtmd/models/gemma4a.cpp`

```
输入: [T, 128] log-mel 谱
  │
  ├─ Conv1D x2 (通过 ggml_conv_2d, kernel=3, stride=2, padding=1)
  │   128 → 64 (时间维度 T → T/2)
  │   64 → 32 (时间维度 T/2 → T/4)
  │   每层后接 LayerNorm + ReLU
  │
  ├─ Flatten (时间维度展平)
  │   32 × (T/4) → 1024
  │
  ├─ Input Projection: 1024 → n_embd (如 512)
  │
  ├─ 12x Conformer Block:
  │   ├─ FFN (Swish + Linear): n_embd → 4*n_embd → n_embd
  │   ├─ Multi-Head Self-Attention (n_head heads, d_head=n_embd/n_head)
  │   │   ├─ Chunked attention (chunk_size=12)
  │   │   ├─ 正弦 RPE (13 positions)
  │   │   ├─ 滑动窗口掩码 (context=24)
  │   │   └─ Softcap (50.0)
  │   ├─ Causal Conv1D (kernel=3)
  │   ├─ FFN (Swish + Linear): n_embd → 4*n_embd → n_embd
  │   └─ RMSNorm (eps=1e-6)
  │
  ├─ Output Projection: n_embd → n_output_embd
  │
  └─ RMSNorm (eps=1e-6)
      输出: [T'', n_output_embd] 音频嵌入
```

### 5.4 分块注意力（Chunked Attention）

**位置**: `src/mtmd/audio/config.zig` (`ChunkedAttentionParams`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_size` | 12 | 块大小 |
| `max_past_horizon` | 12 | 最大过去视野 |
| `context_size` | 24 | 上下文大小 = chunk_size + max_past_horizon |
| `rpe_positions` | 13 | RPE 位置数 = max_past_horizon + 1 |
| `softcap` | 50.0 | Softcap 值 |
| `q_scale_factor` | 1.0 | Q 缩放因子 |
| `k_scale_factor` | 1.0 | K 缩放因子 |

---

## 6. 视觉处理流水线

### 6.1 流水线阶段

参考: llama.cpp `tools/mtmd/clip.cpp`, `gemma4v.cpp`, `gemma4uv.cpp`

```
图像文件 (PNG/JPEG/BMP/GIF)
  │
  ├─[1] helper.zig: bitmapInitFromFile()
  │     通过 stb_image 解码为 RGB 像素数据
  │     输出: Bitmap { nx, ny, data }
  │
  ├─[2] preprocess.zig: loadImage() / fromRawRGB()
  │     双线性插值缩放到目标尺寸（如 224x224）
  │     输出: ProcessedImage { data, width, height }
  │
  ├─[3] preprocess.zig: imageToTensor() / normalizeToTensor()
  │     归一化模式:
  │     - SigLIP: pixel/255.0 * 2 - 1
  │     - Standard: (pixel/255.0 - mean) / std
  │     输出: ggml F32 张量 [3, height, width]
  │
  ├─[4] encoder.zig: VisionEncoder.encode()
  │     ├─ Patch Embedding (卷积投影)
  │     │   输入: [3, 224, 224] → 输出: [768, 196] (14x14 patches)
  │     │
  │     ├─ Patch 归一化 (Gemma4UV 特有)
  │     │
  │     ├─ 位置编码 (可学习 2D 位置嵌入)
  │     │
  │     ├─ 16x ViT Block:
  │     │   ├─ RMSNorm → Multi-Head Self-Attention (12 heads)
  │     │   ├─ RMSNorm → FFN (SiLU/GELU)
  │     │   └─ 残差连接
  │     │
  │     ├─ Pooling (平均池化下采样, n_merge=2)
  │     │
  │     └─ Output: [768, n_tokens]
  │
  ├─[5] postprocess.zig: standardize()
  │     应用 std_bias 和 std_scale
  │
  └─[6] postprocess.zig: projectToLLM()
       ├─ RMSNorm (eps=1e-6)
       ├─ mm_soft_emb_norm_w 缩放
       └─ mm_input_proj_w 线性投影: 768 → 1536
           输出: [1536, n_tokens] 视觉嵌入
```

### 6.2 关键参数

| 参数 | Gemma4V | Gemma4UV | 说明 |
|------|---------|----------|------|
| `image_size` | 224 | 224 | 输入图像尺寸 |
| `patch_size` | 16 | 16 | Patch 大小 |
| `n_embd` | 768 | 768 | 嵌入维度 |
| `n_head` | 12 | 12 | 注意力头数 |
| `n_layer` | 16 | 16 | ViT 层数 |
| `n_ff` | 3072 | 3072 | FFN 中间维度 |
| `n_output_embd` | 1536 | 1536 | 输出投影维度 |
| `n_merge` | 2 | 2 | Pooling kernel size |
| `norm_eps` | 1e-6 | 1e-6 | 归一化 epsilon |
| `ffn_op` | SiLU | SiLU | FFN 激活函数 |
| Patch Norm | 无 | 有 | Gemma4UV 特有 |

### 6.3 视觉编码器变体

**位置**: `src/mtmd/vision/config.zig` (`EncoderType`)

```zig
pub const EncoderType = enum {
    gemma4v,   // 标准 ViT + SigLIP
    gemma4uv,  // 统一视觉编码器（带额外 patch 归一化）
};
```

检测逻辑（`MultiModalManager.detectFromGGUF`）：
- 如果存在 `v.patch_norm.1.weight` 或 `patch_norm_1.weight` → `gemma4uv`
- 否则 → `gemma4v`

### 6.4 可变分辨率支持

视觉编码器支持可变分辨率输入，使用基于视觉 token 预算的动态扩展方式处理图像：

- `estimateOutputTokens()`: 估算给定分辨率图像的 token 数量
- `bestResolution()`: 计算视觉 token 预算下的最佳图像分辨率

---

## 7. Token 化与 Chunk 管理

### 7.1 多模态 Token 化流程

**位置**: `src/mtmd/tokenize.zig`

参考: llama.cpp `mtmd.h` (`mtmd_tokenize`)

```
输入文本: "Describe <__media__> in detail"
媒体列表: [Bitmap(image, 224, 224)]
  │
  ├─ 在文本中查找媒体标记 "<__media__>"
  │
  ├─ 分割为 chunks:
  │   [0] text: "Describe "
  │   [1] image: { nx=196, ny=1, pos=.normal }
  │   [2] text: " in detail"
  │
  ├─ 文本 chunks 通过 tokenizer 编码
  │   （无 BOS/EOS，但解析特殊 token）
  │
  ├─ 图像 chunks 添加开始/结束标记:
  │   "<|image>" + image_tokens + "<image|>"
  │
  ├─ 音频 chunks 添加开始/结束标记:
  │   "<|audio>" + audio_tokens + "<audio|>"
  │
  └─ 返回 InputChunks 序列
```

### 7.2 Chunk 合并策略

相邻的文本 chunks 会被自动合并，减少碎片化：

```zig
// 如果上一个 chunk 也是 text 类型，合并 token 数组
if (chunks.entries.items.len > 0 and
    chunks.entries.items[chunks.entries.items.len - 1].chunk_type == .text) {
    // 合并 token 数组
}
```

### 7.3 媒体标记解析

**位置**: `src/mtmd/mod.zig:358-378`

根据编码器类型自动确定媒体标记：

| 编码器类型 | 图像标记 | 音频标记 |
|-----------|---------|---------|
| gemma4v / gemma4uv | `<|image>` ... `<image|>` | — |
| gemma4a / gemma4ua | — | `<|audio>` ... `<audio|>` |
| 其他 | `<start_of_image>` ... `<end_of_image>` | 空 |

---

## 8. 三阶段 Prefill 机制

### 8.1 设计动机

多模态推理中，媒体 token（图像/音频嵌入）与文本 token 的注意力模式不同：
- **文本 token**: 因果注意力（只能看过去）
- **媒体 token**: 非因果注意力（可以看全部媒体 token，但不能看未来文本）

三阶段 Prefill 将输入拆分为三段，分别处理：

### 8.2 三阶段流程

**位置**: `src/core/engine.zig` (`threeStagePrefill`)

```
输入: [prefix_tokens] [media_tokens] [suffix_tokens]
  │
  ├─ Pass 1: 文本前缀 (causal)
  │   ┌──────────────────────────────────────┐
  │   │ prefix_tokens → 标准因果 prefill     │
  │   │ KV Cache 更新到 prefix 末尾          │
  │   └──────────────────────────────────────┘
  │
  ├─ Pass 2: 媒体 token (non-causal)
  │   ┌──────────────────────────────────────┐
  │   │ media_tokens → 非因果 prefill        │
  │   │ 注入编码器输出作为嵌入                │
  │   │ 媒体 token 之间可以互相看             │
  │   │ KV Cache 更新到 media 末尾           │
  │   └──────────────────────────────────────┘
  │
  └─ Pass 3: 文本后缀 (causal)
      ┌──────────────────────────────────────┐
      │ suffix_tokens → 标准因果 prefill     │
      │ 可以看到 prefix + media + 当前 suffix│
      │ 采样第一个生成 token                 │
      └──────────────────────────────────────┘
```

### 8.3 非因果解码

**位置**: `src/mtmd/mod.zig:471-477`

```zig
pub fn decodeUseNonCausal(self: *const MtmdContext, chunk: ?*const InputChunk) bool {
    if (self.caps.vision_encoder_type.len > 0 and
        std.mem.startsWith(u8, self.caps.vision_encoder_type, "gemma")) {
        if (chunk) |c| return c.chunk_type == .image;
        return true;
    }
    return false;
}
```

Gemma 4 的视觉编码器使用非因果注意力（图像 token 之间可以互相看），而音频编码器使用因果注意力。

### 8.4 M-RoPE 位置编码

**位置**: `src/mtmd/helper.zig:37-82`

支持三种位置编码模式：

| 模式 | 说明 | 位置计算 |
|------|------|---------|
| `normal` | 标准位置编码 | t = pos_0 + i, x=0, y=0 |
| `mrope` | M-RoPE（多模态 RoPE） | t = pos_0, x = i % nx, y = i / nx |
| `hunyuanvl` | HunyuanVL 风格 | 带行结束标记的 2D 位置 |

---

## 9. 媒体标记系统

### 9.1 标记解析

**位置**: `src/chat_template/multimodal.zig`

对话模板渲染时，将媒体文件路径替换为媒体标记：

```
用户输入: --image photo.jpg --audio speech.wav -p "Describe this"
  │
  ├─ 对话模板渲染:
  │   "<|user|>\n<|image|><|audio|>Describe this<|end|>\n<|assistant|>\n"
  │
  ├─ 媒体标记替换:
  │   "<__media__>" 占位符 → 实际 Bitmap 数据
  │
  └─ Token 化:
      [text_chunk] [image_chunk] [audio_chunk] [text_chunk]
```

### 9.2 标记与编码器的映射

| 标记 | 编码器 | 输出维度 | 说明 |
|------|--------|---------|------|
| `<|image|>` | VisionEncoder | [1536, 196] | 224x224 图像 → 14x14=196 patches |
| `<|audio|>` | AudioEncoder | [1536, T'] | T 帧音频 → T' 个嵌入 |

---

## 10. 与 llama.cpp 的对齐策略

### 10.1 逐阶段对齐

每个模块标注参考的 llama.cpp 源文件和行号：

| zllama.zig 模块 | llama.cpp 参考 | 对齐状态 |
|----------------|---------------|---------|
| `audio/loader.zig` | `mtmd-audio.cpp` | ✅ |
| `audio/framing.zig` | `mtmd-audio.cpp` (preprocess + log_mel_spectrogram_worker_thread) | ✅ |
| `audio/mel.zig` | `mtmd-audio.cpp` (fill_mel_filterbank_matrix) | ✅ |
| `audio/log_transform.zig` | `mtmd-audio.cpp` (log_mel_spectrogram_worker_thread) | ✅ |
| `audio/encoder.zig` | `gemma4a.cpp` | ⚠️ Conv2d 输出数值差异 |
| `audio/pipeline.zig` | `mtmd-audio.cpp` (mtmd_audio_preprocessor_gemma4a) | ✅ |
| `vision/encoder.zig` | `gemma4v.cpp`, `gemma4uv.cpp` | ✅ |
| `vision/loader.zig` | `clip.cpp` | ✅ |
| `vision/postprocess.zig` | `gemma4v.cpp` | ✅ |
| `mtmd/mod.zig` | `mtmd.h`, `mtmd.cpp` | ✅ |
| `mtmd/tokenize.zig` | `mtmd.h` (mtmd_tokenize) | ✅ |
| `mtmd/helper.zig` | `mtmd-helper.h` | ✅ |

### 10.2 数值对齐验证

通过 `debug_audio/DATA_ALIGN.md` 记录的逐阶段对比：

| 阶段 | 状态 | 说明 |
|------|------|------|
| 原始音频样本 | ✅ 一致 | 文件大小相同（138,080 字节） |
| Mel 频谱 | ✅ 一致 | 文件大小相同（110,883 字节），10,240 个值 |
| Conv1d 权重 | ✅ 一致 | 逐元素对比通过 |
| Conv2d_0 输出 | ❌ 数值不同 | 形状一致（327,680 个值），数值范围不同 |
| Conv2d_1 输出 | ❌ 数值不同 | 形状一致（20,480 个值），数值范围不同 |
| 最终嵌入 | ❌ 待修复 | 因 Conv2d 不一致导致 |

### 10.3 已知差异

1. **FFT 实现**: zllama 使用 Apple Accelerate vDSP，llama.cpp 使用自定义 C++ FFT
   - vDSP 前向 FFT 输出比标准 DFT 大 2 倍，需除以 2 缩放因子
   - 已通过 `inv_scale: f32 = 0.5` 补偿
   - Mel 频谱数据已验证一致 ✅

2. **Conv2d 输出数值差异**: 根本原因仍在排查中
   - 已排除：权重数据布局、ggml 版本、编译选项、输入数据
   - 已修复：缺少 `ggml_set_input` 导致中间张量内存被释放
   - 待排查：编译优化差异、线程调度、Accelerate BLAS 版本差异

3. **重采样算法**: zllama 使用线性插值，llama.cpp 可能使用 libsamplerate

---

## 11. 调试与验证工具

### 11.1 调试数据输出

**位置**: `src/mtmd/helper.zig:280-356`

`mtmdDebugSaveData()` 函数可将任意浮点数组保存为 JSON 格式文件：

```zig
// 保存 Mel 频谱数据用于对比
helper.mtmdDebugSaveData(io, "debug_audio", "zllama_audio_mel.json",
    "audio_mel", mel_data);
```

输出目录: `debug_audio/`

### 11.2 对比工具

| 工具 | 位置 | 用途 |
|------|------|------|
| `zllama-compare-mtmd-audio` | `src/tools/compare_mtmd_audio.zig` | 逐阶段对比音频编码器输出 |
| `zllama-compare-mtmd-vision` | `src/tools/compare_mtmd_vision.zig` | 逐阶段对比视觉编码器输出 |
| `zllama-compare-logits` | `src/tools/compare_logits.zig` | 对比 logits 输出 |
| `zllama-compare-llamacpp` | `src/tools/compare_with_llamacpp.zig` | 通用对比工具 |

### 11.3 余弦相似度验证

**位置**: `src/mtmd/audio/postprocess.zig:52-68`

```zig
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    // 计算两个向量的余弦相似度
    // 用于验证与 llama.cpp 输出的对齐程度
}
```

### 11.4 单元测试

| 测试文件 | 测试内容 |
|---------|---------|
| `src/tests/test_audio.zig` | 音频编码器单元测试 |
| `src/tests/test_vision.zig` | 视觉编码器单元测试 |
| `src/tests/test_mtmd.zig` | MTMD 模块集成测试 |

---

## 12. 已知问题与待办

### 12.1 当前问题

- [ ] **Conv2d 输出数值差异**: `debug_audio/DATA_ALIGN.md` 记录的 Conv2d_0 输出数值不同，导致后续所有层输出不一致
  - 输入数据（Mel 频谱）和权重已验证一致 ✅
  - ggml 源码和编译选项已验证一致 ✅
  - 待排查：编译优化差异、线程调度、Accelerate BLAS 版本
- [ ] **`encoder_input` 调试数据不可靠**: 在图计算之前保存，读取未初始化内存
- [ ] **重采样算法差异**: 线性插值 vs libsamplerate 可能导致细微数值差异

### 12.2 待办事项

- [ ] 修复 Conv2d 输出不一致的根本原因
- [ ] 添加更多模型的视觉编码器支持（Qwen2-VL 等）
- [ ] 实现视频输入支持
- [ ] 添加 GPU 后端支持（Metal/CUDA）
- [ ] 完善 golden 测试体系（将 llama.cpp 各阶段输出保存为参考文件）
- [ ] 添加 CI 自动对比流水线

### 12.3 性能优化方向

- [ ] 音频预处理使用多线程并行（30 秒分块处理可并行化）
- [ ] 视觉编码器使用 `ggml_graph_plan` + 线程池
- [ ] 减少中间张量的内存分配次数
- [ ] 使用 `mmap` 加载 mmproj 权重文件

---

## 附录 A：关键文件索引

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| `src/mtmd/mod.zig` | 499 | MtmdContext, MultiModalManager, 基础类型 |
| `src/mtmd/helper.zig` | 357 | Chunk 评估, 文件加载, 调试输出 |
| `src/mtmd/tokenize.zig` | 95 | 多模态 token 化 |
| `src/mtmd/preprocess.zig` | 340 | 图像预处理 (resize + 归一化) |
| `src/mtmd/fft.zig` | 201 | Apple Accelerate vDSP FFT |
| `src/mtmd/audio/mod.zig` | 65 | 音频模块入口 |
| `src/mtmd/audio/config.zig` | 124 | 音频配置参数 |
| `src/mtmd/audio/types.zig` | 89 | 音频数据结构 |
| `src/mtmd/audio/loader.zig` | 162 | WAV 加载与重采样 |
| `src/mtmd/audio/framing.zig` | 139 | 分帧与加窗 |
| `src/mtmd/audio/mel.zig` | 123 | Mel 滤波器组 |
| `src/mtmd/audio/log_transform.zig` | 62 | 对数变换 |
| `src/mtmd/audio/encoder.zig` | 838 | Conformer 编码器 |
| `src/mtmd/audio/postprocess.zig` | 63 | 后处理 (softcapping) |
| `src/mtmd/audio/pipeline.zig` | 218 | 音频流水线编排器 |
| `src/mtmd/vision/mod.zig` | 43 | 视觉模块入口 |
| `src/mtmd/vision/config.zig` | 80 | 视觉配置参数 |
| `src/mtmd/vision/types.zig` | 55 | 视觉数据结构 |
| `src/mtmd/vision/loader.zig` | 126 | 视觉权重加载 |
| `src/mtmd/vision/preprocess.zig` | 96 | 图像归一化 |
| `src/mtmd/vision/encoder.zig` | 523 | ViT 编码器 |
| `src/mtmd/vision/postprocess.zig` | 67 | 后处理 (标准化 + 投影) |
| `src/mtmd/vision/pipeline.zig` | 75 | 视觉流水线编排器 |

## 附录 B：参考文档

| 文档 | 内容 |
|------|------|
| `docs/LLAMA.CPP_MTMD.md` | llama.cpp MTMD 实现详解 |
| `docs/MULTIMODAL.md` | 多模态推理模块概述 |
| `docs/audio_flow.md` | 音频处理流水线详细说明 |
| `docs/vision_flow.md` | 视觉处理流水线详细说明 |
| `debug_audio/DATA_ALIGN.md` | 音频推理数据对齐记录 |
| `deps/conv2d.md` | Conv2d 实现分析 |
| `deps/input.md` | 输入数据处理分析 |
| `deps/MTMD_AUDIO.md` | MTMD 音频模块分析 |
