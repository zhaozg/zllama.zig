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
| 音频预处理 | `mtmd_audio_preprocessor_gemma4a` 类 | `mel_spectrogram.zig` 编排器 + 回调 |
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
                    │ encoder.zig      │    │ mel_spectrogram.zig│
                    │   (ViT forward)  │    │ → framing.zig    │
                    │ postprocess.zig  │    │ → mel.zig        │
                    │   (projection)   │    │ → encoder.zig    │
                    │                  │    │   (Conformer)    │
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
│   └── mel_spectrogram.zig # Mel 频谱计算（编排器）
│
└── vision/              # 视觉处理子模块
    ├── mod.zig          # 模块入口，重新导出所有公开 API
    ├── config.zig       # 配置参数（ViT 超参数）
    ├── types.zig        # ViT 层权重和编码器权重类型
    ├── loader.zig       # 从 GGUF 加载视觉编码器权重
    ├── preprocess.zig   # 图像归一化（standard/siglip/passthrough）
    ├── encoder.zig      # ViT 编码器（Gemma4V / Gemma4UV）
    └── postprocess.zig  # 后处理（标准化 + 投影到 LLM 空间）
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
  │   └── mel_spectrogram.zig
  ├── vision/    (通过 @import("vision"))
  │   ├── config.zig
  │   ├── types.zig
  │   ├── loader.zig
  │   ├── preprocess.zig
  │   ├── encoder.zig
  │   └── postprocess.zig
  ├── helper.zig
  ├── tokenize.zig
  ├── preprocess.zig
  └── fft.zig
```

---

## 4. 核心数据结构

### 4.1 `MultiModalManager` — 多模态管理器

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
- `resolveImageMarkers()`: 从 `caps.special_tokens` 解析图像开始/结束标记
- `resolveAudioMarkers()`: 从 `caps.special_tokens` 解析音频开始/结束标记

### 4.2 `MtmdContext` — 多模态上下文

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

### 4.3 `InputChunk` — 输入 Chunk 序列

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

### 4.6 能力检测（自动）

扫描 GGUF tensor 名称识别：

| Tensor 特征 | 推断 |
|------------|------|
| `v.patch_embd.weight` / `mm.input_projection.weight` | 视觉编码器 |
| `patch_norm_1.weight` | `gemma4uv`（否则 `gemma4v`） |
| `a.conv1d.0.weight` / `a.input_projection.weight` | 音频编码器 (`gemma4a`) |

### 4.7 媒体类型系统

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
  ├─[3] mel_spectrogram.zig: processPcmSamples()
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

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_size` | 12 | 块大小 |
| `max_past_horizon` | 12 | 最大过去视野 |
| `context_size` | 24 | 上下文大小 = chunk_size + max_past_horizon |
| `rpe_positions` | 13 | RPE 位置数 = max_past_horizon + 1 |
| `softcap` | 50.0 | Softcap 值 |

### 5.5 GEMMA4A vs GEMMA4UA 对比

| 特性 | GEMMA4A | GEMMA4UA |
|------|---------|----------|
| 编码器 | Conformer (Conv2D + Self-Attention + Conv) | 无编码器 |
| 预处理 | FFT → Mel Spectrogram | 原始波形分帧 |
| 帧大小 | 20ms (320 samples) | 40ms (640 samples) |
| 帧步长 | 10ms (160 samples) | 40ms (640 samples) |
| 下采样 | 2x Conv2D stride-2 (4x 时间缩减) | 无 |
| 位置编码 | 正弦 RPE (chunked local) | 无 |
| 注意力 | Chunked local self-attention | 无 |
| 输出投影 | audio_out_proj + mm_input_proj | mm_input_proj |
| 复杂度 | 高 (多层 Conformer) | 极低 (单层投影) |

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

```zig
pub const EncoderType = enum {
    gemma4v,   // 标准 ViT + SigLIP
    gemma4uv,  // 统一视觉编码器（带额外 patch 归一化）
};
```

检测逻辑（`MultiModalManager.detectFromGGUF`）：
- 如果存在 `v.patch_norm.1.weight` 或 `patch_norm_1.weight` → `gemma4uv`
- 否则 → `gemma4v`

### 6.4 GEMMA4V vs GEMMA4UV 对比

| 特性 | GEMMA4V | GEMMA4UV |
|------|---------|----------|
| Patch 提取 | Conv2D (ggml_conv_2d) | im2col + LayerNorm + Linear |
| 像素缩放 | `2*(x-0.5)` scale_bias | 无 |
| 位置编码 | 查找表 + 加法 | 查找表 + 加法 + LayerNorm |
| 归一化 | RMSNorm | LayerNorm (PyTorch 默认) |
| ViT 层 | 有 (build_vit, 含 2D RoPE) | 无 |
| Pooling | 2D AvgPool (n_merge=3) | 无 (n_merge 已合并到 patch_size) |
| 标准化 | std_bias / std_scale | 无 |
| 投影 | ClippableLinear | 普通 Linear |
| 复杂度 | 高 (完整 ViT) | 低 (仅 patch + 投影) |

### 6.5 可变分辨率支持

视觉编码器支持可变分辨率输入，使用基于视觉 token 预算的动态扩展方式处理图像：

- `estimateOutputTokens()`: 估算给定分辨率图像的 token 数量
- `bestResolution()`: 计算视觉 token 预算下的最佳图像分辨率

### 6.6 图像预处理详解

**Gemma4V / GEMMA4UV 使用的预处理器**: `mtmd_image_preprocessor_dyn_size`

处理步骤:
1. 获取原始图像尺寸
2. 计算 `n_merge` (Gemma4V=3, Gemma4UV=1 但 patch_size 更大)
3. 计算目标尺寸: `calc_size_preserved_ratio()` — 保持宽高比, 对齐到 `patch_size * n_merge` 的倍数, 限制在 `[image_min_pixels, image_max_pixels]` 范围内
4. 调用 `img_tool::resize()` 进行 resize (双线性插值, 可选 padding)
5. 归一化为 float32:
   - `(pixel / 255.0 - mean) / std` (标准 CLIP 归一化)
   - 数据布局: `[H*W, 3]` (CHW 格式)

---

## 7. Token 化与 Chunk 管理

### 7.1 多模态 Token 化流程

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

根据编码器类型自动确定媒体标记：

| 编码器类型 | 图像标记 | 音频标记 |
|-----------|---------|---------|
| gemma4v / gemma4uv | `<|image>` ... `<image|>` | — |
| gemma4a / gemma4ua | — | `<|audio>` ... `<audio|>` |
| 其他 | `<start_of_image>` ... `<end_of_image>` | 空 |

### 7.4 Token 化对比：zllama.zig vs llama.cpp mtmd

| 方面 | zllama.zig | llama.cpp mtmd |
|------|-----------|---------------|
| 输入分割 | `<|image|>` 作为模板占位符 | `<__media__>` 作为分割标记 |
| Token 化方式 | 全量 tokenize + 后处理展开 | 分段 tokenize，按 chunk 类型 |
| 边界标记 | **无**（只有占位符 token） | `img_beg` + chunk + `img_end` |
| 媒体 chunk 表示 | 连续的 N 个相同 token ID | `InputChunk` 类型标记为 image/audio |

### 7.5 核心函数：`tokenizeWithMediaPlaceholders()`

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

**设计理由**：
1. `parse_special=true` 确保 `<|image|>` 被 tokenizer 正确编码为单个特殊 token
2. 展开为 N 个 token 是因为视觉编码器输出 N 个 embedding 向量，每个都需要在 LLM 的序列中占一个位置
3. 保留原始 token ID 允许 LLM 在 Pass 2 中仍然可以查找 per-layer embedding

---

## 8. 三阶段 Prefill 机制

### 8.1 设计动机

多模态推理中，媒体 token（图像/音频嵌入）与文本 token 的注意力模式不同：
- **文本 token**: 因果注意力（只能看过去）
- **媒体 token**: 非因果注意力（可以看全部媒体 token，但不能看未来文本）

三阶段 Prefill 将输入拆分为三段，分别处理。

### 8.2 三阶段流程

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

### 8.4 实现细节

每阶段执行：
1. `context.reset()` — 释放所有之前的 tensor/节点
2. `setNoAlloc(false)` — 允许 tensor 分配
3. 构建计算图（tensor、graph nodes）
4. `setNoAlloc(true)` — 锁定 context
5. Gallocr 分配 + `graph.compute()`

### 8.5 位置编码

| 位置类型 | 描述 | 适用模型 |
|---------|------|---------|
| `MTMD_POS_TYPE_NORMAL` | 顺序位置：pos = pos_0 + i | Gemma 4, LLaMA 等 |
| `MTMD_POS_TYPE_MROPE` | M-RoPE：t=pos_0, x=pos_0+col, y=pos_0+row | Qwen2-VL |
| `MTMD_POS_TYPE_HUNYUANVL` | HunyuanVL 布局：BOI/EOI + 行新行 | HunyuanVL |

Gemma 4 使用 `MTMD_POS_TYPE_NORMAL`，所有维度（t, x, y, z）都使用相同的顺序位置。

---

## 9. 媒体标记系统

### 9.1 标记解析

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

### 9.3 两种标记系统

zllama.zig 中使用**统一标记系统**：

| 标记 | 用途 | 示例 | 在词表中？ | 处理方式 |
|------|------|------|-----------|---------|
| 媒体特殊 token | 对话模板中的媒体占位符 | `<|image|>` `<|audio|>` | ✅ 是 | `parse_special=true` 编码为单个 token |
| 边界标记 | 隐式定义（模板 + 展开） | `<|image|>` 前 ← N个token → `<image|>` 后 | ✅ 是 | 三阶段 prefill 自动划分 prefix/media/suffix |

### 9.4 Gemma 4 的边界标记

根据 llama.cpp mtmd.cpp 的 `init_vision()` 和 `init_audio()`：

| 编码器类型 | img_beg | img_end | aud_beg | aud_end |
|-----------|---------|---------|---------|---------|
| GEMMA4V / GEMMA4UV | `<|image>` | `<image|>` | - | - |
| GEMMA4A / GEMMA4UA | - | - | `<|audio>` | `<audio|>` |
| GEMMA3 / GEMMA3NV | `<start_of_image>` | `<end_of_image>` | - | - |

### 9.5 zllama.zig 与 llama.cpp 的设计对等性

| 检查项 | llama.cpp | zllama.zig | 影响 |
|--------|-----------|------------|------|
| `add_media` 边界标记注入 | ✅ `img_beg` + chunk + `img_end` | ✅ 模板 + 展开（三等分 prefix/media/suffix） | ✅ 对等 |
| Embedding 注入 | `llama_batch.embd` | `embed_override` → `mediaForward()` | ✅ 对等 |
| Non-causal attention | `llama_set_causal_attn(false)` | Pass 2 `causal=false` | ✅ 对等 |
| Token 化方式 | `mtmd_tokenize` 分段处理 | `tokenizeWithMediaPlaceholders` 后处理展开 | 🟡 路径不同，功能对等 |
| 编码器数值精度 | f32 | f32 | ✅ 一致 |
| 三阶段 prefill | per-chunk causal toggle | 三阶段 context reset | ✅ 设计等价 |

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
| `audio/mel_spectrogram.zig` | `mtmd-audio.cpp` (mtmd_audio_preprocessor_gemma4a) | ✅ |
| `vision/encoder.zig` | `gemma4v.cpp`, `gemma4uv.cpp` | ✅ |
| `vision/loader.zig` | `clip.cpp` | ✅ |
| `vision/postprocess.zig` | `gemma4v.cpp` | ✅ |
| `mtmd/mod.zig` | `mtmd.h`, `mtmd.cpp` | ✅ |
| `mtmd/tokenize.zig` | `mtmd.h` (mtmd_tokenize) | ✅ |
| `mtmd/helper.zig` | `mtmd-helper.h` | ✅ |

### 10.2 已知差异

1. **FFT 实现**: zllama 使用 Apple Accelerate vDSP，llama.cpp 使用自定义 C++ FFT
   - vDSP 前向 FFT 输出比标准 DFT 大 2 倍，需除以 2 缩放因子
   - 已通过 `inv_scale: f32 = 0.5` 补偿
   - Mel 频谱数据已验证一致 ✅

2. **Conv2d 输出数值差异**: 根本原因仍在排查中
   - 已排除：权重数据布局、ggml 版本、编译选项、输入数据
   - 已修复：缺少 `ggml_set_input` 导致中间张量内存被释放
   - 待排查：编译优化差异、线程调度、Accelerate BLAS 版本差异

3. **重采样算法**: zllama 使用线性插值，llama.cpp 可能使用 libsamplerate

### 10.3 模型权重加载

Gemma4 的权重从 mmproj GGUF 文件加载。

**Vision 权重 (GEMMA4V)**:

| 张量名模式 | 变量 | 用途 |
|------------|------|------|
| `v.patch_embeddings_0.weight` | `patch_embeddings_0` | Conv2d patch 提取 |
| `v.position_embeddings.weight` | `position_embeddings` | 2D 位置编码查找表 |
| `v.blk.{i}.*` | `layers[i].*` | ViT 各层权重 (Q/K/V/O, FFN, norm) |
| `v.std_bias` | `std_bias` | 标准化偏置 |
| `v.std_scale` | `std_scale` | 标准化缩放 |
| `mm.input_projection.weight` | `mm_input_proj_w` | 多模态投影 |
| `*.input_max/min, *.output_max/min` | `clamp_info_map` | ClippableLinear 参数 |

**Audio 权重 (GEMMA4A)**:

| 张量名模式 | 变量 | 用途 |
|------------|------|------|
| `a.conv1d.{i}.weight` | `sscp_conv_w[i]` | Subsampling Conv2D |
| `a.conv1d.{i}.norm.weight` | `sscp_norm_w[i]` | Conv 后 LayerNorm |
| `a.input_projection.weight` | `sscp_inp_proj_w` | 输入投影 |
| `a.pre_encode.out.weight` | `audio_out_proj_w` | 输出投影 |
| `mm.a.soft_emb_norm.weight` | `mm_soft_emb_norm_w` | 软嵌入归一化 |
| `mm.a.input_projection.weight` | `mm_input_proj_w` | 多模态投影 |
| `{prefix}.blk.{i}.*` | `layers[i].*` | Conformer 各层权重 |
| `{prefix}.blk.{i}.attn_k_rel.weight` | `attn_k_rel_w` | 相对位置注意力 |
| `{prefix}.blk.{i}.per_dim_scale.weight` | `per_dim_scale_w` | Q 维度缩放 |
| `{prefix}.blk.{i}.conv_dw.weight` | `conv_dw_w` | Depthwise Conv1D |

---

## 11. 调试与验证工具

### 11.1 调试数据输出

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

```zig
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    // 计算两个向量的余弦相似度
    // 用于验证与 llama.cpp 输出的对齐程度
}
```

### 11.4 生成参考 logits

使用 `tools/mtmd_ref_logits.cpp` 从 llama.cpp 生成参考 logits：

```bash
# 视觉
mtmd-ref-logits -m model.gguf --mmproj mmproj.gguf --image hello.png -p ":" -o ref_vision.bin

# 音频
mtmd-ref-logits -m model.gguf --mmproj mmproj.gguf --audio hello.wav -p ":" -o ref_audio.bin
```

### 11.5 端到端验证

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

### 11.6 单元测试

| 测试文件 | 测试数 | 覆盖范围 |
|---------|--------|---------|
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
- [ ] 跨平台 FFT 后备（非 macOS）

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
| `src/mtmd/audio/mel_spectrogram.zig` | 218 | Mel 频谱计算（编排器） |
| `src/mtmd/vision/mod.zig` | 43 | 视觉模块入口 |
| `src/mtmd/vision/config.zig` | 80 | 视觉配置参数 |
| `src/mtmd/vision/types.zig` | 55 | 视觉数据结构 |
| `src/mtmd/vision/loader.zig` | 126 | 视觉权重加载 |
| `src/mtmd/vision/preprocess.zig` | 96 | 图像归一化 |
| `src/mtmd/vision/encoder.zig` | 523 | ViT 编码器 |
| `src/mtmd/vision/postprocess.zig` | 67 | 后处理 (标准化 + 投影) |

