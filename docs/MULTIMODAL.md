# 多模态推理模块 — MULTIMODAL

> 参考：[llama.cpp tools/mtmd](deps/llama.cpp/tools/mtmd/)、Gemma 4 多模态实现
> 当前状态：视觉/音频编码器实现完整，parse_special 特殊 token 解析已集成，端到端文本推理已验证。
> **关键问题**：对话模板中媒体占位符的嵌入方式与 llama.cpp mtmd 的 `img_beg`/`img_end`/`aud_beg`/`aud_end` 边界标记机制不匹配，导致媒体嵌入在 token 序列中的位置偏移。

---

## 目录

1. [概述与架构](#概述与架构)
2. [公共基础设施](#公共基础设施)
3. [视觉编码器](#视觉编码器)
4. [音频编码器](#音频编码器)
5. [对话模板与媒体嵌入](#对话模板与媒体嵌入)
6. [端到端推理流程](#端到端推理流程)
7. [参考资源](#参考资源)
8. [质量对比验证方法](#质量对比验证方法)

---

## 概述与架构

zllama.zig 通过 `src/mtmd/` 目录下的模块实现多模态（图像+音频）推理，当前主要支持 **Gemma 4 E2B** 模型的内建编码器。

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

### 核心数据流

```
用户输入 (文本 + 媒体文件)
  │
  ▼
[1] 对话模板渲染 (chat_template/gemma4.zig)
    - 生成含 <|image|>/<|audio|> 占位符的字符串
    - 占位符 INLINE 嵌入（无换行）
  │
  ▼
[2] mtmd tokenize (mtmd/tokenize.zig)
    - 按 <__media__> 标记分割输入
    - 文本段 → tokenize
    - 媒体段 → 创建 InputChunk (image/audio)
    - 添加 img_beg/img_end/aud_beg/aud_end 边界标记 token
  │
  ▼
[3] 三阶段 Prefill (core/prefill.zig)
    - Stage 1: 文本前缀 (causal)
    - Stage 2: 媒体 token (non-causal/bidirectional)
    - Stage 3: 文本后缀 (causal)
  │
  ▼
[4] 自回归解码
```

---

## 公共基础设施

### Chunk 模型 — 输入的多态表示

`mtmd/tokenize.zig` 按 `media_marker`（默认 `<__media__>`）将输入分割为有序的 `InputChunk` 列表：

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
    img_beg: []const u8,        // "<|image>" (Gemma4)
    img_end: []const u8,        // "<image|>" (Gemma4)
    aud_beg: []const u8,        // "<|audio>" (Gemma4)
    aud_end: []const u8,        // "<audio|>" (Gemma4)
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
| parse_special 特殊 token 解析 | ✅ 已集成到 tokenizer |
| 端到端文本推理 (Gemma 4 text) | ✅ 已验证 |
| 输出质量对比 (vs llama.cpp) | ✅ 比较工具已使用三阶段 prefill，正确处理 causal/non-causal 注意力掩码 |
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
| parse_special 特殊 token 解析 | ✅ 已集成到 tokenizer |
| 端到端输出质量验证 | ✅ 比较工具已使用三阶段 prefill，正确处理 causal/non-causal 注意力掩码 |
| 跨平台 FFT 后备 (非 macOS) | 🟡 待完成（当前仅支持 macOS Accelerate） |

---

## 对话模板与媒体嵌入

### ⚠️ 核心问题：两种媒体嵌入机制的冲突

zllama.zig 当前存在**两种并行的媒体嵌入机制**，它们的设计理念不同，导致媒体 token 在序列中的位置偏移：

#### 机制 A：对话模板占位符（chat_template/multimodal.zig）

```
对话模板渲染 → "<|turn>user\n<|image|>Describe this image<turn|>\n<|turn>model\n"
                    ↑ 占位符
```

- 占位符 `<|image|>` 在模板渲染阶段嵌入
- 通过 `tokenizeWithPlaceholders()` 展开为 N 个填充 token
- 填充 token 的 token_id 是硬编码的（如 258880）
- **问题**：填充 token 的 token_id 与真实编码器输出的嵌入不匹配

#### 机制 B：mtmd tokenize（mtmd/tokenize.zig）

```
用户输入 → "Describe this image<__media__>"
                    ↑ media_marker
→ [text:"Describe this image"] [image:bm_0]
  → 添加 img_beg token ("<|image>") + 图像 chunk + img_end token ("<image|>")
```

- 使用 `<__media__>` 标记分割输入
- 媒体段创建 `InputChunk`（image/audio 类型）
- 自动添加 `img_beg`/`img_end` 边界标记 token
- **问题**：边界标记 token 是真实 token（通过 tokenizer 编码），不是填充 token

#### 冲突分析

| 方面 | 机制 A (占位符) | 机制 B (mtmd tokenize) |
|------|----------------|----------------------|
| 标记方式 | `<\|image\|>` 占位符 | `<__media__>` 分割标记 |
| 边界标记 | 无 | `img_beg`/`img_end` token |
| 媒体 token | 填充 token (固定 ID) | InputChunk (编码器输出) |
| 适用场景 | 纯文本模板渲染 | 多模态推理管线 |

**当前问题**：当同时使用两种机制时：
1. 对话模板渲染出含 `<|image|>` 的字符串
2. `tokenizeWithPlaceholders()` 将其展开为填充 token
3. 但 mtmd tokenize 期望的是 `<__media__>` 标记
4. 结果：媒体 token 在序列中的位置偏移，导致三阶段 prefill 的边界计算错误

### 解决方案：统一为 mtmd tokenize 机制

正确的流程应该是：

```
[1] 对话模板渲染 → 含 <__media__> 标记的字符串
    (不是 <|image|> 占位符)

[2] mtmd tokenize → 按 <__media__> 分割
    → [text_chunk, image_chunk, text_chunk, ...]
    → 自动添加 img_beg/img_end 边界标记

[3] 三阶段 prefill
    - Stage 1: 文本前缀 (直到第一个 image/audio chunk)
    - Stage 2: 媒体 token (non-causal)
    - Stage 3: 文本后缀
```

#### Gemma 4 的边界标记

根据 llama.cpp mtmd.cpp 的 `init_vision()` 和 `init_audio()`：

| 编码器类型 | img_beg | img_end | aud_beg | aud_end |
|-----------|---------|---------|---------|---------|
| GEMMA4V / GEMMA4UV | `<\|image>` | `<image\|>` | - | - |
| GEMMA4A / GEMMA4UA | - | - | `<\|audio>` | `<audio\|>` |
| GEMMA3 / GEMMA3NV | `<start_of_image>` | `<end_of_image>` | - | - |

这些边界标记是**真实 token**（通过 tokenizer 的 `parse_special=true` 编码），不是填充 token。

### 对话模板中的媒体标记

Gemma 4 的官方 Jinja 模板使用 `{{- '<|audio|>' -}}` 和 `{{- '<|image|>' -}}` 作为媒体标记。
这些标记在模板渲染后出现在字符串中，然后通过 `parse_special` 被 tokenizer 编码为特殊 token。

**关键区别**：
- `<|image|>` 和 `<|audio|>` 是**特殊 token**（在 tokenizer 词表中），编码为单个 token ID
- `<|image>` 和 `<image|>` 是**边界标记**（也是特殊 token），标记媒体嵌入的开始和结束
- `<__media__>` 是**分割标记**（不在词表中），用于在 mtmd tokenize 阶段分割输入

### 推荐的统一流程

```
用户输入: --image img.png --prompt "Describe this image"
  │
  ▼
[1] 构建 ChatMessage
    msg.role = "user"
    msg.content = "Describe this image"
    msg.media = .{ .type = .image, ... }
  │
  ▼
[2] 对话模板渲染 (gemma4.zig)
    → "<bos><|turn|>user\n<|image|>Describe this image<turn|>\n<|turn|>model\n"
       ↑ 注意：这里使用 <|image|> 特殊 token，不是 <__media__>
  │
  ▼
[3] 分段 tokenize
    - 扫描 <|image|> 占位符位置
    - 文本段: "<bos><|turn|>user\n" → tokenize
    - 占位符: <|image|> → 替换为 img_beg + image_chunk + img_end
    - 文本段: "Describe this image<turn|>\n<|turn|>model\n" → tokenize
  │
  ▼
[4] 三阶段 prefill
    - Stage 1: "<bos><|turn|>user\n" (causal)
    - Stage 2: "<|image>" + image_embeddings + "<image|>" (non-causal)
    - Stage 3: "Describe this image<turn|>\n<|turn|>model\n" (causal)
```

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

根据 llama.cpp `mtmd_image_tokens_get_decoder_pos()`：

| 位置类型 | 描述 | 适用模型 |
|---------|------|---------|
| `MTMD_POS_TYPE_NORMAL` | 顺序位置：pos = pos_0 + i | Gemma 4, LLaMA 等 |
| `MTMD_POS_TYPE_MROPE` | M-RoPE：t=pos_0, x=pos_0+col, y=pos_0+row | Qwen2-VL |
| `MTMD_POS_TYPE_HUNYUANVL` | HunyuanVL 布局：BOI/EOI + 行新行 | HunyuanVL |

Gemma 4 使用 `MTMD_POS_TYPE_NORMAL`，所有维度（t, x, y, z）都使用相同的顺序位置。

---

## 参考资源

- [llama.cpp mtmd 实现](deps/llama.cpp/tools/mtmd/)
- [Gemma 4 技术报告](https://ai.google.dev/gemma)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [本项目的待办事项](TODO.md)

---

## 质量对比验证方法

### 注意力掩码验证 ✅ 已确认

三阶段 prefill（`src/core/prefill.zig`）正确处理不同阶段的注意力掩码：

| 阶段 | 注意力类型 | causal | 实现位置 |
|------|-----------|--------|---------|
| Pass 1: 文本前缀 | Causal（只能看到自身及左侧） | `true` | `Gemma4Graph.build()` → `transformerForward(..., true)` |
| Pass 2: 媒体 tokens | Bidirectional（所有 token 相互可见） | `false` | `mediaForwardFn` → `mediaForward(..., false)` |
| Pass 3: 文本后缀 | Causal + 历史可见（可看到 prefix+media+左侧） | `true` | `Gemma4Graph.build()` → `transformerForward(..., true)` |

验证要点：
- KV cache 在三次 pass 之间持久化（通过独立的 `kv_cache_ctx`），每阶段追加写入
- 比较工具已更新为使用 `threeStagePrefill`，不再使用单次 `forwardWithEmbdOverride`（后者对所有 token 使用相同的 causal 标志）

### 生成参考 logits

使用 zllama-gen-ref 工具从文本模型生成参考 logits：

```bash
zig-out/bin/zllama-gen-ref --model model.gguf -p "prompt" -n 1 -o ref.bin
```

### 视觉对比

```bash
# 第 1 步：用 llama.cpp mtmd 生成参考
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image img.png \
  --jinja -p ":" --logit-binary ref_vision.bin

# 第 2 步：对比 zllama.zig 输出
zig-out/bin/zllama-compare-mtmd-vision \
  --model model.gguf --mmproj mmproj.gguf --image img.png \
  --prompt "Describe this image" --ref-logits ref_vision.bin
```

### 音频对比

```bash
# 第 1 步：用 llama.cpp mtmd 生成参考
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --audio hello.wav \
  --jinja -p ":" --logit-binary ref_audio.bin

# 第 2 步：对比 zllama.zig 输出
zig-out/bin/zllama-compare-mtmd-audio \
  --model model.gguf --mmproj mmproj.gguf --audio hello.wav \
  --prompt "Transcribe the audio" --ref-logits ref_audio.bin
```

> 注意：`--logit-binary` 需要 llama.cpp 对应版本支持。如果该 flag 不可用，
> 可使用 `zllama-gen-ref` 在 zllama.zig 内部生成纯文本参考，或通过修改
> llama.cpp 源码添加 logit 导出功能。

---

## 附录：llama.cpp mtmd 关键设计决策

### 1. 媒体标记 vs 占位符

llama.cpp mtmd 使用**两种不同的标记**：

| 标记 | 用途 | 示例 | 处理方式 |
|------|------|------|---------|
| `media_marker` | 输入分割标记 | `<__media__>` | 在 `mtmd_tokenizer` 中分割输入 |
| `img_beg`/`img_end` | 图像边界标记 | `<\|image>` / `<image\|>` | 作为特殊 token 编码，标记嵌入边界 |
| `aud_beg`/`aud_end` | 音频边界标记 | `<\|audio>` / `<audio\|>` | 同上 |

**关键设计**：
- `media_marker` 是**分割标记**，不在 tokenizer 词表中，仅用于分割输入
- `img_beg`/`img_end` 是**特殊 token**，在 tokenizer 词表中，通过 `parse_special=true` 编码
- 边界标记确保媒体嵌入在 token 序列中有明确的开始和结束位置

### 2. 编码器类型检测

llama.cpp 通过 `clip_get_projector_type()` 检测编码器类型，然后根据类型设置不同的边界标记和预处理器：

```cpp
case PROJECTOR_TYPE_GEMMA4V:
case PROJECTOR_TYPE_GEMMA4UV:
    img_beg = "<|image>";
    img_end = "<image|>";
    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
    break;
```

### 3. 音频预处理器类型

| 类型 | 边界标记 | 预处理器 |
|------|---------|---------|
| GEMMA4A | `<\|audio>` / `<audio\|>` | `mtmd_audio_preprocessor_gemma4a` |
| GEMMA4UA | `<\|audio>` / `<audio\|>` | `mtmd_audio_preprocessor_gemma4ua` |
| QWEN2A | `<\|audio_bos\|>` / `<\|audio_eos\|>` | `mtmd_audio_preprocessor_whisper` |

### 4. 非因果注意力判定

Gemma 4 的视觉和音频编码器输出都需要 non-causal 注意力。这是通过 `mtmd_decode_use_non_causal()` 函数判定的，在构建计算图时根据 chunk 类型设置 causal 标志。

### 5. 位置编码

Gemma 4 使用 `MTMD_POS_TYPE_NORMAL`，所有维度（t, x, y, z）都使用相同的顺序位置。这与 Qwen2-VL 的 M-RoPE 不同，后者需要为每个 token 计算 (t, x, y) 三维位置。
