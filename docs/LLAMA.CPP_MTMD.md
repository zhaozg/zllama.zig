# llama.cpp/Gemma4 多模态处理流程分析 (MTMD)

本文档分析 [mtmd](deps/llama.cpp/tools/mtmd/] 目录下 Gemma4 对 audio (WAV) 和 vision (PNG/JPEG) 的完整处理链路: 从文件读取、预处理、编码、推理到输出的全过程。

---

## 1. 架构总览

Gemma4 多模态支持涉及以下 4 种 projector 类型:

| 类型 | 说明 | 对应图构建器 |
|------|------|-------------|
| `GEMMA4V` | 标准 Vision 编码器 (ViT + 2D RoPE) | `clip_graph_gemma4v` |
| `GEMMA4UV` | Unified Vision Embedder (im2col + LayerNorm) | `clip_graph_gemma4uv` |
| `GEMMA4A` | Audio Conformer 编码器 (FFT + Conv2D + Conformer) | `clip_graph_gemma4a` |
| `GEMMA4UA` | Unified Audio Embedder (无编码器, 直接投影) | `clip_graph_gemma4ua` |

核心组件层次:

```
mtmd-cli.cpp  (入口, 用户交互)
  |
  +-- mtmd-helper.cpp  (文件解码: stb_image / miniaudio)
  +-- mtmd.cpp         (核心库: tokenize / encode / batch)
  +-- mtmd-image.cpp   (图像预处理器)
  +-- mtmd-audio.cpp   (音频预处理器)
  +-- clip.cpp         (CLIP 模型加载 + 图构建 + 推理执行)
  +-- models/gemma4*.cpp  (各模型的具体计算图)
```

---

## 2. 完整 Pipeline 流程

### 2.1 整体流程图

```
用户输入 (PNG/WAV 文件 + 文本 prompt)
  |
  v
[1] 文件解码 (mtmd-helper.cpp)
  |  PNG -> stb_image -> RGB bitmap (uint8)
  |  WAV -> miniaudio -> PCM F32 mono (16kHz)
  v
[2] mtmd_tokenize() (mtmd.cpp)
  |  分割文本与 media marker
  |  对每个 bitmap 调用预处理器
  |  生成 mtmd_input_chunks (text/image/audio 交替)
  v
[3] 预处理 (mtmd-image.cpp / mtmd-audio.cpp)
  |  图像: resize -> normalize -> clip_image_f32
  |  音频: FFT -> mel spectrogram -> clip_image_f32
  v
[4] mtmd_encode_chunk() / mtmd_batch_encode() (mtmd.cpp)
  |  调用 clip_image_batch_encode()
  |  -> clip_get_graph_builder() 选择对应 gemma4 图构建器
  |  -> builder->build() 构建 ggml 计算图
  |  -> ggml_backend_sched 执行推理
  |  输出: float embeddings 向量
  v
[5] mtmd_helper_decode_image_chunk() (mtmd-helper.cpp)
  |  将 embeddings 注入 llama_batch
  |  调用 llama_decode() 送入语言模型
  v
[6] 文本 token 也通过 llama_decode() 送入
  |
  v
[7] 采样生成 (common_sampler)
  |  循环: llama_decode() -> sample() -> 输出 token
  v
[8] 文本输出
```

---

## 3. Vision 处理流程 (PNG 图片)

### 3.1 文件解码

**入口**: `mtmd_helper_bitmap_init_from_file()` (mtmd-helper.cpp)

1. 读取文件到内存 buffer
2. 调用 `mtmd_helper_bitmap_init_from_buf()`:
   - 先检测是否为音频文件 (`audio_helpers::is_audio_file()`)
   - 不是音频 -> 调用 `stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3)` 解码为 RGB 数据
   - 创建 `mtmd_bitmap(nx, ny, data)`, 其中 `is_audio = false`
3. 计算 FNV-1a hash 作为 bitmap ID (用于 KV cache 追踪)

输出: `mtmd_bitmap` (RGB uint8 数据, nx*ny*3 字节)

### 3.2 Tokenize 阶段

**入口**: `mtmd_tokenize()` -> `mtmd_tokenizer::tokenize()` (mtmd.cpp)

1. 用户文本中包含 `<__media__>` 标记, `split_text()` 将文本按标记分割
2. 遇到 media marker 时, 取出对应的 bitmap, 调用 `add_media()`
3. `add_media()` 中对 vision 路径:
   - 添加图像起始 token: `<|image>` (Gemma4V/GEMMA4UV 的 `img_beg`)
   - 将 `mtmd_bitmap` 转换为 `clip_image_u8`
   - 调用 `image_preproc->preprocess(img_u8)` 进行预处理
   - 计算输出 token 数: `clip_n_output_tokens()`
   - 构建 `mtmd_image_tokens` (包含 `clip_image_f32_batch`)
   - 包装为 `mtmd_input_chunk` (type = `MTMD_INPUT_CHUNK_TYPE_IMAGE`)
   - 添加图像结束 token: `<image|>` (Gemma4V/GEMMA4UV 的 `img_end`)

### 3.3 图像预处理

**Gemma4V / GEMMA4UV 使用的预处理器**: `mtmd_image_preprocessor_dyn_size` (mtmd-image.cpp)

处理步骤:
1. 获取原始图像尺寸
2. 计算 `n_merge` (Gemma4V=3, Gemma4UV=1 但 patch_size 更大)
3. 计算目标尺寸: `calc_size_preserved_ratio()` -- 保持宽高比, 对齐到 `patch_size * n_merge` 的倍数, 限制在 `[image_min_pixels, image_max_pixels]` 范围内
4. 调用 `img_tool::resize()` 进行 resize (双线性插值, 可选 padding)
5. 调用 `output.append(hparams, resized_image, true)` -- 归一化为 float32:
   - `(pixel / 255.0 - mean) / std` (标准 CLIP 归一化)
   - 数据布局: `[H*W, 3]` (CHW 格式)

输出: `mtmd_image_preproc_out` 包含 `clip_image_f32` entries

### 3.4 图像编码 (推理)

**入口**: `mtmd_encode_chunk()` -> `clip_image_batch_encode()` (clip.cpp)

#### 3.4.1 图构建器选择

`clip_get_graph_builder()` 根据 `proj_type` 选择:
- `PROJECTOR_TYPE_GEMMA4V` -> `clip_graph_gemma4v`
- `PROJECTOR_TYPE_GEMMA4UV` -> `clip_graph_gemma4uv`

#### 3.4.2 GEMMA4V 计算图 (`gemma4v.cpp`)

```
build_inp_raw()  -- 输入: [W, H, C=3, B=1] float32
  |
  v  scale_bias: patches = 2 * (patches - 0.5)  -- Gemma4 特有的像素缩放
  |
  v  ggml_conv_2d(patch_embeddings_0, stride=patch_size)  -- 卷积提取 patch
  |  输出: [n_patches, n_embd, n_batch]
  |
  v  transpose -> [n_embd, n_patches]
  |
  v  2D 位置编码:
  |  - position_embeddings 查找表 (分 X/Y 两个方向)
  |  - pos_x, pos_y 为输入 (I32)
  |  - emb_x = get_rows(tbl_x, pos_x)
  |  - emb_y = get_rows(tbl_y, pos_y)
  |  - inp += emb_x + emb_y
  |
  v  build_vit()  -- 标准 Vision Transformer
  |  每层:
  |    - RMSNorm
  |    - Q/K/V 投影 -> reshape_3d(d_head, n_head, n_pos)
  |    - 2D RoPE (NEOX ordering, 前半 X 方向, 后半 Y 方向)
  |    - Self-Attention (Q@K^T * scale -> softmax -> @V)
  |    - 输出投影 (o_w)
  |    - 残差连接
  |    - FFN (SwiGLU: up -> silu -> down)
  |    - 残差连接
  |
  v  Gemma4VisionPooler:
  |  - 2D Average Pooling (kernel = n_merge, stride = n_merge)
  |  - 缩放: cur * sqrt(n_embd)
  |
  v  标准化: (cur - std_bias) * std_scale  -- Gemma4 特有
  |
  v  Gemma4MultimodalEmbedder:
  |  - RMSNorm
  |  - mm_input_proj_w 线性投影 (带 ClippableLinear clamp)
  |
  v  输出: [n_mmproj_embd, n_output_tokens] float32
```

**ClippableLinear** (`build_mm`): Gemma4V 特有的机制, 对部分线性层的输入和输出进行 clamp:
```
clamped_input = clamp(x, inp_min, inp_max)
output = clamp(W @ clamped_input, out_min, out_max)
```

#### 3.4.3 GEMMA4UV 计算图 (`gemma4uv.cpp`)

Unified Vision Embedder, 更简洁:

```
build_inp_raw()  -- 输入: [W, H, C, B]
  |
  v  im2col (patch_size x patch_size kernel)
  |  -> [patch_size^2 * C, n_patches_w, n_patches_h]
  |
  v  reshape_2d -> [patch_size^2 * C, n_patches]
  |
  v  LayerNorm (patch_norm_1)  -- 注意: 用标准 LayerNorm, 非 RMSNorm
  |
  v  patch_embeddings_0 线性投影 + patch_bias
  |  -> [n_embd, n_patches]
  |
  v  LayerNorm (patch_norm_2)
  |
  v  2D 位置编码 (同 GEMMA4V, 查找表方式)
  |
  v  LayerNorm (patch_norm_3)  -- pos_norm
  |
  v  Gemma4UnifiedMultimodalEmbedder:
  |  - RMSNorm
  |  - mm_input_proj_w 线性投影
  |
  v  输出: [n_mmproj_embd, n_output_tokens]
```

#### 3.4.4 推理执行

`clip_image_batch_encode()` (clip.cpp):
1. 重置 backend scheduler
2. 调用 `builder->build()` 构建 `ggml_cgraph`
3. `ggml_backend_sched_alloc_graph()` 分配资源
4. 设置输入张量:
   - `set_input_f32("inp_raw", inp_raw)` -- 像素数据, 布局为 `[W, H, 3, B]` (R/G/B 分通道)
   - `set_input_i32("pos_x", ...)` / `set_input_i32("pos_y", ...)` -- 位置索引
5. `ggml_backend_sched_graph_compute()` 执行推理
6. 从输出张量读取 embeddings 到 `out_batch_embd`

### 3.5 输出 Token 数计算

`clip_n_output_tokens()` (clip.cpp) 对 GEMMA4V/GEMMA4UV:
```
n_patches = (W/patch_size) * (H/patch_size)
n_patches /= (n_merge^2)  -- pooling 后的 token 数
```

---

## 4. Audio 处理流程 (WAV 文件)

### 4.1 文件解码

**入口**: `mtmd_helper_bitmap_init_from_file()` (mtmd-helper.cpp)

1. 读取文件到 buffer
2. `is_audio_file()` 检测为音频 (通过文件头 magic bytes)
3. 调用 `decode_audio_from_buf()`:
   - 使用 **miniaudio** 库 (`ma_decoder`)
   - 配置: `ma_format_f32`, 1 channel (mono), `target_sample_rate` (16000Hz)
   - `ma_decoder_init_memory()` 初始化解码器
   - `ma_decoder_get_length_in_pcm_frames()` 获取帧数
   - `ma_decoder_read_pcm_frames()` 解码为 PCM F32 mono 数据
4. 创建 `mtmd_bitmap(n_samples, pcmf32.data())`, 其中 `is_audio = true`

输出: `mtmd_bitmap` (F32 PCM mono 数据, 16kHz)

### 4.2 Tokenize 阶段

`add_media()` 中对 audio 路径 (mtmd.cpp):
1. 添加音频起始 token: `<|audio>` (Gemma4A/GEMMA4UA 的 `aud_beg`)
2. 从 bitmap 提取 PCM samples (float* 指针, n_samples = buf.size / sizeof(float))
3. 调用 `audio_preproc->preprocess(samples, n_samples, mel_spec_chunks)`
4. 对每个 mel_spec chunk:
   - 创建 `clip_image_f32` (尺寸: n_len x n_mel, `is_audio = true`)
   - 计算 token 数: `clip_n_output_tokens(ctx_a, &mel_f32)`
   - 构建 `mtmd_audio_tokens` -> `mtmd_input_chunk` (type = `MTMD_INPUT_CHUNK_TYPE_AUDIO`)
5. 添加音频结束 token: `<audio|>` (Gemma4A/GEMMA4UA 的 `aud_end`)

### 4.3 音频预处理

#### 4.3.1 GEMMA4A 预处理器 (`mtmd_audio_preprocessor_gemma4a`)

**初始化** (`initialize()`):
- 填充 sin/cos 表 (用于 FFT): `cache.fill_sin_cos_table(n_fft=512)`
- 构建 Hann 窗口 (320 点, 零填充到 512): `0.5 - 0.5 * cos(2*PI*i/window_len)`
- 构建 HTK mel 滤波器组: `fill_mel_filterbank_matrix(n_mel, n_fft=512, sr=16000, fmin=0, fmax=8000, use_htk=true)`

**超参数** (clip.cpp):
```
audio_sample_rate = 16000
audio_n_fft       = 512
audio_window_len  = 320   (20ms 帧)
audio_hop_len     = 160   (10ms 步长)
n_mel_bins        = (来自模型)
eps               = 1e-6
```

**预处理** (`preprocess()`):
1. 将音频按 30 秒分块 (模型上下文限制, 每块约 750 tokens)
2. 对每个 30 秒 chunk:
   - 半因果左填充: `pad_left = window_len / 2 = 160`
   - 计算帧数 (匹配 PyTorch unfold 逻辑)
   - 右填充到所需长度
   - 调用 `log_mel_spectrogram()`:
     - 对每帧应用 Hann 窗口
     - 执行 FFT (使用预计算的 sin/cos 表)
     - 计算幅度谱: `|real^2 + imag^2|^0.5`
     - 应用 mel 滤波器组: `mel = filters @ spectrum`
     - 取对数: `log(mel + mel_floor)`
   - 裁剪到 PyTorch 帧数
3. 输出: `vector<mtmd_audio_mel>` (每个 30 秒 chunk 一个 mel spectrogram)

输出结构:
```
mtmd_audio_mel {
    n_len     // 时间帧数
    n_len_org // 原始帧数 (未裁剪)
    n_mel     // mel 频带数
    data      // [n_mel * n_len] float 数据
}
```

#### 4.3.2 GEMMA4UA 预处理器 (`mtmd_audio_preprocessor_gemma4ua`)

**无编码器** 模式, 极其简单:

**初始化**: no-op (不需要 FFT 或滤波器组)

**预处理**:
1. `frame_size = n_mel_bins = 640` (640 samples per token @ 16kHz = 40ms)
2. `n_tokens = ceil(n_samples / frame_size)`
3. 将原始 PCM 波形直接重排为 `[n_tokens, frame_size]` 的矩阵
4. 数据布局为 mel-major: `data[f * n_tokens + t]` (便于 ggml 张量加载为 `[n_tokens, frame_size]`)

超参数:
```
audio_sample_rate = 16000
n_mel_bins        = 640
eps               = 1e-6
```

### 4.4 音频编码 (推理)

#### 4.4.1 GEMMA4A 计算图 (`gemma4a.cpp`)

Conformer 架构, 5 个主要阶段:

```
[1] 输入
    inp_raw: [n_mel, n_len, 1, 1]  (mel spectrogram)
    -> transpose -> [n_len, n_mel, 1, 1]

[2] Subsampling Conv2D (2 层, stride=2)
    每层:
      - Conv2D(kernel=3, stride=2, padding=1)  -- 下采样 2x
      - LayerNorm (channels)
      - ReLU
    2 层后: 时间维度缩减 4x
    -> flatten [ch*freq, time]
    -> input_projection 线性投影 -> [n_embd, n_pos]

[3] Conformer Blocks (n_layer 层)
    每层包含 4 个子模块:

    (a) FFN 1 (half-step):
        RMSNorm -> Linear(up) -> SiLU -> Linear(down) -> RMSNorm
        residual += 0.5 * ffn(x)

    (b) Chunked Local Self-Attention with RPE:
        - RMSNorm
        - Q/K/V 投影 -> reshape [d_head, n_head, n_pos]
        - Q scaling: 1/(sqrt(d_head) * log(2))
        - K scaling: log(1+e)/log(2)
        - 分块: chunk_size=12, context=24, RPE positions=13
        - Q 分块: [D, C, B, H]
        - K/V 重叠窗口提取 (stride=C, window=S)
        - Content attention: Q @ K^T -> [S, C, B, H]
        - Relative Position attention: Q @ RPE^T -> blocked relative shift
        - Softcap: tanh(scores/50) * 50
        - Attention mask
        - Softmax -> @V -> output projection
        - residual += attn(x)

    (c) Convolution Module:
        RMSNorm -> Pointwise Conv1 -> GLU -> Depthwise Conv1D (causal)
        -> RMSNorm -> SiLU -> Pointwise Conv2
        residual += conv(x)

    (d) FFN 2 (half-step):
        RMSNorm -> Linear(up) -> SiLU -> Linear(down) -> RMSNorm
        residual += 0.5 * ffn(x)

    - Layer output RMSNorm

[4] Output Projection
    audio_out_proj_w 线性投影

[5] Audio Multimodal Embedder
    RMSNorm -> mm_soft_emb_norm_w -> mm_input_proj_w 线性投影
    (带 ClippableLinear clamp)

    输出: [n_mmproj_embd, n_output_tokens]
```

**Token 数计算** (GEMMA4A):
```
两次 Conv2D stride-2 下采样:
  n = img->nx()  (时间帧数)
  n = (n-1)/2 + 1  (第一次)
  n = (n-1)/2 + 1  (第二次)
  n_patches = n  (下采样 4x 后的时间步数)
```

#### 4.4.2 GEMMA4UA 计算图 (`gemma4ua.cpp`)

无编码器, 极其简洁:

```
[1] 输入
    inp_raw: [n_tokens, frame_size, 1, 1]  (原始波形分帧)
    -> permute(1,0,2,3) -> [frame_size, n_tokens, 1, 1]

[2] Gemma4UnifiedMultimodalEmbedder
    - RMSNorm (eps=1e-6)
    - mm_input_proj_w 线性投影

    输出: [n_mmproj_embd, n_tokens]
```

**Token 数计算** (GEMMA4UA):
```
n_patches = img->nx()  // 一个 token 对应一个 640-sample 帧
```

#### 4.4.3 推理执行

与 vision 路径相同的 `clip_image_batch_encode()`:
1. 构建计算图: `clip_graph_gemma4a::build()` 或 `clip_graph_gemma4ua::build()`
2. 设置输入: `set_input_f32("inp_raw", mel_data)` (音频 mel 数据直接作为 inp_raw)
3. 对于 GEMMA4A, 还需设置:
   - `pos_emb` (RPE 位置编码)
   - `kq_mask` (分块注意力掩码)
4. `ggml_backend_sched_graph_compute()` 执行推理
5. 读取输出 embeddings

---

## 5. Embeddings 注入语言模型

### 5.1 文本与媒体交替解码

`eval_message()` (mtmd-cli.cpp) 遍历所有 chunks:

1. **Text chunk**: 调用 `mtmd_helper_eval_chunk_single()`:
   - 获取 text tokens
   - 构建 `llama_batch` (token, pos, seq_id)
   - 调用 `llama_decode(lctx, batch)` 推理

2. **Media chunk** (image/audio):
   - 尝试从现有 batch 获取 embeddings, 或创建新 batch
   - 调用 `mtmd_batch_encode()` 批量编码 (多个 media chunk 可合并)
   - 获取 `embd = mtmd_batch_get_output_embd()`
   - 调用 `mtmd_helper_decode_image_chunk()`:
     - 构建 `decode_embd_batch` (embedding 作为 input)
     - 设置位置 (normal 或 M-RoPE 2D)
     - 分批调用 `llama_decode(lctx, batch_embd_view)` 推理

### 5.2 位置编码处理

- **普通模型**: `set_position_normal(n_past, seq_id)` -- 线性递增位置
- **M-RoPE 模型** (如 Qwen-VL): 图像使用 2D 位置 `(x, y)`, 音频使用 1D 位置
- Gemma4 使用普通位置编码

### 5.3 生成阶段

所有 chunks 解码完成后, 进入自回归生成循环:
1. 从最后一个 token 的 logits 采样: `common_sampler_sample()`
2. 新 token 加入 batch, `llama_decode()` 继续推理
3. 遇到 EOS 或 antiprompt 时停止

---

## 6. 关键数据结构

### 6.1 数据流转换链

```
PNG 文件
  -> unsigned char[] (文件 buffer)
  -> stbi_load -> uint8 RGB [H*W*3]
  -> mtmd_bitmap {nx, ny, is_audio=false}
  -> clip_image_u8 {nx, ny, buf=RGB}
  -> mtmd_image_preprocessor::preprocess()
  -> clip_image_f32 {nx, ny, buf=float32 normalized}  (CHW 布局)
  -> clip_image_f32_batch {entries, is_audio=false}
  -> mtmd_image_tokens {nx, ny, batch_f32}
  -> mtmd_input_chunk {type=IMAGE, tokens_image}
  -> clip_image_batch_encode() -> float[] embeddings
  -> llama_batch (embd as input)
  -> llama_decode() -> logits
  -> sample -> output token

WAV 文件
  -> unsigned char[] (文件 buffer)
  -> ma_decoder -> float[] PCM mono [n_samples]
  -> mtmd_bitmap {nx=n_samples, ny=1, is_audio=true}
  -> mtmd_audio_preprocessor::preprocess()
  -> mtmd_audio_mel {n_len, n_mel, data}
  -> clip_image_f32 {nx=n_len, ny=n_mel, is_audio=true}
  -> clip_image_f32_batch {entries, is_audio=true}
  -> mtmd_audio_tokens {n_tokens, batch_f32}
  -> mtmd_input_chunk {type=AUDIO, tokens_audio}
  -> clip_image_batch_encode() -> float[] embeddings
  -> llama_batch (embd as input)
  -> llama_decode() -> logits
  -> sample -> output token
```

### 6.2 核心结构体

| 结构体 | 文件 | 用途 |
|--------|------|------|
| `mtmd_bitmap` | mtmd.cpp | 原始输入数据 (RGB 或 PCM) |
| `clip_image_u8` | clip-model.h | uint8 图像 |
| `clip_image_f32` | clip-model.h | float32 归一化图像/mel |
| `clip_image_f32_batch` | clip-model.h | 批量 float32 数据 |
| `mtmd_image_tokens` | mtmd.cpp | 图像 token 信息 |
| `mtmd_audio_tokens` | mtmd.cpp | 音频 token 信息 |
| `mtmd_input_chunk` | mtmd.cpp | 单个输入块 (text/image/audio) |
| `mtmd_input_chunks` | mtmd.cpp | 输入块序列 |
| `mtmd_batch` | mtmd.cpp | 批量编码上下文 |
| `mtmd_audio_mel` | mtmd-audio.h | mel spectrogram 数据 |
| `mtmd_audio_cache` | mtmd-audio.h | FFT/滤波器组缓存 |

---

## 7. 模型权重加载

Gemma4 的权重从 mmproj GGUF 文件加载 (clip.cpp):

### 7.1 Vision 权重 (GEMMA4V)

| 张量名模式 | 变量 | 用途 |
|------------|------|------|
| `v.patch_embeddings_0.weight` | `patch_embeddings_0` | Conv2d patch 提取 |
| `v.position_embeddings.weight` | `position_embeddings` | 2D 位置编码查找表 |
| `v.blk.{i}.*` | `layers[i].*` | ViT 各层权重 (Q/K/V/O, FFN, norm) |
| `v.std_bias` | `std_bias` | 标准化偏置 |
| `v.std_scale` | `std_scale` | 标准化缩放 |
| `mm.input_projection.weight` | `mm_input_proj_w` | 多模态投影 |
| `*.input_max/min, *.output_max/min` | `clamp_info_map` | ClippableLinear 参数 |

### 7.2 Audio 权重 (GEMMA4A)

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

### 7.3 Unified Audio 权重 (GEMMA4UA)

| 张量名模式 | 变量 | 用途 |
|------------|------|------|
| `mm.a.input_projection.weight` | `mm_input_proj_w` | 唯一的投影矩阵 |

---

## 8. GEMMA4A vs GEMMA4UA 对比

| 特性 | GEMMA4A | GEMMA4UA |
|------|---------|----------|
| 编码器 | Conformer (Conv2D + Self-Attention + Conv) | 无编码器 |
| 预处理 | FFT -> Mel Spectrogram | 原始波形分帧 |
| 帧大小 | 20ms (320 samples) | 40ms (640 samples) |
| 帧步长 | 10ms (160 samples) | 40ms (640 samples) |
| 下采样 | 2x Conv2D stride-2 (4x 时间缩减) | 无 |
| 位置编码 | 正弦 RPE (chunked local) | 无 |
| 注意力 | Chunked local self-attention | 无 |
| 输出投影 | audio_out_proj + mm_input_proj | mm_input_proj |
| 复杂度 | 高 (多层 Conformer) | 极低 (单层投影) |

---

## 9. GEMMA4V vs GEMMA4UV 对比

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

---

## 10. 总结

Gemma4 多模态处理的核心设计理念:

1. **双文件架构**: 语言模型 GGUF + 多模态投影器 mmproj GGUF 分离, 允许独立开发
2. **统一接口**: `libmtmd` 通过 `mtmd_tokenize()` / `mtmd_encode_chunk()` 统一处理所有模态
3. **模块化预处理器**: 不同模型通过继承 `mtmd_image_preprocessor` / `mtmd_audio_preprocessor` 实现特定预处理
4. **计算图构建**: 每个模型变体有独立的 `clip_graph_*::build()` 方法, 使用 ggml 构建计算图
5. **Embedding 注入**: 多模态 embeddings 直接作为 `llama_batch` 的 input embedding, 与文本 token 交替送入语言模型
6. **Gemma4 特有机制**: ClippableLinear (输入/输出 clamp), 像素 scale_bias, std_bias/std_scale 标准化
