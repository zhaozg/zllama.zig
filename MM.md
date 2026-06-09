# 多模态支持

本仓库(zllama.zig) 的多模态支持开发文档。

## 目标模型

Gemma 4 E2B (gemma-4-E2B-it):
- `--model ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf`
- `--mmproj ~/.cache/models/mmproj-F16.gguf`
- `--audio ~/.cache/models/hello.wav`
- `--image ~/.cache/models/ocr.png` (PPM P6 格式，暂不支持 JPEG/PNG)

llama.cpp 参考: `deps/llama.cpp/`（symlink 到上游仓库）

### 测试资源

| 资源 | 路径 | 用途 |
|------|------|------|
| 图像加载库 | `vendor/stb/stb_image.h` | ⏳ 待 vendored，用于 JPEG/PNG 解码 |
| 测试图像 | `~/.cache/models/ocr.png` | OCR 图文问答测试（需要 stb_image 支持）|

---

## 一、架构对齐

llama.cpp `tools/mtmd/` (libmtmd) 与 zllama.zig `src/mm/` 的对应关系：

| llama.cpp libmtmd | zllama.zig mm/ | 状态 |
|---|---|---|
| `mtmd.cpp` / `clip-impl.h` | `mm/manager.zig` | ✅ 已实现 |
| `models/gemma4a.cpp` (Conformer 音频编码器) | `mm/audio.zig` | ✅ 已实现 |
| `models/gemma4v.cpp` (ViT 视觉编码器) | `mm/vision.zig` (Gemma4V) | ✅ 已实现 |
| `models/gemma4uv.cpp` (UV 视觉编码器) | `mm/vision.zig` (Gemma4UV) | ✅ 已实现 |
| `mtmd-audio.cpp` (Mel 频谱预处理) | `mm/preprocess.zig` | ✅ 已实现 |
| `mtmd-audio.cpp` (FFT) | `mm/fft.zig` (Accelerate vDSP) | ✅ 已实现 |

---

## 二、当前实现状态

### 2.1 ✅ 已完成

| 模块 | 文件 | 功能 |
|------|------|------|
| 多模态管理器 | `src/mm/manager.zig` | 加载 mmproj、检测能力、编码调度 |
| 音频编码器 | `src/mm/audio.zig` | Conformer 编码器（子采样 Conv2D + 12 层 Conformer） |
| 视觉编码器 | `src/mm/vision.zig` | ViT + SigLIP (Gemma4V) / im2col + LN (Gemma4UV) |
| 音频预处理 | `src/mm/preprocess.zig` | WAV 加载、Mel 频谱、STFT、滤波器组 |
| FFT 引擎 | `src/mm/fft.zig` | Apple Accelerate vDSP FFT + Hann 窗 |
| CLI 参数 | `src/main.zig` | `--mmproj`、`--audio`、`--image` |
| 构建系统 | `build.zig` | `mm_manager_mod`、`mm_audio_mod`、`mm_vision_mod`、`fft_mod`、`mm_preprocess_mod` |
| 能力检测 | `src/models/registry.zig` | `detectCapabilities()` 检测 has_audio/has_vision |

### 2.2 ✅ 音频到LLM集成 (Phase 1 完成)

**实现方式**：
- `Gemma4Model.forwardWithEmbdOverride()` — 允许用预计算的嵌入替代 token 嵌入
- `transformerForward()` — 提取共享的 transformer 循环，供 `forward()` 和 `forwardWithEmbdOverride()` 复用
- `InferenceEngine.generateWithAudio()` — 完整管线：WAV → Mel 频谱 → Conformer 编码 → 混合嵌入 → LLM 生成

**混合嵌入策略**：
1. 音频编码器输出 `[n_embd, n_audio_tokens]`
2. 创建混合输入：音频嵌入（前 n_audio_tokens 个位置）+ 文本 token 嵌入（后续位置）
3. 通过 `ggml.concat` 拼接，送入 transformer 循环
4. prefill 完成后进入增量解码循环

**提交记录**：`d7c4a75` feat: audio-to-LLM integration with Gemma4 forwardWithEmbdOverride

### 2.3 🔧 进行中：视觉到LLM集成 (Phase 2)

**当前阻塞点**：
- 视觉编码器可运行并输出 `[n_embd, n_vision_tokens]`
- `generateWithImage()` 仍回退到纯文本生成
- 复用 `forwardWithEmbdOverride` 模式即可完成集成（与音频完全相同）

**待办事项**：
1. ❌ `generateWithImage()` 中调用 `forwardWithEmbdOverride` 替代 `self.generate()`
2. ❌ vendored `vendor/stb/stb_image.h` 用于 JPEG/PNG → RGB 解码
3. ❌ 端到端图文问答测试（`ocr.png`）

---

## 三、音频管线详解

### 3.1 音频预处理流程

```
WAV 文件 (16-bit PCM)
  → loadWav()           提取 F32 单声道样本
  → computeMelSpectrogram()
    → 预加重 (coeff=0.97)
    → STFT: 帧长 320 (20ms@16kHz), 帧移 160 (10ms)
    → Hann 窗 + FFT (n_fft=512, Accelerate vDSP)
    → Mel 滤波器组 (128 bins, 80-7600Hz, HTK 尺度)
    → log10 压缩
  → Mel 频谱 [128, n_frames]
```

### 3.2 Conformer 编码器流程

```
Mel 频谱 [n_mel_bins, n_frames]
  → 转置 [n_frames, n_mel_bins] 适配 Conv2D 布局
  → 子采样 Conv2D × 2 (stride=2, pad=1)
    → 每次 conv 后: LayerNorm + ReLU
    → 4× 下采样（时间维度和频率维度各 2× 子采样）
  → 展平 + 输入投影 (a.input_projection)
  → Conformer 层 × 12 (实际层数由 GGUF 检测)
    → FFN 1 (half-step, res_weight=0.5)
    → Chunked Local Self-Attention + RPE
      - chunk_size=12, max_past_horizon=12, context=24
      - Q/K per-dim scaling + softcap=50
    → Conv GLU (pointwise + depthwise causal conv)
    → FFN 2 (half-step, res_weight=0.5)
    → Layer output norm
  → 输出投影 (a.pre_encode.out)
  → 多模态嵌入器 (mm.a.soft_emb_norm + mm.a.input_projection)
  → 输出 [n_embd_llm, n_audio_tokens]
```

### 3.3 张量命名规范

llama.cpp 中 Gemma4A 的张量命名（见 `clip-impl.h`）：
- 子采样: `a.conv1d.{0,1}.weight`, `a.conv1d.{0,1}.bias`, `a.conv1d.{0,1}.norm.weight`
- 输入投影: `a.input_projection.weight`, `a.input_projection.bias`
- Conformer 层: `a.blk.{il}.attn_q.weight`, `.attn_k.weight`, `.ffn_up.weight`, 等
- 输出: `a.pre_encode.out.weight`, `a.pre_encode.out.bias`
- 多模态投影: `mm.a.soft_emb_norm.weight`, `mm.a.input_projection.weight`

代码中 `audio.zig` 已按此命名加载权重。

---

## 四、能力检测

```zig
// registry.zig detectCapabilities()
// 通过检查 GGUF 张量名检测模型的多模态能力
// 音频: a.conv1d.0.weight / a.pre_encode.out.weight / mm.a.input_projection.weight 存在
// 视觉: v.patch_embd.weight / v.position_embd.weight / mm.input_projection.weight 存在
// 元数据: gemma4.audio.encoder_type, gemma4.vision.encoder_type
```

---

## 五、mmproj 文件

mmproj 是一个独立的 GGUF 文件，包含多模态编码器权重（音频 Conformer、ViT 等），与主模型 GGUF 分离。

加载流程（`main.zig` `loadMMProj()`）：
```
1. 读取 mmproj GGUF 文件
2. 检测能力（音频/视觉张量是否存在）
3. 创建 ggml context (2GB)
4. MultiModalManager.init() → AudioEncoder.init() / VisionEncoder.init()
```

---

## 六、已知限制与风险

| 限制 | 说明 | 缓解 |
|------|------|------|
| **视觉到LLM未集成** | 嵌入未注入推理循环，回退到纯文本 | 复用 `forwardWithEmbdOverride` 模式（与音频相同）|
| **仅支持 PPM 图像** | JPEG/PNG 需 stb_image 解码 | 待 vendored `vendor/stb/stb_image.h` |
| **Gemma4 per-layer embedding** | llama.cpp 中 Gemma4 使用 per-layer 输入投影，当前 zllama.zig 未实现 | 简化为单层 token embedding，功能正确但不完全对齐 |
| **MoE 未实现** | Gemma4 可能使用 MoE FFN，当前仅支持密集 FFN | E2B 模型为密集架构，无影响 |
| **共享 KV 层未实现** | Gemma4 后几层复用前面的 KV | 影响 SWA 层效率，但非阻塞 |
| **量化精度** | Q4_K_M 及以下可能产生重复输出 | 建议 Q5_K_M+ 验证正确性 |
| **仅 CPU 后端** | Metal/CUDA 后端待实现 | CPU 速度可用但非最优 |
| **macOS 专用 FFT** | `fft.zig` 使用 Accelerate vDSP，仅 macOS 可用 | Linux/Windows 需纯 Zig FFT 后备 |

---

## 七、下一步路线图

### Phase 1: 音频功能 ✅ 已完成
- [x] `mm/` 模块基础设施
- [x] 音频编码器 (Conformer)
- [x] Mel 频谱预处理
- [x] CLI 参数 (`--audio`, `--mmproj`)
- [x] **音频嵌入 → LLM 注入** (`forwardWithEmbdOverride` + `transformerForward`)
- [ ] 端到端音频转录测试（需 Gemma4 模型 + mmproj）

### Phase 2: 视觉功能 🔧 进行中
- [x] 视觉编码器 (ViT / SigLIP / Gemma4UV)
- [x] CLI 参数 (`--image`)
- [x] PPM P6 图像加载 (`loadPPM`)
- [ ] **视觉嵌入 → LLM 注入** ← 当前阻塞点（复用 `forwardWithEmbdOverride` 模式）
- [ ] vendored `vendor/stb/stb_image.h` 支持 JPEG/PNG 解码
- [ ] 图文问答测试（`ocr.png`）

### Phase 3: 完善
- [ ] Linux FFT 后备实现
- [ ] 多轮对话中 KV Cache 同步
- [ ] 性能优化（预分配、缓存复用）

---

## 八、参考资源

- [Gemma4 音频编码器 (gemma4a.cpp)](deps/llama.cpp/tools/mtmd/models/gemma4a.cpp)
- [Gemma4 视觉编码器 (gemma4v.cpp)](deps/llama.cpp/tools/mtmd/models/gemma4v.cpp)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [DeepWiki - Multimodal Support](https://deepwiki.com/ggml-org/llama.cpp/6.5-python-gguf-tools)
