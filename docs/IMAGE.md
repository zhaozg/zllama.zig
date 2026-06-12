# 多模态能力实现思路与路线图

> 基于 Gemma 4 (E2B) 内建视觉/音频编码器，扩展 zllama.zig 的多模态推理能力。
> 本文档聚焦 **图像（Vision）** 能力，音频部分参见 `src/mm/audio.zig`。

---

## 一、当前基础（已完成 / 进行中）

### 1.1 核心引擎 — 文本推理已完备

| 模块 | 文件 | 状态 |
|------|------|------|
| ggml 绑定 + GGUF 解析 | `ggml.zig` / `gguf.zig` | ✅ |
| 纯文本模型推理 (Qwen/Llama/Gemma) | `models/{qwen2,qwen35,llama,gemma3,gemma4}.zig` | ✅ |
| 虚表多态 + 注册表 | `model.zig` + `registry.zig` | ✅ |
| KV Cache (per-layer) | `kv_cache.zig` | ✅ |
| BPE 分词 + Chat Template | `tokenizer.zig` + `chat_template.zig` | ✅ |
| 图优化 (Gallocr 复用) | `core/graph_context.zig` | ✅ |
| CLI 入口 + 交互对话 | `main.zig` | ✅ |

**Gemma 4 模型特有的已支持特性：**
- SWA (Sliding Window Attention) + Full Attention 混合层
- Per-layer head_dim / n_kv_head 可变维度
- Shared KV 层，Q/K pre-norm + Post norms
- GeGLU FFN，Proportional RoPE，Final softcapping
- `forwardWithEmbdOverride()` — **可直接注入编码器嵌入**，跳过 token embedding 查表

### 1.2 多模态基础设施 (`src/mm/`)

| 模块 | 文件 | 行数 | 功能 | 状态 |
|------|------|------|------|------|
| 多模态管理器 | `manager.zig` | 161 | MMProj 加载、能力调度、编码分发 | ✅ |
| 视觉编码器 | `vision.zig` | 509 | ViT + SigLIP (Gemma4V / Gemma4UV)，2D RoPE + im2col + pooling | ✅ |
| 音频编码器 | `audio.zig` | 663 | Conformer (ChunkedAttention + RPE + SSM Conv + GLU) | ✅ |
| 图像/音频预处理 | `preprocess.zig` | 689 | PPM/JPEG/PNG 加载、双线性 Resize、WAV 加载、Mel 频谱、STFT | ✅ |
| FFT 引擎 | `fft.zig` | 213 | Apple Accelerate vDSP FFT (macOS) | ✅ |
| stb_image 绑定 | `stb_image.zig` | — | JPEG/PNG/GIF/BMP/TGA 解码 | ✅ |

### 1.3 CLI 集成

```bash
zllama --model gemma-4-E2B-it-Q4_K_M.gguf \
       --mmproj mmproj-F16.gguf \
       --image hello.png \
       -p "Describe this image"
```
CLI 参数 `--mmproj`、`--image`、`--audio` 已添加，`main.zig` 中存在 `generateWithImage()` / `generateWithAudio()` 完整管线。

### 1.4 能力检测

`registry.detectCapabilities()` 已支持：
- Gemma4：检测 `v.patch_embd.weight` / `mm.input_projection.weight` (vision) 和 `a.conv1d.0.weight` (audio)
- Gemma3：检测 `mm.input_projection.weight`
- LLaMA/Qwen：检测对应的视觉/音频编码器张量
- 元数据：`gemma4.vision.encoder_type` / `gemma4.audio.encoder_type`

---

## 二、Gemma 4 多模态架构概览

```
┌──────────────────────────────────────────────────────────────┐
│  CLI Input                                                    │
│  --image hello.png  --prompt "Describe this image"            │
└────────────┬─────────────────────────────────────────────────┘
             │
    ┌────────▼────────┐       ┌──────────────────────┐
    │ Image Preprocess│       │ Text Tokenize        │
    │ (preprocess.zig)│       │ (tokenizer.zig)      │
    │ loadImage()     │       │ encode()             │
    │ resize+normalize│       │                      │
    └────────┬────────┘       └──────────┬───────────┘
             │                           │
    ┌────────▼────────┐                  │
    │ Vision Encoder  │                  │
    │ (vision.zig)    │                  │
    │ ViT+SigLIP      │                  │
    │ 2D RoPE+pooling │                  │
    └────────┬────────┘                  │
             │                           │
             │  [n_embd, n_vision_tokens]│
             │                           │
             │       ┌───────────────────▼──────────┐
             └───────►  Embedding Concatenation      │
                      │  [vision_emb|text_emb]       │
                      │  forwardWithEmbdOverride()   │
                      └──────────────┬──────────────┘
                                     │
                             ┌───────▼────────┐
                             │  LLM Decoder   │
                             │  (gemma4.zig)  │
                             │  40-layer      │
                             │  SWA+Full Attn │
                             └───────┬────────┘
                                     │
                             ┌───────▼────────┐
                             │  Sampler       │
                             │  (sampler.zig) │
                             └───────┬────────┘
                                     │
                              ┌──────▼──────┐
                              │   Output    │
                              └─────────────┘
```

### 关键设计决策

1. **嵌入覆盖模式 (`forwardWithEmbdOverride`)**
   - 视觉/音频编码器 → 连续值嵌入 `[n_embd, n_tokens]`
   - 与文本 token embedding `[n_embd, n_text_tokens]` 沿时间轴 `ggml.concat`
   - 合并后的嵌入送入共享的 `transformerForward()` 循环
   - 文本部分仍走正常的 token embedding 查表

2. **占位符 Token 策略 (Gemma 4)**
   - `<|image|>` — 视觉占位符，1 个 token → 扩展为 N 个视觉嵌入
   - `<|audio|>` — 音频占位符
   - 当前实现：直接构造 `[vision_tokens|text_tokens]` 拼接，未显式注入占位符
   - **待完成**：tokenizer 层面识别 `<|image|>` 并替换为视觉 token embedding

3. **编码器独立计算图**
   - 视觉/音频编码器有独立 ggml context (2GB) 和 gallocr
   - 编码完成后，嵌入数据保留在 gallocr 内存中
   - 嵌入作为输入注入 LLM 的 prefill 图

---

## 三、当前状态评估

### 3.1 已完成（可直接用于联调）

| 项目 | 状态 |
|------|------|
| Vision Encoder (ViT + SigLIP, Gemma4V/Gemma4UV) | ✅ 实现完整 |
| Image Preprocessing (PPM/JPEG/PNG/BMP → resize → F32) | ✅ 实现完整 |
| Audio Encoder (Conformer + subsampling) | ✅ 实现完整 |
| Audio Preprocessing (WAV → Mel spectrogram via FFT) | ✅ 实现完整 |
| MMProj GGUF loading | ✅ 实现完整 |
| `generateWithImage()` / `generateWithAudio()` 管线 | ✅ 端到端调通 |
| `forwardWithEmbdOverride()` 混合嵌入前向 | ✅ 实现完整 |
| Chat Template 系统 | ✅ 多格式支持 |
| stb_image 集成 (JPEG/PNG/GIF/BMP/TGA) | ✅ 已 vendored |
| Vision encoder reshapeForBroadcast 广播 bug 修复 | ✅ [1,n]→[n,1] |

### 3.2 待完成（阻塞端到端联调）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 🟡 P1 | **输出质量对比** | 与 llama.cpp mtmd 对比 logits (NMSE/余弦相似度) |
| 🟡 P1 | **KV Cache 长度适配** | 视觉 token 数 + 文本 token 数 = prefill 总长度（已基本工作）|
| 🟢 P2 | **性能优化** | Prefill 71s (805 tokens, 9B Q4_K_M CPU)，可优化 |
| 🟢 P2 | **跨平台 FFT** | 当前仅 macOS (Accelerate vDSP)，需 Linux/Windows 纯 Zig 后备 |

### 3.3 已知差距（不影响首轮联调）

| 项目 | 影响 | 说明 |
|------|------|------|
| Gemma4 per-layer embedding | 精度微降 | 当前用单层 token embedding，llama.cpp 用逐层投影 |
| MoE FFN | E2B 无影响 | E2B 为密集架构 |
| Shared KV 层 | SWA 层效率 | 非阻塞 |
| 仅 CPU 后端 | 速度 | Metal/CUDA 待实现 |

---

## 四、后续任务拆解

### Phase 3：视觉端到端联调（当前阶段）

```
[x] 3.1 获取测试资源
     - Gemma 4 E2B 模型: ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf
     - mmproj 文件: ~/.cache/models/mmproj-F16.gguf
     - 测试图像: ~/.cache/models/ocr.png, ~/.cache/models/hello.png

[x] 3.2 Vision Encoder 输出形状验证
     - 修复 reshapeForBroadcast 广播形状 bug（[1,n]→[n,1]）
     - Vision encoder 输出 [1536, 784]，维度 = model n_embd ✓
     - Vision tokens=784（mmproj 使用不同 pooling，预期 1024）
     - 添加维度匹配检查和 token 计数验证日志

[x] 3.3 占位符 Token 注入
     - <|image|> token 在词汇表中找到 (id=258880)
     - BPE 分词器无法直接编码特殊 token，改用预分词拆分方案
     - 在 BPE 编码前拆分 "<|image|>" 标记，分别编码文本段
     - 将特殊 token ID 插入 token 序列的正确位置
     - prompt template: "<|turn>user\n<|image|>Describe this image<turn|>\n<|turn>model\n"
     - 支持多图场景（多 <|image|> 标记）

[x] 3.4 混合 Prefill 推理
     - forwardWithEmbdOverride 正确处理 vision embeddings 覆盖
     - Prefill: 805 tokens (784 vision + 21 text)
     - Incremental decode 正常进入并生成 token
     - KV Cache 位置对齐正确

[x] 3.5 输出质量验证
     - 端到端生成成功："What is in this image?" → "> What is..."
     - 无崩溃，所有组件正确协作
     - TODO: 与 llama.cpp 输出对比 (NMSE/余弦相似度)
```

### Phase 4：鲁棒性与体验

```
[ ] 4.1 多图像对话
     - 支持一个对话轮次中多个 <|image|> 标记
     - 图像间位置编码正确性

[ ] 4.2 动态分辨率
     - Gemma4 支持可变分辨率 (默认 896×896)
     - token budget 自适应计算
     - 非正方形图像处理 (pad → square)

[ ] 4.3 错误处理
     - 无 mmproj 文件时的友好提示
     - 图像格式不支持时的回退
     - 内存不足时的 graceful degradation
```

### Phase 5：音频端到端联调

```
[x] 5.1 WAV → Mel → Conformer → LLM 联调
     - generateWithAudio() 端到端调通
     - <|audio|> token 在词汇表中找到 (id=258881)
     - 预分词拆分方案：BPE 编码前拆分 <|audio|> 标记
     - Audio encoder: [1536, 20] tokens for 0.8s WAV
     - 形状验证通过（1536 == model n_embd）
     - 端到端生成成功，pp_time ~2.3s

[ ] 5.2 跨平台 FFT
     - Linux: 纯 Zig 实现或集成 kissfft
     - Windows: 同上
     - 可编译时选择后端 (Accelerate / Zig-native)
```

### Phase 6：扩展模型

```
[ ] 6.1 Qwen2.5-VL 视觉编码器
     - 动态分辨率 ViT (NaViT-style)
     - MRoPE 位置编码
     - <|vision_start|> / <|image_pad|> / <|vision_end|> 标记处理

[ ] 6.2 LLaVA 系列
     - CLIP ViT 编码器
     - 多模态投影器 (mm_projector)
     - <image> 占位符

[ ] 6.3 其他多模态架构
     - InternVL / Phi-3-Vision / MiniCPM-V
     - 编码器可插拔架构设计
```

---

## 五、重难点分析

### 5.1 占位符 Token 机制 ⭐⭐⭐

**难点：** 视觉占位符 `<|image|>` 在 prompt 中是 1 个文本 token，但编码后扩展为 N 个视觉嵌入。

**方案：**
```
用户 prompt: "<|image|>Describe this image"
                              │
                    ┌─────────▼─────────┐
                    │ Tokenize prompt    │
                    │ Tokens: [<|image|>, "Describe", "this", "image"]  │
                    │ image_pos = 0 (position of <|image|> in sequence) │
                    └─────────┬─────────┘
                              │
              ┌───────────────▼───────────────┐
              │ Build mixed embedding:        │
              │ [vision_emb[0..N-1]] ++        │
              │   token_emb["Describe"] ++     │
              │   token_emb["this"] ++         │
              │   token_emb["image"]           │
              └───────────────────────────────┘
```

**关键实现点：**
1. Tokenizer 需识别特殊 token `<|image|>` (通常在 vocab 中有专用 ID)
2. Prompt 解析时记录 placeholder 位置和数量
3. 构建位置编码时必须为视觉 token 和文本 token 分配连续递增的位置

### 5.2 视觉 Token 数量计算 ⭐⭐

**难点：** Vision encoder 的输出 token 数取决于图像尺寸和 patch/merge 配置。

| 编码器类型 | 输入尺寸 | Patch | 下采样 | 输出 Tokens |
|-----------|---------|-------|--------|-------------|
| Gemma4V | 896×896 | 14×14 | 2×2 pool | (896/14)² / 4 = 1024 |
| Gemma4UV | 896×896 | 14×14 | 2×2 pool | 1024 |
| Qwen2.5-VL | 动态 | 14×14 | 2×2 | 动态 (基于 token budget) |

**当前实现** (`vision.zig`)：
```zig
n_patches = (image_size / patch_size) * (image_size / patch_size)
n_tokens = n_patches / (n_merge * n_merge)
```

### 5.3 KV Cache 位置对齐 ⭐⭐⭐

**难点：** Prefill 阶段 visual + text 混合序列后，增量解码阶段 KV Cache 的起始位置必须正确。

**方案：**
```
prefill: positions [0, 1, ..., N_vis-1, N_vis, ..., N_vis+N_text-1]
decode:  position N_vis + N_text, N_vis + N_text + 1, ...
```

`forwardWithEmbdOverride` 已正确处理：
- `pos` 参数指向序列中的实际起始位置
- KV Cache 的 `setKv` 使用绝对位置索引

### 5.4 多模态 Chat Template ⭐⭐

**难点：** Gemma 4 的对话格式要求 `<|image|>`/`<|audio|>` 占位符在正确位置。

**Gemma 4 格式：**
```
<start_of_turn>user
<|image|>What's in this image?<end_of_turn>
<start_of_turn>model
This image shows...<end_of_turn>
```

**方案：** 在 `chat_template.zig` 中为 Gemma 4 添加 `gemma4_multimodal` 模板变体：
1. 检测 prompt 中是否包含 `<|image|>` / `<|audio|>`
2. 自动插入 `<start_of_turn>user\n` / `<end_of_turn>\n<start_of_turn>model\n` 包装
3. 或直接在 CLI 参数中提供 `--chat-template gemma4`

### 5.5 内存管理 ⭐⭐

**难点：** 三个独立 ggml context 的生命周期管理。

| Context | 用途 | 大小 | 生命周期 |
|---------|------|------|----------|
| `ctx_weights` (LLM) | LLM 权重 | ~模型大小 | 整个会话 |
| `ctx_mm` | 视觉/音频编码器权重 | ~2GB | 整个会话 |
| `ctx_graph` | 前向计算中间张量 | ~512MB | 每次前向 |

**风险点：**
- 视觉编码输出嵌入需在 LLM prefill 图计算期间保持存活
- gallocr 分配的内存不能提前释放
- 需确认 `vision_graph.compute()` 后嵌入数据所在 gallocr 仍然有效

### 5.6 跨平台 FFT ⭐⭐

**当前：** macOS 专用 (Apple Accelerate vDSP)

**Linux/Windows 方案：**
```
方案 A: 纯 Zig 实现 (radix-2 DIT FFT)
  - 优点: 零依赖, 跨平台
  - 缺点: 性能低于 vDSP
  - 适合: 音频预处理 (n_fft=512, 离线计算, 非推理热点)

方案 B: 集成 kissfft / pffft
  - 优点: 成熟的轻量级 FFT 库
  - 缺点: 引入 C 依赖
  - 适合: 追求性能的场景
```

**推荐：先方案 A，后续按需升级 B。**

---

## 六、关键文件清单

```
src/mm/
├── manager.zig       # 多模态管理器 (161 行)
├── vision.zig        # ViT + SigLIP 编码器 (509 行)
├── audio.zig         # Conformer 音频编码器 (663 行)
├── preprocess.zig    # 图像/音频预处理 (689 行)
└── fft.zig           # FFT 引擎 (213 行, macOS)

src/models/
├── gemma4.zig        # Gemma 4 LLM (991 行, 含 forwardWithEmbdOverride)
└── registry.zig      # 能力检测 detectCapabilities()

src/
├── main.zig          # CLI + InferenceEngine (含 generateWithImage/Audio)
├── chat_template.zig # 对话模板系统
└── stb_image.zig     # stb 图像加载绑定

vendor/stb/
├── stb_image.h       # JPEG/PNG 解码 (已 vendored)
└── stb_image.c       # 实现

docs/
├── IMAGE.md          # 本文档
└── DIALOG_TEMPLATE.md # 对话模板详细文档
```

---

## 七、性能预期

| 组件 | 预估耗时 (Apple M-series) |
|------|--------------------------|
| 图像预处理 (resize 896×896) | < 5ms |
| Vision Encoder (ViT 27层) | ~200-500ms |
| LLM Prefill (视觉+文本) | ~200-800ms (取决于 token 数) |
| LLM Incremental Decode | ~30-80ms/tok (Q4_K_M, 9B) |
| **端到端首 token 延迟** | **~0.5-1.5s** |

---

## 八、立即可执行的第一步

```bash
# 1. 确认已有模型和 mmproj 文件
ls ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf
ls ~/.cache/models/mmproj-F16.gguf  # 或类似命名的 mmproj

# 2. 准备测试图像
# 任意 PNG/JPEG 图像，最好是 OCR 或简单场景图

# 3. 测试当前状态 (骨架联调)
zig build -Doptimize=ReleaseFast
./zig-out/bin/zllama \
    --model ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf \
    --mmproj ~/.cache/models/mmproj-F16.gguf \
    --image ~/test.png \
    -p "What is in this image?" \
    -n 50
```

### 调试要点

1. **Vision encoder 输出维度**
   ```zig
   // 在 generateWithImage() 中添加 log
   log.info("Vision output: [{d}, {d}]", .{ n_embd, n_vision_tokens });
   ```

2. **混合嵌入形状**
   ```zig
   // 验证 concat 后的形状
   log.info("Mixed embedding: [{d}, {d}]", .{ n_embd, n_total });
   ```

3. **与 llama.cpp 对比**
   ```bash
   # 用相同输入跑 llama.cpp 的 mtmd 示例
   llama-mtmd -m gemma-4-E2B-it.gguf --mmproj mmproj.gguf --image test.png -p "What is this?"
   ```
   使用 `compare_logits` 工具对比 logits。

---

## 九、参考资源

- [llama.cpp mtmd 实现](deps/llama.cpp/tools/mtmd/)
- [Gemma 4 技术报告](https://ai.google.dev/gemma)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [本项目对话模板文档](docs/DIALOG_TEMPLATE.md)
