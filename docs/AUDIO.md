# Audio 端到端集成

## 命令

```
 zig-out/bin/zllama -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/.cache/models/mmproj-BF16.gguf \
  --audio ~/.cache/models/hello.wav \
  -p "Transcribe audio to text" -n 100
```

期望输出文本 `hello world`

## 当前状态

音频端到端集成已基本完成。管线如下：

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
  → Conformer 编码器 (audio.zig)
    → 子采样 Conv2D × 2 (stride=2, pad=1)
    → 输入投影
    → Conformer 层 × N (ChunkedAttention + RPE + Conv GLU + FFN)
    → 输出投影
    → 多模态嵌入器 (mm.a.soft_emb_norm + mm.a.input_projection)
  → 输出 [n_embd_llm, n_audio_tokens]
  → forwardWithEmbdOverride() 混合嵌入到 LLM
  → 增量解码生成文本
```

## 与 llama.cpp 的关键差异

### 1. 注意力 permute 布局

| 操作 | llama.cpp (gemma4a.cpp) | zllama.zig (audio.zig) | 等价性 |
|------|------------------------|------------------------|--------|
| Qcur permute | `(0,3,1,2)` → `[D,C,B,H]` | `(0,2,1,3)` → `[D,C,H,B]` | ✅ H/B 轴交换，数学等价 |
| Kblk permute | `(0,3,1,2)` → `[D,S,B,H]` | `(0,2,1,3)` → `[D,S,H,B]` | ✅ H/B 轴交换，数学等价 |
| Vblk permute | `(1,3,0,2)` → `[S,D,B,H]` | `(1,2,3,0)` → `[B,D,H,S]` → Vt=`[S,H,D,B]` | ✅ 最终结果等价 |
| attn@V permute | `(0,2,3,1)` → `[D,H,C,B]` | `(1,3,0,2)` → `[D,C,B,H]` → `(0,3,1,2)` → `[D,H,C,B]` | ✅ 最终结果相同 |

**结论**：虽然中间轴顺序不同（H 和 B 交换），但 ggml 的 `mul_mat` 只 contracts ne[0] 维度，其他维度作为 batch 处理，因此数学上等价。

### 2. k_scale 计算

| 实现 | 公式 | 结果 |
|------|------|------|
| llama.cpp | `logf(1.0f + expf(1.0f)) / logf(2.0f)` | `log2(1+e)` |
| zllama.zig | `@log2(1.0 + @exp(1.0))` | `log2(1+e)` |

✅ 等价

### 3. build_mm clamp 操作

llama.cpp 的 `build_mm` 方法在 `mul_mat` 前后有可选的 `clamp` 操作（通过 `clamp_info_map` 控制），用于限制输入/输出值范围。zllama.zig 未实现此机制。

**影响**：低，因为 `clamp_info_map` 通常为空，此时 `build_mm` 退化为普通 `mul_mat`。

### 4. norm_eps

| 实现 | 值 | 来源 |
|------|-----|------|
| llama.cpp | `1e-6f` | 硬编码 |
| zllama.zig | 从 GGUF 元数据读取 | `clip.audio.attention.layer_norm_epsilon` |

**影响**：低，GGUF 中的值通常也是 `1e-6`。

### 5. 输入张量填充方式

llama.cpp 使用 `ggml_set_input()` 标记 pos_emb 和 kq_mask 为输入张量，由外部填充数据。zllama.zig 在 `encode()` 函数内部通过 `fillSinusoidalPosEmb()` 和 `fillChunkedAttentionMask()` 直接填充。

**影响**：功能等价，实现方式不同。

## 验证方法

1. 使用 `tools/compare_logits.zig` 对比 zllama.zig 和 llama.cpp 的音频编码器输出
2. 运行端到端命令验证输出文本

## 已知问题

- [ ] 端到端音频推理尚未实际验证（需要 `hello.wav` 测试文件）
- [ ] macOS 专用 FFT（Accelerate vDSP），Linux/Windows 需纯 Zig FFT 后备
- [ ] 音频编码器 `norm_eps` 从 GGUF 读取，若缺失则使用默认值

## 部分分析

实现音频支持的关键文件主要集中在 `deps/llama.cpp/tools/mtmd/` 目录中，核心文件如下：

## 一、音频编码器实现

| 文件 | 作用 |
|------|------|
| **`mtmd-audio.h`** | 音频编码器的公共接口声明，定义音频预处理参数、Conformer 编码器的初始化与执行函数。 |
| **`mtmd-audio.cpp`** | 实现音频编码器的核心逻辑：<br>- Mel 频谱提取（STFT、Mel 滤波器组）<br>- Conformer 模型的前向计算（子采样卷积、分块注意力、RPE、GLU 等）<br>- 音频嵌入到 LLM 空间的投影 |
| **`mtmd-audio.cpp`** 内部会调用 `clip.cpp` 中的基础张量操作（`ggml_*` 函数）来构建计算图。 |

## 二、底层基础设施

| 文件 | 作用 |
|------|------|
| **`clip.h`** | 定义多模态编码器的通用接口（图像、音频），包括上下文管理、媒体预处理、张量构建等。 |
| **`clip.cpp`** | 实现媒体编码器的公共部分，例如张量加载、`ggml` 图构建辅助函数。音频编码器会复用其中的内存管理和张量操作。 |
| **`mtmd.h`** | 多模态顶层接口，声明 `mtmd_context`、`mtmd_init_from_file`、`mtmd_encode` 等统一 API。 |
| **`mtmd.cpp`** | 实现多模态调度逻辑，根据媒体类型（图像/音频）调用对应的编码器，管理推理流程。 |

## 三、工具与测试

| 文件 | 作用 |
|------|------|
| **`mtmd-cli.cpp`** | 命令行演示程序，展示如何使用 `libmtmd` 进行音频/图像推理，可作为集成参考。 |
| **`mtmd-helper.h/cpp`** | 辅助函数，例如加载 `mmproj` 文件、解析 GGUF 元数据等。 |
| **`test-2.mp3`** | 测试音频文件（示例）。 |

## 四、与 `zllama.zig` 集成的关键点

如果您希望在不依赖 `llama.cpp` 的情况下自己实现音频编码器，以下是必须实现的模块（对应上述文件的功能）：

1. **音频预处理**（对应 `mtmd-audio.cpp` 中的 `mtmd_audio_preprocessor_gemma4a`）：
   - 从 WAV 读取 PCM
   - 预加重、分帧、加窗、FFT
   - Mel 滤波器组（HTK 尺度，128 bins，80-7600 Hz）
   - 对数压缩

2. **Conformer 编码器**（对应 `mtmd-audio.cpp` 中的 `mtmd_audio_encode`）：
   - 子采样卷积（2 层，stride=2）
   - 12 层 Conformer block（包含 FFN、分块局部自注意力 + RPE、卷积 GLU）
   - 输出投影到 LLM 嵌入空间（1024 → 1536）

3. **资源管理**（对应 `clip.cpp` 中的 `clip_ctx` 和 `mtmd.cpp` 中的 `mtmd_context`）：
   - 加载 `mmproj` 文件中的权重
   - 分配 `ggml` 上下文内存
   - 构建并执行计算图

## 五、与您当前代码的对应关系

您已经实现了 `src/mm/audio.zig`（编码器）和 `src/mm/preprocess.zig`（预处理），与 `mtmd-audio.cpp` 的功能对应。`llama.cpp` 的这些文件可以作为**验证参考**，特别是 `mtmd-audio.cpp` 中的具体参数（如 `C=12`、`P=12`、`softcap=50.0`）和分块注意力的实现细节，有助于调试您的 Zig 版本。

**建议**：将 `llama.cpp` 的 `mtmd-audio.cpp` 与您的 `audio.zig` 逐行对比，重点关注：
- 子采样卷积的输入/输出布局（`permute` 顺序）
- 分块注意力的 `view` 和 `roll` 操作
- RPE 位置编码的生成方式

这能有效定位当前编码器输出语义错误的根源。
