# mtmd-audio-flow

为了将 `zllama.zig` 的音频处理与 `llama.cpp` 对齐，需要对比两者在每个阶段的**输入数据格式、处理参数、输出张量形状和数值范围**。
以下是音频处理流水线的完整阶段分解，每个阶段都标注了关键节点和需对齐的细节。

---

## 📊 音频处理流水线（从 WAV 到嵌入向量）

| 阶段 | 主要任务 | 输入 | 输出 | 关键参数 / 需对齐项 |
|------|---------|------|------|-------------------|
| **1. 文件加载与解码** | 读取 WAV 文件，提取 PCM 数据 | WAV 文件路径 | 音频采样数据（`f32` 数组，范围 -1.0~1.0），采样率（Hz），声道数 | 是否支持多声道（通常只用单声道），采样率转换（如需要，llama.cpp 内部固定 16kHz） |
| **2. 重采样（可选）** | 将音频统一到 16kHz | PCM 数据 + 原采样率 | 16kHz PCM 数据 | 重采样算法（llama.cpp 使用 libsamplerate 或简单线性插值？应保持一致） |
| **3. 分帧（Framing）** | 将 PCM 数据分割为重叠帧 | PCM 数据（16kHz） | 帧序列（每帧 `window_len` 个采样点） | 窗口长度 `window_len` = **320**，帧移 `hop_len` = **160**（10ms 步长） |
| **4. 加窗与 FFT** | 对每帧应用 Hann 窗口，计算功率谱 | 帧数据（320 采样点） | 功率谱（复数或幅度，`n_fft/2+1 = 257` 个 bins） | 窗口类型：**周期性 Hann 窗口**（`torch.hann_window(320, periodic=True)`）；FFT 点数 = **512**（含零填充） |
| **5. 梅尔滤波器组** | 将功率谱映射到梅尔刻度 | 功率谱（257 bins） | 梅尔谱（128 bins） | 梅尔滤波器数量 = **128**；梅尔刻度类型：**HTK**；频率范围：0~8kHz；`mel_floor` = **1e-3** |
| **6. 对数变换** | 对梅尔谱取对数（log-mel） | 梅尔谱（能量值） | log-mel 谱（单位：dB） | 是否加小常数（`1e-6`）避免 log(0)；是否做归一化（**无** Whisper 风格归一化） |
| **7. 全局归一化（可选）** | 对 log-mel 谱进行均值和方差归一化 | log-mel 谱（`[frames, 128]`） | 归一化后的 log-mel 谱 | **Gemma 4 音频不进行归一化**（无均值方差统计），直接输入编码器 |
| **8. 特征张量构建** | 将帧序列堆叠为 `[frames, mel_bins]` 张量 | 归一化 log-mel 谱（或原始） | `[T, 128]` 浮点张量 | 数据类型：**f32**；布局：帧优先（time-major） |
| **9. 音频编码器（Conformer）** | 将 log-mel 谱编码为嵌入序列 | `[T, 128]` 张量 | `[T, 1536]` 嵌入向量（T 为帧数，模型 `n_embd=1536`） | 编码器架构：**12 层 USM 风格 Conformer**；注意力：**因果注意力**；输出投影：1024→1536；RMS Norm eps = **1e-6** |
| **10. 嵌入后处理** | 可能进行 softcap 或投影 | `[T, 1536]` 嵌入 | `[T, 1536]` 最终嵌入 | 是否应用 logit softcapping（Gemma 4 音频 softcap = 50.0）；是否通过一个线性层映射到模型维度（已在编码器输出中完成） |
| **11. 占位符替换** | 将嵌入向量替换文本序列中的 `<\|audio\|>` 占位符 token | 文本 token 序列 + 嵌入向量 | 最终输入 token 序列（嵌入替换后） | 替换位置：**在用户消息开头，紧贴文本**（无额外换行）；需确保占位符 token ID（如 `258881`）对应 `<\|audio\|>`；替换后 `media_count == n_media_tokens` |


## 🔍 重点对齐检查清单

- [ ] **采样率**：是否强制为 16kHz？重采样算法是否与 llama.cpp 一致？
- [ ] **窗口与 FFT**：窗口长度 320，hop 160，零填充至 512，Hann 窗口是否 `periodic=True`？
- [ ] **梅尔滤波器**：HTK 梅尔刻度，128 个滤波器，频率范围 0~8000 Hz，`mel_floor=1e-3`？
- [ ] **对数变换**：是否使用 `log(max(mel, 1e-10))`？是否有归一化？
- [ ] **编码器架构**：12 层 Conformer，因果注意力，RMS Norm eps=1e-6，输出投影维度 1536？
- [ ] **占位符位置**：是否紧贴用户消息，无多余换行或空格？
- [ ] **数值精度**：是否全程使用 `f32`（或 `f64` 在预处理中）？llama.cpp 的音频预处理使用双精度计算某些常量，但最终输入编码器是 `f32`。

---

## 💡 如何获取各阶段的数据进行对比

- **在 zllama.zig 中添加调试输出**：在 `audio.zig` 或 `preprocess.zig` 中的每个阶段结束后，将张量前几个值、形状打印出来，并保存为二进制文件（如 `.npy` 或 `.raw`）。
- **在 llama.cpp 中使用 `--verbose-prompt` 查看嵌入预览**，但更详细的方式是在 `mtmd-audio.cpp` 中插入 `printf`，打印同样阶段的数据。
- **对比 Mel 频谱**：可先对比阶段 7 输出的 log-mel 谱（形状 `[T,128]`），确保均值、方差、数值范围接近。如果这一步就差异大，说明预处理参数不匹配，不必深入编码器。

---

## 🧪 推荐的对齐步骤

1. **先用同一个 WAV 文件**，分别用 `llama.cpp` 和 `zllama.zig` 运行，打印阶段 7（log-mel 谱）和阶段 9（编码器输出）的前几个值。
2. **如果 log-mel 谱一致**，则问题在编码器实现或后续处理（如 softcap）。
3. **如果 log-mel 谱差异大**，则对齐预处理参数。

按照此清单逐一核对，即可精准定位差异点。

### 📊 阶段详细对比

*   **阶段 1-2 (文件加载与重采样)**：两者一致。这是标准预处理，llama.cpp 同样会加载 WAV 并重采样至 16kHz。
*   **阶段 3 (分帧)**：**核心参数一致**。均使用窗口长度 **320**，帧移 **160**。
*   **阶段 4 (加窗与 FFT)**：**核心参数一致**。均使用**周期性 Hann 窗口**，零填充至 **512** 点 FFT。
*   **阶段 5 (梅尔滤波器组)**：**核心参数一致**。均使用 **HTK 梅尔刻度**，**128 个滤波器组**，`mel_floor=1e-3`。
*   **阶段 6-7 (对数变换与全局归一化)**：**行为一致**。均**不进行** Whisper 风格的归一化。llama.cpp 实现中可能包含 `log` 和 `sqrt` 等操作。
*   **阶段 8 (特征张量构建)**：两者一致。输出形状均为 `[T, 128]` 的浮点张量。
*   **阶段 9 (音频编码器-Conformer)**：**架构完全一致**。均为 **12 层 USM 风格 Conformer**：
    *   **层结构**：FFN → Self-Attention → Causal Conv1D → FFN → Norm。
    *   **子采样**：2x Conv2D (stride=2) 带 LayerNorm。
    *   **注意力**：全自注意力 + 正弦 RPE + 滑动窗口掩码 (24)。
    *   **输出**：1024 → 1536 → RMSNorm。
*   **阶段 10 (嵌入后处理)**：**行为一致**。应用 Logit softcapping (50.0)。
*   **阶段 11 (占位符替换)**：**概念一致**。用音频嵌入替换文本中的占位符。llama.cpp 使用 `<__media_...>` 等唯一标识符，替换逻辑与你的 `zllama.zig` 相同。

### 💎 总结与建议

现在最关键的对齐工作，在于**核实具体代码实现中的“数值”与“细节”**：

1.  **核对数值常量和计算公式**：重点关注窗口函数的具体实现、梅尔滤波器组的计算公式、对数变换（`log` 或 `log10`）以及 softcapping 的阈值 (50.0)。
2.  **核对张量操作**：确认 `zllama.zig` 中张量的**形状、维度顺序**是否与 llama.cpp 完全一致。
3.  **利用官方验证标准**：官方实现提到与 PyTorch 的梅尔频谱余弦相似度达到 **0.9998**，你可以此为目标进行验证。

只要在这些细节上做到完全一致，就能确保两个实现的音频处理输出是相同的。


---

## 📁 文件结构

为 `zllama.zig` 设计一个模块化、可测试、高性能的音频处理流水线，关键在于

- **清晰分离阶段**
- **统一数据类型*、
- **集中配置管理**
- **易于调试**。

以下是一套具体的代码组织建议，充分结合 Zig 的特性。

```
src/mtmd/audio/
├── mod.zig            # 公开 API 和模块入口
├── pipeline.zig       # 流水线编排器，串联各阶段
├── config.zig         # 所有配置参数（窗口、FFT、梅尔等）
├── loader.zig         # 文件加载与重采样
├── framing.zig        # 分帧 + 加窗
├── fft.zig            # FFT（可能已有，需集成）
├── mel.zig            # 梅尔滤波器组
├── log_transform.zig  # 对数变换
├── encoder.zig        # Conformer 编码器（调用 GGML）
├── postprocess.zig    # softcapping 等后处理
├── types.zig          # 阶段间传递的数据结构
└── test/              # 单元测试与 golden 测试
    ├── test_pipeline.zig
    └── golden/        # 存放参考输出（二进制）
```

---

## 🧩 核心设计原则

### 1. **阶段独立，输入输出明确**

每个阶段对应一个函数，输入和输出是纯数据结构（`struct`），不依赖全局状态。便于单独测试和替换。

示例（`framing.zig`）：
```zig
pub const FrameOutput = struct {
    frames: []Frame, // 帧数据（按时间排列）
    n_frames: usize,
};

pub fn frame(audio: []const f32, config: *const Config) !FrameOutput {
    // 使用 config.window_len, config.hop_len
    // 返回帧数组
}
```

### 2. **统一配置结构体**
所有参数集中在 `config.zig`，并可通过 `comptime` 静态检查。配置应包含所有超参数，并可序列化为 JSON/YAML 方便对比。

```zig
pub const AudioConfig = struct {
    sample_rate: usize = 16000,
    window_len: usize = 320,
    hop_len: usize = 160,
    n_fft: usize = 512,
    mel_bins: usize = 128,
    mel_floor: f32 = 1e-3,
    // ...
};
```

### 3. **使用 Arena 分配器管理临时内存**
整个流水线可在一个 `ArenaAllocator` 下运行，避免碎片化，且易清理。

```zig
pub fn runPipeline(alloc: std.mem.Allocator, audio_data: []f32, config: AudioConfig) ![]f32 {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const tmp_alloc = arena.allocator();

    const frames = try framing.frames(audio_data, config, tmp_alloc);
    const specs = try fft.compute(frames, config, tmp_alloc);
    const mel = try mel.filter(specs, config, tmp_alloc);
    // ...
    const embeddings = try encoder.encode(mel, config, tmp_alloc);
    return embeddings; // 返回的嵌入由调用者管理（或改用 Arena 传递给上层）
}
```

### 4. **数据持久化以便调试**
提供 `dumpStage` 函数，可将任意阶段的输出保存为 `.npy` 或 `.raw` 文件，便于与 llama.cpp 的输出对比。

```zig
pub fn dumpTensor(path: []const u8, data: anytype) !void {
    // 使用 zig-npy 或自定义二进制格式
}
```

### 5. **实现可选的日志级别**
在关键阶段添加详细日志，但默认关闭（`std.log.scoped(.audio)`），通过环境变量或构建配置开启。

```zig
const log = std.log.scoped(.audio);
log.debug("Mel spectrum shape: {}x{}", .{n_frames, mel_bins});
```

### 6. **利用 `comptime` 优化静态参数**
对于固定参数（如 `mel_bins=128`），可以在编译期生成滤波器组系数，避免运行时计算。

```zig
const MEL_BINS = comptime 128;
const filterbank: [MEL_BINS][257]f32 = comptime generateMelFilterbank();
```

### 7. **分离算法与执行后端**
- `fft.zig` 仅封装 FFT 算法，具体实现（如 `AccelFFT` 或 `ggml`）通过编译期选择。
- `encoder.zig` 调用 `ggml` 计算图，但将输入/输出张量的构建逻辑封装在本模块内。

---

## 🧪 测试策略

- **单元测试**：验证每个阶段的输入输出形状、边界条件。
- **集成测试**：使用预设的 `hello.wav`，将整个流水线输出与 llama.cpp 的导出嵌入对比（余弦相似度需 > 0.999）。
- **Golden 测试**：将 llama.cpp 各阶段的中间输出存为 golden 文件，在 CI 中自动比较。

---

## 📌 与现有 `mtmd/` 模块的集成

现有 `src/mtmd/audio.zig` 可能已包含部分逻辑，可逐步重构为以上结构。建议先提取核心函数，并保持对外接口不变，逐步替换内部实现。

---

## 💡 总结

通过以上组织，你将获得一个**清晰、可维护、易于调试**的音频处理流水线，便于与 llama.cpp 对齐。
同时，这种设计也能轻松扩展到其他模型的音频处理（如 Whisper 或 Qwen2-Audio）。
