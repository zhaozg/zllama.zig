# mtmd-vision-flow

基于与[音频处理](./mtmd-audio-flow.md)相同的设计原则，为 `zllama.zig` 的图像处理流水线设计一份完整的模块化规划。

`llama.cpp` 的图像处理正是通过 `libmtmd` 库和 `clip.cpp` 编码器来完成的，
其核心流程与“加载 → 预处理 → 编码 → 投影 → 替换”逻辑链完全一致。

下面我将你的阶段与 `llama.cpp` 的源码进行逐一比对：

| 你规划的阶段 | `llama.cpp` 对应实现与说明 | 关键源码/参数 |
| :--- | :--- | :--- |
| **1. 文件加载与解码** | **`mtmd` 图像加载器**：通过 `stb_image` 等库将 PNG/JPEG 等格式解码为 RGB 像素数据。 | `clip_image_u8` 结构体存储解码后的图像。 |
| **2. 图像预处理** | **`clip.cpp` 图像预处理**：将图像缩放到模型指定尺寸（如 896x896），并使用均值和标准差进行归一化。 | `clip.vision.image_size`、`image_mean` / `image_std`。 |
| **3. 分块嵌入 (Patch Embedding)** | **`clip.cpp` ViT 编码器**：将图像分割为固定大小的 Patch（如 14x14），并线性投影到嵌入空间。 | `clip.vision.patch_size` 定义 Patch 大小。 |
| **4. 位置编码** | **`clip.cpp` ViT 编码器**：为 Patch 序列添加可学习的位置编码。 | 位置嵌入权重 `v.position_embd.weight`。 |
| **5. ViT 编码器** | **`clip.cpp` ViT 编码器**：通过多层 Transformer 处理 Patch 序列，提取图像特征。 | 关键参数由 GGUF 元数据定义：`block_count`、`attention.head_count`、`layer_norm_epsilon` 等。 |
| **6. 后处理与投影** | **`clip.cpp` 投影器 (Projector)**：将 ViT 输出映射到 LLM 的嵌入空间。 | `clip.projector_type` 定义投影器类型（如 MLP, LDP）。 |
| **7. 占位符替换** | **`mtmd` 上下文管理**：将图像嵌入向量替换文本序列中的媒体占位符。 | 处理媒体标记（`<\|image\|>` 等）和位置索引。 |

### 💡 关键发现与补充

*   **核心处理单元**：`llama.cpp` 的图像处理核心是 **`clip.cpp`**，它负责从加载 `mmproj` 文件到执行 ViT 编码器的全部工作。
*   **Gemma 4 的特殊性**：需要留意的是，Gemma 4 采用了“无编码器 (encoder-free)”设计。对于这种模型，你规划的 **阶段 3 至 6** 可能会被简化为一个或多个**轻量级投影层**。但整体流水线的阶段划分和概念仍然是适用的。
*   **数据格式**：`llama.cpp` 内部使用 `clip_image_u8`（uint8）和 `clip_image_f32`（float32）两种图像数据结构，这与你规划中提到的数据类型一致。
*   **动态分辨率**：部分模型（如 Qwen2-VL）支持动态分辨率，这可能会引入额外的图像分块（tiling）或调整大小步骤。

### ✅ 总结

你设计的**七个阶段**与 `llama.cpp` 的 `libmtmd` 和 `clip.cpp` 实现**完全对齐**，该设计抓住了核心处理链路，可以作为你实现和调试的可靠蓝图。
---

## 📁 文件结构

```
src/mtmd/vision/
├── mod.zig            # 公开 API 和模块入口
├── pipeline.zig       # 流水线编排器，串联各阶段
├── config.zig         # 所有配置参数（尺寸、归一化、分块等）
├── loader.zig         # 图像文件加载与解码（调用 stb_image）
├── preprocess.zig     # 预处理：尺寸调整、归一化、通道转换
├── patch_embed.zig    # 图像分块（Patch Embedding）
├── position_embed.zig # 位置编码（可选，如 ViT 的位置编码）
├── encoder.zig        # ViT 编码器（调用 GGML 或直接实现）
├── postprocess.zig    # 后处理（如投影到模型空间）
├── types.zig          # 阶段间传递的数据结构
└── test/              # 单元测试与 golden 测试
    ├── test_pipeline.zig
    └── golden/        # 存放参考输出（二进制）
```

---

## 🧩 核心阶段详解

### 阶段 1: 文件加载与解码
- **任务**：读取图像文件（PNG/JPEG/WebP），解码为像素数据。
- **输入**：图像文件路径。
- **输出**：`ImageData` 结构（`[height][width][channels]u8`，RGB 或 RGBA）。
- **实现**：`stb_image` 或 Zig 原生图像库。
- **对齐要点**：确保解码后的颜色通道顺序（RGB）与 llama.cpp 一致。

---

### 阶段 2: 预处理
- **任务**：缩放、归一化、通道转换。
- **输入**：原始像素数据（`u8`，0-255）。
- **输出**：浮点张量 `[height, width, channels]f32`（值范围通常为 0~1 或归一化到特定均值/方差）。
- **子步骤**：
  - **尺寸调整**：调整到模型期望尺寸（Gemma 4 Vision：**896x896**）。
  - **归一化**：根据 ViT 训练时的均值和标准差进行标准化（如 `mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]`）。
  - **通道顺序**：确保 RGB 或 BGR（通常为 RGB）。
- **对齐要点**：归一化参数必须与 llama.cpp 一致（可查看 `clip.cpp` 中的 `mean`/`std` 数组）。

---

### 阶段 3: 分块嵌入（Patch Embedding）
- **任务**：将图像分割为固定大小的 Patch，并线性投影到嵌入维度。
- **输入**：预处理后的图像张量 `[H, W, C]`。
- **输出**：Patch 序列 `[num_patches, patch_embd]`。
- **参数**：Patch size（通常为 14×14），`num_patches = (H/patch_size) * (W/patch_size)`。
- **实现**：使用二维卷积或切片操作。
- **对齐要点**：确认 Patch 大小、步长（通常与大小相同）、以及投影矩阵是否与 llama.cpp 一致。

---

### 阶段 4: 位置编码
- **任务**：为 Patch 序列添加位置编码（可学习或固定）。
- **输入**：Patch 序列 `[num_patches, embd]`。
- **输出**：添加位置编码后的张量。
- **对齐要点**：位置编码矩阵的初始化方式（如 Sin/Cos 或可学习）和形状。

---

### 阶段 5: ViT 编码器
- **任务**：通过 Transformer 编码器处理 Patch 序列，提取图像特征。
- **输入**：带位置编码的 Patch 序列 `[num_patches, embd]`。
- **输出**：编码后的图像特征 `[num_patches, embd]`。
- **架构**：
  - **层数**：27 层（Gemma 4 Vision）
  - **注意力头数**：16
  - **隐藏层维度**：1152
  - **前馈网络维度**：约 4 * hidden_size
- **对齐要点**：层数、头数、激活函数（GELU）、LayerNorm/RMSNorm eps（1e-6）。

---

### 阶段 6: 后处理与投影
- **任务**：将 ViT 输出投影到 LLM 的嵌入空间（通常为一个线性层）。
- **输入**：ViT 输出 `[num_patches, embd]`。
- **输出**：最终图像嵌入 `[num_patches, llm_embd]`。
- **对齐要点**：线性层的权重矩阵是否与 llama.cpp 一致（可在 GGUF 的 `mmproj` 中查找）。

---

### 阶段 7: 占位符替换
- **任务**：将图像嵌入替换到文本序列中的 `<|image|>` 占位符位置。
- **输入**：文本 token 序列 + 图像嵌入 `[num_patches, llm_embd]`。
- **输出**：最终输入序列（嵌入替换后）。
- **对齐要点**：
  - 占位符位置：在用户消息开头，紧贴文本（无额外换行）
  - 确保 `media_count == n_media_tokens`（784 个嵌入对应 Gemma 4 的 896x896 图像）

---

## 🔧 配置管理（`config.zig` 示例）

```zig
pub const ImageConfig = struct {
    // 模型固定参数
    image_size: usize = 896,
    patch_size: usize = 14,
    in_channels: usize = 3,
    embd_dim: usize = 1152,
    llm_embd: usize = 1536,  // Gemma 4 的嵌入维度
    num_layers: usize = 27,
    num_heads: usize = 16,
    mlp_ratio: usize = 4,

    // 预处理参数
    mean: [3]f32 = .{0.5, 0.5, 0.5},
    std: [3]f32 = .{0.5, 0.5, 0.5},
    resize_algorithm: ResizeAlgorithm = .bicubic,

    // 特殊标记
    image_token: []const u8 = "<|image|>",
    vision_start: []const u8 = "<|vision_start|>",
    vision_end: []const u8 = "<|vision_end|>",
};
```

---

## 🧪 关键对齐检查清单

- [ ] **预处理**：
  - 图像尺寸调整算法（双线性/双三次）是否一致？
  - 归一化参数（mean/std）是否匹配？
  - 通道顺序（RGB/BGR）是否正确？

- [ ] **Patch Embedding**：
  - Patch 大小和步长是否匹配（14×14，步长 14）？
  - 卷积核权重是否来自 `mmproj` 文件？

- [ ] **ViT 编码器**：
  - 层数、头数、MLP 比例是否一致？
  - LayerNorm/RMSNorm eps 是否为 1e-6？
  - 激活函数（GELU）是否实现正确？

- [ ] **位置编码**：
  - 是否与 llama.cpp 的初始化方式相同（sin/cos 或可学习）？

- [ ] **占位符位置**：
  - 是否在用户消息开头，无额外换行？
  - 替换后 `media_count == n_media_tokens`（784）？

---

## 🚀 性能与调试建议

- **Arena 分配**：整个流水线在 `ArenaAllocator` 中运行，一次释放。
- **中间结果持久化**：提供 `dumpStage` 函数，可保存任意阶段的张量为 `.npy` 或 `.raw`。
- **日志分级**：关键阶段添加 `log.debug`，默认关闭。
- **Golden 测试**：将 llama.cpp 各阶段的输出保存为 golden 文件，CI 中自动对比余弦相似度。

---

## 💡 与现有 `mtmd/vision.zig` 的集成

当前 `src/mtmd/vision.zig` 可能已包含部分实现。建议逐步重构为上述模块化结构，保持对外接口不变，内部逐步替换。可以先用 `--no-chat-template` 和简单图像进行测试，逐阶段验证输出。
