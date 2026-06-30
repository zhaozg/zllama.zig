# mtmd 计算图架构设计

> 本文档定义 `src/mtmd/` 下计算图（computation graph）模块的 Zig 架构设计。
> 参考: `deps/llama.cpp/tools/mtmd/clip-graph.h`、`clip-model.h`、`clip-impl.h`

## 1. 设计目标

1. **结构清晰**：将 ViT 图构建拆分为可组合的构建块（build block），每个块对应一个明确的 ggml 子图。
2. **易调试**：每个构建函数有明确的输入/输出张量形状断言，支持 `std.log` 级别的调试输出。
3. **易维护**：新增模型只需在 `registry.zig` 注册新的 `GraphBuilder` 实现，无需修改核心图构建逻辑。
4. **类型安全**：利用 Zig 的 comptime 和联合类型，在编译期捕获图构建错误。
5. **零成本抽象**：运行时多态通过 `switch` 分发，无虚表开销。

## 2. 核心架构

### 2.1 层次结构

```
src/mtmd/graph/
├── mod.zig              # 模块根，导出所有公共类型
├── builder.zig          # GraphBuilder 接口定义 + 工厂函数
├── vit.zig              # 通用 ViT 图构建（build_vit）
├── attn.zig             # 注意力层构建（build_attn）
├── ffn.zig              # FFN 层构建（build_ffn）
├── norm.zig             # 归一化层构建（build_norm）
├── patch.zig            # Patch embedding 构建（build_inp, build_inp_raw）
├── rope.zig             # 2D RoPE 构建（build_rope_2d）
├── merge.zig            # Patch merge / pixel shuffle 构建
├── stack.zig            # Frame stacking 构建（音频）
├── mm.zig               # 多模态投影器构建（build_mm）
├── debug.zig            # 调试支持（DebugTensorRegistry）
└── types.zig            # 共享类型定义（norm_type, ffn_op_type 等）
```

### 2.2 与 clip-graph.h 的对应关系

| clip-graph.h 成员 | Zig 模块 | 说明 |
|---|---|---|
| `clip_graph` 类 | `builder.zig` 中的 `GraphBuilder` 接口 | 核心图构建器接口 |
| `build_vit()` | `vit.zig` 中的 `buildVit()` | 通用 ViT 图构建 |
| `build_inp()` | `patch.zig` 中的 `buildInp()` | Conv2D patch embedding |
| `build_inp_raw()` | `patch.zig` 中的 `buildInpRaw()` | 原始输入处理 |
| `build_norm()` | `norm.zig` 中的 `buildNorm()` | LayerNorm / RMSNorm |
| `build_ffn()` | `ffn.zig` 中的 `buildFFN()` | FFN (GELU/SiLU/ReLU²) |
| `build_attn()` | `attn.zig` 中的 `buildAttn()` | 多头注意力 |
| `build_rope_2d()` | `rope.zig` 中的 `buildRope2D()` | 2D RoPE |
| `build_patch_merge_permute()` | `merge.zig` 中的 `buildPatchMergePermute()` | Patch merge |
| `build_stack()` | `stack.zig` 中的 `buildStack()` | Frame stacking |
| `build_mm()` | `mm.zig` 中的 `buildMM()` | 矩阵乘法封装 |
| `build_vit_opts` | `types.zig` 中的 `BuildVitOpts` | ViT 构建选项 |
| `resize_position_embeddings()` | `vit.zig` 中的 `resizePositionEmbeddings()` | 位置嵌入插值 |

## 3. 核心类型定义

### 3.1 枚举类型（types.zig）

```zig
pub const FFNOpType = enum { gelu, gelu_erf, silu, gelu_quick, relu_sqr };
pub const NormType = enum { layer_norm, rms_norm };
pub const PatchMergeType = enum { flat, spatial_unpad };
pub const ProjectorType = enum(u16) { mlp, mlp_norm, ldp, ..., unknown };
pub const Modality = enum(u8) { vision, audio };
pub const FlashAttnType = enum(i8) { auto = -1, disabled = 0, enabled = 1 };
pub const ResizeAlgo = enum(u8) { bilinear, bicubic, bicubic_pillow };
pub const PadStyle = enum(u8) { none, ceil, nearest };
```

### 3.2 构建选项（types.zig）

```zig
pub const BuildVitOpts = struct {
    attn_mask: ?*ggml.Tensor = null,
};
```

### 3.3 GraphBuilder 接口（builder.zig）

```zig
pub const GraphBuilder = struct {
    weights: *const VisionEncoderWeights,
    hparams: *const VisionHParams,
    proj_type: ProjectorType,
    img: *const ImageF32,
    ctx0: *ggml.Context,
    gf: *ggml.CGraph,
    n_batch: u32 = 1,
    flash_attn_type: FlashAttnType = .disabled,
    debug_registry: DebugTensorRegistry = .{},

    pub fn build(self: *GraphBuilder) !*ggml.CGraph;
    pub fn buildMM(self: *const GraphBuilder, w: *ggml.Tensor, x: *ggml.Tensor) !*ggml.Tensor;
    pub fn supportBatch(self: *const GraphBuilder) bool;
    pub fn buildVit(...) !*ggml.Tensor;
    pub fn buildInp(...) !*ggml.Tensor;
    pub fn buildNorm(...) !*ggml.Tensor;
    pub fn buildFFN(...) !*ggml.Tensor;
    pub fn buildAttn(...) !*ggml.Tensor;
    pub fn buildRope2D(...) !*ggml.Tensor;
    pub fn buildPatchMergePermute(...) !*ggml.Tensor;
    pub fn buildStack(...) !*ggml.Tensor;
    pub fn buildGemma3Projector(...) !*ggml.Tensor;
    pub fn buildStandardizeAndProject(...) !*ggml.Tensor;
    pub fn createPositionIndices(...) !struct { pos_x, pos_y };
    pub fn debugRegisterTensor(...) !void;
    pub fn debugSaveAll(...) !void;
    pub fn debugSaveTensorByName(...) !void;
};
```

### 3.5 AudioEncoderBackend 接口（graph/mod.zig）

音频编码器后端接口，用于将音频编码器的模型特定实现（权重加载、图构建、token 估算）从框架代码中解耦。

```zig
pub const AudioEncoderBackend = struct {
    name: []const u8,
    loadParams: *const fn (gguf_file: *const gguf.GGUFFile, params: *VisionHParams) void,
    loadWeights: *const fn (allocator: std.mem.Allocator, gguf_file: *const gguf.GGUFFile, ctx: *ggml.Context, w: *VisionEncoderWeights) anyerror!void,
    loadClampInfo: *const fn (allocator: std.mem.Allocator, gguf_file: *const gguf.GGUFFile, w: *VisionEncoderWeights) anyerror!void,
    buildGraph: *const fn (ctx: *ggml.Context, gf: *ggml.CGraph, w: *const VisionEncoderWeights, p: *const VisionHParams, mel_tensor: *ggml.Tensor, clamp_map: *const std.StringHashMap(ClampInfo)) anyerror!*ggml.CGraph,
    estimateOutputTokens: *const fn (n_frames: u32) u32,
};
```

该接口定义在 `graph/mod.zig` 中，由 `audio/encoder.zig` 重新导出。模型特定实现（如 `gemma4a.zig`）通过导出 `pub const backend = graph.AudioEncoderBackend{ ... }` 来注册。

`audio/encoder.zig` 中的 `AudioEncoder` 使用 `AudioEncoderParams`（来自 `config.zig`）作为内部参数类型，在调用 `backend.buildGraph` 时构建临时的 `VisionHParams` 传递给后端。

### 3.4 权重结构（types.zig）

完整定义了 `ViTLayerWeights`、`VisionEncoderWeights`、`MobileNetV5Block`、`YASA2Block`、`YASA2Stage`、`QFormerBlock`、`ClampInfo` 等结构，与 `clip-model.h` 一一对应。

## 4. 构建函数签名

所有构建函数签名与 `clip-graph.h` 对应，详见各模块文件。

## 5. 模型注册与分发

模型特定实现在 `src/mtmd/graph/models/` 下，每个文件实现一个模型的完整图构建。

## 6. 与现有代码的集成

### 6.1 重构路径

1. **第一阶段**：创建 `src/mtmd/graph/` 目录和类型定义 ✅
2. **第二阶段**：实现通用构建块（norm.zig、ffn.zig、attn.zig、patch.zig）✅
3. **第三阶段**：实现 ViT 主干（vit.zig、rope.zig、merge.zig）✅
4. **第四阶段**：实现投影器构建（mm.zig、stack.zig）✅
5. **第五阶段**：实现模型特定构建器（models/gemma4v.zig, gemma4a.zig）✅
6. **第六阶段**：build.zig 集成 + mtmd/mod.zig 集成 ✅
7. **第七阶段**：旧代码兼容层（vision/types.zig 重定向到 graph/types.zig）✅

### 6.2 现有文件调整

| 现有文件 | 调整方式 |
|---|---|
| `src/mtmd/vision/types.zig` | ✅ 已改为兼容层，重新导出 graph/types.zig 的类型 |
| `src/mtmd/vision/encoder.zig` | ✅ 已重构，applyViTBlocks 使用 graph.buildNorm/buildFFN/buildAttn |
| `src/mtmd/audio/encoder.zig` | ✅ 已重构，buildNorm/buildFFN/buildMM 使用 graph 模块实现 |
| `src/mtmd/vision/config.zig` | 保留，作为超参数加载模块 |
| `src/mtmd/vision/loader.zig` | 保留，作为权重加载模块 |
| `src/mtmd/vision/preprocess.zig` | 保留，作为预处理模块 |
| `src/mtmd/vision/postprocess.zig` | 保留，作为后处理模块 |
| `src/mtmd/vision/pipeline.zig` | 保留独立实现 |
| `build.zig` | ✅ 已添加 graph 模块注册（在 audio/vision 模块之前） |
| `src/mtmd/mod.zig` | ✅ 已添加 graph 模块导入 |

## 7. 调试支持

- 张量形状断言
- 调试命名（`setName()`）
- 中间张量数据保存（`mtmdDebugSaveTensor()`）
- 实现:
  1. ✅ `src/mtmd/graph/debug.zig` — `DebugTensorRegistry` 模块，管理调试张量注册表
     - `register()`: 注册张量，设置调试名称，标记 input/output
     - `saveAll()`: 将所有注册张量保存为 JSON 数组文件
     - `saveTensor()`: 保存单个张量数据
     - `saveTensorByName()`: 通过名称在图中查找张量并保存
  2. ✅ `src/ggml/graph.zig` — `CGraph.getTensor()`: 通过名称在图中查找张量
  3. ✅ `src/mtmd/graph/builder.zig` — `GraphBuilder` 集成调试接口
     - `debugRegisterTensor()`: 注册张量调试
     - `debugSaveAll()`: 保存所有调试张量
     - `debugSaveTensorByName()`: 按名称保存单个张量
  4. ✅ `src/mtmd/graph/mod.zig` — 导出 `debug` 模块和 `DebugTensorRegistry`

### 7.1 使用示例

```zig
// 在模型构建器中注册调试张量
const allocator = ...;
try builder.debugRegisterTensor(allocator, cur, "debug_audio_encoder_input", false);

// 构建完成后保存所有调试数据
try builder.debugSaveAll(allocator, "/tmp/debug", "llama_audio");

// 或按名称保存单个张量
try builder.debugSaveTensorByName(allocator, "debug_audio_encoder_input", "/tmp/debug", "encoder_input.json");
```

### 7.2 文件格式

保存的 JSON 数组文件格式（与 llama.cpp `mtmd_debug_save_data` 兼容）:
```json
[
1.234567,
2.345678,
...
9.876543
]
```

## 8. 测试策略

- 单元测试：每个构建块模块包含独立测试
- 集成测试：使用小型模型验证完整图构建
- 数值对比测试：与 llama.cpp 参考实现对比 logits

## 9. 文件结构总览

```
src/mtmd/
├── graph.md                    # 本文档
├── mod.zig                     # mtmd 模块根
├── graph/                      # 计算图构建
│   ├── mod.zig                 # 模块根
│   ├── builder.zig             # GraphBuilder 接口
│   ├── types.zig               # 共享类型定义
│   ├── vit.zig                 # 通用 ViT 构建
│   ├── attn.zig                # 注意力层构建
│   ├── ffn.zig                 # FFN 层构建
│   ├── norm.zig                # 归一化层构建
│   ├── patch.zig               # Patch embedding 构建
│   ├── rope.zig                # 2D RoPE 构建
│   ├── merge.zig               # Patch merge 构建
│   ├── stack.zig               # Frame stacking 构建
│   ├── mm.zig                  # 投影器构建
│   ├── debug.zig               # 调试支持
│   └── models/                 # 模型特定实现
│       ├── gemma4v.zig         # Gemma4V 视觉编码器
│       ├── gemma4a.zig         # Gemma4A 音频编码器
│       ├── gemma4uv.zig        # Gemma4UV 统一视觉编码器
│       ├── qwen2vl.zig         # Qwen2VL 视觉编码器
│       └── qwen3vl.zig         # Qwen3VL 视觉编码器
├── audio/                      # 音频处理（已有）
├── vision/                     # 视觉处理（已有）
├── helper.zig                  # 辅助函数（已有）
├── tokenize.zig                # 分词器（已有）
├── fft.zig                     # FFT 实现（已有）
└── preprocess.zig              # 预处理（已有）
```

## 10. 实现状态

### 10.1 已完成

| 模块 | 文件 | 状态 |
|---|---|---|
| 类型定义 | `src/mtmd/graph/types.zig` | ✅ 完成 |
| GraphBuilder 接口 | `src/mtmd/graph/builder.zig` | ✅ 完成 |
| 归一化层 | `src/mtmd/graph/norm.zig` | ✅ 完成（含测试） |
| FFN 层 | `src/mtmd/graph/ffn.zig` | ✅ 完成（含测试） |
| 注意力层 | `src/mtmd/graph/attn.zig` | ✅ 完成（含测试） |
| Patch embedding | `src/mtmd/graph/patch.zig` | ✅ 完成（含测试） |
| 2D RoPE | `src/mtmd/graph/rope.zig` | ✅ 完成（含测试） |
| Patch merge | `src/mtmd/graph/merge.zig` | ✅ 完成（含测试） |
| Frame stacking | `src/mtmd/graph/stack.zig` | ✅ 完成（含测试） |
| 投影器 | `src/mtmd/graph/mm.zig` | ✅ 完成（含测试） |
| 通用 ViT | `src/mtmd/graph/vit.zig` | ✅ 完成（含测试） |
| 模块根 | `src/mtmd/graph/mod.zig` | ✅ 完成 |
| 调试支持 | `src/mtmd/graph/debug.zig` | ✅ 完成（含测试） |
| Gemma4V 模型 | `src/mtmd/graph/models/gemma4v.zig` | ✅ 完成 |
| Gemma4A 模型 | `src/mtmd/graph/models/gemma4a.zig` | ✅ 完成 |
| Gemma4UV 模型 | `src/mtmd/graph/models/gemma4uv.zig` | ✅ 完成 |
| Qwen2VL 模型 | `src/mtmd/graph/models/qwen2vl.zig` | ✅ 完成 |
| Qwen3VL 模型 | `src/mtmd/graph/models/qwen3vl.zig` | ✅ 完成 |
| resizePositionEmbeddings | `src/mtmd/graph/vit.zig` | ✅ 完成 |
| Attention sinks | `src/mtmd/graph/attn.zig` | ✅ 完成 |
| build.zig 集成 | `build.zig` | ✅ 完成（graph 模块注册） |
| mtmd/mod.zig 集成 | `src/mtmd/mod.zig` | ✅ 完成（graph 模块导入） |
| vision/types.zig 兼容层 | `src/mtmd/vision/types.zig` | ✅ 完成（重定向到 graph/types.zig） |
| AudioEncoderBackend 接口 | `src/mtmd/graph/mod.zig` | ✅ 完成（定义在 graph/mod.zig，由 audio/encoder.zig 重新导出） |
| Gemma4A backend 注册 | `src/mtmd/graph/models/gemma4a.zig` | ✅ 完成（导出 `backend` 常量） |
| AudioEncoder 使用 AudioEncoderParams | `src/mtmd/audio/encoder.zig` | ✅ 完成（内部使用 AudioEncoderParams，调用 backend 时构建 VisionHParams） |
| CGraph.getTensor 绑定 | `src/ggml/graph.zig` | ✅ 完成（ggml_graph_get_tensor 绑定） |

### 10.2 待完成

| 任务 | 优先级 | 说明 |
|---|---|---|
| 集成测试 | 中 | 与 llama.cpp 参考实现对比 |
