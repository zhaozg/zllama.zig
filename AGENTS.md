# zllama.zig AI 编程入口

> 本文档指导 AI 理解并开发 **多模型本地推理引擎**（Zig 0.16.0 + ggml）。
> 请严格遵循以下目标、约束及设计决策。

## 🎯 项目目标

实现**生产级、跨平台**的本地推理引擎，支持 **Qwen 3.5 / LLaMA 3 / gemma 4** 等多模型架构：

- 完全基于 **Zig 0.16.0** 与 **ggml**（C 库），单二进制，无 Python/C++ 运行时。
- 支持 **GGUF v2/v3**，解析混合架构（全注意力 + 线性注意力层）及量化张量。
- **增量解码 & KV Cache**，长上下文（≥32K）内存友好。
- 支持 **多后端**：CPU（默认）、Metal (macOS)、CUDA (Linux)，GPU 后端可选。
- **多模型架构**：通过 GGUF 元数据中 `general.architecture` 自动检测模型类型。
- 代码实现需参考 [llama.cpp](deps/llama.cpp)，确保关键逻辑一致。

## ⚠️ 核心约束

### 1. Zig 0.16.0 & I/O
- 使用 **Juicy Main**：`pub fn main(init: std.process.Init) !void { const io = init.io; ... }`
- 所有 I/O 通过 `std.Io` 实例完成，函数签名显式传递 `io: std.Io`（类似 Allocator 模式）。
- 不再使用 `std.fs.cwd()`；改用 `std.Io.Dir.cwd().openFile(io, ...)`。
- `build.zig` 用 `b.addStaticLibrary` / `addCSourceFiles` 集成 ggml，用 `root_module.addCMacro` 传编译宏。
- 时间用 `std.Io.Clock.now(.awake, io)`，不用 `std.time.Timer`。

### 2. ggml 集成
- **静态编译 ggml 源码**，避免预编译库 ABI 不一致。
- 所有 ggml C API 必须通过 `src/ggml/` 模块封装（类型安全），`c` 命名空间保留供直接调用。
- 分配类操作返回 `!*T`，纯计算返回 `*T`。不透明指针用 `opaque {}`。
- 发现 llama.cpp 使用但未绑定的 ggml 函数，立即在 ggml.zig 中绑定并加测试。
- 参考：`docs/GGML_BINDING.md`

### 3. 模型架构
- 从 GGUF `general.architecture` 自动检测模型类型。
- **Architecture 枚举** 定义在 `src/model.zig`，新增模型需在此加枚举值。
- 文本模型实现在 `src/models/`，共享算子在 `src/layers/`。
- 多模态（视频/音频）实现在 `src/mtmd/` 下。
- 在 `src/models/registry.zig` 的 `createModel` / `detectArchitecture` 中注册新模型。
- 使用 `ModelInstance`（虚表模式）实现运行时多态。

### 4. 内存与性能
- 通过 ggml 加载 gguf，避免模型全量驻留物理内存。
- KV Cache 预分配固定大小张量，增量写入通过 `ggml_view_*` 切片，**禁止每 token 复制历史缓存**。
- 默认启用 `ggml_graph_plan` + 线程池（物理核心数 2/3~3/4）。
- 分配音频/图像 buffer 时使用对齐分配（如 `alignedAlloc`），确保 f32↔float、c_int↔i32。

### 5. 代码风格
- 文件不超过 600 行，接口与实现分离。
- 资源释放用 `defer`，分配失败返回错误。
- 优先代码清晰，必要时牺牲微小性能提升。

### 6. 设计哲学

面向 AI 编程助手与贡献者，定义基于 **编译期强类型（comptime）**、**显式分配（Allocator）** 与 **VTable 接口** 构建的"白盒插件系统"的顶层设计约束。所有代码修改与功能扩展必须遵守以下契约。

#### 一、 核心设计哲学：显式优于隐式，契约重于继承

本项目不使用运行时类型信息（RTTI）或虚函数表继承。我们将 C++ 的"黑盒多态"转化为 Zig 的"白盒组合"。

1.  **接口即结构体（VTable as Struct）**：系统存在两大 VTable 家族：
    - **文本 LLM**：`ModelVTable`（`src/model.zig`），通过 `ModelInstance` 包装 `ptr: *anyopaque` + `vtable: *const ModelVTable` 实现运行时多态。每个模型（llama、qwen35、gemma4 等）在 `src/models/` 下提供自己的 vtable 实例。
    - **多模态编码器**：`VisionEncoderBackend` / `AudioEncoderBackend`（`src/mtmd/graph/mod.zig`），由包含函数指针的结构体承载。各模型后端在 `src/mtmd/graph/models/` 下导出 `.backend` 单例。
    禁止使用 `@ptrCast` 进行不安全的向下转型；如需扩展能力，应在 VTable 中显式新增函数指针字段。

2.  **数据与行为分离**：权重（`Weights`）、超参数（`HParams`）与图构建逻辑严格分离。
    - 文本侧：`GraphBuilder`（`src/core/graph_builder.zig`）持有 `ctx: *ggml.Context`、`graph: *ggml.CGraph`、`params` 和 `allocator`，通过 `buildRmsNorm`/`buildRope`/`buildAttention`/`buildSwiGLU` 等方法构建计算图。
    - 多模态侧：`GraphBuilder`（`src/mtmd/graph/builder.zig`）持有 `weights: *const VisionEncoderWeights`、`hparams: *const VisionHParams`、`ctx0: *ggml.Context`、`gf: *ggml.CGraph`，通过 `buildNorm`/`buildAttn`/`buildFFN`/`buildVit` 等方法构建 ViT 计算图。
    两种 GraphBuilder 的唯一职责是接收输入张量，产出 `ggml_cgraph`——不得持有任何后端（Backend）引用。

3.  **编译期多态（Comptime Dispatch）**：如果某个特性（如是否支持批量编码、非因果注意力）在模型加载后永不改变，优先通过 Backend 字段（如 `supportBatch: bool`）在初始化时确定；调用处通过运行时分支 `if (backend.supportBatch)` 分发。模型 Architecture 枚举在 `src/model.zig` 定义，`fromString()` 在编译期完成 GGUF 字符串→枚举映射。

#### 二、 模块边界与依赖方向（强制约束）

为了防止架构腐化，模块间的依赖必须遵循以下**全局有向无环图（DAG）**规则。从底层（无业务依赖）到顶层（编排一切），共七层：

| 层级 | 角色 | 典型模块 | 允许依赖 | 禁止依赖 |
| :--- | :--- | :--- | :--- | :--- |
| **L0** | 平台抽象 & FFI | `std`、`src/ggml/`（c.zig, context, tensor, graph, backend, ops） | 仅标准库 | 任何业务模块 |
| **L1** | 数据格式 & 接口定义 | `src/gguf.zig`、`src/model.zig`（Architecture, ModelVTable, ModelInstance）、`src/core/memory.zig`（MemoryContext） | L0 | L3+（模型实现层以上） |
| **L2** | 共享算子 & 基础设施 | L2a 文本构建块：`src/layers/`；L2b 多模态构建块：`src/mtmd/graph/{attn,ffn,norm,patch,vit,builder,…}`；L2c 基础设施：`src/core/graph_builder.zig`、`src/kv_cache.zig`、`src/sampler.zig`；L2d 分词 & 模板：`src/tokenizer/`、`src/chat_template/` | L0, L1（仅类型/格式） | 任何具体模型实现 |
| **L3** | 模型实现 | L3a 文本：`src/models/{llama,qwen2,qwen35,qwen3vl,gemma3,gemma4,…}.zig` + `src/models/registry.zig`；L3b 多模态：`src/mtmd/graph/models/{gemma4v,gemma4a,qwen2vl,qwen3vl,…}.zig` | L2（构建块）, L1（接口类型） | 任何 `engine` 或 `backend` 模块；L3a 与 L3b 互不感知 |
| **L4** | 编码器 & 预处理 | `src/mtmd/vision/{encoder,…}`、`src/mtmd/audio/{encoder,pipeline,…}`、`src/mtmd/preprocess.zig`、`src/mtmd/tokenize.zig` | L3b（多模态 Backend 注册）, L2 | L6（引擎层） |
| **L5** | 多模态门面 | `src/mtmd/mod.zig`（MultiModalManager, MtmdContext）、`src/mtmd/helper.zig` | L4, L3, L2 | `InferenceEngine` 结构体 |
| **L6** | 执行引擎 | `src/core/engine_common.zig`（computeGraph）、`src/core/graph_context.zig`（IncContext）、`src/core/decode.zig`、`src/core/prefill.zig`、`src/core/multimodal.zig` | L2, L1, L0（仅 ggml C 绑定） | 任何 L3+ 的模型实现细节、`helper.zig` 的回调 |
| **L7** | 应用入口 | `src/core/engine.zig`（InferenceEngine）、`src/main.zig`、`src/cli_args.zig` | L6, L5, L3a, L2, L1 | —（顶层，无限制） |

**重要铁律（Golden Rule）**：

- **L3 不碰 L6**：L3 层（模型图构建，含 L3a 文本模型与 L3b 多模态图）**永远不能调用** L6 层（执行引擎）的任何函数。图构建只负责"画图纸"（产出 `ggml_cgraph`），不负责"盖楼"（执行 `computeGraph`/`Gallocr` 分配）。

- **文本 ↔ 多模态 隔离**：L3a（`src/models/` 文本 LLM 模型）与 L3b（`src/mtmd/graph/models/` 多模态编码器图）是两个**完全独立的世界**，互不 import。二者仅在以下两个交汇点发生协作：
  - `src/core/multimodal.zig`：三阶段 prefill 中，先调 L3b 的视觉/音频编码产出 embedding，再调 L3a 的 text model 完成带 embedding override 的前向。
  - `src/mtmd/mod.zig`：`MultiModalManager` 持有 `VisionEncoder`/`AudioEncoder`（L4），被 `InferenceEngine`（L7）用作"外挂"。

- **L2 不感知模型**：L2 层（构建块库，含 `src/layers/` 和 `src/mtmd/graph/{attn,ffn,…}`）仅操作 `ggml` 张量，不 import 任何具体模型的 `.zig` 文件。

**多模态模块的内部层次（L2b → L3b → L4 → L5）**：

MTMD（多模态）模块在 L2~L5 之间形成自身的内聚子图：

```
L2b: src/mtmd/graph/{attn,ffn,norm,patch,rope,merge,stack,mm,clamp,vit,builder,types,debug}.zig
        ↓ (仅依赖 L0 ggml + L1 类型)
L3b: src/mtmd/graph/models/{gemma4v,gemma4a,gemma4uv,qwen2vl,qwen3vl}.zig
        ↓ (每个模型导出 .backend 单例: VisionEncoderBackend / AudioEncoderBackend)
L4:  src/mtmd/vision/encoder.zig  (VisionEncoder, 通过 Backend 分发)
     src/mtmd/audio/encoder.zig   (AudioEncoder, 通过 Backend 分发)
     src/mtmd/preprocess.zig      (图像归一化等)
        ↓
L5:  src/mtmd/mod.zig             (MultiModalManager: 根据 GGUF key 选择 Backend)
     src/mtmd/helper.zig          (evalChunks, imageGetDecoderPos)
```

新增视觉/音频模型时，只需在 L3b 新增文件并导出 `.backend`，再在 L4 的 `getBackend()` / `registered_backends` 中注册，L5 的 `detectFromGGUF()` 添加 key 匹配——无需触碰 L6/L7。

#### 三、 内存生命周期契约（Critical Memory Contract）

Zig 以显式内存管理著称，本项目对张量数据的生命周期有极其严格的约定，这是保证运行时稳定的基石。

1.  **所有权归属**：
    - **权重内存（`ctx_weights`）**：由 `VisionEncoder.init()` / `AudioEncoder.init()` 在初始化时加载，并**借出**（Borrow）给各编码器的 `weights` 字段。所有权永不转移。
    - **计算图元数据（`ggml_context`）**：LLM 推理侧由 `InferenceEngine` 持有 `ctx_graph`；增量解码由 `IncContext`（`src/core/graph_context.zig`）管理 `ctx_inc`，通过 `reset()` 回收内存。
    - **图分配器（`Gallocr`）与张量数据**：严格遵循 **"延迟释放（Leak-to-Exit）"** 策略。`engine_common.computeGraph()` 内部创建的 `Gallocr` 和 CPU backend 在 compute 完成后**故意不释放**——一旦释放，`tensor.data` 指针即悬空。这些分配器驻留内存直至进程退出。

2.  **临时张量分配**：在 `GraphBuilder` 中构造临时中间张量（如位置编码、Mask），必须使用传入的 `ctx`（计算上下文）分配。这些张量的生命周期仅局限于 `ggml_cgraph` 执行期间，无需手动释放。

3.  **跨后端数据搬运**：当前主要使用 CPU 后端。当启用 GPU 后端时，输入张量（通常驻留 CPU）必须通过 `ggml.setInput()` 标记为 CPU 输入，Gallocr 的 `reserve()`/`allocGraph()` 会自动安排跨设备拷贝。

#### 四、 VTable/Backend 扩展契约（如何安全添加新模型）

当需要为系统接入新模型时，必须遵循以下扩展路径，严禁随意修改核心枚举或全局状态：

**A. 新增文本 LLM 架构：**
1. 在 `src/model.zig` 的 `Architecture` 枚举中添加新成员，并在 `fromString()` 中添加 GGUF 名称映射。
2. 在 `src/models/` 下创建新文件，实现 `ModelVTable` 的所有必需函数指针（`getParams`、`buildGraph`、`deinit`），导出 `pub const vtable = ModelVTable{...}`。
3. 在 `src/models/registry.zig` 的 `createModel()` 中为新架构添加分支，返回 `ModelInstance`。

**B. 新增视觉编码器后端：**
1. 在 `src/mtmd/graph/models/` 下创建新文件（如 `siglip.zig`），实现 `pub fn buildGraph(ctx, gf, w, p, image_tensor)` 等函数。
2. 导出 `pub const backend = VisionEncoderBackend{ .name = "siglip", .loadParams = ..., .loadWeights = ..., .buildGraph = buildGraph, ... }`。
3. 在 `src/mtmd/graph/mod.zig` 的 `model_graphs` 中注册。
4. 在 `src/mtmd/vision/mod.zig` 的 `registered_backends` 和 `getBackend()` 中注册。
5. 在 `src/mtmd/mod.zig` 的 `MultiModalManager.detectFromGGUF()` 中添加 GGUF key 匹配逻辑。

**C. 关键约束：**
- **严禁**在 `VisionEncoder` 结构体中添加"模型专属"的布尔标志（如 `is_qwen3vl`）。模型差异必须由 Backend 内部的实现逻辑消化，或通过 `VisionHParams` / `BuildVitOpts` 参数化。
- **严禁**在 `helper.zig` 的 `evalChunks` 中使用 `@typeOf` 或 `switch` 硬编码特定模型的 M-RoPE 位置计算。M-RoPE 逻辑通过 `PosType` 枚举（定义在 `src/mtmd/mod.zig`）分发到 `imageGetDecoderPos()` 中统一处理。

#### 五、 零成本抽象与编译期约束（Comptime Gates）

为了保持高性能，我们将运行时决策最小化。Agent 在处理逻辑分支时必须优先考虑编译期求值。

1.  **特性标记（Feature Flags）**：`supportBatch` 等特性定义在 `VisionEncoderBackend` 或 `AudioEncoderBackend` 结构体的 `bool` 字段中，默认值为 `false`。调用处通过运行时 `if` 分支（编译器可能优化为常量）进行分发。对于在模型加载后即确定的路径，优先使用 Backend 字段而非运行时类型检查。

2.  **类型安全的超参数**：所有从 GGUF 解析的超参数（如图像尺寸、Patch 大小、n_merge）必须在读取时立即存入强类型的 `VisionHParams` 结构体（`src/mtmd/graph/types.zig`）中，或存入 `VisionEncoderParams`（`src/mtmd/vision/config.zig`）。禁止在代码逻辑中使用裸 `f32` 或 `usize` 变量进行关键维度计算，除非它们明确源自这些结构体。

3.  **调试代码的隔离**：图构建过程中的调试张量注册通过 `DebugTensorRegistry`（`src/mtmd/graph/debug.zig`）实现，仅在显式启用时生效。`setName` 调用在 `ReleaseFast` 模式下应尽可能消除。

#### 六、 数值对齐与回归防御（Correctness Gate）

本项目是 `llama.cpp` 的 Zig 端口，数值精度是生命线。任何改变计算图拓扑的修改都必须遵循以下验证准则：

1.  **中间层可观测性**：`DebugTensorRegistry` 提供可选的张量捕获钩子，允许 `compare_mtmd_vision.zig` / `compare_mtmd_audio.zig` 等工具捕获特定中间层张量数据（如 Pre-Norm 输出）进行逐层对比。

2.  **随机权重测试**：在无真实模型权重（GGUF）的 CI 环境中，Agent 设计的单元测试应具备生成"随机但确定"（固定种子）权重并执行前向传播的能力。输出张量的形状和数值范围（而非具体数值）应作为回归测试的基准。

3.  **熔断机制（Fuse Breaker）**：如果在实现优化（如算子融合）时发现与 C++ 参考实现存在 NMSE > 1e-5 的偏差，**必须放弃该优化**，优先保证数值一致性。性能优化只能在数值对齐验证通过后进行。

## 七、 Agent 工作流决策矩阵

当你被要求修改或扩展本系统时，请依据以下决策树快速定位设计边界：

| 需求场景 | 设计落脚点 | 禁止行为 |
| :--- | :--- | :--- |
| 新增一种文本模型架构 | `src/models/` 新建文件 + `src/model.zig` 加枚举 + `src/models/registry.zig` 注册 | 修改 `InferenceEngine` 或 `decode.zig` 的主循环 |
| 新增一种视觉/音频编码器 | `src/mtmd/graph/models/` 新建文件 + `src/mtmd/graph/mod.zig` 注册 Backend + `src/mtmd/vision/mod.zig`（或 `audio/mod.zig`）注册 `getBackend` | 修改 `InferenceEngine` 或 `helper.zig` 的主循环 |
| 新增一种预处理方式（如 Foveated） | `src/mtmd/preprocess.zig` 新增函数，在 `src/mtmd/tokenize.zig` 或 `src/mtmd/vision/preprocess.zig` 中调用 | 将预处理代码混入 `graph/` 层 |
| 优化 CPU 计算性能（如算子融合） | 仅修改 `src/mtmd/graph/` 或 `src/layers/` 下的构建块 | 触碰 `ggml/backend.zig` 或内存释放逻辑 |
| 支持新的硬件后端（如 Vulkan） | 扩展 `src/ggml/backend.zig` 的 `DeviceType` 枚举，实现 `detectBestBackend` | 在业务层（`helper.zig` 或 `engine.zig`）添加硬件判断分支 |
| 解决 Logits 精度不匹配 | 追踪 `buildGraph` 中缺失的 `scale`/`bias`/`norm` 操作，对比 llama.cpp 参考实现 | 盲目调整 `ggml` 绑定或后端 buffer 大小 |
| 新增 ggml 算子 | 在 `src/ggml/ops.zig` 封装，在 `src/ggml/mod.zig` 重新导出 | 业务代码直接 `@cImport` 调用裸 C 函数 |

本系统的核心优势在于"确定性"——无论图结构多复杂，只要遵守上述契约，内存一定安全，类型一定匹配，调度一定有序。当你感到困惑时，请回到宪章的第一条：**显式优于隐式，契约重于继承。** 。

## 📦 Import 规则

所有模块通过 `build.zig` 的 `createModule`/`addImport` 注册，**必须使用模块名导入**。

| 模块名 | 根文件 |
|--------|--------|
| `ggml` | `src/ggml/mod.zig` |
| `gguf` | `src/gguf.zig` |
| `model` | `src/model.zig` |
| `registry` | `src/models/registry.zig` |
| `graph_builder` | `src/core/graph_builder.zig` |
| `memory` | `src/core/memory.zig` |
| `tokenizer` | `src/tokenizer/mod.zig` |
| `sampler` | `src/sampler.zig` |
| `kv_cache` | `src/kv_cache.zig` |
| `chat_template` | `src/chat_template/mod.zig` |

**规则**：跨模块引用必须用 `@import("模块名")`。同一模块内的子文件（如 `models/llama.zig` 引用 `model.zig`）可使用相对路径 `@import("../model.zig")`。模块子文件通过根文件的 `pub const` 重新导出。

## 🔒 禁止事项

- ❌ 业务代码直接 `@cImport` 或调用裸 C 函数。
- ❌ 忽略 `std.Io` 接口，使用 `std.fs.cwd()` 等阻塞 API。
- ❌ 硬编码模型超参数（必须从 GGUF 元数据读取）。
- ❌ KV Cache 物理复制；使用 `@alignCast` 假设对齐。
- ❌ 删除功能代码绕开问题；跳过错误处理。
- ❌ 参数硬编码而不从权重实际形状获取。
- ❌ 直接交互式使用 `llama-cli`（导致 hang）；应通过 `echo /exit | llama-cli ...`。
- ❌ 编译安装 llama.cpp（它已安装在 `/usr/local`）。

## 🤖 AI 工作流程

1. **需求理解**：阅读 `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/GGML_BINDING.md`。
2. **编码**：
   - ggml C 调用必须通过 `src/ggml/` 模块封装。
   - 新增算子先在 ggml.zig 中封装，再在业务中使用。
   - 新增模型在 `src/models/` 创建文件 → 在 `model.zig` 加枚举 → 在 `registry.zig` 注册。
3. **验证**：用 `compare_logits` 对比输出，确保 NMSE < 1e-5 或余弦相似度 > 0.999。完成 `forward()` 后立即验证 logits 非全零/NaN。
4. **提交前**：
   - `zig fmt src`
   - `zig build test -Doptimize=ReleaseSafe --summary all` 全部通过
   - `zig-out/bin/zllama -n 5 --model <model.gguf>` 对 tinyllama/Llama-3.2-3B/gemma-4-E2B/Qwen3.5-4B 均正常推理
5. **文档同步**：修改架构或绑定设计后，同步更新对应 `*.md`。

## 📁 项目结构

```
zllama.zig/
├── AGENTS.md                   # 本文件
├── TODO.md                     # 待办与调试指南
├── build.zig / build.zig.zon   # Zig 构建
├── docs/                       # 设计文档
│   ├── ARCHITECTURE.md         # 系统架构
│   ├── GGML_BINDING.md         # ggml.zig 绑定设计
│   ├── FILE_SRUCTURE.md        # 文件组织约定
│   ├── TECHNICAL_CHALLENGES.md # 技术难点
│   ├── TEST.md                 # 测试体系
│   └── HOW_TO_ADD_NEW_MODEL.md # 新增模型指南
├── src/
│   ├── main.zig                # 入口（Juicy Main）
│   ├── model.zig               # Architecture 枚举、ModelVTable 接口
│   ├── gguf.zig                # GGUF 解析
│   ├── kv_cache.zig / sampler.zig / utils.zig
│   ├── core/                   # 引擎、图构建、内存、加载器
│   ├── ggml/                   # ggml C API 绑定
│   ├── layers/                 # 共享算子（attention, rope, rms_norm 等）
│   ├── models/                 # 模型实现 + registry.zig
│   ├── mtmd/                   # 多模态（音频、视觉、预处理）
│   ├── tokenizer/              # BPE 分词器
│   ├── chat_template/          # 对话模板（含 Jinja）
│   ├── vendor/                 # 第三方（stb, minja）
│   ├── tests/                  # 单元测试
│   └── tools/                  # 辅助工具（compare_logits 等）
├── deps/
│   ├── llama.cpp/              # 参考实现（只读）
│   └── ggml/                   # ggml C 源码（submodule）
└── tools/                      # 外部辅助脚本
```

## 🧪 增量开发策略

- **按阶段**：超参数加载 → 张量加载 → 全注意力前向 → 线性注意力前向 → 完整推理循环，每阶段独立验证。
- **参考对齐**：用 `compare_logits.zig` 对比第一个 token logits 与 llama.cpp 参考值。
- **交叉验证**：关键路径（`forward()`、KV Cache）用注释标明 llama.cpp 对应文件和行号。
- **已知差异**：在代码注释或 `docs/` 中记录与参考实现的故意差异及原因。

## 📚 参考材料

- [ggml 源码](deps/ggml/)
- [llama.cpp 模型实现](deps/llama.cpp/src/models/)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Zig 0.16.0 文档](https://ziglang.org/documentation/0.16.0/)

---

**AI 助手应始终以"安全、可维护、高性能"为原则，优先遵循本文档约束。如有歧义，请在对话中提问澄清。**
