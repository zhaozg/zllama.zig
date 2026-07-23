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

### 6. 日志作用域命名规则

所有 `std.log.scoped` 调用必须遵循**层级前缀 + 下划线 + 组件名**的命名规则，确保日志作用域名称在整个项目中唯一且可追溯。

格式：`{layer_prefix}_{component_name}`，用下划线分隔层级（Zig 枚举不支持点号）。同一模块内多个文件共享同一 scope 名。变量名统一用 `log`，冲突时用 `logger`。

| 层级前缀 | 对应模块 |
|---------|---------|
| `ggml` | L0: ggml 绑定 |
| `ggml_gguf` | L0: ggml GGUF 绑定 |
| `ggml_quantize` | L0: ggml 量化 |
| `gguf` | L1: GGUF 解析（主模块） |
| `gguf_parse` | L1: GGUF 解析子模块 |
| `model_*` | L3a: 文本模型 |
| `model_registry` | L3a: 模型注册 |
| `core_*` | L2c/L6: 基础设施/引擎 |
| `layer_*` | L2a: 共享算子 |
| `graph_*` | L2b: 多模态图构建块 |
| `graph_model_*` | L3b: 多模态模型图 |
| `vision_*` | L4: 视觉编码器 |
| `audio_*` | L4: 音频编码器 |
| `mtmd` | L5: 多模态门面（主模块） |
| `mtmd_*` | L5: 多模态门面子模块 |
| `chat_*` | L2d: 对话模板 |
| `tokenizer` | L2d: 分词器（主模块） |
| `tokenizer_vocab` | L2d: 词表 |
| `app_*` | L7: 应用入口 |
| `test_*` | 测试 |
| `tool_*` | 工具 |

唯一性、可追溯、一致性、简洁性（`graph_model_gemma4v` 已是上限）。变量名统一用 `log`，冲突时用 `logger`。

```zig
// core/engine.zig → .core_engine  (与函数名 log 冲突，用 logger)
// models/llama.zig → .model_llama
// mtmd/graph/attn.zig → .graph_attn
```

### 7. 设计哲学

面向 AI 编程助手与贡献者，定义基于 **编译期强类型（comptime）**、**显式分配（Allocator）** 与 **VTable 接口** 构建的"白盒插件系统"的顶层设计约束。

#### 一、 核心设计哲学：显式优于隐式，契约重于继承

1.  **接口即结构体（VTable as Struct）**：两大 VTable 家族——文本 LLM 的 `ModelVTable`（`src/model.zig`）与多模态编码器的 `VisionEncoderBackend`/`AudioEncoderBackend`（`src/mtmd/graph/mod.zig`）。禁止 `@ptrCast` 向下转型；扩展能力应在 VTable 中显式新增函数指针字段。

2.  **数据与行为分离**：权重（`Weights`）、超参数（`HParams`）与图构建逻辑严格分离。文本侧 `GraphBuilder`（`src/core/graph_builder.zig`）与多模态侧 `GraphBuilder`（`src/mtmd/graph/builder.zig`）的唯一职责是接收输入张量、产出 `ggml_cgraph`——不得持有任何后端引用。

3.  **编译期多态（Comptime Dispatch）**：特性标记（如 `supportBatch: bool`）在初始化时确定，运行时通过 `if (backend.supportBatch)` 分发。`Architecture.fromString()` 在编译期完成 GGUF 字符串→枚举映射。

#### 二、 模块边界与依赖方向（强制约束）

> 完整 DAG 架构图及 VTable 家族详解见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

七层 DAG（L0→L7），每层只能依赖更低编号的层：

| 层级 | 角色 | 典型模块 |
| :--- | :--- | :--- |
| **L0** | 平台抽象 & FFI | `std`、`src/ggml/`（c.zig, context, tensor, graph, backend, ops） |
| **L1** | 数据格式 & 接口定义 | `src/gguf.zig`、`src/model.zig`、`src/core/memory.zig` |
| **L2** | 共享算子 & 基础设施 | `src/layers/`、`src/mtmd/graph/{attn,ffn,norm,…}`、`src/core/graph_builder.zig`、`src/kv_cache.zig`、`src/sampler.zig`、`src/tokenizer/`、`src/chat_template/` |
| **L3** | 模型实现 | L3a 文本：`src/models/{llama,qwen2,qwen35,gemma3,gemma4,…}.zig` + `registry.zig`；L3b 多模态：`src/mtmd/graph/models/{gemma4v,gemma4a,…}.zig` |
| **L4** | 编码器 & 预处理 | `src/mtmd/vision/`、`src/mtmd/audio/`、`src/mtmd/preprocess.zig` |
| **L5** | 多模态门面 | `src/mtmd/mod.zig`（MultiModalManager）、`src/mtmd/helper.zig` |
| **L6** | 执行引擎 | `src/core/engine_common.zig`、`src/core/graph_context.zig`、`src/core/decode.zig`、`src/core/prefill.zig`、`src/core/multimodal.zig` |
| **L7** | 应用入口 | `src/core/engine.zig`（InferenceEngine）、`src/main.zig`、`src/cli_args.zig` |

**三条铁律**：
- **L3 不碰 L6**：图构建只负责"画图纸"（产出 `ggml_cgraph`），不负责"盖楼"（`computeGraph`/`Gallocr`）。
- **文本 ↔ 多模态 隔离**：L3a（`src/models/`）与 L3b（`src/mtmd/graph/models/`）互不 import，仅交汇于 `core/multimodal.zig` 和 `mtmd/mod.zig`。
- **L2 不感知模型**：构建块库仅操作 `ggml` 张量，不 import 任何具体模型 `.zig` 文件。

#### 三、 内存生命周期契约（Critical Memory Contract）

1.  **所有权归属**：
    - **权重内存（`ctx_weights`）**：由编码器 `init()` 加载并**借出**给 `weights` 字段，所有权永不转移。
    - **计算图元数据（`ggml_context`）**：LLM 推理侧由 `InferenceEngine` 持有 `ctx_graph`；增量解码由 `IncContext`（`src/core/graph_context.zig`）管理 `ctx_inc`，通过 `reset()` 回收。
    - **图分配器（`Gallocr`）与张量数据**：严格遵循 **"延迟释放（Leak-to-Exit）"** 策略——`computeGraph()` 创建的 `Gallocr` 和 CPU backend 在 compute 完成后**故意不释放**，否则 `tensor.data` 指针悬空。

2.  **临时张量分配**：在 `GraphBuilder` 中构造临时中间张量，使用传入的 `ctx` 分配，生命周期仅限 `ggml_cgraph` 执行期间。

3.  **跨后端数据搬运**：GPU 后端时，输入张量通过 `ggml.setInput()` 标记为 CPU 输入，`Gallocr` 自动安排跨设备拷贝。

#### 四、 VTable/Backend 扩展契约

> 完整新增模型指南（含代码模板）详见 [`docs/HOW_TO_ADD_NEW_MODEL.md`](docs/HOW_TO_ADD_NEW_MODEL.md)。

**A. 新增文本 LLM 架构：**
1. `src/model.zig` 的 `Architecture` 枚举加成员 + `fromString()` 加映射。
2. `src/models/` 下新建文件，实现 `ModelVTable` 函数指针，导出 `pub const vtable = ModelVTable{...}`。
3. `src/models/registry.zig` 的 `createModel()` 加分支。

**B. 新增视觉编码器后端：**
1. `src/mtmd/graph/models/` 下新建文件，实现 `buildGraph` 等函数。
2. 导出 `pub const backend = VisionEncoderBackend{...}`。
3. 在 `src/mtmd/graph/mod.zig`、`src/mtmd/vision/mod.zig`、`src/mtmd/mod.zig` 中注册。

**C. 关键约束：**
- **严禁**在 `VisionEncoder` 中添加模型专属布尔标志（如 `is_qwen3vl`）。差异由 Backend 内部消化或通过 `VisionHParams`/`BuildVitOpts` 参数化。
- **严禁**在 `helper.zig` 的 `evalChunks` 中硬编码特定模型的 M-RoPE 位置计算。通过 `PosType` 枚举分发到 `imageGetDecoderPos()`。

#### 五、 零成本抽象与编译期约束（Comptime Gates）

1.  **特性标记（Feature Flags）**：`supportBatch` 等定义在 Backend 结构体的 `bool` 字段中，运行时 `if` 分发（编译器可优化为常量）。
2.  **类型安全的超参数**：GGUF 解析的超参数立即存入强类型结构体（`VisionHParams`/`VisionEncoderParams`），禁止裸 `f32`/`usize` 进行关键维度计算。
3.  **调试代码的隔离**：`DebugTensorRegistry`（`src/mtmd/graph/debug.zig`）仅在显式启用时生效，`setName` 在 `ReleaseFast` 下应消除。

#### 六、 数值对齐与回归防御（Correctness Gate）

> 完整测试体系详见 [`docs/TEST.md`](docs/TEST.md)。

1.  **中间层可观测性**：`DebugTensorRegistry` 提供张量捕获钩子，支持逐层对比。
2.  **随机权重测试**：单元测试应具备固定种子随机权重前向传播能力，验证形状和数值范围。
3.  **熔断机制（Fuse Breaker）**：优化发现 NMSE > 1e-5 偏差时**必须放弃**，优先保证数值一致性。

## 七、 Agent 工作流决策矩阵

| 需求场景 | 设计落脚点 | 禁止行为 |
| :--- | :--- | :--- |
| 新增文本模型架构 | `src/models/` 新建 + `model.zig` 加枚举 + `registry.zig` 注册 | 修改 `InferenceEngine` 或 `decode.zig` 主循环 |
| 新增视觉/音频编码器 | `src/mtmd/graph/models/` 新建 + 注册 Backend | 修改 `InferenceEngine` 或 `helper.zig` 主循环 |
| 新增预处理方式 | `src/mtmd/preprocess.zig` 新增函数 | 将预处理混入 `graph/` 层 |
| 优化 CPU 性能 | 仅修改 `src/mtmd/graph/` 或 `src/layers/` 构建块 | 触碰 `ggml/backend.zig` 或内存释放逻辑 |
| 支持新硬件后端 | 扩展 `src/ggml/backend.zig` 的 `DeviceType` | 在业务层添加硬件判断分支 |
| 解决 Logits 精度不匹配 | 追踪 `buildGraph` 缺失操作，对比 llama.cpp | 盲目调整 ggml 绑定或后端 buffer |
| 新增 ggml 算子 | `src/ggml/ops.zig` 封装，`mod.zig` 重新导出 | 业务代码直接 `@cImport` |

本系统的核心优势在于"确定性"——**显式优于隐式，契约重于继承。**

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

**规则**：跨模块引用用 `@import("模块名")`。同一模块内子文件可用相对路径 `@import("../model.zig")`。模块子文件通过根文件 `pub const` 重新导出。

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

> 完整目录结构详见 [`docs/FILE_SRUCTURE.md`](docs/FILE_SRUCTURE.md)。

```
zllama.zig/
├── AGENTS.md                   # 本文件（AI 编程入口）
├── TODO.md                     # 待办与调试指南
├── build.zig / build.zig.zon   # Zig 构建
├── docs/                       # 设计文档
├── src/
│   ├── main.zig                # 入口（Juicy Main）
│   ├── model.zig               # Architecture 枚举、ModelVTable 接口
│   ├── gguf.zig                # GGUF 解析
│   ├── core/                   # 引擎、图构建、内存、加载器
│   ├── ggml/                   # ggml C API 绑定
│   ├── layers/                 # 共享算子
│   ├── models/                 # 模型实现 + registry.zig
│   ├── mtmd/                   # 多模态
│   ├── tokenizer/              # BPE 分词器
│   ├── chat_template/          # 对话模板
│   ├── vendor/                 # 第三方（stb, minja）
│   ├── tests/                  # 单元测试
│   └── tools/                  # 辅助工具
├── deps/
│   ├── llama.cpp/              # 参考实现（只读）
│   └── ggml/                   # ggml C 源码（submodule）
└── tools/                      # 外部脚本
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
