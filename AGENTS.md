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

**规则**：模块间引用用 `@import("模块名")`，禁止相对路径导入（如 `@import("../ggml.zig")`）。模块子文件通过根文件的 `pub const` 重新导出。

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
