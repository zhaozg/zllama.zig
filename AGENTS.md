# zllama.zig AI 编程入口

> 本文档用于指导 AI 助手（如 GitHub Copilot、Cursor、Claude）理解并辅助开发 **多模型本地推理引擎**（Zig + ggml）。
> 请严格遵循以下目标、约束及设计决策。

## 🎯 项目目标

实现一个**生产级、高性能、跨平台**的本地推理引擎，支持 **Qwen 3.5 / LLaMA 3** 等多模型架构，满足以下要求：

- 完全基于 **Zig 0.16.0** 与 **ggml**（C 库），单二进制原生运行，无 Python / C++ 运行时依赖。
- 支持 **GGUF v2 / v3** 格式，能够解析混合架构（全注意力 + 线性注意力层）及量化张量。
- 实现**增量解码 & KV Cache**，长上下文（≥32K）内存友好。
- 支持 **多后端**：CPU（默认）、Metal (macOS)、CUDA (Linux)。GPU 后端可选，但不强制。
- 达到**可用推理速度**：9B Q4_K_M ≥15 tok/s，27B Q4_K_M ≥5 tok/s（CPU + 多线程）。
- 提供**最小 BPE 分词器**（从 GGUF 提取词表），无需外部 tokenizer 库。
- **多模型架构**：通过 GGUF 元数据中的 `general.architecture` 字段自动检测模型类型。

## ⚠️ 核心约束

1. **Zig 0.16.0 特定行为**
   - 必须适配 I/O 接口化（`std.Io`），阻塞操作需通过 `io` 实例。
   - `build.zig` 中不得使用已废弃的 `exe.linkSystemLibrary`；应通过 `b.addStaticLibrary` 或 `addCSourceFiles` 集成 ggml 源码。
   - 使用 `root_module.addCMacro` 传递 ggml 编译宏（如 `GGML_USE_METAL`）。

2. **ggml 集成策略**
   - **静态编译 ggml 源码**（推荐），避免预编译库的 ABI 不一致。
   - 所有 ggml C API 必须通过 `ggml.zig` 模块封装为类型安全的 Zig 函数，且保留 `c` 命名空间供高级用户直接调用。
   - 分配类操作（如 `ggml_new_tensor`）必须返回 `!*T` 错误联合，纯计算操作返回 `*T`。
   - 使用 `opaque {}` 类型包装不透明指针（`ggml_context`、`ggml_tensor` 等）。

3. **Zig 0.16.0 I/O 接口化约束**
   - **所有**阻塞 I/O 操作必须通过 Io 实例完成，函数签名应显式传递 Io 参数（类似 Allocator 模式）。
   - 使用 **Juicy Main** 获取预初始化的 Io 实例：`pub fn main(init: std.process.Init) !void { const io = init.io; ... }`。
   - 不再使用 `std.fs.cwd()` 等旧 API，改用 `std.Io.Dir.cwd().openFile(io, ...)` 模式。
   - 如需自定义 Io 实现，使用 `std.Io.Threaded`（稳定、功能完整），`std.Io.Evented` 仍处于实验阶段。

4. **多模型架构实现要求**
   - 必须从 GGUF 元数据读取 **`general.architecture` 字段**，区分模型类型。
   - 模型实现放在 `src/models/` 目录，共享算子放在 `src/layers/` 目录。
   - 新增模型只需在 `registry.zig` 增加枚举值和对应 case。
   - 使用 Zig 的 `switch` 实现零成本运行时多态分发。

5. **时间测量约束（Zig 0.16.0 破坏性变更）**
   - **`std.time.Timer` 已被移除**，`std.time.nanoTimestamp()` 不再是推荐方式。
   - 新 API：使用 `std.Io.Clock.now(.awake, io)` 获取时间戳。

6. **GGUF v3 兼容**
   - v3 使用 64 位字段（`tensor_count`、`metadata_kv_count`），无填充，**张量数据 32 字节对齐**。
   - 解析器需根据 `version` 字段动态选择读取路径，并对齐分配缓冲区。

7. **内存与性能**
   - 使用 **`mmap`**（通过 ggml 后端文件加载）避免模型数据全量驻留物理内存。
   - KV Cache 预分配固定大小张量（`[max_seq_len, n_kv_heads, head_dim]`），增量写入通过 `ggml_view_*` 切片，**禁止每 token 复制历史缓存**。
   - 默认启用 `ggml_graph_plan` + 线程池（物理核心数的 2/3 ~ 3/4）。

8. **测试与可调试性**
   - 提供与 `llama.cpp` 或 HuggingFace 输出对比的测试用例（短 prompt）。
   - 调试模式下可打印张量形状及部分值，Release 模式下完全移除。
   - 善于使用 std.log 模块，利于在开发、调试阶段分析定位问题。

9. **默认化设计原则**
   - **安全、可维护、高性能**，优先考虑代码清晰和正确性，必要时牺牲微小性能提升。
   - 单文件模块应保持清晰的接口和实现分离，避免过度复杂化。
   - 避免文件过大，不能超过 600 行，合理拆分功能模块（如算子、模型实现、核心引擎等）。

## 🔍 软件工程实践：功能对齐与早期问题发现

为保证开发质量和避免后期返工，在编码前、编码中和集成阶段必须遵循以下实践：

### 1. 功能对齐（Requirements Alignment）

- **开发前编写设计检查清单**：针对新模型或新特性，先列出所有必须从 GGUF 读取的超参数、张量名称、特殊逻辑（如 `full_attention_interval`、`rope_sections`）。对照 llama.cpp 参考实现逐项勾选，确保无遗漏。
- **接口契约定义**：模块之间通过 `pub const` 导出的 API 应有明确的输入/输出、错误类型和内存所有权约定。使用 Zig 的文档注释（`///`）描述契约。
- **交叉验证**：对于关键路径（如 `forward()`、KV Cache 更新），用注释标明对应 llama.cpp 的源文件和行号，便于后续比较。

### 2. 提前问题发现（Shift-Left Testing）

- **单元测试先行**：每个新算子（如 `gatedDeltaNet`、`ropeMulti`）必须在 `ggml.zig` 或其测试模块中编写独立测试，使用随机输入对比简单实现或参考实现。
- **数值偏差容忍度**：定义 NMSE（归一化均方误差）< 1e-5 或余弦相似度 > 0.999 为通过。对于量化模型可适当放宽。
- **早期集成验证**：
  - 新模型首次可加载后，立即运行 `zig build test-model`（若存在）或手动执行极短 prompt（如 `-n 1`），验证 `forward()` 不崩溃且输出 logits 不是全零/NaN。
  - 使用 `tools/compare_logits.zig` 对比第一个 token 的 logits 与参考实现（llama.cpp 或已实现的正确模型）。
- **静态分析**：启用 `zig build test -Drelease-safe`，利用 Zig 的内置安全检查（整数溢出、切片边界、可选类型解包）捕获未定义行为。

### 3. 持续回归防御

- **CI 自动化**：每次 PR 必须通过所有单元测试和模型快速冒烟测试（如 `tinyllama`、`Llama-3.2-3B`、`Qwen3.5-0.8B` 各跑 5 token）。
- **基准快照**：将正确模型的 logits 样本（前 5-10 token）签入仓库（`tests/reference_logits/`），通过 CI 对比，防止回归。
- **失败即阻断**：任何数值偏差超出阈值或崩溃应阻断合并，直到修复。

### 4. 增量开发与对比

- **按阶段实现**：将模型实现拆分为：① 超参数加载 + 张量加载测试；② 全注意力层前向（不集成 KV Cache）；③ 线性注意力层前向；④ 完整推理循环。每阶段独立验证。
- **使用 `tools/generate_reference.zig`**：从正确的实现（llama.cpp）生成参考 logits，作为本项目的 Golden Master。
- **分支对比**：在实现新功能时，创建独立分支并定期与 `main` 合并，避免偏离太久。

### 5. 文档化已知差距

- 对于与参考实现的故意差异（如融合算子使用、布局优化），在代码注释或 `docs/` 中明确记录原因和影响范围。
- 维护 `docs/KNOWN_ISSUES.md`，列出未修复的数值偏差或性能问题，并标注优先级。

## 📁 项目结构（AI 应遵循）

```
zllama.zig/
├── AGENTS.md                    # 本文件（AI 编程入口）
├── ROADMAP.md                   # 开发路线图（含已完成功能、待办、调试指南）
├── README.md                    # 项目简介
├── build.zig                    # Zig 构建脚本
├── docs/                        # 设计文档
│   ├── ARCHITECTURE.md          # 系统架构设计
│   ├── GGML_BINDING.md          # ggml.zig 绑定设计
│   ├── TECHNICAL_CHALLENGES.md  # 技术难重点分析
│   ├── TEST.md                  # 测试体系文档
│   └── QWEN35.md                # Qwen3.5 模型实现笔记
├── src/                         # 源码组织具体看 docs/FILE_STRUCTURE.md
├── models/                      # 模型辅助工具
│   ├── ...
│   └── templates/               # jinja file for models
├── deps/zig-jinja/              # zig-jinja 引擎实现库
├── deps/llama.cpp/              # llama.cpp 相关代码（可参考）
└── deps/ggml/                   # ggml 源码（submodule 或拷贝）
```

## 📦 Import 规则（仅使用模块名）

本项目所有模块通过 `build.zig` 中的 `createModule` 定义，模块间引用**必须使用模块名**（由 `addImport` 注册）。

### 核心规则
- **任何文件引用另一个模块的根文件，一律使用模块名**：`@import("模块名")`
- **禁止使用相对路径导入**（如 `@import("ggml.zig")`、`@import("../model.zig")` 等）
- **模块内部细节不对外暴露**：子文件通过根文件的 `pub const` 重新导出

### 模块注册表（来自 `build.zig`）
| 模块名 | 根文件 |
|--------|--------|
| `ggml` | `src/ggml.zig` |
| `gguf` | `src/gguf.zig` |
| `model` | `src/model.zig` |
| `graph_builder` | `src/core/graph_builder.zig` |
| `memory` | `src/core/memory.zig` |
| `registry` | `src/models/registry.zig` |
| `tokenizer` | `src/tokenizer.zig` |
| `sampler` | `src/sampler.zig` |
| `kv_cache` | `src/kv_cache.zig` |

### 示例（正确 ✅）
```zig
// src/model.zig（模块根）
const ggml = @import("ggml");
const graph_builder = @import("graph_builder");

// src/main.zig（模块根）
const model = @import("model");
const tokenizer = @import("tokenizer");
```

### 禁止事项 ❌
- 使用相对路径：`@import("../ggml.zig")`
- 使用文件名：`@import("ggml.zig")`（应写 `@import("ggml")`）
- 混合使用模块名与路径导入同一个文件

## 🤖 AI 工作流程

1. **需求理解**：优先阅读 `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/GGML_BINDING.md`、`docs/TECHNICAL_CHALLENGES.md`。
2. **功能对齐检查**：对照新增功能的设计检查清单，确认所有超参数、张量、特殊逻辑已识别。
3. **代码生成**：
   - 任何涉及 ggml C 调用的代码必须通过 `ggml.zig` 的封装，不可直接 `@cImport` 到业务模块。
   - 新增层或算子时，先在 `ggml.zig` 添加安全封装，再在业务中使用。
   - 新增模型时，在 `src/models/` 下创建新文件，在 `registry.zig` 注册。
   - 资源释放一律使用 `defer`，分配失败返回错误。
   - **遵循 Import 策略**：根文件之间使用模块名导入，非根文件使用相对路径导入。
4. **早期验证**：
   - 完成超参数加载后，编写单元测试验证解析值与 GGUF 元数据一致。
   - 完成张量加载后，运行简单图（如只做一次 embedding lookup）检查形状。
   - 完成首次 `forward()` 后，立即用 `compare_logits` 对比参考实现。
5. **约束检查**：
   - 确保所有阻塞 I/O 都通过传递的 `*std.Io` 参数执行。
   - 检查张量维度是否匹配（使用 `std.debug.assert`）。
   - 验证 GGUF 版本兼容性。
6. **问题定位**：
   - 善于通过 `git diff` 发现改动引入的潜在问题。
   - 必要时在对话中提问澄清约束或设计细节，避免误解导致的实现偏差。
   - 可参考 `llama.cpp` 的实现细节，但必须适配 Zig 0.16.0 的特定要求。
7. **提交前验证**：
   - 运行 `zig build test`（如果存在测试）。
   - 确保未引入未定义行为（如数组越界、空指针解引用）。
   - `zig-out/bin/zllama -n 5 --model ~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` works
   - `zig-out/bin/zllama -n 5 --model ~/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf` works
   - `zig-out/bin/zllama -n 5 --model ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf` works
   - `zig-out/bin/zllama -n 5 --model ~/.cache/models/Qwen3.5-4B-Q4_K_M.gguf` works
   - 多模态模型 gemma-4-E2B-it-Q4_K_M.gguf 的音频、图像主要输出对齐
      - llama-simple -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj ~/.cache/models/mmproj-BF16.gguf --audio ~/.cache/models/hello.wav -p  "Transcribe the audio"
      - llama-simple -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj ~/.cache/models/mmproj-BF16.gguf --image ~/.cache/models/hello.png --prompt "Describe this image"
8. **效果参照**:
   - `llama-simple -m ~/.cache/models/Qwen3.5-4B-Q4_K_M.gguf 你好`
9. **文档同步**：修改架构或绑定设计后，需同步更新对应的 `*.md` 文件。

## 🔒 禁止事项

- ❌ 在业务代码中直接调用 `@cImport` 或裸 C 函数。
- ❌ 忽略 `std.Io` 接口化要求，使用 `std.fs.cwd()` 等阻塞 API。
- ❌ 硬编码模型超参数（必须从 GGUF 元数据读取）。
- ❌ 对 KV Cache 进行不必要的物理复制。
- ❌ 使用不安全的 `@alignCast` 或假设指针对齐。
- ❌ 删除功能代码，绕开问题。
- ❌ 在根文件中使用相对路径导入另一个根文件。
- ❌ 在业务代码中直接导入 `ggml/` 子目录文件。
- ❌ 参数硬编码，没有从权重文件实际形状获取。
- ❌ 直接使用`llama-cli`交互式聊天模式, 导致 run_command 无法处理.

## 📚 参考材料

- [ggml 源码](deps/ggml/)
- [llama.cpp Qwen 支持](deps/llama.cpp/src/models/qwen*.cpp)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Zig 0.16.0 文档](https://ziglang.org/documentation/0.16.0/)

---

**AI 助手应始终以“安全、可维护、高性能”为原则，优先遵循本文档约束。如有歧义，请在对话中提问澄清。**
