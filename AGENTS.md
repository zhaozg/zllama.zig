# ggml.zig AI 编程入口

> 本文档用于指导 AI 助手（如 GitHub Copilot、Cursor、Claude）理解并辅助开发 **Qwen 3.5 本地推理引擎**（Zig + ggml）。
> 请严格遵循以下目标、约束及设计决策。

## 🎯 项目目标

实现一个**生产级、高性能、跨平台**的本地推理引擎，支持 **Qwen 3.5 9B / 27B**（及未来 35B-A3B MoE）模型，满足以下要求：

- 完全基于 **Zig 0.16.0** 与 **ggml**（C 库），单二进制原生运行，无 Python / C++ 运行时依赖。
- 支持 **GGUF v2 / v3** 格式，能够解析混合架构（全注意力 + 线性注意力层）及量化张量。
- 实现**增量解码 & KV Cache**，长上下文（≥32K）内存友好。
- 支持 **多后端**：CPU（默认）、Metal (macOS)、CUDA (Linux)。GPU 后端可选，但不强制。
- 达到**可用推理速度**：9B Q4_K_M ≥15 tok/s，27B Q4_K_M ≥5 tok/s（CPU + 多线程）。
- 提供**最小 BPE 分词器**（从 GGUF 提取词表），无需外部 tokenizer 库。

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

3. **Qwen 3.5 混合架构实现要求**
   - 必须从 GGUF 元数据读取 **`layer_type` 数组**，区分 `full_attention` 与 `linear_attention` 层。
   - 线性注意力层需实现 **1D 因果卷积**（调用 `ggml_conv_1d`）。
   - 正确处理 **`attn_output_gate`**（门控机制）：在残差连接前乘以 gate 张量。
   - 支持 **GQA**（`n_kv_heads` 可能不等于 `n_heads`）。

4. **GGUF v3 兼容**
   - v3 使用 64 位字段（`tensor_count`、`metadata_kv_count`），无填充，**张量数据 32 字节对齐**。
   - 解析器需根据 `version` 字段动态选择读取路径，并对齐分配缓冲区。

5. **内存与性能**
   - 使用 **`mmap`**（通过 ggml 后端文件加载）避免模型数据全量驻留物理内存。
   - KV Cache 预分配固定大小张量（`[max_seq_len, n_kv_heads, head_dim]`），增量写入通过 `ggml_view_*` 切片，**禁止每 token 复制历史缓存**。
   - 默认启用 `ggml_graph_plan` + 线程池（物理核心数的 2/3 ~ 3/4）。

6. **测试与可调试性**
   - 提供与 `llama.cpp` 或 HuggingFace 输出对比的测试用例（短 prompt）。
   - 调试模式下可打印张量形状及部分值，Release 模式下完全移除。

## 📁 项目结构（AI 应遵循）

```
qwen-engine/
├── AGENTS.md                    # 本文件
├── ARCHITECTURE.md              # 系统架构设计
├── GGML_BINDING.md              # ggml.zig 绑定设计
├── TECHNICAL_CHALLENGES.md      # 技术难重点分析
├── build.zig                    # Zig 构建脚本
├── src/
│   ├── main.zig                 # 入口、CLI、I/O 初始化
│   ├── ggml.zig                 # C 绑定 + 安全封装
│   ├── gguf.zig                 # GGUF 解析器（v2/v3）
│   ├── model.zig                # 模型加载、层构建
│   ├── layers/
│   │   ├── full_attn.zig        # 标准注意力
│   │   ├── linear_attn.zig      # 线性注意力（1D conv）
│   │   └── moe.zig              # MoE 路由（可选）
│   ├── kv_cache.zig             # KV Cache 管理
│   ├── tokenizer.zig            # BPE 分词器
│   ├── sampler.zig              # Top-p / Top-k 采样
│   └── backend.zig              # 多后端抽象（CPU/Metal/CUDA）
└── deps/ggml/                   # ggml 源码（submodule 或拷贝）
```

## 🤖 AI 工作流程

1. **需求理解**：优先阅读 `AGENTS.md`、`ARCHITECTURE.md`、`GGML_BINDING.md`、`TECHNICAL_CHALLENGES.md`。
2. **代码生成**：
   - 任何涉及 ggml C 调用的代码必须通过 `ggml.zig` 的封装，不可直接 `@cImport` 到业务模块。
   - 新增层或算子时，先在 `ggml.zig` 添加安全封装，再在业务中使用。
   - 资源释放一律使用 `defer`，分配失败返回错误。
3. **约束检查**：
   - 确保所有阻塞 I/O 都通过传递的 `*std.Io` 参数执行。
   - 检查张量维度是否匹配（使用 `std.debug.assert`）。
   - 验证 GGUF 版本兼容性。
4. **提交前验证**：
   - 运行 `zig build test`（如果存在测试）。
   - 确保未引入未定义行为（如数组越界、空指针解引用）。
5. **文档同步**：修改架构或绑定设计后，需同步更新对应的 `*.md` 文件。

## 🔒 禁止事项

- ❌ 在业务代码中直接调用 `@cImport` 或裸 C 函数。
- ❌ 忽略 `std.Io` 接口化要求，使用 `std.fs.cwd()` 等阻塞 API。
- ❌ 硬编码模型超参数（必须从 GGUF 元数据读取）。
- ❌ 对 KV Cache 进行不必要的物理复制。
- ❌ 使用不安全的 `@alignCast` 或假设指针对齐。

## 📚 参考材料

- [ggml 源码](https://github.com/ggerganov/ggml)
- [llama.cpp Qwen 支持](https://github.com/ggerganov/llama.cpp/tree/master/models/qwen)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Zig 0.16.0 文档](https://ziglang.org/documentation/0.16.0/)

---

**AI 助手应始终以“安全、可维护、高性能”为原则，优先遵循本文档约束。如有歧义，请在对话中提问澄清。**

---
