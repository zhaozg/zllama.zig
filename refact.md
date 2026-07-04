# 整体重构

基于对 `zllama.zig` 项目的全面分析，并结合 `AGENTS.md` 中定义的严格约束与设计目标，提出以下改进建议。
这些建议旨在进一步提升代码质量、可维护性、合规性及性能，确保项目长期健康发展。

---

## 一、重构超长文件，恪守“单文件 ≤600 行”原则

**现状**：分析显示多个核心文件远超 600 行：
- `src/core/engine.zig` – **1506 行**
- `src/models/gemma4.zig` – **1316 行**
- `src/chat_template/mod.zig` – **1262 行**
- `build.zig` – **1221 行**（构建脚本可豁免，但建议拆分模块）
- `src/vocab.zig` – **891 行**

**建议**：
- 将 `engine.zig` 按职责拆分为：
  - `core/engine.zig`（主循环、状态机）
  - `core/prefill.zig`（预填充逻辑）
  - `core/decode.zig`（增量解码循环）
  - `core/chat_loop.zig`（交互式聊天）
  - `core/embedding.zig`（嵌入生成）
- `gemma4.zig` 可拆分为：
  - `models/gemma4.zig`（模型定义、配置）
  - `models/gemma4/attention.zig`（全注意力+线性注意力）
  - `models/gemma4/ffn.zig`（GEGLU）
  - `models/gemma4/kv_cache.zig`（KV 映射）
- `chat_template/mod.zig` 按模板族拆分：`chat_template/templates/llama.zig`、`chat_template/templates/gemma.zig` 等。
- `build.zig` 将 ggml 编译、模块集成、测试目标等逻辑抽取到 `build/` 目录下的辅助脚本（Zig 支持 `build.zig` 调用外部 `.zig` 文件）。

**理由**：遵守 AGENTS.md 第5条“文件不超过600行”，提升可读性与可测试性，便于多人协作。

---

## 二、完善 ggml 绑定与封装，杜绝裸 C 调用

**现状**：分析指出项目依赖 `ggml` C 库，但未明确检查所有业务代码是否通过 `src/ggml/` 模块调用。可能存在直接 `@cImport` 或调用 C 函数的情况。

**建议**：
- 强制代码审查：在 CI 中添加脚本，禁止 `src/` 下除 `ggml/` 目录外的文件出现 `@cImport` 或 `c.` 前缀。
- 补充缺失的 ggml 函数绑定（尤其涉及 `ggml_graph_plan`、`ggml_view_*`、`ggml_cpu_mask` 等）。
- 为每个新增绑定编写单元测试（参考 `docs/GGML_BINDING.md` 要求）。

**理由**：保证类型安全、错误处理统一，并便于未来切换后端（如 Vulkan）。

---

## 三、统一模型注册与架构检测，确保扩展性

**现状**：已存在 `src/models/registry.zig` 和 `model.zig` 中的 `Architecture` 枚举，但可能部分模型（如 Gemma 4）未完全遵循注册流程。

**建议**：
- 确认所有模型文件（`gemma4.zig`, `qwen35.zig`, `llama.zig` 等）均在 `registry.zig` 的 `createModel` 中注册。
- 将 `Architecture` 枚举与 GGUF 元数据 `general.architecture` 的字符串映射集中管理，避免多处硬编码。
- 对于多模态模型，增加 `general.multimodal` 标志，并在 `ModelInstance` 中提供 `encode_image`/`encode_audio` 虚函数。

**理由**：保持新增模型的低成本，符合 `HOW_TO_ADD_NEW_MODEL.md` 指南。

---

## 四、KV Cache 零拷贝约束验证与优化

**现状**：文档强调“禁止每 token 复制历史缓存”，但代码中可能存在未使用 `ggml_view_*` 的地方。

**建议**：
- 审查 `kv_cache.zig` 和模型实现中的 KV 写入逻辑，确保全部通过 `ggml_view_1d`/`ggml_view_2d` 创建切片。
- 添加运行时断言（在 Debug 模式下）检查是否发生了隐式复制（如通过 `ggml_dup` 或 `ggml_cpy` 且源为目标之外）。
- 若使用 `ggml_backend_sched` 调度，确保视图操作与后端内存对齐。

**理由**：长上下文场景下内存效率至关重要，必须从设计上保证。

---

## 五、增强测试体系，强制 Logits 验证

**现状**：AGENTS.md 要求使用 `compare_logits` 验证，且每个 `forward()` 后检查非全零/NaN。但分析未涉及测试覆盖细节。

**建议**：
- 在 CI 中集成 `src/tools/compare_logits.zig`，对每个模型生成前 5 个 token 的 logits，与 llama.cpp 参考值对比（NMSE < 1e-5）。
- 添加 `tests/` 下的集成测试，使用 TinyLlama 等小模型快速验证完整推理流程。
- 为每个新增层（如 RoPE、RMSNorm）添加独立的单元测试，对照数学公式或 PyTorch 参考。

**理由**：确保重构不引入精度回归，保障生产质量。

---

## 六、文档同步与注释规范

**现状**：项目文档较全，但可能因代码演进出现不一致。

**建议**：
- 在 PR 流程中强制要求更新 `docs/` 中对应章节（如新增模型需更新 `ARCHITECTURE.md` 和 `HOW_TO_ADD_NEW_MODEL.md`）。
- 在复杂函数（如 `engine.zig` 中的主循环）开头注释对应的 llama.cpp 文件行号，便于交叉检查。
- 记录故意偏离 llama.cpp 的设计（如自定义调度策略），并说明原因。

**理由**：降低后续维护成本，便于新人理解。

---

## 七、优化 `build.zig` 结构，支持增量编译

**现状**：`build.zig` 超过 1200 行，包含大量 ggml 编译配置、宏定义和模块注册。

**建议**：
- 将 ggml 构建逻辑抽取到 `build_ggml.zig`，通过 `b.addStaticLibrary` 返回对象，减少主文件冗长。
- 使用 Zig 0.16.0 的 `b.dependency` 管理子模块，避免硬编码路径。
- 为不同后端（CPU/Metal/CUDA）提供独立的编译选项，并启用 `-Doptimize` 组合。

**理由**：提高构建可维护性，减少编译时间（利用缓存）。

---

## 八、内存对齐与分配策略审计

**现状**：AGENTS.md 要求音频/图像 buffer 使用对齐分配，并确保类型转换安全。

**建议**：
- 全局检查 `allocator.alignedAlloc` 的使用，尤其 `mtmd/` 下的预处理代码。
- 使用 `@as` 和 `@bitCast` 时增加 `@compileError` 或断言防止未定义行为。
- 对 KV Cache 预分配的大张量，使用 `ggml_new_tensor` 配合对齐参数（ggml 内部保证）。

**理由**：避免偶发崩溃，尤其在不同硬件平台（ARM/x86）上。

---

## 九、明确错误处理与日志策略

**现状**：未统一错误返回和日志输出。

**建议**：
- 定义全局 `log` 模块（或使用 `std.log`），输出级别由编译选项控制（`-Dlog-level`）。
- 所有错误返回时附带上下文（如 `error.FailedToLoadTensor` + 张量名），便于调试。
- 在 `main.zig` 中捕获顶层错误并打印友好信息，避免 panic。

**理由**：提升用户体验，便于问题定位。

---

以上改进建议均紧密围绕 `AGENTS.md` 的核心约束与设计目标，
旨在将 `zllama.zig` 打造成坚固、可扩展、社区友好的本地推理引擎。
我们应当分阶段实施（先拆分文件，再加固测试），并在每次迭代后运行全量测试套件，确保零回归。
