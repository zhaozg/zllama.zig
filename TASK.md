# zllama.zig 开发任务

## 目标

实现 `zllama-simple` 与 `llama-simple` 的功能对齐，使 `zllama-simple` 能够正确加载并推理 Qwen3.5、LLaMA 等模型。

## 参考

- `src/simple_main.zig` — zllama-simple 入口
- `src/main.zig` — zllama 主入口
- `src/tokenize_main.zig` — 分词工具
- `deps/llama.cpp/examples/simple/simple.cpp` — llama-simple 参考实现
- `llama-simple.log` — llama-simple 运行日志（含 Qwen3.5-0.8B 输出）
- `deps/llama.cpp/src/models/qwen35.cpp` — Qwen3.5 模型参考实现
- `deps/llama.cpp/src/models/qwen35moe.cpp` — Qwen3.5 MoE 参考实现

### 使用方式

```bash
# tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
llama-simple -m ~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 你好
zig-out/bin/zllama-simple -m ~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 你好

# Qwen3.5-0.8B-Q4_K_M.gguf
llama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好
zig-out/bin/zllama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好
```

## 验收条件

1. **`zig-out/bin/zllama-simple` 与 `llama-simple` 的推理输出一致**（允许细微差异，如采样随机性引起的输出不同，但整体语义和格式应相似）。
2. **功能对齐**：支持 `-m`、`-n`、`-t`、`-k`、`-tp`、`-th`、`-v`、`-d` 等命令行参数。
3. **流程对齐**：加载模型 → 分词 → 构建计算图 → 推理 → 采样 → 输出 → 性能统计。
4. **模块化设计对齐**：不是简单的 C 到 Zig 代码转换，而是利用 Zig 的特性（comptime、switch、错误联合等）进行合理抽象。
5. **多模型支持**：至少支持 Qwen3.5（含混合注意力/SSM）和 LLaMA 系列模型。

## 当前状态

### 已完成

- ✅ 项目结构搭建（AGENTS.md、ARCHITECTURE.md、GGML_BINDING.md、TECHNICAL_CHALLENGES.md、ROADMAP.md）
- ✅ ggml.zig 安全封装层（C 绑定、Context、Tensor、CGraph、Backend、Ops）
- ✅ GGUF v2/v3 解析器
- ✅ 模型抽象接口（model.zig）
- ✅ 模型注册与工厂函数（registry.zig）
- ✅ Qwen2 模型实现
- ✅ Qwen3.5 模型实现（含混合注意力、SSM 层）
- ✅ LLaMA 模型实现
- ✅ KV Cache 管理
- ✅ BPE 分词器
- ✅ 采样器（贪心采样）
- ✅ zllama-simple 入口（simple_main.zig）
- ✅ 构建脚本（build.zig，含三个可执行文件）
- ✅ 卷积/SSM 相关算子（conv1d、ssmConv、ssmScan、gatedDeltaNet）
- ✅ 基于词汇表的 tokenizer 测试（test_vocab.zig，18 个词汇表）
- ✅ 注意力 mask 修复（diagMaskInf 正确处理 3D 张量）
- ✅ Qwen3.5 Q/gate 交错布局修复（view_3d with interleaved stride）
- ✅ SSM 状态持久化（ctx_kv_cache 分配，不受 ctx_graph.reset() 影响）
- ✅ gdn_output view 修复（使用正确的 stride）
- ✅ tinyllama 推理正确 ✅
- ✅ Llama-3.2-3B 推理正确 ✅
- ✅ **Qwen3.5 SSM 层推理正确性**：修复完成！根本原因是 `simple_main.zig` 和 `main.zig` 中在每 token 增量解码循环中调用了 `model.resetSSMStates()`，导致 SSM 的循环状态（conv_state 和 ssm_state）被重置为零，模型失去记忆能力。修复后 Qwen3.5-0.8B 能生成连贯的中文文本。

### 待完成/待修复

- ❌ **推理正确性验证**：zllama-simple 的输出与 llama-simple 对比，修复可能的计算图构建错误
- ❌ **RoPE 位置编码**：验证 Qwen3.5 的分段 RoPE（dimension_sections）实现
- ✅ **EOG 检测**：tokenizer 新增 `isEog()` 方法和 `eog_ids` 集合，与 llama.cpp 的 `llama_vocab_is_eog()` 对齐。通过名称匹配和 GGUF 元数据收集 EOG tokens，`simple_main.zig` 使用 `isEog()` 替代复杂的 `isSpecialToken` 检查。
- ❌ **性能优化**：当前每 token 重建计算图，应复用图结构
- ❌ **多 prompt token 支持**：验证 batch 推理（n_prompt_tokens > 1）的正确性
- ❌ **测试覆盖**：增加与 llama.cpp 输出对比的测试用例

## 调试指南

### 日志级别

```bash
# 调试模式（显示详细日志）
zig-out/bin/zllama-simple -d -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好

# 详细模式
zig-out/bin/zllama-simple -v -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好
```

### 常见问题

1. **GraphAllocFailed**：ctx_graph 内存不足，增大 mem_size_estimate
2. **输出乱码**：tokenizer 的 decodeSingle 实现有问题，检查 BPE 解码逻辑
3. **输出为空**：采样得到的 token_id 为 0（unk）或 EOS，检查 logits 形状和采样逻辑
4. **速度慢**：每 token 重建计算图导致，后续应实现图复用
5. **Qwen3.5 输出 "0"**：SSM 层（gatedDeltaNet）计算图构建有问题，需要检查 gdn_output 的 view 和 state 管理

## 禁止操作

- ❌ 禁止运行其他 llama 命令行工具（如 llama-cli、llama-perplexity 等）来干扰测试
- ❌ 禁止删除功能代码来绕过问题
- ❌ 禁止在业务代码中直接调用 `@cImport` 或裸 C 函数（必须通过 ggml.zig 封装）
