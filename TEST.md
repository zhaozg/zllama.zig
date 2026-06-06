# 底层功能验证测试

基于词汇表的测试，验证 tokenizer 的编码/解码正确性。

## 测试数据

`deps/llama.cpp/models/` 目录下的数据可用于测试。每个词汇表有三个文件：

- `ggml-vocab-<name>.gguf` — GGUF 格式的词汇表文件
- `ggml-vocab-<name>.gguf.inp` — 测试输入文本（由 `__ggml_vocab_test__` 分隔）
- `ggml-vocab-<name>.gguf.out` — 预期输出 token ID（每行空格分隔）

## 测试策略

参考 llama.cpp 的 `test-tokenizer-0.cpp` 设计：

1. **编码验证**：对每个输入文本进行 tokenize，验证输出 token ID 与预期一致
2. **往返一致性**：验证 `detokenize(tokenize(text))` 基本正确
3. **多词汇表覆盖**：支持所有 llama.cpp 提供的词汇表测试数据

## 支持的词汇表

| 名称 | 类型 | 说明 |
|------|------|------|
| llama-bpe | BPE | LLaMA 家族 BPE 词汇表 |
| llama-spm | SentencePiece | LLaMA 家族 SPM 词汇表 |
| qwen2 | BPE | Qwen2 系列 |
| qwen35 | BPE | Qwen3.5 系列 |
| gpt-2 | BPE | GPT-2 |
| falcon | BPE | Falcon |
| deepseek-coder | BPE | DeepSeek Coder |
| deepseek-llm | BPE | DeepSeek LLM |
| phi-3 | BPE | Phi-3 |
| command-r | BPE | Command-R |
| starcoder | BPE | StarCoder |
| mpt | BPE | MPT |
| refact | BPE | ReFact |
| baichuan | BPE | Baichuan |
| bert-bge | BPE | BERT/BGE |
| gemma-4 | BPE | Gemma-4 |
| nomic-bert-moe | BPE | Nomic BERT MoE |
| aquila | BPE | Aquila |

## 运行测试

```bash
# 运行所有测试
zig build test

# 仅运行词汇表测试
zig build test-vocab

# 仅运行 GGUF 解析测试
zig build test-gguf

# 仅运行层测试
zig build test-layers

# 仅运行架构测试
zig build test-archs

# 仅运行 KV Cache 测试
zig build test-kv-cache
```

## 测试文件结构

```
src/tests/
├── test_gguf.zig          # GGUF 解析测试（手工构造二进制数据）
├── test_layers.zig        # 算子层数值测试（RMSNorm, RoPE, SwiGLU, Attention）
├── test_kv_cache.zig      # KV Cache 功能测试
├── test_archs.zig         # 架构前向测试（随机权重推理验证）
├── test_vocab.zig         # 词汇表测试（基于 llama.cpp 标准测试数据）
├── test_compare_logits.zig # Logits 比较器测试
└── utils.zig              # 测试工具函数
```

## 测试数据格式

### .inp 文件格式

文本由 `__ggml_vocab_test__` 分隔符分割，每个段是一个独立的测试输入：

```
Hello world
__ggml_vocab_test__
 Hello world
__ggml_vocab_test__
Hello World
```

### .out 文件格式

每行对应一个测试输入的预期 token ID 序列，空格分隔：

```
9906 1917
22691 1917
9906 4435
```

## 实现细节

### 测试流程

1. 读取 GGUF 词汇表文件并解析
2. 初始化 Tokenizer
3. 读取 `.inp` 和 `.out` 文件
4. 按分隔符分割输入文本
5. 对每个输入进行 tokenize
6. 比较输出 token ID 与预期值
7. 验证往返一致性

### 错误处理

- 如果 token 数量不匹配，打印详细对比信息
- 如果 token ID 不匹配，打印具体位置和值
- 所有测试数据使用 `testing.allocator` 管理内存

## 与 llama.cpp 的对比

本测试实现与 llama.cpp 的 `test-tokenizer-0.cpp` 功能对齐：

| 特性 | llama.cpp | zllama.zig |
|------|-----------|------------|
| 加载 GGUF 词汇表 | ✅ | ✅ |
| 解析 .inp/.out 文件 | ✅ | ✅ |
| 多线程 tokenize | ✅ | ❌（单线程） |
| 编码验证 | ✅ | ✅ |
| 往返一致性 | ✅ | ✅ |
| Unicode 码点测试 | ✅ | ❌（可扩展） |
| 多词汇表支持 | ✅ | ✅ |

## 推理验证状态

| 模型 | 状态 | 说明 |
|------|------|------|
| tinyllama-1.1b | ✅ | 推理正确，输出合理文本 |
| Llama-3.2-3B | ✅ | 推理正确，输出合理文本 |
| Qwen3.5-0.8B | ✅ | 推理正确！修复了 `resetSSMStates()` 在增量解码循环中被错误调用的问题 |

## 扩展指南

### 添加新的词汇表测试

1. 将 GGUF 词汇表文件放入 `deps/llama.cpp/models/`
2. 创建对应的 `.inp` 和 `.out` 文件
3. 在 `src/tests/test_vocab.zig` 的 `vocab_tests` 数组中添加条目
4. 添加对应的 `test` 块

### 添加新的测试类型

1. 在 `src/tests/` 下创建新的测试文件
2. 在 `src/main.zig` 底部导入该文件
3. 在 `build.zig` 中添加对应的 `b.step()` 定义
