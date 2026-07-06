---
title: 工具使用说明
...

本文档描述 `zllama.zig` 项目提供的所有命令行工具及其用法。

${TOC}

---

## 1. zllama — 主推理引擎

**源码**: `src/main.zig`
**构建产物**: `zig-out/bin/zllama`
**构建命令**: `zig build`（默认构建）

主推理引擎，支持多模型架构的文本生成。

### 用法

```bash
zllama --model <model.gguf> [options]
```

### 参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model <path>` | `-m` | 模型 GGUF 文件路径（**必需**） | — |
| `--prompt <text>` | `-p` | 输入提示词 | 交互式输入 |
| `--n-predict <n>` | `-n` | 生成的 token 数量 | `256` |
| `--temp <f>` | | 温度参数（采样） | `0.7` |
| `--threads <n>` | `-t` | 线程数 | CPU 核心数 |
| `--chat-template <name>` | | 对话模板（如 `llama3`, `qwen`） | 从 GGUF 自动检测 |
| `--color` | | 彩色输出 | 启用 |

### 示例

```bash
# 基础文本补全
zllama -m ~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello" -n 20

# 使用 Qwen3.5 模型
zllama -m ~/.cache/models/Qwen3.5-4B-Q4_K_M.gguf -p "你好" -n 50

# 交互模式（不带 --prompt）
zllama -m ~/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## 2. zllama — LLM 交互式对话

**源码**: `src/main.zig`
**构建产物**: `zig-out/bin/zllama`
**构建命令**: `zig build`（默认构建）

LLM 引擎，支持：
- 文本对话模式（--chat-template）
- 图像多模态输入（--image，需要 --mmproj）
- 长上下文（>=32K token）
- Streaming 输出
- 交互式 REPL

### 用法

```bash
zllama --model <model.gguf> [options]
```

### 参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model <path>` | `-m` | 模型 GGUF 文件路径（**必需**） | — |
| `--mmproj <path>` | | 多模态投影器文件（用于图像输入） | — |
| `--image <path>` | | 图像文件路径（需要 --mmproj） | — |
| `--prompt <text>` | `-p` | 输入提示词 | 交互式 |
| `--n-predict <n>` | `-n` | 生成的 token 数量 | `-1`（无限，直到 EOG） |
| `--temp <f>` | | 温度参数 | `0.8` |
| `--top-k <n>` | | Top-K 采样 | `40` |
| `--top-p <f>` | | Top-P (nucleus) 采样 | `0.95` |
| `--min-p <f>` | | Min-P 采样 | `0.05` |
| `--threads <n>` | `-t` | 线程数 | CPU 核心数 |
| `--chat-template <name>` | | 对话模板 | 从 GGUF 自动检测 |
| `--no-mmap` | | 禁用 mmap 加载 | 启用 mmap |
| `--grammar <string>` | | GBNF 语法约束 | — |
| `--color` | | 彩色输出 | 启用 |
| `--verbose` | `-v` | 详细日志 | — |
| `--help` | `-h` | 显示帮助 | — |

### 示例

```bash
# 文本对话
zllama -m ~/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p "Hello" -n 100

# 图像多模态推理
zllama -m model.gguf --mmproj mmproj.gguf --image photo.jpg -p "Describe this image"

# 交互式对话（REPL 模式）
zllama -m ~/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf --chat-template llama3
```

---

## 3. zllama-tokenize — 分词器工具

**源码**: `src/tokenize_main.zig`
**构建产物**: `zig-out/bin/zllama-tokenize`
**构建命令**: `zig build`（默认构建）

使用给定模型的 tokenizer 对输入文本进行分词，输出 token ID 或 token 字符串。

### 用法

```bash
zllama-tokenize [options]
```

### 参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model <path>` | `-m` | 模型 GGUF 文件路径（**必需**） | — |
| `--prompt <text>` | `-p` | 要分词的文本 | — |
| `--file <path>` | `-f` | 从文件读取 prompt | — |
| `--stdin` | | 从标准输入读取 prompt | — |
| `--ids` | | 仅打印数字 token ID（如 `[1, 2, 3]`） | 打印 token 字符串 |
| `--no-bos` | | 不添加 BOS token | 根据模型自动决定 |
| `--no-escape` | | 不对输入做转义处理（如 `\n`, `\t`） | 自动转义 |
| `--show-count` | | 打印 token 总数 | — |
| `--log-disable` | | 禁用日志输出 | — |
| `--log-level <level>` | | 日志级别（debug/info/warn/err） | info |
| `--verbose` | `-v` | 详细日志（等同 `--log-level info`） | — |
| `--debug` | `-d` | 调试日志（等同 `--log-level debug`） | — |
| `--help` | `-h` | 显示帮助 | — |

> **注意**: `--prompt`, `--file`, `--stdin` 三者必须恰好指定一个。

### 示例

```bash
# 查看 "Hello, world!" 的 token ID
zllama-tokenize -m tinyllama-1.1b.Q4_K_M.gguf -p "Hello, world!" --ids

# 包含转义字符
zllama-tokenize -m Llama-3.2-3B.Q4_K_M.gguf -p "Line1\nLine2" --show-count

# 从文件读取
zllama-tokenize -m Qwen3.5-4B.Q4_K_M.gguf -f input.txt -v
```

---

## 4. zllama-dump-graph — 计算图导出

**源码**: `src/tools/dump_graph.zig`
**构建产物**: `zig-out/bin/zllama-dump-graph`
**构建命令**: `zig build dump-graph`

导出模型推理计算图，支持多种格式。用于调试和理解模型内部计算流程。

### 用法

```bash
zllama-dump-graph [options]
```

### 参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model <path>` | `-m` | 模型 GGUF 文件路径（**必需**） | — |
| `--prompt <text>` | `-p` | 输入提示词 | `"Hello"` |
| `--format <fmt>` | `-f` | 输出格式：`text`, `dot`, `json` | `text` |
| `--help` | `-h` | 显示帮助 | — |

### 输出格式

- **text** — 简洁的节点列表，包含操作类型、维度信息、名称
- **dot** — Graphviz DOT 格式，可用 `dot -Tsvg graph.dot -o graph.svg` 渲染
- **json** — JSON 格式，便于程序化分析

### 示例

```bash
# 文本格式输出
zllama-dump-graph -m tinyllama-1.1b.Q4_K_M.gguf -p "Hi" | tail -20

# 生成 DOT 图并渲染
zllama-dump-graph -m Llama-3.2-3B.Q4_K_M.gguf -f dot > graph.dot
dot -Tsvg graph.dot -o graph.svg

# JSON 格式
zllama-dump-graph -m Qwen3.5-4B.Q4_K_M.gguf -f json | jq '.nodes | length'
```

---

## 5. zllama-gen-ref — 参考 logits 生成

**源码**: `src/tools/generate_reference.zig`
**构建产物**: `zig-out/bin/zllama-gen-ref`
**构建命令**: `zig build gen-ref`

运行模型推理，将 token 的 logits 输出保存到文件，作为后续对比验证的金标准（Golden Master）。

### 用法

```bash
zllama-gen-ref --model <model.gguf> [options]
```

### 参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model <path>` | `-m` | 模型 GGUF 文件路径（**必需**） | — |
| `--prompt <text>` | `-p` | 输入提示词 | `"Hello"` |
| `--output <path>` | `-o` | 输出文件路径 | `ref_logits.bin` |
| `--format <fmt>` | `-f` | 输出格式：`binary` 或 `text` | `binary` |
| `--tokens` | | 输出中包含 token ID | 仅 logits |
| `--help` | `-h` | 显示帮助 | — |

### 输出格式

#### binary（默认）
二进制格式：先写入 token 列表长度（32位无符号整数），再写入 logits 长度（32位无符号整数），然后以 f32 数组写入所有 token 的 logits。

#### text
文本格式：包含注释头（模型路径、prompt、token 数、logits 数），后跟每行一个 f32 值。

### 示例

```bash
# 生成参考 logits
zllama-gen-ref -m tinyllama-1.1b.Q4_K_M.gguf -p "Hello" -o tiny_ref.bin

# 带 token ID 的二进制输出
zllama-gen-ref -m Llama-3.2-3B.Q4_K_M.gguf -p "Hi there" -o llama_ref.bin --tokens

# 文本格式（可读）
zllama-gen-ref -m Qwen3.5-4B.Q4_K_M.gguf -p "你好" -o qwen_ref.txt -f text

# 批量生成多个模型的参考
for m in tinyllama-1.1b Llama-3.2-3B gemma-4-E2B Qwen3.5-4B; do
    zllama-gen-ref -m ~/.cache/models/$m.Q4_K_M.gguf -o /tmp/ref_$m.bin
done
```

---

## 6. zllama-compare-logits — Logits 对比

**源码**: `src/tools/compare_logits.zig`
**构建产物**: `zig-out/bin/zllama-compare-logits`
**构建命令**: `zig build compare-logits`

比较两组 logits 文件，计算多种数值指标，验证实现正确性。

### 用法

```bash
zllama-compare-logits --ref <ref_file> --test <test_file>
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ref <path>` | 参考 logits 文件（**必需**） | — |
| `--test <path>` | 待测试 logits 文件（**必需**） | — |

### 输出的指标

| 指标 | 说明 | 通过标准 |
|------|------|----------|
| NMSE | 归一化均方误差 | < 1e-4 |
| Max Abs Error | 最大绝对误差 | < 0.01 |
| Cosine Similarity | 余弦相似度 | > 0.999 |
| PSNR | 峰值信噪比 | — |
| Mean Abs Error | 平均绝对误差 | — |
| Argmax Match | Token argmax 是否一致 | 一致 |

### 示例

```bash
# 由 zllama-gen-ref 生成参考 → 验证
zllama-gen-ref -m model.gguf -p "Hello" -o ref.bin
zllama-gen-ref -m model_v2.gguf -p "Hello" -o test.bin
zllama-compare-logits --ref ref.bin --test test.bin

# 与 llama.cpp 输出对比
# （需先用 llama.cpp 生成参考 logits）
llama-cli -m model.gguf -p "Hello" --logit-binary ref_llamacpp.bin
zllama-compare-logits --ref ref_llamacpp.bin --test ref.bin
```

---

## 7. zllama-compare-llamacpp — llama.cpp 对齐验证

**源码**: `src/tools/compare_with_llamacpp.zig`
**构建产物**: `zig-out/bin/zllama-compare-llamacpp`
**构建命令**: `zig build compare-llamacpp`

集成式验证工具：加载模型 → 推理 → 与 llama.cpp 参考 logits 对比，一步完成。

### 用法

```bash
zllama-compare-llamacpp --model <model.gguf> --prompt <text> --ref-logits <file> [-n tokens]
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model <path>` | 模型 GGUF 文件路径（**必需**） | — |
| `--prompt <text>` | 输入提示词（**必需**） | — |
| `--ref-logits <file>` | llama.cpp 生成的参考 logits 文件（**必需**） | — |
| `-n <tokens>` | 对比的 token 数量（当前仅支持首个 token） | 1 |

### 工作流

```
llama.cpp 生成参考          zllama.zig 验证
─────────────────          ────────────────
ref_logits.bin  ──────→    加载模型 → 推理 → 对比
```

### 示例

```bash
# 1. 用 llama.cpp 生成参考（llama.cpp 侧）
llama-cli -m model.gguf -p "Hello" --logit-binary ref.bin

# 2. 用本工具对比
zllama-compare-llamacpp --model model.gguf --prompt "Hello" --ref-logits ref.bin

# 输出示例:
# 🟢   ✅ NMSE: 1.234e-06 (threshold: 1e-3)
# 🟢   ✅ Cosine Similarity: 0.999987 (threshold: 0.999)
# 🟢   ✅ Max Abs Error: 0.0012
# 🟢   ✅ Argmax: ours=1234, ref=1234, match=true
```

---

## 8. zllama-compare-mtmd-vision — 多模态视觉验证

**源码**: `src/tools/compare_mtmd_vision.zig`
**构建产物**: `zig-out/bin/zllama-compare-mtmd-vision`
**构建命令**: `zig build compare-mtmd-vision`

多模态视觉编码器输出质量验证。需要模型文件 + mmproj 投影器 + 图像文件。

### 用法

```bash
zllama-compare-mtmd-vision --model <model.gguf> --mmproj <mmproj.gguf> --image <image.png> --prompt <text> --ref-logits <file>
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model <path>` | 模型 GGUF 文件路径（**必需**） | — |
| `--mmproj <path>` | 多模态投影器文件（**必需**） | — |
| `--image <path>` | 输入图像文件（**必需**） | — |
| `--prompt <text>` | 提示词 | `"Describe this image"` |
| `--ref-logits <file>` | 参考 logits 文件（来自 llama.cpp mtmd）（**必需**） | — |

### 工作流

```
1. llama.cpp mtmd 生成参考:
   llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image hello.png --jinja -p ":" --logit-binary ref.bin

2. zllama.zig 验证:
   zllama-compare-mtmd-vision --model model.gguf --mmproj mmproj.gguf --image hello.png --prompt "Describe this image" --ref-logits ref.bin
```

> **注意**: 视觉编码器可能使用不同量化策略，NMSE 阈值放宽至 `1e-3`。

---

## 9. 工作流示例

### 9.1 新模型验证流程

```bash
# Step 1: 确保模型可加载且分词正常
zllama-tokenize -m new_model.gguf -p "Hello, world!" --ids --show-count

# Step 2: 导出计算图检查结构
zllama-dump-graph -m new_model.gguf -p "A" > graph.txt

# Step 3: 生成参考 logits
zllama-gen-ref -m new_model.gguf -p "The capital of France is" -o new_model_ref.bin

# Step 4: 运行推理（快速冒烟测试）
zllama -m new_model.gguf -p "The capital of France is" -n 5

# Step 5: 与 llama.cpp 参考对比
zllama-compare-llamacpp --model new_model.gguf --prompt "Hello" --ref-logits ref_from_llamacpp.bin
```

### 9.2 CI 回归测试

```bash
# 生成所有模型的参考 logits
zig build gen-ref -- --model ~/.cache/models/tinyllama-1.1b.Q4_K_M.gguf -o /tmp/ref_tiny.bin -p "Hello"
zig build gen-ref -- --model ~/.cache/models/Llama-3.2-3B.Q4_K_M.gguf -o /tmp/ref_llama.bin -p "Hello"
zig build gen-ref -- --model ~/.cache/models/gemma-4-E2B.Q4_K_M.gguf -o /tmp/ref_gemma.bin -p "Hello"
zig build gen-ref -- --model ~/.cache/models/Qwen3.5-4B.Q4_K_M.gguf -o /tmp/ref_qwen.bin -p "Hello"

# 对比新旧构建
zig build compare-logits -- --ref /tmp/ref_tiny.bin --test /tmp/ref_tiny_new.bin

# 单元测试
zig build test
```

### 9.3 常用路径参考

模型路径格式（macOS / Linux）:

```bash
~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf       # 1.1B 轻量测试
~/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf           # 3B LLaMA 3.2
~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf                  # 2B Gemma 4
~/.cache/models/Qwen3.5-4B-Q4_K_M.gguf                      # 4B Qwen 3.5
```

---

> **相关文档**: `AGENTS.md`（AI 编程入口）、`docs/TEST.md`（测试体系）
