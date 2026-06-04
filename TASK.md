# zllama-simple 实现

## 对标对象: 🚶‍♂️`llama-simple` —— 专为入门设计的最小实现

根据官方文档，`llama-simple` 被定义为"一个最小实现，专门用来演示 `llama.cpp` 的基础用法模式"，是所有工具中最简单、最纯粹的一个。
它非常适合作为你 `llama.cpp` 对齐之旅的下一站。

它的使用也非常简单，一个命令就能完成从加载模型到生成文本的全过程：

```bash
llama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好
```

选择 `llama-simple` 作为下一个目标，意味着你将深入实现 `llama.cpp` 最核心的逻辑链路，包括：

1.  加载模型和分词器
2.  处理输入提示并编码为 token
3.  执行模型前向传播，获取 logits
4.  实现基本的采样逻辑（如贪心采样）
5.  将生成的 token ID 解码为文本并输出

## zllama-simple 实现目标

- [x] 创建 `src/simple_main.zig` —— zllama-simple 工具入口
- [x] 更新 `build.zig` —— 添加 zllama-simple 可执行文件构建
- [x] 支持命令行参数解析（-m, -n, -t, -k, -tp, -th, -v, -d）
- [x] 支持从命令行参数直接传入 prompt（位置参数）
- [x] 加载模型和分词器
- [x] 编码 prompt 为 token IDs
- [x] 执行模型前向传播（首 token 完整图推理 + 增量解码）
- [x] 贪心采样获取下一个 token
- [x] 解码 token IDs 为文本并输出
- [x] 支持多模型架构（Qwen / LLaMA / tinyllama）
- [x] 编译通过并成功运行
- [ ] 对齐 llama-simple 的推理输出（允许细微差异）

## 参考

- src/main.zig
- src/tokenize_main.zig
- deps/llama.cpp/examples/simple/simple.cpp
- llama-simple.log
- deps/llama.cpp/src/models/qwen35.cpp
- deps/llama.cpp/src/models/qwen35moe.cpp

### 使用方式

```bash
# 基本用法
zig-out/bin/zllama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好

# 指定生成 token 数量
zig-out/bin/zllama-simple -m ~/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -n 50 "Hello"

# 通过 zig build 运行
zig build simple -- -m ~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -n 10 "Hello"
```

## 验收条件

`zig-out/bin/zllama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好`
与 `llama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好` 的推理输出一致（允许细微差异，如采样随机性引起的输出不同，但整体语义和格式应相似）。
