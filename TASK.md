# zllama-simple 与 zllama-simple 对齐

## 参考

- src/main.zig
- src/tokenize_main.zig
- deps/llama.cpp/examples/simple/simple.cpp
- llama-simple.log
- deps/llama.cpp/src/models/qwen35.cpp
- deps/llama.cpp/src/models/qwen35moe.cpp

### 使用方式

```bash
# llama-simple
llama-simple -m ~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 你好
# zllama-simple
zig-out/bin/zllama-simple -m ~/.cache/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 你好
```

## 验收条件

`zig-out/bin/zllama-simple` 与 `llama-simple` 的推理输出一致（允许细微差异，如采样随机性引起的输出不同，但整体语义和格式应相似）。

## 禁止操作

禁止运行其他llama命令行工具。
