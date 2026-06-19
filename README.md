# zllama.zig - 多模型本地推理引擎

> 纯 Zig 实现的高性能本地推理引擎，基于 ggml，支持多模型架构（Qwen / LLaMA / Gemma 等），支持文本生成与嵌入向量，初步支持多模态（图像/音频）。

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0-orange)](https://ziglang.org/)
[![ggml](https://img.shields.io/badge/ggml-latest-blue)](https://github.com/ggerganov/ggml)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## ✨ 特性

- **纯原生二进制**：无 Python、无 C++ 运行时，单文件部署
- **GGUF 模型支持**：v2/v3 兼容，零拷贝内存映射，支持 Q4_K_M / Q8_0 等量化格式
- **多模型架构**：
  - **LLaMA 2/3/3.1**：标准 Transformer 架构
  - **Qwen 2/2.5**：标准 Transformer + GQA
  - **Qwen 3.5**：全注意力层与线性注意力（SSM/GDN）混合架构
  - **Gemma 3**：SWA/Full Attention 混合、Q/K pre-norm、logit softcapping
   - **Gemma 4**：per-layer head_dim、SWA/Full Attention 混合、shared KV、GeGLU FFN
   - **Qwen3-Embedding**：双向注意力 + mean/cls/last 池化 + L2 归一化
  - 可扩展：新增模型只需在 `registry.zig` 注册
- **增量解码 & KV Cache**：长上下文（≥32K）内存友好，支持 per-layer 可变维度
- **多后端**：CPU (默认)、Metal (macOS)、CUDA (Linux)
- **内建 BPE 分词器**：从 GGUF 提取词表，无外部依赖
- **交互式聊天模式**：`-c/--chat` 流式对话、采样参数可调
- **嵌入向量生成**：`--embed` 模式，池化策略可配，支持文本嵌入与语义搜索
- **Benchmark 模式**：`--benchmark` 输出 PP/TG 分离的性能数据
- **Benchmark 模式**：`--benchmark` 输出 PP/TG 分离的性能数据
- **多模态支持**（🚧 进行中）：
  - ViT 图像编码器（gemma4v / gemma4uv）
  - Conformer 音频编码器（ChunkedAttention + SSM Conv）
  - PPM 图像预处理（加载 + Resize + 标准化）

## 🚀 快速开始

### 前置条件

- **Benchmark 模式**：`--benchmark` 输出 PP/TG 分离的性能数据
- （可选）CUDA 12.0+ / Metal 支持的 macOS

- **多模态支持**（🚧 进行中）：
  - ✅ 音频：Conformer 编码器 + Mel 频谱 → LLM（Gemma 4 E2B 已验证）
  - 🚧 视觉：ViT 图像编码器 → LLM（pipeline 就绪，输出内容待验证）
  - WAV/PPM 预处理（加载 + Resize + 标准化）

### 运行推理

```bash
# 交互聊天模式（自动检测模型架构）
./zig-out/bin/zllama -c -m /path/to/model.gguf

# 单次生成
./zig-out/bin/zllama -m model.gguf -p "人工智能的未来是" -n 200

# 嵌入向量生成
./zig-out/bin/zllama --embed -m Qwen3-Embedding-0.6B-Q8_0.gguf -p "你好世界"

# 指定线程数
./zig-out/bin/zllama -m model.gguf --threads 6

# Benchmark 模式
./zig-out/bin/zllama -m model.gguf --benchmark

# 多模态（需要 --mmproj 投影器文件）
./zig-out/bin/zllama -m model.gguf --mmproj /path/to/mmproj.gguf --image input.ppm -p "描述这张图片"
./zig-out/bin/zllama -m model.gguf --mmproj /path/to/mmproj.gguf --audio input.pcm -p "转录这段音频"
```

## 📦 项目结构

```
zllama.zig/
├── src/
│   ├── main.zig           # CLI 入口（Juicy Main）
│   ├── simple_main.zig    # 简化推理入口
│   ├── ggml.zig           # ggml C API 安全封装 + Tensor 方法式算子
│   ├── gguf.zig           # GGUF v2/v3 解析器
│   ├── model.zig          # 模型抽象接口定义
# 多模态图像（需要 --mmproj 投影器文件）
./zig-out/bin/zllama -m model.gguf --mmproj /path/to/mmproj.gguf --image input.ppm -p "描述这张图片"
# 多模态音频（WAV 格式）
./zig-out/bin/zllama -m model.gguf --mmproj /path/to/mmproj.gguf --audio input.wav -p "转录这段音频"
│   ├── layers/            # 通用层实现（算子库）
│   │   ├── rms_norm.zig
│   │   ├── rope.zig
│   │   ├── swiglu.zig
│   │   ├── attention.zig
│   │   ├── linear.zig
│   │   └── embed.zig
│   ├── models/            # 具体模型实现
│   │   ├── registry.zig   # 模型注册与工厂函数
│   │   ├── qwen2.zig      # Qwen2 系列
│   │   ├── qwen35.zig     # Qwen3.5 混合架构
│   │   ├── llama.zig      # LLaMA 家族
│   │   ├── gemma3.zig     # Gemma 3
│   │   └── gemma4.zig     # Gemma 4
│   ├── core/              # 核心引擎
│   │   ├── graph_builder.zig
│   │   ├── graph_context.zig
│   │   └── memory.zig
│   ├── mm/                # 多模态模块
│   │   ├── manager.zig    # 多模态调度器（MMProj 加载）
│   │   ├── vision.zig     # ViT 视觉编码器
│   │   ├── audio.zig      # Conformer 音频编码器
│   │   └── preprocess.zig # 图像/音频预处理
│   ├── ggml/              # ggml 安全封装子模块
│   └── tools/             # 调试工具
│       ├── dump_graph.zig
│       ├── compare_logits.zig
│       └── generate_reference.zig
├── deps/ggml/             # ggml 源码（submodule）
├── build.zig              # Zig 构建脚本
├── AGENTS.md              # AI 协作入口
├── ROADMAP.md             # 开发路线图
├── README.md              # 本文件
└── docs/                  # 设计文档
```

## 🔧 开发与贡献

### 环境搭建

```bash
zig version  # 确保 Zig 0.16.0
git submodule update --init
```

### 运行测试

```bash
zig build test
```

### 扩展新模型

1. 在 `src/models/` 下创建新文件（如 `mistral.zig`）
2. 实现 `init`、`deinit`、`forward`、`params`、`weights` 方法
3. 在 `model.zig` 的 `Architecture` 枚举中添加新类型
4. 在 `registry.zig` 的各个 switch 中添加对应 case

## 📊 性能参考

| 模型 | 量化 | 后端 | 硬件 | 速度 (tok/s) |
|------|------|------|------|---------------|
| Llama-3.2-3B | Q4_K_M | CPU (6 线程) | Apple M2 | 16.1 TG |
| Qwen3.5-0.8B | Q4_K_M | CPU (6 线程) | Apple M2 | 56.7 TG |
| tinyllama-1.1B | Q4_K_M | CPU (6 线程) | Apple M2 | 53.8 TG |
| Gemma 4 E2B | Q4_K_M | CPU (6 线程) | Apple M2 | 待测 |

## 已知问题:

- `zig build test -Doptimize=ReleaseSafe -Dbundle-ggml=true` 出现非法指令错误

## 📄 许可证

本项目采用 MIT 许可证。ggml 部分遵循其原始许可证（MIT）。

## 🙏 致谢

- [ggml](https://github.com/ggerganov/ggml) – 高性能张量计算库
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – 参考实现与 GGUF 规范
- [Qwen 团队](https://github.com/QwenLM/Qwen) – 开源模型架构
- [Google Gemma 团队](https://ai.google.dev/gemma) – Gemma 系列模型
