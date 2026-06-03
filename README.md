# zllama.zig - 多模型本地推理引擎

> 纯 Zig 实现的高性能本地推理引擎，基于 ggml，支持多模型架构（Qwen / LLaMA 等）。

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0-orange)](https://ziglang.org/)
[![ggml](https://img.shields.io/badge/ggml-latest-blue)](https://github.com/ggerganov/ggml)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## ✨ 特性

- **纯原生二进制**：无 Python、无 C++ 运行时，单文件部署
- **GGUF 模型支持**：v2/v3 兼容，零拷贝内存映射，支持 Q4_K_M / Q8_0 等量化格式
- **多模型架构**：
  - **Qwen 3.5**：全注意力层与线性注意力层混合架构，GQA，RoPE，RMSNorm，SwiGLU FFN
  - **LLaMA 2/3/3.1**：标准 Transformer 架构（框架已就绪）
  - 可扩展：新增模型只需在 `registry.zig` 注册
- **增量解码 & KV Cache**：长上下文（≥32K）内存友好，零拷贝视图
- **多后端**：CPU (默认)、Metal (macOS)、CUDA (Linux)
- **内建 BPE 分词器**：从 GGUF 提取词表，无外部依赖
- **交互式 CLI**：流式输出、采样参数可调

## 🚀 快速开始

### 前置条件

- Zig 0.16.0（推荐通过 [zigup](https://github.com/marler8997/zigup) 安装）
- Git（用于拉取 ggml submodule）
- （可选）CUDA 12.0+ / Metal 支持的 macOS

### 构建

```bash
git clone https://github.com/your-repo/zllama.zig.git
cd zllama.zig
git submodule update --init --recursive
zig build -Doptimize=ReleaseFast
```

构建产物位于 `zig-out/bin/zllama`。

### 运行推理

```bash
# 交互模式（自动检测模型架构）
./zig-out/bin/zllama -m /path/to/model.gguf

# 单次生成
./zig-out/bin/zllama -m model.gguf -p "人工智能的未来是" -n 200

# 指定线程数
./zig-out/bin/zllama -m model.gguf --threads 6
```

## 📦 项目结构

```
zllama.zig/
├── src/
│   ├── main.zig           # CLI 入口（Juicy Main）
│   ├── ggml.zig           # ggml C API 安全封装
│   ├── gguf.zig           # GGUF v2/v3 解析器
│   ├── model.zig          # 模型抽象接口定义
│   ├── kv_cache.zig       # KV Cache 管理
│   ├── tokenizer.zig      # BPE 分词器
│   ├── sampler.zig        # 采样算法
│   ├── backend.zig        # 多后端抽象
│   ├── layers/            # 通用层实现（算子库）
│   │   ├── rms_norm.zig
│   │   ├── rope.zig
│   │   ├── swiglu.zig
│   │   ├── attention.zig
│   │   ├── linear.zig
│   │   └── embed.zig
│   ├── models/            # 具体模型实现
│   │   ├── registry.zig   # 模型注册与工厂函数
│   │   ├── qwen.zig       # Qwen 系列
│   │   └── llama.zig      # LLaMA 家族
│   └── core/              # 核心引擎（可选）
├── deps/ggml/             # ggml 源码（submodule）
├── build.zig              # Zig 构建脚本
├── AGENTS.md              # AI 协作入口
├── ARCHITECTURE.md        # 系统架构设计
├── GGML_BINDING.md        # ggml 绑定设计
├── TECHNICAL_CHALLENGES.md # 难点与解决方案
└── ROADMAP.md             # 开发路线图
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
| Qwen 3.5 9B | Q4_K_M | CPU (16 线程) | AMD Ryzen 9 7950X | 18.2 |
| Qwen 3.5 9B | Q4_K_M | Metal | M2 Max (30核 GPU) | 42.5 |
| Qwen 3.5 27B | Q4_K_M | CPU (16 线程) | AMD Ryzen 9 7950X | 6.3 |

## 📄 许可证

本项目采用 MIT 许可证。ggml 部分遵循其原始许可证（MIT）。

## 🙏 致谢

- [ggml](https://github.com/ggerganov/ggml) – 高性能张量计算库
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – 参考实现与 GGUF 规范
- [Qwen 团队](https://github.com/QwenLM/Qwen) – 开源模型架构
