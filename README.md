# zllama.zig - 多模型本地推理引擎

> 纯 Zig 实现的高性能本地推理引擎，基于 ggml，支持多模型架构（Qwen / LLaMA / Gemma 等），
> 支持文本生成与嵌入向量，初步支持多模态（图像/音频）。

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
- **多模态支持**（🚧 进行中）：
  - ViT 图像编码器（gemma4v / gemma4uv）
  - Conformer 音频编码器（ChunkedAttention + SSM Conv）
  - PPM 图像预处理（加载 + Resize + 标准化）

## 🚀 快速开始

### 前置条件

- **Zig 0.16.0**：`zig version` 确认版本
- **ggml submodule**：`git submodule update --init`
- （可选）CUDA 12.0+ / Metal 支持的 macOS

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
./zig-out/bin/zllama -m model.gguf --mmproj /path/to/mmproj.gguf --audio input.wav -p "转录这段音频"
```

## 📦 项目结构

```
zllama.zig/
├── src/
│   ├── main.zig                # CLI 入口（Juicy Main）
│   ├── model.zig               # Architecture 枚举、ModelVTable 接口
│   ├── gguf.zig                # GGUF 解析
│   ├── core/                   # 引擎、图构建、内存、加载器
│   │   ├── engine.zig          # InferenceEngine：统一生命周期管理
│   │   ├── engine_common.zig   # computeGraph / WallTimer / mmapFile
│   │   ├── graph_builder.zig   # GraphBuilder（文本侧）
│   │   ├── graph_context.zig   # IncContext / DecodeStep
│   │   ├── decode.zig          # runDecodeLoop / DecodeCallbacks
│   │   ├── prefill.zig         # threeStagePrefill
│   │   ├── multimodal.zig      # 多模态入口
│   │   ├── memory.zig          # MemoryContext / MemoryVTable
│   │   └── memory_pool.zig     # TempContextPool
│   ├── ggml/                   # ggml C API 安全封装
│   │   ├── mod.zig             # 重新导出
│   │   ├── c.zig               # 裸 C 函数声明
│   │   ├── context.zig         # Context 封装
│   │   ├── tensor.zig          # Tensor 封装
│   │   ├── graph.zig           # CGraph 封装
│   │   ├── backend.zig         # Backend / Gallocr / Scheduler
│   │   ├── ops.zig             # 类型安全算子封装
│   │   ├── threadpool.zig      # ThreadPool 封装
│   │   └── gguf.zig            # GGUF 格式的 ggml 层解析
│   ├── layers/                 # 共享算子（文本侧）
│   │   ├── rms_norm.zig
│   │   ├── rope.zig
│   │   ├── swiglu.zig
│   │   ├── attention.zig
│   │   ├── linear.zig
│   │   ├── embed.zig
│   │   └── pooling.zig
│   ├── models/                 # 具体模型实现
│   │   ├── registry.zig        # 模型注册与工厂函数
│   │   ├── llama.zig           # LLaMA 家族
│   │   ├── qwen2.zig           # Qwen2 系列
│   │   ├── qwen35.zig          # Qwen3.5 混合架构
│   │   ├── gemma3.zig          # Gemma 3
│   │   ├── gemma4.zig          # Gemma 4
│   │   ├── gemma4_graph.zig    # Gemma 4 图构建（buildMM）
│   │   ├── qwen3vl.zig         # Qwen3-VL 文本侧
│   │   └── embedding.zig       # 嵌入模型
│   ├── mtmd/                   # 多模态模块
│   │   ├── mod.zig             # MultiModalManager
│   │   ├── helper.zig          # evalChunks / imageGetDecoderPos
│   │   ├── preprocess.zig      # 图像/音频预处理
│   │   ├── tokenize.zig        # 文本+媒体标记混合 tokenize
│   │   ├── vision/             # 视觉编码器
│   │   │   ├── encoder.zig
│   │   │   ├── preprocess.zig
│   │   │   ├── config.zig
│   │   │   ├── loader.zig
│   │   │   └── postprocess.zig
│   │   ├── audio/              # 音频编码器
│   │   │   ├── encoder.zig
│   │   │   ├── framing.zig
│   │   │   ├── mel.zig
│   │   │   ├── log_transform.zig
│   │   │   └── mel_spectrogram.zig
│   │   └── graph/              # 多模态图构建块
│   │       ├── mod.zig         # VisionEncoderBackend / AudioEncoderBackend
│   │       ├── builder.zig     # GraphBuilder（多模态侧）
│   │       ├── attn.zig / ffn.zig / norm.zig
│   │       ├── vit.zig / patch.zig / rope.zig / merge.zig
│   │       ├── mm.zig / clamp.zig / stack.zig
│   │       ├── types.zig / debug.zig
│   │       └── models/         # 多模态模型图
│   │           ├── gemma4v.zig / gemma4a.zig
│   │           ├── gemma4uv.zig
│   │           ├── qwen2vl.zig / qwen3vl.zig
│   ├── tokenizer/              # BPE 分词器
│   │   ├── mod.zig
│   │   └── vocab.zig
│   ├── chat_template/          # 对话模板
│   │   └── mod.zig
│   ├── sampler.zig             # 采样器
│   ├── kv_cache.zig            # KV Cache
│   ├── cli_args.zig            # CLI 参数定义
│   └── tools/                  # 调试工具
│       ├── dump_graph.zig
│       ├── compare_logits.zig
│       └── generate_reference.zig
├── deps/
│   ├── ggml/                   # ggml 源码（submodule）
│   └── llama.cpp/              # 参考实现（只读）
├── build.zig / build.zig.zon   # Zig 构建
├── AGENTS.md                   # AI 协作入口
├── ROADMAP.md                  # 开发路线图
├── README.md                   # 本文件
└── docs/                       # 设计文档
    ├── ARCHITECTURE.md         # 架构设计文档
    ├── GGML_BINDING.md         # ggml 绑定设计规范
    ├── MTMD.md                 # 多模态模块详细设计
    ├── HOW_TO_ADD_NEW_MODEL.md # 新增模型指南
    ├── MEM.md                  # 内存管理详解
    └── TEST.md                 # 测试体系
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
2. 实现 `ModelVTable` 的函数指针，导出 `pub const vtable = ModelVTable{...}`
3. 在 `model.zig` 的 `Architecture` 枚举中添加新类型
4. 在 `registry.zig` 的 `createModel()` 中添加对应 case

## 📄 许可证

本项目采用 MIT 许可证。ggml 部分遵循其原始许可证（MIT）。

## 🙏 致谢

- [ggml](https://github.com/ggerganov/ggml) – 高性能张量计算库
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – 参考实现与 GGUF 规范
- [Qwen 团队](https://github.com/QwenLM/Qwen) – 开源模型架构
- [Google Gemma 团队](https://ai.google.dev/gemma) – Gemma 系列模型
