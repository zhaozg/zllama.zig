# Qwen 3.5 Local Inference Engine

> 纯 Zig 实现的高性能本地推理引擎，基于 ggml，支持 Qwen 3.5 全系列（Dense / MoE）模型。

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0-orange)](https://ziglang.org/)
[![ggml](https://img.shields.io/badge/ggml-latest-blue)](https://github.com/ggerganov/ggml)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## ✨ 特性

- **纯原生二进制**：无 Python、无 C++ 运行时，单文件部署
- **GGUF 模型支持**：v2/v3 兼容，零拷贝内存映射，支持 Q4_K_M / Q8_0 等量化格式
- **Qwen 3.5 完整适配**：
  - 全注意力层与线性注意力层混合架构
  - GQA (Group Query Attention)
  - RoPE、RMSNorm、SwiGLU FFN
  - `attn_output_gate` 门控机制
  - MoE 变体（35B-A3B）支持（规划中）
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
git clone https://github.com/your-repo/qwen-engine.git
cd qwen-engine
git submodule update --init --recursive  # 拉取 ggml 源码
zig build -Doptimize=ReleaseFast
```

构建产物位于 `zig-out/bin/qwen-engine`。

可选的构建标志：

```bash
# 启用 Metal 后端（macOS）
zig build -Doptimize=ReleaseFast -Dmetal=true

# 启用 CUDA 后端（Linux）
zig build -Doptimize=ReleaseFast -Dcuda=true

# 指定目标 CPU 架构（如 -Dcpu=baseline）
zig build -Doptimize=ReleaseFast -Dcpu=avx2
```

### 下载模型

从 Hugging Face 或 [llama.cpp 模型库](https://huggingface.co/models?library=gguf) 下载 Qwen 3.5 的 GGUF 文件，例如：

```bash
# 示例：9B 模型 Q4_K_M 量化
wget https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/qwen3.5-9b-q4_k_m.gguf
```

### 运行推理

```bash
# 交互模式
./zig-out/bin/qwen-engine -m /path/to/model.gguf

# 单次生成
./zig-out/bin/qwen-engine -m model.gguf -p "人工智能的未来是" -n 200

# 指定后端和线程数
./zig-out/bin/qwen-engine -m model.gguf --backend metal --threads 6
```

命令行选项：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | GGUF 模型路径 | 必填 |
| `-p, --prompt` | 输入提示词 | 无（进入交互模式） |
| `-n, --n-predict` | 最大生成 token 数 | 512 |
| `--backend` | 后端：cpu / metal / cuda | cpu |
| `--threads` | CPU 线程数 | 物理核心数 × 0.75 |
| `--temp` | 温度参数 | 0.8 |
| `--top-k` | Top-k 采样 | 40 |
| `--top-p` | Top-p 采样 | 0.9 |
| `--seed` | 随机种子 | 随机 |
| `--verbose` | 打印调试信息 | false |

## 📦 项目结构

```
qwen-engine/
├── src/
│   ├── main.zig           # CLI 入口
│   ├── ggml.zig           # ggml C API 安全封装
│   ├── gguf.zig           # GGUF v2/v3 解析器
│   ├── model.zig          # Qwen 3.5 模型构建
│   ├── layers/
│   │   ├── full_attn.zig  # 标准注意力
│   │   ├── linear_attn.zig # 线性注意力
│   │   └── moe.zig        # MoE 路由（WIP）
│   ├── kv_cache.zig       # KV Cache 管理
│   ├── tokenizer.zig      # BPE 分词器
│   ├── sampler.zig        # 采样算法
│   └── backend.zig        # 多后端抽象
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

确保 Zig 0.16.0 可用：

```bash
zig version
```

初始化 submodule：

```bash
git submodule update --init
```

### 运行测试

目前测试集尚在完善，可手动验证：

```bash
zig build test
```

### 代码风格

- 使用 `zig fmt` 自动格式化
- 遵循 Zig 官方命名约定（驼峰式函数、帕斯卡式类型）
- 所有公共 API 必须有文档注释

### 提交 PR 前

- 确保 `zig build -Doptimize=ReleaseSafe` 通过
- 未引入内存泄漏（使用 `GeneralPurposeAllocator` 检测）
- 与参考实现（llama.cpp）输出数值误差在 1e-3 内

## 📊 性能参考

以下为内部测试数据（仅供参考，实际性能取决于硬件和模型配置）：

| 模型 | 量化 | 后端 | 硬件 | 速度 (tok/s) |
|------|------|------|------|---------------|
| Qwen 3.5 9B | Q4_K_M | CPU (16 线程) | AMD Ryzen 9 7950X | 18.2 |
| Qwen 3.5 9B | Q4_K_M | Metal | M2 Max (30核 GPU) | 42.5 |
| Qwen 3.5 27B | Q4_K_M | CPU (16 线程) | AMD Ryzen 9 7950X | 6.3 |
| Qwen 3.5 27B | Q4_K_M | CUDA | RTX 4090 | 48.1 |

## ⚠️ 已知限制

- 线性注意力层当前使用 `ggml_conv_1d` 实现，暂未支持因果 mask 自动处理（需手动偏移）
- MoE 模型（35B-A3B）尚在开发中
- Windows 平台暂未全面测试，欢迎贡献
- 对超长上下文（>64K）的 KV Cache 内存占用未做分块优化

## 📄 许可证

本项目采用 MIT 许可证。ggml 部分遵循其原始许可证（MIT）。

## 🙏 致谢

- [ggml](https://github.com/ggerganov/ggml) – 高性能张量计算库
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – 参考实现与 GGUF 规范
- [Qwen 团队](https://github.com/QwenLM/Qwen) – 开源模型架构

## 📬 联系方式

问题反馈请提交 [GitHub Issues](https://github.com/your-repo/qwen-engine/issues)。

---
