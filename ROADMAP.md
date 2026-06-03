# 开发路线图

> **项目名称：** zllama.zig — 纯 Zig 实现的多模型本地推理引擎
>
> **重要声明：** 由于 Zig 0.16.0 较新，部分 API 用法需要在开发过程中逐步验证。遇到无法确认的内容将标记为 **「需查证」** ，而非编造实现。

## 📋 当前状态（2026-06-03）

| 模块 | 状态 | 验证方式 |
|------|------|---------|
| `build.zig` 集成 ggml 源码 | ✅ 已完成 | `zig build` 编译通过 |
| ggml 基础绑定 | ✅ 已完成 | `zig build test` pass |
| GGUF 解析器（v2/v3） | ✅ 已完成 | 能加载 GGUF 文件并打印 tensor 信息 |
| CPU 多线程后端 | ✅ 已完成 | `ggml_backend_cpu_init` 可用 |
| 词表提取（BPE） | ✅ 已完成 | 可从 GGUF 提取 tokenizer.json 格式 |
| 完整 Qwen 前向推理 | ✅ 已完成 | P0-P4 核心目标全部达成 |
| Qwen 3.5 0.8B 可运行 | ✅ 已完成 | `zllama --model ...` works |
| **多模型架构重构** | ✅ **已完成** | 新增 `models/`、`layers/` 目录结构 |
| **模型抽象接口** | ✅ **已完成** | `model.zig` 定义 Architecture 枚举和接口 |
| **模型注册表** | ✅ **已完成** | `registry.zig` 工厂函数 |
| **共享算子库** | ✅ **已完成** | `layers/` 下 6 个算子模块 |
| **Qwen 模型迁移** | ✅ **已完成** | `models/qwen.zig` |
| **LLaMA 模型框架** | ✅ **已完成** | `models/llama.zig` |
| 与 llama.cpp 对比测试 | ⬜ 待完成 | |
| 线性注意力层完整实现 | ⬜ 待完成 | |
| Metal 后端集成 | ⬜ 待完成 | |
| CUDA 后端集成 | ⬜ 待完成 | |

## 代码验证约束（适用于所有开发阶段）

**CI 测试必须包含以下验证**（确保 Zig 0.16.0 规范性）：
- [x] main.zig 使用 `pub fn main(init: std.process.Init) !void` 签名
- [x] 所有文件 I/O 使用 `std.Io.Dir.cwd().openFile(io, ...)` 而非 `std.fs.cwd()`
- [x] 所有时间测量使用 `std.Io.Clock.now(.awake, io).durationTo(...)` 而非旧的 `std.time.Timer`
- [x] 避免任何假设已知的 API 用法，遇到不确定的必须标记 TODO

## 🎯 分阶段里程碑

### 里程碑 P0：ggml 绑定与构建验证 ✅

**目标：** `zig build` 能编译通过，能调用 `ggml_version()`。

- [x] `build.zig` 添加 ggml .c 源文件列表
- [x] 通过 `b.addCSourceFiles` 配置编译宏
- [x] `ggml.zig` 定义 `ggml_init`, `ggml_free`, `ggml_version`
- [x] 创建测试验证 binding
- [x] `zig build test` 通过 ✅

### 里程碑 P1：GGUF 文件加载 ✅

**目标：** 能够完整读取 GGUF 文件格式，解析元数据和张量索引。

- [x] 实现 GGUF header 解析（magic, version）
- [x] 支持 v2（32 位字段）和 v3（64 位字段）
- [x] 解析 metadata KV 对
- [x] 解析 tensor infos
- [x] 实现 `validate()` 检查版本兼容性和对齐要求
- [x] CLI 添加 `--info` 选项

### 里程碑 P2：Embedding 层验证 ✅

**目标：** 将嵌入层张量加载到 ggml，输入一个 token id 获得其词向量。

- [x] 扩展 `gguf.zig` 支持通过 offset 和大小读取张量数据
- [x] `model.zig` 实现 `loadEmbedding`
- [x] 实现 `forwardEmbedding`

### 里程碑 P3：单 Transformer 层前向 ✅

**目标：** 完成一个 Transformer 层的完整前向计算。

- [x] 实现 `loadLayer`
- [x] 实现 `forwardRmsNorm`
- [x] 实现 `forwardFullAttention` 支持 GQA
- [x] 实现 `forwardSwiGLU`
- [x] 实现 `forwardLayer`

### 里程碑 P4：完整推理 + KV Cache ✅

**目标：** 多 token 生成（增量解码），带 KV Cache。

- [x] 实现 `kv_cache.zig`
- [x] 实现位置编码 RoPE
- [x] 实现 `forwardGraph`
- [x] 实现 `generate`
- [x] 支持 GQA
- [x] `zllama --model ... --prompt` 能生成连续文本

### 里程碑 P5：多模型架构重构 ✅

**目标：** 支持多模型架构（Qwen / LLaMA 等）。

- [x] 定义模型抽象接口 `model.zig`
- [x] 创建共享算子库 `layers/`
- [x] 创建模型注册表 `registry.zig`
- [x] 迁移 Qwen 实现到 `models/qwen.zig`
- [x] 创建 LLaMA 模型框架 `models/llama.zig`
- [x] 更新 `main.zig` 使用多模型架构
- [x] 更新所有项目文档

## 🔄 扩展里程碑（阶段 P6+ 可独立进行）

### 里程碑 P6：Qwen 3.5 混合架构完整实现

- [ ] 从 GGUF 元数据读取 `layer_type` 数组
- [ ] 实现 `forwardLinearAttention`（使用 `ggml_conv_1d` 实现 1D 因果卷积）
- [ ] 实现 attn_output_gate 门控机制（若存在）
- [ ] 端到端测试通过

### 里程碑 P7：Metal 后端集成

- [ ] `build.zig` 添加 `-Dmetal=true` 选项
- [ ] 添加 Metal 编译宏 `GGML_USE_METAL`
- [ ] 在 `ggml.zig` 中添加 Metal 后端初始化函数
- [ ] 实现后端选择逻辑（`--backend metal`）

### 里程碑 P8：CUDA 后端集成

- [ ] `build.zig` 添加 `-Dcuda=true` 选项
- [ ] 添加 CUDA 编译宏 `GGML_USE_CUDA`
- [ ] CUDA 后端初始化与内存分配
- [ ] `--backend cuda` 可用

### 里程碑 P9：性能优化与基准测试

- [ ] 实现 `ggml_graph_plan` + 线程池优化
- [ ] 添加 `--benchmark` 模式，输出 tok/s、内存占用等指标
- [ ] 与 llama.cpp 同模型同硬件对比测试
- [ ] 支持更多量化格式（Q5_K_M, Q6_K, Q8_0）

### 里程碑 P10：生态与工具链

- [ ] 交互式 CLI 增强（流式输出、对话模式、多轮记忆）
- [ ] 添加 CI（GitHub Actions，编译 + 测试）
- [ ] 文档完善（API 参考、模型兼容性列表）
- [ ] 提供预编译二进制发布

## ✅ 已确认（无需查证）

| 主题 | 确认结论 | 来源 |
|------|---------|------|
| 时间测量 API | `std.time.Timer` 已移除，改用 `std.Io.Clock.now(.awake, io).durationTo(...)` | Zig 0.16.0 Release Notes |
| Juicy Main | main 函数接受 `std.process.Init` 参数 | Zig 0.16.0 Release Notes |
| I/O 接口化 | 所有阻塞 I/O 必须通过 Io 实例 | Zig 0.16.0 Release Notes |

## 📅 近期计划

| 任务 | 状态 |
|------|------|
| ✅ 完成 P0-P5（核心路径 + 多模型重构） | ✅ |
| ⬜ 里程碑 P6：Qwen 3.5 混合架构完整实现 | 待定 |
| ⬜ 里程碑 P7：Metal 后端集成 | 待定 |
| ⬜ 里程碑 P8：CUDA 后端集成 | 待定 |
| ⬜ 里程碑 P9：性能优化与基准测试 | 待定 |
| ⬜ 里程碑 P10：生态与工具链 | 待定 |

**当前优先级：** 多模型架构重构已完成，下一步可推进 P6（混合架构完整实现）或 P7/P8（GPU 后端）。

**说明：** 本文档会随开发进展持续更新，每次合并 PR 时应同步更新对应的里程碑状态。
