# 开发路线图

本文档描述 Qwen 3.5 本地推理引擎的开发计划、里程碑及长期愿景。

## 🧭 总体目标

构建一个**生产就绪、易于部署、高性能**的本地大模型推理引擎，支持 Qwen 3.5 全系列模型（Dense / MoE），并可作为未来其他 LLM 的基础框架。

测试方法:

- `zig build` pass.
- `zig-out/bin/qwen --model ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf` works.

## 📅 里程碑

### 阶段一：基础框架与 CPU 推理（当前）
**目标：** 在 x86_64 CPU 上稳定运行 Qwen 3.5 9B Q4_K_M，达到 ≥10 tok/s。

#### 已完成 ✅

- [x] 项目初始化和 `build.zig` 配置
- [x] ggml 已安装在 /usr/local 目录
- [x] `ggml.zig` 安全封装（Context, Tensor, CGraph, GgufContext）
- [x] GGUF v2 解析器（元数据 + 张量索引）
- [x] GGUF v3 支持（64 位字段、32 字节对齐）
- [x] 基础 CLI（参数解析、模型加载、信息展示）
- [x] `zig build` 编译通过 ✅
- [x] `zig build test` 测试通过 ✅
- [x] 可执行文件 `zig-out/bin/qwen` 构建成功

#### 进行中 🔄

- [ ] Qwen 3.5 基础架构搭建（全注意力层、RMSNorm、RoPE、SwiGLU）
  - `model.zig` 已实现 `buildForwardGraph`、`loadWeights`、`parseParams`
  - `layers/` 目录已创建但尚未填充具体实现
  - 需要将 `model.zig` 中的图构建与 `main.zig` 集成
- [ ] CPU 后端多线程执行
  - `ggml.zig` 已提供 `cpuNThreads()` 和 `recommendedThreads()`
  - `CGraph.compute()` 已支持多线程参数
  - 但尚未在 `main.zig` 中实际执行推理图
- [ ] 分词器（BPE）实现 + 特殊 token 处理
  - `tokenizer.zig` 已实现 `encode`、`decode`、`vocabSize`
  - 但 `main.zig` 当前未导入 tokenizer 模块
- [ ] 首 token 完整图推理
  - `model.zig` 的 `buildForwardGraph` 已实现
  - 但 `main.zig` 当前简化版未调用
- [ ] 与 llama.cpp 输出对比测试

**预计完成：** 第 1 个月末

---

### 阶段二：KV Cache 与增量推理
**目标：** 支持交互式生成，KV Cache 零拷贝管理，长上下文（32K）内存占用可控。

#### 已完成 ✅

- [x] KV Cache 预分配 + 视图切片（`kv_cache.zig` 已实现 `init`、`getKView`、`getVView`、`setKv`）
- [x] 采样器（温度、top-k、top-p）（`sampler.zig` 已实现 `sample`、`sampleGreedy`）

#### 待完成 ⬜

- [ ] 增量解码图构建（每层复用 Cache）
- [ ] 注意力拼接优化（避免每 token 复制）
- [ ] 交互式 CLI（流式输出）
- [ ] 内存使用 benchmark（32K context）

**预计完成：** 第 2 个月末

---

### 阶段三：Qwen 3.5 混合架构完整实现
**目标：** 支持全注意力和线性注意力的交替层，处理 `attn_output_gate` 等 Qwen 特有结构。

- [ ] 从 GGUF 读取 `layer_type` 数组
- [ ] 线性注意力层实现（1D 因果卷积）
- [ ] 门控机制（`attn_output_gate`）集成
- [ ] 支持 GQA（`n_kv_heads` ≠ `n_heads`）
- [ ] 验证线性注意力输出数值正确性
- [ ] 混合架构端到端测试（使用 Qwen 3.5 9B/27B）

**预计完成：** 第 3 个月末

---

### 阶段四：GPU 后端集成
**目标：** 支持 Metal (macOS) 和 CUDA (Linux)，显著提升推理速度。

- [ ] 后端抽象接口（`backend.zig`）
- [ ] CPU 后端封装为统一接口
- [ ] Metal 后端集成（`ggml-backend-metal`）
- [ ] CUDA 后端集成（`ggml-backend-cuda`）
- [ ] 自动后端检测与选择
- [ ] 设备内存分配与 Host-Device 拷贝优化
- [ ] Metal 与 CPU 性能对比测试

**预计完成：** 第 4 个月末

---

### 阶段五：生产级特性与优化
**目标：** 提高稳定性、易用性、性能上限。

- [ ] 模型热加载与卸载
- [ ] 支持超长上下文（>100K）的分块 KV Cache
- [ ] 异步 token 生成（后台执行 + 实时返回）
- [ ] OpenAI 兼容的 HTTP API 服务器
- [ ] 性能剖析与微调（SIMD 调度、内存带宽优化）
- [ ] 添加单元测试与集成测试（覆盖张量操作、采样、分词）
- [ ] Windows 平台 CI 支持
- [ ] 预编译二进制发布（GitHub Releases）

**预计完成：** 第 6 个月末

---

### 阶段六：扩展模型支持与高级特性
**目标：** 支持 Qwen 3.5 变体（35B-A3B MoE）及其他架构。

- [ ] MoE 路由实现（专家选择 + 负载均衡）
- [ ] 多模态支持（Qwen-VL 集成）
- [ ] 动态量化（部分层使用更高精度）
- [ ] LoRA 适配器热插拔
- [ ] 投机采样（Speculative Decoding）加速
- [ ] 支持其他模型架构（如 LLaMA 3、Mistral）作为编译时选项

**预计完成：** 第 9 个月末

---

## 📊 关键性能指标 (KPI)

| 指标 | 目标值 |
|------|--------|
| **首次生成延迟**（首 token） | < 200ms（9B Q4, CPU） |
| **增量生成速度** | ≥15 tok/s（9B Q4, CPU 16线程）<br>≥5 tok/s（27B Q4, CPU 16线程）<br>≥40 tok/s（9B Q4, RTX 4090） |
| **内存占用**（KV Cache 32K） | < 8GB（9B Q4）<br>< 16GB（27B Q4） |
| **模型加载时间** | < 5 秒（9B GGUF） |
| **跨平台覆盖率** | Linux (x86_64, aarch64), macOS (arm64, x86_64), Windows (x86_64) |

## 🔄 版本规划

### v0.1.0 – 基础 CPU 推理（阶段一、二）
- 支持 9B Q4_K_M CPU 推理
- 交互式 CLI
- GGUF v2 兼容

### v0.2.0 – 混合架构支持（阶段三）
- 完整 Qwen 3.5 9B/27B 功能
- 线性注意力与门控机制

### v0.3.0 – GPU 加速（阶段四）
- Metal / CUDA 后端
- 性能提升 3-5 倍

### v1.0.0 – 生产稳定版（阶段五）
- HTTP API 服务器
- 完善的测试与文档
- Windows 支持

### v1.1.0 – 高级特性（阶段六）
- MoE 模型支持
- 投机采样
- 多模态（可选）

## 🤝 贡献指南

欢迎贡献！我们寻找以下领域的帮助：

- **测试与验证**：在不同硬件上运行 benchmark，报告问题
- **性能优化**：分析 ggml 算子瓶颈，提出改进建议
- **文档**：完善使用指南、API 文档
- **平台移植**：帮助适配 Windows、FreeBSD 等
- **新模型适配**：验证其他模型架构在引擎上的兼容性

请参考 `AGENTS.md` 了解 AI 协作方式，或直接提交 PR。

## 📝 变更日志

详细的版本更新记录见 [CHANGELOG.md](CHANGELOG.md)（待创建）。

---

*本路线图将根据项目进展和社区反馈动态调整。*
