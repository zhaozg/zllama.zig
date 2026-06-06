# 开发路线图

> **项目名称：** zllama.zig — 纯 Zig 实现的多模型本地推理引擎

## 📋 当前状态（2026-06-06）

| 模块 | 状态 |
|------|------|
| ggml 绑定 + GGUF 解析 | ✅ 已完成 |
| CPU 多线程后端 | ✅ 已完成 |
| BPE 分词器 | ✅ 已完成 |
| Qwen2 / Qwen3.5 / LLaMA 推理 | ✅ 已完成 |
| 多模型架构（registry + layers/ + models/） | ✅ 已完成 |
| KV Cache + 增量解码 | ✅ 已完成 |
| 测试体系（GGUF / 架构 / 层 / KV Cache / 词汇表） | ✅ 已完成 |
| 调试工具（dump_graph / compare_logits / gen_ref） | ✅ 已完成 |
| 内存泄漏修复 | ✅ 已完成 |
| 推理正确性验证（tinyllama / Llama-3.2 / Qwen3.5） | ✅ 已完成 |
| 图优化（Gallocr 复用 + 输入张量缓存 + IncContext） | ✅ 已完成 |
| Metal / CUDA 后端 | ⬜ 待完成 |
| CI / 性能基准 / 生态工具 | ⬜ 待完成 |

## 🎯 里程碑

### P0-P4：核心推理路径 ✅
ggml 绑定 → GGUF 加载 → Embedding → 单层前向 → 完整推理 + KV Cache

### P5：多模型架构重构 ✅
模型抽象接口 → 共享算子库 → 注册表 → Qwen / LLaMA 迁移

### P6：Qwen3.5 混合架构 ✅
SSM 参数读取 → forwardSSM → forwardFullAttention → 层类型判断 → SSM 状态管理

### P7：Metal 后端集成 ⬜
`build.zig` 添加 `-Dmetal` → Metal 编译宏 → 后端初始化 → `--backend metal`

### P8：CUDA 后端集成 ⬜
`build.zig` 添加 `-Dcuda` → CUDA 编译宏 → 后端初始化 → `--backend cuda`
### P9：性能优化 ✅
Gallocr 复用 + 增量上下文分离 → 输入张量缓存 + 图结构复用 → `ggml_graph_plan` + 线程池 → `--benchmark` 模式 → 与 llama.cpp 对比 → 更多量化格式

### P10：生态工具 ⬜
流式 CLI → CI（GitHub Actions）→ 文档完善 → 预编译发布

---

## 已完成功能清单

### 核心引擎
- ✅ 项目结构搭建（AGENTS.md、ARCHITECTURE.md、GGML_BINDING.md、TECHNICAL_CHALLENGES.md、ROADMAP.md）
- ✅ ggml.zig 安全封装层（C 绑定、Context、Tensor、CGraph、Backend、Ops）
- ✅ GGUF v2/v3 解析器
- ✅ 模型抽象接口（model.zig）
- ✅ 模型注册与工厂函数（registry.zig）
- ✅ Qwen2 模型实现
- ✅ Qwen3.5 模型实现（含混合注意力、SSM 层）
- ✅ LLaMA 模型实现

### 图优化
- ✅ IncContext 增量上下文分离（独立 512MB 上下文，避免与 prompt 图上下文冲突）
- ✅ Gallocr 跨 token 复用（避免每 token 重复图分析，~1-5ms/次节省）
- ✅ **输入张量缓存**（参考 llama.cpp `llm_graph_input_i` 模式，预分配跨步复用）
- ✅ CGraph 每步新建（极轻量 ~100 字节，避免 ggml_graph_reset 梯度检查陷阱）
- ✅ 上下文内存自动回收（超过 80% 阈值触发 full reset）
- ✅ graph_context.zig 模块（IncContext + DecodeStep API）
- ✅ threadpool.zig 模块（ggml_threadpool Zig 封装，待 ggml 版本升级后启用）
- ✅ ggml_graph_dup 绑定（CGraph.dup 方法）

### 算子与修复
- ✅ 卷积/SSM 相关算子（conv1d、ssmConv、ssmScan、gatedDeltaNet）
- ✅ 基于词汇表的 tokenizer 测试（test_vocab.zig，18 个词汇表）
- ✅ 注意力 mask 修复（diagMaskInf 正确处理 3D 张量）
- ✅ Qwen3.5 Q/gate 交错布局修复（view_3d with interleaved stride）
- ✅ SSM 状态持久化（ctx_kv_cache 分配，不受 ctx_graph.reset() 影响）
- ✅ gdn_output view 修复（使用正确的 stride）
- ✅ RoPE 位置编码修复（buildMultiPositionTensor 布局对齐 MRoPE/IMRoPE）
- ✅ EOG 检测（tokenizer 新增 isEog() 方法和 eog_ids 集合）
- ✅ 多 prompt token 支持（sampleGreedy 正确取最后一个 token 的 logits）
- ✅ 构建脚本（build.zig，含三个可执行文件）

### 推理验证
- ✅ tinyllama 推理正确
- ✅ Llama-3.2-3B 推理正确
- ✅ Qwen3.5 SSM 层推理正确性（修复 resetSSMStates 在增量解码循环中被错误调用的问题）
- ✅ main.zig 段错误修复（缺少 setKVCacheContext 调用导致 Qwen3.5 SSM 状态被释放）
- ✅ 推理正确性验证：zllama-simple 输出与 llama-simple 对比，三种模型输出基本一致

### 测试覆盖
- ✅ GGUF 解析测试（手工构造 v2/v3 二进制数据）
- ✅ 架构检测测试（30+ 用例）
- ✅ 算子层数值测试（RMSNorm, RoPE, SwiGLU, Attention）
- ✅ KV Cache 功能测试
- ✅ 词汇表测试（18 个词汇表，基于 llama.cpp 标准测试数据）
- ✅ Logits 比较器测试
- ✅ `zig build test` 全部通过

## ⏳ 待完成

### 性能优化
- ✅ **Gallocr 复用**：增量解码使用独立 IncContext，Gallocr 跨 token 复用
- ✅ **增量上下文分离**：增量解码使用独立的 512MB ctx_inc
- ✅ **图结构复用**：输入张量预分配跨步复用，参考 llama.cpp `llm_graph_input_i` 模式；CGraph 每步新建（极轻量）；Gallocr 跨步复用；上下文内存 80% 阈值自动回收
- ⬜ **线程池**：持久化 ggml_threadpool（需 ggml 版本升级以支持线程池 API）
- ⬜ **`--benchmark` 模式**：已添加 CLI 标志，待完善输出格式

### 后端支持
- ⬜ **Metal 后端**（macOS GPU 加速）
- ⬜ **CUDA 后端**（Linux GPU 加速）

### 生态工具
- 流式 CLI
- CI（GitHub Actions）
- 性能基准测试
- 预编译发布

---

## 验证体系

### 测试金字塔

```
         ┌──────────┐
         │ 端到端    │  真实模型推理（手动）
        ┌┴──────────┴┐
        │ 架构前向   │  随机权重 + NMSE 对比（CI）
       ┌┴───────────┴┐
       │ 算子层数值  │  各算子独立测试（CI）
      ┌┴────────────┴┐
      │ 单元测试     │  函数级测试（CI）
     ┌┴─────────────┴┐
     │ 编译检查      │  zig build（CI）
     └───────────────┘
```

### 测试模块

| 文件 | 内容 |
|------|------|
| `tests/test_gguf.zig` | GGUF v2/v3 解析测试 |
| `tests/test_archs.zig` | 架构注册与检测（30+ 用例） |
| `tests/test_layers.zig` | RMSNorm / RoPE / SwiGLU 数值测试 |
| `tests/test_kv_cache.zig` | KV Cache 功能测试 |
| `tests/test_vocab.zig` | 词汇表测试（18 个词汇表） |
| `tests/test_compare_logits.zig` | Logits 对比测试 |

### 调试工具

| 工具 | 功能 |
|------|------|
| `tools/dump_graph.zig` | 计算图导出（text / dot / json） |
| `tools/compare_logits.zig` | Logits 对比（NMSE / 余弦相似度 / PSNR） |
| `tools/generate_reference.zig` | 生成参考 logits（binary / text） |

### 关键设计决策

- **注册表**使用虚表（`ModelVTable`），**模型内部**使用编译时多态
- **测试数据**：算子测试用完全随机，端到端测试用真实模型
- **NMSE 阈值**：算子 1e-5，架构前向 1e-4，与 llama.cpp 对比 1e-2

### 风险与缓解

| 风险 | 缓解 |
|------|------|
| Zig 0.16.0 测试框架不成熟 | 使用 `std.testing`，必要时封装辅助函数 |
| 浮点精度差异导致误报 | 宽松阈值（1e-2），关注趋势 |
| 测试执行时间过长 | 分层测试：快速（<1min）+ 完整（按需） |
| 模型文件过大 | 脚本生成 + git-lfs 管理参考输出 |

### 成功标准

1. `zig build` 零错误
2. `zig build test` 全部通过
3. 每个架构都有随机权重前向测试
4. 代码修改不导致已有测试失败
5. 问题能快速定位到具体层/算子
6. 相同输入始终产生相同输出

---

## 调试指南

### 日志级别

```bash
# 调试模式（显示详细日志）
zig-out/bin/zllama-simple -d -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好

# 详细模式
zig-out/bin/zllama-simple -v -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好
```

### 常见问题

1. **GraphAllocFailed**：ctx_graph 内存不足，增大 mem_size_estimate
2. **输出乱码**：tokenizer 的 decodeSingle 实现有问题，检查 BPE 解码逻辑
3. **输出为空**：采样得到的 token_id 为 0（unk）或 EOS，检查 logits 形状和采样逻辑
4. **速度慢**：已通过 Gallocr 复用 + 输入张量缓存优化
5. **Qwen3.5 输出 "0"**：SSM 层（gatedDeltaNet）计算图构建有问题，需要检查 gdn_output 的 view 和 state 管理

---

## 代码验证约束

- [x] `pub fn main(init: std.process.Init) !void`
- [x] 文件 I/O 使用 `std.Io.Dir.cwd().openFile(io, ...)`
- [x] 时间测量使用 `std.Io.Clock.now(.awake, io)`
- [x] 遇到不确定的 API 标记 TODO

## 近期优先级

1. ⬜ Metal / CUDA 后端
2. ⬜ CI + 生态工具
3. ⬜ 线程池（待 ggml 升级）
