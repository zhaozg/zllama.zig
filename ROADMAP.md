# 开发路线图

> **项目名称：** zllama.zig — 纯 Zig 实现的多模型本地推理引擎

## 📋 当前状态（2026-06-08）

| 模块 | 状态 |
|------|------|
| ggml 绑定 + GGUF 解析 | ✅ 已完成 |
| CPU 多线程后端 | ✅ 已完成 |
| BPE 分词器 | ✅ 已完成 |
| Qwen2 / Qwen3.5 / LLaMA 推理 | ✅ 已完成 |
| Gemma 3 推理 | ✅ 已完成（已验证 270m Q8_0） |
| Gemma 4 推理 | ✅ 已完成（已验证 E2B Q4_K_M，内存估算修复） |
| 多模型架构（registry + layers/ + models/） | ✅ 已完成 |
| KV Cache per-layer 可变维度（n_kv_head/head_dim 逐层自适应） | ✅ 已完成 |
| 交互式聊天模式（-c/--chat） | ✅ 已完成 |
| 测试体系（GGUF / 架构 / 层 / KV Cache / 词汇表） | ✅ 已完成 |
| 调试工具（dump_graph / compare_logits / gen_ref） | ✅ 已完成 |
| 推理正确性验证（tinyllama / Llama-3.2 / Qwen3.5 / Gemma3 / Gemma4） | ✅ 已完成 |
| 图优化（Gallocr 复用 + 输入张量缓存 + IncContext） | ✅ 已完成 |
| `--benchmark` 模式（PP/TG 分离 + 格式化输出） | ✅ 已完成 |
| `-Dbundle-ggml` 源码构建 | ✅ 已完成 |
| 多模态支持（音频/图像输入） | ⬜ 规划中 |
| Metal / CUDA 后端 | ⬜ 待完成 |
| CI / 性能基准 / 生态工具 | ⬜ 待完成 |

## 🎯 里程碑

### P0-P4：核心推理路径 ✅
ggml 绑定 → GGUF 加载 → Embedding → 单层前向 → 完整推理 + KV Cache

### P5：多模型架构重构 ✅
模型抽象接口 → 共享算子库 → 注册表 → Qwen / LLaMA 迁移

### P6：Qwen3.5 混合架构 ✅
SSM 参数读取 → forwardSSM → forwardFullAttention → 层类型判断 → SSM 状态管理

### P7：Gemma 3/4 支持 ✅
Gemma 3 推理已验证 → Gemma 4 推理已验证 (E2B Q4_K_M)

### P8：Metal 后端集成 ⬜
`build.zig` 添加 `-Dmetal` → Metal 编译宏 → 后端初始化 → `--backend metal`

### P9：CUDA 后端集成 ⬜
`build.zig` 添加 `-Dcuda` → CUDA 编译宏 → 后端初始化 → `--backend cuda`

### P10：性能优化 ✅
Gallocr 复用 + 增量上下文分离 → 输入张量缓存 + 图结构复用 → `ggml_graph_plan` + 线程池 → `--benchmark` 模式 → 与 llama.cpp 对比 → 更多量化格式

### P11：生态工具 ⬜
流式 CLI → CI（GitHub Actions）→ 文档完善 → 预编译发布

### 多模态支持 ⬜
- ⬜ 音频编码器集成（需升级 ggml 以支持 `mtmd-audio.c` 中的算子）
- ⬜ 图像编码器集成（可选，按需）
- ⬜ 多模态输入缓存与图优化
- ⬜ CLI 交互式多模态输入（`-a/--audio`, `-i/--image`）
- ⬜ 与现有模型架构（Gemma 4 E2B）联调

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
- ✅ Gemma 3 模型实现（混合 SWA/Full Attention、Q/K pre-norm、logit softcapping）
- ✅ Gemma 4 模型实现（per-layer head_dim、SWA/Full Attention 混合、shared KV、GeGLU FFN）
- ✅ 交互式聊天模式（`-c/--chat`，流式对话）

### 内存与 KV Cache
- ✅ KV Cache per-layer 可变维度（Gemma 4 各层 n_kv_head/head_dim 不同，LayerCache 逐层存储实际维度）
- ✅ setKv 自动适配 per-layer 维度差异（使用实际张量维度创建视图）
- ✅ Gemma 4 内存估算修复（改用 `gguf_file.totalTensorDataSize()` 精确计算，替代启发式 0.6 bytes/elem 估算）

### 图优化
- ✅ IncContext 增量上下文分离（独立 512MB 上下文，避免与 prompt 图上下文冲突）
- ✅ Gallocr 跨 token 复用（避免每 token 重复图分析，~1-5ms/次节省）
- ✅ **输入张量缓存**（参考 llama.cpp `llm_graph_input_i` 模式，预分配跨步复用）
- ✅ CGraph 每步新建（极轻量 ~100 字节，避免 ggml_graph_reset 梯度检查陷阱）
- ✅ 上下文内存自动回收（超过 80% 阈值触发 full reset）
- ✅ graph_context.zig 模块（IncContext + DecodeStep API）
- ✅ threadpool.zig 模块（ggml_threadpool Zig 封装，待 ggml 版本升级后启用）
- ✅ ggml_graph_dup 绑定（CGraph.dup 方法）
- ✅ 构建脚本（build.zig）

### 算子与修复
- ✅ 卷积/SSM 相关算子（conv1d、ssmConv、ssmScan、gatedDeltaNet）
- ✅ 注意力 cont4d 修复（显式 cont4d 避免 ggml_is_transposed 边界情况）
- ✅ 注意力 mask 修复（diagMaskInf 正确处理 3D 张量）
- ✅ Qwen3.5 Q/gate 交错布局修复（view_3d with interleaved stride）
- ✅ SSM 状态持久化（ctx_kv_cache 分配，不受 ctx_graph.reset() 影响）
- ✅ gdn_output view 修复（使用正确的 stride）
- ✅ RoPE 位置编码修复（buildMultiPositionTensor 布局对齐 MRoPE/IMRoPE）
- ✅ EOG 检测（tokenizer 新增 isEog() 方法和 eog_ids 集合）
- ✅ 多 prompt token 支持（sampleGreedy 正确取最后一个 token 的 logits）
- ✅ GGUF v3 扩展元数据解析（rope_freq_base_swa、attention_softcapping 等）
- ✅ 图构建 forward_expand API（避免 CGraph 重复初始化）
- ✅ ropeExt 绑定（支持 freq_base 和 freq_scale 参数）

### 推理验证
- ✅ tinyllama 推理正确
- ✅ Llama-3.2-3B 推理正确
- ✅ Qwen3.5 SSM 层推理正确性
- ✅ Gemma 3 270m Q8_0 推理正确
- ✅ Gemma 4 E2B Q4_K_M 推理正确（内存估算修复后加载成功）

### 测试覆盖
- ✅ GGUF 解析测试
- ✅ 架构检测测试（30+ 用例）
- ✅ 算子层数值测试（RMSNorm, RoPE, SwiGLU, Attention）
- ✅ KV Cache 功能测试（含 per-layer 维度验证）
- ✅ 词汇表测试（18 个词汇表）
- ✅ Logits 比较器测试
- ✅ `zig build test` 全部通过

## ⏳ 待完成

### 性能优化
- ✅ **Gallocr 复用**：增量解码使用独立 IncContext，Gallocr 跨 token 复用
- ✅ **增量上下文分离**：增量解码使用独立的 512MB ctx_inc
- ✅ **图结构复用**：输入张量预分配跨步复用
- ⬜ **线程池**：持久化 ggml_threadpool（需 ggml 版本升级）
- ✅ **`--benchmark` 模式**：PP/TG 时间分离 + 格式化 benchmark 输出
- ✅ **`-Dbundle-ggml` 源码构建**

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
         ┌────────┐
         │ 端到端 │  真实模型推理（手动）
        ┌┴────────┴┐
        │ 架构前向 │  随机权重 + NMSE 对比（CI）
       ┌┴──────────┴┐
       │ 算子层数值 │  各算子独立测试（CI）
      ┌┴────────────┴┐
      │ 单元测试     │  函数级测试（CI）
     ┌┴──────────────┴┐
     │ 编译检查       │  zig build（CI）
     └────────────────┘
```

### 调试工具

| 工具 | 功能 |
|------|------|
| `tools/dump_graph.zig` | 计算图导出（text / dot / json） |
| `tools/compare_logits.zig` | Logits 对比（NMSE / 余弦相似度 / PSNR） |
| `tools/generate_reference.zig` | 生成参考 logits（binary / text） |

---

## 调试指南

### 日志级别

```bash
# 调试模式（显示详细日志）
zig-out/bin/zllama -d -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf

# 详细模式
zig-out/bin/zllama -v -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf
```

### 常见问题

1. **GraphAllocFailed**：ctx_graph 内存不足，增大 mem_size_estimate
2. **输出乱码**：tokenizer 的 decodeSingle 实现有问题，检查 BPE 解码逻辑
3. **输出为空**：采样得到的 token_id 为 0（unk）或 EOS，检查 logits 形状和采样逻辑
4. **加载崩溃（GGML_ASSERT obj_new failed）**：内存估算不足，需使用 `totalTensorDataSize()` 精确计算
5. **Gemma 4 KV Cache 维度不匹配**：各层 n_kv_head/head_dim 不同，需用 per-layer 实际维度创建视图

---

## 近期优先级

1. ⬜ Metal / CUDA 后端
2. ⬜ CI + 生态工具
3. ⬜ 线程池（待 ggml 升级）
