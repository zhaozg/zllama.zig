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
| 测试体系（GGUF / 架构 / 层 / KV Cache） | ✅ 已完成 |
| 调试工具（dump_graph / compare_logits / gen_ref） | ✅ 已完成 |
| 内存泄漏修复 | ✅ 已完成 |
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

### P9：性能优化 ⬜
`ggml_graph_plan` + 线程池 → `--benchmark` 模式 → 与 llama.cpp 对比 → 更多量化格式

### P10：生态工具 ⬜
流式 CLI → CI（GitHub Actions）→ 文档完善 → 预编译发布

---

## 验证体系（源自 PLAN.md）

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

## 代码验证约束

- [x] `pub fn main(init: std.process.Init) !void`
- [x] 文件 I/O 使用 `std.Io.Dir.cwd().openFile(io, ...)`
- [x] 时间测量使用 `std.Io.Clock.now(.awake, io)`
- [x] 遇到不确定的 API 标记 TODO

## 近期优先级

1. ⬜ Metal / CUDA 后端
2. ⬜ 性能优化与基准测试
3. ⬜ CI + 生态工具
