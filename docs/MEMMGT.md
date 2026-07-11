# zllama.zig 内存管理与优化指南（修订版）

> **目标**：构建可预测、可扩展、高效的内存体系，支撑多模态大模型推理。
> **适用版本**：基于 Zig 0.16.0 + ggml v0.16，zllama.zig 当前主分支。

---

## 1. 内存使用热点与瓶颈分析

### 1.1 典型工作负载内存构成

| 组件 | 典型规格（Gemma 4 E2B） | 估算大小 | 生命周期 |
|------|--------------------------|----------|-----------|
| **模型权重** | 4-bit 量化（Q4_K_M），约 6.7B 参数 | ~4.0 GB | 进程生命周期（只读） |
| **KV Cache** | 35 层，max_seq_len=131072，head_dim=128，K/V 各 f16 | 35×131072×128×2×2字节 = ~2.3 GB（但实际根据上下文增长） | 会话生命周期（可扩展） |
| **mmproj 多模态权重** | 视觉/音频编码器权重（f16） | ~0.5 GB | 进程生命周期（只读） |
| **输入嵌入** | 文本 token + 媒体 token（如 784×1536 f32） | 784×1536×4 ≈ 4.8 MB | 临时（仅当前 Prefill） |
| **中间激活** | 每层 Q/K/V、注意力分数、FFN 中间值等 | 高峰可达 **2–3 GB** | 临时（单次计算图执行） |
| **ggml_context 元数据** | 张量描述、节点指针等 | 数十 MB | 随 context 生命周期 |

**关键结论**：**中间激活**是内存波动的最大来源，尤其在多模态长序列（如图像 token 数量大）和非因果注意力场景下，峰值可能超过模型权重或 KV Cache。

### 1.2 当前面临的主要问题

1. **ggml_context 固定尺寸**：当前 `Graph context` 硬编码为 2048 MB（见 `src/core/engine.zig`），在纯文本短序列时够用，但在多模态 Prefill 阶段（如 784 个图像 token + 35 层 transformer）所需临时张量超过 2.14 GB，导致 `ggml_new_object` 分配失败（如 Issue #XXX 所示）。

2. **三阶段 Prefill 共用同一个 context**：Pass 1（文本前缀）的中间张量虽被重置，但底层内存池容量并未释放，导致 Pass 2 可用空间因碎片或残留占用而小于总容量。

3. **缺乏峰值测量与动态调整**：未使用 `ggml_gallocr_alloc_graph` 的 `measure` 模式来准确预测所需内存，无法在运行时调整 context 大小。

4. **持久与临时张量混用**：模型权重、位置编码等持久数据与每层的激活张量分配在同一 context 中，加剧了临时可用空间的不足。

5. **`IncContext.reset()` 不彻底**：仅重置分配指针，不回收内存池，无法应对大临时张量场景。

---

## 2. 已完成的内存优化（现状评估）

以下优化已实施，可继续保留：

| 优化项 | 位置 | 说明 |
|--------|------|------|
| **mmap 加载模型** | `src/core/engine_common.zig` | 零拷贝加载 GGUF，减少内存占用和启动时间 |
| **Arena 管理 init 阶段** | `src/core/engine.zig` | 使用 ArenaAllocator 管理临时分配，初始化后自动释放 |
| **KV Cache 预分配** | `src/kv_cache.zig` | 按最大序列长度预先分配，避免动态扩容开销 |
| **WallTimer** | `src/core/engine_common.zig` | 使用 Zig 0.16 的 `std.Io.Clock` 实现高性能计时 |
| **延迟加载多模态** | `src/core/engine.zig` | 仅在需要时加载 mmproj 编码器 |
| **线程池与后端调度** | `src/ggml/backend.zig` | 多线程计算图调度 |


---

## 3. 设计原则与目标

### 3.1 核心设计原则

- **可测量性**：任何计算图构建前必须能准确预测内存需求。
- **弹性分配**：context 大小应随输入动态调整，而非硬编码。
- **关注点分离**：持久数据（权重、KV）与临时数据（激活）使用独立的分配器。
- **可回收性**：确保 `deinit` 能完全释放所有内存，支持进程内模型热切换。
- **零欠载**：绝不因内存不足而崩溃（通过测量 + fallback 机制）。

### 3.2 架构图（新增内存管理层）

```
┌───────────────────────────────────────────────────────────────┐
│                        InferenceEngine                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐  │
│  │ 权重 Context  │  │  KV Cache     │  │  临时 Context池   │  │
│  │ (只读, mmap)  │  │ (独立分配器)  │  │ (可重置/可扩容)   │  │
│  └───────────────┘  └───────────────┘  └───────────────────┘  │
│                                                               │
│  每个 Prefill/Decode 阶段：                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  1. 构建计算图（无分配）                                │  │
│  │  2. ggml_gallocr_measure() → 获得所需字节数             │  │
│  │  3. 若 > 当前临时 context 容量 → 扩展或切换             │  │
│  │  4. 分配并执行                                          │  │
│  │  5. 释放临时 context（或 reset）                        │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. 改进方案（分阶段实施）

### 4.1 短期修复（立即生效，解决崩溃）

| 措施 | 具体操作 | 预期效果 |
|------|----------|----------|
| **增大默认 Graph context** | 将 `src/core/engine.zig` 中 `graph_memory_mb` 从 2048 提升至 4096（或根据 `max_seq_len` 动态计算：`max_seq_len * n_embd * 4`） | 暂时避免多模态崩溃，但浪费内存 |
| **Pass 前强制重置 context** | 在 `threeStagePrefill` 的每个 Pass 之前调用 `ggml_context_reset` 并重新创建必要持久张量 | 减少碎片，提高可用空间 |
| **增加可用空间余量** | 在 `ggml_new_object` 失败前打印 context 使用详情（实现 `ggml_context_usage` 辅助函数） | 便于调试 |

> **注意**：这些是临时措施，长期需采用下列重构。

### 4.2 中期重构（核心设计调整）

#### 4.2.1 引入计算图内存测量

- 在 `src/ggml/graph.zig` 中封装 `measureGraph(graph: *CGraph) !usize`，调用 `ggml_gallocr_alloc_graph` 的测量模式（通过设置 `ggml_allocr_is_measure`）。
- 在 `src/core/prefill.zig` 和 `src/core/decode.zig` 中，在构建图后、分配前先测量，若需求超过当前 context 容量，则动态扩展或创建新 context。

#### 4.2.2 为每个 Prefill 阶段分配独立 context

- 在 `threeStagePrefill` 中，为 Pass 1、Pass 2、Pass 3 分别创建临时 `ggml_context`（通过 `ggml_context_init`），Pass 完成后立即 `ggml_context_free`。
- 利用测量值精确分配所需大小（加少量余量），避免浪费。

#### 4.2.3 分离持久与临时 context

- **持久 Context**：存放模型权重、位置编码、固定嵌入等，生命周期与 `InferenceEngine` 一致，使用 `ggml_context` 或 `ggml_backend_buffer`。
- **临时 Context 池**：管理一组可变大小 context，按需取用，支持 `reset` 和 `free`。

#### 4.2.4 改进 `IncContext.reset()`

- 增加 `reset(force: bool)` 参数：当 `force=true` 时，除了重置指针，还调用 `ggml_context_free` + `ggml_context_init` 重新创建，彻底回收内存（代价较高，仅在大序列重置时使用）。

### 4.3 长期优化（系统级弹性）

- **可增长 Context 包装**：实现 `GrowingContext`，当容量不足时自动创建更大的新 context，复制必要的持久数据（如 token 嵌入），然后释放旧 context。
- **与 Zig Arena 结合**：使用 `std.heap.ArenaAllocator` 管理 `ggml_tensor` 元数据，而数据区由 ggml 管理，统一分配策略。
- **自适应阈值**：根据当前序列长度和模型参数，动态计算所需内存，并提前调整预分配大小（例如在 `engine.generate` 入口处）。
- **内存监控与告警**：集成 `ggml_backend_sched` 的内存使用统计，当使用率 >90% 时触发警告并尝试回收。

---

## 5. 代码修改清单

### 5.1 新增/修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/ggml/graph.zig` | 新增 | `measureGraph`、`measureCGraph` |
| `src/core/memory_pool.zig` | 新增 | 临时 context 池管理 |
| `src/core/engine.zig` | 修改 | 分离权重、KV、临时 context，使用测量逻辑 |
| `src/core/prefill.zig` | 修改 | 采用独立 context + 测量 |
| `src/core/decode.zig` | 修改 | 采用独立 context + 测量 |
| `src/core/graph_context.zig` | 修改 | `reset(force)` 增强 |
| `src/ggml/context.zig` | 新增 | `usage()` 辅助函数，`reset()` 增强 |
| `src/core/engine_common.zig` | 修改 | 移除固定 context 大小硬编码，改用动态 |

### 5.2 关键接口设计（示例）

```zig
// src/ggml/graph.zig
pub fn measureGraph(ctx: *Context, graph: *CGraph) !usize {
    // 创建临时 gallocr 并设置为测量模式
    var gallocr = try Gallocr.init(...);
    defer gallocr.free();
    gallocr.setMeasure(true);
    _ = gallocr.allocGraph(graph);
    return gallocr.getRequiredSize();
}

// src/core/memory_pool.zig
pub const TempContextPool = struct {
    allocator: std.mem.Allocator,
    contexts: std.ArrayList(*Context),
    // 按大小分级管理
    pub fn acquire(self: *TempContextPool, min_size: usize) !*Context {
        // 寻找大小 >= min_size 的空闲 context，若无则创建新
    }
    pub fn release(self: *TempContextPool, ctx: *Context) void {
        ctx.reset(true); // 彻底重置
        // 放回池中
    }
};
```

---

## 6. 实施优先级与路线图

| 优先级 | 任务 | 预计工作量 | 状态 |
|--------|------|------------|------|
| **P0** | 增大默认 context 到 4096 MB（临时缓解） | 0.5 天 | ✅ 已完成 |
| **P0** | 实现 `measureGraph` 并集成到 Prefill | 2 天 | ✅ 已完成 |
| **P0** | 为三阶段 Prefill 独立分配 context | 1.5 天 | ✅ 已完成 |
| **P1** | 分离持久与临时 context | 3 天 | ✅ 已完成 |
| **P1** | 实现 TempContextPool | 2 天 | ✅ 已完成 |
| **P1** | 改进 IncContext.reset(force) | 1 天 | ✅ 已完成 |
| **P1** | ggml/context.zig usage() 辅助函数 | 0.5 天 | ✅ 已完成 |
| **P2** | 可增长 Context 包装 | 4 天 | ⚪ 长期 |
| **P2** | 自适应阈值与监控 | 2 天 | ⚪ 长期 |

---

## 7. 验证与测试标准

- **单元测试**：新增 `test_measure_graph.zig`，验证测量值与实际分配一致。
- **回归测试**：纯文本、多模态（不同尺寸图像）、长上下文（>10k tokens）场景下，内存不超限且无崩溃。
- **压力测试**：连续加载/卸载模型 10 次，检测内存泄漏（使用 `GeneralPurposeAllocator`）。
- **性能测试**：测量引入测量后的额外开销（应 < 5% 总推理时间）。
- P0 验收标准: `zig-out/bin/zllama -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj ~/.cache/models/gemma-4-E2B-mmproj-BF16.gguf --image ~/.cache/models/hello.png -p " " --verbose-prompt` 能够正常运行。

---

## 8. 常见问题与陷阱

- **ggml_context 与 backend 的交互**：测量模式可能不适用于所有后端，需在 CPU 后端上运行。
- **`ggml_allocr` 的线程安全性**：测量和分配应在同一线程，避免竞争。
- **与 KV Cache 的依赖**：KV Cache 的 context 独立于 graph context，需确保不干扰。
- **多模态投影器权重**：mmproj 权重可放在权重 context 中，无需参与临时分配。

---

## 9. 相关文档

- `ARCHITECTURE.md` — 系统分层与数据流
- `TASK_MEM.md` — 内存策略重构任务详细说明
- `TECHNICAL_CHALLENGES.md` — 技术难重点
- `GGML_BUILD.md` — ggml 构建集成

---

**修订历史**：

- 2025-07-11：重写，基于多模态崩溃分析，增加动态测量与独立 context 策略。
- 之前版本：初始创建（记录 mmap、Arena 等优化）。

