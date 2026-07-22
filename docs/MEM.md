# zllama.zig 内存管理设计文档（修订版 v4.0）

> **目标**：基于"构建 → 编译 → 执行"三段式生命周期框架，构建可预测、可扩展、高效的内存体系，支撑多模态大模型推理。
> **适用版本**：Zig 0.16.0 + ggml（v0.16 系列），zllama.zig 当前主分支。

---

## 1. 设计总纲：三段式生命周期框架

### 1.1 核心思想

将 GGML 计算图的整个生命周期严格划分为三个阶段，每个阶段职责单一、边界清晰：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    三段式生命周期框架                               │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 1: 构建期 (Build)    →  只画图，不分配任何内存                │
│  阶段 2: 编译期 (Compile)  →  测量需求，规划内存复用策略            │
│  阶段 3: 执行期 (Exec)     →  分配物理内存，执行计算，释放资源      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 分层架构

```
┌────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine (业务层)                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│  │  权重 Context  │  │   KV Cache    │  │   临时 Context 池    │   │
│  │  (只读, mmap)  │  │  (独立分配器) │  │  (可重置/可扩容)     │   │
│  └───────────────┘  └───────────────┘  └───────────────────────┘   │
├────────────────────────────────────────────────────────────────────┤
│                    封装层 (Zig Wrapper Layer)                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  IncContext: beginStep() → buildGraph() → compute()          │  │
│  │  TempContextPool: acquire() → build() → release()            │  │
│  └──────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────┤
│                     底层 (ggml C API)                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ggml_context / ggml_cgraph / ggml_gallocr / ggml_backend    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 三段式在 Prefill/Decode 中的映射

| 框架阶段 | 文档对应设计 | 关键操作 |
|----------|-------------|----------|
| **构建期** | "构建计算图（无分配）" | `ggml_context` 以 `no_alloc = true` 创建，只添加算子 |
| **编译期** | `measureGraph()` 获得所需字节数 | 图分配器分析生命周期，计算最优内存复用方案 |
| **执行期** | "若 > 当前容量则扩展，分配并执行，释放或 reset" | Zig 分配器划拨物理内存，挂载后执行 `ggml_backend_graph_compute` |

---

## 2. 内存使用热点与瓶颈分析

### 2.1 典型工作负载内存构成

| 组件 | 典型规格（Gemma 4 E2B） | 估算大小 | 生命周期 |
|------|--------------------------|----------|-----------|
| **模型权重** | 4-bit 量化（Q4_K_M），约 6.7B 参数 | ~4.0 GB | 进程生命周期（只读） |
| **KV Cache** | 35 层，max_seq_len=131072，head_dim=128，K/V 各 f16 | 35×131072×128×2×2字节 ≈ 2.3 GB（按需增长） | 会话生命周期（可扩展） |
| **mmproj 多模态权重** | 视觉/音频编码器权重（f16） | ~0.5 GB | 进程生命周期（只读） |
| **输入嵌入** | 文本 token + 媒体 token（如 784×1536 f32） | 784×1536×4 ≈ 4.8 MB | 临时（仅当前 Prefill） |
| **中间激活** | 每层 Q/K/V、注意力分数、FFN 中间值等 | 高峰可达 **2–3 GB** | 临时（单次计算图执行） |
| **ggml_context 元数据** | 张量描述、节点指针等 | 数十 MB | 随 context 生命周期 |

**关键结论**：**中间激活**是内存波动的最大来源，尤其在多模态长序列（如图像 token 数量大）和非因果注意力场景下，峰值可能超过模型权重或 KV Cache。三段式框架的"编译期测量"正是为此设计——在执行前精确获知峰值需求。

### 2.2 当前面临的主要问题（已解决 ✅）

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| 固定尺寸硬编码 | ✅ 已解决 | `estimateGraphSize()` 动态计算，`measureGraph()` 精确测量 |
| 三阶段 Prefill 共用同一个 context | ✅ 已解决 | 每个 Pass 使用独立临时 context，Pass 完成后立即释放 |
| 缺乏峰值测量 | ✅ 已解决 | `measureGraph()` 封装在 `src/ggml/graph.zig`，Prefill 和 Decode 均已集成 |
| 持久与临时张量混用 | ✅ 已解决 | 权重、KV Cache、临时 context 分离管理 |
| `IncContext.reset()` 不彻底 | ✅ 已解决 | `resetFull()` 释放并重建 gallocr + context |
| 裸 malloc/free 管理输入数据 | ✅ 已解决 | 改用 Zig 分配器管理输入张量数据 |

---

## 3. 已完成的内存优化（现状评估）

| 优化项 | 位置 | 说明 |
|--------|------|------|
| **mmap 加载模型** | `src/core/engine_common.zig` | 零拷贝加载 GGUF，减少内存占用和启动时间 |
| **Arena 管理 init 阶段** | `src/core/engine.zig` | 使用 ArenaAllocator 管理临时分配，初始化后自动释放 |
| **KV Cache 预分配** | `src/kv_cache.zig` | 按最大序列长度预先分配，避免动态扩容开销 |
| **WallTimer** | `src/core/engine_common.zig` | 使用 Zig 0.16 的 `std.Io.Clock` 实现高性能计时 |
| **延迟加载多模态** | `src/core/engine.zig` | 仅在需要时加载 mmproj 编码器 |
| **线程池与后端调度** | `src/ggml/backend.zig` | 多线程计算图调度 |
| **measureGraph** | `src/ggml/graph.zig` | 计算图内存测量，精确获知峰值需求 |
| **TempContextPool** | `src/core/memory_pool.zig` | 临时 context 池管理，按需取用 |
| **三阶段独立 context** | `src/core/prefill.zig` | 每个 Pass 使用独立临时 context，Pass 完成后立即释放 |
| **IncContext.resetFull** | `src/core/graph_context.zig` | 完整重置，释放并重建 gallocr + context |
| **usage() 辅助函数** | `src/ggml/context.zig` | 打印 context 内存使用详情 |
| **Zig 分配器管理输入数据** | `src/core/prefill.zig` | 移除裸 malloc/free，改用 Zig allocator |

---

## 4. 设计原则

- **可测量性**：任何计算图构建前必须能准确预测内存需求——这是三段式中"编译期"的核心价值。
- **弹性分配**：context 大小应随输入动态调整，而非硬编码。
- **关注点分离**：持久数据（权重、KV）与临时数据（激活）使用独立的分配器。
- **可回收性**：确保 `deinit` 能完全释放所有内存，支持进程内模型热切换。
- **零欠载**：绝不因内存不足而崩溃（通过测量 + fallback 机制）。
- **可观测性**：内存分配和复用情况可追踪、可监控。

---

## 5. 代码现状与文档差异修正

### 5.1 已实现但文档未更新的内容

| 文档声称"待实施" | 实际状态 | 修正 |
|------------------|----------|------|
| `measureGraph` 待实施 | ✅ 已实现于 `src/ggml/graph.zig:132` | 标记为已完成 |
| 三阶段 Prefill 独立 context | ✅ 已实现于 `src/core/prefill.zig`（textPass/mediaPass/threeStagePrefill） | 标记为已完成 |
| `TempContextPool` 待实施 | ✅ 已实现于 `src/core/memory_pool.zig:19` | 标记为已完成 |
| `IncContext.reset(force)` 待实施 | ✅ 已实现为 `resetFull()` 于 `src/core/graph_context.zig:166` | 标记为已完成 |
| `usage()` 辅助函数 | ✅ 已实现于 `src/ggml/context.zig:150` | 标记为已完成 |
| 分离持久与临时 context | ✅ 权重、KV Cache、临时 context 已分离 | 标记为已完成 |
| 增大默认 context | ✅ `estimateGraphSize()` 返回 2GB（非硬编码 4GB） | 标记为已完成 |

### 5.2 文档中不准确或已过时的内容

| 原内容 | 问题 | 修正 |
|--------|------|------|
| §6.1 "增大默认 Graph context 到 4096 MB" | 实际使用 `estimateGraphSize()` 动态计算，返回 2GB | 删除硬编码建议，改为描述动态计算策略 |
| §6.2.1 "measureGraph 使用临时 gallocr + set_measure" | 实际实现使用 `ggml_gallocr_reserve` + `ggml_gallocr_get_buffer_size` | 更新为实际实现方式 |
| §6.2.2 "为每个 Prefill 阶段分配独立 context" | 已实现，但文档描述过于简略 | 补充实际实现细节 |
| §7.1 "GgmlSession 三段式封装" | 未实现，且当前架构（IncContext + TempContextPool）已覆盖其功能 | 标记为"不适用，已被 IncContext 替代" |
| §7.2 "TempContextPool" 接口设计 | 实际实现与文档设计有差异 | 更新为实际 API |
| §5 "Zig 0.16 适配指南" | 项目已使用 Zig 0.16，`@cImport` 仍保留在 `src/ggml/c.zig` 封装层 | 更新为实际状态 |
| §9 路线图中 P0/P1 状态 | 大部分已实际完成 | 更新状态标记 |

### 5.3 文档中正确且仍需保留的内容

| 内容 | 说明 |
|------|------|
| §1 三段式生命周期框架 | 核心设计思想，完全正确 |
| §2.1 内存构成分析 | 数据准确，保留 |
| §4 设计原则 | 原则正确，保留 |
| §6.3 长期优化（P2） | 可增长 Context、自适应阈值等尚未实现 |
| §10 验证标准 | 测试标准有效，保留 |
| §11 常见问题 | 部分问题已解决，保留作为参考 |

---

## 6. 改进方案（分阶段实施）

### 6.1 已完成 ✅

| 措施 | 位置 | 说明 |
|------|------|------|
| **measureGraph 封装** | `src/ggml/graph.zig` | 使用 `ggml_gallocr_reserve` + `get_buffer_size` 精确测量 |
| **三阶段独立 context** | `src/core/prefill.zig` | textPass/mediaPass/threeStagePrefill 各使用独立临时 context |
| **TempContextPool** | `src/core/memory_pool.zig` | 临时 context 池，支持 acquire/release 复用 |
| **IncContext.resetFull** | `src/core/graph_context.zig` | 完整重置，释放并重建 gallocr + context |
| **usage() 辅助函数** | `src/ggml/context.zig` | 打印 context 内存使用详情 |
| **分离持久与临时 context** | `src/core/engine.zig` | 权重、KV Cache、临时 context 分离 |
| **Zig 分配器管理输入数据** | `src/core/prefill.zig` | 移除裸 malloc/free |

### 6.2 待实施（P1 — 短期）

| 措施 | 优先级 | 说明 |
|------|--------|------|
| **Gallocr 跨 Pass 复用优化** | P1 | 当前 prefill 每个 Pass 都调用 `gallocr.reserve()` + `allocGraph()`，可优化为预规划一次后复用 |
| **Prefill 输入数据生命周期管理** | P1 | `setDataPtr` 传入的 buffer 在 context 释放后悬空，需确保 gallocr compute 完成前数据有效 |
| **mediaPass 中 n_threads 参数未使用** | P1 | `graph.compute(n_threads)` 已使用，但 `_ = params` 表明 params 未使用，可考虑移除 |

### 6.3 长期优化（P2 — 系统级弹性）

| 措施 | 说明 |
|------|------|
| **可增长 Context 包装** | 实现 `GrowingContext`，容量不足时自动创建更大的新 context |
| **与 Zig Arena 结合** | 使用 `std.heap.ArenaAllocator` 管理 `ggml_tensor` 元数据，数据区由 ggml 管理 |
| **自适应阈值** | 根据当前序列长度和模型参数动态计算所需内存，提前调整预分配大小 |
| **内存监控与告警** | 集成 `ggml_backend_sched` 的内存使用统计，使用率 >90% 时触发警告并尝试回收 |
| **GgmlSession 封装** | 若 IncContext + TempContextPool 模式不满足需求，可考虑实现三段式封装 |

---

## 7. 核心接口设计（实际实现）

### 7.1 IncContext（增量解码上下文）

```zig
// src/core/graph_context.zig
pub const IncContext = struct {
    ctx_inc: *ggml.Context,       // 增量解码图上下文
    galloc: ?*ggml.Gallocr,       // 持久化 Gallocr（跨 token 复用）
    galloc_reserved: bool,        // 是否已完成预规划
    cached_input: ?*ggml.Tensor,  // 缓存的输入 token 张量

    pub fn init(allocator, params, ctx_size) !IncContext
    pub fn deinit(self) void
    pub fn getGallocr(self) !*ggml.Gallocr
    pub fn reserveGallocr(self, max_graph) !void
    pub fn resetFull(self) void
    pub fn beginStep(self) !DecodeStep
};
```

### 7.2 TempContextPool（临时 context 池）

```zig
// src/core/memory_pool.zig
pub const TempContextPool = struct {
    allocator: std.mem.Allocator,
    contexts: std.ArrayList(*ggml.Context),
    capacities: std.ArrayList(usize),
    borrowed: std.AutoHashMap(*ggml.Context, void),

    pub fn init(allocator) TempContextPool
    pub fn deinit(self) void
    pub fn acquire(self, min_size) !*ggml.Context
    pub fn release(self, ctx) void
    pub fn printStats(self) void
};
```

### 7.3 Prefill 三阶段接口

```zig
// src/core/prefill.zig
pub fn textPass(
    model_instance, kv_cache_mgr, tokens, start_pos,
    params, n_threads, allocator, gallocr, want_logits,
) !?[]f32

pub fn mediaPass(
    model_instance, kv_cache_mgr, media_token_id, media_count,
    media_embeddings, embd_dim, start_pos,
    params, n_threads, allocator, gallocr, chunk_size,
) !void

pub fn threeStagePrefill(
    graph_ctx, model_instance, kv_cache_mgr,
    prefix_tokens, media_token_id, media_token_count,
    media_embeddings_data, media_embd_dim, suffix_tokens,
    params, n_threads, allocator, gallocr,
) !PrefillResult
```

---

## 8. 代码修改清单（实际状态）

### 8.1 已实现文件

| 文件 | 说明 |
|------|------|
| `src/ggml/graph.zig` | `measureGraph`、`measureGraphDetailed` |
| `src/core/memory_pool.zig` | `TempContextPool` 临时 context 池管理 |
| `src/core/prefill.zig` | 三阶段独立 context + 测量 + Zig 分配器管理 |
| `src/core/graph_context.zig` | `IncContext` + `resetFull()` |
| `src/ggml/context.zig` | `usage()`、`printUsage()`、`usedMem()`、`totalMem()` |
| `src/core/engine.zig` | `estimateGraphSize()` 动态计算 |
| `src/core/engine_common.zig` | `computeGraph()` 使用传入的 gallocr |

### 8.2 未实现（待后续）

| 文件/功能 | 说明 |
|-----------|------|
| `GgmlSession` 三段式封装 | 当前 IncContext + TempContextPool 已覆盖，暂不需要 |
| 可增长 Context 包装 | P2 长期优化 |
| 自适应阈值与监控 | P2 长期优化 |

---

## 9. 实施优先级与路线图（更新版）

| 优先级 | 任务 | 状态 |
|--------|------|------|
| **P0** | 实现 `measureGraph` 并集成到 Prefill | ✅ 已完成 |
| **P0** | 为三阶段 Prefill 独立分配 context | ✅ 已完成 |
| **P0** | 移除裸 malloc/free，改用 Zig 分配器 | ✅ 已完成 |
| **P1** | 分离持久与临时 context | ✅ 已完成 |
| **P1** | 实现 `TempContextPool` | ✅ 已完成 |
| **P1** | 改进 `IncContext.resetFull()` | ✅ 已完成 |
| **P1** | `ggml/context.zig` `usage()` 辅助函数 | ✅ 已完成 |
| **P1** | Gallocr 跨 Pass 复用优化 | ⬜ 待实施 |
| **P1** | Prefill 输入数据生命周期管理 | ⬜ 待实施 |
| **P2** | 可增长 Context 包装 | ⬜ 长期 |
| **P2** | 自适应阈值与监控 | ⬜ 长期 |

---

## 10. 验证与测试标准

- **单元测试**：`test_measure_graph.zig` 验证测量值与实际分配一致。
- **回归测试**：纯文本、多模态（不同尺寸图像）、长上下文（>10k tokens）场景下，内存不超限且无崩溃。
- **压力测试**：连续加载/卸载模型 10 次，检测内存泄漏（使用 `GeneralPurposeAllocator`）。
- **性能测试**：测量引入测量后的额外开销（应 < 5% 总推理时间）。
- **P0 验收标准**：
  ```bash
  zig-out/bin/zllama -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf \
    --mmproj ~/.cache/models/gemma-4-E2B-mmproj-BF16.gguf \
    --image ~/.cache/models/hello.png -p " " --verbose-prompt
  ```
  上述命令能够在 Zig 0.16 环境下正常编译运行。

---

## 11. 常见问题与陷阱

| 问题 | 解决方案 |
|------|----------|
| `setDataPtr` 传入的 buffer 生命周期 | buffer 必须在 `graph.compute()` 完成前保持有效。临时 context 释放后 buffer 悬空 |
| Gallocr 跨 Pass 复用 | gallocr 内部使用张量指针作为哈希键，context 释放后指针失效，需重新 reserve |
| `measureGraph` 与后端兼容性 | 测量模式适用于 CPU 后端，GPU 后端需额外验证 |
| KV Cache 与临时 context 的依赖 | KV Cache 的 context 独立于 graph context，确保不干扰 |
| 多模态投影器权重 | mmproj 权重放在权重 context 中，无需参与临时分配 |
| `ArenaAllocator` 线程安全 | 多线程场景下为每个线程独立创建 Arena |

---

## 12. 相关文档

- `ARCHITECTURE.md` — 系统分层与数据流
- `AGENTS.md` — AI 编程入口与设计哲学
- `TECHNICAL_CHALLENGES.md` — 技术难点总结

---

**修订历史**：

| 日期 | 版本 | 说明 |
|------|------|------|
| 2025-07-11 | v2.0 | 重写，基于多模态崩溃分析，增加动态测量与独立 context 策略 |
| 2025-07-22 | v3.0 | 整合三段式生命周期框架；新增 Zig 0.16 适配指南 |
| 2025-07-22 | v4.0 | 同步代码现状：标记已完成项，修正过时内容，更新接口设计为实际实现 |
