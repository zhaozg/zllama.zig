# zllama.zig 工作计划

> **哲学宣言**：从 `llama.cpp` 的"数学家先跑通公式，再顺手搭个脚手架"的 C 语言田野式生长，走向 Zig 的"显式分配器、编译期泛型、强类型接口"的工程哲学——将数学的优雅与工程的严谨统一于一个可预测、可扩展、可验证的系统。

---

## 一、 工作计划：分阶段实施路线图

### 阶段 0：基础架构巩固（已完成 ✅）

| 任务 | 状态 | 说明 |
|------|------|------|
| 七层 DAG 架构定义 | ✅ | L0-L7 分层，依赖方向强制约束 |
| 两大 VTable 家族 | ✅ | `ModelVTable` + `VisionEncoderBackend`/`AudioEncoderBackend` |
| 三段式生命周期框架 | ✅ | 构建→编译→执行 |
| `measureGraph` 精确测量 | ✅ | `src/ggml/graph.zig` |
| 三阶段 Prefill 独立 context | ✅ | `src/core/prefill.zig` |
| `TempContextPool` | ✅ | `src/core/memory_pool.zig` |
| `IncContext.resetFull()` | ✅ | `src/core/graph_context.zig` |
| Gallocr 所有权上移 | ✅ | `InferenceEngine` 持有 |
| 日志作用域命名规则 | ✅ | 层级前缀 + 下划线 + 组件名 |

### 阶段 1：核心重构（已完成 ✅）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| `core/engine.zig` 拆分 | **P0** ✅ | 将 695 行杂糅代码拆为 `context.zig`（GGUF 加载）、`planner.zig`（图规划）、`executor.zig`（图执行） |
| 依赖倒置强制执行 | **P0** ✅ | `executor` 只依赖 `planner` 给出的 `GraphPlan` 结构体，绝不直接读取 `context` 里的原始指针 |
| KV Cache 环形缓冲区封装 | **P1** ✅ | 将 `kv_cache.zig` 中的裸指针操作封装为 `rotate()`、`evict()`、`copy()` 方法 |
| 消除运行时 `realloc` | **P1** ✅ | 缓存高频使用的图结构（如 Prefill 图），彻底消灭 `ggml_gallocr_needs_realloc` |
| 多后端统一接口 | **P1** ✅ | `Backend` 接口只暴露 `execGraph()`、`allocTensor()`、`copyToDevice()` |
| **GraphPlanner 无状态化** | **P1** ✅ | 移除 `GraphPlanner` 对 `ModelContext` 字段的指针存储，改为所有方法通过参数传递所需引用，彻底消除悬空指针风险 |
| **InferenceEngine.finalizeInit()** | **P1** ✅ | 新增 `finalizeInit()` 方法，在 `InferenceEngine` 移动到最终内存位置后注册 `IncContext` 到内存监控器，解决 `ctx` move 后指针悬空问题 |

### 阶段 2：多模态完善（已完成 ✅）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| Gemma 4 E2B 端到端推理 | **P0** ✅ | 视觉 + 音频 + 文本三模态联合推理已验证 |
| 三阶段 Prefill 稳定性 | **P0** ✅ | prefix → media(non-causal) → suffix 各阶段独立 context |
| 视觉编码器 Backend 注册 | **P0** ✅ | gemma4v, gemma4uv, qwen2vl, qwen3vl |
| 音频编码器 Backend 注册 | **P0** ✅ | gemma4a, gemma4ua |
| M-RoPE 位置计算分发 | **P1** ✅ | 通过 `PosType` 枚举分发到 `imageGetDecoderPos()` |
| 多模态 tokenize | **P1** ✅ | 文本+媒体标记混合 tokenize |

### 阶段 3：性能优化（计划中 📋）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 可增长 Context 包装 | **P2** ✅ | `GrowableContext` 包装 `ggml.Context`，容量不足时自动创建更大的新 context |
| 自适应阈值 | **P2** ✅ | `MemoryEstimator` 根据序列长度和模型参数动态计算所需内存，替代硬编码估算 |
| 内存监控与告警 | **P2** ✅ | `MemoryMonitor` 分级告警（90%/95%/99%），自动回收与扩容 |
| 阶段3 P2 集成 | **P2** ✅ | `MemoryEstimator` 替换硬编码估算，`MemoryMonitor` 接入推理循环 |
| GPU 后端支持 | **P2** | Metal (macOS)、CUDA (Linux) |
| 线程池优化 | **P2** | 物理核心数 2/3~3/4 动态调整 |

### 阶段 4：质量保障（持续 🔄）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 数值对齐测试 | **P0** | 与 llama.cpp 参考实现对比 logits，NMSE < 1e-5 或余弦相似度 > 0.999 |
| 随机权重回归测试 | **P1** | 固定种子生成权重，前向传播形状和数值范围作为基准 |
| 压力测试 | **P1** | 连续加载/卸载模型 10 次，检测内存泄漏 |
| 长上下文测试 | **P1** | >10K tokens 场景下内存不超限且无崩溃 |
| 跨平台编译测试 | **P2** | macOS / Linux / Windows |

---

## 二、 验证标准

### 数值正确性

```bash
# 对比第一个 token logits 与 llama.cpp 参考值
zig-out/bin/compare_logits --model model.gguf --reference ref.bin
# 要求：NMSE < 1e-5 或余弦相似度 > 0.999
```

### 内存安全

```bash
# 使用 GeneralPurposeAllocator 检测内存泄漏
zig build test -Doptimize=ReleaseSafe --summary all
# 连续加载/卸载模型 10 次，零泄漏
```

### 端到端推理

```bash
# 纯文本推理
zig-out/bin/zllama -n 5 --model tinyllama.gguf -p "Hello"

# 多模态推理
zig-out/bin/zllama -m gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj gemma-4-E2B-mmproj-BF16.gguf \
  --image hello.png -p " " --verbose-prompt
```

---

## 三、 与 llama.cpp 的关系

### 对齐策略

- **底层（ggml）**：完全对齐，确保数值一致性。所有 ggml C API 通过 `src/ggml/` 模块封装。
- **上层（模型实现）**：参考 llama.cpp 的模型实现逻辑，但用 Zig 的工程哲学重构——消灭 `if` 森林，引入 VTable 策略模式。
- **差异记录**：在代码注释或 `docs/` 中记录与参考实现的故意差异及原因。

### 已知差异

| 方面 | llama.cpp | zllama.zig |
|------|-----------|------------|
| 语言 | C/C++ | Zig 0.16.0 |
| 多态机制 | `if (arch == ...)` 硬编码 | VTable 策略模式 |
| 内存管理 | 手动 new/delete | Arena + defer + 三段式生命周期 |
| 图构建与执行 | 混合（边构建边执行） | 两阶段分离（规划→执行） |
| KV Cache | 裸指针 k_cache / v_cache | 环形缓冲区封装 |
| 多模态编码器 | mtmd_context + clip 分离 | MultiModalManager 统一管理 |

---

## 四、 决策矩阵

当需要修改或扩展本系统时，依据以下决策树快速定位设计边界：

| 需求场景 | 设计落脚点 | 禁止行为 |
| :--- | :--- | :--- |
| 新增一种文本模型架构 | `src/models/` 新建文件 + `src/model.zig` 加枚举 + `src/models/registry.zig` 注册 | 修改 `InferenceEngine` 或 `decode.zig` 的主循环 |
| 新增一种视觉/音频编码器 | `src/mtmd/graph/models/` 新建文件 + `src/mtmd/graph/mod.zig` 注册 Backend + `src/mtmd/vision/mod.zig`（或 `audio/mod.zig`）注册 `getBackend` | 修改 `InferenceEngine` 或 `helper.zig` 的主循环 |
| 新增一种预处理方式 | `src/mtmd/preprocess.zig` 新增函数，在 `src/mtmd/tokenize.zig` 或 `src/mtmd/vision/preprocess.zig` 中调用 | 将预处理代码混入 `graph/` 层 |
| 优化 CPU 计算性能（如算子融合） | 仅修改 `src/mtmd/graph/` 或 `src/layers/` 下的构建块 | 触碰 `ggml/backend.zig` 或内存释放逻辑 |
| 支持新的硬件后端（如 Vulkan） | 扩展 `src/ggml/backend.zig` 的 `DeviceType` 枚举，实现 `detectBestBackend` | 在业务层（`helper.zig` 或 `engine.zig`）添加硬件判断分支 |
| 解决 Logits 精度不匹配 | 追踪 `buildGraph` 中缺失的 `scale`/`bias`/`norm` 操作，对比 llama.cpp 参考实现 | 盲目调整 `ggml` 绑定或后端 buffer 大小 |
| 新增 ggml 算子 | 在 `src/ggml/ops.zig` 封装，在 `src/ggml/mod.zig` 重新导出 | 业务代码直接 `@cImport` 调用裸 C 函数 |

---

## 五、 遗留问题与已知风险

> 以下问题在最近一次重构（阶段 1 核心重构）后识别，需要在后续迭代中解决。

### 5.1 高优先级（P0）

| 问题 | 描述 | 影响范围 | 建议方案 |
|------|------|----------|----------|
| **PrefillGraphCache 被移除** | `GraphPlanner` 无状态化重构中移除了 `PrefillGraphCache`，每次 prefill 都重新构建图结构 | 重复 prefill 场景（如 chat 多轮对话）性能下降 | 在 `InferenceEngine` 层重新实现缓存，或使用 `GrowableContext` 复用图内存 |
| **`reserveDecodeGallocr` 中 `gallocr` 参数未使用** | `reserveDecodeGallocr` 接收 `gallocr` 参数但使用 `inc_ctx.reserveGallocr()` 内部管理 | 接口混淆，调用者可能误以为 gallocr 在此处被使用 | 移除 `gallocr` 参数，或统一 gallocr 管理路径 |
| **`planDecode` 未在 generate 主路径中使用** | `generate()` 和 `chatLoop()` 直接调用 `decode_mod.runDecodeLoop`，未使用 `planner.planDecode` | `planDecode` 成为死代码，decode 图构建逻辑分散 | 将 `runDecodeLoop` 内部改为调用 `planner.planDecode`，或移除 `planDecode` |

### 5.2 中优先级（P1）

| 问题 | 描述 | 影响范围 | 建议方案 |
|------|------|----------|----------|
| **`ModelContext.toMultimodalContext()` 复制 `tok`** | `toMultimodalContext()` 中 `tok = self.tok.*` 复制了整个 Tokenizer 值 | 存在潜在的重复释放或悬空指针风险 | 改为传递指针 `tok = &self.tok`，或确保 `EngineContext` 不拥有所有权 |
| **`engine.zig.bak` 残留** | 重构后 `engine.zig.bak` 备份文件未清理 | 代码库整洁性 | 删除备份文件 |
| **`main.zig` 中未使用的 import** | `gguf`、`registry`、`graph_builder`、`graph_context`、`memory`、`sampler`、`kv_cache`、`mm`、`preprocess`、`prefill_mod`、`chat_template`、`loadMMProj` 等 import 在 `main.zig` 中未直接使用 | 编译警告，代码可读性 | 清理未使用的 import |
| **`engine_common.zig` 日志作用域命名** | `engine_common.zig` 使用 `.core_engine` 作为日志作用域，与 `engine.zig` 的 `.core_engine` 冲突 | 日志过滤时无法区分两个模块 | 将 `engine_common.zig` 的日志作用域改为 `.core_engine_common` |

### 5.3 低优先级（P2）

| 问题 | 描述 | 影响范围 | 建议方案 |
|------|------|----------|----------|
| **`estimateKVCacheSize` 测试硬编码** | `test "estimateKVCacheSize basic"` 中硬编码了 1.7GB 的期望值 | 测试脆弱，参数变化时容易失败 | 改为基于公式动态计算期望值 |
| **`estimateGraphSize` 测试过于简单** | 仅检查返回值是否为 2GB | 测试无实际验证意义 | 增加更精确的估算验证 |
| **`GraphPlanner` 测试覆盖不足** | 仅有 `init` 和结构体大小测试 | 核心逻辑无测试覆盖 | 增加 mock 测试验证 `planPrefill` 和 `planDecode` |

---

## 六、 近期提交记录

| 提交 | 描述 |
|------|------|
| `689c7ef` | fix: 修复 InferenceEngine.init 中 GraphPlanner.kv_cache_mgr 悬空指针问题 |
| `337f9af` | refactor: 优化 Qwen3VL 模型实现 |
| `1d72663` | refactor: 将 chat_template/_tests.zig 移至 tests/test_chat_template.zig |
| `799f71b` | fix: 修复测试中的内存泄漏和 log.err 导致的测试失败 |
| `2b7f7ce` | 阶段3 P2: 集成 GrowableContext / MemoryEstimator / MemoryMonitor |

---

> **最终目标**：将 `llama.cpp` 的"数学家先跑通公式，再顺手搭个脚手架"的 C 语言田野式生长，转化为 Zig 的"显式分配器、编译期泛型、强类型接口"的工程哲学——让数学的优雅与工程的严谨在 `zllama.zig` 中完美统一。
>
> **衡量标准**：当新增一个模型架构时，开发者只需新建一个文件、注册一个枚举、实现一个 VTable——**完全不用动主推理循环**。当系统运行 131K 上下文时，内存可预测、无泄漏、无 `realloc`。当与 llama.cpp 对比 logits 时，NMSE < 1e-5。
