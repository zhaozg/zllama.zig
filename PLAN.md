# zllama.zig 验证体系与工程化方案

> 基于对 llama.cpp 源码架构的深度分析，提炼其模块化设计、测试体系、调试手段，
> 为 zllama.zig 制定可落地的验证思路与执行方案。

---

## 一、llama.cpp 架构精华提炼

### 1.1 模块化分层

```
┌─────────────────────────────────────────────────────────────┐
│  examples/ (simple, batched, server...)                      │
├─────────────────────────────────────────────────────────────┤
│  llama.cpp / llama.h (统一 C API 层)                         │
├─────────────────────────────────────────────────────────────┤
│  llama-model.cpp / llama-model.h (模型工厂 + 基类)           │
│  llama-arch.cpp / llama-arch.h (架构注册表)                  │
│  llama-hparams.h (超参数)                                    │
│  llama-graph.cpp / llama-graph.h (计算图构建上下文)          │
│  llama-kv-cache.cpp / llama-kv-cache.h (KV Cache)           │
│  llama-memory.h (内存抽象接口)                               │
│  llama-model-loader.cpp / .h (模型加载器)                    │
│  llama-vocab.cpp / .h (词表)                                 │
│  llama-sampler.cpp / .h (采样器)                             │
├─────────────────────────────────────────────────────────────┤
│  src/models/ (131 个模型实现文件)                             │
│  src/models/models.h (基类 + 共享算子)                       │
├─────────────────────────────────────────────────────────────┤
│  ggml (底层张量库)                                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 关键设计模式

| 模式 | llama.cpp 实现 | zllama 借鉴 |
|------|---------------|-------------|
| **模型工厂** | `llama_model_mapping()` 大 switch 返回子类指针 | `registry.zig` 的 switch 分发 |
| **架构注册** | `llm_arch` 枚举 + `LLM_ARCH_NAMES` map | `Architecture` 枚举 + `fromString()` |
| **图构建上下文** | `llm_graph_context` 基类，子类实现 `build()` | 需引入 `GraphContext` 接口 |
| **内存抽象** | `llama_memory_context_i` 虚接口 | 需引入 `MemoryContext` 接口 |
| **模型加载** | `llama_model_loader` 统一加载流程 | 需重构 `gguf.zig` 为加载器 |
| **测试体系** | `test-llama-archs.cpp` 随机权重 + NMSE 对比 | 需建立类似机制 |

### 1.3 测试体系架构

```
src/tests/
├── test-llama-archs.cpp    ← 核心：所有架构的随机权重前向对比
│   ├── 生成随机 GGUF（小模型：2层，256维）
│   ├── CPU 推理获取参考 logits
│   ├── 各后端推理获取 logits
│   └── NMSE 对比（阈值 1e-4）
├── test-gguf.cpp           ← GGUF 格式解析测试
├── test-tokenizer-*.cpp    ← 分词器测试（使用 models/ 下的参考词表）
├── test-rope.cpp           ← RoPE 数值测试
├── test-sampling.cpp       ← 采样器测试
└── test-quantize-fns.cpp   ← 量化函数测试
```

---

## 二、zllama 当前问题诊断

### 2.1 架构问题

1. **缺少统一模型接口**：`model.zig` 定义了 `ModelParams` 和 `ModelWeights`，但各模型实现（`qwen.zig`, `llama.zig`）未统一继承/实现该接口
2. **图构建与推理耦合**：`forward()` 函数同时负责构建计算图和执行推理，难以独立测试
3. **缺少内存抽象**：`kv_cache.zig` 直接管理张量，未抽象出 `MemoryContext` 接口
4. **模型加载分散**：每个模型自行解析 GGUF 元数据和张量，重复代码多
5. **测试覆盖不足**：仅有少量单元测试，缺少端到端前向对比测试

### 2.2 具体差距

| 维度 | llama.cpp | zllama.zig |
|------|-----------|------------|
| 模型数 | 131 个架构 | 3 个架构 |
| 测试用例 | 50+ 测试文件 | 少量内联测试 |
| 随机权重测试 | `test-llama-archs.cpp` | 无 |
| 参考对比 | CPU logits 作为基准 | 无 |
| 回归测试 | CI 自动运行 | 无 |
| 调试工具 | `--verbose`, `--log-level` | 有限 |

---

## 三、验证体系设计方案

### 3.1 总体架构

```
zllama.zig/
├── src/tests/                          ← 新增：独立测试目录
│   ├── test_archs.zig              ← 核心：多架构随机权重前向测试
│   ├── test_gguf.zig               ← GGUF 解析测试
│   ├── test_tokenizer.zig          ← 分词器测试
│   ├── test_layers.zig             ← 算子层数值测试
│   ├── test_kv_cache.zig           ← KV Cache 测试
│   ├── test_sampler.zig            ← 采样器测试
│   ├── fixtures/                   ← 测试用参考数据
│   │   ├── gen_fixture.zig         ← 生成参考输出的工具
│   │   └── ref_outputs/            ← 参考输出（git-lfs 或生成）
│   └── utils.zig                   ← 测试工具函数
├── tools/                          ← 新增：调试/诊断工具
│   ├── dump_graph.zig              ← 计算图可视化
│   ├── compare_logits.zig          ← logits 对比工具
│   └── profile.zig                 ← 性能分析工具
└── src/
    ├── core/                       ← 新增：核心引擎重构
    │   ├── context.zig             ← 推理上下文
    │   ├── graph_builder.zig       ← 图构建辅助
    │   ├── memory.zig              ← 内存抽象接口
    │   └── loader.zig              ← 模型加载器
    ├── models/
    │   ├── registry.zig            ← 模型注册表（重构）
    │   ├── llama.zig               ← LLaMA 模型
    │   ├── qwen.zig                ← Qwen3.5 模型
    │   └── qwen2.zig               ← Qwen2 模型
    └── layers/                     ← 算子层（已有）
```

### 3.2 测试金字塔

```
         ┌──────────┐
         │ 端到端    │  ← 真实模型推理（手动运行）
         │ 推理测试  │
        ┌┴──────────┴┐
        │ 架构前向   │  ← 随机权重 + NMSE 对比（CI 自动）
        │ 对比测试   │
       ┌┴───────────┴┐
       │ 算子层数值  │  ← 各算子独立测试（CI 自动）
       │ 测试        │
      ┌┴────────────┴┐
      │ 单元测试     │  ← 函数级测试（CI 自动）
      │ (GGUF/分词)  │
     ┌┴─────────────┴┐
     │ 编译检查      │  ← zig build（CI 自动）
     └───────────────┘
```

---

## 四、分阶段执行方案

### 第一阶段：基础设施搭建（1-2 周）

#### 4.1.1 重构模型接口

**目标**：建立统一的模型接口，使所有模型实现遵循相同契约。

```zig
// src/core/context.zig - 推理上下文
pub const InferenceContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    ggml_ctx: *ggml.Context,
    graph: *ggml.CGraph,
    memory: *MemoryContext,
    // ...
};

// src/core/memory.zig - 内存抽象接口
pub const MemoryContext = struct {
    // 虚方法表（通过函数指针实现）
    vtable: *const MemoryVTable,
    data: *anyopaque,

    pub const MemoryVTable = struct {
        init: *const fn (data: *anyopaque, params: MemoryParams) anyerror!void,
        deinit: *const fn (data: *anyopaque) void,
        apply: *const fn (data: *anyopaque, ubatch: *const UBatch) anyerror!void,
        get_k: *const fn (data: *anyopaque, layer: usize) *ggml.Tensor,
        get_v: *const fn (data: *anyopaque, layer: usize) *ggml.Tensor,
    };
};

// src/core/graph_builder.zig - 图构建上下文
pub const GraphBuilder = struct {
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    hparams: *const ModelParams,

    pub fn buildNorm(self: *GraphBuilder, input: *ggml.Tensor, weight: *ggml.Tensor, eps: f32) *ggml.Tensor;
    pub fn buildRope(self: *GraphBuilder, input: *ggml.Tensor, pos: *ggml.Tensor, params: RopeParams) *ggml.Tensor;
    pub fn buildAttention(self: *GraphBuilder, input: *ggml.Tensor, layer: *const LayerWeights, kv_cache: ?*MemoryContext) *ggml.Tensor;
    pub fn buildFFN(self: *GraphBuilder, input: *ggml.Tensor, layer: *const LayerWeights) *ggml.Tensor;
};
```

**模型接口契约**：

```zig
// 每个模型必须实现：
pub const ModelInterface = struct {
    init: *const fn (self: *anyopaque, loader: *ModelLoader, io: std.Io) anyerror!void,
    deinit: *const fn (self: *anyopaque) void,
    buildGraph: *const fn (self: *anyopaque, builder: *GraphBuilder, input: *ggml.Tensor, memory: ?*MemoryContext) anyerror!void,
    getParams: *const fn (self: *anyopaque) *const ModelParams,
};
```

#### 4.1.2 建立测试目录结构

```bash
mkdir -p src/tests/fixtures/ref_outputs
mkdir -p tools
```

#### 4.1.3 构建脚本改造

```zig
// build.zig 新增测试目标
const test_archs = b.addTest(.{
    .name = "test-archs",
    .root_source_file = b.path("src/tests/test_archs.zig"),
    // ...
});
const test_archs_step = b.step("test-archs", "Run architecture forward tests");
test_archs_step.dependOn(&test_archs.step);
```

### 第二阶段：核心测试实现（2-3 周）

#### 4.2.1 随机权重前向测试（对标 test-llama-archs.cpp）

**原理**：生成小型随机权重的 GGUF 模型，在 CPU 上推理得到参考 logits，然后验证各模型实现的输出与参考一致。

```zig
// src/tests/test_archs.zig
const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const registry = @import("registry");

// 测试配置
const TestConfig = struct {
    arch: model.Architecture,
    n_layer: u32 = 2,
    n_embd: u32 = 64,
    n_head: u32 = 4,
    n_head_kv: u32 = 2,
    n_ff: u32 = 128,
    n_vocab: u32 = 128,
    seed: u64 = 42,
};

// 生成随机 GGUF
fn generateTestGGUF(allocator: std.mem.Allocator, config: TestConfig) ![]u8 {
    // 1. 创建 GGUF 写入器
    // 2. 写入元数据（架构、维度等）
    // 3. 生成随机权重（正态分布，std=0.01）
    // 4. 返回 GGUF 字节
}

// 运行前向推理
fn runForward(allocator: std.mem.Allocator, gguf_data: []const u8, config: TestConfig) ![]f32 {
    // 1. 解析 GGUF
    // 2. 创建模型实例
    // 3. 构建计算图
    // 4. 执行推理
    // 5. 返回 logits
}

// NMSE 计算
fn nmse(a: []const f32, b: []const f32) f64 {
    // normalized mean squared error
}

test "qwen35 forward consistency" {
    const config = TestConfig{ .arch = .qwen35 };
    const gguf_data = try generateTestGGUF(testing.allocator, config);
    defer testing.allocator.free(gguf_data);

    const logits = try runForward(testing.allocator, gguf_data, config);
    defer testing.allocator.free(logits);

    // 与参考值对比（首次运行生成参考，后续对比）
    // 或与 llama.cpp 输出对比
}
```

#### 4.2.2 算子层数值测试

```zig
// src/tests/test_layers.zig
test "rms_norm numerical" {
    // 1. 创建 ggml context
    // 2. 创建输入张量（已知值）
    // 3. 创建权重张量
    // 4. 调用 rms_norm
    // 5. 验证输出与预期一致
}

test "rope numerical" {
    // 验证 RoPE 旋转后的向量内积与位置差相关
}

test "swiglu numerical" {
    // 验证 SwiGLU 激活函数
}
```

#### 4.2.3 GGUF 解析测试

```zig
// src/tests/test_gguf.zig
test "gguf v3 header parsing" {
    // 构造 v3 格式的二进制数据
    // 验证解析结果
}

test "gguf metadata kv reading" {
    // 验证各种 KV 类型的读取
}

test "gguf tensor loading" {
    // 验证张量加载和对齐
}
```

### 第三阶段：调试与诊断工具（2-3 周）

#### 4.3.1 计算图转储工具

```zig
// tools/dump_graph.zig
// 功能：将计算图导出为 DOT 格式或文本格式
// 用法：zllama-dump-graph --model model.gguf --output graph.dot
```

#### 4.3.2 Logits 对比工具

```zig
// tools/compare_logits.zig
// 功能：对比两个模型的 logits 输出
// 用法：zllama-compare-logits --ref ref.bin --test test.bin
// 输出：NMSE、最大差异、逐 token 分析
```

#### 4.3.3 性能分析工具

```zig
// tools/profile.zig
// 功能：逐层耗时分析
// 用法：zllama-profile --model model.gguf --prompt "hello"
// 输出：每层耗时、总耗时、tok/s
```

### 第四阶段：回归测试与 CI（1-2 周）

#### 4.4.1 参考输出生成

```bash
# 使用 llama.cpp 生成参考输出
llama-simple -m model.gguf -n 5 "你好" --log-format json > ref_outputs/qwen35_hello.json

# 使用 zllama 生成测试输出
zllama-simple -m model.gguf -n 5 "你好" --log-format json > test_outputs/qwen35_hello.json

# 对比
zllama-compare-logits --ref ref_outputs/qwen35_hello.json --test test_outputs/qwen35_hello.json
```

#### 4.4.2 CI 集成

```yaml
# .github/workflows/test.yml
jobs:
  test:
    steps:
      - run: zig build test              # 单元测试
      - run: zig build test-archs        # 架构前向测试
      - run: zig build test-layers       # 算子测试
      - run: zig build test-gguf         # GGUF 解析测试
```

#### 4.4.3 回归测试清单

| 测试项 | 触发条件 | 验证标准 |
|--------|---------|---------|
| `zig build` | 每次提交 | 编译成功 |
| `zig build test` | 每次提交 | 所有测试通过 |
| `test-archs` | PR 提交 | NMSE < 1e-4 |
| `test-layers` | 每次提交 | 数值误差 < 1e-5 |
| 端到端推理 | 手动/发布 | 输出与 llama.cpp 一致 |

---

## 五、具体执行计划（按优先级）

### P0：已完成（`zig build` 通过）

- [x] **重构 `model.zig`**：定义清晰的 `ModelInterface` 接口契约（已实现 `ModelVTable` + `ModelInstance`）
- [x] **重构 `registry.zig`**：使用接口表替代 `*anyopaque` 模式（已使用 `ModelVTable` 虚表）
- [x] **创建 `src/tests/` 目录**：建立测试基础设施
- [x] **实现 `test_layers.zig`**：RMSNorm、RoPE、SwiGLU 数值测试

### P1：短期（1-2 周）

- [x] **实现 `test_gguf.zig`**：GGUF v2/v3 解析测试（手工构造二进制数据，覆盖元数据、张量、对齐、边界条件）
- [x] **实现 `test_archs.zig`**：架构注册与检测测试（覆盖所有支持的架构枚举、detectArchitecture、GraphBuilder 基本操作）
- [x] **创建 `core/memory.zig`**：内存抽象接口
- [x] **创建 `core/graph_builder.zig`**：图构建上下文
- [x] **重构 `kv_cache.zig`**：实现 `MemoryContext` 接口（通过 toMemoryContext() 适配）

> 注：`test_archs.zig` 中的随机权重前向测试（generateTestGGUF + runForward）需要 GGUF 写入器支持，暂标记为 TODO。
> 当前已实现架构枚举、detectArchitecture、GraphBuilder、KVCacheMemory 等基础测试。

### P2：中期（2-4 周）

- [ ] **实现 `tools/dump_graph.zig`**：计算图可视化
- [ ] **实现 `tools/compare_logits.zig`**：Logits 对比
- [ ] **生成参考输出**：使用 llama.cpp 为各模型生成参考 logits
- [ ] **扩展 `test_archs.zig`**：支持 Qwen2、Qwen3.5 架构
- [ ] **实现 `test_kv_cache.zig`**：KV Cache 功能测试

### P3：长期（4-8 周）

- [ ] **CI 集成**：GitHub Actions 自动运行测试
- [ ] **性能回归测试**：跟踪 tok/s 变化
- [ ] **模糊测试**：随机输入验证稳定性
- [ ] **内存泄漏检测**：集成 Zig 的测试分配器
- [ ] **文档生成**：从测试生成模型支持矩阵

---

## 六、关键设计决策

### 6.1 接口 vs 虚表 vs 编译时多态

```zig
// 方案 A：编译时多态（推荐用于核心路径）
pub fn buildGraph(comptime ModelT: type, model: *ModelT, builder: *GraphBuilder) !void {
    // 编译时确定具体模型
}

// 方案 B：虚表（推荐用于注册表）
pub const ModelVTable = struct {
    buildGraph: *const fn (ctx: *GraphBuilder, data: *anyopaque) anyerror!void,
    getParams: *const fn (data: *anyopaque) *const ModelParams,
};

// 方案 C：标记联合（推荐用于简单场景）
pub const ModelInstance = union(enum) {
    llama: *LlamaModel,
    qwen2: *Qwen2Model,
    qwen35: *Qwen35Model,
};
```

**决策**：注册表使用**方案 B（虚表）**，模型内部使用**方案 A（编译时多态）**。

### 6.2 测试数据生成策略

```
策略 1：完全随机（推荐）
  - 优点：无需外部依赖，可重复
  - 缺点：无法与 llama.cpp 直接对比
  - 适用：算子测试、架构一致性测试

策略 2：从真实模型提取（推荐）
  - 优点：可验证实际正确性
  - 缺点：需要真实模型文件
  - 适用：端到端测试、回归测试

策略 3：手工构造（补充）
  - 优点：可测试边界条件
  - 缺点：工作量大
  - 适用：GGUF 解析测试、错误处理测试
```

### 6.3 NMSE 阈值设定

| 测试类型 | 阈值 | 说明 |
|---------|------|------|
| 算子数值测试 | 1e-5 | 纯数学运算 |
| 架构前向测试（同后端） | 1e-4 | 相同后端，不同实现 |
| 架构前向测试（跨后端） | 1e-3 | CPU vs GPU |
| 与 llama.cpp 对比 | 1e-2 | 不同实现，相同算法 |

---

## 七、与 llama.cpp 的对比验证流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 准备阶段                                                  │
│    ├── 下载真实 GGUF 模型                                    │
│    ├── llama.cpp 推理得到参考输出（JSON 格式）                │
│    └── 提取关键中间张量（可选）                               │
├─────────────────────────────────────────────────────────────┤
│ 2. 逐层验证                                                  │
│    ├── RMSNorm: 输入相同 → 输出相同                          │
│    ├── RoPE: 输入相同 → 输出相同                             │
│    ├── Attention: 输入相同 → 输出相同                        │
│    └── FFN: 输入相同 → 输出相同                              │
├─────────────────────────────────────────────────────────────┤
│ 3. 端到端验证                                                │
│    ├── 相同 prompt → 相同 logits                             │
│    ├── 相同采样 → 相同输出文本                               │
│    └── 性能达标（tok/s 在合理范围内）                         │
├─────────────────────────────────────────────────────────────┤
│ 4. 回归验证                                                  │
│    ├── 代码修改后 → 输出不变                                 │
│    └── 新架构添加 → 不影响已有架构                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 八、风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Zig 0.16.0 测试框架不成熟 | 测试编写困难 | 使用 `std.testing`，必要时封装辅助函数 |
| 随机权重测试无法捕获所有 bug | 漏测 | 结合真实模型测试 + 边界条件测试 |
| 与 llama.cpp 浮点精度差异 | 误报 | 使用宽松阈值（1e-2），关注趋势而非绝对值 |
| 测试执行时间过长 | CI 缓慢 | 分层测试：快速测试（<1min）+ 完整测试（按需） |
| 模型文件过大无法纳入仓库 | 测试数据管理困难 | 使用脚本生成 + git-lfs 管理参考输出 |

---

## 九、成功标准

1. **编译通过**：`zig build` 零错误
2. **测试通过**：`zig build test` 全部通过
3. **架构覆盖**：每个支持的架构都有随机权重前向测试
4. **回归保护**：代码修改不会导致已有测试失败
5. **可调试性**：出现问题时能快速定位到具体层/算子
6. **可复现**：相同输入始终产生相同输出（确定性推理）
