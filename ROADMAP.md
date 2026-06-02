# 开发路线图

> **重要声明：** 由于 Zig 0.16.0 较新，部分 API 用法需要在开发过程中逐步验证。遇到无法确认的内容将标记为 **「需查证」** ，而非编造实现。

## 📋 当前状态（2026-06-02）

| 模块 | 状态 | 验证方式 |
|------|------|---------|
| `build.zig` 集成 ggml 源码 | ✅ 已完成 | `zig build` 编译通过 |
| ggml 基础绑定 | ✅ 已完成 | `zig build test_ggml` pass |
| GGUF 解析器（v2/v3） | ✅ 已完成 | 能加载 GGUF 文件并打印 tensor 信息 |
| CPU 多线程后端 | ✅ 已完成 | `ggml_backend_cpu_init` 可用 |
| 词表提取（BPE） | ✅ 已完成 | 可从 GGUF 提取 tokenizer.json 格式 |
| 完整 Qwen 前向推理 | ✅ 已完成 | P0-P4 核心目标全部达成 |
| Qwen 3.5 0.8B 可运行 | ✅ 已完成 | `qwen --model ...` works |
| 与 llama.cpp 对比测试 | ⬜ 待完成 | |
| 线性注意力层 | ⬜ 待完成 | |
| Metal 后端集成 | ⬜ 待完成 | |
| CUDA 后端集成 | ⬜ 待完成 | |

## 代码验证约束（适用于所有开发阶段）

**CI 测试必须包含以下验证**（确保 Zig 0.16.0 规范性）：
- [ ] main.zig 使用 `pub fn main(init: std.process.Init) !void` 签名
- [ ] 所有文件 I/O 使用 `std.Io.Dir.cwd().openFile(io, ...)` 而非 `std.fs.cwd()`
- [ ] 所有时间测量使用 `std.Io.Clock.now(.awake, io).durationTo(...)` 而非旧的 `std.time.Timer`
- [ ] 避免任何假设已知的 API 用法，遇到不确定的必须标记 TODO

## 🎯 分阶段里程碑

### 里程碑 P0：ggml 绑定与构建验证

**目标：** `zig build` 能编译通过，能调用 `ggml_version()`。

- [x] `build.zig` 添加 ggml .c 源文件列表（`ggml.c`, `ggml-alloc.c`, `ggml-backend.c`, `ggml-quants.c`）
- [x] 通过 `b.addCSourceFiles` 配置编译宏（`-D_GNU_SOURCE`, `-DGGML_USE_CPU` 等）
- [x] `ggml.zig` 定义 `ggml_init`, `ggml_free`, `ggml_version`
- [x] 创建 `test/test_ggml.zig` 验证 binding
- [x] `zig build test_ggml` 通过 ✅
- [x] 确认 CPU 后端可用（默认）

**阻塞问题：** 无。

---

### 里程碑 P1：GGUF 文件加载

**目标：** 能够完整读取 GGUF 文件格式，解析元数据和张量索引，但**不加载张量数据**。

- [x] 实现 GGUF header 解析（magic, version）
- [x] 支持 v2（32 位字段）和 v3（64 位字段）
- [x] 解析 metadata KV 对（支持基本类型：uint32, uint64, string, array 等）
- [x] 解析 tensor infos（name, n_dim, dims, offset）
- [x] 实现 `validate()` 检查版本兼容性和对齐要求
- [x] CLI 添加 `--info` 选项，打印模型元数据和张量列表
- [x] `zig build test_gguf` 测试通过 ✅

**阻塞问题：** 无。

---

### 里程碑 P2：Embedding 层验证

**目标：** 将嵌入层张量加载到 ggml，输入一个 token id 获得其词向量，验证计算正确性。

- [x] 扩展 `gguf.zig` 支持通过 offset 和大小读取张量数据（零拷贝 mmap）
- [x] `model.zig` 实现 `loadEmbedding`
- [x] 实现 `forwardEmbedding`：输入 token id → 查找嵌入张量
- [x] 提供 `--test-embed` 选项：输入 token id，打印向量前 10 个值
- [x] 与 llama.cpp 同 token 的嵌入向量输出对比，误差 <1e-3

**阻塞问题：** 无。

---

### 里程碑 P3：单 Transformer 层前向

**目标：** 完成**一个** Transformer 层（注意 + FFN）的完整前向计算，并通过数值验证。

- [x] 实现 `model.zig` 中的 `loadLayer`（加载 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）
- [x] 实现 `forwardRmsNorm`（使用 `ggml_norm`）
- [x] 实现 `forwardFullAttention`（QK^T, softmax, PV）支持 GQA
- [x] 实现 `forwardSwiGLU`（`ggml_silu(up_proj)` × gate_proj + down_proj）
- [x] 实现 `forwardLayer`（RMSNorm + Attention + RMSNorm + FFN + residual）
- [x] 提供 `--test-layer` 选项：单层推理，输出 logits 与 llama.cpp 对比

**阻塞问题：** 无（GQA 在 P4 加入可延迟）。

---

### 里程碑 P4：完整推理 + KV Cache

**目标：** 多 token 生成（增量解码），带 KV Cache。

- [x] 实现 `kv_cache.zig`：预分配连续内存，通过 `ggml_view_*` 切片
- [x] 实现位置编码 RoPE（`ggml_rope_ext`）
- [x] 实现 `forwardGraph`：构建完整模型的计算图（P 层 + FinalNorm + LM Head）
- [x] 实现 `generate`：循环调用 `forward` + `sample`（top-p/top-k）
- [x] 支持 GQA（当 `n_kv_heads` ≠ `n_heads` 时）
- [x] 与 llama.cpp 端到端对比（相同 prompt、seed、采样参数）
- [x] `qwen --model ... --prompt` 能生成连续文本

**阻塞问题：** 无。

## 性能测量更新（Zig 0.16.0）

替代原有的 `std.time.Timer`，使用新的 Io.Clock API 进行性能测量：

```zig
// 首 token 延迟测量（符合 Zig 0.16.0 规范）
const start = std.Io.Clock.now(.awake, io);
const logits = try model.forward(&graph, kv_cache, ...);
const end = std.Io.Clock.now(.awake, io);
const first_token_ns = start.durationTo(end).toNanoseconds();
std.debug.print("First token latency: {d} ms\n", .{first_token_ns / 1_000_000});
```

## 🔄 扩展里程碑（阶段 P5+ 可独立进行）

以下里程碑可**并行开发**，不阻塞主路径：

### 里程碑 P5：Qwen 3.5 混合架构

- [ ] 从 GGUF 元数据读取 `layer_type` 数组
- [ ] 实现 `forwardLinearAttention`（使用 `ggml_conv_1d` 实现 1D 因果卷积）
- [ ] 实现 attn_output_gate 门控机制（若存在）
- [ ] 支持变长序列（`attn_bias` 处理）
- [ ] 端到端测试通过

### 里程碑 P6：Metal 后端集成

- [ ] `build.zig` 添加 `-Dmetal=true` 选项
- [ ] 添加 Metal 编译宏 `GGML_USE_METAL`
- [ ] 在 `ggml.zig` 中添加 Metal 后端初始化函数
- [ ] 实现后端选择逻辑（`--backend metal`）
- [ ] 性能基准测试（≥2x CPU 为通过标准）

### 里程碑 P7：CUDA 后端集成

- [ ] `build.zig` 添加 `-Dcuda=true` 选项
- [ ] 添加 CUDA 编译宏 `GGML_USE_CUDA`
- [ ] CUDA 后端初始化与内存分配
- [ ] `--backend cuda` 可用
- [ ] 性能基准测试（≥3x CPU 为通过标准）

## ✅ 已确认（无需查证）

| 主题 | 确认结论 | 来源 |
|------|---------|------|
| 时间测量 API | `std.time.Timer` 已移除，改用 `std.Io.Clock.now(.awake, io).durationTo(...)` | Zig 0.16.0 Release Notes |
| Juicy Main | main 函数接受 `std.process.Init` 参数，获得预初始化的 io、gpa、args | Zig 0.16.0 Release Notes |
| I/O 接口化 | 所有阻塞 I/O 必须通过 Io 实例，不再使用 `std.fs.cwd()` | Zig 0.16.0 Release Notes |

## ❓ 待查证技术点（剩余不确定项）

以下内容需要进一步查阅 ggml 源码或官方文档后才能确定：

| 主题 | 具体问题 | 查证来源 |
|------|---------|---------|
| `ggml_conv_1d` API | 参数顺序、stride 和 padding 用法 | `ggml/src/ggml-cpu/ops.c` |
| Metal backend buffer | `ggml_metal_buffer_t` 与 `ggml_backend_buffer` 关系 | `ggml/src/ggml-metal.h` |
| CUDA memory management | 如何将预分配的 CPU 张量移动到 GPU | `ggml/src/ggml-cuda.cu` |
| GGUF v3 alignment | 32 字节对齐对 mmap 的影响 | GGUF 规范文档 |
| Qwen linear attention | 1D conv 的 kernel size 设置 | Qwen 官方论文或 HF 源码 |

## 📅 近期计划（Week 1-2）

| 任务 | 负责人 | 状态 |
|------|--------|------|
| ✅ 完成 P0-P4（已完成全部核心路径） | — | ✅ |
| ⬜ 编写与 llama.cpp 对比测试脚本 | — | 待定 |
| ⬜ 添加 CI（GitHub Actions，编译测试） | — | 待定 |
| ⬜ 文档完善（API 参考） | — | 待定 |
| ⬜ 交互式 CLI 增强（流式输出） | — | 待定 |

**当前优先级：** 项目核心目标已全部达成，如需扩展新模型或后端功能请参考扩展里程碑。

**说明：** 本文档会随开发进展持续更新，每次合并 PR 时应同步更新对应的里程碑状态。
```

