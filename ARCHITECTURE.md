# 架构设计文档

> **说明：** 本文档描述的是**当前实现所采用的架构**，支持多模型架构（Qwen / LLaMA 等）。

## 🧭 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      main.zig (Juicy Main)                  │
│  1. 解析 CLI 参数                                            │
│  2. 读取 GGUF 文件                                           │
│  3. 检测模型架构 (registry.detectArchitecture)               │
│  4. 创建模型实例 (registry.createModel)                      │
│  5. 推理循环 (generate)                                      │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   models/       │  │   layers/       │  │   core/         │
│   qwen.zig      │  │   rms_norm.zig  │  │   (可选)        │
│   llama.zig     │  │   rope.zig      │  │                 │
│   registry.zig  │  │   swiglu.zig    │  │                 │
│                 │  │   attention.zig │  │                 │
│                 │  │   linear.zig    │  │                 │
│                 │  │   embed.zig     │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   ggml.zig      │  │   gguf.zig      │  │   kv_cache.zig  │
│   (C 绑定)      │  │   (GGUF 解析)   │  │   (KV Cache)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   ggml (C 库)   │
                    │   CPU/Metal/    │
                    │   CUDA 后端     │
                    └─────────────────┘
```

## 🧱 多模型架构设计

### 模型抽象接口

`src/model.zig` 定义了模型抽象接口：

- **`Architecture` 枚举**：支持的模型架构（qwen2, llama）
- **`ModelParams` 基类**：所有模型共享的通用参数
- **`ModelWeights` 基类**：通用权重结构
- **`ModelFactory` 类型**：工厂函数类型，支持运行时多态

### 模型注册与工厂

`src/models/registry.zig` 提供：

- **`detectArchitecture()`**：从 GGUF 元数据检测模型架构
- **`createModel()`**：根据架构创建模型实例（返回 `*anyopaque`）
- **`forwardModel()`**：执行前向计算（switch 分发）
- **`deinitModel()`**：释放模型资源

### 共享算子库

`src/layers/` 下的每个文件导出纯粹的**算子函数**，接受 `ggml.Tensor` 并返回新的张量：

- `rms_norm.zig`：RMSNorm 归一化
- `rope.zig`：RoPE 位置编码
- `swiglu.zig`：SwiGLU 前馈网络
- `attention.zig`：缩放点积注意力（含 GQA）
- `linear.zig`：线性投影
- `embed.zig`：Token 嵌入

### 具体模型实现

每个模型实现文件（如 `qwen.zig`、`llama.zig`）包含：

- 该模型的超参数结构体（从 GGUF 读取）
- 权重张量的持有
- `init`、`deinit`、`forward` 方法
- 使用 `layers/` 中的共享算子构建计算图

## 数据流与内存管理

```
main(init)
  ├── 获得 io = init.io
  ├── 读取 GGUF 文件
  ├── 解析 GGUF 元数据
  ├── 检测架构 → 创建模型
  ├── 初始化 KV Cache
  ├── 编码 prompt
  ├── 构建计算图 → 执行 → 采样
  └── 增量生成 → 输出文本

内存管理：
  - 权重：通过 setDataPtr 指向 GGUF 文件数据（零拷贝）
  - KV Cache：预分配连续内存，通过 ggml_view_* 切片
  - 计算图：使用 gallocr 分配中间张量内存
```

## 可验证测试模式

| 测试 | 说明 |
|------|------|
| `zig build test` | 运行所有测试 |
| `./qwen --model model.gguf` | 交互式推理 |
| `./qwen --model model.gguf -p "Hello" -n 10` | 单次生成 |

## 扩展新模型

新增模型只需：

1. 在 `src/models/` 下创建新文件（如 `mistral.zig`）
2. 实现 `init`、`deinit`、`forward`、`params`、`weights` 方法
3. 在 `model.zig` 的 `Architecture` 枚举中添加新类型
4. 在 `registry.zig` 的 `fromString`、`createModel`、`forwardModel`、`deinitModel` 中添加对应 case
