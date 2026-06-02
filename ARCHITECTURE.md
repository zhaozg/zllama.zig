# 架构设计文档

> **说明：** 本文档描述的是**当前实现所采用的架构**，而非最终完美设计。架构会随阶段推进而逐步复杂化。每个阶段都会更新本文档。

## 🧭 架构演进路线（阶段式，非终极设计）

### 阶段 P0：最小绑定层

```
┌─────────────────────────────────────────┐
│              build.zig                  │
│  - 添加 ggml .c 源文件                  │
│  - 链接 libc                            │
│  - 可选 Metal / CUDA 标志               │
└─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              ggml.zig                    │
│  仅实现：                                │
│  - ggml_init_params / ggml_init          │
│  - ggml_free                             │
│  - ggml_version                          │
│  注意：ggml 自身不依赖 Zig Io，可直接调用│
│  系统 mmap 进行文件加载（绕过 Io 接口化）│
└──────────────────────────────────────── ─┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│           main.zig (Juicy Main)           │
│  pub fn main(init: std.process.Init) !void│
│  - const io = init.io                     │
│  - 使用 std.Io.Dir.cwd().openFile(io, ...)│
│  - 时间测量：std.Io.Clock.now(.awake, io) │
└───────────────────────────────────────────┘
```

### 阶段 P1：GGUF 解析器

```
┌─────────────────────────────────────────┐
│              gguf.zig                   │
│  - readHeader()：解析 magic, version    │
│  - readMetadata()：解析 KV 元数据       │
│  - readTensorInfos()：解析张量索引      │
│  - validate()：检查 version 兼容性      │
│  不加载实际张量数据（仅索引）           │
└─────────────────────────────────────────┘
```

GGUF 文件布局（参考 llama.cpp/gguf-py）：
```
[Header: magic, version, tensor_count, metadata_kv_count]
[Metadata: key-value pairs]
[Tensor Infos: name, n_dim, dims, offset]
[Tensor Data: 32-byte aligned]
```

v2/v3 差异：v3 的 `tensor_count`、`metadata_kv_count` 为 64 位，且张量数据必须 32 字节对齐。

### 阶段 P2-P3：单层计算图构建

```
┌─────────────────────────────────────────┐
│              model.zig                  │
│  GraphBuilder 结构：                    │
│  - ctx: *ggml_context                   │
│  - cgraph: ggml_cgraph                  │
│  - add_tensor(t: *ggml_tensor)          │
│  - add_op(op, inputs...)                │
│                                         │
│  每层提供：                             │
│  - forward_full_attn()                  │
│  - forward_rms_norm()                   │
│  - forward_rope()                       │
│  - forward_ffn_swiglu()                 │
└─────────────────────────────────────────┘
```

**计算图构建模式（ggml 推荐）：**
1. 创建上下文 `ggml_init(params)`
2. 为模型权重创建张量 `ggml_new_tensor_*()`
3. 构建算子图 `ggml_mul_mat()`, `ggml_add()`, `ggml_norm()`...
4. 设置输出 `ggml_set_output()`
5. 调用 `ggml_graph_compute_with_ctx()` 执行

### 阶段 P4：KV Cache 与增量解码

```
┌─────────────────────────────────────────┐
│            kv_cache.zig                  │
│  KVCache 结构：                          │
│  - k: *ggml_tensor (预分配)              │
│  - v: *ggml_tensor                       │
│  - head_dim: usize                       │
│  - seq_len: usize                        │
│                                          │
│  get_k_view(pos: usize) -> *ggml_tensor  │
│    → ggml_view_2d(k, ...)                │
│  get_v_view(pos: usize) -> *ggml_tensor  │
│  set_kv(pos, k_tensor, v_tensor)         │
└─────────────────────────────────────────┘
```

**关键约束：** `ggml_view_*` 不复制数据，必须在缓存预分配时保证内存连续。

### 阶段 P5-P6：混合架构与多后端

Qwen 3.5 架构需从 GGUF 元数据获取：
- `layer_type` 数组：`"full_attention"` 或 `"linear_attention"`
- `attn_output_gate`：是否存在门控张量（Qwen 3.5 特有）
- `n_kv_heads`：GQA 时的 KV head 数量

线性注意力需要 `ggml_conv_1d` 算子。

## 数据流与内存管理

Io 实例（来自 Juicy Main 的 init.io）必须贯穿整个推理链路：
```
main(init)
  ├── 获得 io = init.io
  ├── 传递给 GGUF 文件加载：std.Io.Dir.cwd().openFile(io, path, .{})
  ├── 传递给 Tokenizer（如需从文件读取词表）
  └── 传递给所有需要 I/O 的调用链

文件 mmap (ggml_backend_file，此操作不受 Io 影响，直接使用系统 mmap)
      │
      ▼
gguf.zig → 解析元数据，获取权重张量 offset
      │
      ▼
model.zig → 创建 ggml_context，通过 offset 映射张量（零拷贝）
      │
      ▼
ggml 计算图 → CPU / Metal / CUDA 执行
      │
      ▼
输出采样 → 下一 token id
```

## 可验证测试模式

每个里程碑需提供对应的测试：

| 阶段 | 测试 |
|------|------|
| P0 | `zig build test_ggml` 调用 `ggml_version()` |
| P1 | `./qwen --model model.gguf --info` |
| P2 | `./qwen --model model.gguf --test-embed` |
| P3 | `./qwen --model model.gguf --test-layer 1` |
| P4 | `./qwen --model model.gguf --prompt "Hello" -n 1` |
```

