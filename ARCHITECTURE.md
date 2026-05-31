# 系统架构设计

## 1. 总体层次

```
┌─────────────────────────────────────────────┐
│           CLI / HTTP Server 前端             │  (Zig, 交互与采样)
├─────────────────────────────────────────────┤
│           Model Inference Engine            │  (Zig, 计算图构建, KV Cache)
├─────────────────────────────────────────────┤
│    GGUF Loader & Tokenizer (BPE)            │  (Zig, 零拷贝映射)
├─────────────────────────────────────────────┤
│           ggml.zig 安全封装层                │  (Zig opaque + @ptrCast)
├─────────────────────────────────────────────┤
│         ggml C API (CPU / Metal / CUDA)      │  (预编译或源码静态链接)
└─────────────────────────────────────────────┘
```

## 2. 核心模块职责

### 2.1 `ggml.zig` – 底层绑定
- 提供 `pub const c = @cImport(...)` 原始 API。
- 封装 `Context`、`Tensor`、`CGraph`、`Backend`、`GgufContext` 为不透明类型，带 `init/deinit`。
- 所有算子（`mulMat`, `rmsNorm`, `rope`, `conv1d`, `add` 等）以零开销方式暴露。

### 2.2 `gguf.zig` – 模型加载
- 读取 GGUF v2/v3 头部，解析元数据键值对（支持 64 位计数字段）。
- 提取模型超参数：`hidden_size`, `n_heads`, `n_kv_heads`, `n_layers`, `layer_type` 数组等。
- 构建张量索引表（名称 → 偏移/大小），不实际加载数据。
- 使用 ggml 的 `gguf_init_from_file` 或自行 mmap 文件，返回 `GgufContext`。

### 2.3 `model.zig` – 计算图构建
- 根据超参数动态分配权重张量视图（指向 GGUF 文件映射内存）。
- 实现 `build_transformer_layer`，根据 `layer_type` 分派到 `full_attn` 或 `linear_attn`。
- 组装首 token 完整图和增量解码图（共享 KV Cache）。
- 处理输出 logits 和采样。

### 2.4 `kv_cache.zig` – 缓存管理
- 为每层预分配两个 3D 张量 `[max_seq_len, n_kv_heads, head_dim]`，初始零。
- 提供 `set_kv(ctx, cache, new_k, cache_idx)` 写入（`ggml_set_1d` 或 `ggml_cpy` 部分）。
- 提供 `get_kv_view(ctx, cache, current_len)` 返回视图，用于注意力拼接。

### 2.5 `tokenizer.zig` – BPE 分词
- 从 GGUF 元数据中读取 `tokenizer.ggml.tokens` 和 `merges`。
- 实现编码：字符串 → token ID 列表（字节对合并，保留特殊 token）。
- 实现解码：token ID → 字符串（直接查表拼接，处理 UTF-8 边界）。

### 2.6 `sampler.zig` – 采样策略
- 接收 logits 张量，应用温度、top-k、top-p 过滤。
- 返回下一个 token ID。

### 2.7 `backend.zig` – 多后端抽象
- 统一 `Backend` 接口：`init`, `deinit`, `copyToDevice`, `computeGraph`。
- 具体实现：`CpuBackend`, `MetalBackend`, `CudaBackend`，封装 ggml 后端 API。

### 2.8 `main.zig` – 入口
- 解析命令行参数（模型路径、提示词、最大生成长度、后端类型等）。
- 初始化 `std.Io`、分配器、随机数生成器。
- 加载模型和分词器，启动推理循环。

## 3. 数据流（以增量解码为例）

1. 用户输入 prompt → 分词器 → `token_ids`。
2. **首 token 前向**：
   - 构建完整序列图（`seq_len = len(prompt)`）。
   - 计算图中为每一层创建 KV Cache 张量（全零）。
   - 执行图，得到 logits，同时 Cache 中填充所有历史 K/V。
   - 采样得到第一个生成 token。
3. **增量循环**：
   - 将新 token 作为输入 `[1, hidden_size]`。
   - 构建增量图：对每一层，计算当前 token 的 Q/K/V → 对 K/V 应用 RoPE → 写入 Cache 的下一个位置（`cache_idx`）→ 使用 `ggml_view` 获取 `[0..cache_idx+1]` 的完整 K/V → 注意力计算。
   - 执行图，得到新 logits，采样，重复直到遇到 EOS 或达到最大长度。
4. 收集生成的 token IDs → 解码为字符串 → 输出。

## 4. 线程模型

- 主线程负责构建计算图和调度。
- ggml 内部线程池执行计算图（通过 `ggml_graph_compute` 传入线程数）。
- **I/O 接口化**：所有文件操作、控制台输入/输出均通过 `std.Io` 实例，避免阻塞主事件循环（如需异步，可后续扩展）。

## 5. 内存布局

- **模型权重**：位于 mmap 的 GGUF 文件区域，只读，无需额外拷贝。
- **计算中间张量**：在 `ggml_context` 的工作缓冲区中分配（初始化时指定 `mem_size`）。
- **KV Cache**：预分配固定大小，独立于 `ggml_context`（但使用同一分配器），生命周期与模型相同。
- **分词器词表**：常驻内存（`StringHashMap`），大小通常 <500MB，可接受。

