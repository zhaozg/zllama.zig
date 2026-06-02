# 修复程序崩溃：空指针解引用 (EXC_BAD_ACCESS, address=0x0)

## 问题

```
debug: reshape3d: [10,80,1,1] -> [10,10,8] nelem=800 expected=800
debug: reshape2d: [10,8,256,1] -> [2048,10] nelem=20480 expected=20480
ATTN_OUT_DEBUG: layer=23 cur ne=10240, attn_out ne=10240
POST_ATTN_DEBUG: layer=23 cur ne=10240, post_attn ne=10240
debug: reshape2d: [1024,1,1,1] -> [1024,1] nelem=1024 expected=1024
FFN_OUT_DEBUG: layer=23 cur ne=10240, ffn_out ne=10240
FFN_OUT_DEBUG: layer=23 cur ne=10240, ffn_out ne=10240
debug: reshape2d: [1024,1,1,1] -> [1024,1] nelem=1024 expected=1024
info(main): Computing forward pass...
load_backend: loaded BLAS backend from /usr/local/Cellar/ggml/0.13.1/libexec/libggml-blas.so
load_backend: loaded CPU backend from /usr/local/Cellar/ggml/0.13.1/libexec/libggml-cpu.so
Process 51848 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
    frame #0: 0x0000000100485601 libggml-base.0.dylib`dequantize_row_q6_K + 327
libggml-base.0.dylib`dequantize_row_q6_K:
->  0x100485601 <+327>: movss  %xmm4, (%rsi,%r11,4)
    0x100485607 <+333>: movsbl 0x2(%r9,%rbx), %eax
    0x10048560d <+339>: xorps  %xmm3, %xmm3
    0x100485610 <+342>: cvtsi2ss %eax, %xmm3
  thread #2, stop reason = EXC_BAD_ACCESS (code=1, address=0x3000)
    frame #0: 0x0000000100485601 libggml-base.0.dylib`dequantize_row_q6_K + 327
libggml-base.0.dylib`dequantize_row_q6_K:
->  0x100485601 <+327>: movss  %xmm4, (%rsi,%r11,4)
    0x100485607 <+333>: movsbl 0x2(%r9,%rbx), %eax
    0x10048560d <+339>: xorps  %xmm3, %xmm3
    0x100485610 <+342>: cvtsi2ss %eax, %xmm3
  thread #3, stop reason = EXC_BAD_ACCESS (code=1, address=0x6000)
    frame #0: 0x0000000100485601 libggml-base.0.dylib`dequantize_row_q6_K + 327
libggml-base.0.dylib`dequantize_row_q6_K:
->  0x100485601 <+327>: movss  %xmm4, (%rsi,%r11,4)
    0x100485607 <+333>: movsbl 0x2(%r9,%rbx), %eax
    0x10048560d <+339>: xorps  %xmm3, %xmm3
    0x100485610 <+342>: cvtsi2ss %eax, %xmm3
  thread #4, stop reason = EXC_BAD_ACCESS (code=1, address=0x9000)
    frame #0: 0x0000000100485601 libggml-base.0.dylib`dequantize_row_q6_K + 327
libggml-base.0.dylib`dequantize_row_q6_K:
->  0x100485601 <+327>: movss  %xmm4, (%rsi,%r11,4)
    0x100485607 <+333>: movsbl 0x2(%r9,%rbx), %eax
    0x10048560d <+339>: xorps  %xmm3, %xmm3
    0x100485610 <+342>: cvtsi2ss %eax, %xmm3
Target 0: (qwen) stopped.
(lldb) bt
* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
  * frame #0: 0x0000000100485601 libggml-base.0.dylib`dequantize_row_q6_K + 327
    frame #1: 0x00000001204cc143 libggml-cpu.so`ggml_compute_forward_get_rows + 476
    frame #2: 0x0000000120497c8b libggml-cpu.so`ggml_graph_compute_thread + 2511
    frame #3: 0x0000000120497299 libggml-cpu.so`ggml_graph_compute.omp_outlined + 274
    frame #4: 0x00000001005df969 libomp.dylib`__kmp_invoke_microtask + 153
    frame #5: 0x0000000100591fce libomp.dylib`__kmp_invoke_task_func + 358
    frame #6: 0x000000010058d7ee libomp.dylib`__kmp_fork_call + 6258
    frame #7: 0x00000001005817d2 libomp.dylib`__kmpc_fork_call + 285
    frame #8: 0x0000000120496d69 libggml-cpu.so`ggml_graph_compute + 274
    frame #9: 0x000000012049920c libggml-cpu.so`ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*) + 120
    frame #10: 0x000000010046bf0d libggml-base.0.dylib`ggml_backend_graph_compute + 18
    frame #11: 0x000000010013ed6d qwen`ggml.CGraph.compute(self=0x0000000157405fc0, _n_threads=6) at ggml.zig:418:45
    frame #12: 0x000000010013f8be qwen`main.InferenceEngine.generate(self=0x00007ff7bfefbd58, prompt=(ptr = "Hello, how are you?", len = 19), max_tokens=256) at main.zig:313:26
    frame #13: 0x0000000100140e49 qwen`main.main(init=<unavailable>) at main.zig:466:20
    frame #14: 0x0000000100141641 qwen`start.callMain at start.zig:737:30
    frame #15: 0x0000000100141479 qwen`start.callMainWithArgs at start.zig:638:20
    frame #16: 0x0000000100141294 qwen`start.main(c_argc=3, c_argv=0x00007ff7bfefd650, c_envp=0x00007ff7bfefd670) at start.zig:663:28
    frame #17: 0x00007ff80776fb28 dyld`start + 3240
(lldb)
```

### 进度鉴定：重大突破，仅剩“最后一公里”！

**这是一个巨大的进步！** 你的计算图（Forward Graph）已经 **100% 成功构建**，没有任何维度或断言错误。程序已经顺利进入了 `ggml_backend_graph_compute` 实际执行计算的阶段。

现在的崩溃（`EXC_BAD_ACCESS, address=0x0`）不再是算法或维度推导逻辑错误，而是纯粹的 **C/Zig 内存管理问题**：**空指针解引用 (NULL Pointer Dereference)**。

---

### 🛑 崩溃核心原因定位

崩溃堆栈显示：
```text
frame #0: libggml-base.0.dylib`dequantize_row_q6_K + 327
frame #1: libggml-cpu.so`ggml_compute_forward_get_rows + 476
```
在 LLM 推理中，`ggml_get_rows` 几乎总是用于 **Token Embedding 查找**（根据输入的 token ID 从权重矩阵中提取向量）。

崩溃的 `address=0x0` 明确表示：在尝试读取或写入内存时，基地址是 `NULL`。在 `ggml_get_rows` 操作中，这只能是以下两种情况之一：
1. **源权重张量**（例如 `token_embd.weight`）的 `.data` 指针是 `NULL`。
2. **目标输出张量**（`get_rows` 操作创建的用于存放结果的张量）的 `.data` 指针是 `NULL`。

结合你之前刚修复了 KV Cache 的 `setZero` (也是因为 `data == NULL`)，**几乎可以断定：你的模型权重张量或中间计算张量，没有被正确分配物理内存。**

---

### 🛠️ 问题处理方案

请按照以下步骤排查和修复内存分配逻辑：

#### 方案 1：使用 GGML 推荐的统一内存分配（强烈推荐）
在现代 GGML/llama.cpp 架构中，最安全、最不易出错的方式是让 Backend 统一分配整个计算图上下文 (`ctx`) 中所有张量的内存。

请检查你的 `InferenceEngine.init` 或模型加载逻辑，确保流程如下：
```zig
// 1. 创建 backend (例如 CPU backend)
var backend = ggml.backend_cpu_init();

// 2. 创建计算图上下文 (ctx)
var ctx = ggml.init(...);

// 3. 加载模型权重并构建整个计算图 (此时所有 tensor 的 .data 都是 null)
// ... 你的 load_model 和 build_forward_graph 逻辑 ...

// 4. 【关键步骤】在开始计算前，让 backend 为 ctx 中所有 .data == null 的张量分配内存
// 这会自动处理权重、KV cache、以及 get_rows 等操作的输出张量
ggml.backend_alloc_ctx_tensors(ctx, backend);

// 5. 现在所有张量都有了合法的内存，可以安全地进行计算
graph.compute(backend, n_threads);
```
*注意：如果你在加载 GGUF 时是手动将文件中的数据 `memcpy` 到张量中，请确保你在 `memcpy` **之前**，已经通过 `allocator` 或 `backend_alloc` 为该张量分配了内存（即 `.data` 不为 `null`）。*

#### 方案 2：诊断打印（定位具体是哪个张量为 NULL）
如果你使用的是手动内存管理，或者想精确定位问题，请在 `model.zig` 中调用 `ggml.get_rows` 的地方（通常是 Embedding 层）添加防御性检查：

```zig
// 假设你的代码类似这样：
// const embd = ggml.get_rows(ctx, model.token_embd, input_tokens);

// 【添加诊断打印】
std.debug.print("DEBUG GET_ROWS:\n", .{});
std.debug.print("  Weight (token_embd) data: {any}, type: {s}\n", .{
    model.token_embd.data, @tagName(model.token_embd.type)
});
std.debug.print("  Input (tokens) data: {any}\n", .{ input_tokens.data });
std.debug.print("  Output (embd) data: {any}\n", .{ embd.data });

// 捕获 NULL 指针
if (model.token_embd.data == null) {
    std.debug.print("FATAL: token_embd.data is NULL! Memory not allocated.\n", .{});
    std.process.exit(1);
}
if (embd.data == null) {
    std.debug.print("FATAL: get_rows output data is NULL! Memory not allocated.\n", .{});
    std.process.exit(1);
}
```

#### 方案 3：检查 Zig 内存分配器的生命周期 (Arena Allocator 陷阱)
如果你在加载模型时使用了 `std.heap.ArenaAllocator`，请务必确保：**在推理生成阶段（`generate` 函数被调用时），这个 Arena 还没有被 `deinit()` 或 `reset()`**。
如果 Arena 被提前释放，所有通过它分配的 `.data` 指针都会变成悬垂指针（Dangling Pointer），在多线程计算时极易触发 `EXC_BAD_ACCESS`。
- **对策**：确保模型权重的内存分配器生命周期覆盖整个 `InferenceEngine` 的存在周期，或者在加载后将其 `toOwnedAllocator()` 转移给 Engine 长期持有。

---

### 🚀 下一步行动

1. 优先应用 **方案 2** 的打印诊断，100% 确认是 Weight 还是 Output 的 `.data` 为 `NULL`。
2. 根据结果，应用 **方案 1** 或 **方案 3** 修复内存分配逻辑。

**总结**：你已成功攻克了 Qwen3.5 混合架构最复杂的维度推导和计算图构建难题。现在的问题只是标准的 C/Zig 内存生命周期管理。补上这块内存分配的拼图，你的引擎就能成功吐出第一个 Token！
