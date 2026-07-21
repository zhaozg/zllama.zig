# ggml.zig 绑定设计

> **项目：** zllama.zig — 纯 Zig 实现的多模型本地推理引擎

## 1. 设计原则

- **全能力暴露**：`ggml.c` 命名空间提供所有原始 C API。
- **安全封装**：opaque 类型 + 错误联合 + defer 资源管理。
- **零开销**：封装函数经 `@ptrCast(@alignCast(...))` 编译后与直接调用 C 函数一致。
- **模块化**：按功能拆分为独立子模块，每个文件不超过 600 行。
- **构建集成**：所有 C 依赖和宏定义在 `build.zig` 中声明，源码无绝对路径。

## 2. 模块结构

```
src/ggml.zig          # 模块入口（facade），重新导出所有子模块
src/ggml/
├── c.zig             # 原始 C API 导入和类型枚举（Type, GgufValueType, GgufValue）
├── context.zig       # ggml_context 封装（Context）
├── tensor.zig        # ggml_tensor 封装（Tensor）
├── graph.zig         # ggml_cgraph 封装（CGraph）
├── backend.zig       # Backend 与 Gallocr 封装
├── ops.zig           # 计算图操作函数（mulMat, rmsNorm, ropeExt, conv1d, ...）
├── quantize.zig      # 量化函数（quantizeChunk, quantizeTensor, quantizedSize, ...）
├── threadpool.zig    # 线程池封装
├── gguf.zig          # GGUF 解析封装
└── utils.zig         # 工具函数（version, CpuFeatures, recommendedThreads）
```

### 2.1 `c.zig` — 原始 C API

```zig
pub const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-cpu.h");
    @cInclude("ggml-backend.h");
    @cInclude("ggml-alloc.h");
    @cInclude("gguf.h");
});

pub const Type = enum(c.ggml_type) { ... };
pub const GgufValueType = enum(c.gguf_type) { ... };
pub const GgufValue = union(enum) { ... };
```

### 2.2 `context.zig` — Context 封装

```zig
pub const Context = opaque {
    pub fn init(mem_size: usize) !*Context { ... }
    pub fn initNoAlloc(mem_size: usize) !*Context { ... }
    pub fn deinit(self: *Context) void { ... }
    pub fn reset(self: *Context) void { ... }
    pub fn newF32(self: *Context, value: f32) *Tensor { ... }
    pub fn newTensor1d(self: *Context, t: Type, ne0: i64) !*Tensor { ... }
    pub fn newTensor2d(self: *Context, t: Type, ne0: i64, ne1: i64) !*Tensor { ... }
    pub fn newTensor3d(self: *Context, t: Type, ne0: i64, ne1: i64, ne2: i64) !*Tensor { ... }
    pub fn newTensor4d(self: *Context, t: Type, ne0: i64, ne1: i64, ne2: i64, ne3: i64) !*Tensor { ... }
    pub fn view1d(self: *Context, a: *Tensor, ne0: i64, offset: usize) *Tensor { ... }
    pub fn view2d(self: *Context, a: *Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) *Tensor { ... }
    pub fn view3d(self: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) *Tensor { ... }
    pub fn view4d(self: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor { ... }
    pub fn usedMem(self: *Context) usize { ... }
    pub fn totalMem(self: *Context) usize { ... }
    pub fn usage(self: *Context) struct { used: usize, total: usize, ratio: f64 } { ... }
    ...
};
```

### 2.3 `tensor.zig` — Tensor 封装

```zig
pub const Tensor = opaque {
    // 元数据查询
    pub fn getName(self: *Tensor) [:0]const u8 { ... }
    pub fn setName(self: *Tensor, name: [:0]const u8) void { ... }
    pub fn getOpName(self: *Tensor) [:0]const u8 { ... }
    pub fn ne(self: *Tensor) [4]i64 { ... }
    pub fn nb(self: *Tensor) [4]usize { ... }
    pub fn strides(self: *Tensor) [4]usize { ... }
    pub fn shape(self: *Tensor) [4]i64 { ... }
    pub fn dataType(self: *Tensor) Type { ... }
    pub fn nElems(self: *Tensor) i64 { ... }
    pub fn nBytes(self: *Tensor) usize { ... }
    pub fn elementSize(self: *Tensor) usize { ... }
    pub fn isContiguous(self: *Tensor) bool { ... }

    // 数据访问
    pub fn dataBytes(self: *Tensor) []u8 { ... }
    pub fn dataF32(self: *Tensor) []f32 { ... }
    pub fn dataI32(self: *Tensor) []i32 { ... }
    pub fn dataF16(self: *Tensor) []u16 { ... }
    pub fn dataBF16(self: *Tensor) []u16 { ... }
    pub fn setDataPtr(self: *Tensor, data: []u8) void { ... }
    pub fn setZero(self: *Tensor) void { ... }
    pub fn dataGet(self: *const Tensor, comptime T: type, allocator: std.mem.Allocator) ![]T { ... }
    pub fn dataSet(self: *Tensor, comptime T: type, data: []const T) !void { ... }

    // 后端数据搬运
    pub fn backendGet(self: *const Tensor, data: []u8, offset: usize) void { ... }
    pub fn backendSet(self: *Tensor, data: []const u8, offset: usize) void { ... }

    // 计算图操作（方法风格）
    pub fn mulMat(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor { ... }
    pub fn mul(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor { ... }
    pub fn add(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor { ... }
    pub fn sub(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor { ... }
    pub fn rmsNorm(self: *Tensor, ctx: *anyopaque, eps: f32) *Tensor { ... }
    pub fn norm(self: *Tensor, ctx: *anyopaque, eps: f32) *Tensor { ... }
    pub fn scale(self: *Tensor, ctx: *anyopaque, s: f32) *Tensor { ... }
    pub fn scaleBias(self: *Tensor, ctx: *anyopaque, s: f32, b: f32) *Tensor { ... }
    pub fn softMax(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn softMaxExt(self: *Tensor, ctx: *anyopaque, mask: ?*Tensor, scaling: f32, max_bias: f32) *Tensor { ... }
    pub fn silu(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn sigmoid(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn gelu(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn geluErf(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn geluQuick(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn sqr(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn tanh(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn relu(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn clamp(self: *Tensor, ctx: *anyopaque, min: f32, max: f32) *Tensor { ... }
    pub fn permute(self: *Tensor, ctx: *anyopaque, axis0: i32, axis1: i32, axis2: i32, axis3: i32) *Tensor { ... }
    pub fn cont(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn cont2d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64) *Tensor { ... }
    pub fn cont3d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64) *Tensor { ... }
    pub fn cont4d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor { ... }
    pub fn reshape2d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64) *Tensor { ... }
    pub fn reshape3d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64) *Tensor { ... }
    pub fn reshape4d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor { ... }
    pub fn pad(self: *Tensor, ctx: *anyopaque, p0: i32, p1: i32, p2: i32, p3: i32) *Tensor { ... }
    pub fn roll(self: *Tensor, ctx: *anyopaque, p0: i32, p1: i32, p2: i32, p3: i32) *Tensor { ... }
    pub fn pool2d(self: *Tensor, ctx: *anyopaque, op: c_uint, k0: i32, k1: i32, s0: i32, s1: i32, p0: f32, p1: f32) *Tensor { ... }
    pub fn getRows(self: *Tensor, ctx: *anyopaque, b: *Tensor) *Tensor { ... }
    pub fn concat(self: *Tensor, ctx: *anyopaque, b: *Tensor, dim: i32) *Tensor { ... }
    pub fn dupTensor(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn view2d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, nb1: usize, offset: usize) *Tensor { ... }
    pub fn view3d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) *Tensor { ... }
    pub fn view4d(self: *Tensor, ctx: *anyopaque, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor { ... }
    pub fn conv2d(self: *Tensor, ctx: *anyopaque, kernel: *Tensor, s0: i32, s1: i32, p0: i32, p1: i32, d0: i32, d1: i32) *Tensor { ... }
    pub fn im2col(self: *Tensor, ctx: *anyopaque, kernel: *Tensor, s0: i32, s1: i32, p0: i32, p1: i32, d0: i32, d1: i32, is_2d: bool, dst_type: Type) *Tensor { ... }
    pub fn ssmConv(self: *Tensor, ctx: *anyopaque, kernel: *Tensor) *Tensor { ... }
    pub fn sumRows(self: *Tensor, ctx: *anyopaque) *Tensor { ... }
    pub fn ropeExt(self: *Tensor, ctx: *anyopaque, pos: *Tensor, freq_factors: ?*Tensor, n_dims: i32, mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) *Tensor { ... }
    ...
};
```

### 2.4 `graph.zig` — CGraph 封装

```zig
pub const CGraph = opaque {
    pub fn init(ctx: *Context) !*CGraph { ... }
    pub fn initReserved(ctx: *Context, n_nodes: i32) !*CGraph { ... }
    pub fn buildForwardExpand(self: *CGraph, tensor: *Tensor) void { ... }
    pub fn compute(self: *CGraph, n_threads: i32) !void { ... }
    pub fn nNodes(self: *CGraph) i32 { ... }
    pub fn getNode(self: *CGraph, i: i32) *Tensor { ... }
    pub fn print(self: *CGraph) void { ... }
    pub fn reset(self: *CGraph) void { ... }
    pub fn dup(ctx: *Context, cgraph: *CGraph) *CGraph { ... }
    pub fn getTensor(self: *CGraph, name: [:0]const u8) ?*Tensor { ... }
    ...
};
```

`compute()` 使用 `ggml_backend` API 自动选择最佳后端（Metal/CUDA/CPU）。

### 2.5 `backend.zig` — Backend 与 Gallocr

```zig
pub const Backend = c.struct_ggml_backend;
pub const BackendBufferType = c.struct_ggml_backend_buffer_type;
pub const Gallocr = opaque { ... };
pub const Scheduler = c.struct_ggml_backend_sched;

pub fn backendCpuInit() !*Backend { ... }
pub fn backendCpuBufferType() *BackendBufferType { ... }
pub fn backendCpuSetNThreads(backend: *Backend, n_threads: i32) void { ... }
pub fn backendAllocCtxTensors(ctx: *Context, backend: *Backend) !void { ... }
pub fn backendAllocCtxTensorsFromBuft(ctx: *Context, buft: *BackendBufferType) !void { ... }
pub fn backendGraphCompute(backend: *Backend, graph: *CGraph) bool { ... }
pub fn backendTensorGet(tensor: *Tensor, data: []u8, offset: usize) void { ... }
pub fn backendTensorSet(tensor: *Tensor, data: []const u8, offset: usize) void { ... }
pub fn backendFree(backend: *Backend) void { ... }
pub fn backendName(backend: *Backend) []const u8 { ... }
pub fn backendIsGpu(backend: *Backend) bool { ... }
pub fn detectBestBackend() !*Backend { ... }
pub fn loadBackends() void { ... }
pub fn logAvailableBackends() void { ... }
pub fn setInput(tensor: *Tensor) void { ... }
```

### 2.6 `ops.zig` — 计算图操作（函数风格）

当前已绑定的操作函数（79 个）：

**基础算术**：`mulMat`, `mul`, `add`, `neg`, `exp`, `cpy`, `cast`, `scale`, `sub`（通过 Tensor 方法）

**归一化**：`rmsNorm`, `l2Norm`, `softMax`, `softMaxExt`, `softMaxAddSinks`

**激活函数**：`silu`, `sigmoid`, `softplus`, `gelu`, `geluErf`, `geluQuick`, `sqr`, `tanh`, `relu`, `clamp`

**GLU 变体**：`swigluSplit`, `gegluSplit`, `gegluErfSplit`, `gegluQuickSplit`

**位置编码**：`ropeExt`, `ropeMulti`, `getRelPos`, `addRelPos`, `addRelPosInplace`

**注意力**：`flashAttnExt`, `flashAttnExtSetPrec`, `flashAttnExtAddSinks`, `diagMaskInf`

**SSM**：`ssmConv`, `ssmScan`, `gatedDeltaNet`

**卷积**：`conv1d`, `conv1dPh`, `conv1dDw`, `conv1dDwPh`, `convTranspose1d`, `conv2d`, `conv2dSkP0`, `conv2dS1Ph`, `conv2dDw`, `convTranspose2dP0`

**形状操作**：`permute`, `cont`, `cont2d`, `cont3d`, `cont4d`, `reshape2d`, `reshape3d`, `reshape4d`, `repeat`, `repeat4d`, `transpose`, `concat`, `getRows`, `dupTensor`

**数据生成**：`arange`, `fill`, `interpolate`, `padReflect1d`, `roll`, `timestepEmbedding`

**池化**：`pool1d`

**自定义算子**：`mapCustom1`, `mapCustom2`, `mapCustom3`, `custom4d`

**量化**：`dequantizeRow`, `dequantizeTensor`

**其他**：`sumRows`, `setOutput`

### 2.7 `quantize.zig` — 量化函数

```zig
pub fn quantizeInit(typ: Type, nrows: i64, n_per_row: i64) !QuantizeCtx { ... }
pub fn quantizeFree(ctx: *QuantizeCtx) void { ... }
pub fn quantizeRequiresImatrix(typ: Type) bool { ... }
pub fn quantizeChunk(typ: Type, src: []const f32, dst: []u8, start_index: i64, nrows: i64, n_per_row: i64, imatrix: ?[]const f32) usize { ... }
pub fn quantizeTensor(typ: Type, src: []const f32, dst: []u8, nrows: i64, n_per_row: i64) usize { ... }
pub fn quantizedSize(typ: Type, nrows: i64, n_per_row: i64) usize { ... }
```

### 2.8 `utils.zig` — 工具函数

```zig
pub fn version() [:0]const u8 { ... }
pub fn cpuNThreads() i32 { ... }
pub const CpuFeatures = struct { ... };
pub fn recommendedThreads() i32 { ... }
pub fn logSet(level: LogLevel) void { ... }
pub fn logSetCallback(callback: c.ggml_log_callback, user_data: ?*anyopaque) void { ... }
```

## 3. 使用方式

所有模块通过 `src/ggml/mod.zig` 统一导出，业务代码只需：

```zig
const ggml = @import("ggml");

var ctx = try ggml.Context.init(1024 * 1024 * 100);
defer ctx.deinit();

const a = try ctx.newTensor1d(.f32, 10);
const b = try ctx.newTensor1d(.f32, 10);
const c = ggml.add(ctx, a, b);

var graph = try ggml.CGraph.init(ctx);
graph.buildForwardExpand(c);
try graph.compute(4);
```

如需直接访问 C API，使用 `ggml.c.ggml_version()` 等。

## 4. 构建集成

ggml 源码通过 `build.zig` 中的 `addCSourceFiles` 静态编译，避免 ABI 不一致。
所有 C 依赖和宏定义在 `build.zig` 中声明，源码无绝对路径。

详见 `docs/GGML_BUILD.md`。

## 5. 关键操作语义（在 Zig 中的正确使用）

### 5.1 `ggml_permute` 语义

`ggml_permute(ctx, tensor, axis0, axis1, axis2, axis3)` 对 4D 张量的维度进行重排。
其内部实现为：

```c
ne[axis0] = a->ne[0];  // axis0 位置获得旧的 ne[0]
ne[axis1] = a->ne[1];  // axis1 位置获得旧的 ne[1]
ne[axis2] = a->ne[2];  // axis2 位置获得旧的 ne[2]
ne[axis3] = a->ne[3];  // axis3 位置获得旧的 ne[3]

result->ne[0] = ne[0];  // 结果 ne[0] = ne[0]
result->ne[1] = ne[1];  // 结果 ne[1] = ne[1]
result->ne[2] = ne[2];  // 结果 ne[2] = ne[2]
result->ne[3] = ne[3];  // 结果 ne[3] = ne[3]
```

**示例**：`[D, H, C, B] -> permute(0, 3, 1, 2)`

```
ne[0] = old ne[0] = D  (axis0=0)
ne[3] = old ne[1] = H  (axis1=3)
ne[1] = old ne[2] = C  (axis2=1)
ne[2] = old ne[3] = B  (axis3=2)
结果: [D, C, B, H]
```

**在 Zig 中的使用**：

```zig
// llama.cpp 风格: [D, H, C, B] -> [D, C, B, H]
const p_llama = ggml.permute(ctx, t, 0, 3, 1, 2);

// 当前 Zig 风格: [D, H, C, B] -> [D, C, H, B]
const p_zig = ggml.permute(ctx, t, 0, 2, 1, 3);
```

**关键区别**：两种风格的区别在于 H（head）和 B（block）维度是否交换位置。
llama.cpp 将 H 放在最外层（ne[3]），Zig 将 H 放在 ne[2] 位置。

### 5.2 `ggml_mul_mat` 语义

`ggml_mul_mat(a, b)` 计算 `a^T @ b`（在 ne[0] 维度上收缩），
其 4D 结果的维度顺序为：

```c
result->ne[0] = a->ne[1];  // 第一个参数的第二维
result->ne[1] = b->ne[1];  // 第二个参数的第二维
result->ne[2] = b->ne[2];  // 沿用第二个参数的第三维
result->ne[3] = b->ne[3];  // 沿用第二个参数的第四维
```

**约束**：`a->ne[0] == b->ne[0]`（收缩维度必须匹配）。

**示例**：

```zig
// A = [2, 3, 4, 5], B = [2, 6, 4, 5]
// mul_mat(A, B): 在 ne[0]=2 上收缩
// 结果: ne[0]=A.ne[1]=3, ne[1]=B.ne[1]=6, ne[2]=B.ne[2]=4, ne[3]=B.ne[3]=5
// 形状: [3, 6, 4, 5]
const C = A.mulMat(ctx, B);
```

**在注意力计算中的应用**：

```zig
// llama.cpp 风格:
// Kp = [D, S, B, H], Qp = [D, C, B, H]
// scores = Kp.mulMat(ctx, Qp)
//   ne[0] = Kp.ne[1] = S
//   ne[1] = Qp.ne[1] = C
//   ne[2] = Qp.ne[2] = B
//   ne[3] = Qp.ne[3] = H
// 结果: [S, C, B, H]

// 当前 Zig 风格:
// Kp = [D, S, H, B], Qp = [D, C, H, B]
// scores = Kp.mulMat(ctx, Qp)
//   ne[0] = Kp.ne[1] = S
//   ne[1] = Qp.ne[1] = C
//   ne[2] = Qp.ne[2] = H
//   ne[3] = Qp.ne[3] = B
// 结果: [S, C, H, B]
```

### 5.3 测试验证

`src/tests/test_permute.zig` 包含以下测试用例，验证上述语义：

| 测试名 | 验证内容 |
|--------|----------|
| `permute semantics: basic 4D` | 验证 permute 基本语义（两种风格） |
| `permute: mul_mat 4D dimension ordering` | 验证 mul_mat 4D 维度顺序 |
| `permute: encoder.zig shape verification` | 验证 encoder.zig 中的 permute 形状 |
| `permute: RPE path` | 验证 RPE 路径计算正常 |

运行测试：

```bash
zig build test -Doptimize=ReleaseSafe
```

### 5.4 常见陷阱

1. **permute 不改变数据**：`ggml_permute` 只改变张量的维度元数据（ne/nb），不移动数据。
   如需实际重排数据，需配合 `ggml_cont`（`ggml.cont(ctx, tensor)`）。

2. **mul_mat 的维度顺序与直觉不同**：结果 ne[0] 来自第一个参数的第二维（a->ne[1]），
   而非第二个参数的第一维。这在 4D 注意力计算中容易出错。

3. **不同 permute 顺序导致不同计算结果**：虽然数学上等价，但 ggml 的 mul_mat
   对高维张量的 batch 广播处理依赖于数据布局（ne 维度顺序），
   不同的 permute 顺序导致不同的数据布局，进而产生不同的计算结果。

### 5.5 ggml_conv_2d 核张量布局（GGUF 维度顺序说明）

`ggml_conv_2d` 期望的 4D 核张量布局为 `[OC, IC, KH, KW]`，其 ne 维度顺序为：
- `ne[0] = KW`（宽度维度，最内层循环）
- `ne[1] = KH`（高度维度）
- `ne[2] = IC`（输入通道）
- `ne[3] = OC`（输出通道）

**关键事实**：`ggml_conv_2d` 的 ne[0] 是 **KW**（宽度先行），而非 KH。

#### GGUF 文件中的核张量布局

gguf-py 在写入 GGUF 文件时，会将 numpy 的 shape（从最外层到最内层，即 `[OC, IC, KH, KW]`）
**反转**后写入文件（见 `gguf_writer.py` 第 268 行：`ti.shape[n_dims - 1 - j]`）。

因此 GGUF 文件中存储的维度顺序为 `[KW, KH, IC, OC]`（从最内层到最外层），
与 ggml 的 `ne[]` 顺序完全一致：
- `dims[0] = KW`（最内层维度）
- `dims[1] = KH`
- `dims[2] = IC`
- `dims[3] = OC`（最外层维度）

#### 权重加载的正确性

zllama 的 `findOrCreateTensor` 将 GGUF 的 `dims[0..3]` 直接传递给 `ggml_new_tensor_4d` 的 `ne0..ne3` 参数：

```zig
// findOrCreateTensor 中:
try ctx.newTensor4d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2]), @intCast(dims[3]))
// 结果: ne[0]=dims[0]=KW, ne[1]=dims[1]=KH, ne[2]=dims[2]=IC, ne[3]=dims[3]=OC
```

这与 `ggml_conv_2d` 的期望完全一致，**无需 permute**。

#### 验证方法

对比 `ggml_conv_2d` 的官方测试 `deps/ggml/tests/test-conv2d.cpp` 第 99 行：

```cpp
model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, KW, KH, IC, OC);
// ne[0]=KW, ne[1]=KH — 确认 ggml_conv_2d 的 ne[0] 是 KW！
```

#### 深度可分离卷积

对于深度可分离卷积（如 Conformer 的 `conv_dw`），核张量是 1D `[KH, 1, IC, 1]`，不存在 KH/KW 混淆问题，无需 permute。

### 5.6 Flatten 操作：permute + reshape2d 模式

音频编码器中的 Flatten 是将 4D Conv2D 输出展平为 2D 以送入全连接层的标准模式。

**llama.cpp 参考**（`deps/llama.cpp/tools/mtmd/models/gemma4a.cpp:55-56`）：

```cpp
// Flatten [freq, time, ch, 1] -> [ch*freq, time]
cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
```

**Zig 实现**（`src/mtmd/audio/encoder.zig`）：

```zig
// Flatten [freq, time, ch, 1] -> [ch*freq, time]
// Matches llama.cpp: ggml_permute(ctx0, cur, 1, 2, 0, 3)
cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
const flat_dim0 = cur.ne()[0] * cur.ne()[1];  // = ch * freq
cur = cur.reshape2d(ctx, flat_dim0, cur.ne()[2]);  // = [ch*freq, time]
```

**维度流**：`[freq, time, ch, 1]` → permute(1,2,0,3) → `[time, ch, freq, 1]` → reshape2d → `[ch*freq, time]`

> **警告**：不要自行"推导"permute 参数。这个 permute 顺序由权重文件的布局决定（`input_projection.weight` 的形状必须与 flatten 输出匹配），直接复制 llama.cpp 的参数即可。

### 5.7 开发黄金法则：始终匹配 llama.cpp

当编写涉及 ggml 维度操作的代码时，遵循以下流程：

1. **找到 llama.cpp 参考**：在 `deps/llama.cpp/tools/mtmd/models/` 或 `deps/llama.cpp/src/` 中找到对应代码。
2. **逐字复制 permute 参数**：ggml_permute 的参数语义特殊（"新轴→原轴"而非"原轴→新轴"），自行推导极易出错。
3. **添加交叉引用注释**：在 Zig 代码中以 `// Matches llama.cpp:` 标注参考源文件和行号。
4. **编写测试验证**：在 `src/tests/test_permute.zig` 中添加对应测试，用 ggml 计算图运行验证。
5. **运行回归测试**：`zig build test -Doptimize=ReleaseSafe --summary all` 确保所有 155+ 测试通过。

常见需要匹配的操作：

| 操作 | 参考文件 | 关键点 |
|------|---------|--------|
| 标准注意力 Q/K/V permute | `llama-graph.cpp:2083-2085` | `permute(0,2,1,3)` |
| 音频 Flatten | `gemma4a.cpp:55` | `permute(1,2,0,3)` |
| 音频 LayerNorm | `gemma4a.cpp:42-44` | `permute(1,2,0,3)` / `permute(2,0,1,3)` |
| 音频 Q blocking | `gemma4a.cpp` | `permute(0,3,1,2)` |
| 音频 K blocking | `gemma4a.cpp` | `permute(0,3,1,2)` |
| 音频 V blocking | `gemma4a.cpp` | `permute(1,3,0,2)` |
| 注意力结果恢复 | `gemma4a.cpp` | `permute(0,2,3,1)` |

### 5.8 调试技巧

当怀疑 permute 顺序有问题时：

```zig
// 1. 打印 permute 前后的形状
const before = cur.ne();
std.log.debug("before permute: [{d},{d},{d},{d}]", .{ before[0], before[1], before[2], before[3] });
cur = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
const after = cur.ne();
std.log.debug("after permute: [{d},{d},{d},{d}]", .{ after[0], after[1], after[2], after[3] });

// 2. 与 llama.cpp 参考对比形状（在 llama.cpp 中添加 printf 打印相同位置）

// 3. 使用 test_permute.zig 编写独立测试
```

## 6. 常见类型不匹配问题（2025-07 修复记录）

### 6.1 `ggml_print_objects` 参数类型

`ggml_print_objects` 接受 `ggml_context*` 而非 `ggml_tensor*`。
在 Tensor 封装中，`print()` 方法应接受 `ctx: *anyopaque` 参数：

```zig
pub fn print(self: *Tensor, ctx: *anyopaque) void {
    _ = self;
    c.ggml_print_objects(@ptrCast(ctx));
}
```

### 6.2 `ggml_pool_2d` 的 `op` 参数类型

`ggml_pool_2d` 的 `op` 参数在 C 头文件中声明为 `enum_ggml_op_pool`，
在 Zig 的 `@cImport` 中映射为 `c_uint`（unsigned int），而非 `c_int`。

```zig
// 正确：op 应为 c_uint
pub fn pool2d(self: *Tensor, ctx: *anyopaque, op: c_uint, k0: i32, k1: i32, s0: i32, s1: i32, p0: f32, p1: f32) *Tensor {
    return wrap(c.ggml_pool_2d(@ptrCast(ctx), @ptrCast(@alignCast(self)), op, k0, k1, s0, s1, p0, p1));
}
```

### 6.3 `ggml_nelements` 返回类型

`ggml_nelements` 返回 `int64_t`，在 Zig 中对应 `i64`，而非 `usize` 或 `i32`。

```zig
pub fn nElems(self: *Tensor) i64 {
    return c.ggml_nelements(@ptrCast(@alignCast(self)));
}
```

### 6.4 `ggml_log_callback` 签名

ggml 日志回调的 `level` 参数类型为 `c_uint`（`enum ggml_log_level`），
而非 `c_int`。回调函数签名必须严格匹配：

```zig
fn defaultLogCallback(level: c_uint, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    const log_level: LogLevel = @enumFromInt(level);
    const msg = std.mem.sliceTo(text, 0);
    std.debug.print("[ggml] [{s}] {s}", .{ log_level.name(), msg });
}
```

### 6.5 `gguf_set_arr_str` 参数类型

`gguf_set_arr_str` 的 `data` 参数类型为 `const char **`，
在 Zig 中对应 `[]const [*:0]const u8`（切片，元素为 C 字符串指针），
而非 `[]const []const u8`。

```zig
pub fn setArrStr(ctx: *GgufCtx, key: [:0]const u8, data: []const [*:0]const u8) void {
    c.gguf_set_arr_str(ctx, key.ptr, @intCast(data.len), data.ptr);
}
```

### 6.6 避免重复定义

在 `opaque {}` 类型中，不允许有同名方法。编辑时需注意：
- 删除旧定义后再添加新定义
- 使用 `codedb_outline` 检查是否有重复符号
- 编译错误 `duplicate opaque member name` 表明存在重复定义

### 6.7 编译检查清单

修改 ggml 绑定后，运行以下命令验证：

```bash
# 1. 运行测试（检查类型安全）
zig build test -Doptimize=ReleaseSafe --summary all

# 2. 构建所有目标（检查链接）
zig build -Doptimize=ReleaseSafe

# 3. 格式化代码
zig fmt src/ggml/
```

### 6.8 `quantizedSize` 乘法溢出修复

`quantizedSize` 函数中，`row_size * nrows` 的乘法在 `i64` 类型下可能溢出。
修复方式：先分别转换为 `usize` 再相乘。

```zig
// 修复前（可能溢出）
return @as(usize, @intCast(row_size * nrows));

// 修复后
return @as(usize, @intCast(row_size)) * @as(usize, @intCast(nrows));
```

### 6.9 `conv2d_dw` 的 F32 类型约束

`ggml_conv_2d_dw` 内部使用 `im2col` + `mul_mat` 实现。`im2col` 的输出类型为 F16，
而 `mul_mat` 在 CPU 后端要求 `src1` 为 F32（当 `src0` 为 F32 时 `vec_dot_type` 为 F32，
`src1->type != vec_dot_type` 进入转换分支，断言 `src1->type == GGML_TYPE_F32`）。

因此 `conv2d_dw` 的输入张量必须为 F32 类型，不能使用 F16 输入。

### 6.10 `ssmConv` 和 `ssmScan` 的重复定义修复

`ops.zig` 中 `ssmConv` 和 `ssmScan` 存在重复定义，导致编译错误 `duplicate struct member name`。
修复方式：删除重复的函数定义，保留完整的实现。

---

## 7. 绑定覆盖状态

### 7.1 当前覆盖统计

| 类别 | C API 总数 | Zig 绑定数 | 覆盖率 |
|------|-----------|-----------|--------|
| 计算图操作（ops） | ~237 | ~235 | ~99% |
| Context 操作 | ~15 | ~15 | ~100% |
| Tensor 元数据 | ~30 | ~30 | ~100% |
| Backend 操作 | ~25 | ~20 | ~80% |
| 量化函数 | ~10 | ~6 | ~60% |
| **总计** | **~317** | **~306** | **~96.5%** |
| **总计** | **~317** | **~215** | **~68%** |
| **总计** | **~317** | **~145** | **~46%** |

### 7.2 未绑定的重要 C API 列表

以下 C API 尚未绑定（仅剩 2 个 deprecated API，已被替代）：

| C API | 说明 | 替代方案 |
|-------|------|---------|
| `ggml_rope_custom` | 自定义 RoPE（已废弃） | 使用 `ggml_rope_ext`（已绑定为 `ropeExt`） |
| `ggml_rope_custom_inplace` | 原地自定义 RoPE（已废弃） | 使用 `ggml_rope_ext_inplace`（已绑定为 `ropeExtInplace`） |

**绑定覆盖率已达 ~99%，仅剩 2 个 deprecated API 未绑定。**

### 7.3 测试覆盖

| 测试文件                                     | 覆盖内容                                                 | 测试数   |
| ---------                                    | ---------                                                | -------- |
| `src/tests/test_ggml_arange.zig`             | `arange` 操作                                            | 2        |
| `src/tests/test_ggml_cont.zig`               | `cont` 操作                                              | 2        |
| `src/tests/test_ggml_dup.zig`                | `dup` 操作（跨类型、view）                               | 3        |
| `src/tests/test_ggml_customop.zig`           | 自定义算子                                               | 2        |
| `src/tests/test_ggml_interpolate.zig`        | 插值操作                                                 | 2        |
| `src/tests/test_ggml_pad_reflect_1d.zig`     | 反射填充                                                 | 2        |
| `src/tests/test_ggml_pool.zig`               | 池化操作                                                 | 2        |
| `src/tests/test_ggml_rel_pos.zig`            | 相对位置编码                                             | 2        |
| `src/tests/test_ggml_roll.zig`               | 滚动操作                                                 | 2        |
| `src/tests/test_ggml_timestep_embedding.zig` | 时间步嵌入                                               | 2        |
| `src/tests/test_ggml_gguf.zig`               | GGUF 绑定                                                | 2        |
| `src/tests/test_ggml_conv.zig`               | 卷积操作（conv1d, conv2d, conv1d_dw, conv_transpose 等） | 9        |
| `src/tests/test_ggml_quantize_fns.zig`       | 量化函数（往返精度、类型特征表、反量化）                 | 21       |
| **总计**                                     |                                                          | ~53      |

运行测试：

```bash
# 运行所有 ggml 绑定测试
zig build test-ggml

# 运行所有测试
zig build test -Doptimize=ReleaseSafe --summary all
```
