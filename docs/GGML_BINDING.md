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
    pub fn newTensor1d(self: *Context, t: Type, ne0: i64) !*Tensor { ... }
    pub fn newTensor2d(self: *Context, t: Type, ne0: i64, ne1: i64) !*Tensor { ... }
    pub fn newTensor3d(self: *Context, t: Type, ne0: i64, ne1: i64, ne2: i64) !*Tensor { ... }
    pub fn view1d(self: *Context, a: *Tensor, ne0: i64, offset: usize) *Tensor { ... }
    pub fn view2d(self: *Context, a: *Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) *Tensor { ... }
    pub fn view3d(self: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) *Tensor { ... }
    ...
};
```

### 2.3 `tensor.zig` — Tensor 封装

```zig
pub const Tensor = opaque {
    pub fn getName(self: *Tensor) [:0]const u8 { ... }
    pub fn setName(self: *Tensor, name: [:0]const u8) void { ... }
    pub fn ne(self: *Tensor) [4]i64 { ... }
    pub fn nb(self: *Tensor) [4]usize { ... }
    pub fn strides(self: *Tensor) [4]usize { ... }
    pub fn shape(self: *Tensor) [4]i64 { ... }
    pub fn dataBytes(self: *Tensor) []u8 { ... }
    pub fn dataF32(self: *Tensor) []f32 { ... }
    pub fn dataI32(self: *Tensor) []i32 { ... }
    pub fn setDataPtr(self: *Tensor, data: []u8) void { ... }
    pub fn setZero(self: *Tensor) void { ... }
    ...
};
```

### 2.4 `graph.zig` — CGraph 封装

```zig
pub const CGraph = opaque {
    pub fn init(ctx: *Context) !*CGraph { ... }
    pub fn buildForwardExpand(self: *CGraph, tensor: *Tensor) void { ... }
    pub fn compute(self: *CGraph, n_threads: i32) !void { ... }
    ...
};
```

`compute()` 使用 `ggml_backend` API 自动选择最佳后端（Metal/CUDA/CPU）。

### 2.5 `backend.zig` — Backend 与 Gallocr

```zig
pub const Backend = c.struct_ggml_backend;
pub const BackendBufferType = c.struct_ggml_backend_buffer_type;

pub fn backendCpuInit() !*Backend { ... }
pub fn backendCpuBufferType() *BackendBufferType { ... }
pub fn backendAllocCtxTensors(ctx: *Context, backend: *Backend) !void { ... }

pub const Gallocr = opaque {
    pub fn init(buft: *BackendBufferType) !*Gallocr { ... }
    pub fn allocGraph(self: *Gallocr, graph: *CGraph) bool { ... }
    pub fn free(self: *Gallocr) void { ... }
};
```

### 2.6 `ops.zig` — 计算图操作

```zig
pub fn mulMat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor { ... }
pub fn rmsNorm(ctx: *Context, a: *Tensor, eps: f32) *Tensor { ... }
pub fn ropeExt(ctx: *Context, a: *Tensor, pos: *Tensor, mode: i32, n_dims: i32, ...) *Tensor { ... }
pub fn conv1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: i32, p0: i32, d0: i32) *Tensor { ... }
pub fn ssmConv(ctx: *Context, sx: *Tensor, kernel: *Tensor) *Tensor { ... }
pub fn ssmScan(ctx: *Context, sx: *Tensor, B: *Tensor, C: *Tensor, dt: *Tensor, A: *Tensor, state: *Tensor) *Tensor { ... }
pub fn setOutput(tensor: *Tensor) void { ... }
...
```

### 2.7 `utils.zig` — 工具函数

```zig
pub fn version() [:0]const u8 { ... }
pub fn cpuNThreads() i32 { ... }
pub const CpuFeatures = struct { ... };
pub fn recommendedThreads() i32 { ... }
```

## 3. 使用方式

所有模块通过 `src/ggml.zig` 统一导出，业务代码只需：

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

### 5.5 ggml_conv_2d 核张量布局（GGUF vs ggml 核转置陷阱）

`ggml_conv_2d` 期望的 4D 核张量布局为 `[OC, IC, KH, KW]`，其 ne 维度顺序为：
- `ne[0] = KW`（宽度维度，最内层循环）
- `ne[1] = KH`（高度维度）
- `ne[2] = IC`（输入通道）
- `ne[3] = OC`（输出通道）

**关键事实**：`ggml_conv_2d` 的 ne[0] 是 **KW**（宽度先行），而非 KH。

#### GGUF 文件中的核张量布局

GGUF 中 Gemma Conformer 的卷积权重（如 `a.conv1d.0.weight`）存储的维度语义可能为：
- `ne[0] = KH`（高度维度，最内层循环）
- `ne[1] = KW`（宽度维度）
- `ne[2] = IC`（输入通道）
- `ne[3] = OC`（输出通道）

#### 核转置问题

即使 `KH == KW`（如 3×3 核），两种布局在 3×3 核矩阵内部的元素排列顺序不同：

| 布局 | 3×3 核内遍历顺序 | 核矩阵（以 data[0..8] 填充） |
|------|-----------------|--------------------------|
| GGUF (ne[0]=KH) | `(kh=0,kw=0) → (1,0) → (2,0) → (0,1) → ...` | `[[d0,d3,d6],[d1,d4,d7],[d2,d5,d8]]` |
| ggml (ne[0]=KW) | `(kw=0,kh=0) → (1,0) → (2,0) → (0,1) → ...` | `[[d0,d1,d2],[d3,d4,d5],[d6,d7,d8]]` |

两种布局产生的 3×3 核矩阵是**互为转置**的关系。直接使用 GGUF 的核张量调用 `ggml_conv_2d`，等于对每个输入/输出通道的 3×3 核做了**隐式转置**，导致完全不同的卷积结果。

#### 解决方案

在 `conv2d` 调用前，使用 `permute(1, 0, 2, 3)` 交换 KH 和 KW 维度，将 GGUF 布局转换为 ggml 期望的布局：

```zig
// GGUF 核: ne[0]=KH, ne[1]=KW → ggml conv2d 期望: ne[0]=KW, ne[1]=KH
const kernel = conv_w_raw.permute(ctx, 1, 0, 2, 3).cont(ctx);
cur = cur.conv2d(ctx, kernel, 2, 2, 1, 1, 1, 1);
```

> **验证方法**：对比 `ggml_conv_2d` 的官方测试 `deps/ggml/tests/test-conv2d.cpp` 第 99 行：
>
> ```cpp
> model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, KW, KH, IC, OC);
> // ne[0]=KW, ne[1]=KH — 确认 ggml_conv_2d 的 ne[0] 是 KW！
> ```

#### 检测方法

若卷积输出与参考实现严重偏离（NMSE > 1.0，余弦相似度 < 0.5），且：
- 输入数据已验证一致
- 权重数据已验证一致
- 卷积 stride/padding/dilation 参数已验证一致

则很可能是核转置问题。修复后卷积输出应与手工 NumPy 计算结果匹配。

#### 深度可分离卷积

对于深度可分离卷积（如 Conformer 的 `conv_dw`），核张量是 1D `[KH, 1, IC, 1]`，不存在 KH/KW 混淆问题，无需 permute。
