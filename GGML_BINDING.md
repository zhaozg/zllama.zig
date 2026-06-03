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
