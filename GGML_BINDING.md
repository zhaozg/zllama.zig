# ggml.zig 绑定设计

## 1. 设计原则

- **全能力暴露**：`ggml.c` 命名空间提供所有原始 C API。
- **安全封装**：opaque 类型 + 错误联合 + defer 资源管理。
- **零开销**：封装函数经 `@ptrCast` 编译后与直接调用 C 函数一致。
- **构建集成**：所有 C 依赖和宏定义在 `build.zig` 中声明，源码无绝对路径。

## 2. 文件结构

```zig
// ggml.zig
pub const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-backend.h");
    @cInclude("gguf.h");
});

pub const Type = enum(c.ggml_type) {
    f32 = c.GGML_TYPE_F32,
    f16 = c.GGML_TYPE_F16,
    q4_0 = c.GGML_TYPE_Q4_0,
    q4_K = c.GGML_TYPE_Q4_K,
    // ...
};

pub const Context = opaque {
    pub fn init(mem_size: usize) !*Context { ... }
    pub fn deinit(self: *Context) void { ... }
    pub fn newTensor1d(self: *Context, t: Type, ne0: u32) !*Tensor { ... }
    // ...
};

pub const Tensor = opaque {
    pub fn getName(self: *Tensor) [:0]const u8 { ... }
    pub fn getData(self: *Tensor) []u8 { ... }
    pub fn getShape(self: *Tensor) [4]i64 { ... }
};

pub const CGraph = opaque {
    pub fn init(ctx: *Context) !*CGraph { ... }
    pub fn buildForwardExpand(self: *CGraph, result: *Tensor) void { ... }
    pub fn compute(self: *CGraph, n_threads: u32) !void { ... }
};

pub const Backend = opaque {
    pub fn initCPU() !*Backend { ... }
    pub fn initMetal() !*Backend { ... }
    pub fn initCUDA() !*Backend { ... }
    // ...
};
```

## 3. 关键算子封装

```zig
pub fn mulMat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor { ... }
pub fn rmsNorm(ctx: *Context, a: *Tensor, eps: f32) *Tensor { ... }
pub fn conv1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: i32, s1: i32, p0: i32) *Tensor { ... }
pub fn ropeExt(ctx: *Context, a: *Tensor, pos: *Tensor, ...) *Tensor { ... }
pub fn ssmConv(ctx: *Context, x: *Tensor, w: *Tensor) *Tensor { ... }
```

## 4. 构建集成

ggml 源码通过 `build.zig` 中的 `addCSourceFiles` 静态编译，避免 ABI 不一致。

## 5. 使用示例

```zig
const ggml = @import("ggml");

var ctx = try ggml.Context.init(1024 * 1024 * 100);
defer ctx.deinit();

const a = try ctx.newTensor1d(.f32, 10);
const b = try ctx.newTensor1d(.f32, 10);
const c = ctx.add(a, b);

var graph = try ggml.CGraph.init(ctx);
defer graph.deinit();
graph.buildForwardExpand(c);
try graph.compute(4);
```
