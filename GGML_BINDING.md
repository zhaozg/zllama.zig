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
    pub fn init(mem_size: usize) !*Context {
        const params = c.struct_ggml_init_params{
            .mem_size = mem_size,
            .mem_buffer = null,
            .no_alloc = false,
        };
        const ctx = c.ggml_init(params);
        if (ctx == null) return error.OutOfMemory;
        return @ptrCast(*Context, ctx);
    }
    pub fn deinit(self: *Context) void {
        c.ggml_free(@ptrCast(*c.struct_ggml_context, self));
    }
    pub fn newTensor1d(self: *Context, t: Type, ne0: u32) !*Tensor { ... }
    // ... 其他创建方法
};

pub const Tensor = opaque {
    pub fn getName(self: *Tensor) [:0]const u8 { ... }
    pub fn getData(self: *Tensor) []u8 { ... } // 返回切片
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
    pub fn deinit(self: *Backend) void { ... }
    pub fn graphCompute(self: *Backend, graph: *CGraph) !void { ... }
};

pub const GgufContext = opaque {
    pub fn initFromFile(path: [*:0]const u8) !*GgufContext { ... }
    pub fn deinit(self: *GgufContext) void { ... }
    pub fn findTensor(self: *GgufContext, name: [*:0]const u8) ?*Tensor { ... }
    pub fn getMetadata(self: *GgufContext, key: []const u8) !GgufValue { ... }
};
```

## 3. 关键算子封装示例

```zig
pub fn mulMat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @ptrCast(*Tensor, c.ggml_mul_mat(
        @ptrCast(*c.struct_ggml_context, ctx),
        @ptrCast(*c.struct_ggml_tensor, a),
        @ptrCast(*c.struct_ggml_tensor, b),
    ));
}

pub fn rmsNorm(ctx: *Context, a: *Tensor, eps: f32) *Tensor {
    return @ptrCast(*Tensor, c.ggml_rms_norm(
        @ptrCast(*c.struct_ggml_context, ctx),
        @ptrCast(*c.struct_ggml_tensor, a),
        eps,
    ));
}

pub fn conv1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: i32, s1: i32, p0: i32) *Tensor {
    return @ptrCast(*Tensor, c.ggml_conv_1d(
        @ptrCast(*c.struct_ggml_context, ctx),
        @ptrCast(*c.struct_ggml_tensor, a),
        @ptrCast(*c.struct_ggml_tensor, b),
        s0, s1, p0,
    ));
}
```

## 4. 构建集成（`build.zig` 片段）

```zig
const lib = b.addStaticLibrary(.{
    .name = "ggml",
    .target = target,
    .optimize = optimize,
});
lib.addIncludePath(.{ .path = "deps/ggml/include" });
lib.addCSourceFiles(.{
    .files = &.{
        "deps/ggml/src/ggml.c",
        "deps/ggml/src/ggml-backend.c",
        "deps/ggml/src/ggml-quants.c",
        "deps/ggml/src/gguf.c",
    },
    .flags = &.{"-std=c11", "-O3", "-march=native"},
});
lib.linkLibC();

if (target.result.os.tag == .macos) {
    lib.linkFramework("Metal");
    lib.linkFramework("Foundation");
    lib.defineCMacro("GGML_USE_METAL", "1");
}
if (target.result.os.tag == .linux and cuda_available) {
    lib.linkSystemLibrary("cuda");
    lib.defineCMacro("GGML_USE_CUDA", "1");
}

const exe = b.addExecutable(.{ .name = "qwen", .root_source_file = .{ .path = "src/main.zig" } });
exe.root_module.addImport("ggml", lib);
```

## 5. 使用示例

```zig
const ggml = @import("ggml");

var ctx = try ggml.Context.init(1024 * 1024 * 100);
defer ctx.deinit();

const a = try ctx.newTensor1d(.f32, 10);
const b = try ctx.newTensor1d(.f32, 10);
// 设置数据...
const c = ctx.add(a, b);

var graph = try ggml.CGraph.init(ctx);
defer graph.deinit();
graph.buildForwardExpand(c);
try graph.compute(4);
// 读取 c 结果...
```

