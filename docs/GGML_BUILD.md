# GGML 构建集成文档

> 本文档描述 zllama.zig 的 ggml 集成架构、两种构建模式及已知坑位。

---

## 📦 两种构建模式

### 模式一：系统库（默认）

```bash
zig build
```

链接系统预装的 ggml（Homebrew `/usr/local/Cellar/ggml/...`）。
- 优点：编译快，无需维护 C 源文件列表
- 缺点：依赖外部库，无法定制优化；可能 API/ABI 不一致

### 模式二：源码捆绑（`-Dbundle-ggml`）

```bash
zig build -Dbundle-ggml
```

从 `deps/ggml/` 源码直接编译静态库并链接。
- 优点：零外部依赖，单二进制分发；自主控制编译宏和优化标志
- 缺点：首次编译较慢（~30 个 C/C++ 源文件）；升级 ggml 需同步维护源文件列表

---

## 🏗️ build.zig 架构

```
build.zig
├── build()                         → 顶层：双模式分支
│   ├── buildGgmlFromSource()       → bundle 模式：编译 deps/ggml/ 源码
│   └── createModule("ggml")        → 统一模块输出
│       ├── bundle: addIncludePath + linkLibrary(lib)
│       └── system: linkSystemLibrary
└── 各 executable                   → 链接 ggml_mod
```

### 核心流程

```zig
// 1. 检测 -Dbundle-ggml 选项
const bundle_ggml = b.option(bool, "bundle-ggml", ...) orelse false;

// 2. 按需编译 ggml 静态库
const ggml_lib: ?*std.Build.Step.Compile = if (bundle_ggml)
    buildGgmlFromSource(b, target, optimize)
else
    null;

// 3. 创建统一模块
const ggml_mod = b.createModule(.{
    .root_source_file = b.path("src/ggml.zig"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,
});

// 4. 根据模式配置模块
if (bundle_ggml) {
    ggml_mod.addIncludePath(b.path("deps/ggml/include"));
    ggml_mod.addIncludePath(b.path("deps/ggml/src"));
    ggml_mod.addCMacro("GGML_USE_CPU", "1");
    ggml_mod.linkLibrary(ggml_lib.?);
} else {
    ggml_mod.addIncludePath(.{ .cwd_relative = "/usr/local/include" });
    ggml_mod.linkSystemLibrary("ggml-base", .{});
    ggml_mod.linkSystemLibrary("ggml", .{});
}
```

---

## 📁 ggml v0.13.1 源文件结构

```
deps/ggml/src/
├── ggml.c                    # 核心 C（张量、图、上下文）
├── ggml.cpp                  # C++ 辅助
├── ggml-alloc.c              # 内存分配器
├── ggml-quants.c             # 量化算子
├── ggml-backend.cpp          # 后端注册
├── ggml-backend-dl.cpp       # 动态加载后端
├── ggml-backend-reg.cpp      # 后端注册表
├── ggml-backend-meta.cpp     # 元后端
├── ggml-threading.cpp        # 多线程
├── ggml-opt.cpp              # 优化器
├── ggml-cpu/                 # CPU 后端
│   ├── ggml-cpu.c
│   ├── ggml-cpu.cpp
│   ├── quants.c
│   ├── repack.cpp
│   ├── hbm.cpp, traits.cpp
│   ├── binary-ops.cpp, unary-ops.cpp
│   ├── vec.cpp, ops.cpp
│   ├── amx/amx.cpp, amx/mmq.cpp        # AMX（x86_64 矩阵加速）
│   └── arch/x86/                        # x86_64 架构特定
│       ├── cpu-feats.cpp
│       ├── quants.c
│       └── repack.cpp
├── ggml-metal/               # Metal 后端（aarch64 macOS）
└── gguf.cpp                  # GGUF 解析（ggml 侧）
```

---

## 🔧 buildGgmlFromSource 实现要点

### 1. 模块创建（使用 `createModule` + `addLibrary`）

```zig
const lib_mod = b.createModule(.{
    .target = target,
    .optimize = optimize,
    .link_libc = true,
    .link_libcpp = true,   // ← 关键：混合 C/C++ 源文件
});
```

> **Zig 0.16.0 约束**：不再使用 `b.addStaticLibrary()`，改用 `b.addLibrary(.{ .linkage = .static })`。

### 2. 头文件路径（三级）

```zig
lib_mod.addIncludePath(b.path("deps/ggml/include"));
lib_mod.addIncludePath(b.path("deps/ggml/src"));
lib_mod.addIncludePath(b.path("deps/ggml/src/ggml-cpu"));
```

### 3. 必需宏定义

```zig
lib_mod.addCMacro("GGML_USE_CPU", "1");          // 启用 CPU 后端
lib_mod.addCMacro("GGML_BACKEND_DL", "1");       // 启用动态后端加载
lib_mod.addCMacro("GGML_VERSION", "\"0.13.1\""); // ⚠️ 必填！ggml.c:515 使用
lib_mod.addCMacro("GGML_COMMIT", "\"1e33fed3\""); // ⚠️ 必填！ggml.c:519 使用
```

> **坑位 1**：`GGML_VERSION` 和 `GGML_COMMIT` 宏缺失会导致编译失败。ggml.c 中 `ggml_version()` 和 `ggml_commit()` 函数直接使用这些宏，CMake 构建时由构建系统注入，Zig 构建需手动添加。

### 4. x86_64 优化标志（仅施加于 CPU 后端文件）

```zig
const x86_opt_flags: []const []const u8 = if (cpu_arch == .x86_64)
    &.{ "-mavx2", "-mfma", "-mf16c", "-mavx", "-msse4.2" }
else
    &.{};

// C 基础标志
const c_base_flags = &.{ "-std=c11", "-Wno-unused-function", ... };

// C++ 基础标志
const cpp_base_flags = &.{ "-std=c++17", "-Wno-unused-function", ... };

// CPU 后端文件 = 基础标志 + x86_64 优化标志
lib_mod.addCSourceFile(.{
    .file = b.path("deps/ggml/src/ggml-cpu/quants.c"),
    .flags = c_base_flags ++ x86_opt_flags,
});
```

> **设计决策**：x86_64 优化标志仅施加于 `ggml-cpu/` 下的文件，核心 `ggml.c` 不施加——避免交叉编译时破坏可移植性。

### 5. macOS 框架链接

```zig
switch (os) {
    .macos => {
        lib_mod.linkFramework("Foundation", .{});
        lib_mod.linkFramework("Accelerate", .{});
        lib_mod.addCMacro("GGML_USE_ACCELERATE", "1");
        lib_mod.addCMacro("ACCELERATE_NEW_LAPACK", "1");
        lib_mod.addCMacro("ACCELERATE_LAPACK_ILP64", "1");
    },
    // ...
}
```

### 6. LTO（仅 Linux）

```zig
if (os == .linux) {
    lib.lto = .full;
    lib.use_lld = true;
}
// macOS 上 LTO 需要 LLD，但 LLD 不支持 Mach-O
```

---

## 🐛 已知坑位与修复

### 坑位 1：GGML_VERSION / GGML_COMMIT 宏缺失

- **现象**：`zig build -Dbundle-ggml` 编译失败，`ggml.c:515: use of undeclared identifier 'GGML_VERSION'`
- **原因**：ggml.c 中 `ggml_version()` 和 `ggml_commit()` 使用 CMake 注入的宏，Zig 构建时不会自动设置
- **修复**：在 `buildGgmlFromSource` 中添加：
  ```zig
  lib_mod.addCMacro("GGML_VERSION", "\"0.13.1\"");
  lib_mod.addCMacro("GGML_COMMIT", "\"1e33fed3\"");
  ```

### 坑位 2：ggml_graph_nbytes 的空指针算术 UB

- **现象**：运行时 crash，`applying non-zero offset 96 to null pointer`
- **原因**：`deps/ggml/src/ggml.c` 中 `ggml_graph_nbytes` 使用 `void* p = 0` 做地址算术计算结构体偏移量，这在 C 中是 UB。Zig 的 C 编译器（clang with UBSan-like checks）会 trap 这种操作
- **修复**：将 `void*` 算术改为 `uintptr_t` 整数算术：
  ```c
  // 原代码（UB）
  void * p = 0;
  incr_ptr_aligned(&p, sizeof(struct ggml_cgraph), 1);
  // ...
  return (size_t) p;

  // 修复后
  uintptr_t nbytes = 0;
  nbytes = GGML_PAD(nbytes, 1);
  nbytes += sizeof(struct ggml_cgraph);
  // ...
  return (size_t) nbytes;
  ```
- **位置**：`deps/ggml/src/ggml.c` 行 7069-7085

### 坑位 3：Metal .m 文件与 Zig C 编译器

- **现象**：尝试添加 `ggml-metal.m` 时编译失败
- **原因**：Zig 的 C 编译器不支持 Objective-C（`.m`）文件；Metal 后端需要 clang 编译
- **现状**：Metal 后端在 bundle 模式下暂不可用；x86_64 上使用 Accelerate 框架已足够
- **未来方向**：aarch64 macOS 上可预编译 Metal 后端为 `.a` 再链接

### 坑位 4：`ggml_gallocr_needs_realloc` 警告

- **现象**：测试运行时输出 `ggml_gallocr_needs_realloc: graph has different number of nodes`
- **原因**：`ggml_gallocr_alloc_graph` 在检测到图结构变化时会自动重新分配缓冲区。这是 ggml 后端的正常行为，不是错误
- **影响**：不影响测试结果，但会导致测试输出包含警告信息
- **处理**：在测试中可忽略此警告。如果测试因此失败，检查图构建是否正确（节点数是否一致）

---

## 🧪 测试体系

### 测试步骤

`build.zig` 中定义了多个测试步骤：

```bash
# 运行所有测试
zig build test

# 运行 ggml 绑定专项测试（14 个子测试）
zig build test-ggml

# 运行特定子测试
zig build test-ggml-conv
zig build test-ggml-quantize-fns
zig build test-ggml-dup
zig build test-ggml-arange
zig build test-ggml-cont
zig build test-ggml-customop
zig build test-ggml-interpolate
zig build test-ggml-pad-reflect-1d
zig build test-ggml-pool
zig build test-ggml-rel-pos
zig build test-ggml-roll
zig build test-ggml-timestep-embedding
zig build test-ggml-gguf
```

### test-ggml 步骤结构

`test-ggml` 步骤聚合了 14 个子测试，覆盖以下 ggml 绑定功能：

| 子测试 | 覆盖内容 | 测试数 |
|--------|---------|--------|
| test-ggml-arange | `arange` 等差数列操作 | 2 |
| test-ggml-cont | `cont` 连续化操作 | 2 |
| test-ggml-dup | `dup` 复制操作（跨类型、view） | 3 |
| test-ggml-customop | 自定义算子（mapCustom1/2/3） | 2 |
| test-ggml-interpolate | 插值操作 | 2 |
| test-ggml-pad-reflect-1d | 反射填充 | 2 |
| test-ggml-pool | 池化操作 | 2 |
| test-ggml-rel-pos | 相对位置编码 | 2 |
| test-ggml-roll | 滚动操作 | 2 |
| test-ggml-timestep-embedding | 时间步嵌入 | 2 |
| test-ggml-gguf | GGUF 绑定 | 2 |
| test-ggml-conv | 卷积操作（conv1d, conv2d, conv1d_dw, conv_transpose 等） | 9 |
| test-ggml-quantize-fns | 量化函数（往返精度、类型特征表、反量化） | 21 |
| **总计** | | **~53** |

### 测试文件位置

所有 ggml 绑定测试位于 `src/tests/` 目录下：

```
src/tests/
├── test_ggml_arange.zig
├── test_ggml_cont.zig
├── test_ggml_dup.zig
├── test_ggml_customop.zig
├── test_ggml_interpolate.zig
├── test_ggml_pad_reflect_1d.zig
├── test_ggml_pool.zig
├── test_ggml_rel_pos.zig
├── test_ggml_roll.zig
├── test_ggml_timestep_embedding.zig
├── test_ggml_gguf.zig
├── test_ggml_conv.zig          # 新增：卷积算子测试
└── test_ggml_quantize_fns.zig  # 新增：量化函数测试
```

### 测试设计原则

1. **独立可运行**：每个测试文件通过 `build.zig` 中的独立步骤运行，不依赖 `test_runner.zig`
2. **使用 ggml 模块**：通过 `@import("ggml")` 导入，不直接调用 C API
3. **验证数值正确性**：对已知输入验证输出值，而非仅检查形状
4. **处理平台差异**：对可能因平台/后端不同而失败的操作使用 `error.SkipZigTest`
5. **内存安全**：使用 `defer` 确保 Context 释放

---

## ✅ 验证清单

```bash
# 1. 系统库模式构建
zig build

# 2. 源码捆绑模式构建
zig build -Dbundle-ggml

# 3. 双模式测试
zig build test
zig build -Dbundle-ggml test

# 4. ggml 绑定专项测试
zig build test-ggml

# 5. 三模型推理验证（两种模式各一遍）
for model in tinyllama Llama-3.2 Qwen3.5; do
    zig-out/bin/zllama -n 5 --model ~/.cache/models/${model}*.gguf 你好
done

# 6. Benchmark 模式
zig-out/bin/zllama --benchmark -n 20 --model ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf 你好
```

---

## 📋 维护清单（ggml 升级时）

当更新 `deps/ggml` 到新版本时，需要检查以下内容：

| 检查项 | 位置 |
|--------|------|
| `GGML_VERSION` / `GGML_COMMIT` 宏值 | `build.zig` `buildGgmlFromSource` |
| 源文件列表（新增/删除） | `build.zig` `buildGgmlFromSource` |
| `ggml_graph_nbytes` 的 `void*` 算术是否仍存在 | `deps/ggml/src/ggml.c` |
| 新增子目录（如 ggml-cpu/arch/xxx） | `build.zig` 的 include / 源文件路径 |
| CMakeLists.txt 中新增的宏定义 | `build.zig` `addCMacro` |
| 新增 ggml C API 是否需要在 Zig 中绑定 | `src/ggml/ops.zig` + `src/ggml/mod.zig` |
| 新增 ggml 操作是否需要测试覆盖 | `src/tests/test_ggml_*.zig` + `build.zig` |
