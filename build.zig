const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // -Dbundle-ggml: build ggml from source instead of using system-installed libraries
    const bundle_ggml = b.option(bool, "bundle-ggml", "Build ggml from source instead of using system libraries") orelse false;

    // Build ggml from source when bundling, otherwise nil
    const ggml_lib: ?*std.Build.Step.Compile = if (bundle_ggml)
        buildGgmlFromSource(b, target, optimize)
    else
        null;

    // ======================================================================
    // ggml 模块（C 绑定 + 安全封装）
    // ======================================================================
    const ggml_mod = b.createModule(.{
        .root_source_file = b.path("src/ggml.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    if (bundle_ggml) {
        // 从源码构建时：使用 deps/ggml 中的头文件，链接自建静态库
        ggml_mod.addIncludePath(b.path("deps/ggml/include"));
        ggml_mod.addIncludePath(b.path("deps/ggml/src"));
        ggml_mod.addCMacro("GGML_USE_CPU", "1");
        ggml_mod.linkLibrary(ggml_lib.?);
    } else {
        // 使用系统安装的 ggml
        ggml_mod.addIncludePath(.{ .cwd_relative = "/usr/local/include" });
        ggml_mod.linkSystemLibrary("ggml-base", .{});
        ggml_mod.linkSystemLibrary("ggml", .{});
    }

    // macOS 框架和加速
    if (target.result.os.tag == .macos) {
        ggml_mod.linkFramework("Foundation", .{});
        ggml_mod.linkFramework("Accelerate", .{});
        ggml_mod.addCMacro("GGML_USE_ACCELERATE", "1");
    }

    // ======================================================================
    // 内部模块
    // ======================================================================
    const gguf_mod = b.createModule(.{
        .root_source_file = b.path("src/gguf.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gguf_mod.addImport("ggml", ggml_mod);

    const memory_mod = b.createModule(.{
        .root_source_file = b.path("src/core/memory.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    memory_mod.addImport("ggml", ggml_mod);

    const kv_cache_mod = b.createModule(.{
        .root_source_file = b.path("src/kv_cache.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    kv_cache_mod.addImport("ggml", ggml_mod);
    kv_cache_mod.addImport("memory", memory_mod);

    // --- 层模块（算子库） ---
    const rms_norm_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/rms_norm.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    rms_norm_mod.addImport("ggml", ggml_mod);

    const rope_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/rope.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    rope_mod.addImport("ggml", ggml_mod);

    const swiglu_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/swiglu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    swiglu_mod.addImport("ggml", ggml_mod);

    const attention_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/attention.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    attention_mod.addImport("ggml", ggml_mod);

    const embed_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/embed.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    embed_mod.addImport("ggml", ggml_mod);

    const pooling_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/pooling.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    pooling_mod.addImport("ggml", ggml_mod);


    const weight_loader_mod = b.createModule(.{
        .root_source_file = b.path("src/core/weight_loader.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    weight_loader_mod.addImport("ggml", ggml_mod);
    weight_loader_mod.addImport("gguf", gguf_mod);


    const model_mod = b.createModule(.{
        .root_source_file = b.path("src/model.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    model_mod.addImport("ggml", ggml_mod);
    model_mod.addImport("gguf", gguf_mod);
    model_mod.addImport("memory", memory_mod);
    model_mod.addImport("kv_cache", kv_cache_mod);
    model_mod.addImport("rms_norm", rms_norm_mod);
    model_mod.addImport("rope", rope_mod);
    model_mod.addImport("swiglu", swiglu_mod);
    model_mod.addImport("attention", attention_mod);
    model_mod.addImport("embed", embed_mod);
    model_mod.addImport("pooling", pooling_mod);

    model_mod.addImport("weight_loader", weight_loader_mod);


    const graph_builder_mod = b.createModule(.{
        .root_source_file = b.path("src/core/graph_builder.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    graph_builder_mod.addImport("ggml", ggml_mod);
    graph_builder_mod.addImport("rope", rope_mod);

    graph_builder_mod.addImport("model", model_mod);
    graph_builder_mod.addImport("memory", memory_mod);

    const graph_context_mod = b.createModule(.{
        .root_source_file = b.path("src/core/graph_context.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    graph_context_mod.addImport("ggml", ggml_mod);
    graph_context_mod.addImport("model", model_mod);
    graph_context_mod.addImport("graph_builder", graph_builder_mod);
    graph_context_mod.addImport("memory", memory_mod);




    model_mod.addImport("graph_builder", graph_builder_mod);

    model_mod.addImport("graph_builder", graph_builder_mod);

    const engine_common_mod = b.createModule(.{
        .root_source_file = b.path("src/core/engine_common.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    engine_common_mod.addImport("ggml", ggml_mod);
    engine_common_mod.addImport("model", model_mod);

    const registry_mod = b.createModule(.{

        .root_source_file = b.path("src/models/registry.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    registry_mod.addImport("ggml", ggml_mod);
    registry_mod.addImport("gguf", gguf_mod);
    registry_mod.addImport("model", model_mod);
    registry_mod.addImport("graph_builder", graph_builder_mod);
    registry_mod.addImport("memory", memory_mod);

    const tokenizer_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tokenizer_mod.addImport("ggml", ggml_mod);
    tokenizer_mod.addImport("gguf", gguf_mod);
    tokenizer_mod.addImport("ggml", ggml_mod);
    tokenizer_mod.addImport("gguf", gguf_mod);

    const sampler_mod = b.createModule(.{
        .root_source_file = b.path("src/sampler.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    sampler_mod.addImport("ggml", ggml_mod);

    // ======================================================================

    // --- 多模态模块 ---
    const mm_audio_mod = b.createModule(.{
        .root_source_file = b.path("src/mm/audio.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_audio_mod.addImport("ggml", ggml_mod);
    mm_audio_mod.addImport("gguf", gguf_mod);
    mm_audio_mod.addImport("weight_loader", weight_loader_mod);


    const mm_vision_mod = b.createModule(.{
        .root_source_file = b.path("src/mm/vision.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_vision_mod.addImport("ggml", ggml_mod);
    mm_vision_mod.addImport("gguf", gguf_mod);
    mm_vision_mod.addImport("weight_loader", weight_loader_mod);

    const fft_mod = b.createModule(.{
        .root_source_file = b.path("src/mm/fft.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    fft_mod.linkFramework("Accelerate", .{});


    const utils_mod = b.createModule(.{
        .root_source_file = b.path("src/utils.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    utils_mod.addImport("ggml", ggml_mod);
    utils_mod.addImport("gguf", gguf_mod);
    utils_mod.addImport("tokenizer", tokenizer_mod);


    const mm_manager_mod = b.createModule(.{
        .root_source_file = b.path("src/mm/manager.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_manager_mod.addImport("ggml", ggml_mod);
    mm_manager_mod.addImport("gguf", gguf_mod);
    mm_manager_mod.addImport("model", model_mod);
    mm_manager_mod.addImport("audio", mm_audio_mod);
    mm_manager_mod.addImport("vision", mm_vision_mod);
    mm_manager_mod.addImport("vision", mm_vision_mod);

    const mm_preprocess_mod = b.createModule(.{
        .root_source_file = b.path("src/mm/preprocess.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_preprocess_mod.addImport("ggml", ggml_mod);
    mm_preprocess_mod.addImport("fft", fft_mod);

    // stb_image — vendor/stb/stb_image.h wrapper for JPEG/PNG decoding
    const stb_image_mod = b.createModule(.{
        .root_source_file = b.path("src/stb_image.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    stb_image_mod.addIncludePath(b.path("vendor/stb"));
    stb_image_mod.addCSourceFile(.{ .file = b.path("vendor/stb/stb_image.c") });

    mm_preprocess_mod.addImport("stb_image", stb_image_mod);

    // 主可执行文件 zllama
    // ======================================================================
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe_mod.addImport("ggml", ggml_mod);
    exe_mod.addImport("gguf", gguf_mod);
    exe_mod.addImport("model", model_mod);
    exe_mod.addImport("registry", registry_mod);
    exe_mod.addImport("graph_builder", graph_builder_mod);
    exe_mod.addImport("memory", memory_mod);
    exe_mod.addImport("tokenizer", tokenizer_mod);
    exe_mod.addImport("sampler", sampler_mod);
    exe_mod.addImport("graph_context", graph_context_mod);
    exe_mod.addImport("mm", mm_manager_mod);
    exe_mod.addImport("preprocess", mm_preprocess_mod);
    exe_mod.addImport("kv_cache", kv_cache_mod);
    exe_mod.addImport("stb_image", stb_image_mod);
    exe_mod.addImport("pooling", pooling_mod);
    exe_mod.addImport("engine_common", engine_common_mod);


    const exe = b.addExecutable(.{
        .name = "zllama",
        .root_module = exe_mod,
    });

    // ======================================================================
    // zllama-tokenize
    // ======================================================================
    const tokenize_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenize_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tokenize_mod.addImport("ggml", ggml_mod);
    tokenize_mod.addImport("gguf", gguf_mod);
    tokenize_mod.addImport("tokenizer", tokenizer_mod);
    tokenize_mod.addImport("utils", utils_mod);


    const tokenize_exe = b.addExecutable(.{
        .name = "zllama-tokenize",
        .root_module = tokenize_mod,
    });

    // ======================================================================
    // zllama-simple
    // ======================================================================
    const simple_mod = b.createModule(.{
        .root_source_file = b.path("src/simple_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    simple_mod.addImport("ggml", ggml_mod);
    simple_mod.addImport("gguf", gguf_mod);
    simple_mod.addImport("model", model_mod);
    simple_mod.addImport("registry", registry_mod);
    simple_mod.addImport("graph_builder", graph_builder_mod);
    simple_mod.addImport("memory", memory_mod);
    simple_mod.addImport("tokenizer", tokenizer_mod);
    simple_mod.addImport("sampler", sampler_mod);
    simple_mod.addImport("kv_cache", kv_cache_mod);
    simple_mod.addImport("stb_image", stb_image_mod);
    simple_mod.addImport("engine_common", engine_common_mod);

    simple_mod.addImport("graph_context", graph_context_mod);
    simple_mod.addImport("mm", mm_manager_mod);
    simple_mod.addImport("preprocess", mm_preprocess_mod);
    const simple_exe = b.addExecutable(.{
        .name = "zllama-simple",
        .root_module = simple_mod,
    });

    // ======================================================================
    // 测试
    // ======================================================================
    const test_step = b.step("test", "Run all tests");

    const test_root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_root_mod.addImport("ggml", ggml_mod);
    test_root_mod.addImport("gguf", gguf_mod);
    test_root_mod.addImport("model", model_mod);
    test_root_mod.addImport("registry", registry_mod);
    test_root_mod.addImport("graph_builder", graph_builder_mod);
    test_root_mod.addImport("memory", memory_mod);
    test_root_mod.addImport("tokenizer", tokenizer_mod);
    test_root_mod.addImport("sampler", sampler_mod);
    test_root_mod.addImport("kv_cache", kv_cache_mod);
    test_root_mod.addImport("graph_context", graph_context_mod);
    test_root_mod.addImport("mm", mm_manager_mod);

    const test_unit = b.addTest(.{
        .name = "unit-tests",
        .root_module = test_root_mod,
    });
    test_step.dependOn(&test_unit.step);

    _ = b.step("test-layers", "Run layer tests only");
    _ = b.step("test-gguf", "Run GGUF tests only");
    _ = b.step("test-archs", "Run architecture tests only");
    _ = b.step("test-kv-cache", "Run KV Cache tests only");
    _ = b.step("test-vocab", "Run vocab-based tokenizer tests only");

    // ======================================================================
    // 工具可执行文件
    // ======================================================================
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/dump_graph.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-dump-graph",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("dump-graph", "Run zllama-dump-graph tool");
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/compare_logits.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-compare-logits",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("compare-logits", "Run zllama-compare-logits tool");
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/generate_reference.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-gen-ref",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("gen-ref", "Run zllama-gen-ref tool");
    }

    // ======================================================================
    // 安装与运行
    // ======================================================================
    b.installArtifact(exe);
    b.installArtifact(tokenize_exe);
    b.installArtifact(simple_exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run zllama.zig engine");
    run_step.dependOn(&run_cmd.step);

    const tokenize_run_cmd = b.addRunArtifact(tokenize_exe);
    tokenize_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        tokenize_run_cmd.addArgs(args);
    }
    const tokenize_run_step = b.step("tokenize", "Run zllama-tokenize tool");
    tokenize_run_step.dependOn(&tokenize_run_cmd.step);

    const simple_run_cmd = b.addRunArtifact(simple_exe);
    simple_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        simple_run_cmd.addArgs(args);
    }
    const simple_run_step = b.step("simple", "Run zllama-simple tool");
    simple_run_step.dependOn(&simple_run_cmd.step);
}

// ============================================================================
// 从源码构建 ggml 静态库
// ============================================================================

/// 构建 ggml 静态库（CPU 后端）。
/// 返回一个 Step.Compile，可通过 Module.linkLibrary() 链接。
fn buildGgmlFromSource(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const src_dir = "deps/ggml/src";
    const cpu_dir = src_dir ++ "/ggml-cpu";
    const amx_dir = cpu_dir ++ "/amx";
    const cpu_arch = target.result.cpu.arch;
    const os = target.result.os.tag;

    // 创建 ggml 库模块（无 Zig 根文件，仅有 C/C++ link_objects）
    const lib_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    // 头文件路径
    lib_mod.addIncludePath(b.path("deps/ggml/include"));
    lib_mod.addIncludePath(b.path("deps/ggml/src"));
    lib_mod.addIncludePath(b.path("deps/ggml/src/ggml-cpu"));

    // 宏定义
    lib_mod.addCMacro("GGML_USE_CPU", "1");
    lib_mod.addCMacro("GGML_BACKEND_DL", "1");
    lib_mod.addCMacro("GGML_VERSION", "\"0.13.1\"");
    lib_mod.addCMacro("GGML_COMMIT", "\"1e33fed3\"");
    // x86_64 优化标志（所有 CPU 后端文件共享）
    const x86_opt_flags: []const []const u8 = if (cpu_arch == .x86_64)
        &.{ "-mavx2", "-mfma", "-mf16c", "-mavx", "-msse4.2" }
    else
        &.{};
    // C 文件基础标志
    const c_base_flags = &.{ "-std=c11", "-Wno-unused-function", "-Wno-unused-variable", "-Wno-missing-braces", "-Wno-implicit-function-declaration" };

    // C++ 文件基础标志
    const cpp_base_flags = &.{ "-std=c++17", "-Wno-unused-function", "-Wno-unused-variable", "-Wno-missing-braces" };

    // 系统库链接（通过模块）
    switch (os) {
        .windows => {
            lib_mod.linkSystemLibrary("ole32", .{});
            lib_mod.linkSystemLibrary("ws2_32", .{});
        },
        .macos, .ios, .tvos, .watchos => {
            lib_mod.linkFramework("Foundation", .{});
            lib_mod.linkFramework("Accelerate", .{});
            lib_mod.addCMacro("GGML_USE_ACCELERATE", "1");
            lib_mod.addCMacro("ACCELERATE_NEW_LAPACK", "1");
            lib_mod.addCMacro("ACCELERATE_LAPACK_ILP64", "1");
        },
        .linux => {
            lib_mod.linkSystemLibrary("pthread", .{});
        },
        else => {},
    }

    // ---- 辅助：连接标志切片 ----
    const ConcatFlags = struct {
        fn concat(alloc: std.mem.Allocator, a: []const []const u8, extra: []const []const u8) []const []const u8 {
            if (extra.len == 0) return a;
            const result = alloc.alloc([]const u8, a.len + extra.len) catch @panic("OOM");
            @memcpy(result[0..a.len], a);
            @memcpy(result[a.len..], extra);
            return result;
        }
    }.concat;

    // ---- 添加 C 源文件 ----
    inline for (.{ "ggml.c", "ggml-quants.c", "ggml-alloc.c" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(src_dir ++ "/" ++ file),
            .flags = c_base_flags,
        });
    }

    // ---- 添加 C++ 核心源文件 ----
    inline for (.{ "ggml.cpp", "ggml-backend.cpp", "ggml-backend-dl.cpp", "ggml-backend-reg.cpp", "ggml-backend-meta.cpp", "ggml-threading.cpp" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(src_dir ++ "/" ++ file),
            .flags = cpp_base_flags,
        });
    }

    // ---- 添加 CPU 后端 C 文件（含架构优化标志） ----
    inline for (.{ "ggml-cpu.c", "quants.c" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(cpu_dir ++ "/" ++ file),
            .flags = ConcatFlags(b.allocator, c_base_flags, x86_opt_flags),
        });
    }

    // ---- 添加 CPU 后端 C++ 文件（含架构优化标志） ----
    inline for (.{ "ggml-cpu.cpp", "repack.cpp", "hbm.cpp", "traits.cpp", "binary-ops.cpp", "unary-ops.cpp", "vec.cpp", "ops.cpp" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(cpu_dir ++ "/" ++ file),
            .flags = ConcatFlags(b.allocator, cpp_base_flags, x86_opt_flags),
        });
    }

    // ---- 添加 AMX C++ 文件（含架构优化标志） ----
    inline for (.{ "amx.cpp", "mmq.cpp" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(amx_dir ++ "/" ++ file),
            .flags = ConcatFlags(b.allocator, cpp_base_flags, x86_opt_flags),
        });
    }

    // ---- 架构特定源文件 ----
    switch (cpu_arch) {
        .x86_64 => {
            const arch_dir = cpu_dir ++ "/arch/x86";
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/cpu-feats.cpp"),
                .flags = cpp_base_flags,
            });
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/quants.c"),
                .flags = ConcatFlags(b.allocator, c_base_flags, x86_opt_flags),
            });
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/repack.cpp"),
                .flags = ConcatFlags(b.allocator, cpp_base_flags, x86_opt_flags),
            });
        },
        .aarch64 => {
            const arch_dir = cpu_dir ++ "/arch/arm";
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/cpu-feats.cpp"),
                .flags = cpp_base_flags,
            });
        },
        else => {
            // 其他架构：无架构特定优化文件
        },
    }

    // 创建静态库
    const lib = b.addLibrary(.{
        .name = "ggml",
        .root_module = lib_mod,
        .linkage = .static,
    });
    // 链接时优化（macOS 上 LTO 需要 LLD，但 LLD 不支持 Mach-O，故仅在 Linux 上启用）
    if (os == .linux) {
        lib.lto = .full;
        lib.use_lld = true;
    }

    return lib;
}
