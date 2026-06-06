const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 系统头文件路径
    const include_path: std.Build.LazyPath = .{ .cwd_relative = "/usr/local/include" };

    // --- ggml 模块（C 绑定 + 安全封装） ---
    const ggml_mod = b.createModule(.{
        .root_source_file = b.path("src/ggml.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ggml_mod.addIncludePath(include_path);
    ggml_mod.linkSystemLibrary("ggml-base", .{});
    ggml_mod.linkSystemLibrary("ggml", .{});

    if (target.result.os.tag == .macos) {
        ggml_mod.linkFramework("Metal", .{});
        ggml_mod.linkFramework("Foundation", .{});
        ggml_mod.linkFramework("Accelerate", .{});
        ggml_mod.addCMacro("GGML_USE_METAL", "1");
        ggml_mod.addCMacro("GGML_USE_ACCELERATE", "1");
    }

    // --- 内部模块 ---
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

    const graph_builder_mod = b.createModule(.{
        .root_source_file = b.path("src/core/graph_builder.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    graph_builder_mod.addImport("ggml", ggml_mod);
    graph_builder_mod.addImport("model", model_mod);
    graph_builder_mod.addImport("memory", memory_mod);

    // model_mod 需要 graph_builder_mod，所以 graph_builder_mod 必须在 model_mod 之后定义
    // 但 model_mod 已经创建了，所以可以添加导入
    model_mod.addImport("graph_builder", graph_builder_mod);

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

    const sampler_mod = b.createModule(.{
        .root_source_file = b.path("src/sampler.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    sampler_mod.addImport("ggml", ggml_mod);

    // 辅助函数：添加系统库和框架
    const addSystemLibs = struct {
        fn apply(mod: *std.Build.Module, t: std.Build.ResolvedTarget, inc: std.Build.LazyPath) void {
            mod.addIncludePath(inc);
            mod.linkSystemLibrary("ggml-base", .{});
            mod.linkSystemLibrary("ggml", .{});
            if (t.result.os.tag == .macos) {
                mod.linkFramework("Metal", .{});
                mod.linkFramework("Foundation", .{});
                mod.linkFramework("Accelerate", .{});
                mod.addCMacro("GGML_USE_METAL", "1");
                mod.addCMacro("GGML_USE_ACCELERATE", "1");
            }
        }
    }.apply;

    // --- 主可执行文件 ---
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
    exe_mod.addImport("kv_cache", kv_cache_mod);
    addSystemLibs(exe_mod, target, include_path);

    const exe = b.addExecutable(.{
        .name = "zllama",
        .root_module = exe_mod,
    });

    // --- zllama-tokenize 可执行文件 ---
    const tokenize_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenize_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tokenize_mod.addImport("ggml", ggml_mod);
    tokenize_mod.addImport("gguf", gguf_mod);
    tokenize_mod.addImport("tokenizer", tokenizer_mod);
    addSystemLibs(tokenize_mod, target, include_path);

    const tokenize_exe = b.addExecutable(.{
        .name = "zllama-tokenize",
        .root_module = tokenize_mod,
    });

    // --- zllama-simple 可执行文件 ---
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
    addSystemLibs(simple_mod, target, include_path);

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
    addSystemLibs(test_root_mod, target, include_path);

    const test_unit = b.addTest(.{
        .name = "unit-tests",
        .root_module = test_root_mod,
    });
    test_step.dependOn(&test_unit.step);

    const test_layers_step = b.step("test-layers", "Run layer tests only");
    test_layers_step.dependOn(&test_unit.step);

    const test_gguf_step = b.step("test-gguf", "Run GGUF tests only");
    test_gguf_step.dependOn(&test_unit.step);

    const test_archs_step = b.step("test-archs", "Run architecture tests only");
    test_archs_step.dependOn(&test_unit.step);

    const test_kv_cache_step = b.step("test-kv-cache", "Run KV Cache tests only");
    test_kv_cache_step.dependOn(&test_unit.step);

    const test_vocab_step = b.step("test-vocab", "Run vocab-based tokenizer tests only");
    test_vocab_step.dependOn(&test_unit.step);

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
        addSystemLibs(mod, target, include_path);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-dump-graph",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("dump-graph", "Run zllama-dump-graph tool");
        run_step.dependOn(&run_cmd.step);
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
        addSystemLibs(mod, target, include_path);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-compare-logits",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("compare-logits", "Run zllama-compare-logits tool");
        run_step.dependOn(&run_cmd.step);
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
        addSystemLibs(mod, target, include_path);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-gen-ref",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("gen-ref", "Run zllama-gen-ref tool");
        run_step.dependOn(&run_cmd.step);
    }

    // --- 安装与运行 ---
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
