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

    if (target.result.os.tag == .macos) {
        ggml_mod.linkFramework("Metal", .{});
        ggml_mod.linkFramework("Foundation", .{});
        ggml_mod.linkFramework("Accelerate", .{});
        ggml_mod.addCMacro("GGML_USE_METAL", "1");
        ggml_mod.addCMacro("GGML_USE_ACCELERATE", "1");
    }

    // --- 主可执行文件 ---
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe_mod.addImport("ggml", ggml_mod);
    exe_mod.addIncludePath(include_path);
    exe_mod.linkSystemLibrary("ggml-base", .{});

    const exe = b.addExecutable(.{
        .name = "qwen",
        .root_module = exe_mod,
    });

    // --- 测试 ---
    const test_step = b.step("test", "Run all tests");

    // ggml.zig 测试
    const ggml_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ggml.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ggml_test_mod.addIncludePath(include_path);
    ggml_test_mod.linkSystemLibrary("ggml-base", .{});
    if (target.result.os.tag == .macos) {
        ggml_test_mod.linkFramework("Metal", .{});
        ggml_test_mod.linkFramework("Foundation", .{});
        ggml_test_mod.linkFramework("Accelerate", .{});
        ggml_test_mod.addCMacro("GGML_USE_METAL", "1");
        ggml_test_mod.addCMacro("GGML_USE_ACCELERATE", "1");
    }

    const ggml_test = b.addTest(.{
        .name = "ggml-test",
        .root_module = ggml_test_mod,
    });
    test_step.dependOn(&ggml_test.step);

    // gguf.zig 测试
    const gguf_test_mod = b.createModule(.{
        .root_source_file = b.path("src/gguf.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gguf_test_mod.addImport("ggml", ggml_mod);
    gguf_test_mod.addIncludePath(include_path);
    gguf_test_mod.linkSystemLibrary("ggml-base", .{});

    const gguf_test = b.addTest(.{
        .name = "gguf-test",
        .root_module = gguf_test_mod,
    });
    test_step.dependOn(&gguf_test.step);

    // main.zig 测试
    const main_test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    main_test_mod.addImport("ggml", ggml_mod);
    main_test_mod.addIncludePath(include_path);
    main_test_mod.linkSystemLibrary("ggml-base", .{});

    const main_test = b.addTest(.{
        .name = "main-test",
        .root_module = main_test_mod,
    });
    test_step.dependOn(&main_test.step);

    // --- 安装与运行 ---
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run qwen engine");
    run_step.dependOn(&run_cmd.step);
}
