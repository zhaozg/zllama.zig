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
    exe_mod.linkSystemLibrary("ggml", .{});

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
    tokenize_mod.addIncludePath(include_path);
    tokenize_mod.linkSystemLibrary("ggml-base", .{});
    tokenize_mod.linkSystemLibrary("ggml", .{});

    const tokenize_exe = b.addExecutable(.{
        .name = "zllama-tokenize",
        .root_module = tokenize_mod,
    });

    // --- 测试 ---
    const test_step = b.step("test", "Run all tests");

    // 创建一个根模块用于测试，包含整个 src/ 目录
    // 这样子目录中的文件可以使用相对路径导入
    const test_root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_root_mod.addImport("ggml", ggml_mod);
    test_root_mod.addIncludePath(include_path);
    test_root_mod.linkSystemLibrary("ggml-base", .{});
    test_root_mod.linkSystemLibrary("ggml", .{});
    if (target.result.os.tag == .macos) {
        test_root_mod.linkFramework("Metal", .{});
        test_root_mod.linkFramework("Foundation", .{});
        test_root_mod.linkFramework("Accelerate", .{});
        test_root_mod.addCMacro("GGML_USE_METAL", "1");
        test_root_mod.addCMacro("GGML_USE_ACCELERATE", "1");
    }

    // 使用根模块运行所有测试
    const test_unit = b.addTest(.{
        .name = "all-tests",
        .root_module = test_root_mod,
    });
    test_step.dependOn(&test_unit.step);

    // --- 安装与运行 ---
    b.installArtifact(exe);
    b.installArtifact(tokenize_exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run zllama.zig engine");
    run_step.dependOn(&run_cmd.step);

    // --- tokenize 运行步骤 ---
    const tokenize_run_cmd = b.addRunArtifact(tokenize_exe);
    tokenize_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        tokenize_run_cmd.addArgs(args);
    }
    const tokenize_run_step = b.step("tokenize", "Run zllama-tokenize tool");
    tokenize_run_step.dependOn(&tokenize_run_cmd.step);
}
