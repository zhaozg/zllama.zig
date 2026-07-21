//! 工具函数
//!
//! 提供 ggml 版本信息、CPU 特性检测等工具函数。

const std = @import("std");
const builtin = @import("builtin");
const cmod = @import("c.zig");
const c = cmod.c;

// ============================================================================
// 版本信息
// ============================================================================

/// 获取 ggml 版本字符串
pub fn version() [:0]const u8 {
    return std.mem.sliceTo(c.ggml_version(), 0);
}

/// 获取逻辑 CPU 核心数（含超线程）
pub fn logicalCpuCount() i32 {
    return @as(i32, @intCast(std.Thread.getCpuCount() catch 4));
}

/// 获取物理 CPU 核心数（排除超线程）
/// 在 macOS 上通过 sysctl hw.physicalcpu 获取真实物理核心数，
/// 其他平台回退到 logicalCpuCount()。
pub fn physicalCpuCount() i32 {
    if (builtin.os.tag == .macos) {
        var count: c_int = 0;
        var count_len: usize = @sizeOf(c_int);
        const name: [:0]const u8 = "hw.physicalcpu";
        const rc = std.posix.system.sysctlbyname(name, &count, &count_len, null, 0);
        if (std.posix.errno(rc) == .SUCCESS and count > 0) {
            return @intCast(count);
        }
    }
    return logicalCpuCount();
}

/// 向后兼容别名：逻辑 CPU 核心数
pub fn cpuNThreads() i32 {
    return logicalCpuCount();
}

// ============================================================================
// CPU 特性检测
// ============================================================================

/// CPU 特性集合
pub const CpuFeatures = struct {
    pub fn hasSse3() bool {
        return c.ggml_cpu_has_sse3() != 0;
    }
    pub fn hasSsse3() bool {
        return c.ggml_cpu_has_ssse3() != 0;
    }
    pub fn hasAvx() bool {
        return c.ggml_cpu_has_avx() != 0;
    }
    pub fn hasAvxVnni() bool {
        return c.ggml_cpu_has_avx_vnni() != 0;
    }
    pub fn hasAvx2() bool {
        return c.ggml_cpu_has_avx2() != 0;
    }
    pub fn hasBmi2() bool {
        return c.ggml_cpu_has_bmi2() != 0;
    }
    pub fn hasF16c() bool {
        return c.ggml_cpu_has_f16c() != 0;
    }
    pub fn hasFma() bool {
        return c.ggml_cpu_has_fma() != 0;
    }
    pub fn hasAvx512() bool {
        return c.ggml_cpu_has_avx512() != 0;
    }
    pub fn hasNeon() bool {
        return c.ggml_cpu_has_neon() != 0;
    }
    pub fn hasArmFma() bool {
        return c.ggml_cpu_has_arm_fma() != 0;
    }
    pub fn hasFp16Va() bool {
        return c.ggml_cpu_has_fp16_va() != 0;
    }
    pub fn hasDotprod() bool {
        return c.ggml_cpu_has_dotprod() != 0;
    }
    pub fn hasWasmSimd() bool {
        return c.ggml_cpu_has_wasm_simd() != 0;
    }
    pub fn hasSve() bool {
        return c.ggml_cpu_has_sve() != 0;
    }
    pub fn hasSme() bool {
        return c.ggml_cpu_has_sme() != 0;
    }
};

/// 计算推荐线程数（物理核心数的 3/4，最少 1）
/// 利用物理核心避免超线程竞争，为系统预留余量。
pub fn recommendedThreads() i32 {
    const n = physicalCpuCount();
    if (n <= 4) return n;
    return @max(1, @divTrunc(n * 3, 4));
}

fn defaultLogCallback(level: c_uint, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    const log_level: LogLevel = @enumFromInt(level);
    const msg = std.mem.sliceTo(text, 0);
    // 使用 std.debug.print 输出，它内部使用 Io 实例
    std.debug.print("[ggml] [{s}] {s}", .{ log_level.name(), msg });
}
// ============================================================================
// 日志系统
pub const LogLevel = enum(c_uint) {
    none = 0,
    debug = 1,
    info = 2,
    warn = 3,
    err = 4,
    cont = 5,

    pub fn name(self: LogLevel) []const u8 {
        return switch (self) {
            .none => "NONE",
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
            .cont => "CONT",
        };
    }
};

/// 设置 ggml 日志回调
pub fn logSet() void {
    c.ggml_log_set(defaultLogCallback, null);
}

/// 设置自定义 ggml 日志回调
pub fn logSetCallback(callback: *const fn (level: c_uint, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void, user_data: ?*anyopaque) void {
    c.ggml_log_set(callback, user_data);
}

// ============================================================================
// 测试
// ============================================================================

// ============================================================================
// 跨平台 I/O 工具
// ============================================================================

/// 跨平台 write 函数，写入到文件描述符
/// 在 macOS 上使用系统调用，在 Linux 上使用系统调用
pub fn writeToFd(fd: i32, data: []const u8) usize {
    // 使用 @cImport 直接调用 POSIX write
    // 但为了避免在业务代码中 @cImport，我们在 ggml 封装层中提供此功能
    return c.write(fd, data.ptr, data.len);
}

const testing = std.testing;

test "ggml version" {
    const v = version();
    try testing.expect(v.len > 0);
    std.debug.print("ggml version: {s}\n", .{v});
}

test "CpuFeatures" {
    _ = CpuFeatures.hasAvx2();
    _ = CpuFeatures.hasNeon();
    _ = CpuFeatures.hasSve();
}

test "recommendedThreads" {
    const n = recommendedThreads();
    try testing.expect(n >= 1);
}

test "LogLevel enum" {
    try testing.expectEqual(LogLevel.none, @as(LogLevel, @enumFromInt(0)));
    try testing.expectEqual(LogLevel.info, @as(LogLevel, @enumFromInt(2)));
    try testing.expectEqualStrings("INFO", LogLevel.info.name());
}
