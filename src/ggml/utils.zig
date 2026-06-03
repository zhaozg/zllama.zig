//! 工具函数
//!
//! 提供 ggml 版本信息、CPU 特性检测等工具函数。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;

// ============================================================================
// 版本信息
// ============================================================================

/// 获取 ggml 版本字符串
pub fn version() [:0]const u8 {
    return std.mem.sliceTo(c.ggml_version(), 0);
}

/// 获取 CPU 核心数（推荐线程数）
pub fn cpuNThreads() i32 {
    return @as(i32, @intCast(std.Thread.getCpuCount() catch 4));
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

/// 计算推荐线程数（物理核心数的 2/3 ~ 3/4）
pub fn recommendedThreads() i32 {
    const n = cpuNThreads();
    if (n <= 4) return n;
    return @max(1, @divTrunc(n * 3, 4));
}

// ============================================================================
// 测试
// ============================================================================

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
