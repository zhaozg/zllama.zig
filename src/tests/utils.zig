//! 测试工具函数
//!
//! 提供测试中常用的辅助函数：
//! - NMSE 计算
//! - 随机张量生成
//! - 测试配置管理

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");

/// 测试配置
pub const TestConfig = struct {
    arch: model_if.Architecture,
    n_layer: u32 = 2,
    n_embd: u32 = 64,
    n_head: u32 = 4,
    n_kv_head: u32 = 2,
    n_ff: u32 = 128,
    n_vocab: u32 = 128,
    n_head_dim: u32 = 16,
    max_seq_len: u32 = 128,
    rope_theta: f32 = 10000000.0,
    rope_dim: u32 = 16,
    norm_eps: f32 = 1e-6,
    seed: u64 = 42,
};

/// 计算归一化均方误差 (NMSE)
/// nmse = mse(a, b) / mse(a, 0)
pub fn nmse(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var mse_a_b: f64 = 0.0;
    var mse_a_0: f64 = 0.0;
    for (a, b) |a_val, b_val| {
        const a_f64: f64 = @floatCast(a_val);
        const b_f64: f64 = @floatCast(b_val);
        const diff = a_f64 - b_f64;
        mse_a_b += diff * diff;
        mse_a_0 += a_f64 * a_f64;
    }
    if (mse_a_0 == 0.0) return 0.0;
    return mse_a_b / mse_a_0;
}

/// 生成随机 GGUF 数据（用于测试）
/// 返回分配的字节数组，调用者负责释放
pub fn generateTestGGUF(allocator: std.mem.Allocator, config: TestConfig) ![]u8 {
    _ = allocator;
    _ = config;
    // TODO: 实现随机 GGUF 生成
    // 1. 创建 GGUF 写入器
    // 2. 写入元数据（架构、维度等）
    // 3. 生成随机权重（正态分布，std=0.01）
    // 4. 返回 GGUF 字节
    @panic("Not implemented yet");
}

/// 创建测试用的 ggml context
pub fn createTestContext(_: std.mem.Allocator) !*ggml.Context {
    const mem_size: usize = 64 * 1024 * 1024; // 64MB
    return try ggml.Context.initNoAlloc(mem_size);
}

const testing = std.testing;

test "nmse identical" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(@as(f64, 0.0), nmse(&a, &b));
}

test "nmse different" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expect(nmse(&a, &b) < 1e-10);
}

test "TestConfig defaults" {
    const c = TestConfig{ .arch = .llama };
    try testing.expectEqual(@as(u32, 2), c.n_layer);
    try testing.expectEqual(@as(u64, 42), c.seed);
}
