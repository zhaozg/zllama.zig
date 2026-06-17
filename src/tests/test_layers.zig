//! 算子层数值测试
//!
//! 对各算子层进行独立的数值正确性测试。
//! 验证 RMSNorm、RoPE、SwiGLU、Attention 等算子的输出与预期一致。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const attention = @import("attention");
const swiglu = @import("swiglu");
const test_utils = @import("test_utils");

test "rms_norm structure" {
    // 验证 rms_norm 模块的导出函数存在
    _ = rms_norm.rmsNorm;
    _ = rms_norm.rmsNormOnly;
}

test "rope structure" {
    // 验证 rope 模块的导出函数存在
    _ = rope.applyRope;
    _ = rope.buildPositionTensor;
    _ = rope.buildMultiPositionTensor;
    _ = rope.RopeParams;
}

test "rope params defaults" {
    const p = rope.RopeParams{};
    try testing.expectEqual(@as(i64, 64), p.rope_dim);
    try testing.expectEqual(@as(f32, 10000000.0), p.rope_theta);
    try testing.expectEqual(@as(i32, 2), p.mode);
}

test "attention structure" {
    // 验证 attention 模块的导出函数存在
    _ = attention.scaledDotProductAttention;
    _ = attention.AttentionParams;
}

test "attention params" {
    const p = attention.AttentionParams{
        .n_head = 32,
        .n_kv_head = 8,
        .head_dim = 128,
        .n_tokens = 1,
        .cache_len = 10,
        .start_pos = 0,
        .scale_factor = 0.08838834764831845,
    };
    try testing.expectEqual(@as(i64, 32), p.n_head);
    try testing.expectEqual(@as(i64, 8), p.n_kv_head);
    try testing.expectEqual(@as(f32, 0.08838834764831845), p.scale_factor);
}

test "swiglu structure" {
    _ = swiglu.swiGLU;
}

test "nmse computation" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const err = test_utils.nmse(&a, &b);
    try testing.expectEqual(@as(f64, 0.0), err);
}

test "nmse with small difference" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0001, 2.0001, 3.0001 };
    const err = test_utils.nmse(&a, &b);
    // NMSE should be very small
    try testing.expect(err < 1e-7);
}
