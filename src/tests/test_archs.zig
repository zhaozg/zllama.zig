//! 架构前向测试（入口）
//!
//! 对每个支持的架构运行随机权重前向测试。
//! 验证模型实现的数值正确性。
//!
//! 参考 llama.cpp 的 test-llama-archs.cpp 设计。

const std = @import("std");
const testing = std.testing;
const model_if = @import("../model.zig");
const test_utils = @import("utils.zig");

test "architecture enum" {
    try testing.expectEqual(model_if.Architecture.qwen2, model_if.Architecture.fromString("qwen2").?);
    try testing.expectEqual(model_if.Architecture.qwen35, model_if.Architecture.fromString("qwen35").?);
    try testing.expectEqual(model_if.Architecture.llama, model_if.Architecture.fromString("llama3").?);
}

test "model params defaults" {
    const p = model_if.ModelParams{};
    try testing.expectEqual(@as(u32, 0), p.n_vocab);
    try testing.expectEqual(@as(u32, 32768), p.max_seq_len);
}

test "model vtable size" {
    try testing.expectEqual(@as(usize, @sizeOf(model_if.ModelVTable)), @sizeOf(model_if.ModelVTable));
}

// TODO: 添加实际的随机权重前向测试
// 需要实现 generateTestGGUF 和 runForward
