//! ggml.gguf 绑定层测试
//!
//! 测试 src/ggml/gguf.zig 中的 Context 封装和 MetadataValue 类型。

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

test "ggml.gguf.Context type size" {
    try testing.expectEqual(@sizeOf(*ggml.gguf.Context), @sizeOf(*anyopaque));
}

test "ggml.gguf.InitParams default" {
    const p = ggml.gguf.InitParams{};
    try testing.expectEqual(false, p.no_alloc);
    try testing.expect(p.ctx == null);
}

test "ggml.gguf.MetadataValue conversions" {
    const v = ggml.gguf.MetadataValue{ .int32 = 42 };
    try testing.expectEqual(@as(i64, 42), try v.asInt());
    try testing.expectError(error.TypeMismatch, v.asString());

    const s = ggml.gguf.MetadataValue{ .string = "hello" };
    try testing.expectEqualStrings("hello", try s.asString());
}

test "ggml.gguf.MetadataValue float conversion" {
    const v = ggml.gguf.MetadataValue{ .float32 = 3.14 };
    try testing.expectApproxEqAbs(@as(f64, 3.14), try v.asFloat(), 0.001);
}

test "ggml.gguf.MetadataValue bool conversion" {
    const v_true = ggml.gguf.MetadataValue{ .bool = true };
    try testing.expect(try v_true.asBool());

    const v_false = ggml.gguf.MetadataValue{ .bool = false };
    try testing.expect(!try v_false.asBool());
}

test "ggml.gguf.MetadataValue type mismatch" {
    const str_val = ggml.gguf.MetadataValue{ .string = "hello" };
    try testing.expectError(error.TypeMismatch, str_val.asInt());
    try testing.expectError(error.TypeMismatch, str_val.asFloat());
    try testing.expectError(error.TypeMismatch, str_val.asBool());

    const int_val = ggml.gguf.MetadataValue{ .int32 = 42 };
    try testing.expectError(error.TypeMismatch, int_val.asString());
}
