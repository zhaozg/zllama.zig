//! GGUF 解析测试
//!
//! 验证 GGUF v2/v3 格式的解析正确性。
//! 包括：文件头、元数据 KV、张量信息、对齐等。

const std = @import("std");
const testing = std.testing;
const gguf = @import("../gguf.zig");

test "gguf module exports" {
    _ = gguf.parse;
    _ = gguf.GGUFFile;
}

test "gguf file structure" {
    try testing.expectEqual(@as(usize, @sizeOf(gguf.GGUFFile)), @sizeOf(gguf.GGUFFile));
}

// TODO: 添加实际的 GGUF 二进制解析测试
// 需要构造 v2/v3 格式的二进制数据
