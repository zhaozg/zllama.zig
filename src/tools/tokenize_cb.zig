//! 工具间共享的多模态 tokenize 回调
//!
//! 提供 `tokenizeTextSegment` 函数，供 compare_mtmd_vision 和 compare_mtmd_audio 共用。

const std = @import("std");
const tokenizer = @import("tokenizer");

/// 标准的多模态 tokenize 回调函数
/// 与 llama.cpp 的 tokenizeTextSegment 行为一致：
/// 将文本编码为 token ID 序列。
pub fn tokenizeTextSegment(ctx: ?*anyopaque, text: []const u8, alloc: std.mem.Allocator) ![]u32 {
    const tok: *tokenizer.Tokenizer = @ptrCast(@alignCast(ctx orelse return error.NullCtx));
    var result = try tok.encode(text, false, false);
    defer result.deinit(alloc);
    return try result.toOwnedSlice(alloc);
}
