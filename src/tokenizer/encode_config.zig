//! 编码配置（EncodeConfig），被 encode.zig、encode_spm.zig、encode_word.zig 共享
const std = @import("std");
const mod = @import("mod.zig");
const types = mod.types;
const trie = mod.trie;

/// 编码所需的配置和回调函数
pub const EncodeConfig = struct {
    allocator: std.mem.Allocator,
    special: types.SpecialTokens,
    pre_type: types.PreTokenizerType,
    model: types.TokenizerModel,
    vocab: std.ArrayListUnmanaged(types.VocabEntry),
    merges: std.StringHashMap(u32),
    trie_root: *const trie.TrieNode,
    tokenToStringFn: *const fn (token_id: u32, ctx: ?*anyopaque) ?[]const u8,
    textToTokenFn: *const fn (text: []const u8, ctx: ?*anyopaque) ?u32,
    byteToTokenIdFn: *const fn (byte: u8, ctx: ?*anyopaque) u32,
    bytesToUnicodeFn: ?*const fn (byte: u8, ctx: ?*anyopaque) []const u8 = null,
    unicodeToByte: ?*const std.StringHashMap(u8) = null,
    tokenScoreFn: *const fn (token_id: u32, ctx: ?*anyopaque) f32 = undefined,
    escape_whitespaces: bool = false,
    ctx: ?*anyopaque,
    /// 缓存的特殊 token 列表（按 text 长度降序排列），用于 parse_special 模式
    cache_special_tokens: ?[]const mod.CacheSpecialToken = null,
};
