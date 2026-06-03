//! Trie 前缀树实现
//!
//! 用于 BPE 分词器的贪婪最长匹配编码。
//! 将词表中的所有 token 字符串构建为 Trie 树，支持 O(n) 时间复杂度的前缀匹配。

const std = @import("std");

// ============================================================================
// Trie 节点
// ============================================================================

pub const TrieNode = struct {
    children: std.AutoHashMap(u8, *TrieNode),
    token_id: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator) TrieNode {
        return TrieNode{
            .children = std.AutoHashMap(u8, *TrieNode).init(allocator),
        };
    }

    pub fn deinit(self: *TrieNode, allocator: std.mem.Allocator) void {
        var it = self.children.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit(allocator);
            allocator.destroy(entry.value_ptr.*);
        }
        self.children.deinit();
    }
};

// ============================================================================
// 匹配结果
// ============================================================================

pub const MatchResult = struct {
    token_id: u32,
    len: usize,
};

// ============================================================================
// Trie 操作
// ============================================================================

/// 向 Trie 中添加一个 token 字符串
pub fn addToTrie(root: *TrieNode, token_str: []const u8, token_id: u32, allocator: std.mem.Allocator) !void {
    var node = root;
    for (token_str) |c| {
        const gop = try node.children.getOrPut(c);
        if (!gop.found_existing) {
            const new_node = try allocator.create(TrieNode);
            new_node.* = TrieNode.init(allocator);
            gop.value_ptr.* = new_node;
        }
        node = gop.value_ptr.*;
    }
    node.token_id = token_id;
}

/// 在 Trie 中查找从 pos 开始的最长匹配
pub fn longestMatch(root: *const TrieNode, text: []const u8, pos: usize) ?MatchResult {
    var node = root;
    var best: ?MatchResult = null;
    var cl: usize = 0;
    var i = pos;
    while (i < text.len) {
        if (node.children.get(text[i])) |child| {
            node = child;
            cl += 1;
            if (node.token_id) |tid| best = MatchResult{ .token_id = tid, .len = cl };
            i += 1;
        } else break;
    }
    return best;
}
