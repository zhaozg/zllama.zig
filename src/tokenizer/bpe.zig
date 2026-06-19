//! BPE 合并逻辑
//!
//! 实现 Byte Pair Encoding (BPE) 合并算法。
//! 根据合并规则（merges 表），反复合并最优先的相邻 token 对。
//!
//! 参考 llama.cpp 的 llm_tokenizer_bpe_session::tokenize 实现：
//! 1. 将每个词拆分为 UTF-8 字符（或字节）
//! 2. 使用优先级队列反复合并 rank 最低的相邻 token 对
//! 3. 合并后的 token 字符串必须在词表中存在

const std = @import("std");

const log = std.log.scoped(.tokenizer);

// ============================================================================
// 符号和 Bigram 类型
// ============================================================================

const Symbol = struct {
    token_id: u32,
    prev: i32,
    next: i32,
};

const Bigram = struct {
    left: i32,
    right: i32,
    rank: u32,

    /// Zig PriorityQueue 是最小堆（0.16），pop() 返回 lessThan 中 .lt 的元素
    /// BPE 要求 rank 最小（最频繁）的 pair 先弹出合并
    /// 因此：rank 较小 → .lt（优先级高，先弹出）
    fn lessThan(context: void, a: @This(), b: @This()) std.math.Order {
        _ = context;
        if (a.rank < b.rank) return .lt;
        if (a.rank > b.rank) return .gt;
        if (a.left < b.left) return .lt;
        if (a.left > b.left) return .gt;
        return .eq;
    }
};

// ============================================================================
// BPE 合并（基于优先级队列的正确实现）
// ============================================================================

/// 应用 BPE 合并规则到 token 列表
/// 使用优先级队列，反复合并 rank 最低的相邻 token 对
/// 参考 llama.cpp 的 add_new_bigram + work_queue 实现
pub fn applyBpeMerges(
    tokens: *std.ArrayListUnmanaged(u32),
    merges: std.StringHashMap(u32),
    tokenToStringFn: *const fn (token_id: u32, ctx: ?*anyopaque) ?[]const u8,
    textToTokenFn: *const fn (text: []const u8, ctx: ?*anyopaque) ?u32,
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
) !void {
    if (tokens.items.len < 2) return;
    if (merges.count() == 0) return;

    // 构建符号链表
    var symbols = std.ArrayListUnmanaged(Symbol){ .items = &.{}, .capacity = 0 };
    defer symbols.deinit(allocator);

    // 初始化符号链表
    for (tokens.items, 0..) |tid, i| {
        try symbols.append(allocator, Symbol{
            .token_id = tid,
            .prev = @as(i32, @intCast(i)) - 1,
            .next = if (i + 1 < tokens.items.len) @as(i32, @intCast(i + 1)) else -1,
        });
    }

    // 使用优先级队列（最小堆）
    var work_queue = std.PriorityQueue(Bigram, void, Bigram.lessThan).initContext({});
    defer work_queue.deinit(allocator);

    // 初始化队列：添加所有相邻 pair
    {
        var i: i32 = 0;
        while (i < @as(i32, @intCast(symbols.items.len)) - 1) {
            addNewBigram(&symbols, &work_queue, i, i + 1, merges, tokenToStringFn, textToTokenFn, ctx, allocator);
            i += 1;
        }
    }

    // 反复合并 rank 最低的 pair
    while (work_queue.count() > 0) {
        const bigram = work_queue.pop().?;

        const left_idx = @as(usize, @intCast(bigram.left));
        const right_idx = @as(usize, @intCast(bigram.right));

        // 检查符号是否已被合并
        if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) continue;
        if (symbols.items[left_idx].next != bigram.right) continue;
        if (symbols.items[right_idx].prev != bigram.left) continue;

        // 获取左右 token 的文本
        const left_str = tokenToStringFn(symbols.items[left_idx].token_id, ctx) orelse continue;
        const right_str = tokenToStringFn(symbols.items[right_idx].token_id, ctx) orelse continue;

        // 关键检查：重新计算当前 rank，与存储的 rank 比较
        // 如果 token 在入队后被合并过，token_id 改变 → left_str/right_str 改变 → rank 不同 → 跳过
        // 对应 llama.cpp: if (left_token + right_token != bigram.text) continue;
        const current_key = try std.fmt.allocPrint(allocator, "{s} {s}", .{ left_str, right_str });
        defer allocator.free(current_key);
        const current_rank = merges.get(current_key) orelse continue;
        if (current_rank != bigram.rank) continue;

        // 查找合并后的 token ID
        const merged_str = try std.fmt.allocPrint(allocator, "{s}{s}", .{ left_str, right_str });
        defer allocator.free(merged_str);
        const merged_token_id = textToTokenFn(merged_str, ctx) orelse continue;

        // 合并：将右符号合并到左符号
        symbols.items[left_idx].token_id = merged_token_id;
        symbols.items[left_idx].next = symbols.items[right_idx].next;

        // 无效化右符号：设置 prev = -1 以标记为已合并（对齐 llama.cpp 行为）
        symbols.items[right_idx].prev = -1;

        // 更新右符号的后继的前驱指针
        if (symbols.items[right_idx].next >= 0) {
            const next_idx = @as(usize, @intCast(symbols.items[right_idx].next));
            symbols.items[next_idx].prev = bigram.left;
        }

        // 添加新的 bigram
        addNewBigram(&symbols, &work_queue, symbols.items[left_idx].prev, bigram.left, merges, tokenToStringFn, textToTokenFn, ctx, allocator);
        addNewBigram(&symbols, &work_queue, bigram.left, symbols.items[left_idx].next, merges, tokenToStringFn, textToTokenFn, ctx, allocator);
    }

    // 从符号链表重建 token 列表
    tokens.clearRetainingCapacity();
    {
        var i: i32 = 0;
        while (i >= 0) {
            const idx = @as(usize, @intCast(i));
            try tokens.append(allocator, symbols.items[idx].token_id);
            i = symbols.items[idx].next;
        }
    }
}

/// 添加 bigram 到优先级队列
/// 对齐 llama.cpp: 即使 merged token 不在词表中，仍然入队（rank = maxInt）
/// 因为多步 BPE 合并中，中间结果可能不在词表但最终结果在
fn addNewBigram(
    symbols: *std.ArrayListUnmanaged(Symbol),
    queue: *std.PriorityQueue(Bigram, void, Bigram.lessThan),
    left: i32,
    right: i32,
    merges: std.StringHashMap(u32),
    tokenToStringFn: *const fn (token_id: u32, ctx: ?*anyopaque) ?[]const u8,
    textToTokenFn: *const fn (text: []const u8, ctx: ?*anyopaque) ?u32,
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
) void {
    if (left < 0 or right < 0) return;
    const left_idx = @as(usize, @intCast(left));
    const right_idx = @as(usize, @intCast(right));
    if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) return;

    const left_str = tokenToStringFn(symbols.items[left_idx].token_id, ctx) orelse return;
    const right_str = tokenToStringFn(symbols.items[right_idx].token_id, ctx) orelse return;

    // 查找合并规则的 rank
    const key = std.fmt.allocPrint(allocator, "{s} {s}", .{ left_str, right_str }) catch return;
    defer allocator.free(key);

    const rank = merges.get(key) orelse {
        return;
    };

    // 检查合并后的 token 是否在词表中
    // 对齐 llama.cpp 行为：如果合并后的 token 不在词表中，则不入队
    const merged_str = std.fmt.allocPrint(allocator, "{s}{s}", .{ left_str, right_str }) catch return;
    defer allocator.free(merged_str);
    if (textToTokenFn(merged_str, ctx) == null) return;

    queue.push(allocator, .{
        .left = left,
        .right = right,
        .rank = rank,
    }) catch {};
}
