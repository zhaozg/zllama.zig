//! SPM 编码（基于字符的 bigram 合并）
const std = @import("std");
const mod = @import("mod.zig");
const unicode = mod.unicode;
const encode_config = @import("encode_config.zig");
const EncodeConfig = encode_config.EncodeConfig;

/// SPM bigram 用于优先级队列
const SpmBigram = struct {
    left: i32,
    right: i32,
    score: f32,
    size: usize,

    fn lessThan(context: void, a: @This(), b: @This()) std.math.Order {
        _ = context;
        if (a.score > b.score) return .lt;
        if (a.score < b.score) return .gt;
        if (a.left < b.left) return .lt;
        if (a.left > b.left) return .gt;
        return .eq;
    }
};

/// SPM 符号
const SpmSymbol = struct {
    text: []const u8,
    n: usize,
    prev: i32,
    next: i32,
};

/// SPM 编码：将文本编码为 token ID 列表
pub fn encodeSPM(
    text: []const u8,
    config: *const EncodeConfig,
    allocator: std.mem.Allocator,
) !std.ArrayListUnmanaged(u32) {
    var tokens: std.ArrayListUnmanaged(u32) = .empty;

    if (text.len == 0) return tokens;

    var symbols = std.ArrayListUnmanaged(SpmSymbol){ .items = &.{}, .capacity = 0 };
    defer symbols.deinit(allocator);

    var offs: usize = 0;
    var index: i32 = 0;
    while (offs < text.len) {
        const ch_len = unicode.charLen(text, offs);
        try symbols.append(allocator, SpmSymbol{
            .text = text[offs..],
            .n = ch_len,
            .prev = index - 1,
            .next = if (offs + ch_len >= text.len) -1 else index + 1,
        });
        offs += ch_len;
        index += 1;
    }

    if (symbols.items.len == 0) return tokens;

    var work_queue = std.PriorityQueue(SpmBigram, void, SpmBigram.lessThan).initContext({});
    defer work_queue.deinit(allocator);

    for (1..symbols.items.len) |i| {
        tryAddSpmBigram(&symbols, &work_queue, @intCast(i - 1), @intCast(i), config);
    }

    while (work_queue.count() > 0) {
        const bigram = work_queue.pop().?;

        const left_idx = @as(usize, @intCast(bigram.left));
        const right_idx = @as(usize, @intCast(bigram.right));

        if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) continue;
        if (symbols.items[left_idx].next != bigram.right) continue;
        if (symbols.items[right_idx].prev != bigram.left) continue;
        if (symbols.items[left_idx].n + symbols.items[right_idx].n != bigram.size) continue;

        symbols.items[left_idx].n += symbols.items[right_idx].n;
        symbols.items[right_idx].n = 0;

        symbols.items[left_idx].next = symbols.items[right_idx].next;
        if (symbols.items[right_idx].next >= 0) {
            symbols.items[@as(usize, @intCast(symbols.items[right_idx].next))].prev = bigram.left;
        }

        tryAddSpmBigram(&symbols, &work_queue, symbols.items[left_idx].prev, bigram.left, config);
        tryAddSpmBigram(&symbols, &work_queue, bigram.left, symbols.items[left_idx].next, config);
    }

    {
        var i: i32 = 0;
        while (i >= 0) {
            const idx = @as(usize, @intCast(i));
            const sym = &symbols.items[idx];
            if (sym.n > 0) {
                const token_text = sym.text[0..sym.n];
                if (config.textToTokenFn(token_text, config.ctx)) |tid| {
                    try tokens.append(allocator, tid);
                } else {
                    for (token_text) |byte| {
                        const tid = config.byteToTokenIdFn(byte, config.ctx);
                        try tokens.append(allocator, tid);
                    }
                }
            }
            i = sym.next;
        }
    }

    return tokens;
}

/// 尝试添加 SPM bigram 到优先级队列
fn tryAddSpmBigram(
    symbols: *std.ArrayListUnmanaged(SpmSymbol),
    queue: *std.PriorityQueue(SpmBigram, void, SpmBigram.lessThan),
    left: i32,
    right: i32,
    config: *const EncodeConfig,
) void {
    if (left < 0 or right < 0) return;
    const left_idx = @as(usize, @intCast(left));
    const right_idx = @as(usize, @intCast(right));
    if (left_idx >= symbols.items.len or right_idx >= symbols.items.len) return;

    const left_sym = &symbols.items[left_idx];
    const right_sym = &symbols.items[right_idx];

    const merged_text = left_sym.text[0 .. left_sym.n + right_sym.n];
    const token_id = config.textToTokenFn(merged_text, config.ctx) orelse return;

    const score = config.tokenScoreFn(token_id, config.ctx);

    queue.push(config.allocator, SpmBigram{
        .left = left,
        .right = right,
        .score = score,
        .size = merged_text.len,
    }) catch {};
}
