//! BPE 合并逻辑
//!
//! 实现 Byte Pair Encoding (BPE) 合并算法。
//! 根据合并规则（merges 表），反复合并最优先的相邻 token 对。

const std = @import("std");

const log = std.log.scoped(.tokenizer);

// ============================================================================
// BPE 合并
// ============================================================================

/// 应用 BPE 合并规则到 token 列表
/// 反复查找并合并最优先的相邻 token 对，直到无法继续
pub fn applyBpeMerges(
    tokens: *std.ArrayListUnmanaged(u32),
    merges: std.StringHashMap(u32),
    tokenToStringFn: *const fn (token_id: u32, ctx: ?*anyopaque) ?[]const u8,
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
) !void {
    if (tokens.items.len < 2) return;
    if (merges.count() == 0) return;

    var changed = true;
    while (changed) {
        changed = false;
        var best_idx: ?usize = null;
        var best_rank: u32 = std.math.maxInt(u32);
        var i: usize = 0;
        while (i + 1 < tokens.items.len) {
            const ls = tokenToStringFn(tokens.items[i], ctx) orelse {
                i += 1;
                continue;
            };
            const rs = tokenToStringFn(tokens.items[i + 1], ctx) orelse {
                i += 1;
                continue;
            };

            // 使用空格分隔的格式查找合并规则（与 merges 中的 key 格式一致）
            // 注意：key 必须分配在堆上，因为 merges 中的 key 是长期存在的
            const key = try std.fmt.allocPrint(allocator, "{s} {s}", .{ ls, rs });
            defer allocator.free(key);

            if (merges.get(key)) |rank| {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                }
            }
            i += 1;
        }

        if (best_idx) |idx| {
            // 合并 token[idx] 和 token[idx+1]
            // 简化实现：直接删除第二个 token，保留第一个
            _ = tokens.orderedRemove(idx + 1);
            changed = true;
        }
    }
}
