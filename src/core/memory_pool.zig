//! 临时 Context 池管理（docs/MEM.md §7.2）
//!
//! 管理一组可变大小的 ggml_context，按需取用，支持 reset 和 free。
//! 用于分离持久 context（权重、KV Cache）与临时 context（中间激活）。
//!
//! 设计原则：
//! - 按大小分级管理：context 按容量大小分组，避免频繁创建/销毁
//! - 延迟释放：释放的 context 放回池中复用，减少 ggml_init/free 开销
//! - 自动扩容：当池中无足够大的 context 时，自动创建新的
//!
//! 参考：llama.cpp 的 llama_context 内存管理

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.core_mem_pool);

/// 临时 Context 池
pub const TempContextPool = struct {
    allocator: std.mem.Allocator,
    /// 空闲 context 列表（按大小升序排列）
    free_list: std.ArrayList(*ggml.Context),
    /// 已借出的 context 列表（用于调试和泄漏检测）
    borrowed_list: std.ArrayList(*ggml.Context),
    /// 统计信息
    stats: PoolStats,

    /// 池统计信息
    pub const PoolStats = struct {
        /// 创建的 context 总数
        total_created: u32 = 0,
        /// 销毁的 context 总数
        total_destroyed: u32 = 0,
        /// 借出次数
        total_acquires: u32 = 0,
        /// 归还次数
        total_releases: u32 = 0,
        /// 因容量不足而扩容的次数
        total_resizes: u32 = 0,
        /// 当前空闲 context 数
        current_free: u32 = 0,
        /// 当前借出 context 数
        current_borrowed: u32 = 0,
    };

    /// 初始化池
    pub fn init(allocator: std.mem.Allocator) TempContextPool {
        return TempContextPool{
            .allocator = allocator,
            .free_list = std.ArrayList(*ggml.Context).init(allocator),
            .borrowed_list = std.ArrayList(*ggml.Context).init(allocator),
            .stats = .{},
        };
    }

    /// 释放池中所有资源
    pub fn deinit(self: *TempContextPool) void {
        // 释放所有空闲 context
        for (self.free_list.items) |ctx| {
            ctx.deinit();
            self.stats.total_destroyed += 1;
        }
        self.free_list.deinit();

        // 警告：仍有借出的 context
        if (self.borrowed_list.items.len > 0) {
            log.warn("TempContextPool deinit: {d} contexts still borrowed, freeing them", .{
                self.borrowed_list.items.len,
            });
            for (self.borrowed_list.items) |ctx| {
                ctx.deinit();
                self.stats.total_destroyed += 1;
            }
        }
        self.borrowed_list.deinit();
    }

    /// 从池中获取一个大小 >= min_size 的临时 context。
    /// 如果池中有合适的空闲 context，直接返回（已 reset）；
    /// 否则创建新的 context。
    pub fn acquire(self: *TempContextPool, min_size: usize) !*ggml.Context {
        self.stats.total_acquires += 1;

        // 查找大小 >= min_size 的空闲 context
        for (self.free_list.items, 0..) |ctx, i| {
            const total = ctx.totalMem();
            if (total >= min_size) {
                // 找到合适的 context，从空闲列表中移除
                _ = self.free_list.swapRemove(i);
                // 确保 context 已重置
                ctx.reset();
                try self.borrowed_list.append(ctx);
                self.stats.current_free = @intCast(self.free_list.items.len);
                self.stats.current_borrowed = @intCast(self.borrowed_list.items.len);
                log.debug("TempContextPool: reused context (size={d:.1} MB, requested={d:.1} MB)", .{
                    @as(f64, @floatFromInt(total)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(min_size)) / (1024.0 * 1024.0),
                });
                return ctx;
            }
        }

        // 没有合适的空闲 context，创建新的
        // 使用 min_size 加 20% 余量，避免频繁扩容
        const margin: usize = @intFromFloat(@as(f64, @floatFromInt(min_size)) * 0.20);
        const ctx_size = min_size + margin;
        const ctx = try ggml.Context.initNoAlloc(ctx_size);
        self.stats.total_created += 1;
        try self.borrowed_list.append(ctx);
        self.stats.current_free = @intCast(self.free_list.items.len);
        self.stats.current_borrowed = @intCast(self.borrowed_list.items.len);
        log.debug("TempContextPool: created new context (size={d:.1} MB)", .{
            @as(f64, @floatFromInt(ctx_size)) / (1024.0 * 1024.0),
        });
        return ctx;
    }

    /// 将 context 归还到池中。
    /// context 会被彻底重置（reset），然后放回空闲列表。
    pub fn release(self: *TempContextPool, ctx: *ggml.Context) void {
        self.stats.total_releases += 1;

        // 从借出列表中移除
        const idx = self.indexOf(self.borrowed_list.items, ctx);
        if (idx) |i| {
            _ = self.borrowed_list.swapRemove(i);
        } else {
            log.warn("TempContextPool: releasing unknown context (not in borrowed list)", .{});
        }

        // 彻底重置 context
        ctx.reset();

        // 放回空闲列表
        self.free_list.append(ctx) catch |err| {
            log.err("TempContextPool: failed to return context to free list ({})", .{err});
            ctx.deinit();
            self.stats.total_destroyed += 1;
            return;
        };

        self.stats.current_free = @intCast(self.free_list.items.len);
        self.stats.current_borrowed = @intCast(self.borrowed_list.items.len);
        log.debug("TempContextPool: released context back to pool", .{});
    }

    /// 获取当前池中空闲 context 数量
    pub fn freeCount(self: *const TempContextPool) u32 {
        return @intCast(self.free_list.items.len);
    }

    /// 获取当前借出的 context 数量
    pub fn borrowedCount(self: *const TempContextPool) u32 {
        return @intCast(self.borrowed_list.items.len);
    }

    /// 打印池统计信息
    pub fn printStats(self: *const TempContextPool) void {
        log.info("TempContextPool stats:", .{});
        log.info("  Created:    {d}", .{self.stats.total_created});
        log.info("  Destroyed:  {d}", .{self.stats.total_destroyed});
        log.info("  Acquires:   {d}", .{self.stats.total_acquires});
        log.info("  Releases:   {d}", .{self.stats.total_releases});
        log.info("  Resizes:    {d}", .{self.stats.total_resizes});
        log.info("  Free:       {d}", .{self.stats.current_free});
        log.info("  Borrowed:   {d}", .{self.stats.current_borrowed});
    }

    /// 在切片中查找 context 的索引
    fn indexOf(slice: []const *ggml.Context, ctx: *const ggml.Context) ?usize {
        for (slice, 0..) |item, i| {
            if (@intFromPtr(item) == @intFromPtr(ctx)) return i;
        }
        return null;
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "TempContextPool init and deinit" {
    var pool = TempContextPool.init(testing.allocator);
    defer pool.deinit();

    try testing.expectEqual(@as(u32, 0), pool.freeCount());
    try testing.expectEqual(@as(u32, 0), pool.borrowedCount());
}

test "TempContextPool acquire and release" {
    var pool = TempContextPool.init(testing.allocator);
    defer pool.deinit();

    const ctx = try pool.acquire(64 * 1024);
    defer pool.release(ctx);

    try testing.expectEqual(@as(u32, 1), pool.borrowedCount());
    try testing.expectEqual(@as(u32, 0), pool.freeCount());
    try testing.expect(ctx.totalMem() >= 64 * 1024);
}

test "TempContextPool reuse" {
    var pool = TempContextPool.init(testing.allocator);
    defer pool.deinit();

    const ctx1 = try pool.acquire(64 * 1024);
    const ptr1 = @intFromPtr(ctx1);
    pool.release(ctx1);

    try testing.expectEqual(@as(u32, 1), pool.freeCount());
    try testing.expectEqual(@as(u32, 0), pool.borrowedCount());

    // 再次获取应该复用同一个 context
    const ctx2 = try pool.acquire(64 * 1024);
    const ptr2 = @intFromPtr(ctx2);
    try testing.expectEqual(ptr1, ptr2);
    pool.release(ctx2);
}

test "TempContextPool auto-create larger" {
    var pool = TempContextPool.init(testing.allocator);
    defer pool.deinit();

    const ctx_small = try pool.acquire(64 * 1024);
    const ptr_small = @intFromPtr(ctx_small);
    pool.release(ctx_small);

    // 请求更大的 context，应创建新的
    const ctx_large = try pool.acquire(1024 * 1024);
    const ptr_large = @intFromPtr(ctx_large);
    try testing.expect(ptr_large != ptr_small);
    try testing.expect(ctx_large.totalMem() >= 1024 * 1024);
    pool.release(ctx_large);
}
