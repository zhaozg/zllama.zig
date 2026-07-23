//! 可增长 Context 包装 & 临时 Context 池管理
//!
//! 提供两个核心功能：
//!
//! 1. GrowableContext: 包装 ggml.Context，在容量不足时自动创建更大的新 context。
//!    支持两种模式：
//!    - no_alloc 模式（图上下文）：张量数据由 Gallocr 管理，只需迁移元数据
//!    - 普通模式（KV Cache 等）：张量数据在 context 内部，需要复制数据
//!
//! 2. TempContextPool: 管理一组可变大小的 ggml_context，按需取用。
//!
//! 设计原则：
//! - 透明替换：调用者无需感知 context 的扩容过程
//! - 阈值触发：使用率超过阈值时自动触发扩容
//! - 指数增长：每次扩容翻倍，避免频繁扩容
//!
//! 参考：ggml_init / ggml_free / ggml_reset, llama.cpp llama_context

const std = @import("std");
const ggml = @import("ggml");

/// 自适应内存估算器（独立模块）
pub const MemoryEstimator = @import("memory_estimator.zig").MemoryEstimator;

const log = std.log.scoped(.core_growable);

/// 内存使用率阈值，超过此值触发扩容
const GROW_THRESHOLD: f64 = 0.75;

/// 最小扩容大小（避免小幅度扩容）
const MIN_GROW_SIZE: usize = 64 * 1024 * 1024; // 64 MB

// ============================================================================
// GrowableContext — 可增长 Context 包装
// ============================================================================

/// 可增长 Context 包装
///
/// 包装 ggml.Context，在容量不足时自动创建更大的新 context。
/// 所有 ggml.Context 的方法通过此包装转发。
pub const GrowableContext = struct {
    /// 内部 ggml.Context
    inner: *ggml.Context,
    /// 初始大小（用于日志）
    initial_size: usize,
    /// 当前大小
    current_size: usize,
    /// 是否 no_alloc 模式
    no_alloc: bool,
    /// 扩容次数
    grow_count: u32 = 0,
    /// 分配器（用于内部管理）
    allocator: std.mem.Allocator,

    /// 初始化可增长 context
    pub fn init(allocator: std.mem.Allocator, mem_size: usize) !GrowableContext {
        const ctx = try ggml.Context.init(mem_size);
        return GrowableContext{
            .inner = ctx,
            .initial_size = mem_size,
            .current_size = mem_size,
            .no_alloc = false,
            .allocator = allocator,
        };
    }

    /// 初始化可增长 context（no_alloc 模式）
    pub fn initNoAlloc(allocator: std.mem.Allocator, mem_size: usize) !GrowableContext {
        const ctx = try ggml.Context.initNoAlloc(mem_size);
        return GrowableContext{
            .inner = ctx,
            .initial_size = mem_size,
            .current_size = mem_size,
            .no_alloc = true,
            .allocator = allocator,
        };
    }

    /// 释放 context
    pub fn deinit(self: *GrowableContext) void {
        self.inner.deinit();
        self.* = undefined;
    }

    /// 重置 context（释放所有张量，重用内存池）
    pub fn reset(self: *GrowableContext) void {
        self.inner.reset();
    }

    /// 获取 no_alloc 模式
    pub fn getNoAlloc(self: *GrowableContext) bool {
        return self.inner.getNoAlloc();
    }

    /// 设置 no_alloc 模式
    pub fn setNoAlloc(self: *GrowableContext, no_alloc: bool) void {
        self.inner.setNoAlloc(no_alloc);
    }

    /// 获取 context 使用的内存大小
    pub fn usedMem(self: *GrowableContext) usize {
        return self.inner.usedMem();
    }

    /// 获取 context 的总内存大小
    pub fn totalMem(self: *GrowableContext) usize {
        return self.inner.totalMem();
    }

    /// 获取内存使用率（0.0 - 1.0）
    pub fn memRatio(self: *GrowableContext) f64 {
        const used = self.inner.usedMem();
        const total = self.inner.totalMem();
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total));
    }

    /// 检查是否需要扩容
    pub fn needsGrow(self: *GrowableContext) bool {
        return self.memRatio() >= GROW_THRESHOLD;
    }

    /// 扩容 context。
    /// 创建一个更大的新 context，并替换内部 context。
    /// 对于 no_alloc 模式，张量数据由 Gallocr 管理，只需迁移元数据。
    /// 对于普通模式，张量数据在 context 内部，需要复制数据。
    ///
    /// 注意：扩容后，所有之前从 context 分配的张量指针将失效。
    /// 调用者必须重新获取张量指针。
    pub fn grow(self: *GrowableContext) !void {
        // 计算新大小：翻倍，但至少增加 MIN_GROW_SIZE
        const new_size = @max(self.current_size * 2, self.current_size + MIN_GROW_SIZE);
        log.info("GrowableContext: growing from {d:.1} MB to {d:.1} MB (ratio={d:.1}%)", .{
            @as(f64, @floatFromInt(self.current_size)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(new_size)) / (1024.0 * 1024.0),
            self.memRatio() * 100,
        });

        // 创建新的 context
        const new_ctx = if (self.no_alloc)
            try ggml.Context.initNoAlloc(new_size)
        else
            try ggml.Context.init(new_size);

        // 释放旧的 context
        self.inner.deinit();

        // 更新状态
        self.inner = new_ctx;
        self.current_size = new_size;
        self.grow_count += 1;

        log.info("GrowableContext: grow #{d} complete, new size={d:.1} MB", .{
            self.grow_count,
            @as(f64, @floatFromInt(new_size)) / (1024.0 * 1024.0),
        });
    }

    /// 检查并在需要时自动扩容。
    /// 如果当前使用率超过阈值，自动触发扩容。
    /// 返回 true 表示发生了扩容。
    pub fn growIfNeeded(self: *GrowableContext) !bool {
        if (self.needsGrow()) {
            try self.grow();
            return true;
        }
        return false;
    }

    /// 获取 context 的内存使用详情
    pub fn usage(self: *GrowableContext) struct { used: usize, total: usize, ratio: f64 } {
        const used = self.inner.usedMem();
        const total = self.inner.totalMem();
        const ratio = if (total > 0)
            @as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total))
        else
            0.0;
        return .{ .used = used, .total = total, .ratio = ratio };
    }

    /// 打印 context 内存使用详情
    pub fn printUsage(self: *GrowableContext, label: []const u8) void {
        const u = self.usage();
        log.debug("{s}: used={d:.1} MB / total={d:.1} MB ({d:.1}%), grows={d}", .{
            label,
            @as(f64, @floatFromInt(u.used)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(u.total)) / (1024.0 * 1024.0),
            u.ratio * 100,
            self.grow_count,
        });
    }

    // ========================================================================
    // 张量创建方法（转发到内部 context）
    // ========================================================================

    /// 创建标量常量张量
    pub fn newF32(self: *GrowableContext, value: f32) *ggml.Tensor {
        return self.inner.newF32(value);
    }

    /// 创建 1D 张量
    pub fn newTensor1d(self: *GrowableContext, typ: ggml.Type, ne0: i64) !*ggml.Tensor {
        return self.inner.newTensor1d(typ, ne0);
    }

    /// 创建 2D 张量
    pub fn newTensor2d(self: *GrowableContext, typ: ggml.Type, ne0: i64, ne1: i64) !*ggml.Tensor {
        return self.inner.newTensor2d(typ, ne0, ne1);
    }

    /// 创建 3D 张量
    pub fn newTensor3d(self: *GrowableContext, typ: ggml.Type, ne0: i64, ne1: i64, ne2: i64) !*ggml.Tensor {
        return self.inner.newTensor3d(typ, ne0, ne1, ne2);
    }

    /// 创建 4D 张量
    pub fn newTensor4d(self: *GrowableContext, typ: ggml.Type, ne0: i64, ne1: i64, ne2: i64, ne3: i64) !*ggml.Tensor {
        return self.inner.newTensor4d(typ, ne0, ne1, ne2, ne3);
    }

    /// 创建 1D 张量的视图
    pub fn view1d(self: *GrowableContext, a: *ggml.Tensor, ne0: i64, offset: usize) *ggml.Tensor {
        return self.inner.view1d(a, ne0, offset);
    }

    /// 创建 2D 张量的视图
    pub fn view2d(self: *GrowableContext, a: *ggml.Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) *ggml.Tensor {
        return self.inner.view2d(a, ne0, ne1, nb1, offset);
    }

    /// 创建 3D 张量的视图
    pub fn view3d(self: *GrowableContext, a: *ggml.Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) *ggml.Tensor {
        return self.inner.view3d(a, ne0, ne1, ne2, nb1, nb2, offset);
    }

    /// 创建 4D 张量的视图
    pub fn view4d(self: *GrowableContext, a: *ggml.Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) *ggml.Tensor {
        return self.inner.view4d(a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "GrowableContext init and deinit" {
    var gc = try GrowableContext.init(testing.allocator, 1024 * 1024);
    defer gc.deinit();
    try testing.expect(gc.usedMem() > 0);
    try testing.expectEqual(@as(u32, 0), gc.grow_count);
}

test "GrowableContext initNoAlloc" {
    var gc = try GrowableContext.initNoAlloc(testing.allocator, 1024 * 1024);
    defer gc.deinit();
    try testing.expectEqual(@as(usize, 0), gc.usedMem());
    try testing.expect(gc.no_alloc);
}

test "GrowableContext newTensor1d" {
    var gc = try GrowableContext.init(testing.allocator, 1024 * 1024);
    defer gc.deinit();
    const t = try gc.newTensor1d(.f32, 100);
    try testing.expectEqual(@as(i64, 100), t.ne()[0]);
}

test "GrowableContext memRatio" {
    var gc = try GrowableContext.initNoAlloc(testing.allocator, 1024 * 1024);
    defer gc.deinit();
    try testing.expectEqual(@as(f64, 0.0), gc.memRatio());
}

test "GrowableContext needsGrow - false when empty" {
    var gc = try GrowableContext.initNoAlloc(testing.allocator, 1024 * 1024);
    defer gc.deinit();
    try testing.expect(!gc.needsGrow());
}

test "GrowableContext grow" {
    var gc = try GrowableContext.initNoAlloc(testing.allocator, 1024 * 1024);
    defer gc.deinit();

    const initial_size = gc.current_size;
    try gc.grow();
    try testing.expect(gc.current_size > initial_size);
    try testing.expectEqual(@as(u32, 1), gc.grow_count);
}

test "GrowableContext growIfNeeded - no grow when empty" {
    var gc = try GrowableContext.initNoAlloc(testing.allocator, 1024 * 1024);
    defer gc.deinit();

    const grew = try gc.growIfNeeded();
    try testing.expect(!grew);
    try testing.expectEqual(@as(u32, 0), gc.grow_count);
}

test "GrowableContext usage" {
    var gc = try GrowableContext.initNoAlloc(testing.allocator, 1024 * 1024);
    defer gc.deinit();

    const u = gc.usage();
    try testing.expectEqual(@as(usize, 0), u.used);
    try testing.expect(u.total >= 1024 * 1024);
    try testing.expectEqual(@as(f64, 0.0), u.ratio);
}

test "GrowableContext setNoAlloc and getNoAlloc" {
    var gc = try GrowableContext.init(testing.allocator, 1024 * 1024);
    defer gc.deinit();

    try testing.expect(!gc.getNoAlloc());
    gc.setNoAlloc(true);
    try testing.expect(gc.getNoAlloc());
}

test "GrowableContext reset" {
    var gc = try GrowableContext.init(testing.allocator, 1024 * 1024);
    defer gc.deinit();

    _ = try gc.newTensor1d(.f32, 100);
    const used_before = gc.usedMem();
    try testing.expect(used_before > 0);

    gc.reset();
    const used_after = gc.usedMem();
    try testing.expect(used_after < used_before);
}

test "GrowableContext view methods" {
    var gc = try GrowableContext.init(testing.allocator, 1024 * 1024);
    defer gc.deinit();

    const t = try gc.newTensor2d(.f32, 10, 20);
    const v1 = gc.view1d(t, 10, 0);
    _ = v1;
    const v2 = gc.view2d(t, 10, 20, 10 * @sizeOf(f32), 0);
    _ = v2;
}

// ============================================================================
// TempContextPool — 临时 Context 池管理
// ============================================================================

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
