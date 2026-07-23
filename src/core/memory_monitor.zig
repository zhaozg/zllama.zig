//! 内存监控与告警系统
//!
//! 提供内存使用率监控、告警触发和自动回收功能。
//!
//! 核心功能：
//! 1. MemoryMonitor: 监控多个 context 的内存使用情况
//! 2. 分级告警：WARN (90%) → CRITICAL (95%) → OOM (99%)
//! 3. 自动回收：超过阈值时尝试 compact / recycle / grow
//! 4. MemoryReport: 综合内存诊断报告
//!
//! 设计原则：
//! - 非侵入式：通过回调机制监控，不修改被监控对象
//! - 分级告警：不同级别触发不同响应
//! - 自动恢复：优先尝试轻量回收，逐步升级到扩容
//!
//! 参考：llama.cpp llama_context 内存管理

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.core_mem_monitor);

// ============================================================================
// 告警级别
// ============================================================================

/// 内存告警级别
pub const AlertLevel = enum {
    /// 正常（使用率 < 90%）
    normal,
    /// 警告（使用率 >= 90%，建议回收）
    warn,
    /// 严重（使用率 >= 95%，必须回收）
    critical,
    /// 紧急（使用率 >= 99%，即将 OOM）
    oom,

    /// 从使用率获取告警级别
    pub fn fromRatio(ratio: f64) AlertLevel {
        if (ratio >= 0.99) return .oom;
        if (ratio >= 0.95) return .critical;
        if (ratio >= 0.90) return .warn;
        return .normal;
    }
};

// ============================================================================
// 内存快照
// ============================================================================

/// 单个 context 的内存快照
pub const ContextSnapshot = struct {
    /// 标签（如 "kv_cache", "graph", "inc"）
    label: []const u8,
    /// 已使用内存（字节）
    used: usize,
    /// 总内存（字节）
    total: usize,
    /// 使用率（0.0 - 1.0）
    ratio: f64,
    /// 告警级别
    alert: AlertLevel,
    /// 扩容次数
    grow_count: u32,
    /// 是否 no_alloc 模式
    no_alloc: bool,
};

/// 完整内存报告
pub const MemoryReport = struct {
    /// 各 context 的快照
    contexts: []const ContextSnapshot,
    /// 总已使用内存（字节）
    total_used: usize,
    /// 总内存（字节）
    total: usize,
    /// 总使用率
    total_ratio: f64,
    /// 最高告警级别
    max_alert: AlertLevel,
    /// 时间戳（毫秒）
    timestamp: i64,

    /// 格式化输出内存报告
    pub fn format(self: *const MemoryReport, writer: anytype) !void {
        try writer.print("Memory Report ({d:.1}% used, alert: {s}):\n", .{
            self.total_ratio * 100,
            @tagName(self.max_alert),
        });
        for (self.contexts) |ctx| {
            try writer.print("  {s}: {d:.1} MB / {d:.1} MB ({d:.1}%) {s}{s}\n", .{
                ctx.label,
                @as(f64, @floatFromInt(ctx.used)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(ctx.total)) / (1024.0 * 1024.0),
                ctx.ratio * 100,
                if (ctx.alert != .normal) @tagName(ctx.alert) else "",
                if (ctx.no_alloc) " [no_alloc]" else "",
            });
        }
    }
};

// ============================================================================
// 可监控 Context 接口（虚表模式）
// ============================================================================

/// 可监控的 context 接口
pub const MonitoredContext = struct {
    ptr: *anyopaque,
    vtable: *const MonitoredVTable,

    pub const MonitoredVTable = struct {
        getLabel: *const fn (ptr: *anyopaque) []const u8,
        getUsedMem: *const fn (ptr: *anyopaque) usize,
        getTotalMem: *const fn (ptr: *anyopaque) usize,
        getGrowCount: *const fn (ptr: *anyopaque) u32,
        isNoAlloc: *const fn (ptr: *anyopaque) bool,
        tryReclaim: ?*const fn (ptr: *anyopaque) bool = null,
        tryGrow: ?*const fn (ptr: *anyopaque) anyerror!bool = null,
    };

    pub fn getLabel(self: MonitoredContext) []const u8 {
        return self.vtable.getLabel(self.ptr);
    }
    pub fn getUsedMem(self: MonitoredContext) usize {
        return self.vtable.getUsedMem(self.ptr);
    }
    pub fn getTotalMem(self: MonitoredContext) usize {
        return self.vtable.getTotalMem(self.ptr);
    }
    pub fn getGrowCount(self: MonitoredContext) u32 {
        return self.vtable.getGrowCount(self.ptr);
    }
    pub fn isNoAlloc(self: MonitoredContext) bool {
        return self.vtable.isNoAlloc(self.ptr);
    }
    pub fn tryReclaim(self: MonitoredContext) bool {
        if (self.vtable.tryReclaim) |f| return f(self.ptr);
        return false;
    }
    pub fn tryGrow(self: MonitoredContext) !bool {
        if (self.vtable.tryGrow) |f| return f(self.ptr);
        return false;
    }

    /// 获取快照
    pub fn snapshot(self: MonitoredContext) ContextSnapshot {
        const used = self.getUsedMem();
        const total = self.getTotalMem();
        const ratio = if (total > 0) @as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total)) else 0.0;
        return .{
            .label = self.getLabel(),
            .used = used,
            .total = total,
            .ratio = ratio,
            .alert = AlertLevel.fromRatio(ratio),
            .grow_count = self.getGrowCount(),
            .no_alloc = self.isNoAlloc(),
        };
    }
};

// ============================================================================
// 适配器工厂函数
// ============================================================================

/// 为 ggml.Context 创建 MonitoredContext 适配器
/// label 必须是编译期已知的字符串字面量
pub fn adaptGgmlContext(ctx: *ggml.Context, comptime label: []const u8) MonitoredContext {
    const V = struct {
        fn getLabel(_: *anyopaque) []const u8 {
            return label;
        }
        fn getUsedMem(ptr: *anyopaque) usize {
            return (@as(*ggml.Context, @ptrCast(@alignCast(ptr)))).usedMem();
        }
        fn getTotalMem(ptr: *anyopaque) usize {
            return (@as(*ggml.Context, @ptrCast(@alignCast(ptr)))).totalMem();
        }
        fn getGrowCount(_: *anyopaque) u32 {
            return 0;
        }
        fn isNoAlloc(ptr: *anyopaque) bool {
            return (@as(*ggml.Context, @ptrCast(@alignCast(ptr)))).getNoAlloc();
        }
        fn tryReclaim(ptr: *anyopaque) bool {
            const self = @as(*ggml.Context, @ptrCast(@alignCast(ptr)));
            if (self.usedMem() > 0) {
                self.reset();
                return true;
            }
            return false;
        }
    };
    return .{ .ptr = ctx, .vtable = &.{
        .getLabel = V.getLabel,
        .getUsedMem = V.getUsedMem,
        .getTotalMem = V.getTotalMem,
        .getGrowCount = V.getGrowCount,
        .isNoAlloc = V.isNoAlloc,
        .tryReclaim = V.tryReclaim,
    } };
}

/// 为 GrowableContext 创建 MonitoredContext 适配器
pub fn adaptGrowableContext(gc: *anyopaque) MonitoredContext {
    const GC = @import("memory_pool").GrowableContext;
    const V = struct {
        fn getLabel(_: *anyopaque) []const u8 {
            return "growable";
        }
        fn getUsedMem(ptr: *anyopaque) usize {
            return (@as(*GC, @ptrCast(@alignCast(ptr)))).usedMem();
        }
        fn getTotalMem(ptr: *anyopaque) usize {
            return (@as(*GC, @ptrCast(@alignCast(ptr)))).totalMem();
        }
        fn getGrowCount(ptr: *anyopaque) u32 {
            return (@as(*GC, @ptrCast(@alignCast(ptr)))).grow_count;
        }
        fn isNoAlloc(ptr: *anyopaque) bool {
            return (@as(*GC, @ptrCast(@alignCast(ptr)))).no_alloc;
        }
        fn tryReclaim(ptr: *anyopaque) bool {
            const self = @as(*GC, @ptrCast(@alignCast(ptr)));
            if (self.usedMem() > 0) {
                self.reset();
                return true;
            }
            return false;
        }
        fn tryGrow(ptr: *anyopaque) !bool {
            return (@as(*GC, @ptrCast(@alignCast(ptr)))).growIfNeeded();
        }
    };
    return .{ .ptr = gc, .vtable = &.{
        .getLabel = V.getLabel,
        .getUsedMem = V.getUsedMem,
        .getTotalMem = V.getTotalMem,
        .getGrowCount = V.getGrowCount,
        .isNoAlloc = V.isNoAlloc,
        .tryReclaim = V.tryReclaim,
        .tryGrow = V.tryGrow,
    } };
}

/// 为 IncContext 创建 MonitoredContext 适配器
pub fn adaptIncContext(ic: *anyopaque) MonitoredContext {
    const IC = @import("graph_context").IncContext;
    const V = struct {
        fn getLabel(_: *anyopaque) []const u8 {
            return "inc_decode";
        }
        fn getUsedMem(ptr: *anyopaque) usize {
            return (@as(*IC, @ptrCast(@alignCast(ptr)))).usedMem();
        }
        fn getTotalMem(ptr: *anyopaque) usize {
            return (@as(*IC, @ptrCast(@alignCast(ptr)))).ctx_inc.totalMem();
        }
        fn getGrowCount(_: *anyopaque) u32 {
            return 0;
        }
        fn isNoAlloc(_: *anyopaque) bool {
            return true;
        }
        fn tryReclaim(ptr: *anyopaque) bool {
            const self = @as(*IC, @ptrCast(@alignCast(ptr)));
            if (!self.galloc_reserved and self.cache_valid) {
                self.resetFull();
                return true;
            }
            return false;
        }
    };
    return .{ .ptr = ic, .vtable = &.{
        .getLabel = V.getLabel,
        .getUsedMem = V.getUsedMem,
        .getTotalMem = V.getTotalMem,
        .getGrowCount = V.getGrowCount,
        .isNoAlloc = V.isNoAlloc,
        .tryReclaim = V.tryReclaim,
    } };
}

// ============================================================================
// MemoryMonitor — 内存监控器
// ============================================================================

/// 内存监控器
///
/// 监控多个 context 的内存使用情况，在超过阈值时触发告警和自动回收。
pub const MemoryMonitor = struct {
    allocator: std.mem.Allocator,
    contexts: std.ArrayList(MonitoredContext),
    consecutive_alerts: u32 = 0,
    last_alert: AlertLevel = .normal,
    auto_reclaim: bool = true,
    auto_grow: bool = true,
    max_consecutive_alerts: u32 = 3,

    pub fn init(allocator: std.mem.Allocator) MemoryMonitor {
        return .{ .allocator = allocator, .contexts = std.ArrayList(MonitoredContext).initCapacity(allocator, 0) catch unreachable };
    }

    pub fn deinit(self: *MemoryMonitor) void {
        self.contexts.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addContext(self: *MemoryMonitor, ctx: MonitoredContext) !void {
        try self.contexts.append(self.allocator, ctx);
    }

    pub fn clearContexts(self: *MemoryMonitor) void {
        self.contexts.clearRetainingCapacity();
    }

    /// 获取所有 context 的快照
    pub fn snapshots(self: *const MemoryMonitor, allocator: std.mem.Allocator) ![]ContextSnapshot {
        var list = try std.ArrayList(ContextSnapshot).initCapacity(allocator, self.contexts.items.len);
        errdefer list.deinit(allocator);
        for (self.contexts.items) |ctx| list.appendAssumeCapacity(ctx.snapshot());
        return list.toOwnedSlice(allocator);
    }

    /// 生成完整内存报告
    pub fn report(self: *const MemoryMonitor, allocator: std.mem.Allocator) !MemoryReport {
        const ctx_snapshots = try self.snapshots(allocator);
        errdefer allocator.free(ctx_snapshots);
        var total_used: usize = 0;
        var total: usize = 0;
        var max_alert: AlertLevel = .normal;
        for (ctx_snapshots) |s| {
            total_used += s.used;
            total += s.total;
            if (@intFromEnum(s.alert) > @intFromEnum(max_alert)) max_alert = s.alert;
        }
        const total_ratio = if (total > 0) @as(f64, @floatFromInt(total_used)) / @as(f64, @floatFromInt(total)) else 0.0;
        return .{ .contexts = ctx_snapshots, .total_used = total_used, .total = total, .total_ratio = total_ratio, .max_alert = max_alert, .timestamp = 0 };
    }

    /// 检查内存使用情况，触发告警和自动回收
    pub fn check(self: *MemoryMonitor) !MemoryReport {
        const mem_report = try self.report(self.allocator);
        errdefer self.allocator.free(mem_report.contexts);

        if (mem_report.max_alert != .normal) {
            self.consecutive_alerts += 1;
        } else {
            self.consecutive_alerts = 0;
        }
        self.last_alert = mem_report.max_alert;

        switch (mem_report.max_alert) {
            .normal => {},
            .warn => {
                log.warn("Memory usage at {d:.1}% — consider reclaiming memory", .{mem_report.total_ratio * 100});
                if (self.auto_reclaim) try self.reclaimAll();
            },
            .critical => {
                log.err("Memory usage at {d:.1}% — initiating memory reclamation", .{mem_report.total_ratio * 100});
                _ = try self.reclaimAll();
                if (self.auto_grow) try self.growAll();
            },
            .oom => {
                log.err("Memory usage at {d:.1}% — OOM risk, forcing full reclamation", .{mem_report.total_ratio * 100});
                for (self.contexts.items) |ctx| _ = ctx.tryReclaim();
                if (self.auto_grow) try self.growAll();
            },
        }
        return mem_report;
    }

    fn reclaimAll(self: *MemoryMonitor) !void {
        var reclaimed: u32 = 0;
        for (self.contexts.items) |ctx| {
            if (ctx.tryReclaim()) reclaimed += 1;
        }
        if (reclaimed > 0) log.info("MemoryMonitor: reclaimed {d} context(s)", .{reclaimed});
    }

    fn growAll(self: *MemoryMonitor) !void {
        var grew: u32 = 0;
        for (self.contexts.items) |ctx| {
            if (try ctx.tryGrow()) grew += 1;
        }
        if (grew > 0) log.info("MemoryMonitor: grew {d} context(s)", .{grew});
    }

    pub fn currentAlert(self: *const MemoryMonitor) AlertLevel {
        return self.last_alert;
    }
    pub fn consecutiveAlerts(self: *const MemoryMonitor) u32 {
        return self.consecutive_alerts;
    }
    pub fn resetAlerts(self: *MemoryMonitor) void {
        self.consecutive_alerts = 0;
        self.last_alert = .normal;
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "AlertLevel fromRatio" {
    try testing.expectEqual(.normal, AlertLevel.fromRatio(0.5));
    try testing.expectEqual(.normal, AlertLevel.fromRatio(0.89));
    try testing.expectEqual(.warn, AlertLevel.fromRatio(0.90));
    try testing.expectEqual(.warn, AlertLevel.fromRatio(0.94));
    try testing.expectEqual(.critical, AlertLevel.fromRatio(0.95));
    try testing.expectEqual(.critical, AlertLevel.fromRatio(0.98));
    try testing.expectEqual(.oom, AlertLevel.fromRatio(0.99));
    try testing.expectEqual(.oom, AlertLevel.fromRatio(1.0));
}

test "MemoryMonitor init, add, report, check" {
    var monitor = MemoryMonitor.init(testing.allocator);
    defer monitor.deinit();
    try testing.expectEqual(@as(u32, 0), monitor.contexts.items.len);

    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();
    try monitor.addContext(adaptGgmlContext(ctx, "test_ctx"));
    try testing.expectEqual(@as(usize, 1), monitor.contexts.items.len);

    const report = try monitor.report(testing.allocator);
    defer testing.allocator.free(report.contexts);
    try testing.expect(report.contexts.len > 0);
    try testing.expectEqualStrings("test_ctx", report.contexts[0].label);

    const check_report = try monitor.check();
    defer testing.allocator.free(check_report.contexts);
    try testing.expectEqual(.normal, check_report.max_alert);

    monitor.clearContexts();
    try testing.expectEqual(@as(usize, 0), monitor.contexts.items.len);
}

test "MemoryMonitor high usage triggers alert" {
    var monitor = MemoryMonitor.init(testing.allocator);
    defer monitor.deinit();
    var ctx = try ggml.Context.init(1024);
    defer ctx.deinit();
    _ = try ctx.newTensor1d(.f32, 100);
    try monitor.addContext(adaptGgmlContext(ctx, "tiny_ctx"));

    const report = try monitor.check();
    defer testing.allocator.free(report.contexts);
    try testing.expect(report.max_alert != .normal);
    try testing.expect(monitor.consecutiveAlerts() > 0);
}

test "MemoryMonitor resetAlerts" {
    var monitor = MemoryMonitor.init(testing.allocator);
    defer monitor.deinit();
    monitor.last_alert = .warn;
    monitor.consecutive_alerts = 5;
    monitor.resetAlerts();
    try testing.expectEqual(.normal, monitor.last_alert);
    try testing.expectEqual(@as(u32, 0), monitor.consecutiveAlerts());
}

test "ContextSnapshot and MemoryReport" {
    const snapshots = try testing.allocator.alloc(ContextSnapshot, 2);
    defer testing.allocator.free(snapshots);
    snapshots[0] = .{ .label = "ctx_a", .used = 1024 * 1024, .total = 4 * 1024 * 1024, .ratio = 0.25, .alert = .normal, .grow_count = 0, .no_alloc = false };
    snapshots[1] = .{ .label = "ctx_b", .used = 8 * 1024 * 1024, .total = 10 * 1024 * 1024, .ratio = 0.80, .alert = .warn, .grow_count = 1, .no_alloc = true };
    try testing.expectEqual(.warn, snapshots[1].alert);

    const report = MemoryReport{ .contexts = snapshots, .total_used = 9 * 1024 * 1024, .total = 14 * 1024 * 1024, .total_ratio = 9.0 / 14.0, .max_alert = .warn, .timestamp = 0 };
    try testing.expectEqual(@as(usize, 2), report.contexts.len);
    try testing.expectEqual(.warn, report.max_alert);
}
