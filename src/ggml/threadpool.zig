//! ggml_threadpool 封装
//!
//! 提供 ggml_threadpool 的类型安全 Zig 封装，用于持久化线程池，
//! 避免每 token 创建/销毁线程的开销。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;

const log = std.log.scoped(.ggml);

// ============================================================================
// 线程池封装
// ============================================================================

/// ggml_threadpool 的不透明指针封装
pub const ThreadPool = opaque {
    /// 初始化线程池
    pub fn init(n_threads: i32) !*ThreadPool {
        var c_params = c.ggml_threadpool_params_default(n_threads);
        const tp = c.ggml_threadpool_new(&c_params);
        if (tp == null) {
            log.err("Failed to create thread pool with {d} threads", .{n_threads});
            return error.ThreadPoolInitFailed;
        }
        log.debug("Thread pool created with {d} threads", .{n_threads});
        return @as(*ThreadPool, @ptrCast(tp));
    }

    /// 释放线程池
    pub fn deinit(self: *ThreadPool) void {
        c.ggml_threadpool_free(@as(*c.struct_ggml_threadpool, @ptrCast(self)));
    }

    /// 获取线程数
    pub fn nThreads(self: *ThreadPool) i32 {
        return c.ggml_threadpool_get_n_threads(@as(*c.struct_ggml_threadpool, @ptrCast(self)));
    }

    /// 暂停线程池
    pub fn pause(self: *ThreadPool) void {
        c.ggml_threadpool_pause(@as(*c.struct_ggml_threadpool, @ptrCast(self)));
    }

    /// 恢复线程池
    pub fn wake(self: *ThreadPool) void {
        c.ggml_threadpool_resume(@as(*c.struct_ggml_threadpool, @ptrCast(self)));
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "ThreadPool basic" {
    const tp = try ThreadPool.init(2);
    defer tp.deinit();
    try testing.expectEqual(@as(i32, 2), tp.nThreads());
}
