//! 增量解码优化上下文
//!
//! 提供增量解码阶段的性能优化：
//! 1. 独立的小型增量上下文（避免每次重建 prompt 图上下文）
//! 2. Gallocr 跨 token 复用（避免重复图分析）
//! 3. 输入张量预分配和复用
//!
//! 注意：图结构本身每 token 仍然重建（ggml 张量创建非常轻量），
//! 但通过复用 Gallocr 和独立的增量上下文，消除了主要的重复开销。

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const log = std.log.scoped(.graph_ctx);

/// 增量解码优化上下文
///
/// 管理增量解码阶段的资源复用。
pub const IncContext = struct {
    /// 分配器
    allocator: std.mem.Allocator,

    /// 增量解码图上下文（独立于 prompt 上下文，更小）
    ctx_inc: *ggml.Context,

    /// 模型参数引用
    params: *const model_if.ModelParams,

    /// 持久化 Gallocr（跨 token 复用）
    galloc: ?*ggml.Gallocr = null,

    /// 是否已初始化（已创建 Gallocr）
    initialized: bool = false,

    /// 上下文内存大小估算
    ctx_size: usize,

    /// 初始化增量上下文
    pub fn init(
        allocator: std.mem.Allocator,
        params: *const model_if.ModelParams,
        ctx_size: usize,
    ) !IncContext {
        const ctx_inc = try ggml.Context.initNoAlloc(ctx_size);

        return IncContext{
            .allocator = allocator,
            .ctx_inc = ctx_inc,
            .params = params,
            .ctx_size = ctx_size,
        };
    }

    /// 释放资源
    pub fn deinit(self: *IncContext) void {
        if (self.galloc) |g| g.free();
        self.ctx_inc.deinit();
        self.* = undefined;
    }

    /// 获取或创建 Gallocr（首次调用时创建）
    pub fn getGallocr(self: *IncContext) !*ggml.Gallocr {
        if (self.galloc) |g| return g;

        const buft = ggml.backendCpuBufferType();
        const g = try ggml.Gallocr.init(buft);
        self.galloc = g;
        self.initialized = true;
        return g;
    }
    /// 为增量解码步骤准备计算环境
    /// 调用者获得 ctx 来创建图，然后可以重用 galloc
    pub fn beginStep(self: *IncContext) !struct {
        ctx: *ggml.Context,
        galloc: *ggml.Gallocr,
    } {
        // 重置增量上下文（释放上次的中间张量）
        self.ctx_inc.reset();

        const g = try self.getGallocr();
        return .{ .ctx = self.ctx_inc, .galloc = g };
    }

    /// 获取上下文的内存使用情况（调试用）
    pub fn usedMem(self: *const IncContext) usize {
        return self.ctx_inc.usedMem();
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "IncContext init and deinit" {
    const params = model_if.ModelParams{
        .n_vocab = 32000,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 32,
        .n_layer = 1,
        .n_ff = 11008,
        .max_seq_len = 2048,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };

    var ic = try IncContext.init(testing.allocator, &params, 64 * 1024 * 1024);
    defer ic.deinit();

    try testing.expect(!ic.initialized);
    try testing.expectEqual(@as(usize, 0), ic.usedMem());
}

test "IncContext gallocr reuse" {
    const params = model_if.ModelParams{
        .n_vocab = 32000,
        .n_embd = 4096,
        .n_head = 32,
        .n_head_dim = 128,
        .n_kv_head = 32,
        .n_layer = 1,
        .n_ff = 11008,
        .max_seq_len = 2048,
        .rope_theta = 1000000.0,
        .rope_dim = 64,
        .norm_eps = 1e-5,
    };

    var ic = try IncContext.init(testing.allocator, &params, 64 * 1024 * 1024);
    defer ic.deinit();

    // First call creates gallocr
    const g1 = try ic.getGallocr();
    try testing.expect(ic.initialized);

    // Second call returns same gallocr
    const g2 = try ic.getGallocr();
    try testing.expectEqual(@intFromPtr(g1), @intFromPtr(g2));
}
