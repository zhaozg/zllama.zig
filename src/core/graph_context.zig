//! 增量解码优化上下文
//!
//! 提供增量解码阶段的性能优化：
//! 1. 独立的小型增量上下文（避免每次重建 prompt 图上下文）
//! 2. Gallocr 跨 token 复用（避免重复图分析）
//! 3. 输入张量预分配和复用（参考 llama.cpp llm_graph_input_i 模式）
//! 4. 上下文内存自动回收（超过 80% 阈值时触发 full reset）
//!
//! 图结构复用策略：
//! - 输入张量（token）预分配一次，跨步复用（避免 setNoAlloc + newTensor1d）
//! - Gallocr 跨步复用（避免重复图分析，~1-5ms/次节省）
//! - CGraph 每步新建（极轻量，仅 ~100 字节 struct 分配）
//! - 中间张量随 buildGraph 自然累积，超过阈值自动回收
//!
//! 参考：llama.cpp 的 llm_graph_input_i 模式 — 缓存输入张量指针，避免重复创建

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const log = std.log.scoped(.graph_ctx);

/// 内存回收阈值：上下文使用率达到此比例时触发 full reset
const RECYCLE_THRESHOLD: f64 = 0.80;

/// 单步计算环境
/// 包含预分配的输入张量，避免每 token 重复创建
pub const DecodeStep = struct {
    /// 增量上下文
    ctx: *ggml.Context,
    /// Gallocr（跨步复用）
    galloc: *ggml.Gallocr,
    /// 预分配的输入 token 张量 [1] i32（跨步复用）
    input_token: *ggml.Tensor,
    /// 新建的 CGraph 对象（每步新建，极轻量）
    graph: *ggml.CGraph,

    /// 设置输入 token 数据
    pub fn setToken(self: *const DecodeStep, token_id: i32) void {
        const data = self.input_token.dataBytes();
        const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..1];
        slice[0] = token_id;
    }
};

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

    /// 是否已初始化 Gallocr
    galloc_ready: bool = false,

    /// 上下文内存大小
    ctx_size: usize,

    // ---- 输入张量缓存 ----
    /// 缓存的输入 token 张量（ctx.reset() 后失效）
    cached_input: ?*ggml.Tensor = null,
    /// 缓存是否有效（false 表示需要重建缓存）
    cache_valid: bool = false,
    /// 自上次 full reset 以来的步数
    steps_since_reset: u32 = 0,

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
        self.galloc_ready = true;
        return g;
    }

    /// 检查是否需要内存回收
    fn needsRecycle(self: *const IncContext) bool {
        const used = self.ctx_inc.usedMem();
        const total = self.ctx_inc.totalMem();
        if (total == 0) return false;
        const ratio = @as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total));
        return ratio >= RECYCLE_THRESHOLD;
    }

    /// 完整重置（回收所有上下文内存）
    /// 当上下文接近满时自动调用，也支持手动调用
    pub fn resetFull(self: *IncContext) void {
        self.ctx_inc.reset();
        self.cached_input = null;
        self.cache_valid = false;
        self.steps_since_reset = 0;
        log.debug("IncContext: full reset (ctx recycled, {d}%)", .{@as(u32, @intFromFloat(self.memRatio() * 100))});
    }

    /// 为增量解码步骤准备计算环境
    ///
    /// 复用策略：
    /// - 输入张量：预分配一次，跨步复用
    /// - CGraph：每步新建（极轻量，避免 ggml_graph_reset 梯度检查）
    /// - Gallocr：跨步复用（避免重复图分析）
    /// - 自动检测内存压力：超过阈值时触发 full reset
    pub fn beginStep(self: *IncContext) !DecodeStep {
        const g = try self.getGallocr();

        // 检查是否需要内存回收
        if (self.cache_valid and self.needsRecycle()) {
            self.resetFull();
        }

        if (!self.cache_valid) {
            // 首次或 full reset 后：预分配输入张量
            self.ctx_inc.setNoAlloc(false);
            self.cached_input = try self.ctx_inc.newTensor1d(.i32, 1);
            self.ctx_inc.setNoAlloc(true);
            self.cache_valid = true;
            self.steps_since_reset = 1;
            log.debug("IncContext: input tensor cached (step 1)", .{});
        } else {
            self.steps_since_reset += 1;
        }

        // 每步创建新 graph（极轻量，~100 字节 struct 分配）
        const graph = try ggml.CGraph.initReserved(self.ctx_inc, 16384);

        return DecodeStep{
            .ctx = self.ctx_inc,
            .galloc = g,
            .input_token = self.cached_input.?,
            .graph = graph,
        };
    }

    /// 获取上下文的内存使用情况（调试用）
    pub fn usedMem(self: *const IncContext) usize {
        return self.ctx_inc.usedMem();
    }

    /// 获取上下文的内存使用率（0.0 - 1.0）
    pub fn memRatio(self: *const IncContext) f64 {
        const used = self.ctx_inc.usedMem();
        const total = self.ctx_inc.totalMem();
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total));
    }

    /// 获取自上次 full reset 以来的步数
    pub fn stepsSinceReset(self: *const IncContext) u32 {
        return self.steps_since_reset;
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

    try testing.expect(!ic.galloc_ready);
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
    try testing.expect(ic.galloc_ready);

    // Second call returns same gallocr
    const g2 = try ic.getGallocr();
    try testing.expectEqual(@intFromPtr(g1), @intFromPtr(g2));
}

test "IncContext beginStep - input tensor reuse" {
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

    // First step: creates cache
    const s1 = try ic.beginStep();
    try testing.expect(ic.cache_valid);
    try testing.expectEqual(@as(u32, 1), ic.stepsSinceReset());

    // Set token and verify
    s1.setToken(42);
    const d1 = s1.input_token.dataBytes();
    const sl1 = @as([*]i32, @ptrCast(@alignCast(d1.ptr)))[0..1];
    try testing.expectEqual(@as(i32, 42), sl1[0]);

    // Second step: reuses input tensor, creates new graph
    const s2 = try ic.beginStep();
    try testing.expect(ic.cache_valid);
    try testing.expectEqual(@as(u32, 2), ic.stepsSinceReset());

    // Same input tensor pointer (reused)
    try testing.expectEqual(@intFromPtr(s1.input_token), @intFromPtr(s2.input_token));
    // Different graph pointer (fresh each step)
    try testing.expect(@intFromPtr(s1.graph) != @intFromPtr(s2.graph));

    // Token data independent
    s2.setToken(99);
    const d2 = s2.input_token.dataBytes();
    const sl2 = @as([*]i32, @ptrCast(@alignCast(d2.ptr)))[0..1];
    try testing.expectEqual(@as(i32, 99), sl2[0]);
}

test "IncContext resetFull" {
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

    _ = try ic.beginStep();
    try testing.expect(ic.cache_valid);

    ic.resetFull();
    try testing.expect(!ic.cache_valid);
    try testing.expectEqual(@as(u32, 0), ic.stepsSinceReset());

    // After reset, beginStep creates fresh cache
    _ = try ic.beginStep();
    try testing.expect(ic.cache_valid);
    try testing.expectEqual(@as(u32, 1), ic.stepsSinceReset());
}
