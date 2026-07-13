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
//! Gallocr 预规划（Pre-reservation）：
//! 为防止 ggml_gallocr_needs_realloc 的重复分配循环（deps/guide.md），
//! 在增量解码循环开始前必须调用 reserveGallocrForDecode() 构建 worst-case 图
//! 并调用 ggml_gallocr_reserve 一次。这样后续每次 allocGraph 都复用同一份
//! 缓冲区规划，不再触发重新分配。
//!
//! 关键约束：gallocr 内部使用张量指针作为哈希键，因此 pre-reservation 后
//! **不得**调用 resetFull() 销毁上下文。如需回收内存，必须同时重建 gallocr
//! 并重新执行预规划。
//!
//! 参考：llama.cpp 的 llm_graph_input_i 模式 — 缓存输入张量指针，避免重复创建

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const log = std.log.scoped(.core_graph_ctx);

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

    /// Gallocr 是否已完成预规划（reserve）
    /// 预规划后不能调用 resetFull()，否则张量指针失效
    galloc_reserved: bool = false,

    /// 上下文内存大小
    ctx_size: usize,

    /// ---- 输入张量缓存 ----
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

    /// 使用 worst-case 计算图预分配 Gallocr 缓冲区。
    /// 必须在所有 beginStep() 调用之前调用，且只需调用一次。
    ///
    /// 预规划后，后续 decode 循环中的 allocGraph 将不再触发重新分配，
    /// 从而实现"一次规划，反复执行"的高效内存管理。
    ///
    /// **重要**：调用后不得调用 resetFull()，否则 gallocr 内部哈希表
    /// 中缓存的张量指针全部失效，导致每次 allocGraph 都触发重新分配。
    pub fn reserveGallocr(self: *IncContext, max_graph: *ggml.CGraph) !void {
        const g = try self.getGallocr();
        if (!g.reserve(max_graph)) {
            log.err("Gallocr reserve failed — worst-case graph too large for buffer", .{});
            return error.GallocrReserveFailed;
        }
        self.galloc_reserved = true;
        const buf_size = g.getBufferSize(0);
        log.info("IncContext: gallocr reserved with {d:.1} MB compute buffer", .{
            @as(f64, @floatFromInt(buf_size)) / (1024.0 * 1024.0),
        });
    }

    /// 检查是否需要内存回收
    fn needsRecycle(self: *const IncContext) bool {
        const used = self.ctx_inc.usedMem();
        const total = self.ctx_inc.totalMem();
        if (total == 0) return false;
        const ratio = @as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total));
        return ratio >= RECYCLE_THRESHOLD;
    }

    /// 完整重置（回收所有上下文内存和 Gallocr 内存）
    ///
    /// 新语义（TASK_MEM.md）：
    /// 1. 释放并重新创建 Gallocr（回收计算图内存）
    /// 2. 重置 ggml_context（回收张量元数据）
    /// 3. 设置 is_dirty 标记，强制下次 beginStep 重新构建图节点
    ///
    /// **警告**：如果 gallocr 已完成预规划（galloc_reserved == true），
    /// 调用此方法会使 gallocr 的预规划失效。调用者必须在 reset 后
    /// 重新调用 reserveGallocr() 进行预规划。
    pub fn resetFull(self: *IncContext) void {
        // 1. 释放并重新创建 Gallocr
        if (self.galloc) |g| {
            g.free();
            self.galloc = null;
        }
        self.galloc_ready = false;
        if (self.galloc_reserved) {
            log.warn("IncContext: resetFull with gallocr_reserved=true — gallocr reservation invalidated", .{});
            self.galloc_reserved = false;
        }

        // 2. 重置 ggml_context
        self.ctx_inc.reset();

        // 3. 清除缓存，强制下次 beginStep 重新构建
        self.cached_input = null;
        self.cache_valid = false;
        self.steps_since_reset = 0;
        log.debug("IncContext: full reset (ctx + gallocr recycled)", .{});
    }

    /// 为增量解码步骤准备计算环境
    ///
    /// 复用策略：
    /// - 输入张量：预分配一次，跨步复用
    /// - CGraph：每步新建（极轻量，避免 ggml_graph_reset 梯度检查）
    /// - Gallocr：跨步复用（避免重复图分析）
    /// - 自动检测内存压力：超过阈值时触发 full reset
    ///
    /// 如果 gallocr 已预规划，不会触发 resetFull（因为会破坏 gallocr 哈希表）。
    /// 上下文溢出会导致后续 allocGraph 失败，调用者应负责在溢出前重建 gallocr。
    pub fn beginStep(self: *IncContext) !DecodeStep {
        const g = try self.getGallocr();

        // 检查是否需要内存回收
        // 注意：如果 gallocr 已预规划，跳过回收检查（reset 会破坏哈希表）
        if (!self.galloc_reserved and self.cache_valid and self.needsRecycle()) {
            self.resetFull();
        }

        // 如果 gallocr 已预规划但上下文接近满，发出警告
        if (self.galloc_reserved and self.cache_valid and self.needsRecycle()) {
            log.warn("IncContext: context {d:.0}% full but gallocr_reserved — will need re-reservation soon", .{
                self.memRatio() * 100,
            });
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
    try testing.expect(!ic.galloc_reserved);
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

test "IncContext resetFull clears gallocr_reserved" {
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

    // Manually mark as reserved
    ic.galloc_reserved = true;
    try testing.expect(ic.galloc_reserved);

    ic.resetFull();
    try testing.expect(!ic.galloc_reserved);
}

test "IncContext resetFull frees and recreates gallocr" {
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

    // Create gallocr
    const g1 = try ic.getGallocr();
    _ = g1;
    try testing.expect(ic.galloc_ready);
    try testing.expect(ic.galloc != null);

    // resetFull should free gallocr and set it to null
    ic.resetFull();
    try testing.expect(!ic.galloc_ready);
    try testing.expect(ic.galloc == null);

    // After reset, getGallocr creates a new one
    const g2 = try ic.getGallocr();
    try testing.expect(ic.galloc_ready);
    try testing.expect(ic.galloc != null);
    try testing.expect(g2 != null);
}
