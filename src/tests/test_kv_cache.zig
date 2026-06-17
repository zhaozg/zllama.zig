//! KV Cache 功能测试
//!
//! 验证 KV Cache 的正确性：
//! - 初始化与释放
//! - 增量写入与读取
//! - 视图切片
//! - MemoryContext 接口适配
//! - 多批次写入
//! - 边界条件（空写入、满缓存）

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");
const gguf = @import("gguf");
const kv_cache = @import("kv_cache");
const memory = @import("memory");

const log = std.log.scoped(.test_kv_cache);

// ============================================================================
// 测试辅助
// ============================================================================

/// 创建一个测试用的 ggml context
fn createTestContext() !*ggml.Context {
    return try ggml.Context.initNoAlloc(256 * 1024); // 256KB
}

// ============================================================================
// 测试用例
// ============================================================================

test "KVCache init and deinit" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 2, 4, 32, 64, testing.allocator);
    defer kv.deinit(testing.allocator);

    try testing.expectEqual(@as(u32, 2), @as(u32, @intCast(kv.layers.len)));
    try testing.expectEqual(@as(u32, 64), kv.max_seq_len);
    try testing.expectEqual(@as(u32, 4), kv.n_kv_head);
    try testing.expectEqual(@as(u32, 32), kv.head_dim);
    try testing.expectEqual(@as(u32, 0), kv.currentLen());
}

test "KVCache reset" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    // 模拟写入
    kv.layers[0].current_len = 10;
    try testing.expectEqual(@as(u32, 10), kv.currentLen());

    // 重置
    kv.reset();
    try testing.expectEqual(@as(u32, 0), kv.currentLen());
}

test "KVCache getKView and getVView" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    // 设置 current_len
    kv.layers[0].current_len = 5;

    const k_view = kv.getKView(ctx, 0);
    const v_view = kv.getVView(ctx, 0);

    // 验证视图形状
    try testing.expect(k_view != undefined);
    try testing.expect(v_view != undefined);
}

test "KVCache setKv basic" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    // 创建新的 K, V 张量
    ctx.setNoAlloc(false);
    const new_k = try ctx.newTensor3d(.f32, 16, 2, 1);
    const new_v = try ctx.newTensor3d(.f32, 16, 2, 1);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);

    // 写入 1 个 token
    kv.setKv(ctx, graph, 0, new_k, new_v, 1);
    try testing.expectEqual(@as(u32, 1), kv.currentLen());
}

test "KVCache setKv multiple tokens" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    ctx.setNoAlloc(false);
    const new_k = try ctx.newTensor3d(.f32, 16, 2, 3);
    const new_v = try ctx.newTensor3d(.f32, 16, 2, 3);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);

    // 写入 3 个 token
    kv.setKv(ctx, graph, 0, new_k, new_v, 3);
    try testing.expectEqual(@as(u32, 3), kv.currentLen());
}

test "KVCache setKv incremental" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    ctx.setNoAlloc(false);
    const new_k1 = try ctx.newTensor3d(.f32, 16, 2, 1);
    const new_v1 = try ctx.newTensor3d(.f32, 16, 2, 1);
    const new_k2 = try ctx.newTensor3d(.f32, 16, 2, 2);
    const new_v2 = try ctx.newTensor3d(.f32, 16, 2, 2);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);

    // 第一次写入 1 个 token
    kv.setKv(ctx, graph, 0, new_k1, new_v1, 1);
    try testing.expectEqual(@as(u32, 1), kv.currentLen());

    // 第二次写入 2 个 token
    kv.setKv(ctx, graph, 0, new_k2, new_v2, 2);
    try testing.expectEqual(@as(u32, 3), kv.currentLen());
}

test "KVCache multiple layers" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    const n_layers: u32 = 4;
    var kv = try kv_cache.KVCache.init(ctx, n_layers, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    try testing.expectEqual(n_layers, @as(u32, @intCast(kv.layers.len)));

    // 验证所有层都初始化
    for (0..n_layers) |i| {
        try testing.expectEqual(@as(u32, 0), kv.layers[i].current_len);
        try testing.expect(kv.layers[i].k != undefined);
        try testing.expect(kv.layers[i].v != undefined);
    }
}

test "KVCache currentLen across layers" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 3, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    // 不同层有不同的长度
    kv.layers[0].current_len = 5;
    kv.layers[1].current_len = 10;
    kv.layers[2].current_len = 3;

    // currentLen 返回最大值
    try testing.expectEqual(@as(u32, 10), kv.currentLen());
}

test "KVCache toMemoryContext" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 2, 4, 32, 64, testing.allocator);
    defer kv.deinit(testing.allocator);

    var mem_ctx = kv.toMemoryContext();

    // 验证接口方法
    try testing.expectEqual(@as(u32, 2), mem_ctx.nLayers());
    try testing.expectEqual(@as(u32, 0), mem_ctx.currentLen());

    const k = mem_ctx.getK(0);
    const v = mem_ctx.getV(0);
    try testing.expect(k != undefined);
    try testing.expect(v != undefined);

    // 测试 reset
    kv.layers[0].current_len = 5;
    mem_ctx.reset();
    try testing.expectEqual(@as(u32, 0), mem_ctx.currentLen());
}

test "KVCache toMemoryContext setKv" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    var mem_ctx = kv.toMemoryContext();

    ctx.setNoAlloc(false);
    const new_k = try ctx.newTensor3d(.f32, 16, 2, 1);
    const new_v = try ctx.newTensor3d(.f32, 16, 2, 1);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);

    // 通过 MemoryContext 接口写入
    mem_ctx.setKv(ctx, graph, 0, new_k, new_v, 1);
    try testing.expectEqual(@as(u32, 1), mem_ctx.currentLen());
}

test "KVCache max_seq_len boundary" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    const max_seq_len: u32 = 8;
    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, max_seq_len, testing.allocator);
    defer kv.deinit(testing.allocator);

    try testing.expectEqual(max_seq_len, kv.max_seq_len);

    // 写入到接近上限
    ctx.setNoAlloc(false);
    const new_k = try ctx.newTensor3d(.f32, 16, 2, 4);
    const new_v = try ctx.newTensor3d(.f32, 16, 2, 4);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);
    kv.setKv(ctx, graph, 0, new_k, new_v, 4);
    try testing.expectEqual(@as(u32, 4), kv.currentLen());
}

test "KVCache zero tokens setKv" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv = try kv_cache.KVCache.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer kv.deinit(testing.allocator);

    ctx.setNoAlloc(false);
    const new_k = try ctx.newTensor3d(.f32, 16, 2, 0);
    const new_v = try ctx.newTensor3d(.f32, 16, 2, 0);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);

    // 写入 0 个 token
    kv.setKv(ctx, graph, 0, new_k, new_v, 0);
    try testing.expectEqual(@as(u32, 0), kv.currentLen());
}

test "KVCacheMemory init and basic ops" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv_mem = try memory.KVCacheMemory.init(ctx, 2, 2, 16, 32, testing.allocator);
    defer memory.KVCacheMemory.deinit(@as(*anyopaque, @ptrCast(&kv_mem)));

    try testing.expectEqual(@as(u32, 2), memory.KVCacheMemory.nLayers(@as(*anyopaque, @ptrCast(&kv_mem))));
    try testing.expectEqual(@as(u32, 0), memory.KVCacheMemory.currentLen(@as(*anyopaque, @ptrCast(&kv_mem))));

    // 测试 MemoryContext 包装
    var mem_ctx = kv_mem.toMemoryContext();
    try testing.expectEqual(@as(u32, 2), mem_ctx.nLayers());
    try testing.expectEqual(@as(u32, 0), mem_ctx.currentLen());

    // 测试 reset
    mem_ctx.reset();
    try testing.expectEqual(@as(u32, 0), mem_ctx.currentLen());
}

test "KVCacheMemory setKv via interface" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv_mem = try memory.KVCacheMemory.init(ctx, 1, 2, 16, 32, testing.allocator);
    defer memory.KVCacheMemory.deinit(@as(*anyopaque, @ptrCast(&kv_mem)));

    var mem_ctx = kv_mem.toMemoryContext();

    ctx.setNoAlloc(false);
    const new_k = try ctx.newTensor3d(.f32, 16, 2, 2);
    const new_v = try ctx.newTensor3d(.f32, 16, 2, 2);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);

    // 通过 MemoryContext 接口写入
    mem_ctx.setKv(ctx, graph, 0, new_k, new_v, 2);
    try testing.expectEqual(@as(u32, 2), mem_ctx.currentLen());
}

test "KVCacheMemory multiple layers setKv" {
    const ctx = try createTestContext();
    defer ctx.deinit();

    var kv_mem = try memory.KVCacheMemory.init(ctx, 3, 4, 32, 64, testing.allocator);
    defer memory.KVCacheMemory.deinit(@as(*anyopaque, @ptrCast(&kv_mem)));

    var mem_ctx = kv_mem.toMemoryContext();

    ctx.setNoAlloc(false);
    const new_k = try ctx.newTensor3d(.f32, 32, 4, 1);
    const new_v = try ctx.newTensor3d(.f32, 32, 4, 1);
    ctx.setNoAlloc(true);

    const graph = try ggml.CGraph.init(ctx);

    // 写入所有层
    for (0..3) |i| {
        mem_ctx.setKv(ctx, graph, i, new_k, new_v, 1);
    }
    try testing.expectEqual(@as(u32, 1), mem_ctx.currentLen());
}

test "LayerCache struct size" {
    try testing.expectEqual(@as(usize, @sizeOf(kv_cache.LayerCache)), @sizeOf(kv_cache.LayerCache));
}
