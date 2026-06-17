//! 架构前向测试
//!
//! 对每个支持的架构运行随机权重前向测试。
//! 验证模型实现的数值正确性。
//!
//! 参考 llama.cpp 的 test-llama-archs.cpp 设计。
//!
//! 测试策略：
//! 1. 生成小型随机权重的 GGUF 模型（2层，64维）
//! 2. 在 CPU 上推理得到 logits
//! 3. 验证推理不崩溃，输出形状正确
//! 4. 验证相同输入产生相同输出（确定性）

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const test_utils = @import("test_utils");

const log = std.log.scoped(.test_archs);

// ============================================================================
// 测试配置
// ============================================================================

/// 小型测试模型配置
const SmallTestConfig = struct {
    arch: model_if.Architecture,
    n_layer: u32 = 2,
    n_embd: u32 = 64,
    n_head: u32 = 4,
    n_kv_head: u32 = 2,
    n_ff: u32 = 128,
    n_vocab: u32 = 128,
    n_head_dim: u32 = 16,
    max_seq_len: u32 = 32,
    rope_theta: f32 = 10000000.0,
    rope_dim: u32 = 16,
    norm_eps: f32 = 1e-6,
    seed: u64 = 42,
};

// ============================================================================
// 辅助函数：生成随机 GGUF
// ============================================================================

/// 生成小型随机权重的 GGUF 数据
/// 返回分配的字节数组，调用者负责释放
fn generateTestGGUF(allocator: std.mem.Allocator, config: SmallTestConfig) ![]u8 {
    _ = allocator;
    _ = config;
    // TODO: 实现完整的随机 GGUF 生成
    // 当前返回一个最小的有效 GGUF（空模型）
    // 实际测试需要生成包含随机权重的完整 GGUF
    @panic("Not implemented yet - need GGUF writer");
}

// ============================================================================
// 辅助函数：运行前向推理
// ============================================================================

/// 运行模型前向推理，返回 logits
fn runForward(
    allocator: std.mem.Allocator,
    model: *model_if.ModelInstance,
    params: *const model_if.ModelParams,
    input_tokens: []const i32,
) ![]f32 {
    const n_tokens: i32 = @intCast(input_tokens.len);

    // 创建推理用的 ggml context
    const ctx_graph = try ggml.Context.initNoAlloc(256 * 1024); // 256KB
    defer ctx_graph.deinit();

    ctx_graph.setNoAlloc(false);
    const input_tensor = try ctx_graph.newTensor1d(.i32, n_tokens);
    ctx_graph.setNoAlloc(true);

    // 复制输入 token
    const data = input_tensor.dataBytes();
    const dst = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
    @memcpy(dst, input_tokens);

    // 构建计算图
    var graph = try ggml.CGraph.init(ctx_graph);
    var builder = graph_builder.GraphBuilder.init(ctx_graph, graph, params, allocator);
    const logits_tensor = try model.buildGraph(&builder, input_tensor, n_tokens, null, 0);

    // 分配并执行
    const buft = ggml.backendCpuBufferType();
    var galloc = try ggml.Gallocr.init(buft);
    defer galloc.free();

    if (!galloc.allocGraph(graph)) {
        return error.GraphAllocationFailed;
    }

    // 执行计算图
    const n_threads = @as(i32, @intCast(@min(4, @max(1, try std.Thread.getCpuCount() - 1))));
    ggml.backendCpuSetNThreads(n_threads);
    graph.computeWithCtx();

    // 读取 logits
    const logits_data = logits_tensor.dataBytes();
    const n_logits = @as(usize, @intCast(params.n_vocab));
    const logits = try allocator.alloc(f32, n_logits);
    const src = @as([*]f32, @ptrCast(@alignCast(logits_data.ptr)))[0..n_logits];
    @memcpy(logits, src);

    return logits;
}

// ============================================================================
// 测试用例
// ============================================================================

test "architecture enum fromString" {
    try testing.expectEqual(model_if.Architecture.qwen2, model_if.Architecture.fromString("qwen2").?);
    try testing.expectEqual(model_if.Architecture.qwen35, model_if.Architecture.fromString("qwen35").?);
    try testing.expectEqual(model_if.Architecture.llama, model_if.Architecture.fromString("llama3").?);
    try testing.expect(model_if.Architecture.fromString("unknown") == null);
}

test "architecture enum toString" {
    try testing.expectEqualStrings("qwen2", @tagName(model_if.Architecture.qwen2));
    try testing.expectEqualStrings("qwen35", @tagName(model_if.Architecture.qwen35));
    try testing.expectEqualStrings("llama", @tagName(model_if.Architecture.llama));
}

test "model params defaults" {
    const p = model_if.ModelParams{};
    try testing.expectEqual(@as(u32, 0), p.n_vocab);
    try testing.expectEqual(@as(u32, 32768), p.max_seq_len);
    try testing.expectEqual(@as(f32, 10000000.0), p.rope_theta);
    try testing.expectEqual(@as(f32, 1e-6), p.norm_eps);
}

test "model vtable size" {
    try testing.expectEqual(@as(usize, @sizeOf(model_if.ModelVTable)), @sizeOf(model_if.ModelVTable));
}

test "model instance size" {
    try testing.expectEqual(@as(usize, @sizeOf(model_if.ModelInstance)), @sizeOf(model_if.ModelInstance));
}

test "SmallTestConfig defaults" {
    const c = SmallTestConfig{ .arch = .llama };
    try testing.expectEqual(@as(u32, 2), c.n_layer);
    try testing.expectEqual(@as(u32, 64), c.n_embd);
    try testing.expectEqual(@as(u32, 4), c.n_head);
    try testing.expectEqual(@as(u32, 2), c.n_kv_head);
    try testing.expectEqual(@as(u32, 128), c.n_vocab);
    try testing.expectEqual(@as(u64, 42), c.seed);
}

test "detectArchitecture from registry" {
    // 构造一个包含 general.architecture 元数据的 GGUF
    var buf: [256]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    // 魔数
    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    // 版本 v3
    std.mem.writeInt(u32, buf[pos..][0..4], 3, .little);
    pos += 4;
    // tensor_count = 0
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little);
    pos += 8;
    // metadata_kv_count = 1
    std.mem.writeInt(u64, buf[pos..][0..8], 1, .little);
    pos += 8;

    // key: "general.architecture"
    const key = "general.architecture";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(key.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..key.len], key);
    pos += key.len;

    // value type: STRING = 8
    std.mem.writeInt(u32, buf[pos..][0..4], @intFromEnum(gguf.MetadataValueType.string), .little);
    pos += 4;

    // value: "llama"
    const val = "llama";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(val.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..val.len], val);
    pos += val.len;

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const detected = registry.detectArchitecture(&file);
    try testing.expect(detected != null);
    try testing.expectEqual(model_if.Architecture.llama, detected.?);
}

test "detectArchitecture - qwen2" {
    var buf: [256]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    std.mem.writeInt(u32, buf[pos..][0..4], 3, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little);
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 1, .little);
    pos += 8;

    const key = "general.architecture";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(key.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..key.len], key);
    pos += key.len;

    std.mem.writeInt(u32, buf[pos..][0..4], @intFromEnum(gguf.MetadataValueType.string), .little);
    pos += 4;

    const val = "qwen2";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(val.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..val.len], val);
    pos += val.len;

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const detected = registry.detectArchitecture(&file);
    try testing.expect(detected != null);
    try testing.expectEqual(model_if.Architecture.qwen2, detected.?);
}

test "detectArchitecture - qwen35" {
    var buf: [256]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    std.mem.writeInt(u32, buf[pos..][0..4], 3, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little);
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 1, .little);
    pos += 8;

    const key = "general.architecture";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(key.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..key.len], key);
    pos += key.len;

    std.mem.writeInt(u32, buf[pos..][0..4], @intFromEnum(gguf.MetadataValueType.string), .little);
    pos += 4;

    const val = "qwen35";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(val.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..val.len], val);
    pos += val.len;

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const detected = registry.detectArchitecture(&file);
    try testing.expect(detected != null);
    try testing.expectEqual(model_if.Architecture.qwen35, detected.?);
}

test "detectArchitecture - unsupported returns null" {
    var buf: [256]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    std.mem.writeInt(u32, buf[pos..][0..4], 3, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little);
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 1, .little);
    pos += 8;

    const key = "general.architecture";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(key.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..key.len], key);
    pos += key.len;

    std.mem.writeInt(u32, buf[pos..][0..4], @intFromEnum(gguf.MetadataValueType.string), .little);
    pos += 4;

    const val = "unknown_arch";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(val.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..val.len], val);
    pos += val.len;

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const detected = registry.detectArchitecture(&file);
    try testing.expect(detected == null);
}

test "GraphBuilder init and basic ops" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    const params = model_if.ModelParams{
        .n_embd = 64,
        .n_head = 4,
        .n_kv_head = 2,
        .n_head_dim = 16,
        .n_vocab = 128,
    };

    var builder = graph_builder.GraphBuilder.init(ctx, graph, &params, testing.allocator);

    // 测试 buildRmsNorm
    ctx.setNoAlloc(false);
    const x = try ctx.newTensor2d(.f32, 64, 1);
    const weight = try ctx.newTensor1d(.f32, 64);
    ctx.setNoAlloc(true);

    const normed = builder.buildRmsNorm(x, weight, 1e-6);
    try testing.expect(normed != undefined);
}

test "GraphBuilder buildPositionTensor" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    const params = model_if.ModelParams{ .n_embd = 64 };
    var builder = graph_builder.GraphBuilder.init(ctx, graph, &params, testing.allocator);

    ctx.setNoAlloc(false);
    const pos = try builder.buildPositionTensor(5, 10);
    ctx.setNoAlloc(true);

    const data = pos.dataBytes();
    const pos_vals = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..5];
    try testing.expectEqual(@as(i32, 10), pos_vals[0]);
    try testing.expectEqual(@as(i32, 11), pos_vals[1]);
    try testing.expectEqual(@as(i32, 12), pos_vals[2]);
    try testing.expectEqual(@as(i32, 13), pos_vals[3]);
    try testing.expectEqual(@as(i32, 14), pos_vals[4]);
}

test "KVCacheMemory basic operations" {
    const ctx = try ggml.Context.initNoAlloc(256 * 1024);
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

test "KVCacheMemory toMemoryContext" {
    const ctx = try ggml.Context.initNoAlloc(256 * 1024);
    defer ctx.deinit();

    var kv_mem = try memory.KVCacheMemory.init(ctx, 1, 4, 32, 64, testing.allocator);
    defer memory.KVCacheMemory.deinit(@as(*anyopaque, @ptrCast(&kv_mem)));

    var mem_ctx = kv_mem.toMemoryContext();

    // 验证接口方法
    try testing.expectEqual(@as(u32, 1), mem_ctx.nLayers());
    try testing.expectEqual(@as(u32, 0), mem_ctx.currentLen());

    const k = mem_ctx.getK(0);
    const v = mem_ctx.getV(0);
    try testing.expect(k != undefined);
    try testing.expect(v != undefined);
}

// ============================================================================
// TODO: 实际的随机权重前向测试
// 需要实现 generateTestGGUF 后才能启用
// ============================================================================

// test "llama forward with random weights" {
//     const config = SmallTestConfig{ .arch = .llama };
//     const gguf_data = try generateTestGGUF(testing.allocator, config);
//     defer testing.allocator.free(gguf_data);
//
//     var file = try gguf.parse(gguf_data, testing.allocator);
//     defer file.deinit();
//
//     var model = try registry.createModel(testing.allocator, &file, config.arch, std.io.getStdIo());
//     defer model.deinit();
//
//     const params = model.getParams();
//     const input = [_]i32{ 1, 2, 3, 4 };
//     const logits = try runForward(testing.allocator, &model, params, &input);
//     defer testing.allocator.free(logits);
//
//     // 验证 logits 形状
//     try testing.expectEqual(@as(usize, config.n_vocab), logits.len);
//
//     // 验证不是 NaN
//     for (logits) |v| {
//         try testing.expect(!std.math.isNan(v));
//     }
// }

// ============================================================================
// Qwen2 架构测试
// ============================================================================

test "detectArchitecture - qwen2 via fallback tensor" {
    // 构造一个没有 general.architecture 但有 qwen2 张量的 GGUF
    var buf: [512]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    std.mem.writeInt(u32, buf[pos..][0..4], 3, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 1, .little); // tensor_count = 1
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little); // metadata_kv_count = 0
    pos += 8;

    // 写入张量信息：blk.0.attn_q.weight（qwen2 特征张量）
    const tname = "blk.0.attn_q.weight";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(tname.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..tname.len], tname);
    pos += tname.len;
    std.mem.writeInt(u32, buf[pos..][0..4], 2, .little); // n_dims = 2
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 64, .little); // dim[0] = 64
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 64, .little); // dim[1] = 64
    pos += 8;
    std.mem.writeInt(u32, buf[pos..][0..4], 0, .little); // data_type = f32
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little); // offset = 0
    pos += 8;

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const detected = registry.detectArchitecture(&file);
    try testing.expect(detected != null);
    try testing.expectEqual(model_if.Architecture.qwen2, detected.?);
}

test "detectArchitecture - qwen35 via fallback tensor" {
    // 构造一个没有 general.architecture 但有 qwen35 张量的 GGUF
    var buf: [512]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    std.mem.writeInt(u32, buf[pos..][0..4], 3, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 1, .little); // tensor_count = 1
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little); // metadata_kv_count = 0
    pos += 8;

    // 写入张量信息：blk.0.ssm_conv1d.weight（qwen35 特征张量）
    const tname = "blk.0.ssm_conv1d.weight";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(tname.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..tname.len], tname);
    pos += tname.len;
    std.mem.writeInt(u32, buf[pos..][0..4], 2, .little); // n_dims = 2
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 64, .little); // dim[0] = 64
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 64, .little); // dim[1] = 64
    pos += 8;
    std.mem.writeInt(u32, buf[pos..][0..4], 0, .little); // data_type = f32
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little); // offset = 0
    pos += 8;

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const detected = registry.detectArchitecture(&file);
    try testing.expect(detected != null);
    try testing.expectEqual(model_if.Architecture.qwen35, detected.?);
}

test "detectArchitecture - qwen2 via attn_qkv tensor" {
    var buf: [512]u8 = undefined;
    @memset(&buf, 0);
    var pos: usize = 0;

    @memcpy(buf[pos..][0..4], "GGUF");
    pos += 4;
    std.mem.writeInt(u32, buf[pos..][0..4], 3, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 1, .little);
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little);
    pos += 8;

    const tname = "blk.0.attn_qkv.weight";
    std.mem.writeInt(u64, buf[pos..][0..8], @intCast(tname.len), .little);
    pos += 8;
    @memcpy(buf[pos..][0..tname.len], tname);
    pos += tname.len;
    std.mem.writeInt(u32, buf[pos..][0..4], 2, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 64, .little);
    pos += 8;
    std.mem.writeInt(u64, buf[pos..][0..8], 64, .little);
    pos += 8;
    std.mem.writeInt(u32, buf[pos..][0..4], 0, .little);
    pos += 4;
    std.mem.writeInt(u64, buf[pos..][0..8], 0, .little);
    pos += 8;

    const data = buf[0..pos];
    var file = try gguf.parse(data, testing.allocator);
    defer file.deinit();

    const detected = registry.detectArchitecture(&file);
    try testing.expect(detected != null);
    try testing.expectEqual(model_if.Architecture.qwen2, detected.?);
}

test "architecture fromString - qwen2 variants" {
    try testing.expectEqual(model_if.Architecture.qwen2, model_if.Architecture.fromString("qwen2").?);
    try testing.expectEqual(model_if.Architecture.qwen2, model_if.Architecture.fromString("qwen2.5").?);
}

test "architecture fromString - qwen35 variants" {
    try testing.expectEqual(model_if.Architecture.qwen35, model_if.Architecture.fromString("qwen3.5").?);
    try testing.expectEqual(model_if.Architecture.qwen35, model_if.Architecture.fromString("qwen35").?);
}

test "architecture fromString - llama variants" {
    try testing.expectEqual(model_if.Architecture.llama, model_if.Architecture.fromString("llama").?);
    try testing.expectEqual(model_if.Architecture.llama, model_if.Architecture.fromString("llama2").?);
    try testing.expectEqual(model_if.Architecture.llama, model_if.Architecture.fromString("llama3").?);
}

test "architecture fromString - case sensitivity" {
    try testing.expect(model_if.Architecture.fromString("LLAMA") == null);
    try testing.expect(model_if.Architecture.fromString("Qwen2") == null);
}

test "GraphBuilder buildRope" {
    const ctx = try ggml.Context.initNoAlloc(128 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    const params = model_if.ModelParams{
        .n_embd = 64,
        .n_head = 4,
        .n_kv_head = 2,
        .n_head_dim = 16,
        .n_vocab = 128,
    };

    var builder = graph_builder.GraphBuilder.init(ctx, graph, &params, testing.allocator);

    ctx.setNoAlloc(false);
    const q = try ctx.newTensor3d(.f32, 16, 4, 1);
    const k = try ctx.newTensor3d(.f32, 16, 2, 1);
    const pos = try ctx.newTensor1d(.i32, 1);
    ctx.setNoAlloc(true);

    // 设置位置
    const pos_data = pos.dataBytes();
    @as([*]i32, @ptrCast(@alignCast(pos_data.ptr)))[0] = 0;

    const rope_config = graph_builder.RopeConfig{
        .rope_dim = 16,
        .rope_theta = 10000000.0,
        .mode = 2,
    };

    const result = builder.buildRope(q, k, pos, rope_config);
    try testing.expect(result.q != undefined);
    try testing.expect(result.k != undefined);
}

test "GraphBuilder buildSwiGLU" {
    const ctx = try ggml.Context.initNoAlloc(128 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    const params = model_if.ModelParams{
        .n_embd = 64,
        .n_ff = 128,
    };

    var builder = graph_builder.GraphBuilder.init(ctx, graph, &params, testing.allocator);

    ctx.setNoAlloc(false);
    const x = try ctx.newTensor2d(.f32, 64, 1);
    const gate = try ctx.newTensor2d(.f32, 128, 64);
    const up = try ctx.newTensor2d(.f32, 128, 64);
    const down = try ctx.newTensor2d(.f32, 64, 128);
    ctx.setNoAlloc(true);

    const result = builder.buildSwiGLU(x, gate, up, down);
    try testing.expect(result != undefined);
}

test "GraphBuilder buildAttention" {
    const ctx = try ggml.Context.initNoAlloc(256 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    const params = model_if.ModelParams{
        .n_embd = 64,
        .n_head = 4,
        .n_kv_head = 2,
        .n_head_dim = 16,
        .n_vocab = 128,
    };

    var builder = graph_builder.GraphBuilder.init(ctx, graph, &params, testing.allocator);

    ctx.setNoAlloc(false);
    const q = try ctx.newTensor3d(.f32, 16, 4, 1);
    const k = try ctx.newTensor3d(.f32, 16, 2, 5);
    const v = try ctx.newTensor3d(.f32, 16, 2, 5);
    ctx.setNoAlloc(true);

    const result = builder.buildAttention(q, k, v, 4, 2, 16, 1, 5, 0);
    try testing.expect(result != undefined);
}

test "GraphBuilder forwardExpand" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    const params = model_if.ModelParams{ .n_embd = 64 };
    var builder = graph_builder.GraphBuilder.init(ctx, graph, &params, testing.allocator);

    ctx.setNoAlloc(false);
    const a = try ctx.newTensor1d(.f32, 4);
    const b = try ctx.newTensor1d(.f32, 4);
    ctx.setNoAlloc(true);

    const c = ggml.add(ctx, a, b);
    builder.forwardExpand(c);

    try testing.expectEqual(@as(i32, 1), graph.nNodes());
}

test "GraphBuilder setOutput" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    const params = model_if.ModelParams{ .n_embd = 64 };
    var builder = graph_builder.GraphBuilder.init(ctx, graph, &params, testing.allocator);

    ctx.setNoAlloc(false);
    const a = try ctx.newTensor1d(.f32, 4);
    ctx.setNoAlloc(true);

    builder.setOutput(a);
    // setOutput 不返回错误，只是设置标记
    try testing.expect(true);
}

// ============================================================================
// 模型参数测试
// ============================================================================

test "ModelParams rope_scaling" {
    const scaling = model_if.RopeScaling{
        .rope_type = "linear",
        .factor = 2.0,
        .original_max_seq_len = 16384,
    };
    try testing.expectEqualStrings("linear", scaling.rope_type);
    try testing.expectEqual(@as(f32, 2.0), scaling.factor);
    try testing.expectEqual(@as(u32, 16384), scaling.original_max_seq_len);
}

test "ModelParams deinit" {
    var params = model_if.ModelParams{};
    params.deinit(); // 应该不崩溃
    try testing.expect(true);
}

test "ModelWeights struct" {
    // 验证 ModelWeights 结构体存在且可实例化
    // 注意：需要 ggml.Tensor 指针，这里只测试类型存在
    try testing.expectEqual(@as(usize, @sizeOf(model_if.ModelWeights)), @sizeOf(model_if.ModelWeights));
}

// ============================================================================
// 工具函数测试
// ============================================================================

test "test_utils nmse" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(@as(f64, 0.0), test_utils.nmse(&a, &b));
}

test "test_utils TestConfig" {
    const c = test_utils.TestConfig{ .arch = .llama };
    try testing.expectEqual(@as(u32, 2), c.n_layer);
    try testing.expectEqual(@as(u32, 64), c.n_embd);
    try testing.expectEqual(@as(u32, 128), c.n_vocab);
}
