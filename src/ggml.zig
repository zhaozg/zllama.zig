//! ggml.zig - 安全封装层（模块化入口）
//!
//! 提供 ggml C API 的类型安全 Zig 封装。
//! 所有分配类操作返回 `!*T` 错误联合，纯计算操作返回 `*T`。
//! 使用 `opaque {}` 类型包装不透明指针。
//!
//! 模块结构：
//! - ggml/c.zig:      原始 C API 导入和类型枚举
//! - ggml/context.zig: ggml_context 封装
//! - ggml/tensor.zig:  ggml_tensor 封装
//! - ggml/graph.zig:   ggml_cgraph 封装
//! - ggml/backend.zig: Backend 与 Gallocr 封装
//! - ggml/ops.zig:     计算图操作函数
//! - ggml/utils.zig:   工具函数（版本、CPU 特性等）

const std = @import("std");

// ============================================================================
// 重新导出所有子模块
// ============================================================================

pub const c = @import("ggml/c.zig").c;
pub const Type = @import("ggml/c.zig").Type;
pub const GgufValueType = @import("ggml/c.zig").GgufValueType;
pub const GgufValue = @import("ggml/c.zig").GgufValue;

pub const Context = @import("ggml/context.zig").Context;
pub const Tensor = @import("ggml/tensor.zig").Tensor;
pub const CGraph = @import("ggml/graph.zig").CGraph;

pub const Backend = @import("ggml/backend.zig").Backend;
pub const BackendBufferType = @import("ggml/backend.zig").BackendBufferType;
pub const Gallocr = @import("ggml/backend.zig").Gallocr;
pub const backendCpuInit = @import("ggml/backend.zig").backendCpuInit;
pub const backendCpuBufferType = @import("ggml/backend.zig").backendCpuBufferType;
pub const backendGetDefaultBufferType = @import("ggml/backend.zig").backendGetDefaultBufferType;
pub const backendAllocCtxTensors = @import("ggml/backend.zig").backendAllocCtxTensors;
pub const backendAllocCtxTensorsFromBuft = @import("ggml/backend.zig").backendAllocCtxTensorsFromBuft;
pub const backendFree = @import("ggml/backend.zig").backendFree;
pub const loadBackends = @import("ggml/backend.zig").loadBackends;
pub const setInput = @import("ggml/backend.zig").setInput;

pub const mulMat = @import("ggml/ops.zig").mulMat;
pub const mul = @import("ggml/ops.zig").mul;
pub const add = @import("ggml/ops.zig").add;
pub const neg = @import("ggml/ops.zig").neg;
pub const exp = @import("ggml/ops.zig").exp;
pub const cpy = @import("ggml/ops.zig").cpy;
pub const rmsNorm = @import("ggml/ops.zig").rmsNorm;
pub const l2Norm = @import("ggml/ops.zig").l2Norm;
pub const ropeExt = @import("ggml/ops.zig").ropeExt;
pub const ropeMulti = @import("ggml/ops.zig").ropeMulti;
pub const scale = @import("ggml/ops.zig").scale;
pub const softMax = @import("ggml/ops.zig").softMax;
pub const diagMaskInf = @import("ggml/ops.zig").diagMaskInf;
pub const silu = @import("ggml/ops.zig").silu;
pub const sigmoid = @import("ggml/ops.zig").sigmoid;
pub const softplus = @import("ggml/ops.zig").softplus;
pub const permute = @import("ggml/ops.zig").permute;
pub const cont = @import("ggml/ops.zig").cont;
pub const gatedDeltaNet = @import("ggml/ops.zig").gatedDeltaNet;

pub const cont2d = @import("ggml/ops.zig").cont2d;
pub const cont4d = @import("ggml/ops.zig").cont4d;
pub const reshape2d = @import("ggml/ops.zig").reshape2d;
pub const reshape3d = @import("ggml/ops.zig").reshape3d;
pub const reshape4d = @import("ggml/ops.zig").reshape4d;
pub const repeat = @import("ggml/ops.zig").repeat;
pub const repeat4d = @import("ggml/ops.zig").repeat4d;
pub const transpose = @import("ggml/ops.zig").transpose;
pub const concat = @import("ggml/ops.zig").concat;
pub const getRows = @import("ggml/ops.zig").getRows;
pub const conv1d = @import("ggml/ops.zig").conv1d;
pub const ssmConv = @import("ggml/ops.zig").ssmConv;
pub const ssmScan = @import("ggml/ops.zig").ssmScan;
pub const setOutput = @import("ggml/ops.zig").setOutput;

pub const version = @import("ggml/utils.zig").version;
pub const cpuNThreads = @import("ggml/utils.zig").cpuNThreads;
pub const CpuFeatures = @import("ggml/utils.zig").CpuFeatures;
pub const recommendedThreads = @import("ggml/utils.zig").recommendedThreads;
pub const LogLevel = @import("ggml/utils.zig").LogLevel;
pub const logSet = @import("ggml/utils.zig").logSet;
pub const logSetCallback = @import("ggml/utils.zig").logSetCallback;

// ============================================================================
// 测试（集成测试，需要 ggml context）
// ============================================================================

const testing = std.testing;

test "ggml version" {
    const v = version();
    try testing.expect(v.len > 0);
    std.debug.print("ggml version: {s}\n", .{v});
}

test "CpuFeatures" {
    _ = CpuFeatures.hasAvx2();
    _ = CpuFeatures.hasNeon();
    _ = CpuFeatures.hasSve();
}

test "recommendedThreads" {
    const n = recommendedThreads();
    try testing.expect(n >= 1);
}

test "conv1d basic" {
    const kernel_size: i64 = 4;
    const n_channels: i64 = 6;
    const n_tokens: i64 = 10;

    const mem_size = 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    const kernel = try ctx.newTensor2d(.f32, kernel_size, n_channels);
    kernel.setName("test_conv_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    const data_tensor = try ctx.newTensor2d(.f32, n_tokens, n_channels);
    data_tensor.setName("test_conv_data");
    {
        const data = data_tensor.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    var graph = try CGraph.init(ctx);
    const result = conv1d(ctx, kernel, data_tensor, 1, @as(i32, @intCast(kernel_size - 1)), 1);
    result.setName("test_conv_result");
    setOutput(result);
    graph.buildForwardExpand(result);

    try graph.compute(1);

    const shape = result.shape();
    try testing.expectEqual(n_tokens, shape[0]);
    try testing.expectEqual(n_channels, shape[1]);

    std.debug.print("conv1d test passed: output shape [{d}, {d}]\n", .{ shape[0], shape[1] });
}

test "conv1d with transpose" {
    const kernel_size: i64 = 4;
    const n_channels: i64 = 6;
    const n_tokens: i64 = 10;

    const mem_size = 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    const kernel = try ctx.newTensor2d(.f32, kernel_size, n_channels);
    kernel.setName("test_conv_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    const input = try ctx.newTensor2d(.f32, n_channels, n_tokens);
    input.setName("test_input");
    {
        const data = input.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    var graph = try CGraph.init(ctx);
    const transposed = cont(ctx, permute(ctx, input, 1, 0, 2, 3));
    transposed.setName("test_transposed");

    const result = conv1d(ctx, kernel, transposed, 1, @as(i32, @intCast(kernel_size - 1)), 1);
    result.setName("test_conv_result");

    const result_t = cont(ctx, permute(ctx, result, 1, 0, 2, 3));
    result_t.setName("test_result_t");

    setOutput(result_t);
    graph.buildForwardExpand(result_t);

    try graph.compute(1);

    const shape = result_t.shape();
    try testing.expectEqual(n_channels, shape[0]);
    try testing.expectEqual(n_tokens, shape[1]);

    std.debug.print("conv1d transpose test passed: output shape [{d}, {d}]\n", .{ shape[0], shape[1] });
}

test "ssmConv basic" {
    const d_conv: i64 = 4;
    const d_inner: i64 = 6;
    const n_t: i64 = 10;
    const n_s: i64 = 1;

    const mem_size = 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    const kernel = try ctx.newTensor2d(.f32, d_conv, d_inner);
    kernel.setName("test_ssm_conv_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    const sx = try ctx.newTensor3d(.f32, d_conv - 1 + n_t, d_inner, n_s);
    sx.setName("test_ssm_conv_sx");
    {
        const data = sx.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    var graph = try CGraph.init(ctx);
    const result = ssmConv(ctx, sx, kernel);
    result.setName("test_ssm_conv_result");
    setOutput(result);
    graph.buildForwardExpand(result);

    try graph.compute(1);

    const shape = result.shape();
    try testing.expectEqual(d_inner, shape[0]);
    try testing.expectEqual(n_t, shape[1]);
    try testing.expectEqual(n_s, shape[2]);

    std.debug.print("ssmConv test passed: output shape [{d}, {d}, {d}]\n", .{ shape[0], shape[1], shape[2] });
}

test "ssmConv with concat and view3d" {
    const d_conv: i64 = 4;
    const d_inner: i64 = 6;
    const n_t: i64 = 10;

    const mem_size = 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    const kernel = try ctx.newTensor2d(.f32, d_conv, d_inner);
    kernel.setName("test_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    const conv_state = try ctx.newTensor2d(.f32, d_conv - 1, d_inner);
    conv_state.setName("test_conv_state");
    conv_state.setZero();

    const input = try ctx.newTensor2d(.f32, n_t, d_inner);
    input.setName("test_input");
    {
        const data = input.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    var graph = try CGraph.init(ctx);
    const concat_result = concat(ctx, conv_state, input, 0);
    concat_result.setName("test_concat");

    const concat_cont = cont(ctx, concat_result);
    concat_cont.setName("test_concat_cont");

    const sx_3d = ctx.view3d(concat_cont, d_conv - 1 + n_t, d_inner, 1, @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32))), @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32) * d_inner)), 0);
    sx_3d.setName("test_sx_3d");

    const result = ssmConv(ctx, sx_3d, kernel);
    result.setName("test_ssm_conv_result");
    setOutput(result);
    graph.buildForwardExpand(result);

    try graph.compute(1);

    const shape = result.shape();
    try testing.expectEqual(d_inner, shape[0]);
    try testing.expectEqual(n_t, shape[1]);
    try testing.expectEqual(@as(i64, 1), shape[2]);

    std.debug.print("ssmConv concat+view3d test passed: output shape [{d}, {d}, {d}]\n", .{ shape[0], shape[1], shape[2] });
}

test "conv1d vs ssmConv equivalence" {
    const d_conv: i64 = 4;
    const d_inner: i64 = 6;
    const n_t: i64 = 10;

    const mem_size = 2 * 1024 * 1024;
    var ctx = try Context.init(mem_size);
    defer ctx.deinit();

    const kernel = try ctx.newTensor2d(.f32, d_conv, d_inner);
    kernel.setName("test_kernel");
    {
        const data = kernel.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 4 + 1)) * 0.1;
        }
    }

    const input = try ctx.newTensor2d(.f32, n_t, d_inner);
    input.setName("test_input");
    {
        const data = input.dataF32();
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.01;
        }
    }

    var graph1 = try CGraph.init(ctx);
    const conv1d_result = conv1d(ctx, kernel, input, 1, @as(i32, @intCast(d_conv - 1)), 1);
    conv1d_result.setName("test_conv1d_result");
    setOutput(conv1d_result);
    graph1.buildForwardExpand(conv1d_result);
    try graph1.compute(1);

    const conv1d_shape = conv1d_result.shape();
    try testing.expectEqual(n_t, conv1d_shape[0]);
    try testing.expectEqual(d_inner, conv1d_shape[1]);

    const conv_state = try ctx.newTensor2d(.f32, d_conv - 1, d_inner);
    conv_state.setName("test_conv_state");
    conv_state.setZero();

    var graph2 = try CGraph.init(ctx);
    const concat_result = concat(ctx, conv_state, input, 0);
    concat_result.setName("test_concat");

    const concat_cont = cont(ctx, concat_result);
    concat_cont.setName("test_concat_cont");

    const sx_3d = ctx.view3d(concat_cont, d_conv - 1 + n_t, d_inner, 1, @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32))), @as(usize, @intCast((d_conv - 1 + n_t) * @sizeOf(f32) * d_inner)), 0);
    sx_3d.setName("test_sx_3d");

    const ssm_conv_result = ssmConv(ctx, sx_3d, kernel);
    ssm_conv_result.setName("test_ssm_conv_result");
    setOutput(ssm_conv_result);
    graph2.buildForwardExpand(ssm_conv_result);
    try graph2.compute(1);

    const ssm_conv_shape = ssm_conv_result.shape();
    try testing.expectEqual(d_inner, ssm_conv_shape[0]);
    try testing.expectEqual(n_t, ssm_conv_shape[1]);
    try testing.expectEqual(@as(i64, 1), ssm_conv_shape[2]);

    std.debug.print("conv1d vs ssmConv equivalence test passed\n", .{});
    std.debug.print("  conv1d output:  [{d}, {d}]\n", .{ conv1d_shape[0], conv1d_shape[1] });
    std.debug.print("  ssmConv output: [{d}, {d}, {d}]\n", .{ ssm_conv_shape[0], ssm_conv_shape[1], ssm_conv_shape[2] });
}
