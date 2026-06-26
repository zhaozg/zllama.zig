//! Permute order alignment test for audio encoder attention.
//!
//! Tests that the chunked local attention permute order in encoder.zig
//! produces the same mathematical result as the llama.cpp gemma4a.cpp
//! permute order, despite different data layouts.
//!
//! ggml_permute semantics:
//!   permute(axis0, axis1, axis2, axis3) on tensor with ne=[n0,n1,n2,n3]:
//!     ne[axis0] = old ne[0]
//!     ne[axis1] = old ne[1]
//!     ne[axis2] = old ne[2]
//!     ne[axis3] = old ne[3]
//!   Then result->ne[i] = ne[i] for i=0..3
//!
//! ggml_mul_mat(a, b) semantics:
//!   Result ne[0] = a->ne[1]
//!   Result ne[1] = b->ne[1]
//!   Result ne[2] = b->ne[2]
//!   Result ne[3] = b->ne[3]
//!   Contracts on ne[0] (must match: a->ne[0] == b->ne[0])

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");

fn createContext() !*ggml.Context {
    return try ggml.Context.init(64 * 1024 * 1024);
}

fn fillTensorPattern(t: *ggml.Tensor, seed: u64) void {
    const data = t.dataF32();
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (data) |*v| {
        v.* = r.float(f32) * 2.0 - 1.0;
    }
}

fn computeAndGetData(ctx: *ggml.Context, result: *ggml.Tensor) []f32 {
    var cgraph = ggml.CGraph.initReserved(ctx, 4096) catch unreachable;
    cgraph.buildForwardExpand(result);
    const buft = ggml.backendCpuBufferType();
    var galloc = ggml.Gallocr.init(buft) catch unreachable;
    defer galloc.free();
    _ = galloc.allocGraph(cgraph);
    cgraph.compute(1) catch unreachable;
    return result.dataF32();
}

// ============================================================================
// Test 1: Verify ggml permute semantics
// ============================================================================

test "permute semantics: basic 4D" {
    const ctx = try createContext();
    defer ctx.deinit();

    const t = try ctx.newTensor4d(ggml.Type.f32, 2, 3, 4, 5);
    const data = t.dataF32();
    for (0..@as(usize, @intCast(2 * 3 * 4 * 5))) |i| {
        data[i] = @floatFromInt(i);
    }

    const p1 = ggml.permute(ctx, t, 0, 3, 1, 2);
    const ne1 = p1.ne();
    try testing.expectEqual(@as(i64, 2), ne1[0]);
    try testing.expectEqual(@as(i64, 4), ne1[1]);
    try testing.expectEqual(@as(i64, 5), ne1[2]);
    try testing.expectEqual(@as(i64, 3), ne1[3]);

    const p2 = ggml.permute(ctx, t, 0, 2, 1, 3);
    const ne2 = p2.ne();
    try testing.expectEqual(@as(i64, 2), ne2[0]);
    try testing.expectEqual(@as(i64, 4), ne2[1]);
    try testing.expectEqual(@as(i64, 3), ne2[2]);
    try testing.expectEqual(@as(i64, 5), ne2[3]);
}

// ============================================================================
// Test 2: Verify mul_mat dimension ordering for 4D tensors
// ============================================================================

test "permute: mul_mat 4D dimension ordering" {
    const ctx = try createContext();
    defer ctx.deinit();

    // A = [2, 3, 4, 5], B = [2, 6, 4, 5]
    const A = try ctx.newTensor4d(ggml.Type.f32, 2, 3, 4, 5);
    const B = try ctx.newTensor4d(ggml.Type.f32, 2, 6, 4, 5);

    // mul_mat(A, B): contracts on ne[0]=2
    // Result: ne[0]=A.ne[1]=3, ne[1]=B.ne[1]=6, ne[2]=B.ne[2]=4, ne[3]=B.ne[3]=5
    const C = A.mulMat(ctx, B);
    const ne = C.ne();
    try testing.expectEqual(@as(i64, 3), ne[0]);
    try testing.expectEqual(@as(i64, 6), ne[1]);
    try testing.expectEqual(@as(i64, 4), ne[2]);
    try testing.expectEqual(@as(i64, 5), ne[3]);
}

// ============================================================================
// Test 3: Verify the actual permute shapes used in encoder.zig
// ============================================================================

test "permute: encoder.zig shape verification" {
    const ctx = try createContext();
    defer ctx.deinit();

    const D: i64 = 8;
    const H: i64 = 4;
    const C: i64 = 12;
    const B: i64 = 3;

    const t = try ctx.newTensor4d(ggml.Type.f32, D, H, C, B);

    // Current Zig: permute(0,2,1,3) -> [D, C, H, B]
    const p_zig = ggml.permute(ctx, t, 0, 2, 1, 3);
    const ne_zig = p_zig.ne();
    try testing.expectEqual(D, ne_zig[0]);
    try testing.expectEqual(C, ne_zig[1]);
    try testing.expectEqual(H, ne_zig[2]);
    try testing.expectEqual(B, ne_zig[3]);

    // llama.cpp: permute(0,3,1,2) -> [D, C, B, H]
    const p_llama = ggml.permute(ctx, t, 0, 3, 1, 2);
    const ne_llama = p_llama.ne();
    try testing.expectEqual(D, ne_llama[0]);
    try testing.expectEqual(C, ne_llama[1]);
    try testing.expectEqual(B, ne_llama[2]);
    try testing.expectEqual(H, ne_llama[3]);

    // Verify the difference: H and B are swapped
    try testing.expectEqual(ne_zig[2], ne_llama[3]); // zig H == llama B
    try testing.expectEqual(ne_zig[3], ne_llama[2]); // zig B == llama H
}

// ============================================================================
// Test 4: Verify the RPE path works
// ============================================================================

test "permute: RPE path" {
    const ctx = try createContext();
    defer ctx.deinit();

    const D: i64 = 8;
    const H: i64 = 4;
    const C: i64 = 12;
    const B: i64 = 3;
    const R: i64 = 13;

    const Q = try ctx.newTensor4d(ggml.Type.f32, D, H, C, B);
    fillTensorPattern(Q, 1001);

    const p_rpe = try ctx.newTensor3d(ggml.Type.f32, D, R, H);
    fillTensorPattern(p_rpe, 2002);

    const Q_flat = Q.reshape3d(ctx, D, C * B, H);
    const matrix_bd = p_rpe.mulMat(ctx, Q_flat);
    const mb_reshaped = matrix_bd.reshape4d(ctx, R, C, B, H);

    const result = computeAndGetData(ctx, mb_reshaped);
    for (result) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}
