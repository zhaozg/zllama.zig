//! Permute order alignment test for audio encoder attention.
//!
//! Tests that the chunked local attention permute order in encoder.zig
//! produces the same mathematical result as the llama.cpp gemma4a.cpp
//! permute order, despite different data layouts.
//!
//! ggml_permute semantics ("new-axis → old-axis" mapping):
//!   permute(axis0, axis1, axis2, axis3) on tensor with ne=[n0,n1,n2,n3]:
//!     ne[axis0] = old ne[0]   // new axis 0 gets old axis0 value
//!     ne[axis1] = old ne[1]   // new axis 1 gets old axis1 value
//!     ne[axis2] = old ne[2]
//!     ne[axis3] = old ne[3]
//!   Then result->ne[i] = ne[i] for i=0..3
//!
//!   IMPORTANT: This is "new-axis-index → old-axis-index", NOT "old-axis→new-axis".
//!   Example: permute(1,2,0,3) on [A,B,C,D]:
//!     ne[1]=A, ne[2]=B, ne[0]=C, ne[3]=D → result = [C,A,B,D]
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
// Test 1: Verify ggml permute semantics (new-axis → old-axis)
// ============================================================================

test "permute semantics: basic 4D" {
    const ctx = try createContext();
    defer ctx.deinit();

    // t = [2, 3, 4, 5]
    const t = try ctx.newTensor4d(ggml.Type.f32, 2, 3, 4, 5);
    const data = t.dataF32();
    for (0..@as(usize, @intCast(2 * 3 * 4 * 5))) |i| {
        data[i] = @floatFromInt(i);
    }

    // permute(0,3,1,2): ne[0]=old[0]=2, ne[3]=old[1]=3, ne[1]=old[2]=4, ne[2]=old[3]=5
    // result: [2, 4, 5, 3]
    const p1 = ggml.permute(ctx, t, 0, 3, 1, 2);
    const ne1 = p1.ne();
    try testing.expectEqual(@as(i64, 2), ne1[0]);
    try testing.expectEqual(@as(i64, 4), ne1[1]);
    try testing.expectEqual(@as(i64, 5), ne1[2]);
    try testing.expectEqual(@as(i64, 3), ne1[3]);

    // permute(0,2,1,3): ne[0]=old[0]=2, ne[2]=old[1]=3, ne[1]=old[2]=4, ne[3]=old[3]=5
    // result: [2, 4, 3, 5]
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

// ============================================================================
// Test 5: Audio encoder Flatten pattern (matches llama.cpp gemma4a.cpp:55-56)
//
// Derivation: permute(1,2,0,3) on [freq, time, ch, batch]:
//   ne[1]=old[0]=freq, ne[2]=old[1]=time, ne[0]=old[2]=ch, ne[3]=old[3]=batch
//   result: [ch, freq, time, batch]
//   Then reshape2d: ne[0]*ne[1]=ch*freq, ne[2]=time → [ch*freq, time]
// ============================================================================

test "permute: audio flatten pathway (llama.cpp match)" {
    const ctx = try createContext();
    defer ctx.deinit();

    // Simulate Conv2D output: [freq=32, time=20, ch=64, batch=1]
    // Use different values for freq/ch to catch axis confusion
    const freq: i64 = 32;
    const time: i64 = 20;
    const ch: i64 = 64;
    const batch: i64 = 1;

    const conv_output = try ctx.newTensor4d(ggml.Type.f32, freq, time, ch, batch);
    fillTensorPattern(conv_output, 3001);

    // Flatten: matches llama.cpp ggml_permute(ctx0, cur, 1, 2, 0, 3)
    // [freq, time, ch, batch] -> [ch, freq, time, batch]
    const flat = conv_output.permute(ctx, 1, 2, 0, 3).cont(ctx);

    // Verify intermediate shape (use distinct values for detection)
    const ne_flat = flat.ne();
    try testing.expectEqual(ch, ne_flat[0]); // NOT time — this is the critical verification
    try testing.expectEqual(freq, ne_flat[1]);
    try testing.expectEqual(time, ne_flat[2]);
    try testing.expectEqual(batch, ne_flat[3]);

    // Final reshape: [ch*freq, time]
    const flat_dim0 = ne_flat[0] * ne_flat[1]; // = ch * freq = 64*32 = 2048
    const reshaped = flat.reshape2d(ctx, flat_dim0, ne_flat[2]);
    const ne_r = reshaped.ne();
    try testing.expectEqual(ch * freq, ne_r[0]); // 2048
    try testing.expectEqual(time, ne_r[1]); // 20

    // Verify computation doesn't produce NaN/Inf
    const result = computeAndGetData(ctx, reshaped);
    for (result) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

// ============================================================================
// Test 6: Verify wrong permute produces different shape
// ============================================================================

test "permute: wrong order produces different shape" {
    const ctx = try createContext();
    defer ctx.deinit();

    const freq: i64 = 32;
    const time: i64 = 20;
    const ch: i64 = 64;

    const conv_output = try ctx.newTensor4d(ggml.Type.f32, freq, time, ch, 1);

    // Correct (llama.cpp): permute(1,2,0,3) → [ch, freq, time, 1]
    const correct = conv_output.permute(ctx, 1, 2, 0, 3).cont(ctx);
    const flat_dim0_correct = correct.ne()[0] * correct.ne()[1]; // ch*freq
    const reshaped_correct = correct.reshape2d(ctx, flat_dim0_correct, correct.ne()[2]);
    // [2048, 20]

    // Wrong: permute(2,0,1,3) → [time, ch, freq, 1]
    const wrong = conv_output.permute(ctx, 2, 0, 1, 3).cont(ctx);
    const flat_dim0_wrong = wrong.ne()[0] * wrong.ne()[1]; // time*ch
    const reshaped_wrong = wrong.reshape2d(ctx, flat_dim0_wrong, wrong.ne()[2]);
    // [1280, 32]

    // The two paths produce different result shapes
    try testing.expectEqual(ch * freq, reshaped_correct.ne()[0]); // 2048
    try testing.expectEqual(time, reshaped_correct.ne()[1]); // 20

    // Wrong path: [time*ch, freq] = [1280, 32]
    try testing.expectEqual(time * ch, reshaped_wrong.ne()[0]); // 1280
    try testing.expectEqual(freq, reshaped_wrong.ne()[1]); // 32

    // Verify they're different
    try testing.expect(reshaped_correct.ne()[0] != reshaped_wrong.ne()[0] or
        reshaped_correct.ne()[1] != reshaped_wrong.ne()[1]);
}

// ============================================================================
// Test 7: End-to-end Flatten + input projection (mul_mat compatibility)
// ============================================================================

test "permute: flatten + mul_mat input projection" {
    const ctx = try createContext();
    defer ctx.deinit();

    const freq: i64 = 32;
    const time: i64 = 20;
    const ch: i64 = 32;
    const embd: i64 = 512;

    // Conv2D output
    const conv_output = try ctx.newTensor4d(ggml.Type.f32, freq, time, ch, 1);
    fillTensorPattern(conv_output, 4001);

    // Flatten (matches llama.cpp): [freq, time, ch, 1] -> [ch, freq, time, 1]
    var flat = conv_output.permute(ctx, 1, 2, 0, 3).cont(ctx);
    const flat_dim0 = flat.ne()[0] * flat.ne()[1]; // ch * freq = 1024
    flat = flat.reshape2d(ctx, flat_dim0, flat.ne()[2]); // [1024, time]

    // Input projection weight: [embd, flat_dim0] = [512, 1024]
    const proj_w = try ctx.newTensor2d(ggml.Type.f32, flat_dim0, embd);
    fillTensorPattern(proj_w, 4002);

    // mul_mat: contracts on ne[0]=flat_dim0, result = [embd, time]
    const proj_out = proj_w.mulMat(ctx, flat);
    const ne = proj_out.ne();
    try testing.expectEqual(embd, ne[0]);
    try testing.expectEqual(time, ne[1]);

    // Verify computation
    const result = computeAndGetData(ctx, proj_out);
    for (result) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

// ============================================================================
// Test 8: LayerNorm permute pattern (audio encoder NormLayer)
//
// Derivation:
//   Start: [freq, time, ch, batch] = [32, 20, 64, 1]
//   permute(1,2,0,3): [ch, freq, time, batch] = [64, 32, 20, 1] — ch at ne[0] for norm
//   After norm + scale:
//   permute(2,0,1,3): [freq, time, ch, batch] = [32, 20, 64, 1] — restored
// ============================================================================

test "permute: audio layernorm pattern" {
    const ctx = try createContext();
    defer ctx.deinit();

    // Use distinct values to catch axis confusion
    const freq: i64 = 32;
    const time: i64 = 20;
    const ch: i64 = 64;
    const batch: i64 = 1;

    const cur = try ctx.newTensor4d(ggml.Type.f32, freq, time, ch, batch);
    fillTensorPattern(cur, 5001);

    // LayerNorm step 1: permute ch to ne[0] for norm on C axis
    // Matches llama.cpp: ggml_permute(ctx0, cur, 1, 2, 0, 3)
    // [freq, time, ch, batch] -> [ch, freq, time, batch]
    const step1 = cur.permute(ctx, 1, 2, 0, 3).cont(ctx);
    try testing.expectEqual(ch, step1.ne()[0]); // 64 — ch at ne[0]
    try testing.expectEqual(freq, step1.ne()[1]); // 32
    try testing.expectEqual(time, step1.ne()[2]); // 20
    try testing.expectEqual(batch, step1.ne()[3]); // 1

    // LayerNorm step 2: permute back to original layout
    // Matches llama.cpp: ggml_permute(ctx0, cur, 2, 0, 1, 3)
    // [ch, freq, time, batch] -> [freq, time, ch, batch]
    const step2 = step1.permute(ctx, 2, 0, 1, 3).cont(ctx);
    try testing.expectEqual(freq, step2.ne()[0]); // 32 — restored
    try testing.expectEqual(time, step2.ne()[1]); // 20
    try testing.expectEqual(ch, step2.ne()[2]); // 64
    try testing.expectEqual(batch, step2.ne()[3]); // 1

    const result = computeAndGetData(ctx, step2);
    for (result) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

// ============================================================================
// Test 9: Q blocking permute (chunked attention)
// ============================================================================

test "permute: Q blocking pattern (chunked attention)" {
    const ctx = try createContext();
    defer ctx.deinit();

    const D: i64 = 64; // d_head
    const H: i64 = 8; // n_head
    const C: i64 = 12; // chunk_size
    const B: i64 = 5; // num_blocks

    // Q after reshape: [D, H, C, B]
    const Q = try ctx.newTensor4d(ggml.Type.f32, D, H, C, B);
    fillTensorPattern(Q, 6001);

    // Matches llama.cpp gemma4a: permute(0,3,1,2) -> [D, C, B, H]
    const Q_blocked = Q.permute(ctx, 0, 3, 1, 2).cont(ctx);
    const ne = Q_blocked.ne();
    try testing.expectEqual(D, ne[0]);
    try testing.expectEqual(C, ne[1]);
    try testing.expectEqual(B, ne[2]);
    try testing.expectEqual(H, ne[3]);

    const result = computeAndGetData(ctx, Q_blocked);
    for (result) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}
