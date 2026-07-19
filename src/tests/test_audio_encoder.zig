//! Audio encoder unit tests for the zllama.zig multimodal pipeline.
//!
//! Tests the core building blocks of the Conformer audio encoder:
//!   - buildNorm: RMS/Layer normalization with optional weight/bias
//!   - buildFFN: FFN with SiLU/GELU activation, optional gate/bias, and clamp
//!   - buildMM: Matrix multiplication with optional input/output clamp
//!
//! These functions are the shared building blocks from src/mtmd/graph/.
//! They are tested in isolation using synthetic ggml tensors and graphs.
//!
//! Reference: llama.cpp tools/mtmd/clip.cpp (clip_graph_gemma4a)

const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");
const graph = @import("graph");

// ============================================================================
// Test helpers
// ============================================================================

/// Create a ggml context with no_alloc mode for testing.
/// Memory size is generous to accommodate graph building.
fn createTestCtx() !*ggml.Context {
    return try ggml.Context.initNoAlloc(64 * 1024 * 1024);
}

/// Fill a tensor with deterministic pseudo-random data.
/// Uses a simple LCG for reproducibility.
fn fillTensor(t: *ggml.Tensor, seed: u64) void {
    const data = t.dataF32();
    var rng = std.Random.DefaultPrng.init(seed);
    const rand = rng.random();
    for (data) |*v| {
        v.* = rand.float(f32) * 2.0 - 1.0; // [-1, 1]
    }
}

/// Fill a tensor with a constant value.
fn fillTensorConst(t: *ggml.Tensor, val: f32) void {
    const data = t.dataF32();
    @memset(data, val);
}

/// Fill a tensor with sequential values (for deterministic testing).
fn fillTensorSeq(t: *ggml.Tensor, start: f32, step: f32) void {
    const data = t.dataF32();
    var cur = start;
    for (data) |*v| {
        v.* = cur;
        cur += step;
    }
}

/// Compute a simple reference RMS norm on a slice.
fn referenceRmsNorm(data: []const f32, eps: f32) f32 {
    var sum_sq: f64 = 0.0;
    for (data) |v| {
        const vf: f64 = @floatCast(v);
        sum_sq += vf * vf;
    }
    const mean_sq: f64 = sum_sq / @as(f64, @floatFromInt(data.len));
    const rms: f64 = @sqrt(mean_sq + @as(f64, @floatCast(eps)));
    return @floatCast(rms);
}

/// Build and compute a simple graph with one output tensor.
/// Returns the output tensor's data after computation.
fn computeGraph(ctx: *ggml.Context, output: *ggml.Tensor, n_threads: i32) !void {
    var cgraph = try ggml.CGraph.initReserved(ctx, 1024);
    cgraph.buildForwardExpand(output);
    try cgraph.compute(n_threads);
}

// ============================================================================
// buildNorm tests (using graph.buildNorm)
// ============================================================================

test "buildNorm: RMS norm without weight/bias" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    // Create input tensor [8, 4] (8 features, 4 positions)
    const n_features: i64 = 8;
    const n_pos: i64 = 4;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.25);

    const eps: f32 = 1e-6;

    // Call graph.buildNorm with RMS norm, no weight/bias
    const result = try graph.buildNorm(ctx, input, null, null, .rms_norm, eps, "test_norm");

    // Set output and compute
    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify: each column should be RMS-normalized independently
    const out_data = result.dataF32();
    for (0..@as(usize, @intCast(n_pos))) |p| {
        const col = out_data[p * @as(usize, @intCast(n_features)) .. (p + 1) * @as(usize, @intCast(n_features))];
        // After RMS norm, the RMS of the output should be 1.0
        const out_rms = referenceRmsNorm(col, eps);
        try testing.expectApproxEqAbs(@as(f32, 1.0), out_rms, 1e-5);
    }
}

test "buildNorm: RMS norm with weight" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 8;
    const n_pos: i64 = 4;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.25);

    // Create weight tensor [8]
    const weight = try ctx.newTensor1d(ggml.Type.f32, n_features);
    fillTensorSeq(weight, 1.0, 0.1); // [1.0, 1.1, 1.2, ...]

    const eps: f32 = 1e-6;

    const result = try graph.buildNorm(ctx, input, weight, null, .rms_norm, eps, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify: output = rms_norm(input) * weight
    const out_data = result.dataF32();
    const w_data = weight.dataF32();

    // Compute expected: for each column, normalize then multiply by weight
    for (0..@as(usize, @intCast(n_pos))) |p| {
        const col_start = p * @as(usize, @intCast(n_features));
        // Get the input column
        var input_col: [8]f32 = undefined;
        for (0..@as(usize, @intCast(n_features))) |f| {
            input_col[f] = input.dataF32()[col_start + f];
        }
        const rms = referenceRmsNorm(&input_col, eps);
        for (0..@as(usize, @intCast(n_features))) |f| {
            const expected = (input_col[f] / rms) * w_data[f];
            try testing.expectApproxEqAbs(expected, out_data[col_start + f], 1e-5);
        }
    }
}

test "buildNorm: RMS norm with weight and bias" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 8;
    const n_pos: i64 = 4;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.25);

    const weight = try ctx.newTensor1d(ggml.Type.f32, n_features);
    fillTensorSeq(weight, 1.0, 0.1);

    const bias = try ctx.newTensor1d(ggml.Type.f32, n_features);
    fillTensorSeq(bias, 0.0, 0.05); // [0.0, 0.05, 0.10, ...]

    const eps: f32 = 1e-6;

    const result = try graph.buildNorm(ctx, input, weight, bias, .rms_norm, eps, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify: output = rms_norm(input) * weight + bias
    const out_data = result.dataF32();
    const w_data = weight.dataF32();
    const b_data = bias.dataF32();

    for (0..@as(usize, @intCast(n_pos))) |p| {
        const col_start = p * @as(usize, @intCast(n_features));
        var input_col: [8]f32 = undefined;
        for (0..@as(usize, @intCast(n_features))) |f| {
            input_col[f] = input.dataF32()[col_start + f];
        }
        const rms = referenceRmsNorm(&input_col, eps);
        for (0..@as(usize, @intCast(n_features))) |f| {
            const expected = (input_col[f] / rms) * w_data[f] + b_data[f];
            try testing.expectApproxEqAbs(expected, out_data[col_start + f], 1e-5);
        }
    }
}

test "buildNorm: Layer norm without weight/bias" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 8;
    const n_pos: i64 = 4;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.25);

    const eps: f32 = 1e-6;

    const result = try graph.buildNorm(ctx, input, null, null, .layer_norm, eps, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify: each column should be Layer-normalized
    const out_data = result.dataF32();
    for (0..@as(usize, @intCast(n_pos))) |p| {
        const col = out_data[p * @as(usize, @intCast(n_features)) .. (p + 1) * @as(usize, @intCast(n_features))];
        // After LayerNorm, mean should be ~0 and std should be ~1
        var sum: f64 = 0.0;
        var sum_sq: f64 = 0.0;
        for (col) |v| {
            const vf: f64 = @floatCast(v);
            sum += vf;
            sum_sq += vf * vf;
        }
        const n: f64 = @floatFromInt(col.len);
        const mean: f64 = sum / n;
        const variance: f64 = sum_sq / n - mean * mean;
        try testing.expectApproxEqAbs(@as(f32, 0.0), @as(f32, @floatCast(mean)), 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 1.0), @as(f32, @floatCast(@sqrt(variance))), 1e-4);
    }
}

test "buildNorm: RMS norm with zero input" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 8;
    const n_pos: i64 = 2;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorConst(input, 0.0);

    const eps: f32 = 1e-6;

    const result = try graph.buildNorm(ctx, input, null, null, .rms_norm, eps, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // With all-zero input and eps, output should be all zeros
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    }
}

test "buildNorm: RMS norm with single element" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 1;
    const n_pos: i64 = 3;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 2.0, 1.0); // [2.0, 3.0, 4.0]

    const eps: f32 = 1e-6;

    const result = try graph.buildNorm(ctx, input, null, null, .rms_norm, eps, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // For single element, rms_norm(x) = x / sqrt(x^2 + eps) ≈ sign(x)
    const out_data = result.dataF32();
    try testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), out_data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), out_data[2], 1e-5);
}

test "buildNorm: RMS norm with large epsilon" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorConst(input, 1.0);

    const eps: f32 = 1.0; // Large epsilon

    const result = try graph.buildNorm(ctx, input, null, null, .rms_norm, eps, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // With large eps, rms = sqrt(1 + 1) = sqrt(2) ≈ 1.414
    // output = 1.0 / 1.414 ≈ 0.707
    const out_data = result.dataF32();
    const expected: f32 = 1.0 / @sqrt(1.0 + eps);
    for (out_data) |v| {
        try testing.expectApproxEqAbs(expected, v, 1e-5);
    }
}

test "buildNorm: 3D tensor RMS norm" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    // 3D tensor [features, height, width]
    const n_features: i64 = 4;
    const n_h: i64 = 3;
    const n_w: i64 = 2;
    const input = try ctx.newTensor3d(ggml.Type.f32, n_features, n_h, n_w);
    fillTensor(input, 12345);

    const eps: f32 = 1e-6;

    const result = try graph.buildNorm(ctx, input, null, null, .rms_norm, eps, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // RMS norm operates on the last dimension (ne[0])
    const out_data = result.dataF32();
    const in_data = input.dataF32();
    const total_cols = @as(usize, @intCast(n_h * n_w));

    for (0..total_cols) |col| {
        const start = col * @as(usize, @intCast(n_features));
        const in_col = in_data[start .. start + @as(usize, @intCast(n_features))];
        const out_col = out_data[start .. start + @as(usize, @intCast(n_features))];
        const rms = referenceRmsNorm(in_col, eps);
        for (0..@as(usize, @intCast(n_features))) |f| {
            const expected = in_col[f] / rms;
            try testing.expectApproxEqAbs(expected, out_col[f], 1e-5);
        }
    }
}

// ============================================================================
// buildFFN tests (using graph.buildFFN)
// ============================================================================

test "buildFFN: SiLU without gate (simple FFN)" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    // Input [n_features, n_pos]
    const n_features: i64 = 8;
    const n_ff: i64 = 16;
    const n_pos: i64 = 2;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, -1.0, 0.25);

    // Up projection [n_ff, n_features] (ggml mulMat: weight.ne[0] == input.ne[0])
    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    // Down projection [n_features, n_ff]
    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, null, null, null, down_w, null, .silu, -1);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify shape: [n_features, n_pos]
    const ne = result.ne();
    try testing.expectEqual(n_features, ne[0]);
    try testing.expectEqual(n_pos, ne[1]);

    // Verify all values are finite
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildFFN: SwiGLU (with gate)" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 8;
    const n_ff: i64 = 16;
    const n_pos: i64 = 2;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, -1.0, 0.25);

    // Up projection [n_ff, n_features]
    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    // Gate projection [n_ff, n_features]
    const gate_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(gate_w, 77);

    // Down projection [n_features, n_ff]
    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, null, gate_w, null, down_w, null, .silu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify shape
    const result = try graph.buildFFN(ctx, input, up_w, null, gate_w, null, down_w, null, .silu, -1);
    try testing.expectEqual(n_pos, ne[1]);

    // Verify all values are finite
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildFFN: GELU without gate" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 8;
    const n_ff: i64 = 16;
    const n_pos: i64 = 2;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, -1.0, 0.25);

    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, null, null, null, down_w, null, .gelu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildFFN: with up bias" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_ff: i64 = 8;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.5);

    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    const up_b = try ctx.newTensor1d(ggml.Type.f32, n_ff);
    fillTensorConst(up_b, 0.1);

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, up_b, null, null, down_w, null, .silu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildFFN: with down bias" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_ff: i64 = 8;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.5);

    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const down_b = try ctx.newTensor1d(ggml.Type.f32, n_features);
    fillTensorConst(down_b, 0.2);

    const result = try graph.buildFFN(ctx, input, up_w, null, null, null, down_w, down_b, .silu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildFFN: GEGLU with gate bias" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_ff: i64 = 8;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.5);

    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    const gate_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(gate_w, 77);

    const gate_b = try ctx.newTensor1d(ggml.Type.f32, n_ff);
    fillTensorConst(gate_b, -0.05);

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, null, gate_w, gate_b, down_w, null, .gelu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildFFN: relu_sqr activation" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_ff: i64 = 8;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, -1.0, 0.5);

    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, null, null, null, down_w, null, .relu_sqr, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildFFN: zero input" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_ff: i64 = 8;
    const n_pos: i64 = 2;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorConst(input, 0.0);

    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, null, null, null, down_w, null, .silu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // With zero input and no bias, output should be zero
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-5);
    }
}

test "buildFFN: no up weight (passthrough)" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.5);

    // No up weight, no gate, no down - just silu activation on input
    // graph.buildFFN requires up and down weights, so we test with them
    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_features);
    fillTensorConst(up_w, 1.0); // identity-like

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_features);
    fillTensorConst(down_w, 1.0); // identity-like

    const result = try graph.buildFFN(ctx, input, up_w, null, null, null, down_w, null, .silu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

// ============================================================================
// buildMM tests (using graph.buildMM)
// ============================================================================

test "buildMM: basic matrix multiply without clamp" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    // ggml mulMat: weight.ne[0] must equal input.ne[0]
    // Weight [n_out, n_in]
    const out_features: i64 = 4;
    const in_features: i64 = 8;
    const n_pos: i64 = 3;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test_mm.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensor(x, 123);

    const result = try graph.buildMM(ctx, w, x, null);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify shape: [out_features, n_pos]
    const ne = result.ne();
    try testing.expectEqual(out_features, ne[0]);
    try testing.expectEqual(n_pos, ne[1]);

    // Verify values are finite
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildMM: with input clamp" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const out_features: i64 = 4;
    const in_features: i64 = 8;
    const n_pos: i64 = 2;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test_clamp_mm.weight");

    // Create input with large values that will be clamped
    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensorConst(x, 10.0); // All values = 10.0

    const clamp_info = graph.ClampInfo{
        .inp_min = -1.0,
        .inp_max = 1.0, // Clamp input to [-1, 1]
        .out_min = -std.math.floatMax(f32),
        .out_max = std.math.floatMax(f32),
    };

    const result = try graph.buildMM(ctx, w, x, &clamp_info);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Since input is clamped to [-1, 1], all 10.0 values become 1.0
    const out_data = result.dataF32();

    // Compute expected: w @ clamp(x, -1, 1) = w @ 1.0 (since all x=10.0 clamped to 1.0)
    const w_data = w.dataF32();
    for (0..@as(usize, @intCast(n_pos))) |p| {
        for (0..@as(usize, @intCast(out_features))) |f| {
            var expected: f32 = 0.0;
            for (0..@as(usize, @intCast(in_features))) |i| {
                expected += w_data[f * @as(usize, @intCast(in_features)) + i];
            }
            try testing.expectApproxEqAbs(expected, out_data[p * @as(usize, @intCast(out_features)) + f], 1e-4);
        }
    }
}

test "buildMM: with output clamp" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const out_features: i64 = 4;
    const in_features: i64 = 8;
    const n_pos: i64 = 2;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test_out_clamp.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensor(x, 123);

    const clamp_info = graph.ClampInfo{
        .inp_min = -std.math.floatMax(f32),
        .inp_max = std.math.floatMax(f32),
        .out_min = -0.5,
        .out_max = 0.5, // Clamp output to [-0.5, 0.5]
    };

    const result = try graph.buildMM(ctx, w, x, &clamp_info);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // All output values should be in [-0.5, 0.5]
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(v >= -0.5 - 1e-5);
        try testing.expect(v <= 0.5 + 1e-5);
    }
}

test "buildMM: with both input and output clamp" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const out_features: i64 = 4;
    const in_features: i64 = 8;
    const n_pos: i64 = 2;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test_both_clamp.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensorConst(x, 5.0);

    const clamp_info = graph.ClampInfo{
        .inp_min = -2.0,
        .inp_max = 2.0,
        .out_min = -1.0,
        .out_max = 1.0,
    };

    const result = try graph.buildMM(ctx, w, x, &clamp_info);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Input clamped to [-2, 2], output clamped to [-1, 1]
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(v >= -1.0 - 1e-5);
        try testing.expect(v <= 1.0 + 1e-5);
    }
}

test "buildMM: without clamp (null clamp_info)" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const out_features: i64 = 4;
    const in_features: i64 = 8;
    const n_pos: i64 = 2;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test_no_clamp.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensor(x, 123);

    // null clamp_info - no clamp applied
    const result = try graph.buildMM(ctx, w, x, null);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Should be equivalent to plain mulMat
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildMM: 1D input (vector)" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const out_features: i64 = 4;
    const in_features: i64 = 8;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test_1d.weight");

    // 1D input [in_features]
    const x = try ctx.newTensor1d(ggml.Type.f32, in_features);
    fillTensor(x, 123);

    const result = try graph.buildMM(ctx, w, x, null);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Verify shape: [out_features]
    const ne = result.ne();
    try testing.expectEqual(out_features, ne[0]);
    try testing.expectEqual(@as(i64, 1), ne[1]);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildMM: large matrix multiply" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    // Larger dimensions to stress-test
    const out_features: i64 = 32;
    const in_features: i64 = 64;
    const n_pos: i64 = 10;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test_large.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensor(x, 123);

    const result = try graph.buildMM(ctx, w, x, null);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const ne = result.ne();
    try testing.expectEqual(out_features, ne[0]);
    try testing.expectEqual(n_pos, ne[1]);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "buildMM: zero weight matrix" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const out_features: i64 = 4;
    const in_features: i64 = 8;
    const n_pos: i64 = 2;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensorConst(w, 0.0);
    w.setName("test_zero.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensor(x, 123);

    const result = try graph.buildMM(ctx, w, x, null);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Zero weight -> zero output
    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    }
}

test "buildMM: identity-like matrix" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const dim: i64 = 4;
    const n_pos: i64 = 2;

    // Create an identity-like weight matrix [n_out, n_in] = [dim, dim]
    const w = try ctx.newTensor2d(ggml.Type.f32, dim, dim);
    {
        const w_data = w.dataF32();
        @memset(w_data, 0.0);
        for (0..@as(usize, @intCast(dim))) |i| {
            w_data[i * @as(usize, @intCast(dim)) + i] = 1.0;
        }
    }
    w.setName("test_identity.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, dim, n_pos);
    fillTensorSeq(x, 1.0, 1.0); // [1,2,3,4] per column

    const result = try graph.buildMM(ctx, w, x, null);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    // Identity matrix -> output == input
    const out_data = result.dataF32();
    const in_data = x.dataF32();
    for (out_data, in_data) |o, i| {
        try testing.expectApproxEqAbs(i, o, 1e-5);
    }
}

// ============================================================================
// Edge cases and error conditions
// ============================================================================

test "buildNorm: empty weight does not affect buildNorm" {
    // buildNorm doesn't use clamp_map, so this is a sanity check
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 1.0, 1.0);

    const result = try graph.buildNorm(ctx, input, null, null, .rms_norm, 1e-6, "test_norm");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
    }
}

test "buildFFN: empty clamp_map" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const n_features: i64 = 4;
    const n_ff: i64 = 8;
    const n_pos: i64 = 1;
    const input = try ctx.newTensor2d(ggml.Type.f32, n_features, n_pos);
    fillTensorSeq(input, 0.5, 0.5);

    const up_w = try ctx.newTensor2d(ggml.Type.f32, n_ff, n_features);
    fillTensor(up_w, 42);

    const down_w = try ctx.newTensor2d(ggml.Type.f32, n_features, n_ff);
    fillTensor(down_w, 99);

    const result = try graph.buildFFN(ctx, input, up_w, null, null, null, down_w, null, .silu, "test_ffn");

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
    }
}

test "buildMM: null clamp_info" {
    var ctx = try createTestCtx();
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    defer ctx.setNoAlloc(true);

    const out_features: i64 = 4;
    const in_features: i64 = 8;
    const n_pos: i64 = 2;

    const w = try ctx.newTensor2d(ggml.Type.f32, out_features, in_features);
    fillTensor(w, 42);
    w.setName("test.weight");

    const x = try ctx.newTensor2d(ggml.Type.f32, in_features, n_pos);
    fillTensor(x, 123);

    const result = try graph.buildMM(ctx, w, x, null);

    ggml.setOutput(result);
    try computeGraph(ctx, result, 1);

    const out_data = result.dataF32();
    for (out_data) |v| {
        try testing.expect(!std.math.isNan(v));
    }
}
