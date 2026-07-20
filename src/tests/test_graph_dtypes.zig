const std = @import("std");
const testing = std.testing;
const ggml = @import("ggml");
const graph = @import("graph");

const log = std.log.scoped(.test_graph_dtypes);

const TestDType = enum { f32, f16, bf16 };

fn toGgmlType(dtype: TestDType) ggml.Type {
    return switch (dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
    };
}

fn typeSize(dtype: TestDType) usize {
    return switch (dtype) {
        .f32 => 4,
        .f16 => 2,
        .bf16 => 2,
    };
}

fn writeF32AsDType(dest: []u8, value: f32, dtype: TestDType) void {
    switch (dtype) {
        .f32 => {
            const buf = @as([4]u8, @bitCast(value));
            @memcpy(dest[0..4], &buf);
        },
        .f16 => {
            const f16_bits = ggml.c.ggml_fp32_to_fp16(value);
            const buf = @as([2]u8, @bitCast(@as(u16, @intCast(f16_bits))));
            @memcpy(dest[0..2], &buf);
        },
        .bf16 => {
            const bits: u32 = @as(u32, @bitCast(value));
            const high16: u16 = @truncate(bits >> 16);
            const buf = @as([2]u8, @bitCast(high16));
            @memcpy(dest[0..2], &buf);
        },
    }
}

fn readDTypeAsF32(bytes: []const u8, dtype: TestDType) f32 {
    switch (dtype) {
        .f32 => {
            const copy_len = @min(bytes.len, 4);
            var buf: [4]u8 = .{0} ** 4;
            @memcpy(buf[0..copy_len], bytes[0..copy_len]);
            return @as(f32, @bitCast(buf));
        },
        .f16 => {
            const bits: u16 = @as(u16, @bitCast(bytes[0..2].*));
            return ggml.c.ggml_fp16_to_fp32(@as(c_ushort, bits));
        },
        .bf16 => {
            const low16: u16 = @as(u16, @bitCast(bytes[0..2].*));
            const bits: u32 = @as(u32, low16) << 16;
            return @as(f32, @bitCast(bits));
        },
    }
}

fn fillTensorWithValue(tensor: *ggml.Tensor, value: f32, dtype: TestDType) void {
    const elem_size = typeSize(dtype);
    const bytes = tensor.dataBytes();
    var i: usize = 0;
    while (i < bytes.len) : (i += elem_size) {
        writeF32AsDType(bytes[i..], value, dtype);
    }
}

fn readTensorAsF32(tensor: *ggml.Tensor, dtype: TestDType, idx: usize) f32 {
    const elem_size = typeSize(dtype);
    const bytes = tensor.dataBytes();
    const offset = idx * elem_size;
    return readDTypeAsF32(bytes[offset..], dtype);
}

fn checkNoNaNOrInf(tensor: *ggml.Tensor, dtype: TestDType) !void {
    const n_elems = @as(usize, @intCast(tensor.nElems()));
    for (0..n_elems) |i| {
        const val = readTensorAsF32(tensor, dtype, i);
        if (std.math.isNan(val) or std.math.isInf(val)) {
            return error.NaNOrInfDetected;
        }
    }
}

fn runSimpleGraph(
    build_fn: *const fn (ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) anyerror!*ggml.Tensor,
    dtype: TestDType,
) !struct { result: *ggml.Tensor, ctx: *ggml.Context } {
    var ctx = try ggml.Context.init(8 * 1024 * 1024);
    errdefer ctx.deinit();

    var gf = try ggml.CGraph.init(ctx);
    errdefer _ = &gf;

    const result = try build_fn(ctx, gf, dtype);
    gf.compute(-1) catch {};
    return .{ .result = result, .ctx = ctx };
}

fn buildNormGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const ggml_type = toGgmlType(dtype);
    const n_embd: i64 = 32;
    const n_patches: i64 = 8;

    const cur = try ctx.newTensor2d(ggml_type, n_embd, n_patches);
    const mw = try ctx.newTensor1d(ggml_type, n_embd);

    fillTensorWithValue(cur, 1.0, dtype);
    fillTensorWithValue(mw, 1.0, dtype);

    const result = try graph.buildNorm(ctx, cur, mw, null, .layer_norm, 1e-5, -1);
    gf.buildForwardExpand(result);
    return result;
}

fn buildRMSNormGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const ggml_type = toGgmlType(dtype);
    const n_embd: i64 = 32;
    const n_patches: i64 = 8;

    const cur = try ctx.newTensor2d(ggml_type, n_embd, n_patches);
    const mw = try ctx.newTensor1d(ggml_type, n_embd);

    fillTensorWithValue(cur, 1.0, dtype);
    fillTensorWithValue(mw, 1.0, dtype);

    const result = try graph.buildNorm(ctx, cur, mw, null, .rms_norm, 1e-5, -1);
    gf.buildForwardExpand(result);
    return result;
}

test "buildNorm: LayerNorm f32" {
    const r = try runSimpleGraph(buildNormGraph, .f32);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f32);
}

test "buildNorm: LayerNorm f16" {
    const r = try runSimpleGraph(buildNormGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildNorm: LayerNorm bf16" {
    const r = try runSimpleGraph(buildNormGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .bf16);
}

test "buildNorm: RMSNorm f32" {
    const r = try runSimpleGraph(buildRMSNormGraph, .f32);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f32);
}

test "buildNorm: RMSNorm f16" {
    const r = try runSimpleGraph(buildRMSNormGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildNorm: RMSNorm bf16" {
    const r = try runSimpleGraph(buildRMSNormGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .bf16);
}

fn buildFFNGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const ggml_type = toGgmlType(dtype);
    const n_embd: i64 = 32;
    const n_ff: i64 = 64;
    const n_patches: i64 = 8;

    const cur = try ctx.newTensor2d(ggml_type, n_embd, n_patches);
    const up_w = try ctx.newTensor2d(ggml_type, n_ff, n_embd);
    const down_w = try ctx.newTensor2d(ggml_type, n_embd, n_ff);

    fillTensorWithValue(cur, 0.5, dtype);
    fillTensorWithValue(up_w, 0.1, dtype);
    fillTensorWithValue(down_w, 0.1, dtype);

    const result = try graph.buildFFN(ctx, cur, up_w, null, null, null, down_w, null, .silu, -1, graph.defaultBuildMM, null, null);
    gf.buildForwardExpand(result);
    return result;
}

fn buildFFNGateGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const ggml_type = toGgmlType(dtype);
    const n_embd: i64 = 32;
    const n_ff: i64 = 64;
    const n_patches: i64 = 8;

    const cur = try ctx.newTensor2d(ggml_type, n_embd, n_patches);
    const up_w = try ctx.newTensor2d(ggml_type, n_ff, n_embd);
    const gate_w = try ctx.newTensor2d(ggml_type, n_ff, n_embd);
    const down_w = try ctx.newTensor2d(ggml_type, n_embd, n_ff);

    fillTensorWithValue(cur, 0.5, dtype);
    fillTensorWithValue(up_w, 0.1, dtype);
    fillTensorWithValue(gate_w, 0.1, dtype);
    fillTensorWithValue(down_w, 0.1, dtype);

    const result = try graph.buildFFN(ctx, cur, up_w, null, gate_w, null, down_w, null, .gelu, -1, graph.defaultBuildMM, null, null);
    gf.buildForwardExpand(result);
    return result;
}

test "buildFFN: SiLU f32" {
    const r = try runSimpleGraph(buildFFNGraph, .f32);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f32);
}

test "buildFFN: SiLU f16" {
    const r = try runSimpleGraph(buildFFNGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildFFN: SiLU bf16" {
    const r = try runSimpleGraph(buildFFNGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .bf16);
}

test "buildFFN: GELU+Gate f32" {
    const r = try runSimpleGraph(buildFFNGateGraph, .f32);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f32);
}

test "buildFFN: GELU+Gate f16" {
    const r = try runSimpleGraph(buildFFNGateGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildFFN: GELU+Gate bf16" {
    const r = try runSimpleGraph(buildFFNGateGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .bf16);
}

fn buildAttnFlashGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const ggml_type = toGgmlType(dtype);
    const n_embd: i64 = 32;
    const n_head: i64 = 4;
    const d_head: i64 = n_embd / n_head;
    const n_patches: i64 = 8;
    const n_batch: i64 = 1;

    const wo = try ctx.newTensor2d(ggml_type, n_embd, n_embd);
    const q = try ctx.newTensor4d(ggml_type, d_head, n_head, n_patches, n_batch);
    const k = try ctx.newTensor4d(ggml_type, d_head, n_head, n_patches, n_batch);
    const v = try ctx.newTensor4d(ggml_type, d_head, n_head, n_patches, n_batch);

    fillTensorWithValue(wo, 0.1, dtype);
    fillTensorWithValue(q, 0.5, dtype);
    fillTensorWithValue(k, 0.5, dtype);
    fillTensorWithValue(v, 0.5, dtype);

    const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
    const result = try graph.buildAttn(ctx, gf, wo, null, q, k, v, null, kq_scale, -1, null, .enabled, graph.defaultBuildMM, null, null);
    return result;
}

fn buildAttnNonFlashGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const ggml_type = toGgmlType(dtype);
    const n_embd: i64 = 32;
    const n_head: i64 = 4;
    const d_head: i64 = n_embd / n_head;
    const n_patches: i64 = 8;
    const n_batch: i64 = 1;

    const wo = try ctx.newTensor2d(ggml_type, n_embd, n_embd);
    const q = try ctx.newTensor4d(ggml_type, d_head, n_head, n_patches, n_batch);
    const k = try ctx.newTensor4d(ggml_type, d_head, n_head, n_patches, n_batch);
    const v = try ctx.newTensor4d(ggml_type, d_head, n_head, n_patches, n_batch);

    fillTensorWithValue(wo, 0.1, dtype);
    fillTensorWithValue(q, 0.5, dtype);
    fillTensorWithValue(k, 0.5, dtype);
    fillTensorWithValue(v, 0.5, dtype);

    const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
    const result = try graph.buildAttn(ctx, gf, wo, null, q, k, v, null, kq_scale, -1, null, .disabled, graph.defaultBuildMM, null, null);
    return result;
}

test "buildAttn: flash f32" {
    const r = try runSimpleGraph(buildAttnFlashGraph, .f32);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f32);
}

test "buildAttn: flash f16" {
    const r = try runSimpleGraph(buildAttnFlashGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildAttn: flash bf16" {
    const r = try runSimpleGraph(buildAttnFlashGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .bf16);
}

test "buildAttn: non-flash f32" {
    const r = try runSimpleGraph(buildAttnNonFlashGraph, .f32);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f32);
}

test "buildAttn: non-flash f16" {
    const r = try runSimpleGraph(buildAttnNonFlashGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildAttn: non-flash bf16" {
    const r = try runSimpleGraph(buildAttnNonFlashGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .bf16);
}

fn buildRope2DGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const ggml_type = toGgmlType(dtype);
    const d_head: i64 = 32;
    const n_head: i64 = 4;
    const n_patches: i64 = 8;
    const n_batch: i64 = 1;

    const cur = try ctx.newTensor4d(ggml_type, d_head, n_head, n_patches, n_batch);
    fillTensorWithValue(cur, 0.5, dtype);

    const indices = try graph.createPositionIndices(ctx, n_patches, 4);
    const freq_base: f32 = 10000.0;

    const result = try graph.buildRope2D(ctx, cur, indices.pos_x, indices.pos_y, freq_base, false);
    gf.buildForwardExpand(result);
    return result;
}

test "buildRope2D: f32" {
    const r = try runSimpleGraph(buildRope2DGraph, .f32);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 4), r.result.ne()[1]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[2]);
    try testing.expectEqual(@as(i64, 1), r.result.ne()[3]);
    try checkNoNaNOrInf(r.result, .f32);
}

test "buildRope2D: f16" {
    const r = try runSimpleGraph(buildRope2DGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 4), r.result.ne()[1]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[2]);
    try testing.expectEqual(@as(i64, 1), r.result.ne()[3]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildRope2D: bf16" {
    const r = try runSimpleGraph(buildRope2DGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 4), r.result.ne()[1]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[2]);
    try testing.expectEqual(@as(i64, 1), r.result.ne()[3]);
    try checkNoNaNOrInf(r.result, .bf16);
}

fn buildMixedPrecisionGraph(ctx: *ggml.Context, gf: *ggml.CGraph, dtype: TestDType) !*ggml.Tensor {
    const act_type = toGgmlType(dtype);
    const n_embd: i64 = 32;
    const n_patches: i64 = 8;

    const cur = try ctx.newTensor2d(act_type, n_embd, n_patches);
    const mw = try ctx.newTensor1d(.f32, n_embd);

    fillTensorWithValue(cur, 1.0, dtype);
    fillTensorWithValue(mw, 1.0, .f32);

    const result = try graph.buildNorm(ctx, cur, mw, null, .rms_norm, 1e-5, -1);
    gf.buildForwardExpand(result);
    return result;
}

test "buildNorm: mixed f32 weight + f16 act" {
    const r = try runSimpleGraph(buildMixedPrecisionGraph, .f16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .f16);
}

test "buildNorm: mixed f32 weight + bf16 act" {
    const r = try runSimpleGraph(buildMixedPrecisionGraph, .bf16);
    defer r.ctx.deinit();
    try testing.expectEqual(@as(i64, 32), r.result.ne()[0]);
    try testing.expectEqual(@as(i64, 8), r.result.ne()[1]);
    try checkNoNaNOrInf(r.result, .bf16);
}

test "ggml cast: f32 -> f16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.f32, 10);
    {
        const buf = try testing.allocator.alloc(f32, 10);
        defer testing.allocator.free(buf);
        for (0..10) |i| buf[i] = @as(f32, @floatFromInt(i));
        try t.dataSet(f32, buf);
    }

    const t_f16 = ggml.cast(ctx, t, .f16);
    var gf = try ggml.CGraph.init(ctx);
    defer _ = &gf;
    gf.buildForwardExpand(t_f16);
    gf.compute(-1) catch {};

    const data_f16 = t_f16.dataF16();
    for (0..10) |i| {
        const expected: f32 = @floatFromInt(i);
        const actual = ggml.c.ggml_fp16_to_fp32(data_f16[i]);
        try testing.expectApproxEqAbs(expected, actual, 1e-3);
    }
}

test "ggml cast: f32 -> bf16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.f32, 10);
    {
        const buf = try testing.allocator.alloc(f32, 10);
        defer testing.allocator.free(buf);
        for (0..10) |i| buf[i] = @as(f32, @floatFromInt(i));
        try t.dataSet(f32, buf);
    }

    const t_bf16 = ggml.cast(ctx, t, .bf16);
    var gf = try ggml.CGraph.init(ctx);
    defer _ = &gf;
    gf.buildForwardExpand(t_bf16);
    gf.compute(-1) catch {};

    const data_bf16 = t_bf16.dataBF16();
    for (0..10) |i| {
        const expected: f32 = @floatFromInt(i);
        const bf16_val = @as(ggml.c.ggml_bf16_t, @bitCast(data_bf16[i]));
        const actual = ggml.c.ggml_bf16_to_fp32(bf16_val);
        try testing.expectApproxEqAbs(expected, actual, 1e-2);
    }
}

test "ggml cast: f16 -> f32" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.f16, 10);
    {
        const buf = try testing.allocator.alloc(u16, 10);
        defer testing.allocator.free(buf);
        for (0..10) |i| {
            buf[i] = @as(u16, @intCast(ggml.c.ggml_fp32_to_fp16(@as(f32, @floatFromInt(i)))));
        }
        try t.dataSet(u16, buf);
    }

    const t_f32 = ggml.cast(ctx, t, .f32);
    var gf = try ggml.CGraph.init(ctx);
    defer _ = &gf;
    gf.buildForwardExpand(t_f32);
    gf.compute(-1) catch {};

    const data_f32 = t_f32.dataF32();
    for (0..10) |i| {
        const expected: f32 = @floatFromInt(i);
        try testing.expectApproxEqAbs(expected, data_f32[i], 1e-3);
    }
}

test "ggml cast: bf16 -> f32" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.bf16, 10);
    {
        const buf = try testing.allocator.alloc(u16, 10);
        defer testing.allocator.free(buf);
        for (0..10) |i| {
            const f32_val: f32 = @floatFromInt(i);
            const bits: u32 = @as(u32, @bitCast(f32_val));
            buf[i] = @as(u16, @truncate(bits >> 16));
        }
        try t.dataSet(u16, buf);
    }

    const t_f32 = ggml.cast(ctx, t, .f32);
    var gf = try ggml.CGraph.init(ctx);
    defer _ = &gf;
    gf.buildForwardExpand(t_f32);
    gf.compute(-1) catch {};

    const data_f32 = t_f32.dataF32();
    for (0..10) |i| {
        const expected: f32 = @floatFromInt(i);
        try testing.expectApproxEqAbs(expected, data_f32[i], 1e-2);
    }
}

test "tensor: create and fill f16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor2d(.f16, 4, 4);
    try testing.expectEqual(@as(i64, 4), t.ne()[0]);
    try testing.expectEqual(@as(i64, 4), t.ne()[1]);
    try testing.expectEqual(ggml.Type.f16, t.dataType());

    fillTensorWithValue(t, 3.14, .f16);

    const val = readTensorAsF32(t, .f16, 0);
    try testing.expectApproxEqAbs(@as(f32, 3.14), val, 1e-2);
}

test "tensor: create and fill bf16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor2d(.bf16, 4, 4);
    try testing.expectEqual(@as(i64, 4), t.ne()[0]);
    try testing.expectEqual(@as(i64, 4), t.ne()[1]);
    try testing.expectEqual(ggml.Type.bf16, t.dataType());

    fillTensorWithValue(t, 3.14, .bf16);

    const val = readTensorAsF32(t, .bf16, 0);
    try testing.expectApproxEqAbs(@as(f32, 3.14), val, 1e-1);
}

fn computeNMSE(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var mse_ab: f64 = 0.0;
    var mse_a0: f64 = 0.0;
    for (a, b) |av, bv| {
        const af: f64 = @floatCast(av);
        const bf: f64 = @floatCast(bv);
        const diff = af - bf;
        mse_ab += diff * diff;
        mse_a0 += af * af;
    }
    if (mse_a0 == 0.0) return 0.0;
    return mse_ab / mse_a0;
}

fn tensorToF32Slice(tensor: *ggml.Tensor, dtype: TestDType, allocator: std.mem.Allocator) ![]f32 {
    const n = @as(usize, @intCast(tensor.nElems()));
    const result = try allocator.alloc(f32, n);
    for (0..n) |i| {
        result[i] = readTensorAsF32(tensor, dtype, i);
    }
    return result;
}

test "precision: f16 vs f32 for RMSNorm" {
    const allocator = testing.allocator;

    const ref_r = try runSimpleGraph(buildRMSNormGraph, .f32);
    defer ref_r.ctx.deinit();

    const test_r = try runSimpleGraph(buildRMSNormGraph, .f16);
    defer test_r.ctx.deinit();

    const ref_data = try tensorToF32Slice(ref_r.result, .f32, allocator);
    defer allocator.free(ref_data);
    const test_data = try tensorToF32Slice(test_r.result, .f16, allocator);
    defer allocator.free(test_data);

    const nmse_val = computeNMSE(ref_data, test_data);
    try testing.expect(nmse_val < 1e-3);
    log.info("RMSNorm f16 vs f32 NMSE: {d:.6}", .{nmse_val});
}

test "precision: bf16 vs f32 for RMSNorm" {
    const allocator = testing.allocator;

    const ref_r = try runSimpleGraph(buildRMSNormGraph, .f32);
    defer ref_r.ctx.deinit();

    const test_r = try runSimpleGraph(buildRMSNormGraph, .bf16);
    defer test_r.ctx.deinit();

    const ref_data = try tensorToF32Slice(ref_r.result, .f32, allocator);
    defer allocator.free(ref_data);
    const test_data = try tensorToF32Slice(test_r.result, .bf16, allocator);
    defer allocator.free(test_data);

    const nmse_val = computeNMSE(ref_data, test_data);
    try testing.expect(nmse_val < 1e-2);
    log.info("RMSNorm bf16 vs f32 NMSE: {d:.6}", .{nmse_val});
}

test "precision: f16 vs f32 for FFN SiLU" {
    const allocator = testing.allocator;

    const ref_r = try runSimpleGraph(buildFFNGraph, .f32);
    defer ref_r.ctx.deinit();

    const test_r = try runSimpleGraph(buildFFNGraph, .f16);
    defer test_r.ctx.deinit();

    const ref_data = try tensorToF32Slice(ref_r.result, .f32, allocator);
    defer allocator.free(ref_data);
    const test_data = try tensorToF32Slice(test_r.result, .f16, allocator);
    defer allocator.free(test_data);

    const nmse_val = computeNMSE(ref_data, test_data);
    try testing.expect(nmse_val < 1e-3);
    log.info("FFN SiLU f16 vs f32 NMSE: {d:.6}", .{nmse_val});
}

test "precision: bf16 vs f32 for FFN SiLU" {
    const allocator = testing.allocator;

    const ref_r = try runSimpleGraph(buildFFNGraph, .f32);
    defer ref_r.ctx.deinit();

    const test_r = try runSimpleGraph(buildFFNGraph, .bf16);
    defer test_r.ctx.deinit();

    const ref_data = try tensorToF32Slice(ref_r.result, .f32, allocator);
    defer allocator.free(ref_data);
    const test_data = try tensorToF32Slice(test_r.result, .bf16, allocator);
    defer allocator.free(test_data);

    const nmse_val = computeNMSE(ref_data, test_data);
    try testing.expect(nmse_val < 1e-2);
    log.info("FFN SiLU bf16 vs f32 NMSE: {d:.6}", .{nmse_val});
}

test "precision: f16 vs f32 for FlashAttn" {
    const allocator = testing.allocator;

    const ref_r = try runSimpleGraph(buildAttnFlashGraph, .f32);
    defer ref_r.ctx.deinit();

    const test_r = try runSimpleGraph(buildAttnFlashGraph, .f16);
    defer test_r.ctx.deinit();

    const ref_data = try tensorToF32Slice(ref_r.result, .f32, allocator);
    defer allocator.free(ref_data);
    const test_data = try tensorToF32Slice(test_r.result, .f16, allocator);
    defer allocator.free(test_data);

    const nmse_val = computeNMSE(ref_data, test_data);
    try testing.expect(nmse_val < 1e-3);
    log.info("FlashAttn f16 vs f32 NMSE: {d:.6}", .{nmse_val});
}

test "precision: bf16 vs f32 for FlashAttn" {
    const allocator = testing.allocator;

    const ref_r = try runSimpleGraph(buildAttnFlashGraph, .f32);
    defer ref_r.ctx.deinit();

    const test_r = try runSimpleGraph(buildAttnFlashGraph, .bf16);
    defer test_r.ctx.deinit();

    const ref_data = try tensorToF32Slice(ref_r.result, .f32, allocator);
    defer allocator.free(ref_data);
    const test_data = try tensorToF32Slice(test_r.result, .bf16, allocator);
    defer allocator.free(test_data);

    const nmse_val = computeNMSE(ref_data, test_data);
    try testing.expect(nmse_val < 1e-2);
    log.info("FlashAttn bf16 vs f32 NMSE: {d:.6}", .{nmse_val});
}

test "edge: zero tensor f16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor2d(.f16, 4, 4);
    fillTensorWithValue(t, 0.0, .f16);

    for (0..16) |i| {
        const val = readTensorAsF32(t, .f16, i);
        try testing.expectEqual(@as(f32, 0.0), val);
    }
}

test "edge: zero tensor bf16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor2d(.bf16, 4, 4);
    fillTensorWithValue(t, 0.0, .bf16);

    for (0..16) |i| {
        const val = readTensorAsF32(t, .bf16, i);
        try testing.expectEqual(@as(f32, 0.0), val);
    }
}

test "edge: large values f16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.f16, 4);
    fillTensorWithValue(t, 1000.0, .f16);

    const val = readTensorAsF32(t, .f16, 0);
    try testing.expectApproxEqAbs(@as(f32, 1000.0), val, 1.0);
}

test "edge: large values bf16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.bf16, 4);
    fillTensorWithValue(t, 1000.0, .bf16);

    const val = readTensorAsF32(t, .bf16, 0);
    try testing.expectApproxEqAbs(@as(f32, 1000.0), val, 1.0);
}

test "edge: negative values f16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.f16, 4);
    fillTensorWithValue(t, -3.14, .f16);

    const val = readTensorAsF32(t, .f16, 0);
    try testing.expectApproxEqAbs(@as(f32, -3.14), val, 1e-2);
}

test "edge: negative values bf16" {
    var ctx = try ggml.Context.init(1024 * 1024);
    defer ctx.deinit();

    const t = try ctx.newTensor1d(.bf16, 4);
    fillTensorWithValue(t, -3.14, .bf16);

    const val = readTensorAsF32(t, .bf16, 0);
    try testing.expectApproxEqAbs(@as(f32, -3.14), val, 1e-1);
}
