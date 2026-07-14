//! Image preprocessing.
//!
//! Provides normalization, dynamic resize, and ggml tensor creation.
//! Ref: llama.cpp tools/mtmd/clip.cpp, mtmd-image.cpp

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.vision_preprocess);

pub const NormalizeMode = enum { standard, siglip, passthrough };

pub fn normalizeToTensor(
    ctx: *ggml.Context,
    image_data: []const u8,
    img_width: u32,
    img_height: u32,
    mean: [3]f32,
    std_val: [3]f32,
    mode: NormalizeMode,
) !*ggml.Tensor {
    const W: usize = @intCast(img_width);
    const H: usize = @intCast(img_height);
    const wh: usize = W * H;

    var inp = try ctx.newTensor3d(ggml.Type.f32, @intCast(img_width), @intCast(img_height), 3);
    inp.setName("inp_raw");

    const no_alloc = ctx.getNoAlloc();
    if (no_alloc) {
        const ds = @as(usize, @intCast(inp.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(ds) orelse return error.OutOfMemory))[0..ds];
        @memset(buf, 0);
        inp.setDataPtr(buf);
    }

    const n_elems = @as(usize, @intCast(inp.nElems()));
    const dst = try std.heap.page_allocator.alloc(f32, n_elems);
    defer std.heap.page_allocator.free(dst);

    switch (mode) {
        .standard => {
            const mr = mean[0];
            const mg = mean[1];
            const mb = mean[2];
            const sr = std_val[0];
            const sg = std_val[1];
            const sb = std_val[2];
            for (0..H) |y| {
                for (0..W) |x| {
                    const si = (y * W + x) * 3;
                    const db = y * W + x;
                    dst[db] = (@as(f32, @floatFromInt(image_data[si + 0])) / 255.0 - mr) / sr;
                    dst[db + wh] = (@as(f32, @floatFromInt(image_data[si + 1])) / 255.0 - mg) / sg;
                    dst[db + 2 * wh] = (@as(f32, @floatFromInt(image_data[si + 2])) / 255.0 - mb) / sb;
                }
            }
        },
        .siglip => {
            for (0..H) |y| {
                for (0..W) |x| {
                    const si = (y * W + x) * 3;
                    const db = y * W + x;
                    dst[db] = @as(f32, @floatFromInt(image_data[si + 0])) / 255.0 * 2.0 - 1.0;
                    dst[db + wh] = @as(f32, @floatFromInt(image_data[si + 1])) / 255.0 * 2.0 - 1.0;
                    dst[db + 2 * wh] = @as(f32, @floatFromInt(image_data[si + 2])) / 255.0 * 2.0 - 1.0;
                }
            }
        },
        .passthrough => {
            for (0..H) |y| {
                for (0..W) |x| {
                    const si = (y * W + x) * 3;
                    const db = y * W + x;
                    dst[db] = @floatFromInt(image_data[si + 0]);
                    dst[db + wh] = @floatFromInt(image_data[si + 1]);
                    dst[db + 2 * wh] = @floatFromInt(image_data[si + 2]);
                }
            }
        },
    }
    try inp.dataSet(f32, dst);
    return inp;
}

pub fn calcSizePreservedRatio(
    width: u32,
    height: u32,
    align_size: u32,
    min_pixels: u32,
    max_pixels: u32,
) struct { w: u32, h: u32 } {
    log.debug("calcSizePreservedRatio: input {d}x{d}, align={d}, min={d}, max={d}", .{ width, height, align_size, min_pixels, max_pixels });
    const fa: f64 = @floatFromInt(align_size);
    const round = struct {
        fn r(v: f64, f: f64) u32 {
            return @intFromFloat(@round(v / f) * f);
        }
    }.r;
    const floor = struct {
        fn f(v: f64, f2: f64) u32 {
            return @intFromFloat(@floor(v / f2) * f2);
        }
    }.f;
    const ceil = struct {
        fn c(v: f64, f2: f64) u32 {
            return @intFromFloat(@ceil(v / f2) * f2);
        }
    }.c;

    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    var hb: u32 = @max(align_size, round(h, fa));
    var wb: u32 = @max(align_size, round(w, fa));

    log.debug("calcSizePreservedRatio: initial round hb={d} wb={d} product={d}", .{ hb, wb, hb * wb });

    if (hb * wb > max_pixels) {
        log.debug("calcSizePreservedRatio: too large, downscaling...", .{});
        const beta: f64 = @sqrt((h * w) / @as(f64, @floatFromInt(max_pixels)));
        hb = @max(align_size, floor(h / beta, fa));
        wb = @max(align_size, floor(w / beta, fa));
    } else if (hb * wb < min_pixels) {
        log.debug("calcSizePreservedRatio: too small, upscaling...", .{});
        const beta: f64 = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (h * w));
        hb = ceil(h * beta, fa);
        wb = ceil(w * beta, fa);
    }
    log.debug("calcSizePreservedRatio: result {d}x{d} product={d}", .{ wb, hb, wb * hb });
    return .{ .w = wb, .h = hb };
}

/// Bilinear resize that outputs u8 (matching llama.cpp behavior exactly).
pub fn resizeBilinearU8(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) ![]u8 {
    log.debug("resizeBilinearU8: {d}x{d} -> {d}x{d}", .{ src_w, src_h, dst_w, dst_h });
    const sw: i32 = @intCast(src_w);
    const sh: i32 = @intCast(src_h);
    const dw: i32 = @intCast(dst_w);
    const dh: i32 = @intCast(dst_h);
    const out: []u8 = try allocator.alloc(u8, @as(usize, @intCast(dw * dh * 3)));

    const x_ratio: f64 = if (dw > 1) @as(f64, @floatFromInt(sw - 1)) / @as(f64, @floatFromInt(dw - 1)) else 0.0;
    const y_ratio: f64 = if (dh > 1) @as(f64, @floatFromInt(sh - 1)) / @as(f64, @floatFromInt(dh - 1)) else 0.0;

    for (0..@as(usize, @intCast(dh))) |dy| {
        const py: f64 = @as(f64, @floatFromInt(@as(i32, @intCast(dy)))) * y_ratio;
        const y0: i32 = @min(@as(i32, @intFromFloat(@floor(py))), sh - 1);
        const y1: i32 = @min(y0 + 1, sh - 1);
        const yf: f64 = py - @floor(py);
        for (0..@as(usize, @intCast(dw))) |dx| {
            const px: f64 = @as(f64, @floatFromInt(@as(i32, @intCast(dx)))) * x_ratio;
            const x0: i32 = @min(@as(i32, @intFromFloat(@floor(px))), sw - 1);
            const x1: i32 = @min(x0 + 1, sw - 1);
            const xf: f64 = px - @floor(px);

            const idx00: usize = @as(usize, @intCast((y0 * sw + x0) * 3));
            const idx01: usize = @as(usize, @intCast((y0 * sw + x1) * 3));
            const idx10: usize = @as(usize, @intCast((y1 * sw + x0) * 3));
            const idx11: usize = @as(usize, @intCast((y1 * sw + x1) * 3));
            const db: usize = (dy * @as(usize, @intCast(dw)) + dx) * 3;

            inline for (0..3) |c| {
                const v00: f64 = @floatFromInt(src[idx00 + c]);
                const v01: f64 = @floatFromInt(src[idx01 + c]);
                const v10: f64 = @floatFromInt(src[idx10 + c]);
                const v11: f64 = @floatFromInt(src[idx11 + c]);
                const top: f64 = v00 * (1.0 - xf) + v01 * xf;
                const bot: f64 = v10 * (1.0 - xf) + v11 * xf;
                const val: f64 = top * (1.0 - yf) + bot * yf;
                out[db + c] = @intFromFloat(val);
            }
        }
    }
    return out;
}

/// Legacy: resize to f32 in CHW layout.
pub fn resizeBilinear(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) ![]f32 {
    log.debug("resizeBilinear (legacy): {d}x{d} -> {d}x{d}", .{ src_w, src_h, dst_w, dst_h });
    const sw: usize = @intCast(src_w);
    const sh: usize = @intCast(src_h);
    const dw: usize = @intCast(dst_w);
    const dh: usize = @intCast(dst_h);
    const out: []f32 = try allocator.alloc(f32, dw * dh * 3);

    const rx: f64 = if (dw > 1) @as(f64, @floatFromInt(sw - 1)) / @as(f64, @floatFromInt(dw - 1)) else 0.0;
    const ry: f64 = if (dh > 1) @as(f64, @floatFromInt(sh - 1)) / @as(f64, @floatFromInt(dh - 1)) else 0.0;

    for (0..dh) |dy| {
        const syf: f64 = @as(f64, @floatFromInt(dy)) * ry;
        const sy0: usize = @intFromFloat(@floor(syf));
        const sy1: usize = @min(sy0 + 1, sh - 1);
        const fy: f64 = syf - @as(f64, @floatFromInt(sy0));
        for (0..dw) |dx| {
            const sxf: f64 = @as(f64, @floatFromInt(dx)) * rx;
            const sx0: usize = @intFromFloat(@floor(sxf));
            const sx1: usize = @min(sx0 + 1, sw - 1);
            const fx: f64 = sxf - @as(f64, @floatFromInt(sx0));

            const idx00: usize = (sy0 * sw + sx0) * 3;
            const idx01: usize = (sy0 * sw + sx1) * 3;
            const idx10: usize = (sy1 * sw + sx0) * 3;
            const idx11: usize = (sy1 * sw + sx1) * 3;
            const db: usize = dy * dw + dx;
            const wh2: usize = dw * dh;

            inline for (0..3) |c| {
                const v00: f64 = @floatFromInt(src[idx00 + c]);
                const v01: f64 = @floatFromInt(src[idx01 + c]);
                const v10: f64 = @floatFromInt(src[idx10 + c]);
                const v11: f64 = @floatFromInt(src[idx11 + c]);
                const top: f64 = v00 * (1.0 - fx) + v01 * fx;
                const bot: f64 = v10 * (1.0 - fx) + v11 * fx;
                out[c * wh2 + db] = @floatCast(top * (1.0 - fy) + bot * fy);
            }
        }
    }
    return out;
}

/// Resize then normalize to f32 CHW.
pub fn resizeAndNormalize(
    ctx: *ggml.Context,
    allocator: std.mem.Allocator,
    image_data: []const u8,
    img_width: u32,
    img_height: u32,
    mean: [3]f32,
    std_val: [3]f32,
    align_size: u32,
    min_pixels: u32,
    max_pixels: u32,
) !struct { tensor: *ggml.Tensor, new_width: u32, new_height: u32 } {
    log.warn("resizeAndNormalize: START {d}x{d} align={d} min={d} max={d}", .{ img_width, img_height, align_size, min_pixels, max_pixels });
    log.warn("resizeAndNormalize: image_data[0..5] = {d}, {d}, {d}, {d}, {d}", .{ image_data[0], image_data[1], image_data[2], image_data[3], image_data[4] });

    const size = calcSizePreservedRatio(img_width, img_height, align_size, min_pixels, max_pixels);
    log.warn("resize: {d}x{d} -> {d}x{d}", .{ img_width, img_height, size.w, size.h });

    const resized_u8 = try resizeBilinearU8(allocator, image_data, img_width, img_height, size.w, size.h);
    defer allocator.free(resized_u8);
    log.warn("resizeAndNormalize: resized_u8[0..5] = {d}, {d}, {d}, {d}, {d}", .{ resized_u8[0], resized_u8[1], resized_u8[2], resized_u8[3], resized_u8[4] });

    const wh: usize = @as(usize, @intCast(size.w)) * @as(usize, @intCast(size.h));
    const normalized = try allocator.alloc(f32, wh * 3);
    defer allocator.free(normalized);

    const mr: f64 = @floatCast(mean[0]);
    const mg: f64 = @floatCast(mean[1]);
    const mb: f64 = @floatCast(mean[2]);
    const sr: f64 = @floatCast(std_val[0]);
    const sg: f64 = @floatCast(std_val[1]);
    const sb: f64 = @floatCast(std_val[2]);

    for (0..@as(usize, @intCast(size.h))) |y| {
        for (0..@as(usize, @intCast(size.w))) |x| {
            const hwc_idx = (y * @as(usize, @intCast(size.w)) + x) * 3;
            const chw_idx = y * @as(usize, @intCast(size.w)) + x;
            const r_u8: f64 = @floatFromInt(resized_u8[hwc_idx + 0]);
            const g_u8: f64 = @floatFromInt(resized_u8[hwc_idx + 1]);
            const b_u8: f64 = @floatFromInt(resized_u8[hwc_idx + 2]);
            normalized[chw_idx]          = @as(f32, @floatCast((r_u8 / 255.0 - mr) / sr));
            normalized[chw_idx + wh]     = @as(f32, @floatCast((g_u8 / 255.0 - mg) / sg));
            normalized[chw_idx + 2 * wh] = @as(f32, @floatCast((b_u8 / 255.0 - mb) / sb));
        }
    }

    var inp = try ctx.newTensor3d(ggml.Type.f32, @intCast(size.w), @intCast(size.h), 3);
    inp.setName("inp_raw");

    const no_alloc = ctx.getNoAlloc();
    if (no_alloc) {
        const ds = @as(usize, @intCast(inp.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(ds) orelse return error.OutOfMemory))[0..ds];
        @memset(buf, 0);
        inp.setDataPtr(buf);
    }
    try inp.dataSet(f32, normalized);
    return .{ .tensor = inp, .new_width = size.w, .new_height = size.h };
}

test "calcSizePreservedRatio: no resize" {
    const r = calcSizePreservedRatio(224, 224, 48, 18432, 589824);
    try std.testing.expectEqual(@as(u32, 224), r.w);
}

test "calcSizePreservedRatio: downscale" {
    const r = calcSizePreservedRatio(1024, 1024, 48, 18432, 589824);
    try std.testing.expect(r.w < 1024);
    try std.testing.expect(r.w % 48 == 0);
}


test "resizeBilinear: downscale" {
    const src = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128, 0, 0, 0, 255, 255, 255, 64, 64, 64, 192, 192, 192 };
    const out = try resizeBilinear(std.testing.allocator, &src, 4, 2, 2, 1);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqual(@as(usize, 6), out.len);
}

test "resizeAndNormalize: basic" {
    const allocator = std.testing.allocator;
    const src_w: u32 = 4;
    const src_h: u32 = 4;
    var image_data: [48]u8 = [_]u8{128} ** 48;

    const ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();

    const mean: [3]f32 = .{ 0.5, 0.5, 0.5 };
    const std_val: [3]f32 = .{ 0.5, 0.5, 0.5 };

    const result = try resizeAndNormalize(ctx, allocator, &image_data, src_w, src_h, mean, std_val, 48, 0, 589824);
    try std.testing.expectEqual(@as(u32, 48), result.new_width);
    try std.testing.expectEqual(@as(u32, 48), result.new_height);

    const tensor = result.tensor;
    const data = try tensor.dataGet(f32, allocator);
    defer allocator.free(data);

    const expected: f32 = (128.0 / 255.0 - 0.5) / 0.5;
    try std.testing.expectApproxEqAbs(expected, data[0], 1e-5);
    try std.testing.expect(data[0] < 0.5);
}
