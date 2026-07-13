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

pub fn resizeBilinear(
    allocator: std.mem.Allocator,
    src: []const u8,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) ![]f32 {
    log.debug("resizeBilinear: {d}x{d} -> {d}x{d}", .{ src_w, src_h, dst_w, dst_h });
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
    log.debug("resizeAndNormalize: START {d}x{d} align={d} min={d} max={d}", .{ img_width, img_height, align_size, min_pixels, max_pixels });
    const size = calcSizePreservedRatio(img_width, img_height, align_size, min_pixels, max_pixels);
    log.debug("resize: {d}x{d} -> {d}x{d}", .{ img_width, img_height, size.w, size.h });

    const resized = try resizeBilinear(allocator, image_data, img_width, img_height, size.w, size.h);
    log.debug("resizeAndNormalize: resizeBilinear done, len={d}", .{resized.len});
    defer allocator.free(resized);

    const wh: usize = @as(usize, @intCast(size.w)) * @as(usize, @intCast(size.h));
    const normalized = try allocator.alloc(f32, wh * 3);
    defer allocator.free(normalized);

    const mr: f64 = @floatCast(mean[0]);
    const mg: f64 = @floatCast(mean[1]);
    const mb: f64 = @floatCast(mean[2]);
    const sr: f64 = @floatCast(std_val[0]);
    const sg: f64 = @floatCast(std_val[1]);
    const sb: f64 = @floatCast(std_val[2]);

    inline for (0..3) |c| {
        const m: f64 = switch (c) {
            0 => mr,
            1 => mg,
            2 => mb,
            else => unreachable,
        };
        const s: f64 = switch (c) {
            0 => sr,
            1 => sg,
            2 => sb,
            else => unreachable,
        };
        for (0..wh) |i| {
            const rf: f64 = @floatCast(resized[c * wh + i]);
            normalized[c * wh + i] = @as(f32, @floatCast(rf / 255.0 - m)) / @as(f32, @floatCast(s));
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
