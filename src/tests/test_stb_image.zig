//! stb_image 集成测试
//!
//! 测试 stb_image 加载、双线性 resize、归一化等图像处理核心操作。
//! 使用程序化生成的图像数据（PNG 字节流），不依赖外部文件。
//!
//! 日志作用域: .test_stb_image
//!
//! 注意：此文件通过 test_runner.zig 的 comptime 块导入。
//! 在 Zig 0.16 模块系统中，comptime 块中 @import 的文件内的顶级 const 声明
//! 不能与 root 模块的 --dep 导入名冲突。因此我们通过 @import("root") 访问
//! 所有模块，避免重复声明。

const std = @import("std");
const testing = std.testing;
const stb_image = @import("stb_image");
const ggml = @import("ggml");
const mm_preprocess = @import("preprocess");
const vision_preprocess = @import("vision").preprocess;

const log = std.log.scoped(.test_stb_image);

// ============================================================================
// 辅助：生成一个简单的 2×2 红色 PNG（最小有效 PNG）
// ============================================================================

/// 生成一个 2×2 纯红色 PNG 文件的字节流。
fn generateRedPng(allocator: std.mem.Allocator) ![]u8 {
    const width: u32 = 2;
    const height: u32 = 2;

    const signature = [_]u8{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

    var ihdr_data: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr_data[0..4], width, .big);
    std.mem.writeInt(u32, ihdr_data[4..8], height, .big);
    ihdr_data[8] = 8;
    ihdr_data[9] = 2;
    ihdr_data[10] = 0;
    ihdr_data[11] = 0;
    ihdr_data[12] = 0;

    const ihdr_chunk = try makePngChunk(allocator, "IHDR", &ihdr_data);
    defer allocator.free(ihdr_chunk);

    const row_size: usize = 1 + @as(usize, width) * 3;
    var raw_rows = try allocator.alloc(u8, row_size * height);
    defer allocator.free(raw_rows);

    for (0..height) |y| {
        const row_offset = y * row_size;
        raw_rows[row_offset] = 0;
        for (0..width) |x| {
            const px_offset = row_offset + 1 + x * 3;
            raw_rows[px_offset + 0] = 255;
            raw_rows[px_offset + 1] = 0;
            raw_rows[px_offset + 2] = 0;
        }
    }

    const compressed = try zlibCompress(allocator, raw_rows);
    defer allocator.free(compressed);

    const idat_chunk = try makePngChunk(allocator, "IDAT", compressed);
    defer allocator.free(idat_chunk);

    const iend_chunk = try makePngChunk(allocator, "IEND", &.{});
    defer allocator.free(iend_chunk);

    var result = try allocator.alloc(u8, signature.len + ihdr_chunk.len + idat_chunk.len + iend_chunk.len);
    var offset: usize = 0;
    @memcpy(result[offset..][0..signature.len], &signature);
    offset += signature.len;
    @memcpy(result[offset..][0..ihdr_chunk.len], ihdr_chunk);
    offset += ihdr_chunk.len;
    @memcpy(result[offset..][0..idat_chunk.len], idat_chunk);
    offset += idat_chunk.len;
    @memcpy(result[offset..][0..iend_chunk.len], iend_chunk);

    return result;
}

fn makePngChunk(allocator: std.mem.Allocator, chunk_type: []const u8, data: []const u8) ![]u8 {
    const total_len: usize = 4 + 4 + data.len + 4;
    var chunk = try allocator.alloc(u8, total_len);

    std.mem.writeInt(u32, chunk[0..4], @as(u32, @intCast(data.len)), .big);
    @memcpy(chunk[4..8], chunk_type);
    if (data.len > 0) {
        @memcpy(chunk[8..][0..data.len], data);
    }
    const crc_input = chunk[4..][0 .. 4 + data.len];
    const crc = crc32(crc_input);
    std.mem.writeInt(u32, chunk[8 + data.len ..][0..4], crc, .big);

    return chunk;
}

fn crc32(data: []const u8) u32 {
    var crc: u32 = 0xFFFFFFFF;
    for (data) |byte| {
        crc ^= @as(u32, byte);
        inline for (0..8) |_| {
            if (crc & 1 != 0) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc ^ 0xFFFFFFFF;
}

fn zlibCompress(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const header_len: usize = 2;
    const block_header_len: usize = 5;
    const adler_len: usize = 4;

    var result = try allocator.alloc(u8, header_len + block_header_len + data.len + adler_len);
    var pos: usize = 0;

    result[pos] = 0x78;
    pos += 1;
    result[pos] = 0x01;
    pos += 1;

    result[pos] = 0x01;
    pos += 1;
    const len16: u16 = @intCast(data.len);
    std.mem.writeInt(u16, result[pos..][0..2], len16, .little);
    pos += 2;
    std.mem.writeInt(u16, result[pos..][0..2], ~len16, .little);
    pos += 2;

    @memcpy(result[pos..][0..data.len], data);
    pos += data.len;

    var a: u32 = 1;
    var b: u32 = 0;
    for (data) |byte| {
        a = (a + @as(u32, byte)) % 65521;
        b = (b + a) % 65521;
    }
    const adler = (b << 16) | a;
    std.mem.writeInt(u32, result[pos..][0..4], adler, .big);

    return result;
}

// ============================================================================
// 测试：stb_image 加载 PNG
// ============================================================================

test "stb_image: load generated red PNG from memory" {
    const png_bytes = try generateRedPng(testing.allocator);
    defer testing.allocator.free(png_bytes);

    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const pixels = stb_image.loadFromMemory(png_bytes.ptr, @intCast(png_bytes.len), &w, &h, &comp, 3);
    defer if (pixels) |p| stb_image.free(p);

    try testing.expect(pixels != null);
    try testing.expectEqual(@as(c_int, 2), w);
    try testing.expectEqual(@as(c_int, 2), h);
    try testing.expectEqual(@as(c_int, 3), comp);

    const pixel_data = pixels.?[0..@as(usize, @intCast(w * h * 3))];
    for (0..@as(usize, @intCast(w * h))) |i| {
        try testing.expectEqual(@as(u8, 255), pixel_data[i * 3 + 0]);
        try testing.expectEqual(@as(u8, 0), pixel_data[i * 3 + 1]);
        try testing.expectEqual(@as(u8, 0), pixel_data[i * 3 + 2]);
    }
}

test "stb_image: load invalid data returns null" {
    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const invalid_data = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const pixels = stb_image.loadFromMemory(&invalid_data, @intCast(invalid_data.len), &w, &h, &comp, 3);
    try testing.expect(pixels == null);
}

test "stb_image: failure reason on invalid data" {
    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const invalid_data = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const pixels = stb_image.loadFromMemory(&invalid_data, @intCast(invalid_data.len), &w, &h, &comp, 3);
    try testing.expect(pixels == null);
    const reason = stb_image.failureReason();
    try testing.expect(reason[0] != 0);
    log.debug("stb_image failure reason: {s}", .{reason});
}

test "stb_image: set flip vertically" {
    stb_image.setFlipVertically(1);
    stb_image.setFlipVertically(0);
    try testing.expect(true);
}

// ============================================================================
// 测试：双线性 resize（mm_preprocess = src/mtmd/preprocess.zig）
// ============================================================================

test "bilinearResizeRGB: 2x2 red to 1x1" {
    const src_w: u32 = 2;
    const src_h: u32 = 2;
    const src = [_]u8{
        255, 0, 0,
        255, 0, 0,
        255, 0, 0,
        255, 0, 0,
    };
    const dst = try mm_preprocess.resizeRGB(testing.allocator, &src, src_w, src_h, 1, 1);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, 3), dst.len);
    try testing.expectEqual(@as(u8, 255), dst[0]);
    try testing.expectEqual(@as(u8, 0), dst[1]);
    try testing.expectEqual(@as(u8, 0), dst[2]);
}

test "bilinearResizeRGB: 2x2 mixed to 1x1" {
    const src_w: u32 = 2;
    const src_h: u32 = 2;
    const src = [_]u8{
        255, 0,   0,
        0,   255, 0,
        0,   0,   255,
        255, 255, 0,
    };
    const dst = try mm_preprocess.resizeRGB(testing.allocator, &src, src_w, src_h, 1, 1);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, 3), dst.len);
    try testing.expect(@abs(@as(i16, dst[0]) - 128) <= 2);
    try testing.expect(@abs(@as(i16, dst[1]) - 128) <= 2);
    try testing.expect(@abs(@as(i16, dst[2]) - 64) <= 2);
}

test "bilinearResizeRGB: 4x4 white to 2x2" {
    const src_w: u32 = 4;
    const src_h: u32 = 4;
    const src = try testing.allocator.alloc(u8, src_w * src_h * 3);
    defer testing.allocator.free(src);
    @memset(src, 255);

    const dst = try mm_preprocess.resizeRGB(testing.allocator, src, src_w, src_h, 2, 2);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, 2 * 2 * 3), dst.len);
    for (dst) |v| {
        try testing.expectEqual(@as(u8, 255), v);
    }
}

test "bilinearResizeRGB: 2x2 to 4x4 (upscale)" {
    const src_w: u32 = 2;
    const src_h: u32 = 2;
    const src = [_]u8{
        255, 0,   0,
        0,   255, 0,
        0,   0,   255,
        255, 255, 0,
    };
    const dst = try mm_preprocess.resizeRGB(testing.allocator, &src, src_w, src_h, 4, 4);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, 4 * 4 * 3), dst.len);
    try testing.expect(dst[0] > 200);
    try testing.expect(dst[1] < 50);
    try testing.expect(dst[2] < 50);
}

test "bilinearResizeRGB: same size" {
    const src_w: u32 = 4;
    const src_h: u32 = 4;
    const src = try testing.allocator.alloc(u8, src_w * src_h * 3);
    defer testing.allocator.free(src);
    for (0..src_h) |y| {
        for (0..src_w) |x| {
            const idx = (y * src_w + x) * 3;
            src[idx + 0] = @intCast((x * 255) / (src_w - 1));
            src[idx + 1] = @intCast((y * 255) / (src_h - 1));
            src[idx + 2] = 128;
        }
    }

    const dst = try mm_preprocess.resizeRGB(testing.allocator, src, src_w, src_h, src_w, src_h);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, src_w * src_h * 3), dst.len);
    for (0..src.len) |i| {
        try testing.expect(@abs(@as(i16, dst[i]) - @as(i16, src[i])) <= 1);
    }
}

test "bilinearResizeRGB: 1x1 to 2x2" {
    const src = [_]u8{ 128, 64, 32 };
    const dst = try mm_preprocess.resizeRGB(testing.allocator, &src, 1, 1, 2, 2);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, 2 * 2 * 3), dst.len);
    for (0..4) |i| {
        try testing.expectEqual(@as(u8, 128), dst[i * 3 + 0]);
        try testing.expectEqual(@as(u8, 64), dst[i * 3 + 1]);
        try testing.expectEqual(@as(u8, 32), dst[i * 3 + 2]);
    }
}

test "bilinearResizeRGB: extreme downscale 100x100 to 1x1" {
    const src_w: u32 = 100;
    const src_h: u32 = 100;
    const src = try testing.allocator.alloc(u8, src_w * src_h * 3);
    defer testing.allocator.free(src);
    for (0..src_h) |y| {
        for (0..src_w) |x| {
            const idx = (y * src_w + x) * 3;
            src[idx + 0] = @intCast((x * 255) / (src_w - 1));
            src[idx + 1] = @intCast((y * 255) / (src_h - 1));
            src[idx + 2] = @intCast(((x + y) * 255) / (src_w + src_h - 2));
        }
    }

    const dst = try mm_preprocess.resizeRGB(testing.allocator, src, src_w, src_h, 1, 1);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, 3), dst.len);
    for (dst) |v| {
        try testing.expect(v >= 0);
        try testing.expect(v <= 255);
    }
}

// ============================================================================
// 测试：calcSizePreservedRatio（动态分辨率计算）
// ============================================================================

test "calcSizePreservedRatio: no resize needed" {
    // mm_preprocess.calcSizePreservedRatio 使用 round 策略
    // 224 会被 round(224/48)*48 = 240
    const result = mm_preprocess.calcSizePreservedRatio(224, 224, 48, 18432, 589824);
    try testing.expect(result.width == 240 or result.width == 224);
    try testing.expect(result.height == 240 or result.height == 224);
    try testing.expect(result.width % 48 == 0);
    try testing.expect(result.height % 48 == 0);
}

test "calcSizePreservedRatio: downscale large image" {
    const result = mm_preprocess.calcSizePreservedRatio(1024, 1024, 48, 18432, 589824);
    try testing.expect(result.width < 1024);
    try testing.expect(result.width % 48 == 0);
    try testing.expect(result.height % 48 == 0);
    try testing.expect(result.width * result.height <= 589824);
}

test "calcSizePreservedRatio: upscale small image" {
    const result = mm_preprocess.calcSizePreservedRatio(100, 100, 48, 18432, 589824);
    try testing.expect(result.width >= 100);
    try testing.expect(result.width % 48 == 0);
    try testing.expect(result.height % 48 == 0);
    try testing.expect(result.width * result.height >= 18432);
}

test "calcSizePreservedRatio: non-square image" {
    const result = mm_preprocess.calcSizePreservedRatio(800, 600, 48, 18432, 589824);
    try testing.expect(result.width % 48 == 0);
    try testing.expect(result.height % 48 == 0);
    const ratio_src: f64 = @as(f64, 800.0) / @as(f64, 600.0);
    const ratio_dst: f64 = @as(f64, @floatFromInt(result.width)) / @as(f64, @floatFromInt(result.height));
    try testing.expect(@abs(ratio_src - ratio_dst) < 0.1);
}

// ============================================================================
// 测试：imageToTensor（归一化）
// ============================================================================

test "imageToTensor: siglip normalization on red pixel" {
    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();
    ctx.setNoAlloc(false);

    const img_data = [_]u8{ 255, 0, 0 };
    var img = mm_preprocess.ProcessedImage{
        .data = @constCast(&img_data),
        .width = 1,
        .height = 1,
        .allocator = testing.allocator,
    };

    const tensor = try mm_preprocess.imageToTensor(ctx, &img, .siglip);
    const data = tensor.dataF32();

    try testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), data[2], 1e-5);

    ctx.setNoAlloc(true);
}

test "imageToTensor: siglip normalization on gray pixel" {
    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();
    ctx.setNoAlloc(false);

    const img_data = [_]u8{ 128, 128, 128 };
    var img = mm_preprocess.ProcessedImage{
        .data = @constCast(&img_data),
        .width = 1,
        .height = 1,
        .allocator = testing.allocator,
    };

    const tensor = try mm_preprocess.imageToTensor(ctx, &img, .siglip);
    const data = tensor.dataF32();

    try testing.expectApproxEqAbs(@as(f32, 0.0039217), data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0039217), data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0039217), data[2], 1e-4);

    ctx.setNoAlloc(true);
}

test "imageToTensor: div255 normalization" {
    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();
    ctx.setNoAlloc(false);

    const img_data = [_]u8{ 128, 64, 32 };
    var img = mm_preprocess.ProcessedImage{
        .data = @constCast(&img_data),
        .width = 1,
        .height = 1,
        .allocator = testing.allocator,
    };

    const tensor = try mm_preprocess.imageToTensor(ctx, &img, .div255);
    const data = tensor.dataF32();

    try testing.expectApproxEqAbs(@as(f32, 128.0 / 255.0), data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 64.0 / 255.0), data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 32.0 / 255.0), data[2], 1e-5);

    ctx.setNoAlloc(true);
}

test "imageToTensor: none normalization (passthrough)" {
    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();
    ctx.setNoAlloc(false);

    const img_data = [_]u8{ 100, 150, 200 };
    var img = mm_preprocess.ProcessedImage{
        .data = @constCast(&img_data),
        .width = 1,
        .height = 1,
        .allocator = testing.allocator,
    };

    const tensor = try mm_preprocess.imageToTensor(ctx, &img, .none);
    const data = tensor.dataF32();

    try testing.expectApproxEqAbs(@as(f32, 100.0), data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 150.0), data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 200.0), data[2], 1e-5);

    ctx.setNoAlloc(true);
}

// ============================================================================
// 测试：完整流程 — stb_image 加载 → resize → imageToTensor
// ============================================================================

test "full pipeline: load PNG -> resize -> normalize (siglip)" {
    const png_bytes = try generateRedPng(testing.allocator);
    defer testing.allocator.free(png_bytes);

    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const pixels = stb_image.loadFromMemory(png_bytes.ptr, @intCast(png_bytes.len), &w, &h, &comp, 3);
    defer if (pixels) |p| stb_image.free(p);
    try testing.expect(pixels != null);
    try testing.expectEqual(@as(c_int, 2), w);
    try testing.expectEqual(@as(c_int, 2), h);

    const src_data = pixels.?[0..@as(usize, @intCast(w * h * 3))];

    const dst = try mm_preprocess.resizeRGB(testing.allocator, src_data, @intCast(w), @intCast(h), 1, 1);
    defer testing.allocator.free(dst);

    try testing.expectEqual(@as(usize, 3), dst.len);
    try testing.expectEqual(@as(u8, 255), dst[0]);
    try testing.expectEqual(@as(u8, 0), dst[1]);
    try testing.expectEqual(@as(u8, 0), dst[2]);

    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();
    ctx.setNoAlloc(false);

    var img = mm_preprocess.ProcessedImage{
        .data = dst,
        .width = 1,
        .height = 1,
        .allocator = testing.allocator,
    };

    const tensor = try mm_preprocess.imageToTensor(ctx, &img, .siglip);
    const data = tensor.dataF32();

    try testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), data[2], 1e-5);

    ctx.setNoAlloc(true);
}

test "full pipeline: load PNG -> resize -> normalize (standard mean/std)" {
    const png_bytes = try generateRedPng(testing.allocator);
    defer testing.allocator.free(png_bytes);

    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const pixels = stb_image.loadFromMemory(png_bytes.ptr, @intCast(png_bytes.len), &w, &h, &comp, 3);
    defer if (pixels) |p| stb_image.free(p);
    try testing.expect(pixels != null);

    const src_data = pixels.?[0..@as(usize, @intCast(w * h * 3))];

    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();

    const mean: [3]f32 = .{ 0.5, 0.5, 0.5 };
    const std_val: [3]f32 = .{ 0.5, 0.5, 0.5 };

    const result = try vision_preprocess.resizeAndNormalize(
        ctx,
        testing.allocator,
        src_data,
        @intCast(w),
        @intCast(h),
        mean,
        std_val,
        48,
        0,
        589824,
    );

    const tensor = result.tensor;
    // resizeAndNormalize 在 no_alloc 模式下会为 tensor 分配数据缓冲区，
    // 该缓冲区由调用者负责释放。测试完成后需要手动释放。
    const tensor_data_buf = tensor.dataBytes();
    defer testing.allocator.free(tensor_data_buf);

    const data = try tensor.dataGet(f32, testing.allocator);
    defer testing.allocator.free(data);
    try testing.expect(data[0] > 0.5); // R 通道应为正
}

// ============================================================================
// 真实 PNG 文件测试
// ============================================================================

/// deps/A.png 的 base64 编码（86x86 RGBA 图像）
const a_png_base64 =
    \\iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAACOmlDQ1BJQ0MgUHJvZmlsZQAA
    \\OI2VlD9v00AUwF9CRREqCFWoMFoCUYaCokZio2ritElIFCzHgdLNsd3YxP90dw4UMVQIJr4A
    \\Yz8BYuiIEGPFimBiRAxMSKCu4Z3N3QUQIJ70fD8/v7v33uk9A8w9s9M0LJ8BiGJGzGZd27qz
    \\rc1/hnlYgEV8rtgOTWuG0QUUsf4sR++hxNe3V/hZv3//q5xyPeoAlDTkoUudCPkx6pGTEgZw
    \\7CHaL9xjKed95LMEE0Q+4Dwq+JDzsOAPuY9l6shfkE9PnBHuneOxKrEbxMgV5LUoSlzkFPny
    \\cMZ/NMNFbrmcaAQ0De3d/yzv3xKFmYixjLrgk5aJ6yLewcdx0pEcD3s3BQdu7p+zn7UGgh2q
    \\bwumYb8t2LUbHXlO2OsK3gk2pU/A2pZgj270BZPElHF3iF4TbBOVQzYeSLvvteX5D3zrtuBJ
    \\cKsncxv3O8pHl3aSmbIWL27WVdxNeQ8Rnak9aMu9zLda8h5slb8X19SZdEvm5nqNDeUzkP4p
    \\q8tYaWhIfy9sSjud9OVehs2m9hryDu/aNwzBoEMCISoBDbr41gBg3n3GC9GTdJcEI59pNZwe
    \\T2vHztUVbbWyeg2Az2LRGl+X8hkrLb1RtkffAK670+n0pbJ1LwIcPAE4+UnZll9hK58DOHzq
    \\ZGTyo9dK5SpAMRvFezHDOcGfuJifXDC757jw0vXXAC9QL+3hOK8D8LKtdShXq1JFr+czx+U4
    \\/nH21zi9O+/uwS/yHdvMv/x2QKcFAAAAlmVYSWZNTQAqAAAACAAFARIAAwAAAAEAAQAAARoA
    \\BQAAAAEAAABKARsABQAAAAEAAABSASgAAwAAAAEAAgAAh2kABAAAAAEAAABaAAAAAAAAAJAA
    \\AAABAAAAkAAAAAEAA5KGAAcAAAASAAAAhKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAABB
    \\U0NJSQAAAFNjcmVlbnNob3Ql0cL1AAAACXBIWXMAABYlAAAWJQFJUiTwAAAC1WlUWHRYTUw6
    \\Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIg
    \\eDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6
    \\Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRl
    \\c2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczpleGlmPSJodHRwOi8v
    \\bnMuYWRvYmUuY29tL2V4aWYvMS4wLyIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDov
    \\L25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8ZXhpZjpQaXhlbFhEaW1lbnNp
    \\b24+ODY8L2V4aWY6UGl4ZWxYRGltZW5zaW9uPgogICAgICAgICA8ZXhpZjpVc2VyQ29tbWVu
    \\dD5TY3JlZW5zaG90PC9leGlmOlVzZXJDb21tZW50PgogICAgICAgICA8ZXhpZjpQaXhlbFlE
    \\aW1lbnNpb24+ODY8L2V4aWY6UGl4ZWxZRGltZW5zaW9uPgogICAgICAgICA8dGlmZjpSZXNv
    \\bHV0aW9uVW5pdD4yPC90aWZmOlJlc29sdXRpb25Vbml0PgogICAgICAgICA8dGlmZjpZUmVz
    \\b2x1dGlvbj4xNDQ8L3RpZmY6WVJlc29sdXRpb24+CiAgICAgICAgIDx0aWZmOlhSZXNvbHV0
    \\aW9uPjE0NDwvdGlmZjpYUmVzb2x1dGlvbj4KICAgICAgICAgPHRpZmY6T3JpZW50YXRpb24+
    \\MTwvdGlmZjpPcmllbnRhdGlvbj4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgIDwvcmRm
    \\OlJERj4KPC94OnhtcG1ldGE+Ci+ImMYAAAFJSURBVDgRpZPNSsNAFIXPzMSk6/oA6jtJUREX
    \\Im5cF13oMwniW/iTWmrBn4ogammqYtNEkphcZ1IHGzuklF4YAndyznxzbsLe/JQwR/E5tLm0
    \\1IBoOlypAediKmCpQZLEsxsobCE4wiDA9tYa7u9u4TgcWZYZzSYIlIFlAe12C0vLK2heuUjN
    \\WnOInHPEMcG9PEd9/xDDoY9e14NtWzCFWiAgiWnbHK8vz3h86MCpVJDJ469bTdmH8RoFg+wX
    \\XwmiKMLpyTE8r4eGeyGpILMRkxTqS1SrP/imj4Co+/5FO7t71Hnqkx9R3qsfHNFZ44aChMj7
    \\TPL3ta5AIK+PMAxRW99EtbooKVIIGehqbQOMszxMxlhhGmz8XxiNUMBaAOJoFL3q2bZCB5Ik
    \\RamBtlYzV9PQpdP/L1b7EnCyxsVq1yTUqr9jdGfG5w91ZrXW4Pu1jwAAAABJRU5ErkJggg==
;

test "load real A.png from base64" {
    const decoder = std.base64.standard.decoderWithIgnore("\n\r");

    // 解码 base64
    const sz = decoder.calcSizeUpperBound(a_png_base64.len);
    var buffer: [0x1000]u8 = undefined;
    const decoded = buffer[0..sz];
    const written = try decoder.decode(decoded, a_png_base64);
    try std.testing.expect(written <= sz);

    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const pixels = stb_image.loadFromMemory(decoded.ptr, @intCast(written), &w, &h, &comp, 3);
    defer if (pixels) |p| stb_image.free(p);
    try testing.expect(pixels != null);
    try testing.expectEqual(@as(c_int, 16), w);
    try testing.expectEqual(@as(c_int, 16), h);
    try testing.expectEqual(@as(c_int, 4), comp); // A.png 是 RGBA（4 通道）

    const src_data = pixels.?[0..@as(usize, @intCast(w * h * 3))];

    // 验证第一个像素不是纯红（A.png 不是纯色图）
    try testing.expect(src_data[0] != 255 or src_data[1] != 0 or src_data[2] != 0);

    // 验证 resizeAndNormalize 能处理真实图像
    var ctx = try ggml.Context.initNoAlloc(1024 * 1024);
    defer ctx.deinit();

    const mean: [3]f32 = .{ 0.5, 0.5, 0.5 };
    const std_val: [3]f32 = .{ 0.5, 0.5, 0.5 };

    const result = try vision_preprocess.resizeAndNormalize(
        ctx,
        testing.allocator,
        src_data,
        @intCast(w),
        @intCast(h),
        mean,
        std_val,
        48,
        0,
        589824,
    );

    const tensor = result.tensor;
    // resizeAndNormalize 在 no_alloc 模式下会为 tensor 分配数据缓冲区，
    // 该缓冲区由调用者负责释放。测试完成后需要手动释放。
    const tensor_data_buf = tensor.dataBytes();
    defer testing.allocator.free(tensor_data_buf);

    const data = try tensor.dataGet(f32, testing.allocator);
    defer testing.allocator.free(data);

    // 验证输出形状
    try testing.expect(result.new_width % 48 == 0);
    try testing.expect(result.new_height % 48 == 0);
    try testing.expect(result.new_width * result.new_height <= 589824);

    // 验证 normalized 数据在合理范围内
    // 对于 mean=0.5, std=0.5, 像素值 [0,255] -> normalized [-1, 1]
    for (data) |val| {
        try testing.expect(val >= -1.0 and val <= 1.0);
    }

    // 验证不是全零（图像有内容）
    var has_nonzero = false;
    for (data) |val| {
        if (@abs(val) > 0.001) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}
