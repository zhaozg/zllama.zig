//! Tests for the mtmd multi-modal module
const std = @import("std");
const testing = std.testing;
const model = @import("model");
const mtmd = @import("mtmd");

test "Bitmap: image creation" {
    const bm = mtmd.Bitmap.initImage(224, 224, null);
    try testing.expectEqual(@as(u32, 224), bm.nx);
    try testing.expectEqual(@as(u32, 224), bm.ny);
    try testing.expect(!bm.is_audio);
    try testing.expect(bm.isPlaceholder());
}

test "ImageTokens: normal" {
    const img = mtmd.ImageTokens{ .nx = 7, .ny = 7, .pos = .normal };
    try testing.expectEqual(@as(u32, 49), img.nTokens());
}

test "InputChunk: text" {
    const tokens = [_]i32{ 1, 2, 3 };
    const chunk = mtmd.InputChunk{ .chunk_type = .text, .tokens_text = &tokens };
    try testing.expectEqual(@as(u32, 3), chunk.nTokens());
}

test "InputChunks: total tokens" {
    var chunks = mtmd.InputChunks.init(testing.allocator);
    defer chunks.deinit();
    const t = try testing.allocator.dupe(i32, &.{ 1, 2, 3 });
    try chunks.append(.{ .chunk_type = .text, .tokens_text = t });
    try chunks.append(.{ .chunk_type = .image, .tokens_image = .{ .nx = 4, .ny = 4 } });
    try testing.expectEqual(@as(usize, 2), chunks.size());
}

test "Caps: default" {
    const caps = mtmd.Caps{};
    try testing.expect(!caps.inp_vision);
}

test "DecoderPos: normal" {
    const img = mtmd.ImageTokens{ .nx = 3, .ny = 1, .pos = .normal };
    var positions: [3]mtmd.DecoderPos = undefined;
    mtmd.helper.imageGetDecoderPos(img, 10, &positions);
    try testing.expectEqual(@as(u32, 10), positions[0].t);
}

test "tokenize: text only" {
    var tc = try createTestContext(testing.allocator);
    defer tc.deinit();
    const input = mtmd.InputText{ .text = "Hello", .add_special = false };
    var chunks = try mtmd.tokenize.tokenize(tc.ctx, undefined, testing.allocator, input, &.{});
    defer chunks.deinit();
    try testing.expectEqual(@as(usize, 1), chunks.size());
}

test "tokenize: marker mismatch" {
    var tc = try createTestContext(testing.allocator);
    defer tc.deinit();
    const input = mtmd.InputText{ .text = "<__media__>", .add_special = false };
    try testing.expectError(error.MarkerBitmapMismatch, mtmd.tokenize.tokenize(tc.ctx, undefined, testing.allocator, input, &.{}));
}

test "tokenize: image marker" {
    var tc = try createTestContext(testing.allocator);
    defer tc.deinit();
    const input = mtmd.InputText{ .text = "Look: <__media__>", .add_special = false };
    var chunks = try mtmd.tokenize.tokenize(tc.ctx, undefined, testing.allocator, input, &.{mtmd.Bitmap.initPlaceholderImage(224, 224)});
    defer chunks.deinit();
    try testing.expect(chunks.size() >= 2);
}

test "tokenize: audio marker" {
    var tc = try createTestContextAudio(testing.allocator);
    defer tc.deinit();
    const input = mtmd.InputText{ .text = "Listen: <__media__>", .add_special = false };
    var chunks = try mtmd.tokenize.tokenize(tc.ctx, undefined, testing.allocator, input, &.{mtmd.Bitmap.initPlaceholderAudio(16000)});
    defer chunks.deinit();
    try testing.expect(chunks.size() >= 2);
}

const TestContext = struct {
    ctx: *mtmd.MtmdContext,
    mgr: *mtmd.MultiModalManager,

    fn deinit(self: *TestContext) void {
        const allocator = self.ctx.allocator;
        self.mgr.deinit();
        self.ctx.deinit();
        allocator.destroy(self.mgr);
    }
};

fn createTestContext(allocator: std.mem.Allocator) !TestContext {
    const caps = model.ModelCapabilities{ .has_vision = true, .vision_encoder_type = "gemma4v" };
    const mgr = try allocator.create(mtmd.MultiModalManager);
    mgr.* = .{ .allocator = allocator, .capabilities = caps, .audio_encoder = null, .vision_encoder = null };
    const ctx = try mtmd.MtmdContext.init(allocator, mgr, 2560, mtmd.contextParamsDefault(), null);
    return TestContext{ .ctx = ctx, .mgr = mgr };
}

fn createTestContextAudio(allocator: std.mem.Allocator) !TestContext {
    const caps = model.ModelCapabilities{ .has_audio = true, .audio_encoder_type = "gemma4a", .audio_sample_rate = 16000 };
    const mgr = try allocator.create(mtmd.MultiModalManager);
    mgr.* = .{ .allocator = allocator, .capabilities = caps, .audio_encoder = null, .vision_encoder = null };
    const ctx = try mtmd.MtmdContext.init(allocator, mgr, 2560, mtmd.contextParamsDefault(), null);
    return TestContext{ .ctx = ctx, .mgr = mgr };
}
