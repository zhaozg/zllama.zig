//! mtmd — Multi-modal decoding module
//!
//! 高层多模态上下文，包装 MultiModalManager 并提供：
//! - 媒体标记管理（<|image|> / <|audio|> 等）
//! - Chunk 化输入处理（text / image / audio 混合）
//! - 能力检测与查询
//!
//! Reference: deps/llama.cpp/tools/mtmd/mtmd.h

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model = @import("model");
const mm = @import("mm");
const preprocess = @import("preprocess");
const tokenizer = @import("tokenizer");

const log = std.log.scoped(.mtmd);

pub const audio = @import("audio");
pub const vision = @import("vision");
pub const helper = @import("helper");
pub const tokenize = @import("tokenize");
pub const manager = mm;
pub const preprocess_mod = preprocess;

// ============================================================================
// 基础类型定义
// ============================================================================

pub const ChunkType = enum(u8) { text, image, audio };
pub const PosType = enum(u8) { normal, mrope, hunyuanvl };
pub const DecoderPos = struct { t: u32 = 0, x: u32 = 0, y: u32 = 0, z: u32 = 0 };

pub const ContextParams = struct {
    use_gpu: bool = true,
    print_timings: bool = true,
    n_threads: u32 = 4,
    media_marker: []const u8 = "<__media__>",
    warmup: bool = true,
    image_min_tokens: i32 = -1,
    image_max_tokens: i32 = -1,
};

pub fn defaultMarker() []const u8 {
    return "<__media__>";
}

pub fn contextParamsDefault() ContextParams {
    return .{};
}

pub const Caps = struct { inp_vision: bool = false, inp_audio: bool = false };

pub const Bitmap = struct {
    nx: u32,
    ny: u32,
    is_audio: bool = false,
    id: ?[]const u8 = null,
    data: ?[]const u8 = null,
    allocator: ?std.mem.Allocator = null,

    pub fn initImage(nx: u32, ny: u32, data: ?[]const u8) Bitmap {
        return .{ .nx = nx, .ny = ny, .data = data };
    }
    pub fn initAudio(n: u32, data: ?[]const u8) Bitmap {
        return .{ .nx = n, .ny = 1, .is_audio = true, .data = data };
    }
    pub fn initPlaceholderImage(nx: u32, ny: u32) Bitmap {
        return .{ .nx = nx, .ny = ny };
    }
    pub fn initPlaceholderAudio(n: u32) Bitmap {
        return .{ .nx = n, .ny = 1, .is_audio = true };
    }
    pub fn isPlaceholder(self: Bitmap) bool {
        return self.data == null;
    }
    pub fn nBytes(self: Bitmap) usize {
        return if (self.data) |d| d.len else 0;
    }
    pub fn deinit(self: *Bitmap) void {
        if (self.allocator) |a| {
            if (self.data) |d| a.free(d);
            self.data = null;
            self.allocator = null;
        }
    }
};

pub const ImageTokens = struct {
    nx: u32 = 0,
    ny: u32 = 0,
    pos: PosType = .normal,
    image_idx: u32 = 0,
    id: ?[]const u8 = null,
    patches: ?*anyopaque = null,
    patch_count: u32 = 0,

    pub fn nTokens(self: ImageTokens) u32 {
        return switch (self.pos) {
            .hunyuanvl => (self.nx + 1) * self.ny + 2,
            else => self.nx * self.ny,
        };
    }
    pub fn isPlaceholder(self: ImageTokens) bool {
        return self.patches == null;
    }
};

pub const InputChunk = struct {
    chunk_type: ChunkType,
    tokens_text: ?[]const i32 = null,
    tokens_image: ?ImageTokens = null,
    tokens_audio_n: u32 = 0,
    id: ?[]const u8 = null,

    pub fn nPos(self: InputChunk) u32 {
        return switch (self.chunk_type) {
            .text => @intCast(self.tokens_text.?.len),
            .image => {
                const img = &(self.tokens_image orelse return 0);
                return switch (img.pos) {
                    .mrope, .hunyuanvl => @max(@max(img.nx, img.ny), 1),
                    .normal => img.nTokens(),
                };
            },
            .audio => self.tokens_audio_n,
        };
    }
    pub fn nTokens(self: InputChunk) u32 {
        return switch (self.chunk_type) {
            .text => @intCast(self.tokens_text.?.len),
            .image => if (self.tokens_image) |img| img.nTokens() else 0,
            .audio => self.tokens_audio_n,
        };
    }
};

pub const InputChunks = struct {
    entries: std.ArrayList(InputChunk),
    allocator: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator) InputChunks {
        return .{ .entries = std.ArrayList(InputChunk).initCapacity(a, 0) catch @panic("OOM"), .allocator = a };
    }
    pub fn deinit(self: *InputChunks) void {
        for (self.entries.items) |*chunk| {
            if (chunk.tokens_text) |t| self.allocator.free(t);
        }
        self.entries.clearAndFree(self.allocator);
    }
    pub fn size(self: InputChunks) usize {
        return self.entries.items.len;
    }
    pub fn get(self: InputChunks, idx: usize) *const InputChunk {
        return &self.entries.items[idx];
    }
    pub fn append(self: *InputChunks, chunk: InputChunk) !void {
        try self.entries.append(self.allocator, chunk);
    }
    pub fn totalTokens(self: InputChunks) usize {
        var t: usize = 0;
        for (self.entries.items) |c| t += c.nTokens();
        return t;
    }
    pub fn totalPos(self: InputChunks) usize {
        var t: usize = 0;
        for (self.entries.items) |c| t += c.nPos();
        return t;
    }
};

pub const InputText = struct { text: []const u8, add_special: bool = true, parse_special: bool = true };

// ============================================================================
// 媒体标记辅助函数
// ============================================================================

/// 根据能力检测结果确定图像标记
fn resolveImageMarkers(caps: *const model.ModelCapabilities) struct { beg: []const u8, end: []const u8 } {
    if (!caps.has_vision) return .{ .beg = "", .end = "" };
    if (std.mem.eql(u8, caps.vision_encoder_type, "gemma4v") or
        std.mem.eql(u8, caps.vision_encoder_type, "gemma4uv"))
    {
        return .{ .beg = "<|image>", .end = "<image|>" };
    }
    return .{ .beg = "<start_of_image>", .end = "<end_of_image>" };
}

/// 根据能力检测结果确定音频标记
fn resolveAudioMarkers(caps: *const model.ModelCapabilities) struct { beg: []const u8, end: []const u8 } {
    if (!caps.has_audio) return .{ .beg = "", .end = "" };
    if (std.mem.eql(u8, caps.audio_encoder_type, "gemma4a") or
        std.mem.eql(u8, caps.audio_encoder_type, "gemma4ua"))
    {
        return .{ .beg = "<|audio>", .end = "<audio|>" };
    }
    return .{ .beg = "", .end = "" };
}

// ============================================================================
// MtmdContext — 多模态上下文
// ============================================================================

pub const MtmdContext = struct {
    allocator: std.mem.Allocator,
    mm_manager: *mm.MultiModalManager,
    caps: model.ModelCapabilities,
    params: ContextParams,
    n_embd_text: i32,
    tok: ?*tokenizer.Tokenizer = null,
    media_marker: []const u8,
    img_beg: []const u8 = "",
    img_end: []const u8 = "",
    aud_beg: []const u8 = "",
    aud_end: []const u8 = "",
    pos_type: PosType = .normal,
    output_embd: ?[]f32 = null,

    /// 从已有的 MultiModalManager 初始化 MtmdContext（堆分配）
    /// 调用者负责通过 deinit() 释放
    pub fn init(
        allocator: std.mem.Allocator,
        mm_manager: *mm.MultiModalManager,
        text_n_embd: i32,
        params: ContextParams,
        tok: ?*tokenizer.Tokenizer,
    ) !*MtmdContext {
        const caps = mm_manager.capabilities;
        const img = resolveImageMarkers(&caps);
        const aud = resolveAudioMarkers(&caps);
        log.info("Image markers: {s}{s}", .{ img.beg, img.end });
        log.info("Audio markers: {s}{s}", .{ aud.beg, aud.end });
        const self = try allocator.create(MtmdContext);
        self.* = .{
            .allocator = allocator,
            .mm_manager = mm_manager,
            .caps = caps,
            .params = params,
            .n_embd_text = text_n_embd,
            .tok = tok,
            .media_marker = params.media_marker,
            .img_beg = img.beg,
            .img_end = img.end,
            .aud_beg = aud.beg,
            .aud_end = aud.end,
            .pos_type = .normal,
        };
        return self;
    }

    /// 从 mmproj 文件路径初始化 MtmdContext（便捷方法）
    /// 内部打开 GGUF 文件、检测能力、创建 MultiModalManager
    pub fn initFromPath(
        allocator: std.mem.Allocator,
        mmproj_path: []const u8,
        io: std.Io,
        text_n_embd: i32,
        params: ContextParams,
        tok: ?*tokenizer.Tokenizer,
    ) !*MtmdContext {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
        defer file.close(io);
        var gf = try gguf.GGUFFile.init(allocator, file, io);
        defer gf.deinit();
        const caps = mm.MultiModalManager.detectFromGGUF(&gf);
        const buf_size: usize = 512 * 1024 * 1024;
        const ctx = try ggml.Context.init(allocator, buf_size, false);
        const mgr = try allocator.create(mm.MultiModalManager);
        mgr.* = try mm.MultiModalManager.init(allocator, &gf, ctx, caps);
        return try MtmdContext.init(allocator, mgr, text_n_embd, params, tok);
    }

    pub fn deinit(self: *MtmdContext) void {
        if (self.output_embd) |e| self.allocator.free(e);
        self.mm_manager.deinit();
        self.allocator.destroy(self.mm_manager);
        self.allocator.destroy(self);
    }

    pub fn supportVision(self: *const MtmdContext) bool {
        return self.caps.has_vision;
    }
    pub fn supportAudio(self: *const MtmdContext) bool {
        return self.caps.has_audio;
    }
    pub fn getAudioSampleRate(self: *const MtmdContext) i32 {
        return self.caps.audio_sample_rate;
    }
    pub fn decodeUseNonCausal(self: *const MtmdContext, chunk: ?*const InputChunk) bool {
        if (self.caps.vision_encoder_type.len > 0 and std.mem.startsWith(u8, self.caps.vision_encoder_type, "gemma")) {
            if (chunk) |c| return c.chunk_type == .image;
            return true;
        }
        return false;
    }
    pub fn decodeUseMRope(self: *const MtmdContext) bool {
        return self.pos_type == .mrope or self.pos_type == .hunyuanvl;
    }
    pub fn getMarker(self: *const MtmdContext) []const u8 {
        return self.media_marker;
    }
    pub fn getOutputEmbd(self: *MtmdContext) ?[]f32 {
        return self.output_embd;
    }
};

/// 从 mmproj 文件快速检测能力（不创建完整上下文）
pub fn getCapFromFile(allocator: std.mem.Allocator, mmproj_path: []const u8, io: std.Io) !Caps {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
    defer file.close(io);
    var gf = try gguf.GGUFFile.init(allocator, file, io);
    defer gf.deinit();
    const caps = mm.MultiModalManager.detectFromGGUF(&gf);
    return .{ .inp_vision = caps.has_vision, .inp_audio = caps.has_audio };
}
