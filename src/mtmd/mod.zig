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
const audio = @import("audio");
const vision = @import("vision");
const tokens = @import("tokenizer");
const graph = @import("graph");

const log = std.log.scoped(.mtmd);

pub const audio_mod = audio;
pub const vision_mod = vision;
pub const helper = @import("helper");
pub const tokenize = @import("tokenize");
pub const graph_mod = graph;

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

pub fn defaultMarker() []const u8 { return "<__media__>"; }
pub fn contextParamsDefault() ContextParams { return .{}; }
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
    /// Preprocessed image pixel data (RGB u8, owned by caller)
    raw_pixels: ?[]const u8 = null,
    patch_count: u32 = 0,

    pub fn nTokens(self: ImageTokens) u32 {
        return switch (self.pos) {
            .hunyuanvl => (self.nx + 1) * self.ny + 2,
            else => self.nx * self.ny,
        };
    }
    pub fn isPlaceholder(self: ImageTokens) bool {
        return self.raw_pixels == null and self.nx == 0;
    }
    pub fn getRawPixels(self: ImageTokens) ?[]const u8 {
        return self.raw_pixels;
    }
};

pub const InputChunk = struct {
    chunk_type: ChunkType,
    tokens_text: ?[]const i32 = null,
    tokens_image: ?ImageTokens = null,
    tokens_audio_n: u32 = 0,
    id: ?[]const u8 = null,
    /// Audio Mel spectrogram data (owned by caller), [n_mel_bins * n_frames] f32
    mel_data: ?[]const f32 = null,
    mel_bins: u32 = 0,
    mel_frames: u32 = 0,
    /// Raw audio PCM data (f32 samples as bytes, from Bitmap.data), used when Mel not pre-computed
    audio_data: ?[]const u8 = null,

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
    pub fn getMelData(self: InputChunk) ?[]const f32 { return self.mel_data; }
    pub fn getMelBins(self: InputChunk) ?u32 { return if (self.mel_data != null) self.mel_bins else null; }
    pub fn getMelFrames(self: InputChunk) ?u32 { return if (self.mel_data != null) self.mel_frames else null; }
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
            if (chunk.mel_data) |m| self.allocator.free(m);
        }
        self.entries.clearAndFree(self.allocator);
    }
    pub fn size(self: InputChunks) usize { return self.entries.items.len; }
    pub fn get(self: InputChunks, idx: usize) *const InputChunk { return &self.entries.items[idx]; }
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
// 多模态输入类型
// ============================================================================

pub const MediaType = enum { text, image, audio };

pub const MediaInput = struct {
    media_type: MediaType,
    text: ?[]const u8 = null,
    image_data: ?[]const u8 = null,
    image_width: u32 = 0,
    image_height: u32 = 0,
    mel_data: ?[]const f32 = null,
    mel_bins: u32 = 0,
    mel_frames: u32 = 0,
    mel_tensor: ?*ggml.Tensor = null,
    audio_length_sec: f32 = 0,
};

// ============================================================================
// 多模态管理器
// ============================================================================

pub const MultiModalManager = struct {
    allocator: std.mem.Allocator,
    capabilities: model.ModelCapabilities,
    audio_encoder: ?audio.AudioEncoder = null,
    vision_encoder: ?vision.VisionEncoder = null,

    pub fn detectFromGGUF(gf: *const gguf.GGUFFile) model.ModelCapabilities {
        var caps = model.ModelCapabilities{};
        if (gf.findTensor("v.patch_embd.weight") != null or
            gf.findTensor("v.position_embd.weight") != null or
            gf.findTensor("mm.input_projection.weight") != null or
            gf.findTensor("mm.soft_emb_norm.weight") != null)
        {
            caps.has_vision = true;
            if (gf.findTensor("v.patch_norm.1.weight") != null or
                gf.findTensor("patch_norm_1.weight") != null)
            {
                caps.vision_encoder_type = "gemma4uv";
            } else if (gf.findTensor("v.blk.0.attn_qkv.weight") != null) {
                caps.vision_encoder_type = "qwen3vl";
            } else if (gf.findTensor("v.blk.0.attn_q.weight") != null) {
                caps.vision_encoder_type = "qwen2vl";
            } else {
                caps.vision_encoder_type = "gemma4v";
            }
        }
        if (gf.findTensor("a.conv1d.0.weight") != null or
            gf.findTensor("a.input_projection.weight") != null or
            gf.findTensor("a.pre_encode.out.weight") != null or
            gf.findTensor("mm.a.input_projection.weight") != null)
        {
            caps.has_audio = true;
            caps.audio_encoder_type = "gemma4a";
            caps.audio_sample_rate = 16000;
            if (gf.getU32("gemma4.audio.sample_rate")) |v| caps.audio_sample_rate = @intCast(v);
        }
        return caps;
    }

    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        caps: model.ModelCapabilities,
    ) !MultiModalManager {
        var audio_enc: ?audio.AudioEncoder = null;
        var vision_enc: ?vision.VisionEncoder = null;

        if (caps.has_audio) {
            const backend = audio.getBackend(caps.audio_encoder_type) orelse {
                log.err("Unknown audio encoder type: '{s}'", .{caps.audio_encoder_type});
                return error.UnknownAudioEncoder;
            };
            audio_enc = try audio.AudioEncoder.init(io, gguf_file, ctx, allocator, backend);
            log.info("Audio encoder initialized: backend={s}", .{backend.name});
        }

        if (caps.has_vision) {
            const backend = vision.getBackend(caps.vision_encoder_type) orelse {
                log.err("Unknown vision encoder type: '{s}'", .{caps.vision_encoder_type});
                return error.UnknownVisionEncoder;
            };
            vision_enc = try vision.VisionEncoder.init(gguf_file, ctx, allocator, backend);
            log.info("Vision encoder initialized: backend={s}", .{backend.name});
        }

        return MultiModalManager{
            .allocator = allocator,
            .capabilities = caps,
            .audio_encoder = audio_enc,
            .vision_encoder = vision_enc,
        };
    }

    pub fn deinit(self: *MultiModalManager) void {
        if (self.audio_encoder) |*enc| enc.deinit(self.allocator);
        if (self.vision_encoder) |*enc| enc.deinit(self.allocator);
    }

    /// Encode a single multimodal input, returning an already-computed embedding tensor.
    pub fn encodeMedia(
        self: *MultiModalManager,
        io: std.Io,
        ctx: *ggml.Context,
        cgraph: *ggml.CGraph,
        input: MediaInput,
        n_threads: i32,
    ) !*ggml.Tensor {
        return switch (input.media_type) {
            .text => error.TextEncodingNotSupportedHere,
            .image => {
                if (self.vision_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.VisionEncoderNotAvailable;
                    return enc.encode(io, ctx, cgraph, input.image_data.?, input.image_width, input.image_height, n_threads);
                }
                return error.VisionEncoderNotAvailable;
            },
            .audio => {
                if (self.audio_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.AudioEncoderNotAvailable;
                    if (input.mel_tensor) |mt| {
                        return enc.encode(io, ctx, cgraph, mt, n_threads);
                    } else if (input.mel_data) |md| {
                        return enc.encodeRaw(io, ctx, cgraph, md, input.mel_bins, input.mel_frames, n_threads);
                    } else {
                        return error.NoAudioData;
                    }
                }
                return error.AudioEncoderNotAvailable;
            },
        };
    }

    pub fn estimateTokenCount(self: *const MultiModalManager, input: MediaInput) u32 {
        return switch (input.media_type) {
            .text => 0,
            .image => {
                if (self.vision_encoder) |*enc| return enc.estimateOutputTokens(input.image_width, input.image_height);
                return 0;
            },
            .audio => {
                if (self.audio_encoder) |*enc| return enc.estimateOutputTokens(input.audio_length_sec);
                return 0;
            },
        };
    }

    pub fn supportsMediaType(self: *const MultiModalManager, media_type: MediaType) bool {
        return switch (media_type) {
            .text => true,
            .image => self.capabilities.has_vision,
            .audio => self.capabilities.has_audio,
        };
    }

    pub fn formatCapabilities(self: *const MultiModalManager, writer: anytype) !void {
        try writer.print("Multi-modal capabilities:\n", .{});
        try writer.print("  Text  : yes\n", .{});
        try writer.print("  Vision: {s}", .{if (self.capabilities.has_vision) "yes" else "no"});
        if (self.capabilities.has_vision) try writer.print(" ({s})", .{self.capabilities.vision_encoder_type});
        try writer.print("\n", .{});
        try writer.print("  Audio : {s}", .{if (self.capabilities.has_audio) "yes" else "no"});
        if (self.capabilities.has_audio) try writer.print(" ({s}, {d} Hz)", .{ self.capabilities.audio_encoder_type, self.capabilities.audio_sample_rate });
        try writer.print("\n", .{});
    }
};

// ============================================================================
// 媒体标记辅助函数
// ============================================================================

fn resolveImageMarkers(caps: *const model.ModelCapabilities) struct { beg: []const u8, end: []const u8 } {
    if (!caps.has_vision) return .{ .beg = "", .end = "" };
    if (std.mem.eql(u8, caps.vision_encoder_type, "gemma4v") or
        std.mem.eql(u8, caps.vision_encoder_type, "gemma4uv"))
        return .{ .beg = "<|image>", .end = "<image|>" };
    if (std.mem.eql(u8, caps.vision_encoder_type, "qwen3vl") or
        std.mem.eql(u8, caps.vision_encoder_type, "qwen2vl"))
        return .{ .beg = "<|vision_start|>", .end = "<|vision_end|>" };
    return .{ .beg = "<start_of_image>", .end = "<end_of_image>" };
}

fn resolveAudioMarkers(caps: *const model.ModelCapabilities) struct { beg: []const u8, end: []const u8 } {
    if (!caps.has_audio) return .{ .beg = "", .end = "" };
    if (std.mem.eql(u8, caps.audio_encoder_type, "gemma4a") or
        std.mem.eql(u8, caps.audio_encoder_type, "gemma4ua"))
        return .{ .beg = "<|audio>", .end = "<audio|>" };
    return .{ .beg = "", .end = "" };
}

fn resolvePosType(caps: *const model.ModelCapabilities) PosType {
    if (caps.has_vision) {
        if (std.mem.eql(u8, caps.vision_encoder_type, "qwen3vl") or
            std.mem.eql(u8, caps.vision_encoder_type, "qwen2vl"))
            return .mrope;
    }
    return .normal;
}

// ============================================================================
// MtmdContext — 多模态上下文
// ============================================================================

pub const MtmdContext = struct {
    allocator: std.mem.Allocator,
    mm_manager: *MultiModalManager,
    caps: model.ModelCapabilities,
    params: ContextParams,
    n_embd_text: i32,
    tok: ?*tokens.Tokenizer = null,
    media_marker: []const u8,
    img_beg: []const u8 = "",
    img_end: []const u8 = "",
    aud_beg: []const u8 = "",
    aud_end: []const u8 = "",
    pos_type: PosType = .normal,
    output_embd: ?[]f32 = null,

    pub fn init(
        allocator: std.mem.Allocator,
        mm_manager: *MultiModalManager,
        text_n_embd: i32,
        params: ContextParams,
        tokenizer_: ?*tokens.Tokenizer,
    ) !*MtmdContext {
        const caps = mm_manager.capabilities;
        const img = resolveImageMarkers(&caps);
        const aud = resolveAudioMarkers(&caps);
        const self = try allocator.create(MtmdContext);
        self.* = .{
            .allocator = allocator,
            .mm_manager = mm_manager,
            .caps = caps,
            .params = params,
            .n_embd_text = text_n_embd,
            .tok = tokenizer_,
            .media_marker = params.media_marker,
            .img_beg = img.beg,
            .img_end = img.end,
            .aud_beg = aud.beg,
            .aud_end = aud.end,
            .pos_type = resolvePosType(&caps),
        };
        return self;
    }

    pub fn initFromPath(
        allocator: std.mem.Allocator,
        mmproj_path: []const u8,
        io: std.Io,
        text_n_embd: i32,
        params: ContextParams,
        tokenizer_: ?*tokens.Tokenizer,
    ) !*MtmdContext {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
        defer file.close(io);
        var gf = try gguf.GGUFFile.init(allocator, file, io);
        defer gf.deinit();
        const caps = MultiModalManager.detectFromGGUF(&gf);
        const ctx = try ggml.Context.init(512 * 1024 * 1024);
        const mgr = try allocator.create(MultiModalManager);
        mgr.* = try MultiModalManager.init(io, allocator, &gf, ctx, caps);
        return try MtmdContext.init(allocator, mgr, text_n_embd, params, tokenizer_);
    }

    pub fn deinit(self: *MtmdContext) void {
        if (self.output_embd) |e| self.allocator.free(e);
        self.allocator.destroy(self);
    }

    pub fn supportVision(self: *const MtmdContext) bool { return self.caps.has_vision; }
    pub fn supportAudio(self: *const MtmdContext) bool { return self.caps.has_audio; }
    pub fn getAudioSampleRate(self: *const MtmdContext) i32 { return self.caps.audio_sample_rate; }

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
    pub fn getMarker(self: *const MtmdContext) []const u8 { return self.media_marker; }
    pub fn getOutputEmbd(self: *MtmdContext) ?[]f32 { return self.output_embd; }
};

pub fn getCapFromFile(allocator: std.mem.Allocator, mmproj_path: []const u8, io: std.Io) !Caps {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
    defer file.close(io);
    var gf = try gguf.GGUFFile.init(allocator, file, io);
    defer gf.deinit();
    const caps = MultiModalManager.detectFromGGUF(&gf);
    return .{ .inp_vision = caps.has_vision, .inp_audio = caps.has_audio };
}
