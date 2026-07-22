//! mtmd — Multi-modal decoding module
//!
//! 多模态功能组件的**暴露面与集成接口**。
//!
//! 本模块作为 mtmd 功能组件的公共门面（Facade），对外提供明确、必要的集成接口。
//! 外部模块（如 src/core/multimodal.zig、src/core/engine.zig、src/core/loader.zig）
//! 应**仅通过本模块的公共 API** 与 mtmd 交互，不得直接访问 mtmd 内部子模块。
//!
//! ## 设计原则
//!
//! - **接口收敛**：只暴露必要的公共类型和函数，内部实现细节对外部不可见。
//! - **分层清晰**：本模块位于 L5 层，依赖 L0-L4 层，不依赖 L6 层（引擎层）。
//! - **职责单一**：本模块负责多模态编码器的生命周期管理、媒体标记解析、chunk 化输入处理。
//!   引擎层的多模态编排逻辑（prefill、decode 循环）在 src/core/multimodal.zig 中实现。
//!
//! ## 公共 API 概览
//!
//! ### 核心类型
//! - `MultiModalManager` — 多模态管理器，管理视觉/音频编码器的生命周期
//! - `MtmdContext` — 多模态上下文，提供媒体标记解析、chunk 化输入处理
//! - `InputChunks` / `InputChunk` — 输入数据块（text/image/audio 混合）
//! - `Bitmap` / `ImageTokens` — 图像/音频数据表示
//! - `ChunkType` / `PosType` / `DecoderPos` — 枚举类型
//! - `ContextParams` / `Caps` — 配置参数
//! - `MediaInput` / `MediaType` — 多模态输入类型
//!
//! ### 核心函数
//! - `MultiModalManager.detectFromGGUF()` — 从 GGUF 元数据检测多模态能力
//! - `MultiModalManager.init()` — 初始化多模态管理器
//! - `MultiModalManager.encodeMedia()` — 编码单个多模态输入
//! - `MtmdContext.init()` — 初始化多模态上下文
//! - `MtmdContext.initFromPath()` — 从 mmproj 文件路径初始化
//! - `tokenize()` — 按媒体标记分割文本并生成 InputChunks
//! - `evalChunks()` — 评估所有 chunks（编码媒体 + 解码文本）
//! - `getCapFromFile()` — 从 mmproj 文件获取能力信息
//!
//! ### 子模块（通过本模块重新导出）
//! - `preprocess` — 图像预处理（resize、归一化等）
//! - `vision` — 视觉编码器模块
//! - `audio` — 音频编码器模块
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

// ============================================================================
// 子模块重新导出（公共 API）
// ============================================================================

/// 图像预处理模块（resize、归一化、动态分辨率计算等）
pub const preprocess = @import("preprocess");

/// 视觉编码器模块（VisionEncoder、VisionEncoderBackend 等）
pub const vision_mod = vision;

/// 音频编码器模块（AudioEncoder、AudioEncoderBackend 等）
pub const audio_mod = audio;

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
    /// Check if this bitmap can be temporally merged with another.
    /// Matches llama.cpp mtmd_bitmap::can_merge_with().
    /// [QWEN_VIDEO] can (temporal) merge if both are images with same size.
    pub fn canMergeWith(self: Bitmap, other: Bitmap) bool {
        return !self.is_audio and !other.is_audio and self.nx == other.nx and self.ny == other.ny;
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
    /// Pixel width of the preprocessed image (for encoding).
    nx: u32 = 0,
    /// Pixel height of the preprocessed image (for encoding).
    ny: u32 = 0,
    /// Actual encoder output token count (set by estimateOutputTokens).
    /// When > 0, nTokens() returns this value instead of computing from nx/ny.
    n_tokens: u32 = 0,
    pos: PosType = .normal,
    image_idx: u32 = 0,
    /// Temporal merge factor (for qwen-vl style temporal merge, default 1).
    n_temporal_merge: u32 = 1,
    id: ?[]const u8 = null,
    /// Preprocessed image pixel data (RGB u8, owned by caller)
    raw_pixels: ?[]const u8 = null,

    pub fn nTokens(self: ImageTokens) u32 {
        if (self.n_tokens > 0) return self.n_tokens;
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
    pub fn getMelData(self: InputChunk) ?[]const f32 {
        return self.mel_data;
    }
    pub fn getMelBins(self: InputChunk) ?u32 {
        return if (self.mel_data != null) self.mel_bins else null;
    }
    pub fn getMelFrames(self: InputChunk) ?u32 {
        return if (self.mel_data != null) self.mel_frames else null;
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
            if (chunk.mel_data) |m| self.allocator.free(m);
            if (chunk.tokens_image) |*img| {
                if (img.raw_pixels) |rp| self.allocator.free(@constCast(rp));
            }
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

        // ====================================================================
        // 1. 优先从 GGUF 元数据键读取（匹配 C++ clip.cpp 行为）
        //    C++ 参考: clip.cpp clip_model_loader 构造函数
        //    - clip.has_vision_encoder (bool)
        //    - clip.has_audio_encoder (bool)
        //    - clip.projector_type (string) — 或 clip.vision.projector_type / clip.audio.projector_type
        // ====================================================================
        const has_vision_meta = gf.getBool("clip.has_vision_encoder") orelse false;
        const has_audio_meta = gf.getBool("clip.has_audio_encoder") orelse false;

        const proj_type = gf.getString("clip.projector_type") orelse "";
        const vision_proj_type = gf.getString("clip.vision.projector_type") orelse
            gf.getString("gemma4.vision.projector_type") orelse "";
        const audio_proj_type = gf.getString("clip.audio.projector_type") orelse
            gf.getString("gemma4.audio.projector_type") orelse "";

        // ====================================================================
        // 2. 视觉编码器检测
        // ====================================================================
        if (has_vision_meta) {
            caps.has_vision = true;
            if (vision_proj_type.len > 0) {
                caps.vision_encoder_type = vision_proj_type;
            } else if (proj_type.len > 0) {
                caps.vision_encoder_type = proj_type;
            } else {
                caps.vision_encoder_type = detectVisionEncoderByTensors(gf);
            }
        } else {
            if (gf.findTensor("v.patch_embd.weight") != null or
                gf.findTensor("v.position_embd.weight") != null or
                gf.findTensor("mm.soft_emb_norm.weight") != null)
            {
                caps.has_vision = true;
                caps.vision_encoder_type = detectVisionEncoderByTensors(gf);
            }
        }

        // ====================================================================
        // 3. 音频编码器检测
        // ====================================================================
        if (has_audio_meta) {
            caps.has_audio = true;
            if (audio_proj_type.len > 0) {
                caps.audio_encoder_type = audio_proj_type;
            } else if (proj_type.len > 0) {
                caps.audio_encoder_type = proj_type;
            } else {
                caps.audio_encoder_type = "gemma4a";
            }
            caps.audio_sample_rate = 16000;
            if (gf.getU32("gemma4.audio.sample_rate")) |v| caps.audio_sample_rate = @intCast(v);
        } else {
            if (gf.findTensor("a.conv1d.0.weight") != null or
                gf.findTensor("a.input_projection.weight") != null or
                gf.findTensor("a.pre_encode.out.weight") != null or
                gf.findTensor("mm.a.input_projection.weight") != null)
            {
                caps.has_audio = true;
                caps.audio_encoder_type = "gemma4a";
                caps.audio_sample_rate = 16000;
                if (gf.getU32("gemma4.audio.sample_rate")) |v| caps.audio_sample_rate = @intCast(v);
            } else if (gf.findTensor("mm.input_projection.weight") != null and
                !caps.has_vision and
                gf.findTensor("a.conv1d.0.weight") == null)
            {
                caps.has_audio = true;
                caps.audio_encoder_type = "gemma4ua";
                caps.audio_sample_rate = 16000;
                if (gf.getU32("gemma4.audio.sample_rate")) |v| caps.audio_sample_rate = @intCast(v);
            }
        }

        // ====================================================================
        // 4. 填充特殊 Token 标记
        // ====================================================================
        if (caps.has_vision) {
            if (std.mem.startsWith(u8, caps.vision_encoder_type, "gemma4")) {
                caps.special_tokens.img_beg = "<|image>";
                caps.special_tokens.img_end = "<image|>";
            } else if (std.mem.eql(u8, caps.vision_encoder_type, "qwen3vl") or
                std.mem.eql(u8, caps.vision_encoder_type, "qwen2vl"))
            {
                caps.special_tokens.img_beg = "<|vision_start|>";
                caps.special_tokens.img_end = "<|vision_end|>";
            } else {
                caps.special_tokens.img_beg = "<start_of_image>";
                caps.special_tokens.img_end = "<end_of_image>";
            }
        }
        if (caps.has_audio) {
            if (std.mem.startsWith(u8, caps.audio_encoder_type, "gemma4")) {
                caps.special_tokens.aud_beg = "<|audio>";
                caps.special_tokens.aud_end = "<audio|>";
            }
        }
        return caps;
    }

    fn detectVisionEncoderByTensors(gf: *const gguf.GGUFFile) []const u8 {
        if (gf.findTensor("v.patch_norm.1.weight") != null or
            gf.findTensor("patch_norm_1.weight") != null)
        {
            return "gemma4uv";
        }
        if (gf.findTensor("v.patch_embd_1.weight") != null) {
            return "qwen2vl";
        }
        if (gf.findTensor("v.blk.0.attn_qkv.weight") != null) {
            return "qwen3vl";
        }
        if (gf.findTensor("v.std_bias") != null or
            gf.findTensor("v.std_scale") != null)
        {
            return "gemma4v";
        }
        if (gf.findTensor("v.blk.0.attn_q.weight") != null) {
            return "gemma4v";
        }
        return "gemma4v";
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
            vision_enc = try vision.VisionEncoder.init(io, gguf_file, ctx, allocator, backend, caps.vision_encoder_type);
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
                log.debug("encodeMedia: image {d}x{d}", .{ input.image_width, input.image_height });
                if (self.vision_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.VisionEncoderNotAvailable;
                    return enc.encode(io, ctx, cgraph, input.image_data.?, input.image_width, input.image_height, n_threads);
                }
                return error.VisionEncoderNotAvailable;
            },
            .audio => {
                log.debug("encodeMedia: audio", .{});
                if (self.audio_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.AudioEncoderNotAvailable;
                    if (input.mel_tensor) |mt| {
                        return enc.encode(io, ctx, cgraph, mt);
                    } else if (input.mel_data) |md| {
                        return enc.encodeRaw(io, ctx, cgraph, md, input.mel_bins, input.mel_frames);
                    } else {
                        return error.NoAudioData;
                    }
                }
                return error.AudioEncoderNotAvailable;
            },
        };
    }

    pub fn estimateTokenCount(self: *const MultiModalManager, io: std.Io, input: MediaInput) u32 {
        return switch (input.media_type) {
            .text => 0,
            .image => {
                if (self.vision_encoder) |*enc| return enc.estimateOutputTokens(io, input.image_width, input.image_height);
                return 0;
            },
            .audio => {
                if (self.audio_encoder) |*enc| return enc.estimateOutputTokens(io, input.audio_length_sec);
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
    if (caps.special_tokens.img_beg.len > 0 and caps.special_tokens.img_end.len > 0) {
        return .{ .beg = caps.special_tokens.img_beg, .end = caps.special_tokens.img_end };
    }
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
    if (caps.special_tokens.aud_beg.len > 0 and caps.special_tokens.aud_end.len > 0) {
        return .{ .beg = caps.special_tokens.aud_beg, .end = caps.special_tokens.aud_end };
    }
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

pub fn getCapFromFile(allocator: std.mem.Allocator, mmproj_path: []const u8, io: std.Io) !Caps {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, mmproj_path, .{ .mode = .read_only });
    defer file.close(io);
    var gf = try gguf.GGUFFile.init(allocator, file, io);
    defer gf.deinit();
    const caps = MultiModalManager.detectFromGGUF(&gf);
    return .{ .inp_vision = caps.has_vision, .inp_audio = caps.has_audio };
}

// ============================================================================
// 高层集成函数
// ============================================================================

/// 按媒体标记分割文本并生成 InputChunks。
/// 这是 mtmd 模块对外提供的核心 tokenize 函数。
/// 内部委托给 mtmd/tokenize.zig 实现。
pub const tokenize = @import("tokenize").tokenize;

/// 评估所有 chunks（编码媒体 + 解码文本）。
/// 这是 mtmd 模块对外提供的核心 chunk 评估函数。
/// 内部委托给 mtmd/helper.zig 实现。
///
/// 注意：此函数接收 `computeGraphFn` 回调参数，用于执行 ggml 计算图。
/// 这样设计是为了消除 helper.zig 对 L6 层（engine_common）的直接依赖，
/// 由调用者（L6/L7 层）注入 computeGraph 实现。
pub const evalChunks = @import("helper").evalChunks;

/// 获取图像解码器位置（用于 M-RoPE）。
pub const imageGetDecoderPos = @import("helper").imageGetDecoderPos;

/// 从文件加载 Bitmap（图像或音频）。
pub const bitmapInitFromFile = @import("helper").bitmapInitFromFile;

/// 从缓冲区加载 Bitmap（图像或音频）。
pub const bitmapInitFromBuf = @import("helper").bitmapInitFromBuf;
