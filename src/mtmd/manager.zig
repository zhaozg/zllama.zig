//! 多模态管理器
//!
//! 协调音频编码器和视觉编码器的加载与推理，
//! 提供统一的多模态输入处理接口。
//!
//! 参考: llama.cpp tools/mtmd/mtmd.h, mtmd.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model = @import("model");
const audio = @import("audio");
const vision = @import("vision");

const log = std.log.scoped(.mm);

// ============================================================================
// 多模态输入类型
// ============================================================================

pub const MediaType = enum {
    text,
    image,
    audio,
};

pub const MediaInput = struct {
    media_type: MediaType,
    /// 文本数据（仅 text 类型）
    text: ?[]const u8 = null,
    /// 图像数据：RGB 字节 [height][width][3]，值范围 [0, 255]
    image_data: ?[]const u8 = null,
    image_width: u32 = 0,
    image_height: u32 = 0,
    /// 音频数据：Mel 频谱 F32 [n_mel_bins, n_frames]
    mel_data: ?[]const f32 = null,
    mel_bins: u32 = 0,
    mel_frames: u32 = 0,
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

    /// 从 GGUF 文件检测多模态能力
    /// 检查 mmproj 文件中是否存在视觉/音频编码器相关的张量
    pub fn detectFromGGUF(gf: *const gguf.GGUFFile) model.ModelCapabilities {
        var caps = model.ModelCapabilities{};
        if (gf.findTensor("v.patch_embd.weight") != null or
            gf.findTensor("v.position_embd.weight") != null or
            gf.findTensor("mm.input_projection.weight") != null or
            gf.findTensor("mm.soft_emb_norm.weight") != null)
        {
            caps.has_vision = true;
            // 与 vision.zig 加载代码保持一致的命名前缀
            caps.vision_encoder_type = if (gf.findTensor("v.patch_norm.1.weight") != null or
                gf.findTensor("patch_norm_1.weight") != null)
                "gemma4uv"
            else
                "gemma4v";
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

    /// 初始化多模态管理器
    /// @param gguf_file 包含多模态编码器权重的 GGUF 文件（通常为 mmproj 文件）
    /// @param ctx ggml 权重上下文
    pub fn init(
        allocator: std.mem.Allocator,
        gguf_file: *const gguf.GGUFFile,
        ctx: *ggml.Context,
        caps: model.ModelCapabilities,
    ) !MultiModalManager {
        var audio_enc: ?audio.AudioEncoder = null;
        var vision_enc: ?vision.VisionEncoder = null;

        if (caps.has_audio) {
            audio_enc = try audio.AudioEncoder.init(gguf_file, ctx, allocator);
            log.info("Audio encoder initialized", .{});
        }

        if (caps.has_vision) {
            vision_enc = try vision.VisionEncoder.init(gguf_file, ctx, allocator);
            log.info("Vision encoder initialized", .{});
        }

        return MultiModalManager{
            .allocator = allocator,
            .capabilities = caps,
            .audio_encoder = audio_enc,
            .vision_encoder = vision_enc,
        };
    }

    /// 释放多模态管理器
    pub fn deinit(self: *MultiModalManager) void {
        if (self.audio_encoder) |*enc| {
            enc.deinit(self.allocator);
        }
        if (self.vision_encoder) |*enc| {
            enc.deinit(self.allocator);
        }
    }

    /// 编码单个多模态输入，返回嵌入 tokens
    pub fn encodeMedia(
        self: *MultiModalManager,
        ctx: *ggml.Context,
        graph: *ggml.CGraph,
        input: MediaInput,
    ) !*ggml.Tensor {
        return switch (input.media_type) {
            .text => error.TextEncodingNotSupportedHere,
            .image => {
                if (self.vision_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.VisionEncoderNotAvailable;
                    return enc.encode(ctx, graph, input.image_data.?, input.image_width, input.image_height);
                }
                return error.VisionEncoderNotAvailable;
            },
            .audio => {
                if (self.audio_encoder) |*enc| {
                    if (!enc.isAvailable()) return error.AudioEncoderNotAvailable;
                    return enc.encode(ctx, graph, input.mel_data.?, input.mel_bins, input.mel_frames);
                }
                return error.AudioEncoderNotAvailable;
            },
        };
    }

    /// 估算多模态输入的 token 数量
    pub fn estimateTokenCount(self: *const MultiModalManager, input: MediaInput) u32 {
        return switch (input.media_type) {
            .text => 0,
            .image => {
                if (self.vision_encoder) |*enc| {
                    return enc.estimateOutputTokens(input.image_width, input.image_height);
                }
                return 0;
            },
            .audio => {
                if (self.audio_encoder) |*enc| {
                    return enc.estimateOutputTokens(input.audio_length_sec);
                }
                return 0;
            },
        };
    }

    /// 检查模型是否支持指定媒体类型
    pub fn supportsMediaType(self: *const MultiModalManager, media_type: MediaType) bool {
        return switch (media_type) {
            .text => true,
            .image => self.capabilities.has_vision,
            .audio => self.capabilities.has_audio,
        };
    }

    /// 格式化多模态能力描述
    pub fn formatCapabilities(self: *const MultiModalManager, writer: anytype) !void {
        try writer.print("Multi-modal capabilities:\n", .{});
        try writer.print("  Text  : yes\n", .{});
        try writer.print("  Vision: {s}", .{if (self.capabilities.has_vision) "yes" else "no"});
        if (self.capabilities.has_vision) {
            try writer.print(" ({s})", .{self.capabilities.vision_encoder_type});
        }
        try writer.print("\n", .{});
        try writer.print("  Audio : {s}", .{if (self.capabilities.has_audio) "yes" else "no"});
        if (self.capabilities.has_audio) {
            try writer.print(" ({s}, {d} Hz)", .{ self.capabilities.audio_encoder_type, self.capabilities.audio_sample_rate });
        }
        try writer.print("\n", .{});
    }
};
