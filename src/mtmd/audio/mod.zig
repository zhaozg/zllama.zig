//! 音频处理模块入口
//!
//! 提供完整的音频处理流水线，从 WAV 文件加载到 Conformer 编码器推理。
//! 所有公开 API 通过此模块重新导出。
//!
//! 架构:
//! - config.zig: 配置参数管理
//! - types.zig: 阶段间传递的数据结构
//! - loader.zig: WAV 文件加载与重采样
//! - framing.zig: 分帧 + 加窗
//! - mel.zig: Mel 滤波器组
//! - log_transform.zig: 对数变换
//! - encoder.zig: Conformer 编码器
//! - postprocess.zig: 后处理（softcapping 等）
//! - mel_spectrogram.zig: Mel 频谱计算

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
pub const config = @import("config.zig");
pub const types = @import("types.zig");
pub const loader = @import("loader.zig");
pub const framing = @import("framing.zig");
pub const mel = @import("mel.zig");
pub const log_transform = @import("log_transform.zig");
pub const encoder = @import("encoder.zig");
pub const postprocess = @import("postprocess.zig");
pub const mel_spectrogram = @import("mel_spectrogram.zig");

// ============================================================================
// 便捷类型别名（保持向后兼容）
// ============================================================================

pub const AudioEncoderParams = config.AudioEncoderParams;
pub const AudioPreprocessParams = config.AudioPreprocessParams;

pub const WavInfo = types.WavInfo;
pub const ProcessedAudio = types.ProcessedAudio;

pub const AudioEncoderBackend = encoder.AudioEncoderBackend;
pub const AudioEncoder = encoder.AudioEncoder;

/// 注册的音频编码器后端列表
/// 新增音频模型时在此注册
const registered_backends = struct {
    pub const gemma4a = @import("graph").model_graphs.gemma4a.backend;
    pub const gemma4ua = @import("graph").model_graphs.gemma4ua.backend;
};

/// 根据模型类型名称查找对应的后端
pub fn getBackend(name: []const u8) ?*const AudioEncoderBackend {
    if (std.mem.eql(u8, name, "gemma4a")) return &registered_backends.gemma4a;
    if (std.mem.eql(u8, name, "gemma4ua")) return &registered_backends.gemma4ua;
    return null;
}

// ============================================================================
// 便捷函数别名（保持向后兼容）
// ============================================================================

pub const loadWav = loader.loadWav;
pub const computeMelSpectrogram = mel_spectrogram.processPcmSamples;
pub const melToTensor = mel_spectrogram.melToTensor;

/// Gemma4UA 专用预处理：直接将原始 PCM 样本分帧，不做 FFT/Mel 频谱
/// 参考: llama.cpp mtmd_audio_preprocessor_gemma4ua::preprocess()
pub fn processRawWaveform(
    allocator: std.mem.Allocator,
    samples: []const f32,
    frame_size: u32,
) !types.ProcessedAudio {
    if (samples.len == 0) return error.EmptyAudio;
    if (frame_size == 0) return error.InvalidFrameSize;

    const n_tokens = (@as(u32, @intCast(samples.len)) + frame_size - 1) / frame_size;

    // 分配 mel-major 布局的数据: data[f * n_tokens + t]
    // 这样 ggml 张量加载为 [n_tokens, frame_size] 即 ne[0]=n_tokens, ne[1]=frame_size
    const total_size = @as(usize, frame_size) * @as(usize, n_tokens);
    const data = try allocator.alloc(f32, total_size);
    @memset(data, 0.0);

    for (0..@as(usize, n_tokens)) |t| {
        for (0..@as(usize, frame_size)) |f| {
            const src_idx = t * @as(usize, frame_size) + f;
            // mel-major: data[f * n_tokens + t]
            data[f * @as(usize, n_tokens) + t] = if (src_idx < samples.len) samples[src_idx] else 0.0;
        }
    }

    return .{
        .data = data,
        .n_mel_bins = frame_size,
        .n_frames = n_tokens,
        .allocator = allocator,
    };
}

// ============================================================================
// 常量重新导出（保持向后兼容）
// ============================================================================

pub const AUDIO_SAMPLE_RATE: u32 = config.DEFAULT_SAMPLE_RATE;
pub const AUDIO_FRAME_LENGTH: u32 = config.DEFAULT_FRAME_LENGTH;
pub const AUDIO_HOP_LENGTH: u32 = config.DEFAULT_HOP_LENGTH;
pub const AUDIO_N_FFT: u32 = config.DEFAULT_N_FFT;
pub const AUDIO_N_MEL_BINS: u32 = config.DEFAULT_N_MEL_BINS;
pub const AUDIO_MEL_F_MIN: f32 = config.DEFAULT_MEL_F_MIN;
pub const AUDIO_MEL_F_MAX: f32 = config.DEFAULT_MEL_F_MAX;
pub const AUDIO_PRE_EMPHASIS: f32 = config.DEFAULT_PRE_EMPHASIS;
pub const AUDIO_LOG_OFFSET: f32 = config.DEFAULT_LOG_OFFSET;
