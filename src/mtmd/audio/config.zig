//! 音频处理配置参数
//!
//! 所有音频处理阶段的配置参数集中管理。
//! 参数可从 GGUF 元数据加载，也可使用默认值。
//!
//! 参考: llama.cpp mtmd-audio.cpp, gemma4a.cpp

const std = @import("std");
const gguf = @import("gguf");

const log = std.log.scoped(.audio_config);

// ============================================================================
// 默认音频参数常量
// ============================================================================

/// 默认音频采样率 (Gemma4 E2B)
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
/// STFT 窗口长度 (20ms @ 16kHz, matches llama.cpp gemma4a window_len=320)
pub const DEFAULT_FRAME_LENGTH: u32 = 320;
/// STFT 帧移 (10ms @ 16kHz)
pub const DEFAULT_HOP_LENGTH: u32 = 160;
/// FFT 点数（2 的幂）
pub const DEFAULT_N_FFT: u32 = 512;
/// Mel 滤波器组数量
pub const DEFAULT_N_MEL_BINS: u32 = 128;
/// Mel 最低频率
pub const DEFAULT_MEL_F_MIN: f32 = 0.0;
/// Mel 最高频率
pub const DEFAULT_MEL_F_MAX: f32 = 8000.0;
/// 预加重系数 (gemma4a disables pre-emphasis)
pub const DEFAULT_PRE_EMPHASIS: f32 = 0.0;
/// 对数偏移（防止 log(0)，matches gemma4a mel_floor=0.001）
pub const DEFAULT_LOG_OFFSET: f32 = 0.001;

// ============================================================================
// 音频预处理参数
// ============================================================================

/// 音频预处理参数（可从 GGUF 元数据加载）
pub const AudioPreprocessParams = struct {
    sample_rate: u32 = DEFAULT_SAMPLE_RATE,
    frame_length: u32 = DEFAULT_FRAME_LENGTH,
    hop_length: u32 = DEFAULT_HOP_LENGTH,
    n_fft: u32 = DEFAULT_N_FFT,
    n_mel_bins: u32 = DEFAULT_N_MEL_BINS,
    mel_f_min: f32 = DEFAULT_MEL_F_MIN,
    mel_f_max: f32 = DEFAULT_MEL_F_MAX,
    pre_emphasis: f32 = DEFAULT_PRE_EMPHASIS,
    log_offset: f32 = DEFAULT_LOG_OFFSET,

    /// 从音频编码器参数构建（部分参数从 GGUF 元数据加载）
    pub fn fromAudioEncoder(n_mel_bins: u32) AudioPreprocessParams {
        return .{ .n_mel_bins = n_mel_bins };
    }

    /// 从 GGUF 元数据加载音频预处理参数
    pub fn fromGGUF(gguf_file: *const gguf.GGUFFile) AudioPreprocessParams {
        var p = AudioPreprocessParams{};
        if (gguf_file.getU32("clip.audio.num_mel_bins")) |v| p.n_mel_bins = v;
        if (gguf_file.getU32("clip.audio.sample_rate")) |v| p.sample_rate = v;
        if (gguf_file.getU32("clip.audio.n_fft")) |v| p.n_fft = v;
        if (gguf_file.getU32("clip.audio.hop_length")) |v| p.hop_length = v;
        if (gguf_file.getU32("clip.audio.window_length")) |v| p.frame_length = v;
        return p;
    }
};

// ============================================================================
// 音频编码器参数
// ============================================================================

/// 音频编码器超参数
pub const AudioEncoderParams = struct {
    /// 输入特征维度（mel bins）
    n_mel_bins: u32 = 128,
    /// 模型嵌入维度
    n_embd: u32 = 512,
    /// 注意力头数
    n_head: u32 = 8,
    /// 每头维度
    d_head: u32 = 64,
    /// Conformer 层数 (Gemma 4 E2B 使用 12 层)
    n_layer: u32 = 12,
    /// FFN 中间维度
    n_ff: u32 = 2048,
    /// 输出投影维度（匹配 LLM 嵌入维度，从 GGUF 元数据加载）
    /// Gemma 4 4B: 1536, Gemma 4 9B: 2560
    n_output_embd: u32 = 2560,
    /// 音频采样率
    sample_rate: u32 = 16000,
    /// 最大音频长度（秒）
    max_audio_length_sec: f32 = 30.0,
    /// 归一化 epsilon (loaded from GGUF: clip.audio.attention.layer_norm_epsilon)
    /// Gemma 4 所有模型（文本/视觉/音频）均使用 1e-6
    norm_eps: f32 = 1e-6,
};

// ============================================================================
// Chunked attention 参数
// ============================================================================

/// 分块局部注意力参数（匹配 llama.cpp gemma4a.cpp）
pub const ChunkedAttentionParams = struct {
    /// 块大小
    chunk_size: i64 = 12,
    /// 最大过去视野（context_left - 1）
    max_past_horizon: i64 = 12,
    /// 上下文大小 = chunk_size + max_past_horizon
    context_size: i64 = 24,
    /// RPE 位置数 = max_past_horizon + 1
    rpe_positions: i64 = 13,
    /// Softcap 值
    softcap: f32 = 50.0,
    /// Q 缩放因子
    q_scale_factor: f32 = 1.0,
    /// K 缩放因子
    k_scale_factor: f32 = 1.0,
};
