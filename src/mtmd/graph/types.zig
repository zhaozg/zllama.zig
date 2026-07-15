//! mtmd 计算图共享类型定义
//!
//! 定义计算图构建中使用的枚举类型、构建选项和权重结构。
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h, clip-model.h, clip-impl.h

const std = @import("std");
const ggml = @import("ggml");

// ============================================================================
// 枚举类型
// ============================================================================

/// FFN 激活函数类型
/// 参考: clip-model.h ffn_op_type
pub const FFNOpType = enum(u8) {
    gelu,
    gelu_erf,
    silu,
    gelu_quick,
    relu_sqr,
};

/// 归一化层类型
/// 参考: clip-model.h norm_type
pub const NormType = enum(u8) {
    layer_norm,
    rms_norm,
};

/// Patch merge 类型
/// 参考: clip-model.h patch_merge_type
pub const PatchMergeType = enum(u8) {
    flat,
    spatial_unpad,
};

/// 缩放算法
/// 参考: clip-model.h resize_algo
pub const ResizeAlgo = enum(u8) {
    bilinear,
    bicubic,
    bicubic_pillow,
};

/// 填充样式
/// 参考: clip-model.h pad_style
pub const PadStyle = enum(u8) {
    none,
    ceil,
    nearest,
};

/// 投影器类型（与 clip-impl.h 的 projector_type 对应）
pub const ProjectorType = enum(u16) {
    mlp,
    mlp_norm,
    ldp,
    ldpv2,
    minicpmv,
    glm_edge,
    qwen2vl,
    qwen3vl,
    step3vl,
    gemma3,
    gemma3nv,
    gemma3na,
    gemma4v,
    gemma4a,
    gemma4uv,
    gemma4ua,
    phi4,
    idefics3,
    pixtral,
    qwen25vl,
    ultravox,
    internvl,
    llama4,
    qwen2a,
    qwen3a,
    glma,
    qwen25o,
    voxtral,
    meralion,
    music_flamingo,
    lfm2,
    kimivl,
    paddleocr,
    lightonocr,
    cogvlm,
    janus_pro,
    dots_ocr,
    deepseekocr,
    deepseekocr2,
    lfm2a,
    glm4v,
    youtuvl,
    yasa2,
    kimik25,
    nemotron_v2_vl,
    exaone4_5,
    hunyuanvl,
    minicpmv4_6,
    granite_speech,
    mimovl,
    granite4_vision,
    unknown,

    pub fn toString(self: ProjectorType) []const u8 {
        return switch (self) {
            .mlp => "mlp",
            .mlp_norm => "mlp_norm",
            .ldp => "ldp",
            .ldpv2 => "ldpv2",
            .minicpmv => "resampler",
            .glm_edge => "adapter",
            .qwen2vl => "qwen2vl_merger",
            .qwen25vl => "qwen2.5vl_merger",
            .qwen3vl => "qwen3vl_merger",
            .step3vl => "step3vl",
            .gemma3 => "gemma3",
            .gemma3nv => "gemma3nv",
            .gemma3na => "gemma3na",
            .gemma4v => "gemma4v",
            .gemma4a => "gemma4a",
            .gemma4uv => "gemma4uv",
            .gemma4ua => "gemma4ua",
            .phi4 => "phi4",
            .idefics3 => "idefics3",
            .pixtral => "pixtral",
            .ultravox => "ultravox",
            .internvl => "internvl",
            .llama4 => "llama4",
            .qwen2a => "qwen2a",
            .qwen3a => "qwen3a",
            .glma => "glma",
            .qwen25o => "qwen2.5o",
            .voxtral => "voxtral",
            .meralion => "meralion",
            .music_flamingo => "musicflamingo",
            .lfm2 => "lfm2",
            .kimivl => "kimivl",
            .paddleocr => "paddleocr",
            .lightonocr => "lightonocr",
            .cogvlm => "cogvlm",
            .janus_pro => "janus_pro",
            .dots_ocr => "dots_ocr",
            .deepseekocr => "deepseekocr",
            .deepseekocr2 => "deepseekocr2",
            .lfm2a => "lfm2a",
            .glm4v => "glm4v",
            .youtuvl => "youtuvl",
            .yasa2 => "yasa2",
            .kimik25 => "kimik25",
            .nemotron_v2_vl => "nemotron_v2_vl",
            .exaone4_5 => "exaone4_5",
            .hunyuanvl => "hunyuanvl",
            .minicpmv4_6 => "minicpmv4_6",
            .granite_speech => "granite_speech",
            .mimovl => "mimovl",
            .granite4_vision => "granite4_vision",
            .unknown => "unknown",
        };
    }

    pub fn fromString(str: []const u8) ProjectorType {
        inline for (std.meta.fields(ProjectorType)) |field| {
            const pt: ProjectorType = @enumFromInt(field.value);
            if (std.mem.eql(u8, pt.toString(), str)) return pt;
        }
        return .unknown;
    }
};

/// 模态类型
pub const Modality = enum(u8) {
    vision,
    audio,
};

/// Flash attention 类型
pub const FlashAttnType = enum(i8) {
    auto = -1,
    disabled = 0,
    enabled = 1,
};

// ============================================================================
// 构建选项
// ============================================================================

/// ViT 构建选项
/// 参考: clip-graph.h build_vit_opts
/// build_mm 函数指针类型
/// 对应 C++ clip_graph::build_mm() — 虚拟函数，允许子类添加 clamp 等操作
/// 参数: (ctx, weight, input) -> output
pub const BuildMMFn = *const fn (ctx: *ggml.Context, w: *ggml.Tensor, x: *ggml.Tensor) *ggml.Tensor;

/// 带上下文的 build_mm 函数指针类型
/// 用于 GraphBuilder 等需要捕获 self 的场景
/// 默认 build_mm 实现：直接调用 ggml_mul_mat
pub fn defaultBuildMM(ctx: *ggml.Context, w: *ggml.Tensor, x: *ggml.Tensor) *ggml.Tensor {
    return w.mulMat(ctx, x);
}

/// 构建 ViT 选项
pub const BuildVitOpts = struct {
    attn_mask: ?*ggml.Tensor = null,
    /// 是否对 V 应用 RMSNorm（gemma4v 需要）
    v_norm: bool = false,
    /// V 归一化的 epsilon
    v_norm_eps: f32 = 1e-6,
    /// KQ 缩放因子（默认 1/sqrt(d_head)，gemma4v 使用 1.0）
    kq_scale: ?f32 = null,
    /// build_mm 回调（对应 C++ clip_graph::build_mm 虚拟函数）
    /// 默认使用 ggml_mul_mat，子类可覆盖以添加 clamp 等操作
    build_mm: BuildMMFn = defaultBuildMM,
};

// ============================================================================
// 超参数
// ============================================================================

/// 视觉编码器超参数
/// 参考: clip-model.h clip_hparams
pub const VisionHParams = struct {
    image_size: u32 = 0,
    patch_size: u32 = 0,
    n_embd: u32 = 0,
    n_ff: u32 = 0,
    projection_dim: u32 = 0,
    n_head: u32 = 0,
    n_head_kv: u32 = 0,
    n_layer: u32 = 0,
    n_merge: u32 = 0,

    // 预处理参数
    image_longest_edge: u32 = 0,
    image_min_pixels: i32 = -1,
    image_max_pixels: i32 = -1,
    image_mean: [3]f32 = .{ 0.0, 0.0, 0.0 },
    image_std: [3]f32 = .{ 1.0, 1.0, 1.0 },
    preproc_min_tiles: u32 = 0,
    preproc_max_tiles: u32 = 0,
    preproc_image_size: u32 = 0,

    // 模型参数
    ffn_op: FFNOpType = .gelu,
    mm_patch_merge_type: PatchMergeType = .flat,
    eps: f32 = 1e-6,
    rope_theta: f32 = 0.0,
    vision_feature_layer: []const u32 = &.{},
    attn_window_size: u32 = 0,
    n_wa_pattern: u32 = 0,
    wa_layer_indexes: []const u32 = &.{},
    wa_pattern_mode: []const u32 = &.{},

    // SAM (DeepSeek-OCR)
    sam_n_layer: u32 = 0,
    sam_n_head: u32 = 0,
    sam_n_embd: u32 = 0,

    // Granite4 Vision
    proj_spatial_offsets: []const u32 = &.{},
    downsample_query_side: u32 = 0,
    downsample_window_side: u32 = 0,

    // 音频
    n_mel_bins: u32 = 0,
    proj_stack_factor: u32 = 0,
    audio_chunk_size: u32 = 0,
    audio_conv_kernel_size: u32 = 0,
    audio_max_pos_emb: u32 = 0,
    audio_proj_window_size: u32 = 0,
    audio_proj_downsample_rate: u32 = 0,
    audio_proj_head_count: u32 = 0,

    // 音频预处理
    audio_chunk_len: i32 = -1,
    audio_sample_rate: i32 = -1,
    audio_n_fft: i32 = -1,
    audio_window_len: i32 = -1,
    audio_hop_len: i32 = -1,

    // MiniCPM-V
    minicpmv_version: u32 = 0,
    minicpmv_query_num: u32 = 0,
    insert_layer_id: u32 = 0,

    // 自定义 token 限制
    custom_image_min_tokens: i32 = -1,
    custom_image_max_tokens: i32 = -1,

    // 预热
    warmup_image_size: u32 = 0,
    warmup_audio_size: u32 = 3000,

    pub fn setLimitImageTokens(self: *VisionHParams, n_tokens_min: u32, n_tokens_max: u32) void {
        const cur_merge: u32 = if (self.n_merge == 0) 1 else self.n_merge;
        // Use u64 to avoid overflow in patch_area computation
        const patch_area: u64 = @as(u64, self.patch_size) * @as(u64, self.patch_size) * @as(u64, cur_merge) * @as(u64, cur_merge);
        // Use i64 for token*patch multiplication to avoid overflow
        const tokens_min: i64 = @intCast(if (self.custom_image_min_tokens > 0) @as(u32, @intCast(self.custom_image_min_tokens)) else n_tokens_min);
        const tokens_max: i64 = @intCast(if (self.custom_image_max_tokens > 0) @as(u32, @intCast(self.custom_image_max_tokens)) else n_tokens_max);
        const patch_area_i64: i64 = @intCast(patch_area);
        self.image_min_pixels = @intCast(@min(tokens_min * patch_area_i64, std.math.maxInt(i32)));
        self.image_max_pixels = @intCast(@min(tokens_max * patch_area_i64, std.math.maxInt(i32)));
        // Guard against non-positive image_max_pixels in sqrt
        const max_px: u32 = @intCast(@max(0, self.image_max_pixels));
        const px: f64 = @floatFromInt(max_px);
        self.warmup_image_size = @intFromFloat(@sqrt(px));
    }
};

// ============================================================================
// 图像类型
// ============================================================================

/// F32 图像数据
/// 参考: clip-impl.h clip_image_f32
pub const ImageF32 = struct {
    /// 对于图像: buf.len == nx * ny * 3 (RGBRGBRGB...)
    /// 对于序列: buf.len == nx * ny * 3 * nt (nt 次重复)
    /// 对于音频: buf.len == nx * ny (单通道)
    buf: []const f32,
    nx: u32,
    ny: u32,
    /// 是否为全局视图（如 DeepSeek-OCR）
    add_viewsep: bool = false,
    /// 是否追加 learned newline（如 Granite4 Vision）
    add_newline: bool = false,

    pub fn isPlaceholder(self: *const ImageF32) bool {
        return self.buf.len == 0;
    }

    pub fn nElements(self: *const ImageF32) usize {
        return @as(usize, self.nx) * @as(usize, self.ny) * 3;
    }
};

/// U8 图像数据
/// 参考: clip-impl.h clip_image_u8
pub const ImageU8 = struct {
    buf: []const u8,
    nx: u32,
    ny: u32,

    pub fn isPlaceholder(self: *const ImageU8) bool {
        return self.buf.len == 0;
    }

    pub fn nElements(self: *const ImageU8) usize {
        return @as(usize, self.nx) * @as(usize, self.ny) * 3;
    }

    pub fn getPixel(self: *const ImageU8, x: u32, y: u32) [3]u8 {
        if (self.isPlaceholder()) return .{ 0, 0, 0 };
        const idx = (y * self.nx + x) * 3;
        return .{ self.buf[idx], self.buf[idx + 1], self.buf[idx + 2] };
    }
};

/// F32 图像批次
/// 参考: clip-impl.h clip_image_f32_batch
pub const ImageF32Batch = struct {
    entries: []const ImageF32,
    is_audio: bool = false,
};

// ============================================================================
// 权重结构
// ============================================================================

/// ViT 单层权重
/// 参考: clip-model.h clip_layer
pub const ViTLayerWeights = struct {
    // LayerNorm 1 (pre-attention)
    ln_1_w: ?*ggml.Tensor = null,
    ln_1_b: ?*ggml.Tensor = null,

    // Attention
    k_w: ?*ggml.Tensor = null,
    k_b: ?*ggml.Tensor = null,
    q_w: ?*ggml.Tensor = null,
    q_b: ?*ggml.Tensor = null,
    v_w: ?*ggml.Tensor = null,
    v_b: ?*ggml.Tensor = null,
    qkv_w: ?*ggml.Tensor = null,
    qkv_b: ?*ggml.Tensor = null,
    o_w: ?*ggml.Tensor = null,
    o_b: ?*ggml.Tensor = null,

    // Attention sinks
    attn_sinks: ?*ggml.Tensor = null,

    // Q/K norm
    k_norm: ?*ggml.Tensor = null,
    q_norm: ?*ggml.Tensor = null,

    // Post-attention norm
    attn_post_norm_w: ?*ggml.Tensor = null,

    // FFN
    ff_up_w: ?*ggml.Tensor = null,
    ff_up_b: ?*ggml.Tensor = null,
    ff_gate_w: ?*ggml.Tensor = null,
    ff_gate_b: ?*ggml.Tensor = null,
    ff_down_w: ?*ggml.Tensor = null,
    ff_down_b: ?*ggml.Tensor = null,

    // LayerNorm 2 (pre-FFN)
    ln_2_w: ?*ggml.Tensor = null,
    ln_2_b: ?*ggml.Tensor = null,

    // Post-FFN norm
    ff_post_norm_w: ?*ggml.Tensor = null,

    // Layer scale
    ls_1_w: ?*ggml.Tensor = null,
    ls_2_w: ?*ggml.Tensor = null,
    ls_out_w: ?*ggml.Tensor = null, // gemma4

    // Qwen3VL deepstack merger
    deepstack_norm_w: ?*ggml.Tensor = null,
    deepstack_norm_b: ?*ggml.Tensor = null,
    deepstack_fc1_w: ?*ggml.Tensor = null,
    deepstack_fc1_b: ?*ggml.Tensor = null,
    deepstack_fc2_w: ?*ggml.Tensor = null,
    deepstack_fc2_b: ?*ggml.Tensor = null,

    // SAM rel_pos
    rel_pos_w: ?*ggml.Tensor = null,
    rel_pos_h: ?*ggml.Tensor = null,

    // LFM2
    ff_norm_w: ?*ggml.Tensor = null,
    ff_norm_b: ?*ggml.Tensor = null,
    ff_norm_1_w: ?*ggml.Tensor = null,
    ff_norm_1_b: ?*ggml.Tensor = null,
    ff_up_1_w: ?*ggml.Tensor = null,
    ff_up_1_b: ?*ggml.Tensor = null,
    ff_down_1_w: ?*ggml.Tensor = null,
    ff_down_1_b: ?*ggml.Tensor = null,
    pos_bias_u: ?*ggml.Tensor = null,
    pos_bias_v: ?*ggml.Tensor = null,
    norm_conv_w: ?*ggml.Tensor = null,
    norm_conv_b: ?*ggml.Tensor = null,
    linear_pos_w: ?*ggml.Tensor = null,
    conv_norm_w: ?*ggml.Tensor = null,
    conv_norm_b: ?*ggml.Tensor = null,
    conv_dw_w: ?*ggml.Tensor = null,
    conv_dw_b: ?*ggml.Tensor = null,
    conv_pw1_w: ?*ggml.Tensor = null,
    conv_pw1_b: ?*ggml.Tensor = null,
    conv_pw2_w: ?*ggml.Tensor = null,
    conv_pw2_b: ?*ggml.Tensor = null,

    // Gemma4 audio conformer per-layer
    attn_pre_norm_w: ?*ggml.Tensor = null,
    attn_k_rel_w: ?*ggml.Tensor = null,
    per_dim_scale_w: ?*ggml.Tensor = null,
    per_dim_k_scale_w: ?*ggml.Tensor = null,
    ff_post_norm_1_w: ?*ggml.Tensor = null,

    // GraniteSpeech conformer per-layer
    attn_rel_pos_emb: ?*ggml.Tensor = null,

    // GraniteSpeech QFormer cross-attention
    cross_attn_q_w: ?*ggml.Tensor = null,
    cross_attn_q_b: ?*ggml.Tensor = null,
    cross_attn_k_w: ?*ggml.Tensor = null,
    cross_attn_k_b: ?*ggml.Tensor = null,
    cross_attn_v_w: ?*ggml.Tensor = null,
    cross_attn_v_b: ?*ggml.Tensor = null,
    cross_attn_o_w: ?*ggml.Tensor = null,
    cross_attn_o_b: ?*ggml.Tensor = null,
    cross_attn_norm_w: ?*ggml.Tensor = null,
    cross_attn_norm_b: ?*ggml.Tensor = null,

    pub fn hasDeepstack(self: *const ViTLayerWeights) bool {
        return self.deepstack_fc1_w != null;
    }
};

/// MobileNetV5 块权重
/// 参考: clip-model.h mobilenetv5_block
pub const MobileNetV5Block = struct {
    s0_conv_exp_w: ?*ggml.Tensor = null,
    s0_bn1_w: ?*ggml.Tensor = null,
    s0_conv_pwl_w: ?*ggml.Tensor = null,
    s0_bn2_w: ?*ggml.Tensor = null,
    dw_start_w: ?*ggml.Tensor = null,
    dw_start_bn_w: ?*ggml.Tensor = null,
    pw_exp_w: ?*ggml.Tensor = null,
    pw_exp_bn_w: ?*ggml.Tensor = null,
    dw_mid_w: ?*ggml.Tensor = null,
    dw_mid_bn_w: ?*ggml.Tensor = null,
    pw_proj_w: ?*ggml.Tensor = null,
    pw_proj_bn_w: ?*ggml.Tensor = null,
    layer_scale_w: ?*ggml.Tensor = null,
    attn_q_w: ?*ggml.Tensor = null,
    attn_k_w: ?*ggml.Tensor = null,
    attn_v_w: ?*ggml.Tensor = null,
    attn_o_w: ?*ggml.Tensor = null,
    attn_k_dw_w: ?*ggml.Tensor = null,
    attn_k_norm_w: ?*ggml.Tensor = null,
    attn_v_dw_w: ?*ggml.Tensor = null,
    attn_v_norm_w: ?*ggml.Tensor = null,
    attn_norm_w: ?*ggml.Tensor = null,
};

/// YASA2 块权重
/// 参考: clip-model.h yasa2_block
pub const YASA2Block = struct {
    dw_w: ?*ggml.Tensor = null,
    dw_b: ?*ggml.Tensor = null,
    ln_w: ?*ggml.Tensor = null,
    ln_b: ?*ggml.Tensor = null,
    pw1_w: ?*ggml.Tensor = null,
    pw1_b: ?*ggml.Tensor = null,
    grn_w: ?*ggml.Tensor = null,
    grn_b: ?*ggml.Tensor = null,
    pw2_w: ?*ggml.Tensor = null,
    pw2_b: ?*ggml.Tensor = null,
};

/// YASA2 阶段权重
/// 参考: clip-model.h yasa2_stage
pub const YASA2Stage = struct {
    down_ln_w: ?*ggml.Tensor = null,
    down_ln_b: ?*ggml.Tensor = null,
    down_conv_w: ?*ggml.Tensor = null,
    down_conv_b: ?*ggml.Tensor = null,
    blocks: []YASA2Block = &.{},
};

/// QFormer 投影器块
/// 参考: clip-model.h qf_block
pub const QFormerBlock = struct {
    qf_proj_query: ?*ggml.Tensor = null,
    qf_proj_norm_w: ?*ggml.Tensor = null,
    qf_proj_norm_b: ?*ggml.Tensor = null,
    qf_proj_linear_w: ?*ggml.Tensor = null,
    qf_proj_linear_b: ?*ggml.Tensor = null,
    qf_proj_post_norm_w: ?*ggml.Tensor = null,
    qf_proj_post_norm_b: ?*ggml.Tensor = null,
    qf_proj_img_pos: ?*ggml.Tensor = null,
    qf_proj_layers: []ViTLayerWeights = &.{},
};

/// Gemma4 裁剪信息
/// 参考: clip-model.h clip_model::clamp_info
pub const ClampInfo = struct {
    inp_max: f32,
    inp_min: f32,
    out_max: f32,
    out_min: f32,
};

/// 完整视觉编码器权重
/// 参考: clip-model.h clip_model
pub const VisionEncoderWeights = struct {
    // Embeddings
    class_embedding: ?*ggml.Tensor = null,
    patch_embeddings_0: ?*ggml.Tensor = null,
    patch_embeddings_1: ?*ggml.Tensor = null,
    patch_bias: ?*ggml.Tensor = null,
    position_embeddings: ?*ggml.Tensor = null,
    norm_embd_w: ?*ggml.Tensor = null,
    norm_embd_b: ?*ggml.Tensor = null,

    // Indexed patch embedding norms
    patch_norm_1_w: ?*ggml.Tensor = null,
    patch_norm_1_b: ?*ggml.Tensor = null,
    patch_norm_2_w: ?*ggml.Tensor = null,
    patch_norm_2_b: ?*ggml.Tensor = null,
    patch_norm_3_w: ?*ggml.Tensor = null,
    patch_norm_3_b: ?*ggml.Tensor = null,

    // Pre/post LN
    pre_ln_w: ?*ggml.Tensor = null,
    pre_ln_b: ?*ggml.Tensor = null,
    post_ln_w: ?*ggml.Tensor = null,
    post_ln_b: ?*ggml.Tensor = null,

    // ViT layers
    layers: []ViTLayerWeights = &.{},

    // Qwen3VL deepstack
    n_deepstack_layers: u32 = 0,

    // MM projector
    mm_fc_w: ?*ggml.Tensor = null,
    mm_fc_b: ?*ggml.Tensor = null,
    mm_ffn_up_w: ?*ggml.Tensor = null,
    mm_ffn_up_b: ?*ggml.Tensor = null,
    mm_ffn_gate_w: ?*ggml.Tensor = null,
    mm_ffn_gate_b: ?*ggml.Tensor = null,
    mm_ffn_down_w: ?*ggml.Tensor = null,
    mm_ffn_down_b: ?*ggml.Tensor = null,
    mm_post_norm_w: ?*ggml.Tensor = null,
    mm_post_norm_b: ?*ggml.Tensor = null,

    // LLaVA projection
    mm_input_norm_w: ?*ggml.Tensor = null,
    mm_input_norm_b: ?*ggml.Tensor = null,
    mm_0_w: ?*ggml.Tensor = null,
    mm_0_b: ?*ggml.Tensor = null,
    mm_2_w: ?*ggml.Tensor = null,
    mm_2_b: ?*ggml.Tensor = null,

    // Image newline / view separator
    image_newline: ?*ggml.Tensor = null,
    view_seperator: ?*ggml.Tensor = null,

    // Gemma3
    mm_input_proj_w: ?*ggml.Tensor = null,
    mm_soft_emb_norm_w: ?*ggml.Tensor = null,

    // Pixtral / GLM4V
    token_embd_img_break: ?*ggml.Tensor = null,
    mm_patch_merger_w: ?*ggml.Tensor = null,
    mm_patch_merger_b: ?*ggml.Tensor = null,

    // Gemma4
    std_bias: ?*ggml.Tensor = null,
    std_scale: ?*ggml.Tensor = null,

    // MobileNetV5 (Gemma3n)
    mobilenet_stem_conv_w: ?*ggml.Tensor = null,
    mobilenet_stem_conv_b: ?*ggml.Tensor = null,
    mobilenet_stem_norm_w: ?*ggml.Tensor = null,
    mobilenet_blocks: []MobileNetV5Block = &.{},
    mobilenet_stage_ends: []const u32 = &.{},

    // YASA2
    yasa_patch_w: ?*ggml.Tensor = null,
    yasa_patch_b: ?*ggml.Tensor = null,
    yasa_patch_ln_w: ?*ggml.Tensor = null,
    yasa_patch_ln_b: ?*ggml.Tensor = null,
    yasa_backbone_ln_w: ?*ggml.Tensor = null,
    yasa_backbone_ln_b: ?*ggml.Tensor = null,
    yasa_vision_pos_embed: ?*ggml.Tensor = null,
    yasa_stages: []YASA2Stage = &.{},

    // CogVLM
    mm_post_fc_norm_w: ?*ggml.Tensor = null,
    mm_post_fc_norm_b: ?*ggml.Tensor = null,
    mm_h_to_4h_w: ?*ggml.Tensor = null,
    mm_gate_w: ?*ggml.Tensor = null,
    mm_4h_to_h_w: ?*ggml.Tensor = null,
    mm_boi: ?*ggml.Tensor = null,
    mm_eoi: ?*ggml.Tensor = null,

    // HunyuanVL
    mm_pre_norm_w: ?*ggml.Tensor = null,
    mm_img_begin: ?*ggml.Tensor = null,
    mm_img_end: ?*ggml.Tensor = null,

    // DeepSeek OCR SAM
    patch_embed_proj_w: ?*ggml.Tensor = null,
    patch_embed_proj_b: ?*ggml.Tensor = null,
    pos_embed: ?*ggml.Tensor = null,
    neck_0_w: ?*ggml.Tensor = null,
    neck_1_w: ?*ggml.Tensor = null,
    neck_1_b: ?*ggml.Tensor = null,
    neck_2_w: ?*ggml.Tensor = null,
    neck_3_w: ?*ggml.Tensor = null,
    neck_3_b: ?*ggml.Tensor = null,
    net_2: ?*ggml.Tensor = null,
    net_3: ?*ggml.Tensor = null,
    n_sam_layers: u32 = 12,
    sam_layers: []ViTLayerWeights = &.{},

    // DeepSeek OCR 2
    resample_query_768: ?*ggml.Tensor = null,
    resample_query_1024: ?*ggml.Tensor = null,

    // LFM2 audio
    pre_encode_conv_X_w: [7]?*ggml.Tensor = .{null} ** 7,
    pre_encode_conv_X_b: [7]?*ggml.Tensor = .{null} ** 7,
    pre_encode_out_w: ?*ggml.Tensor = null,
    pre_encode_out_b: ?*ggml.Tensor = null,

    // Gemma4 audio conformer
    sscp_conv_w: [2]?*ggml.Tensor = .{null} ** 2,
    sscp_conv_b: [2]?*ggml.Tensor = .{null} ** 2,
    sscp_norm_w: [2]?*ggml.Tensor = .{null} ** 2,
    sscp_inp_proj_w: ?*ggml.Tensor = null,
    sscp_inp_proj_b: ?*ggml.Tensor = null,
    audio_out_proj_w: ?*ggml.Tensor = null,
    audio_out_proj_b: ?*ggml.Tensor = null,

    // GraniteSpeech encoder
    inp_proj_w: ?*ggml.Tensor = null,
    inp_proj_b: ?*ggml.Tensor = null,
    ctc_out_w: ?*ggml.Tensor = null,
    ctc_out_b: ?*ggml.Tensor = null,
    ctc_out_mid_w: ?*ggml.Tensor = null,
    ctc_out_mid_b: ?*ggml.Tensor = null,

    // QFormer projector blocks
    qf_proj_blocks: []QFormerBlock = &.{},

    // Ultravox / Whisper encoder
    conv1d_1_w: ?*ggml.Tensor = null,
    conv1d_1_b: ?*ggml.Tensor = null,
    conv1d_2_w: ?*ggml.Tensor = null,
    conv1d_2_b: ?*ggml.Tensor = null,
    conv_out_w: ?*ggml.Tensor = null,
    conv_out_b: ?*ggml.Tensor = null,
    mm_norm_pre_w: ?*ggml.Tensor = null,
    mm_norm_pre_b: ?*ggml.Tensor = null,
    mm_norm_mid_w: ?*ggml.Tensor = null,

    // Qwen3A
    conv2d_1_w: ?*ggml.Tensor = null,
    conv2d_1_b: ?*ggml.Tensor = null,
    conv2d_2_w: ?*ggml.Tensor = null,
    conv2d_2_b: ?*ggml.Tensor = null,
    conv2d_3_w: ?*ggml.Tensor = null,
    conv2d_3_b: ?*ggml.Tensor = null,

    // MiniCPMV
    mm_model_pos_embed_k: ?*ggml.Tensor = null,
    mm_model_query: ?*ggml.Tensor = null,
    mm_model_proj: ?*ggml.Tensor = null,
    mm_model_proj_b: ?*ggml.Tensor = null,
    mm_model_kv_proj: ?*ggml.Tensor = null,
    mm_model_attn_q_w: ?*ggml.Tensor = null,
    mm_model_attn_q_b: ?*ggml.Tensor = null,
    mm_model_attn_k_w: ?*ggml.Tensor = null,
    mm_model_attn_k_b: ?*ggml.Tensor = null,
    mm_model_attn_v_w: ?*ggml.Tensor = null,
    mm_model_attn_v_b: ?*ggml.Tensor = null,
    mm_model_attn_o_w: ?*ggml.Tensor = null,
    mm_model_attn_o_b: ?*ggml.Tensor = null,
    mm_model_ln_q_w: ?*ggml.Tensor = null,
    mm_model_ln_q_b: ?*ggml.Tensor = null,
    mm_model_ln_kv_w: ?*ggml.Tensor = null,
    mm_model_ln_kv_b: ?*ggml.Tensor = null,
    mm_model_ln_post_w: ?*ggml.Tensor = null,
    mm_model_ln_post_b: ?*ggml.Tensor = null,

    // MiniCPM-V 4.6 ViT merger
    vit_merger_ln1_w: ?*ggml.Tensor = null,
    vit_merger_ln1_b: ?*ggml.Tensor = null,
    vit_merger_attn_q_w: ?*ggml.Tensor = null,
    vit_merger_attn_q_b: ?*ggml.Tensor = null,
    vit_merger_attn_k_w: ?*ggml.Tensor = null,
    vit_merger_attn_k_b: ?*ggml.Tensor = null,
    vit_merger_attn_v_w: ?*ggml.Tensor = null,
    vit_merger_attn_v_b: ?*ggml.Tensor = null,
    vit_merger_attn_o_w: ?*ggml.Tensor = null,
    vit_merger_attn_o_b: ?*ggml.Tensor = null,
    vit_merger_ds_ln_w: ?*ggml.Tensor = null,
    vit_merger_ds_ln_b: ?*ggml.Tensor = null,
    vit_merger_ds_up_w: ?*ggml.Tensor = null,
    vit_merger_ds_up_b: ?*ggml.Tensor = null,
    vit_merger_ds_down_w: ?*ggml.Tensor = null,
    vit_merger_ds_down_b: ?*ggml.Tensor = null,

    // MSFA (Multi-Scale Fusion Adapter)
    msfa_concat_conv_w: ?*ggml.Tensor = null,
    msfa_concat_norm_w: ?*ggml.Tensor = null,
    msfa_ffn_expand_w: ?*ggml.Tensor = null,
    msfa_ffn_project_w: ?*ggml.Tensor = null,
    msfa_ffn_expand_bn: ?*ggml.Tensor = null,
    msfa_ffn_project_bn: ?*ggml.Tensor = null,

    // GLM-Edge adapter
    mm_model_adapter_conv_w: ?*ggml.Tensor = null,
    mm_model_adapter_conv_b: ?*ggml.Tensor = null,

    // MobileVLM projection
    mm_model_mlp_1_w: ?*ggml.Tensor = null,
    mm_model_mlp_1_b: ?*ggml.Tensor = null,
    mm_model_mlp_3_w: ?*ggml.Tensor = null,
    mm_model_mlp_3_b: ?*ggml.Tensor = null,
    mm_model_block_1_block_0_0_w: ?*ggml.Tensor = null,
    mm_model_block_1_block_0_1_w: ?*ggml.Tensor = null,
    mm_model_block_1_block_0_1_b: ?*ggml.Tensor = null,
    mm_model_block_1_block_1_fc1_w: ?*ggml.Tensor = null,
    mm_model_block_1_block_1_fc1_b: ?*ggml.Tensor = null,
    mm_model_block_1_block_1_fc2_w: ?*ggml.Tensor = null,
    mm_model_block_1_block_1_fc2_b: ?*ggml.Tensor = null,
    mm_model_block_1_block_2_0_w: ?*ggml.Tensor = null,
    mm_model_block_1_block_2_1_w: ?*ggml.Tensor = null,
    mm_model_block_1_block_2_1_b: ?*ggml.Tensor = null,
    mm_model_block_2_block_0_0_w: ?*ggml.Tensor = null,
    mm_model_block_2_block_0_1_w: ?*ggml.Tensor = null,
    mm_model_block_2_block_0_1_b: ?*ggml.Tensor = null,
    mm_model_block_2_block_1_fc1_w: ?*ggml.Tensor = null,
    mm_model_block_2_block_1_fc1_b: ?*ggml.Tensor = null,
    mm_model_block_2_block_1_fc2_w: ?*ggml.Tensor = null,
    mm_model_block_2_block_1_fc2_b: ?*ggml.Tensor = null,
    mm_model_block_2_block_2_0_w: ?*ggml.Tensor = null,
    mm_model_block_2_block_2_1_w: ?*ggml.Tensor = null,
    mm_model_block_2_block_2_1_b: ?*ggml.Tensor = null,

    // MobileVLM V2 projection
    mm_model_mlp_0_w: ?*ggml.Tensor = null,
    mm_model_mlp_0_b: ?*ggml.Tensor = null,
    mm_model_mlp_2_w: ?*ggml.Tensor = null,
    mm_model_mlp_2_b: ?*ggml.Tensor = null,
    mm_model_peg_0_w: ?*ggml.Tensor = null,
    mm_model_peg_0_b: ?*ggml.Tensor = null,

    // Yi type models
    mm_1_w: ?*ggml.Tensor = null,
    mm_1_b: ?*ggml.Tensor = null,
    mm_3_w: ?*ggml.Tensor = null,
    mm_3_b: ?*ggml.Tensor = null,
    mm_4_w: ?*ggml.Tensor = null,
    mm_4_b: ?*ggml.Tensor = null,

    // Gemma4 clamp info
    clamp_info_map: std.StringHashMap(ClampInfo) = undefined,
};
