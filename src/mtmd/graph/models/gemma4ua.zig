//! Gemma4UA 音频编码器 — 轻量级实现
//!
//! 实现 Gemma4UnifiedAudioEmbedder 的计算图构建。
//! 与 Gemma4A 不同，Gemma4UA 跳过 Conformer blocks，
//! 仅对输入 Mel 频谱做 RMSNorm + 线性投影到 LLM 嵌入空间。
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/gemma4ua.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("../mod.zig");
const weight_loader = @import("weight_loader");

const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const VisionHParams = graph.VisionHParams;
const ViTLayerWeights = graph.ViTLayerWeights;
const ClampInfo = graph.ClampInfo;

const log = std.log.scoped(.gemma4ua_graph);

// ============================================================================
// 音频编码器后端注册
// ============================================================================
pub const backend = graph.AudioEncoderBackend{
    .name = "gemma4ua",
    .loadParams = loadParams,
    .loadWeights = loadWeights,
    .loadClampInfo = loadClampInfo,
    .buildGraph = buildGraphFromWeights,
    .estimateOutputTokens = estimateOutputTokens,
};

pub fn loadParams(io: std.Io, gguf_file: *const gguf.GGUFFile, params: *graph.VisionHParams) void {
    _ = io;
    _ = gguf_file;
    _ = params;
    // Gemma4UA 参数已由 encoder.zig 从 clip.audio.* 或 gemma4.audio.* 前缀加载
}

/// 从 GGUF 加载 Gemma4UA 音频编码器所有权重到 VisionEncoderWeights
pub fn loadWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    w: *VisionEncoderWeights,
) !void {
    _ = io;

    // Gemma4UA 只需要多模态投影权重
    // mm.input_projection.weight — 投影到 LLM 嵌入空间
    w.mm_input_proj_w = findTensorInGGUF(ctx, gguf_file, "mm.input_projection.weight") catch null;

    // mm.soft_emb_norm.weight — RMSNorm 权重（可选）
    w.mm_soft_emb_norm_w = findTensorInGGUF(ctx, gguf_file, "mm.soft_emb_norm.weight") catch null;

    // Gemma4UA 没有 Conformer layers
    w.layers = try allocator.alloc(ViTLayerWeights, 0);

    log.info("Gemma4UA weights loaded (no Conformer blocks)", .{});
}

/// 从 GGUF 加载 Gemma4UA 音频编码器的 clamp 信息
pub fn loadClampInfo(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    var weight_names = std.ArrayList([]const u8).initCapacity(allocator, 0) catch |err| return err;
    defer weight_names.deinit(allocator);

    if (w.mm_input_proj_w) |t| try weight_names.append(allocator, t.getName());

    w.clamp_info_map = try graph.clamp.loadClampInfoFromWeightNames(allocator, gguf_file, weight_names.items);
    log.info("Gemma4UA clamp info loaded: {d} entries", .{w.clamp_info_map.count()});
}

/// 从 VisionEncoderWeights 构建计算图的包装函数
fn buildGraphFromWeights(
    io: std.Io,
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const graph.VisionHParams,
    mel_tensor: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) !*ggml.CGraph {
    _ = io;
    return buildGraph(ctx, gf, w, p, mel_tensor, clamp_map);
}

/// 估算输出 token 数量
pub fn estimateOutputTokens(io: std.Io, n_frames: u32) u32 {
    _ = io;
    // Gemma4UA 没有下采样，输出 token 数等于输入帧数
    return n_frames;
}

// ============================================================================
// 权重加载辅助函数
// ============================================================================

fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name);
}

// ============================================================================
// 计算图构建
// ============================================================================

/// 构建 Gemma4UA 完整计算图
///
/// Gemma4UA 是 Gemma4UnifiedAudioEmbedder 的轻量级实现。
/// 处理流程:
///   1. 输入 Mel 频谱 [n_frames, n_mel_bins, 1, 1]
///   2. Transpose + cont: [n_mel_bins, n_frames, 1, 1]
///   3. RMSNorm (embedding_pre_projection_norm)
///   4. 线性投影到 LLM 嵌入空间 (mm.input_projection.weight)
///
/// 参考: llama.cpp gemma4ua.cpp build()
pub fn buildGraph(
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const graph.VisionHParams,
    mel_tensor: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) !*ggml.CGraph {
    const eps: f32 = if (p.eps > 0) p.eps else 1e-6;

    const n_frames: i64 = mel_tensor.ne()[0];
    const n_mel_bins: i64 = mel_tensor.ne()[1];

    log.info("Gemma4UA graph: frames={d}, mel_bins={d}, embd={d}", .{ n_frames, n_mel_bins, p.n_embd });

    // 1. 输入 Mel 频谱 [n_frames, n_mel_bins, 1, 1]
    //    参考: clip.cpp build_inp_raw() → ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, nx, ny, C, B)
    var cur = mel_tensor;

    // 2. Transpose + cont: [n_mel_bins, n_frames, 1, 1]
    //    对应 C++: ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3))
    cur = ggml.cont(ctx, ggml.permute(ctx, cur, 1, 0, 2, 3));
    cur.setName("gemma4ua_input");

    // 3. Gemma4UnifiedMultimodalEmbedder
    //    embedding_pre_projection_norm
    cur = cur.rmsNorm(ctx, eps);
    cur.setName("gemma4ua_norm");

    // 4. 投影到 LLM 嵌入空间 (with clamp)
    if (w.mm_input_proj_w) |proj_w| {
        cur = buildMMWithClamp(ctx, proj_w, cur, clamp_map);
        cur.setName("mm_proj");
    }

    // 构建计算图
    gf.buildForwardExpand(cur);

    log.info("Gemma4UA graph built successfully", .{});
    return gf;
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 带 clamp 的矩阵乘法
/// 对应 C++ clip_graph_gemma4v::build_mm()
fn buildMMWithClamp(
    ctx: *ggml.Context,
    w: *ggml.Tensor,
    x: *ggml.Tensor,
    clamp_map: *const std.StringHashMap(ClampInfo),
) *ggml.Tensor {
    const name = w.getName();
    if (clamp_map.get(name)) |ci| {
        const clamped = x.clamp(ctx, ci.inp_min, ci.inp_max);
        var out = w.mulMat(ctx, clamped);
        out = out.clamp(ctx, ci.out_min, ci.out_max);
        return out;
    } else {
        return w.mulMat(ctx, x);
    }
}
