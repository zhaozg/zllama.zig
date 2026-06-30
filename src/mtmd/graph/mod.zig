//! mtmd 计算图构建模块
//!
//! 提供 Vision Transformer (ViT) 计算图构建的通用构建块。
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");

pub const types = @import("types.zig");
pub const builder = @import("builder.zig");
pub const vit = @import("vit.zig");
pub const attn = @import("attn.zig");
pub const ffn = @import("ffn.zig");
pub const norm = @import("norm.zig");

// 模型特定构建器
pub const model_graphs = struct {
    pub const gemma4v = @import("models/gemma4v.zig");
    pub const gemma4a = @import("models/gemma4a.zig");
    pub const gemma4uv = @import("models/gemma4uv.zig");
    pub const qwen2vl = @import("models/qwen2vl.zig");
    pub const qwen3vl = @import("models/qwen3vl.zig");
};

pub const patch = @import("patch.zig");
pub const rope = @import("rope.zig");
pub const merge = @import("merge.zig");
pub const stack = @import("stack.zig");
pub const mm = @import("mm.zig");

// 重新导出核心类型
pub const GraphBuilder = builder.GraphBuilder;
pub const ProjectorType = types.ProjectorType;
pub const NormType = types.NormType;
pub const FFNOpType = types.FFNOpType;
pub const PatchMergeType = types.PatchMergeType;
pub const BuildVitOpts = types.BuildVitOpts;
pub const FlashAttnType = types.FlashAttnType;
pub const VisionHParams = types.VisionHParams;
pub const VisionEncoderWeights = types.VisionEncoderWeights;
pub const ViTLayerWeights = types.ViTLayerWeights;
pub const ImageF32 = types.ImageF32;
pub const ImageU8 = types.ImageU8;
pub const ImageF32Batch = types.ImageF32Batch;
pub const MobileNetV5Block = types.MobileNetV5Block;
pub const YASA2Block = types.YASA2Block;
pub const YASA2Stage = types.YASA2Stage;
pub const QFormerBlock = types.QFormerBlock;
pub const ClampInfo = types.ClampInfo;
pub const Modality = types.Modality;
pub const ResizeAlgo = types.ResizeAlgo;
pub const PadStyle = types.PadStyle;

/// 音频编码器后端接口
/// 每个音频模型实现此接口，提供模型特定的操作
pub const AudioEncoderBackend = struct {
    name: []const u8,
    loadParams: *const fn (gguf_file: *const gguf.GGUFFile, params: *VisionHParams) void,
    loadWeights: *const fn (allocator: std.mem.Allocator, gguf_file: *const gguf.GGUFFile, ctx: *ggml.Context, w: *VisionEncoderWeights) anyerror!void,
    loadClampInfo: *const fn (allocator: std.mem.Allocator, gguf_file: *const gguf.GGUFFile, w: *VisionEncoderWeights) anyerror!void,
    buildGraph: *const fn (ctx: *ggml.Context, gf: *ggml.CGraph, w: *const VisionEncoderWeights, p: *const VisionHParams, mel_tensor: *ggml.Tensor, clamp_map: *const std.StringHashMap(ClampInfo)) anyerror!*ggml.CGraph,
    estimateOutputTokens: *const fn (n_frames: u32) u32,
};

// 重新导出构建函数
pub const buildVit = vit.buildVit;
pub const resizePositionEmbeddings = vit.resizePositionEmbeddings;
pub const buildAttn = attn.buildAttn;
pub const buildFFN = ffn.buildFFN;
pub const buildNorm = norm.buildNorm;
pub const reshapeForBroadcast = norm.reshapeForBroadcast;
pub const buildInp = patch.buildInp;
pub const buildInpRaw = patch.buildInpRaw;
pub const buildRope2D = rope.buildRope2D;
pub const createPositionIndices = rope.createPositionIndices;
pub const buildPatchMergePermute = merge.buildPatchMergePermute;
pub const buildStack = stack.buildStack;
pub const buildMM = mm.buildMM;
pub const buildGemma3Projector = mm.buildGemma3Projector;
pub const buildMLPProjector = mm.buildMLPProjector;
pub const buildStandardizeAndProject = mm.buildStandardizeAndProject;
