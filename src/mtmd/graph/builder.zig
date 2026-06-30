//! GraphBuilder 接口定义
//!
//! 定义计算图构建器的核心接口，每个模型变体实现此接口。
//! 参考: deps/llama.cpp/tools/mtmd/clip-graph.h clip_graph

const std = @import("std");
const ggml = @import("ggml");
const types = @import("types.zig");
const debug_mod = @import("debug.zig");
const vit_builder = @import("vit.zig");
const attn_builder = @import("attn.zig");
const ffn_builder = @import("ffn.zig");
const norm_builder = @import("norm.zig");
const patch_builder = @import("patch.zig");
const rope_builder = @import("rope.zig");
const merge_builder = @import("merge.zig");
const stack_builder = @import("stack.zig");
const mm_builder = @import("mm.zig");

const ProjectorType = types.ProjectorType;
const NormType = types.NormType;
const FFNOpType = types.FFNOpType;
const BuildVitOpts = types.BuildVitOpts;
const FlashAttnType = types.FlashAttnType;
const VisionHParams = types.VisionHParams;
const VisionEncoderWeights = types.VisionEncoderWeights;
const ViTLayerWeights = types.ViTLayerWeights;
const ImageF32 = types.ImageF32;
const ClampInfo = types.ClampInfo;

const log = std.log.scoped(.graph_builder);

/// 计算图构建器
///
/// 每个模型变体实现此接口，提供特定的图构建逻辑。
/// 使用 Zig 的 switch 实现零成本运行时多态分发。
pub const GraphBuilder = struct {
    /// 模型权重引用
    weights: *const VisionEncoderWeights,
    /// 超参数引用
    hparams: *const VisionHParams,
    /// 投影器类型
    proj_type: ProjectorType,
    /// 输入图像数据
    img: *const ImageF32,
    /// ggml 上下文
    ctx0: *ggml.Context,
    /// 计算图
    gf: *ggml.CGraph,
    /// 批处理大小（默认 1）
    n_batch: u32 = 1,
    /// Flash attention 类型
    flash_attn_type: FlashAttnType = .disabled,
    /// 调试张量注册表
    debug_registry: debug_mod.DebugTensorRegistry = .{},

    /// 派生常量
    pub fn patchSize(self: *const GraphBuilder) u32 {
        return self.hparams.patch_size;
    }
    pub fn nEmbd(self: *const GraphBuilder) u32 {
        return self.hparams.n_embd;
    }
    pub fn nHead(self: *const GraphBuilder) u32 {
        return self.hparams.n_head;
    }
    pub fn nHeadKV(self: *const GraphBuilder) u32 {
        return if (self.hparams.n_head_kv > 0) self.hparams.n_head_kv else self.hparams.n_head;
    }
    pub fn dHead(self: *const GraphBuilder) u32 {
        return self.hparams.n_embd / self.hparams.n_head;
    }
    pub fn nLayer(self: *const GraphBuilder) u32 {
        return self.hparams.n_layer;
    }
    pub fn nMMProjEmbd(self: *const GraphBuilder) u32 {
        return self.hparams.projection_dim;
    }
    pub fn eps(self: *const GraphBuilder) f32 {
        return self.hparams.eps;
    }

    /// 构建完整的计算图
    /// 返回计算图指针
    pub fn build(self: *GraphBuilder) !*ggml.CGraph {
        // 默认实现: 调用 buildVit 构建 ViT
        // 模型特定实现应覆盖此方法
        _ = self;
        log.warn("GraphBuilder.build() not implemented for this model type", .{});
        return error.NotImplemented;
    }

    /// 矩阵乘法封装（支持 LoRA、clamping 等钩子）
    pub fn buildMM(self: *const GraphBuilder, w: *ggml.Tensor, x: *ggml.Tensor) !*ggml.Tensor {
        // 查找 clamp info
        const clamp_info = if (self.weights.clamp_info_map.get(w.name())) |ci| ci else null;
        return mm_builder.buildMM(self.ctx0, w, x, clamp_info);
    }

    /// 是否支持批处理
    pub fn supportBatch(self: *const GraphBuilder) bool {
        _ = self;
        return false;
    }

    /// 构建 ViT 主干
    pub fn buildVit(
        self: *GraphBuilder,
        inp: *ggml.Tensor,
        n_pos: i64,
        norm_t: NormType,
        ffn_t: FFNOpType,
        learned_pos_embd: ?*ggml.Tensor,
        add_pos: *const fn (*ggml.Context, *ggml.Tensor, *const ViTLayerWeights, *ggml.Tensor) *ggml.Tensor,
        opts: BuildVitOpts,
    ) !*ggml.Tensor {
        return vit_builder.buildVit(
            self.ctx0,
            inp,
            n_pos,
            norm_t,
            ffn_t,
            learned_pos_embd,
            self.weights,
            self.hparams,
            add_pos,
            opts,
        );
    }

    /// 构建 Conv2D patch embedding
    pub fn buildInp(self: *GraphBuilder) !*ggml.Tensor {
        const w = self.weights;
        const p = self.hparams;

        return patch_builder.buildInp(
            self.ctx0,
            null, // inp_raw - caller should provide
            w.patch_embeddings_0 orelse return error.MissingPatchEmbedding,
            w.patch_bias,
            p.image_size,
            p.image_size,
            2.0,
            -1.0,
        );
    }

    /// 构建原始输入处理
    pub fn buildInpRaw(self: *GraphBuilder, channels: u32) !*ggml.Tensor {
        return patch_builder.buildInpRaw(self.ctx0, channels);
    }

    /// 构建归一化层
    pub fn buildNorm(
        self: *GraphBuilder,
        cur: *ggml.Tensor,
        mw: *ggml.Tensor,
        mb: ?*ggml.Tensor,
        norm_type: NormType,
        norm_eps: f32,
        name: []const u8,
    ) !*ggml.Tensor {
        return norm_builder.buildNorm(self.ctx0, cur, mw, mb, norm_type, norm_eps, name);
    }

    /// 构建 FFN 层
    pub fn buildFFN(
        self: *GraphBuilder,
        cur: *ggml.Tensor,
        up: *ggml.Tensor,
        up_b: ?*ggml.Tensor,
        gate: ?*ggml.Tensor,
        gate_b: ?*ggml.Tensor,
        down: *ggml.Tensor,
        down_b: ?*ggml.Tensor,
        type_op: FFNOpType,
        name: []const u8,
    ) !*ggml.Tensor {
        return ffn_builder.buildFFN(self.ctx0, cur, up, up_b, gate, gate_b, down, down_b, type_op, name);
    }

    /// 构建注意力层
    pub fn buildAttn(
        self: *GraphBuilder,
        wo: *ggml.Tensor,
        wo_b: ?*ggml.Tensor,
        q_cur: *ggml.Tensor,
        k_cur: *ggml.Tensor,
        v_cur: *ggml.Tensor,
        kq_mask: ?*ggml.Tensor,
        kq_scale: f32,
        name: []const u8,
        sinks: ?*ggml.Tensor,
    ) !*ggml.Tensor {
        return attn_builder.buildAttn(
            self.ctx0,
            wo,
            wo_b,
            q_cur,
            k_cur,
            v_cur,
            kq_mask,
            kq_scale,
            @intCast(self.nHead()),
            name,
            sinks,
        );
    }

    /// 构建 2D RoPE
    pub fn buildRope2D(
        self: *GraphBuilder,
        cur: *ggml.Tensor,
        pos_a: *ggml.Tensor,
        pos_b: *ggml.Tensor,
        freq_base: f32,
        interleave_freq: bool,
    ) !*ggml.Tensor {
        return rope_builder.buildRope2D(self.ctx0, cur, pos_a, pos_b, freq_base, interleave_freq);
    }

    /// 构建 patch merge
    pub fn buildPatchMergePermute(
        self: *GraphBuilder,
        cur: *ggml.Tensor,
        scale_factor: u32,
        n_patches_x: i64,
        n_patches_y: i64,
    ) !*ggml.Tensor {
        return merge_builder.buildPatchMergePermute(self.ctx0, cur, scale_factor, n_patches_x, n_patches_y);
    }

    /// 构建 frame stacking
    pub fn buildStack(
        self: *GraphBuilder,
        cur: *ggml.Tensor,
        stack_factor: u32,
        n_embed: u32,
    ) !*ggml.Tensor {
        return stack_builder.buildStack(self.ctx0, cur, stack_factor, n_embed);
    }

    /// 构建 Gemma3 风格投影器
    pub fn buildGemma3Projector(self: *GraphBuilder, cur: *ggml.Tensor) !*ggml.Tensor {
        return mm_builder.buildGemma3Projector(self.ctx0, cur, self.weights, self.hparams.eps);
    }

    /// 构建标准化 + 投影
    pub fn buildStandardizeAndProject(self: *GraphBuilder, cur: *ggml.Tensor) !*ggml.Tensor {
        return mm_builder.buildStandardizeAndProject(self.ctx0, cur, self.weights, self.hparams.eps);
    }

    /// 创建位置索引张量
    pub fn createPositionIndices(
        self: *GraphBuilder,
        n_patches: i64,
        n_patches_x: i64,
    ) !struct { pos_x: *ggml.Tensor, pos_y: *ggml.Tensor } {
        return rope_builder.createPositionIndices(self.ctx0, n_patches, n_patches_x);
    }

    /// 注册一个张量用于调试
    ///
    /// 参数:
    ///   - allocator: 分配器
    ///   - tensor: 要调试的张量
    ///   - debug_name: 调试名称（会通过 setName 设置到张量）
    ///   - is_input: true 表示输入张量，false 表示输出张量
    ///
    /// 此函数会:
    ///   1. 设置张量的调试名称 (setName)
    ///   2. 如果是输入张量，调用 setInput；否则调用 setOutput
    ///   3. 将条目添加到调试注册表
    pub fn debugRegisterTensor(
        self: *GraphBuilder,
        allocator: std.mem.Allocator,
        tensor: *ggml.Tensor,
        debug_name: []const u8,
        is_input: bool,
    ) !void {
        try self.debug_registry.register(allocator, tensor, debug_name, is_input);
    }

    /// 保存所有调试张量数据到文件
    ///
    /// 参数:
    ///   - allocator: 分配器
    ///   - storage_path: 存储目录路径
    ///   - file_prefix: 文件名前缀
    ///
    /// 每个张量保存为一个 JSON 数组文件:
    ///   {storage_path}/{file_prefix}_{debug_name}.json
    pub fn debugSaveAll(
        self: *GraphBuilder,
        allocator: std.mem.Allocator,
        storage_path: []const u8,
        file_prefix: []const u8,
    ) !void {
        try self.debug_registry.saveAll(allocator, storage_path, file_prefix);
    }

    /// 保存单个调试张量数据到文件（通过调试名称在图中查找）
    ///
    /// 参数:
    ///   - allocator: 分配器
    ///   - debug_name: 调试名称（必须已通过 setName 设置）
    ///   - storage_path: 存储目录路径
    ///   - filename: 输出文件名
    pub fn debugSaveTensorByName(
        self: *GraphBuilder,
        allocator: std.mem.Allocator,
        debug_name: [:0]const u8,
        storage_path: []const u8,
        filename: []const u8,
    ) !void {
        try debug_mod.DebugTensorRegistry.saveTensorByName(
            allocator,
            self.gf,
            debug_name,
            storage_path,
            filename,
        );
    }
};
