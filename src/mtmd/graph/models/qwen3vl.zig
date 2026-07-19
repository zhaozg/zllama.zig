//! Qwen3VL 视觉编码器图构建
//!
//! 实现 Qwen3VL 视觉编码器的计算图构建。
//! 继承 Qwen2VL 的 temporal merge + spatial merge，但增加:
//!   - patch_bias (必须存在)
//!   - 可学习位置嵌入 (resize_position_embeddings, 必须存在)
//!   - QKV fused weight (attn_qkv.weight + attn_qkv.bias)
//!   - Deepstack features
//!   - class_embedding 必须为 null
//!
//! 参考: deps/llama.cpp/tools/mtmd/models/qwen3vl.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("../mod.zig");

const GraphBuilder = graph.GraphBuilder;
const NormType = graph.NormType;
const FFNOpType = graph.FFNOpType;
const defaultBuildMM = graph.defaultBuildMM;
const BuildVitOpts = graph.BuildVitOpts;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const VisionHParams = graph.VisionHParams;
const ViTLayerWeights = graph.ViTLayerWeights;
const ImageF32 = graph.ImageF32;

const log = std.log.scoped(.graph_model_qwen3vl);

// ============================================================================
// 视觉编码器后端注册
// ============================================================================

/// Qwen3VL 视觉编码器后端实例
pub const backend = graph.VisionEncoderBackend{
    .name = "qwen3vl",
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
    // Qwen3VL 参数已由 encoder.zig 从 clip.vision.* 前缀加载
}

pub fn loadClampInfo(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    _ = gguf_file;
    w.clamp_info_map = std.StringHashMap(graph.ClampInfo).init(allocator);
}

/// 从 GGUF 加载 Qwen3VL 视觉编码器所有权重到 VisionEncoderWeights
pub fn loadWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    w: *VisionEncoderWeights,
) !void {
    _ = io;
    // Patch embedding (v.patch_embd.*)
    w.patch_embeddings_0 = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight") catch null;
    w.patch_embeddings_1 = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.weight.1") catch null;
    w.patch_bias = findTensorInGGUF(ctx, gguf_file, "v.patch_embd.bias") catch null;

    // 位置编码
    w.position_embeddings = findTensorInGGUF(ctx, gguf_file, "v.position_embd.weight") catch null;

    // Pre/Post LN
    w.pre_ln_w = findTensorInGGUF(ctx, gguf_file, "v.pre_ln.weight") catch null;
    w.pre_ln_b = findTensorInGGUF(ctx, gguf_file, "v.pre_ln.bias") catch null;
    w.post_ln_w = findTensorInGGUF(ctx, gguf_file, "v.post_ln.weight") catch null;
    w.post_ln_b = findTensorInGGUF(ctx, gguf_file, "v.post_ln.bias") catch null;

    // 多模态投影
    w.mm_0_w = findTensorInGGUF(ctx, gguf_file, "mm.0.weight") catch null;
    w.mm_0_b = findTensorInGGUF(ctx, gguf_file, "mm.0.bias") catch null;
    w.mm_1_w = findTensorInGGUF(ctx, gguf_file, "mm.2.weight") catch null;
    w.mm_1_b = findTensorInGGUF(ctx, gguf_file, "mm.2.bias") catch null;

    // 检测实际层数
    var actual_n_layer: u32 = 0;
    for (0..64) |il| {
        var buf: [32]u8 = undefined;
        const test_name = try std.fmt.bufPrint(&buf, "v.blk.{d}.attn_qkv.weight", .{il});
        if (gguf_file.findTensor(test_name) == null) break;
        actual_n_layer = @intCast(il + 1);
    }

    const n_layer: usize = @intCast(actual_n_layer);
    w.layers = try allocator.alloc(ViTLayerWeights, n_layer);

    for (0..n_layer) |il| {
        const prefix = try std.fmt.allocPrint(allocator, "v.blk.{d}", .{il});
        defer allocator.free(prefix);
        w.layers[il] = loadViTLayer(ctx, gguf_file, prefix) catch |err| {
            log.err("Failed to load ViT layer {d}: {}\n", .{ il, err });
            return err;
        };
    }

    // 加载 Deepstack 权重（全局命名空间 v.deepstack.{idx}.*）
    // 从 clip.vision.is_deepstack_layers 元数据读取哪些层有 deepstack
    const is_deepstack_meta = (@constCast(gguf_file)).getBoolArray("clip.vision.is_deepstack_layers");
    const is_deepstack_owned = is_deepstack_meta == null;
    const is_deepstack = is_deepstack_meta orelse try allocator.alloc(bool, 0);
    if (is_deepstack_owned) {
        defer allocator.free(is_deepstack);
    }

    // 为每个有 deepstack 的层加载权重
    for (0..n_layer) |il| {
        if (il < is_deepstack.len and is_deepstack[il]) {
            var buf: [64]u8 = undefined;
            const ds_prefix = try std.fmt.bufPrint(&buf, "v.deepstack.{d}", .{il});
            w.layers[il].deepstack_norm_w = findTensorInGGUF(ctx, gguf_file, try std.fmt.allocPrint(allocator, "{s}.norm.weight", .{ds_prefix})) catch null;
            w.layers[il].deepstack_norm_b = findTensorInGGUF(ctx, gguf_file, try std.fmt.allocPrint(allocator, "{s}.norm.bias", .{ds_prefix})) catch null;
            w.layers[il].deepstack_fc1_w = findTensorInGGUF(ctx, gguf_file, try std.fmt.allocPrint(allocator, "{s}.fc1.weight", .{ds_prefix})) catch null;
            w.layers[il].deepstack_fc1_b = findTensorInGGUF(ctx, gguf_file, try std.fmt.allocPrint(allocator, "{s}.fc1.bias", .{ds_prefix})) catch null;
            w.layers[il].deepstack_fc2_w = findTensorInGGUF(ctx, gguf_file, try std.fmt.allocPrint(allocator, "{s}.fc2.weight", .{ds_prefix})) catch null;
            w.layers[il].deepstack_fc2_b = findTensorInGGUF(ctx, gguf_file, try std.fmt.allocPrint(allocator, "{s}.fc2.bias", .{ds_prefix})) catch null;
        }
    }

    log.info("Qwen3VL weights loaded: {d} ViT layers\n", .{n_layer});
}

/// 从 VisionEncoderWeights 构建计算图的包装函数
fn buildGraphFromWeights(
    io: std.Io,
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const graph.VisionHParams,
    image_tensor: *ggml.Tensor,
) !*ggml.CGraph {
    _ = io;
    const img_buf = try image_tensor.dataGet(f32, std.heap.page_allocator);
    // Note: img_buf is intentionally leaked (leak-to-exit) since ImageF32
    // is used during graph construction and the data must remain valid.
    const img = ImageF32{
        .buf = img_buf,
        .nx = p.image_size,
        .ny = p.image_size,
    };

    var hparams = VisionHParams{
        .image_size = p.image_size,
        .patch_size = p.patch_size,
        .n_embd = p.n_embd,
        .n_head = p.n_head,
        .n_layer = p.n_layer,
        .n_ff = p.n_ff,
        .projection_dim = p.projection_dim,
        .n_merge = p.n_merge,
        .eps = p.eps,
        .rope_theta = p.rope_theta,
        .ffn_op = p.ffn_op,
    };

    var builder = GraphBuilder{
        .weights = w,
        .hparams = &hparams,
        .proj_type = .qwen3vl,
        .img = &img,
        .ctx0 = ctx,
        .gf = gf,
    };

    return buildGraph(&builder);
}

/// 估算输出 token 数量
pub fn estimateOutputTokens(io: std.Io, img_width: u32, img_height: u32, patch_size: u32, n_merge: u32) u32 {
    _ = io;
    const patches_x = (img_width + patch_size - 1) / patch_size;
    const patches_y = (img_height + patch_size - 1) / patch_size;
    const n_patches = patches_x * patches_y;
    const merge = if (n_merge > 0) n_merge else 1;
    return n_patches / (merge * merge);
}

// ============================================================================
// 权重加载辅助函数
// ============================================================================

fn findTensorInGGUF(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    const weight_loader = @import("weight_loader");
    return weight_loader.findOrCreateTensor(ctx, gguf_file, name);
}

fn loadViTLayer(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    prefix: []const u8,
) !ViTLayerWeights {
    const weight_loader = @import("weight_loader");
    var layer = ViTLayerWeights{};

    // Attention — Qwen3VL uses fused QKV (attn_qkv.weight + attn_qkv.bias)
    layer.ln_1_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln1.weight") catch null;
    layer.ln_1_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln1.bias") catch null;
    layer.qkv_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_qkv.weight") catch null;
    layer.qkv_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_qkv.bias") catch null;
    layer.o_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_out.weight") catch null;
    layer.o_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "attn_out.bias") catch null;

    // FFN — Qwen3VL ViT uses standard FFN (up + down, NO gate)
    // Note: no ffn_gate.weight in Qwen3VL mmproj
    layer.ln_2_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln2.weight") catch null;
    layer.ln_2_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ln2.bias") catch null;
    layer.ff_up_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.weight") catch null;
    layer.ff_up_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_up.bias") catch null;
    layer.ff_gate_w = null; // Qwen3VL ViT has no gate in FFN
    layer.ff_gate_b = null;
    layer.ff_down_w = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.weight") catch null;
    layer.ff_down_b = weight_loader.loadLayerWeight(ctx, gguf_file, prefix, "ffn_down.bias") catch null;

    // Deepstack weights are loaded globally in loadWeights, not per-layer here
    // (they use v.deepstack.{idx}.* naming, not v.blk.{d}.deepstack_*)

    return layer;
}

// ============================================================================
// 原始 buildGraph 函数（保留向后兼容）
// ============================================================================

/// 构建 Qwen3VL 完整计算图
///
/// 处理流程:
///   1. Temporal merge: 两个 Conv2D 相加 (patch_embeddings_0 + patch_embeddings_1)
///   2. Spatial merge: permute + reshape 合并空间维度
///   3. patch_bias 添加 (必须存在)
///   4. 可学习位置嵌入 (resize_position_embeddings, 必须存在)
///   5. Pre-LN (可选)
///   6. ViT blocks (LayerNorm + QKV fused 自注意力 + M-RoPE + FFN + Deepstack)
///   7. Post-LN (可选)
///   8. 多模态投影 (FFN) + Deepstack concat
///
/// 参考: llama.cpp qwen3vl.cpp build()
pub fn buildGraph(
    builder: *GraphBuilder,
) !*ggml.CGraph {
    const ctx = builder.ctx0;
    const w = builder.weights;
    const p = builder.hparams;
    const img = builder.img;

    const n_embd: i64 = @intCast(p.n_embd);
    const n_head: i64 = @intCast(p.n_head);
    const d_head = @divExact(n_embd, n_head);
    const img_width: u32 = p.image_size;
    const img_height: u32 = p.image_size;
    const patch_size: i32 = @intCast(p.patch_size);
    const n_patches_x: i64 = @divTrunc(@as(i64, @intCast(img_width)), patch_size);
    const n_patches_y: i64 = @divTrunc(@as(i64, @intCast(img_height)), patch_size);
    const n_patches: i64 = n_patches_x * n_patches_y;
    const n_batch: i64 = 1;
    const eps = p.eps;
    const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    // Reference: norm_type norm_t = NORM_TYPE_NORMAL; (LayerNorm)
    const norm_t: NormType = .layer_norm;

    // Reference: int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};
    const mrope_sections = [_]i32{
        @intCast(@divExact(d_head, @as(i64, 4))),
        @intCast(@divExact(d_head, @as(i64, 4))),
        @intCast(@divExact(d_head, @as(i64, 4))),
        @intCast(@divExact(d_head, @as(i64, 4))),
    };

    // Reference: const int merge_factor = hparams.n_merge > 0 ? hparams.n_merge * hparams.n_merge : 4;
    const merge_factor: i64 = if (p.n_merge > 0) @intCast(p.n_merge * p.n_merge) else 4;

    log.info("Qwen3VL graph: embd={d}, head={d}, d_head={d}, patches={d}x{d}={d}, merge_factor={d}\n", .{ n_embd, n_head, d_head, n_patches_x, n_patches_y, n_patches, merge_factor });

    // Reference: GGML_ASSERT(model.patch_bias != nullptr);
    // Reference: GGML_ASSERT(model.position_embeddings != nullptr);
    // Reference: GGML_ASSERT(model.class_embedding == nullptr);
    if (w.patch_bias == null) return error.MissingPatchBias;
    if (w.position_embeddings == null) return error.MissingPositionEmbeddings;

    // ============================================================================
    // 1. Temporal merge: build_inp_with_temporal_merge()
    // Reference: ggml_tensor * inp = build_inp_with_temporal_merge();
    // ============================================================================
    var inp: *ggml.Tensor = undefined;
    {
        // Reference: build_inp_raw() -> ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx(), img.ny(), 3, n_batch)
        const inp_raw = try ctx.newTensor4d(ggml.Type.f32, @as(i64, @intCast(img_width)), @as(i64, @intCast(img_height)), 3, n_batch);
        inp_raw.setName("inp_raw");

        // In no_alloc mode, the tensor data pointer is NULL.
        // We need to allocate the data manually so we can write to it.
        const no_alloc = ctx.getNoAlloc();
        if (no_alloc) {
            const data_size = @as(usize, @intCast(inp_raw.nBytes()));
            const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
            @memset(buf, 0);
            inp_raw.setDataPtr(buf);
        }

        {
            const n_elems = @as(usize, @intCast(inp_raw.nElems()));
            const dst = try std.heap.page_allocator.alloc(f32, n_elems);
            defer std.heap.page_allocator.free(dst);
            const src = img.buf;
            const H: usize = @intCast(img_height);
            const W: usize = @intCast(img_width);
            // HWC->CHW conversion: src is HWC layout, inp_raw expects CHW layout
            // Ref: llama.cpp clip.cpp lines 3645-3665
            for (0..H) |y| {
                for (0..W) |x| {
                    const hwc_idx = (y * W + x) * 3;
                    const chw_base = y * W + x;
                    dst[chw_base] = src[hwc_idx]; // R
                    dst[chw_base + H * W] = src[hwc_idx + 1]; // G
                    dst[chw_base + 2 * H * W] = src[hwc_idx + 2]; // B
                }
            }
            try inp_raw.dataSet(f32, dst);
        }

        // Reference: ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1)

        // Reference: ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1)
        // Reference: ggml_add(ctx0, conv0, conv1)
        const pe0 = w.patch_embeddings_0 orelse return error.MissingPatchEmbedding;
        const kw: i32 = @intCast(pe0.ne()[0]);
        const kh: i32 = @intCast(pe0.ne()[1]);
        var conv0 = inp_raw.conv2d(ctx, pe0, kw, kh, 0, 0, 1, 1);
        conv0.setName("conv0");

        if (w.patch_embeddings_1) |pe1| {
            var conv1 = inp_raw.conv2d(ctx, pe1, kw, kh, 0, 0, 1, 1);
            conv1.setName("conv1");
            inp = conv0.add(ctx, conv1);
            inp.setName("temporal_merge");
        } else {
            inp = conv0;
        }
    }

    // ============================================================================
    // 2. Spatial merge
    // Reference: ggml_permute(ctx0, inp, 1, 2, 0, 3) -> [w, h, c, b] -> [c, w, h, b]
    // ============================================================================
    // 2. Spatial merge
    // Reference: ggml_permute(ctx0, inp, 1, 2, 0, 3) -> [w, h, c, b] -> [c, w, h, b]
    // ============================================================================
    {
        inp = inp.permute(ctx, 1, 2, 0, 3).cont(ctx);
        inp.setName("spatial_permuted");

        // Reference: ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size)
        inp = inp.cont4d(ctx, n_embd * 2, @divTrunc(n_patches_x, @as(i64, 2)), n_patches_y, n_batch);
        inp.setName("spatial_reshaped_1");

        // Reference: ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2))
        inp = inp.reshape4d(ctx, n_embd * 2, @divTrunc(n_patches_x, @as(i64, 2)), 2, n_batch * @divTrunc(n_patches_y, @as(i64, 2)));
        inp.setName("spatial_reshaped_2");

        // Reference: ggml_permute(ctx0, inp, 0, 2, 1, 3)
        inp = inp.permute(ctx, 0, 2, 1, 3).cont(ctx);
        inp.setName("spatial_permuted_2");

        // Reference: ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size)
        inp = ggml.cont(ctx, inp).reshape3d(ctx, n_embd, n_patches_x * n_patches_y, n_batch);
        inp.setName("spatial_merged");
        inp.setName("spatial_reshaped_1");

        // Reference: ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2))
        inp = inp.reshape4d(ctx, n_embd * 2, @divExact(n_patches_x, @as(i64, 2)), 2, n_batch * @divExact(n_patches_y, @as(i64, 2)));
        inp.setName("spatial_reshaped_2");

        // Reference: ggml_permute(ctx0, inp, 0, 2, 1, 3)
        inp = inp.permute(ctx, 0, 2, 1, 3).cont(ctx);
        inp.setName("spatial_permuted_2");

        // Reference: ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size)
        inp = ggml.cont(ctx, inp).reshape3d(ctx, n_embd, n_patches_x * n_patches_y, n_batch);
        inp.setName("spatial_merged");
    }

    // ============================================================================
    // 3. Add patch bias
    // Reference: if (model.patch_bias != nullptr) { inp = ggml_add(ctx0, inp, model.patch_bias); }
    // ============================================================================
    if (w.patch_bias) |pb| {
        inp = inp.add(ctx, pb);
        inp.setName("patch_bias");
    }

    // ============================================================================
    // 4. 可学习位置嵌入 (resize_position_embeddings)
    // Reference: ggml_tensor * learned_pos_embd = resize_position_embeddings();
    // ============================================================================
    if (w.position_embeddings) |pos_embd| {
        // Reference: resize_position_embeddings() — interpolates to match n_patches
        var learned_pos_embd = try graph.resizePositionEmbeddings(ctx, pos_embd, n_patches, 0);

        // Apply same spatial merge to position embeddings
        // Reference: ggml_cont_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size)
        learned_pos_embd = learned_pos_embd.cont4d(ctx, n_embd * 2, @divExact(n_patches_x, @as(i64, 2)), n_patches_y, n_batch);
        learned_pos_embd.setName("pos_embd_reshaped_1");

        // Reference: ggml_reshape_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2))
        learned_pos_embd = learned_pos_embd.reshape4d(ctx, n_embd * 2, @divExact(n_patches_x, @as(i64, 2)), 2, n_batch * @divExact(n_patches_y, @as(i64, 2)));
        learned_pos_embd.setName("pos_embd_reshaped_2");

        // Reference: ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3)
        learned_pos_embd = learned_pos_embd.permute(ctx, 0, 2, 1, 3).cont(ctx);
        learned_pos_embd.setName("pos_embd_permuted");

        // Reference: ggml_cont_3d(ctx0, learned_pos_embd, n_embd, n_patches_x * n_patches_y, batch_size)
        learned_pos_embd = ggml.cont(ctx, learned_pos_embd).reshape3d(ctx, n_embd, n_patches_x * n_patches_y, n_batch);
        learned_pos_embd.setName("pos_embd_merged");

        // Reference: inp = ggml_add(ctx0, inp, learned_pos_embd);
        inp = inp.add(ctx, learned_pos_embd);
        inp.setName("inp_pos_emb");
    }

    var inpL = inp;

    // ============================================================================
    // 5. M-RoPE positions
    // Reference: ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    // ============================================================================
    const num_position_ids = n_patches * 4;
    const positions = try ctx.newTensor1d(ggml.Type.i32, num_position_ids);
    positions.setName("positions");

    // In no_alloc mode, the tensor data pointer is NULL.
    // We need to allocate the data manually so we can write to it.
    const no_alloc_pos = ctx.getNoAlloc();
    if (no_alloc_pos) {
        const data_size = @as(usize, @intCast(positions.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(data_size) orelse return error.OutOfMemory))[0..data_size];
        @memset(buf, 0);
        positions.setDataPtr(buf);
    }

    ggml.setInput(positions);

    {
        const data = positions.dataI32();
        for (0..@as(usize, @intCast(n_patches))) |i| {
            const pi: i32 = @intCast(i);
            data[i * 4 + 0] = pi;
            data[i * 4 + 1] = pi;
            data[i * 4 + 2] = pi;
            data[i * 4 + 3] = pi;
        }
    }
    // ============================================================================
    // 6. Pre-LN (optional)
    // Reference: if (model.pre_ln_w) { inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1); }
    // ============================================================================
    if (w.pre_ln_w) |pln_w| {
        inpL = try graph.buildNorm(ctx, inpL, pln_w, w.pre_ln_b, norm_t, eps, -1);
    }

    // ============================================================================
    // 7. Deepstack features
    // Reference: ggml_tensor * deepstack_features = nullptr;
    // ============================================================================
    var deepstack_features: ?*ggml.Tensor = null;

    // ============================================================================
    // 8. ViT blocks
    // Reference: for (int il = 0; il < n_layer; il++)
    // ============================================================================
    for (w.layers, 0..) |*layer, il| {
        var cur = inpL; // inpL = residual, cur = hidden_states

        // Reference: cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cur = try graph.buildNorm(ctx, cur, layer.ln_1_w orelse return error.MissingNormWeight, layer.ln_1_b, norm_t, eps, @intCast(il));

        // Reference: self-attention block
        {
            // Reference: cur = build_mm(layer.qkv_w, cur);
            // Reference: cur = ggml_add(ctx0, cur, layer.qkv_b);
            // Use ggml.mulMat directly since Qwen3VL doesn't use clamping
            cur = ggml.mulMat(ctx, layer.qkv_w orelse return error.MissingQKVWeight, cur);
            if (layer.qkv_b) |qkv_b| {
                cur = cur.add(ctx, qkv_b);
            }

            // Reference: ggml_view_3d for Q, K, V split
            // Q: offset 0, K: offset n_embd, V: offset 2*n_embd
            const row_size = ggml.Type.rowSize(cur.dataType(), d_head);
            const nb1 = cur.nb()[1];

            var Qcur = ctx.view3d(cur, d_head, n_head, n_patches, row_size, nb1, 0);
            Qcur.setName("Qcur");

            var Kcur = ctx.view3d(cur, d_head, n_head, n_patches, row_size, nb1, @as(usize, @intCast(ggml.Type.rowSize(cur.dataType(), n_embd))));
            Kcur.setName("Kcur");

            var Vcur = ctx.view3d(cur, d_head, n_head, n_patches, row_size, nb1, @as(usize, @intCast(ggml.Type.rowSize(cur.dataType(), 2 * n_embd))));
            Vcur.setName("Vcur");

            // Reference: M-RoPE
            // ggml_rope_multi(ctx0, Qcur, positions, nullptr, d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1)
            const rope_type_vision: i32 = 24; // GGML_ROPE_TYPE_VISION
            Qcur = ggml.ropeMulti(ctx, Qcur, positions, @intCast(@divExact(d_head, @as(i64, 2))), &mrope_sections, rope_type_vision, 32768, 10000, 1, 0, 1, 32, 1);
            Qcur.setName("Qcur_rope");
            Kcur = ggml.ropeMulti(ctx, Kcur, positions, @intCast(@divExact(d_head, @as(i64, 2))), &mrope_sections, rope_type_vision, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur.setName("Kcur_rope");

            // Reference: cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cur = try graph.buildAttn(
                ctx,
                builder.gf,
                layer.o_w orelse return error.MissingOutputWeight,
                layer.o_b,
                Qcur,
                Kcur,
                Vcur,
                null,
                kq_scale,
                @intCast(il),
                layer.attn_sinks,
                builder.flash_attn_type,
                defaultBuildMM,
                null,
                null,
            );
        }

        cur = try graph.buildNorm(ctx, cur, layer.ln_2_w orelse return error.MissingNormWeight, layer.ln_2_b, norm_t, eps, -1);

        // Reference: ffn
        // build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il)
        cur = try graph.buildFFN(
            ctx,
            cur,
            layer.ff_up_w,
            layer.ff_up_b,
            layer.ff_gate_w,
            layer.ff_gate_b,
            layer.ff_down_w,
            layer.ff_down_b,
            p.ffn_op,
            @intCast(il),
            defaultBuildMM,
            null,
            null,
        );

        // Reference: cur = ggml_add(ctx0, inpL, cur); (residual 2)
        cur = inpL.add(ctx, cur);
        inpL = cur;

        // Reference: Deepstack feature extraction
        // if (layer.has_deepstack()) { ... }
        if (layer.hasDeepstack()) {
            // Reference: ggml_reshape_3d(ctx0, cur, n_embd * merge_factor, n_pos / merge_factor, batch_size)
            var feat = ggml.cont(ctx, cur).reshape3d(ctx, n_embd * merge_factor, @divExact(n_patches, merge_factor), n_batch);
            feat.setName("deepstack_reshape");

            // Reference: build_norm(feat, layer.deepstack_norm_w, layer.deepstack_norm_b, norm_t, eps, il)
            feat = try graph.buildNorm(ctx, feat, layer.deepstack_norm_w orelse return error.MissingNormWeight, layer.deepstack_norm_b, norm_t, eps, -1);
            feat = try graph.buildFFN(
                ctx,
                feat,
                layer.deepstack_fc1_w,
                layer.deepstack_fc1_b,
                null,
                null,
                layer.deepstack_fc2_w,
                layer.deepstack_fc2_b,
                .gelu,
                -1,
                defaultBuildMM,
                null,
                null,
            );

            // Reference: concat along feature dimension
            // Reference: concat along feature dimension
            if (deepstack_features) |dsf| {
                deepstack_features = dsf.concat(ctx, feat, 0);
                deepstack_features.?.setName("deepstack_concat");
            } else {
                deepstack_features = feat;
            }
        }
    }

    // ============================================================================
    // 9. Post-LN (optional)
    // Reference: if (model.post_ln_w) { inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer); }
    // ============================================================================
    if (w.post_ln_w) |poln_w| {
        inpL = try graph.buildNorm(ctx, inpL, poln_w, w.post_ln_b, norm_t, eps, -1);
    }

    // ============================================================================
    // 10. Multimodal projection
    // Reference: ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size)
    // ============================================================================
    var embeddings = inpL;
    embeddings = ggml.cont(ctx, embeddings).reshape3d(ctx, n_embd * 4, @divExact(n_patches, @as(i64, 4)), n_batch);
    embeddings.setName("mm_reshape");

    embeddings = try graph.buildFFN(
        ctx,
        embeddings,
        w.mm_0_w,
        w.mm_0_b,
        null,
        null,
        w.mm_1_w,
        w.mm_1_b,
        .gelu,
        -1,
        defaultBuildMM,
        null,
        null,
    );
    embeddings.setName("mm_proj");
    if (deepstack_features) |dsf| {
        embeddings = embeddings.concat(ctx, dsf, 0);
        embeddings.setName("mm_with_deepstack");
    }

    embeddings.setName("mm_output");

    // Reference: ggml_build_forward_expand(gf, embeddings);
    builder.gf.buildForwardExpand(embeddings);

    log.info("Qwen3VL graph built successfully\n", .{});
    return builder.gf;
}
