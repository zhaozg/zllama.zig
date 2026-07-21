//! Gemma4V visual encoder graph builder.
//!
//! 1:1 translation of deps/llama.cpp/tools/mtmd/models/gemma4v.cpp
//!
//! Pipeline:
//!   1. build_inp_raw() -> 4D [W, H, C, B] f32 tensor, set_input
//!   2. scale_bias(inp_raw, 2.0, -1.0)  — patches = 2 * (patches - 0.5)
//!   3. Conv2D patch embedding (no patch_bias)
//!   4. reshape_3d -> [n_patches, n_embd, B], then cont(transpose) -> [n_embd, n_patches, B]
//!   5. 2D position embeddings (pos_x/pos_y lookup tables)
//!   6. ViT blocks (RMS norm) with 2D RoPE (NEOX ordering), V RMSNorm
//!   7. Pool 2D (avg, kernel=n_merge) + scale(sqrt(n_embd))
//!   8. Standardization: (hidden - std_bias) * std_scale
//!   9. Multimodal embedder: rms_norm -> build_mm(mm_input_proj_w) with clamp
//!
//! NOTE: gemma4v does NOT use mm_soft_emb_norm_w (that's gemma3 only).

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const graph = @import("../mod.zig");
const vit_builder = @import("../vit.zig");
const weight_loader = @import("weight_loader");

const BuildVitOpts = graph.BuildVitOpts;
const VisionHParams = graph.VisionHParams;
const VisionEncoderWeights = graph.VisionEncoderWeights;
const ViTLayerWeights = graph.ViTLayerWeights;
const ClampInfo = graph.ClampInfo;
const GraphBuilder = graph.GraphBuilder;
const ImageF32 = graph.ImageF32;

const log = std.log.scoped(.graph_model_gemma4v);

// ============================================================================
pub const backend = graph.VisionEncoderBackend{
    .name = "gemma4v",
    .supportBatch = true,
    .buildGraph = buildGraphFromWeights,
    .loadParams = loadParams,
    .loadWeights = loadWeights,
    .loadClampInfo = loadClampInfo,
    .estimateOutputTokens = estimateOutputTokens,
};

// ============================================================================
// Params (ref: clip.cpp GEMMA4V case)
// ============================================================================

pub fn loadParams(io: std.Io, gf: *const gguf.GGUFFile, p: *VisionHParams) void {
    _ = io;
    p.rope_theta = 100.0;
    p.n_merge = 3;
    if (gf.getU32("clip.vision.projector.scale_factor")) |v| p.n_merge = v else if (gf.getU32("gemma4.vision.projector.scale_factor")) |v| p.n_merge = v;
    p.setLimitImageTokens(40, 280);
    p.warmup_image_size = 256;
    log.info("Gemma4V: n_merge={d} rope_theta={d:.1}", .{ p.n_merge, p.rope_theta });
}

// ============================================================================
// Weight Loading
// ============================================================================

fn loadLayer(ctx: *ggml.Context, gf: *const gguf.GGUFFile, prefix: []const u8) !ViTLayerWeights {
    var l = ViTLayerWeights{};

    // Attention weights (required)
    l.q_w = try weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_q.weight");
    l.k_w = try weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_k.weight");
    l.v_w = try weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_v.weight");
    l.o_w = try weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_out.weight");

    // Attention biases (optional)
    l.q_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_q.bias") catch null;
    l.k_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_k.bias") catch null;
    l.v_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_v.bias") catch null;
    l.o_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_out.bias") catch null;

    // FFN weights (required)
    l.ff_up_w = try weight_loader.loadLayerWeight(ctx, gf, prefix, "ffn_up.weight");
    l.ff_gate_w = try weight_loader.loadLayerWeight(ctx, gf, prefix, "ffn_gate.weight");
    l.ff_down_w = try weight_loader.loadLayerWeight(ctx, gf, prefix, "ffn_down.weight");

    // FFN biases (optional)
    l.ff_up_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "ffn_up.bias") catch null;
    l.ff_gate_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "ffn_gate.bias") catch null;
    l.ff_down_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "ffn_down.bias") catch null;

    // LayerNorm weights & biases (optional)
    l.ln_1_w = weight_loader.loadLayerWeight(ctx, gf, prefix, "ln1.weight") catch null;
    l.ln_1_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "ln1.bias") catch null;
    l.ln_2_w = weight_loader.loadLayerWeight(ctx, gf, prefix, "ln2.weight") catch null;
    l.ln_2_b = weight_loader.loadLayerWeight(ctx, gf, prefix, "ln2.bias") catch null;

    // Post-attn / post-FFN norms (optional, gemma4-specific)
    l.attn_post_norm_w = weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_post_norm.weight") catch null;
    l.ff_post_norm_w = weight_loader.loadLayerWeight(ctx, gf, prefix, "ffn_post_norm.weight") catch null;

    // Per-head Q/K norms (optional)
    l.k_norm = weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_k_norm.weight") catch null;
    l.q_norm = weight_loader.loadLayerWeight(ctx, gf, prefix, "attn_q_norm.weight") catch null;

    // Layer scales (optional, gemma4-specific)
    l.ls_1_w = weight_loader.loadLayerWeight(ctx, gf, prefix, "ls1.weight") catch null;
    l.ls_2_w = weight_loader.loadLayerWeight(ctx, gf, prefix, "ls2.weight") catch null;
    l.ls_out_w = weight_loader.loadLayerWeight(ctx, gf, prefix, "out_scale.weight") catch null;

    return l;
}

pub fn loadWeights(io: std.Io, alloc: std.mem.Allocator, gf: *const gguf.GGUFFile, ctx: *ggml.Context, w: *VisionEncoderWeights) anyerror!void {
    // Load patch embedding weight (required)
    log.debug("Loading patch_embeddings_0 (v.patch_embd.weight)...", .{});
    w.patch_embeddings_0 = try weight_loader.findOrCreateTensor(ctx, gf, "v.patch_embd.weight");
    if (w.patch_embeddings_0) |t| {
        log.debug("  patch_embeddings_0 loaded: shape=[{d},{d},{d},{d}] type={s}", .{ t.ne()[0], t.ne()[1], t.ne()[2], t.ne()[3], @tagName(t.dataType()) });
    }

    // Load position embedding weight (required)
    log.debug("Loading position_embeddings (v.position_embd.weight)...", .{});
    w.position_embeddings = try weight_loader.findOrCreateTensor(ctx, gf, "v.position_embd.weight");
    if (w.position_embeddings) |t| {
        log.debug("  position_embeddings loaded: shape=[{d},{d},{d},{d}] type={s}", .{ t.ne()[0], t.ne()[1], t.ne()[2], t.ne()[3], @tagName(t.dataType()) });
    }

    // Load standardization weights (optional)
    log.debug("Loading std_bias (v.std_bias)...", .{});
    w.std_bias = weight_loader.findOrCreateTensor(ctx, gf, "v.std_bias") catch null;
    if (w.std_bias) |t| {
        log.debug("  std_bias loaded: shape=[{d},{d},{d},{d}] type={s}", .{ t.ne()[0], t.ne()[1], t.ne()[2], t.ne()[3], @tagName(t.dataType()) });
    } else {
        log.debug("  std_bias NOT found (optional)", .{});
    }

    log.debug("Loading std_scale (v.std_scale)...", .{});
    w.std_scale = weight_loader.findOrCreateTensor(ctx, gf, "v.std_scale") catch null;
    if (w.std_scale) |t| {
        log.debug("  std_scale loaded: shape=[{d},{d},{d},{d}] type={s}", .{ t.ne()[0], t.ne()[1], t.ne()[2], t.ne()[3], @tagName(t.dataType()) });
    } else {
        log.debug("  std_scale NOT found (optional)", .{});
    }

    // Load multimodal projection weight (required)
    log.debug("Loading mm_input_proj_w (mm.input_projection.weight)...", .{});
    w.mm_input_proj_w = try weight_loader.findOrCreateTensor(ctx, gf, "mm.input_projection.weight");
    if (w.mm_input_proj_w) |t| {
        log.debug("  mm_input_proj_w loaded: shape=[{d},{d},{d},{d}] type={s}", .{ t.ne()[0], t.ne()[1], t.ne()[2], t.ne()[3], @tagName(t.dataType()) });
    }

    // NOTE: gemma4v does NOT use mm_soft_emb_norm_w (that's gemma3 only)

    // Detect ViT layer count from GGUF
    var n: u32 = 0;
    for (0..64) |il| {
        var buf: [32]u8 = undefined;
        if (gf.findTensor(try std.fmt.bufPrint(&buf, "v.blk.{d}.attn_q.weight", .{il})) == null) break;
        n = @intCast(il + 1);
    }
    log.debug("Detected {d} ViT layers from GGUF", .{n});

    // Load all ViT layer weights
    w.layers = try alloc.alloc(ViTLayerWeights, n);
    for (0..n) |il| {
        const pfx = try std.fmt.allocPrint(alloc, "v.blk.{d}", .{il});
        defer alloc.free(pfx);
        log.debug("Loading ViT layer {d}/{d} (prefix: {s})...", .{ il + 1, n, pfx });
        w.layers[il] = try loadLayer(ctx, gf, pfx);
        log.debug("  -> ViT layer {d}/{d} loaded successfully", .{ il + 1, n });
    }
    log.info("Gemma4V: {d} layers loaded", .{n});

    if (w.layers[0].ln_1_w) |t| {
        try graph.debug.saveTensor(io, alloc, "debug_vision", "zllama_vision_04a_1_layer0_ln_1_w.json", t);
    }
    if (w.layers[0].ln_1_b) |t| {
        try graph.debug.saveTensor(io, alloc, "debug_vision", "zllama_vision_04a_2_layer0_ln_1_b.json", t);
    }
}

pub fn loadClampInfo(io: std.Io, allocator: std.mem.Allocator, gf: *const gguf.GGUFFile, w: *VisionEncoderWeights) anyerror!void {
    _ = io;
    var weight_names = std.ArrayList([]const u8).initCapacity(allocator, 0) catch |err| return err;
    defer weight_names.deinit(allocator);

    // C++: for (auto * tensor : tensors_to_load) {
    //         if (string_ends_with(name, ".weight")) {
    //             ... load clamp info for all weights ...
    //         }
    //       }
    // In zllama, we need to collect all weight tensor names that could have clamp info.
    // These include: patch_embd, position_embd, all layer weights (qkv, ff_up, ff_gate, ff_down, ln_1, ln_2, etc.),
    // and mm_input_proj_w.

    // Add mm_input_proj_w (always has clamp info in gemma4v)
    if (w.mm_input_proj_w) |t| try weight_names.append(allocator, t.getName());

    // Add all layer weights that end with ".weight"
    for (w.layers) |layer| {
        if (layer.qkv_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.q_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.k_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.v_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.o_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_up_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_gate_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_down_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ln_1_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ln_2_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.attn_post_norm_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.ff_post_norm_w) |t| try weight_names.append(allocator, t.getName());
        if (layer.k_norm) |t| try weight_names.append(allocator, t.getName());
        if (layer.q_norm) |t| try weight_names.append(allocator, t.getName());
    }

    w.clamp_info_map = try graph.clamp.loadClampInfoFromWeightNames(allocator, gf, weight_names.items);
    log.info("Gemma4V clamp info loaded: {d} entries", .{w.clamp_info_map.count()});
}

// ============================================================================
// Output Estimate
// ============================================================================

pub fn estimateOutputTokens(io: std.Io, iw: u32, ih: u32, ps: u32, nm: u32) u32 {
    _ = io;
    const m: u32 = if (nm == 0) 1 else nm;
    return (iw / ps / m) * (ih / ps / m);
}

// ============================================================================
// Gemma4V 模型私有数据
// ============================================================================

/// Gemma4V 模型私有数据，通过 BuildVitOpts.data 传递给回调函数
pub const Gemma4VData = struct {
    pos_x: *ggml.Tensor,
    pos_y: *ggml.Tensor,
    freq_base: f32,
    clamp_info_map: *const std.StringHashMap(ClampInfo),
};

/// 2D RoPE callback for ViT blocks.
/// 1:1 translation of deps/llama.cpp/tools/mtmd/models/gemma4v.cpp add_pos lambda
///
/// C++ lambda:
///   auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
///       const int64_t n_dim  = cur->ne[0];
///       const int64_t n_head = cur->ne[1];
///       const int64_t n_pos  = cur->ne[2];
///       // first half: view4d + rope_ext(NEOX)
///       // second half: view4d + rope_ext(NEOX)
///       // concat
///   };
fn addPos(ctx: *ggml.Context, cur: *ggml.Tensor, _: *const ViTLayerWeights, data: ?*anyopaque) *ggml.Tensor {
    const d = @as(*const Gemma4VData, @ptrCast(@alignCast(data orelse unreachable)));

    // C++: const int64_t n_dim  = cur->ne[0];
    const n_dim = cur.ne()[0];
    // C++: const int64_t n_head = cur->ne[1];
    const n_head = cur.ne()[1];
    // C++: const int64_t n_pos  = cur->ne[2];
    const n_pos = cur.ne()[2];
    // n_batch is captured from outer scope in C++ lambda [&]
    const n_batch = cur.ne()[3];

    // C++: const int64_t n_dim/2
    const n_dim_half = @divExact(n_dim, 2);

    // ---- first half ----
    // C++: first = ggml_view_4d(ctx0, cur, n_dim/2, n_head, n_pos, n_batch,
    //       cur->nb[1], cur->nb[2], cur->nb[3], 0);
    var first = cur.view4d(ctx, n_dim_half, n_head, n_pos, n_batch, cur.nb()[1], cur.nb()[2], cur.nb()[3], 0);

    // C++: first = ggml_rope_ext(ctx0, first, pos_x, nullptr,
    //       n_dim/2, GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta,
    //       1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    // GGML_ROPE_TYPE_NEOX = 2
    first = first.ropeExt(ctx, d.pos_x, null, @intCast(n_dim_half), 2, // GGML_ROPE_TYPE_NEOX
        0, // n_ctx_orig
        d.freq_base, // freq_base
        1.0, // freq_scale
        0.0, // ext_factor
        1.0, // attn_factor
        0.0, // beta_fast
        0.0 // beta_slow
    );

    // ---- second half ----
    // C++: second = ggml_view_4d(ctx0, cur, n_dim/2, n_head, n_pos, n_batch,
    //       cur->nb[1], cur->nb[2], cur->nb[3],
    //       n_dim/2 * ggml_element_size(cur));
    var second = cur.view4d(ctx, n_dim_half, n_head, n_pos, n_batch, cur.nb()[1], cur.nb()[2], cur.nb()[3], @as(usize, @intCast(n_dim_half)) * cur.elementSize());

    // C++: second = ggml_rope_ext(ctx0, second, pos_y, nullptr,
    //       n_dim/2, GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta,
    //       1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    second = second.ropeExt(ctx, d.pos_y, null, @intCast(n_dim_half), 2, // GGML_ROPE_TYPE_NEOX
        0, // n_ctx_orig
        d.freq_base, // freq_base
        1.0, // freq_scale
        0.0, // ext_factor
        1.0, // attn_factor
        0.0, // beta_fast
        0.0 // beta_slow
    );

    // C++: cur = ggml_concat(ctx0, first, second, 0);
    return first.concat(ctx, second, 0);
}

// ============================================================================
// Graph
// ============================================================================

/// Create position index tensors for 2D position embeddings.
/// Corresponds to C++: ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches) + set_input
/// The data is filled with x/y indices for each patch position.
fn posIndices(ctx: *ggml.Context, nx: i32, ny: i32, comptime d: enum { x, y }) !*ggml.Tensor {
    const n: usize = @intCast(@as(i32, nx * ny));
    const data = try std.heap.page_allocator.alloc(i32, n);
    defer std.heap.page_allocator.free(data);
    var idx: usize = 0;
    switch (d) {
        .x => for (0..@as(usize, @intCast(ny))) |_| {
            for (0..@as(usize, @intCast(nx))) |x| {
                data[idx] = @intCast(x);
                idx += 1;
            }
        },
        .y => for (0..@as(usize, @intCast(ny))) |y| {
            for (0..@as(usize, @intCast(nx))) |_| {
                data[idx] = @intCast(y);
                idx += 1;
            }
        },
    }
    const nm: [:0]const u8 = if (d == .x) "pos_x" else "pos_y";
    var t = try ctx.newTensor1d(ggml.Type.i32, @intCast(n));
    t.setName(nm);
    ggml.setInput(t);
    if (ctx.getNoAlloc()) {
        const sz = @as(usize, @intCast(t.nBytes()));
        const buf = @as([*]u8, @ptrCast(std.c.malloc(sz) orelse return error.OutOfMemory))[0..sz];
        @memcpy(buf, @as([*]const u8, @ptrCast(data.ptr))[0..sz]);
        t.setDataPtr(buf);
    } else {
        try t.dataSet(i32, data);
    }
    return t;
}

/// build_mm callback with clamp support.
/// Corresponds to C++ clip_graph_gemma4v::build_mm()
/// data points to Gemma4VData (for clamp_info_map)
fn buildMMWithClamp(ctx: *ggml.Context, wt: *ggml.Tensor, x: *ggml.Tensor, data: ?*anyopaque) *ggml.Tensor {
    const d = @as(*const Gemma4VData, @ptrCast(@alignCast(data orelse unreachable)));
    if (d.clamp_info_map.get(wt.getName())) |ci| {
        // C++: ggml_tensor * clamped = ggml_clamp(ctx0, x, clamp_info.inp_min, clamp_info.inp_max);
        //       ggml_tensor * out = ggml_mul_mat(ctx0, w, clamped);
        //       out = ggml_clamp(ctx0, out, clamp_info.out_min, clamp_info.out_max);
        return wt.mulMat(ctx, x.clamp(ctx, ci.inp_min, ci.inp_max)).clamp(ctx, ci.out_min, ci.out_max);
    }
    // C++: return ggml_mul_mat(ctx0, w, x);
    return wt.mulMat(ctx, x);
}

/// Direct build_mm with clamp info (used for final mm_input_proj_w projection).
fn buildMMWithClampDirect(ctx: *ggml.Context, wt: *ggml.Tensor, x: *ggml.Tensor, cm: *const std.StringHashMap(ClampInfo)) *ggml.Tensor {
    if (cm.get(wt.getName())) |ci| {
        return wt.mulMat(ctx, x.clamp(ctx, ci.inp_min, ci.inp_max)).clamp(ctx, ci.out_min, ci.out_max);
    }
    return wt.mulMat(ctx, x);
}

/// Fill the inp_raw tensor with image pixel data, performing HWC→CHW conversion.
///
/// The image tensor (image_tensor) is created by resizeAndNormalize as a 3D [W, H, C] tensor
/// with HWC layout (matching llama.cpp convention):
///   ne[0]=W (innermost), ne[1]=H, ne[2]=C
/// Memory: [R0,G0,B0, R1,G1,B1, ..., R_n,G_n,B_n]
///
/// The inp_raw tensor is 4D [W, H, C, B] with CHW layout (ggml column-major):
///   ne[0]=W, ne[1]=H, ne[2]=C, ne[3]=B
/// Memory: [R0,R1,..., G0,G1,..., B0,B1,...]
///
/// So we need HWC->CHW conversion when filling inp_raw.
/// Ref: clip.cpp lines 3645-3665 does HWC->CHW conversion.
fn fillInpRawFromImage(
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    img: *const ImageF32,
    img_w: u32,
    img_h: u32,
) !void {
    const inp_raw = gf.getTensor("inp_raw") orelse return error.TensorNotFound;
    const W: usize = @intCast(img_w);
    const H: usize = @intCast(img_h);
    const n = W * H;
    const src = img.buf;

    // In no_alloc mode, we must allocate the tensor data manually before calling dataSet.
    if (ctx.getNoAlloc()) {
        const sz = @as(usize, @intCast(inp_raw.nBytes()));
        const data_buf = @as([*]u8, @ptrCast(std.c.malloc(sz) orelse return error.OutOfMemory))[0..sz];
        inp_raw.setDataPtr(data_buf);
    }

    // Allocate temporary buffer
    const n_elems = @as(usize, @intCast(inp_raw.nElems()));
    const dst = try std.heap.page_allocator.alloc(f32, n_elems);
    defer std.heap.page_allocator.free(dst);

    // HWC->CHW conversion: src is HWC layout, inp_raw expects CHW layout
    // Ref: clip.cpp lines 3645-3665
    for (0..H) |y| {
        for (0..W) |x| {
            const hwc_idx = (y * W + x) * 3;
            const chw_base = y * W + x;
            dst[chw_base] = src[hwc_idx]; // R channel
            dst[chw_base + n] = src[hwc_idx + 1]; // G channel
            dst[chw_base + 2 * n] = src[hwc_idx + 2]; // B channel
        }
    }

    std.debug.print("inp_raw: {} {} {} {}\n", .{ dst[0], dst[1], dst[2], dst[3] });
    try inp_raw.dataSet(f32, dst);
}

/// Build Gemma4V full compute graph.
///
/// 1:1 translation of deps/llama.cpp/tools/mtmd/models/gemma4v.cpp clip_graph_gemma4v::build()
///
/// Pipeline (matching C++ exactly):
///   1. build_inp_raw() -> 4D [W, H, C, B] f32 tensor, set_input
///   2. scale_bias(inp_raw, 2.0, -1.0)  — patches = 2 * (patches - 0.5)
///   3. Conv2D patch embedding (no patch_bias)
///   4. reshape_3d -> [n_patches, n_embd, B], then cont(transpose) -> [n_embd, n_patches, B]
///   5. 2D position embeddings (pos_x/pos_y lookup tables)
///   6. ViT blocks (RMS norm) with 2D RoPE (NEOX ordering), V RMSNorm
///   7. Pool 2D (avg, kernel=n_merge) + scale(sqrt(n_embd))
///   8. Standardization: (hidden - std_bias) * std_scale
///   9. Multimodal embedder: rms_norm -> build_mm(mm_input_proj_w) with clamp
pub fn buildGraph(
    builder: *GraphBuilder,
) !*ggml.CGraph {
    const ctx = builder.ctx0;
    const gf = builder.gf;
    const w = builder.weights;
    const p = builder.hparams;

    const ps: i32 = @intCast(p.patch_size);
    // Use image_height if set (non-zero), otherwise fall back to image_size (square)
    const img_h: i32 = if (p.image_height > 0) @intCast(p.image_height) else @intCast(p.image_size);
    const img_w: i32 = @intCast(p.image_size);
    const npx: i32 = @divTrunc(img_w, ps);
    const npy: i32 = @divTrunc(img_h, ps);
    const np: i32 = npx * npy;
    const ne: i32 = @intCast(p.n_embd);
    const n_batch: i32 = 1;

    log.debug("buildGraph: image_size={d}x{d} patch_size={d} npx={d} npy={d} np={d} ne={d}", .{ img_w, img_h, p.patch_size, npx, npy, np, ne });

    // ========================================================================
    // Step 1: build_inp_raw() — create input tensor
    // C++: ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx(), img.ny(), channels, n_batch);
    //       ggml_set_name(inp_raw, "inp_raw");
    //       ggml_set_input(inp_raw);
    // ========================================================================
    const inp_raw = try ctx.newTensor4d(ggml.Type.f32, @as(i64, img_w), @as(i64, img_h), 3, n_batch);
    inp_raw.setName("inp_raw");
    ggml.setInput(inp_raw);

    // ========================================================================
    // Step 2: scale_bias — patches = 2 * (patches - 0.5)
    // C++: inp_raw = ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f);
    //       ggml_set_name(inp_raw, "inp_raw_scaled");
    //       ggml_set_output(inp_raw);
    // ========================================================================
    log.debug("Step 1: scale_bias", .{});
    var cur = inp_raw.scaleBias(ctx, 2.0, -1.0);
    cur.setName("inp_raw_scaled");
    ggml.setOutput(cur);

    log.debug("*** n_patches={} n_embd={} n_batch={} patch_size={}", .{ np, ne, 1, ps });

    // ========================================================================
    // Step 3: Conv2D patch embedding
    // C++: inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    //       ggml_set_name(inp, "inp_conv_2d");
    //       ggml_set_output(inp);
    // ========================================================================
    const pe_w = w.patch_embeddings_0 orelse return error.MissingPatchEmbeddings;
    cur = cur.conv2d(ctx, pe_w, ps, ps, 0, 0, 1, 1);

    // ========================================================================
    // Step 4: reshape + transpose
    // C++: inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
    //       ggml_set_name(inp, "inp_reshape_3d");
    //       ggml_set_output(inp);
    //       inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    //       ggml_set_name(inp, "inp_final");
    //       ggml_set_output(inp);
    //       // note: no patch bias
    // ========================================================================
    cur = cur.reshape3d(ctx, np, ne, 1);
    cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
    cur.setName("inp_final");
    ggml.setOutput(cur);

    // ========================================================================
    // Step 5: 2D Position embeddings (x/y lookup)
    // C++: ggml_tensor * pos_x = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    //       ggml_set_name(pos_x, "pos_x");
    //       ggml_set_input(pos_x);
    //       ggml_tensor * pos_y = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    //       ggml_set_name(pos_y, "pos_y");
    //       ggml_set_input(pos_y);
    // ========================================================================
    const pos_x = try posIndices(ctx, npx, npy, .x);
    const pos_y = try posIndices(ctx, npx, npy, .y);

    // C++: const int64_t pos_size = model.position_embeddings->ne[1];
    //       const size_t  nb1      = ggml_row_size(model.position_embeddings->type, n_embd);
    //       ggml_tensor * tbl_x = ggml_view_2d(ctx0, model.position_embeddings,
    //                                            n_embd, pos_size, nb1, 0);
    //       ggml_tensor * tbl_y = ggml_view_2d(ctx0, model.position_embeddings,
    //                                            n_embd, pos_size, nb1, pos_size * nb1);
    //       ggml_tensor * emb_x = ggml_get_rows(ctx0, tbl_x, pos_x);
    //       ggml_tensor * emb_y = ggml_get_rows(ctx0, tbl_y, pos_y);
    //       inp = ggml_add(ctx0, inp, emb_x);
    //       inp = ggml_add(ctx0, inp, emb_y);
    //       cb(inp, "pos_embd", -1);
    //       ggml_set_output(inp);
    // ========================================================================
    const pe = w.position_embeddings orelse return error.MissingPositionEmbeddings;
    const psz: i64 = pe.ne()[1];
    const nb: usize = @intCast(ggml.Type.rowSize(pe.dataType(), @intCast(ne)));
    const tx = pe.view2d(ctx, ne, psz, nb, 0);
    const ty = pe.view2d(ctx, ne, psz, nb, @as(usize, @intCast(psz)) * nb);
    log.debug("Step 3: position embeddings (psz={d})", .{psz});
    cur = cur.add(ctx, ggml.getRows(ctx, tx, pos_x));
    cur = cur.add(ctx, ggml.getRows(ctx, ty, pos_y));
    cur.setName("pos_embd");
    ggml.setOutput(cur);

    // Create Gemma4VData struct for passing to callbacks
    const gemma4v_data = Gemma4VData{
        .pos_x = pos_x,
        .pos_y = pos_y,
        .freq_base = p.rope_theta,
        .clamp_info_map = &w.clamp_info_map,
    };

    // ========================================================================
    // Step 6: ViT blocks with 2D RoPE
    // C++: kq_scale = 1.0f;
    //       ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_RMS, hparams.ffn_op,
    //                                      nullptr, // pos embd is already handled above
    //                                      add_pos);
    //       ggml_set_name(cur, "vit_output");
    //       ggml_set_output(cur);
    // ========================================================================
    cur = try vit_builder.buildVit(ctx, gf, cur, np, .rms_norm, p.ffn_op, null, // learned_pos_embd — already handled above
        w, p, addPos, .{
            .v_norm_eps = p.eps,
            .kq_scale = 1.0,
            .v_norm = true,
            .flash_attn_type = builder.flash_attn_type,
            .build_mm = buildMMWithClamp,
            .data = @ptrCast(@constCast(&gemma4v_data)),
        });
    cur.setName("vit_output");
    ggml.setOutput(cur);

    // ========================================================================
    // Step 7: Pool 2D (avg, kernel=n_merge) + scale(sqrt(n_embd))
    // C++: const int kernel_size = hparams.n_merge;
    //       GGML_ASSERT(kernel_size > 0);
    //       cur = ggml_cont_4d(ctx0, ggml_transpose(ctx0, cur), n_patches_x, n_patches_y, n_embd, n_batch);
    //       cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
    //       const int out_x = n_patches_x / kernel_size;
    //       const int out_y = n_patches_y / kernel_size;
    //       cur = ggml_reshape_3d(ctx0, cur, out_x * out_y, n_embd, n_batch);
    //       cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    //       cur = ggml_scale(ctx0, cur, sqrtf((float)n_embd));
    //       cb(cur, "pooled", -1);
    //       ggml_set_output(cur);
    // ========================================================================
    log.debug("Step 5: pool2d (n_merge={d})", .{p.n_merge});
    const ks: i32 = @intCast(p.n_merge);
    if (ks > 0) {
        cur = ggml.cont4d(ctx, ggml.transpose(ctx, cur), npx, npy, ne, 1);

        cur.setName("pooled_cont_4d");
        ggml.setOutput(cur);

        cur = cur.pool2d(ctx, @intCast(@intFromEnum(ggml.PoolOp.avg)), ks, ks, ks, ks, 0.0, 0.0);

        cur.setName("pooled_pool_2d");
        ggml.setOutput(cur);

        const ox: i64 = @divTrunc(npx, ks);
        const oy: i64 = @divTrunc(npy, ks);
        cur = cur.reshape3d(ctx, ox * oy, ne, 1);

        cur.setName("pooled_reshape_3d");
        ggml.setOutput(cur);
        cur = ggml.cont(ctx, ggml.transpose(ctx, cur));

        cur.setName("pooled_cont");
        ggml.setOutput(cur);

        cur = cur.scale(ctx, @sqrt(@as(f32, @floatFromInt(ne))));
        cur.setName("pooled");
        ggml.setOutput(cur);
    }

    // ========================================================================
    // Step 8: Standardization
    // C++: if (model.std_bias && model.std_scale) {
    //         cur = ggml_sub(ctx0, cur, model.std_bias);
    //         cur = ggml_mul(ctx0, cur, model.std_scale);
    //         cb(cur, "std_scaled", -1);
    //         ggml_set_output(cur);
    //     }
    // ========================================================================
    log.debug("Step 6: standardization (std_bias={any} std_scale={any})", .{ w.std_bias != null, w.std_scale != null });
    if (w.std_bias != null and w.std_scale != null) {
        cur = cur.sub(ctx, w.std_bias.?);
        cur = cur.mul(ctx, w.std_scale.?);
        cur.setName("std_scaled");
    }

    // ========================================================================
    // Step 9: Multimodal embedder
    // C++: // Gemma4MultimodalEmbedder
    //       cur = ggml_rms_norm(ctx0, cur, hparams.eps);
    //       cur = build_mm(model.mm_input_proj_w, cur);
    //       cb(cur, "mm_output", -1);
    //       ggml_set_output(cur);
    // ========================================================================
    log.debug("Step 7: multimodal embedder", .{});
    cur = cur.rmsNorm(ctx, p.eps);
    // NOTE: gemma4v does NOT use mm_soft_emb_norm_w
    if (w.mm_input_proj_w) |proj| {
        cur = buildMMWithClampDirect(ctx, proj, cur, &w.clamp_info_map);
    }
    cur.setName("mm_output");
    ggml.setOutput(cur);

    // ========================================================================
    // C++: ggml_build_forward_expand(gf, cur);
    //       return gf;
    // ========================================================================
    log.debug("buildGraph complete", .{});
    gf.buildForwardExpand(cur);
    return gf;
}

/// Build Gemma4V full compute graph (standalone parameter version).
/// Creates GraphBuilder from standalone parameters, then delegates to buildGraph(builder).
/// This function serves as the Backend.buildGraph entry point.
pub fn buildGraphFromWeights(
    io: std.Io,
    ctx: *ggml.Context,
    gf: *ggml.CGraph,
    w: *const VisionEncoderWeights,
    p: *const VisionHParams,
    image_tensor: *ggml.Tensor,
) anyerror!*ggml.CGraph {
    const img_buf = try image_tensor.dataGet(f32, std.heap.page_allocator);
    // Note: img_buf is intentionally leaked (leak-to-exit) since ImageF32
    // is used during graph construction and the data must remain valid.
    const img_h: u32 = if (p.image_height > 0) p.image_height else p.image_size;
    const img_w: u32 = p.image_size;
    const img = ImageF32{
        .buf = img_buf,
        .nx = img_w,
        .ny = img_h,
    };

    try graph.debug.saveData(io, "debug_vision", "zllama_vision_00_images.json", "images", img_buf);

    var hparams = p.*;
    var builder = GraphBuilder{
        .weights = w,
        .hparams = &hparams,
        .proj_type = .gemma4v,
        .img = &img,
        .ctx0 = ctx,
        .gf = gf,
    };

    // 1. Build graph (creates inp_raw tensor, sets name, marks as input)
    _ = try buildGraph(&builder);

    // 2. Fill inp_raw with pixel data (separate from graph construction, matching llama.cpp pattern)
    try fillInpRawFromImage(ctx, gf, &img, img_w, img_h);

    return gf;
}
