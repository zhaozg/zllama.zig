//! Gemma4V visual encoder graph builder.
//!
//! Ref: deps/llama.cpp/tools/mtmd/models/gemma4v.cpp
//!
//! Pipeline:
//!   1. scale_bias(inp_raw, 2.0, -1.0)
//!   2. Conv2D patch embedding (no patch_bias)
//!   3. 2D position embeddings (pos_x/pos_y lookup tables)
//!   4. ViT blocks (RMS norm) with 2D RoPE (NEOX ordering)
//!   5. Pool 2D (avg, kernel=n_merge) + scale(sqrt(n_embd))
//!   6. Standardization: (hidden - std_bias) * std_scale
//!   7. Multimodal embedder: rms_norm -> build_mm(mm_input_proj_w)
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

fn findTensor(ctx: *ggml.Context, gf: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return weight_loader.findOrCreateTensor(ctx, gf, name);
}

fn loadLayer(ctx: *ggml.Context, gf: *const gguf.GGUFFile, prefix: []const u8, alloc: std.mem.Allocator) !ViTLayerWeights {
    var l = ViTLayerWeights{};
    const fields = [_]struct { suffix: []const u8, ptr: *?*ggml.Tensor }{
        .{ .suffix = "ln1.weight", .ptr = &l.ln_1_w },
        .{ .suffix = "attn_q.weight", .ptr = &l.q_w },
        .{ .suffix = "attn_k.weight", .ptr = &l.k_w },
        .{ .suffix = "attn_v.weight", .ptr = &l.v_w },
        .{ .suffix = "attn_out.weight", .ptr = &l.o_w },
        .{ .suffix = "ln2.weight", .ptr = &l.ln_2_w },
        .{ .suffix = "ffn_up.weight", .ptr = &l.ff_up_w },
        .{ .suffix = "ffn_gate.weight", .ptr = &l.ff_gate_w },
        .{ .suffix = "ffn_down.weight", .ptr = &l.ff_down_w },
        .{ .suffix = "attn_post_norm.weight", .ptr = &l.attn_post_norm_w },
        .{ .suffix = "ffn_post_norm.weight", .ptr = &l.ff_post_norm_w },
        .{ .suffix = "attn_k_norm.weight", .ptr = &l.k_norm },
        .{ .suffix = "attn_q_norm.weight", .ptr = &l.q_norm },
    };
    for (fields) |f| {
        const full = try std.fmt.allocPrint(alloc, "{s}.{s}", .{ prefix, f.suffix });
        defer alloc.free(full);
        f.ptr.* = findTensor(ctx, gf, full) catch null;
    }
    return l;
}

pub fn loadWeights(io: std.Io, alloc: std.mem.Allocator, gf: *const gguf.GGUFFile, ctx: *ggml.Context, w: *VisionEncoderWeights) anyerror!void {
    _ = io;
    w.patch_embeddings_0 = findTensor(ctx, gf, "v.patch_embd.weight") catch null;
    w.position_embeddings = findTensor(ctx, gf, "v.position_embd.weight") catch null;
    w.std_bias = findTensor(ctx, gf, "v.std_bias") catch null;
    w.std_scale = findTensor(ctx, gf, "v.std_scale") catch null;
    w.mm_input_proj_w = findTensor(ctx, gf, "mm.input_projection.weight") catch null;
    // NOTE: gemma4v does NOT use mm_soft_emb_norm_w (that's gemma3 only)
    // w.mm_soft_emb_norm_w is intentionally NOT loaded

    var n: u32 = 0;
    for (0..64) |il| {
        var buf: [32]u8 = undefined;
        if (gf.findTensor(try std.fmt.bufPrint(&buf, "v.blk.{d}.attn_q.weight", .{il})) == null) break;
        n = @intCast(il + 1);
    }
    w.layers = try alloc.alloc(ViTLayerWeights, n);
    for (0..n) |il| {
        const pfx = try std.fmt.allocPrint(alloc, "v.blk.{d}", .{il});
        defer alloc.free(pfx);
        w.layers[il] = try loadLayer(ctx, gf, pfx, alloc);
    }
    log.info("Gemma4V: {d} layers", .{n});
}
pub fn loadClampInfo(io: std.Io, allocator: std.mem.Allocator, gf: *const gguf.GGUFFile, w: *VisionEncoderWeights) anyerror!void {
    _ = io;
    var weight_names = std.ArrayList([]const u8).initCapacity(allocator, 0) catch |err| return err;
    defer weight_names.deinit(allocator);

    if (w.mm_input_proj_w) |t| try weight_names.append(allocator, t.getName());

    w.clamp_info_map = try graph.clamp.loadClampInfoFromWeightNames(allocator, gf, weight_names.items);
    log.info("Gemma4V clamp info loaded: {d} entries", .{w.clamp_info_map.count()});
}
// Output Estimate
// ============================================================================

pub fn estimateOutputTokens(io: std.Io, iw: u32, ih: u32, ps: u32, nm: u32) u32 {
    _ = io;
    const m: u32 = if (nm == 0) 1 else nm;
    return (iw / ps / m) * (ih / ps / m);
}

// ============================================================================
// 2D RoPE state (module-level, set before buildVit call)
// ============================================================================

var rope_pos_x: ?*ggml.Tensor = null;
var rope_pos_y: ?*ggml.Tensor = null;
var rope_freq_base: f32 = 100.0;

/// 2D RoPE callback for ViT blocks.
/// Ref: deps/llama.cpp/tools/mtmd/models/gemma4v.cpp add_pos lambda
fn addPos(ctx: *ggml.Context, cur: *ggml.Tensor, _: *const ViTLayerWeights) *ggml.Tensor {
    const d_head = cur.ne()[0];
    const n_head = cur.ne()[1];
    const n_patches = cur.ne()[2];
    const n_batch = cur.ne()[3];
    const d_head_half = @divExact(d_head, 2);

    // First half: use pos_x
    const first = cur.view4d(
        ctx,
        d_head_half,
        n_head,
        n_patches,
        n_batch,
        cur.nb()[1],
        cur.nb()[2],
        cur.nb()[3],
        0,
    );
    const rope_first = first.ropeExt(
        ctx,
        rope_pos_x orelse unreachable,
        null,
        @intCast(d_head_half),
        2, // GGML_ROPE_TYPE_NEOX
        0,
        rope_freq_base,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );

    // Second half: use pos_y
    const offset: usize = @intCast(d_head_half * @sizeOf(f32));
    const second = cur.view4d(
        ctx,
        d_head_half,
        n_head,
        n_patches,
        n_batch,
        cur.nb()[1],
        cur.nb()[2],
        cur.nb()[3],
        offset,
    );
    const rope_second = second.ropeExt(
        ctx,
        rope_pos_y orelse unreachable,
        null,
        @intCast(d_head_half),
        2, // GGML_ROPE_TYPE_NEOX
        0,
        rope_freq_base,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );

    return rope_first.concat(ctx, rope_second, 0);
}

// ============================================================================
// Graph
// ============================================================================

const Dim = enum { x, y };

fn posIndices(ctx: *ggml.Context, nx: i32, ny: i32, d: Dim) !*ggml.Tensor {
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

fn buildMM(ctx: *ggml.Context, wt: *ggml.Tensor, x: *ggml.Tensor, cm: *const std.StringHashMap(ClampInfo)) *ggml.Tensor {
    if (cm.get(wt.getName())) |ci| {
        return wt.mulMat(ctx, x.clamp(ctx, ci.inp_min, ci.inp_max)).clamp(ctx, ci.out_min, ci.out_max);
    }
    return wt.mulMat(ctx, x);
}

/// 构建 Gemma4V 完整计算图（GraphBuilder 版本）
/// 接收 GraphBuilder，从中提取 ctx, gf, weights, hparams 等公共参数。
/// 从 builder.img 获取图像数据并创建输入 tensor。
pub fn buildGraph(
    builder: *GraphBuilder,
) !*ggml.CGraph {
    const ctx = builder.ctx0;
    const gf = builder.gf;
    const w = builder.weights;
    const p = builder.hparams;
    _ = builder.img; // img data is filled after buildGraph returns (see buildGraphFromWeights)

    const ps: i32 = @intCast(p.patch_size);
    const npx: i32 = @divTrunc(@as(i32, @intCast(p.image_size)), ps);
    const npy: i32 = @divTrunc(@as(i32, @intCast(p.image_size)), ps);
    const np: i64 = @as(i64, npx) * @as(i64, npy);
    const ne: i64 = @intCast(p.n_embd);

    log.debug("buildGraph: image_size={d} patch_size={d} npx={d} npy={d} np={d} ne={d}", .{ p.image_size, p.patch_size, npx, npy, np, ne });
    // 1. 创建输入张量 (match C++ build_inp_raw: 4D [W, H, C, 1], set_input)
    //    Graph construction only — data is filled separately after buildGraph returns.
    //    Ref: llama.cpp clip-graph.h build_inp_raw() creates empty tensor, clip.cpp fills data later.
    const n_batch: i64 = 1;
    const inp_raw = try ctx.newTensor4d(ggml.Type.f32, @as(i64, @intCast(p.image_size)), @as(i64, @intCast(p.image_size)), 3, n_batch);
    inp_raw.setName("inp_raw");
    ggml.setInput(inp_raw);

    // 2. Scale+bias: patches * 2 - 1
    //    Ref: gemma4v.cpp: inp_raw = ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f);
    log.debug("Step 1: scale_bias", .{});
    var cur = inp_raw.scaleBias(ctx, 2.0, -1.0);
    cur.setName("inp_raw_scaled");
    cur = cur.reshape3d(ctx, np, ne, 1);
    cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
    cur.setName("inp");

    // 4. 2D Position embeddings (x/y lookup)
    const pe = w.position_embeddings orelse return error.MissingPositionEmbeddings;
    const psz: i64 = pe.ne()[1];
    const nb: usize = @intCast(ggml.Type.rowSize(pe.dataType(), @intCast(ne)));
    const tx = pe.view2d(ctx, ne, psz, nb, 0);
    const ty = pe.view2d(ctx, ne, psz, nb, @as(usize, @intCast(psz)) * nb);
    log.debug("Step 3: position embeddings (psz={d})", .{psz});
    cur = cur.add(ctx, ggml.getRows(ctx, tx, try posIndices(ctx, npx, npy, .x)));
    cur = cur.add(ctx, ggml.getRows(ctx, ty, try posIndices(ctx, npx, npy, .y)));
    cur.setName("pos_embd");

    // Create pos_x/pos_y tensors for 2D RoPE (used in addPos callback)
    rope_pos_x = try posIndices(ctx, npx, npy, .x);
    rope_pos_y = try posIndices(ctx, npx, npy, .y);
    rope_freq_base = p.rope_theta;

    // 5. ViT blocks with 2D RoPE
    log.debug("Step 4: ViT blocks (n_layer={d})", .{p.n_layer});
    cur = try vit_builder.buildVit(ctx, cur, np, .rms_norm, p.ffn_op, null, w, p, addPos, .{
        .v_norm = true,
        .v_norm_eps = p.eps,
        .kq_scale = 1.0,
    });
    cur.setName("vit_output");

    // 6. Pool 2D (avg, kernel=n_merge) + scale
    log.debug("Step 5: pool2d (n_merge={d})", .{p.n_merge});
    const ks: i32 = @intCast(p.n_merge);
    if (ks > 0) {
        cur = ggml.cont4d(ctx, ggml.transpose(ctx, cur), npx, npy, ne, 1);
        cur = cur.pool2d(ctx, @intCast(@intFromEnum(ggml.PoolOp.avg)), ks, ks, ks, ks, 0.0, 0.0);
        const ox: i64 = @divTrunc(npx, ks);
        const oy: i64 = @divTrunc(npy, ks);
        cur = cur.reshape3d(ctx, ox * oy, ne, 1);
        cur = ggml.cont(ctx, ggml.transpose(ctx, cur));
        cur = cur.scale(ctx, @sqrt(@as(f32, @floatFromInt(ne))));
        cur.setName("pooled");
    }

    // 7. Standardization
    log.debug("Step 6: standardization (std_bias={any} std_scale={any})", .{ w.std_bias != null, w.std_scale != null });
    if (w.std_bias != null and w.std_scale != null) {
        cur = cur.sub(ctx, w.std_bias.?);
        // std_scale is [n_embd] 1D tensor, use directly (ggml_mul broadcasts)
        cur = cur.mul(ctx, w.std_scale.?);
        cur.setName("std_scaled");
    }

    // 8. Multimodal embedder
    log.debug("Step 7: multimodal embedder", .{});
    cur = cur.rmsNorm(ctx, p.eps);
    // NOTE: gemma4v does NOT use mm_soft_emb_norm_w
    // Only apply mm_input_proj_w (with optional clamping)
    if (w.mm_input_proj_w) |proj| {
        cur = buildMM(ctx, proj, cur, &w.clamp_info_map);
    }
    cur.setName("mm_output");

    log.debug("buildGraph complete", .{});
    gf.buildForwardExpand(cur);
    return gf;
}

/// 构建 Gemma4V 完整计算图（独立参数版本）
/// 从独立参数创建 GraphBuilder，然后委托给 buildGraph(builder)。
/// 此函数作为 Backend.buildGraph 的入口，兼容现有调用方式。
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
    const img = ImageF32{
        .buf = img_buf,
        .nx = p.image_size,
        .ny = p.image_size,
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
    _ = buildGraph(&builder) catch unreachable;

    // 2. Fill inp_raw with pixel data (separate from graph construction, matching llama.cpp pattern)
    //    Ref: llama.cpp clip.cpp lines 3645-3665: set_input_f32("inp_raw", inp_raw)
    //
    //    The image tensor (image_tensor) is created by resizeAndNormalize as a 3D [W, H, C] tensor
    //    with HWC layout (matching llama.cpp convention):
    //      ne[0]=W (innermost), ne[1]=H, ne[2]=C
    //    Memory: [R0,G0,B0, R1,G1,B1, ..., R_n,G_n,B_n]
    //
    //    The inp_raw tensor is 4D [W, H, C, B] with CHW layout (ggml column-major):
    //      ne[0]=W, ne[1]=H, ne[2]=C, ne[3]=B
    //    Memory: [R0,R1,..., G0,G1,..., B0,B1,...]
    //
    //    So we need HWC->CHW conversion when filling inp_raw.
    //    Ref: clip.cpp lines 3645-3665 does HWC->CHW conversion.

    // 2. Fill inp_raw with pixel data (separate from graph construction, matching llama.cpp pattern)
    //    Ref: llama.cpp clip.cpp lines 3645-3665: set_input_f32("inp_raw", inp_raw)
    //
    //    The image tensor (image_tensor) is created by resizeAndNormalize as a 3D [W, H, C] tensor
    //    with HWC layout (matching llama.cpp convention):
    //      ne[0]=W (innermost), ne[1]=H, ne[2]=C
    //    Memory: [R0,G0,B0, R1,G1,B1, ..., R_n,G_n,B_n]
    //
    //    The inp_raw tensor is 4D [W, H, C, B] with CHW layout (ggml column-major):
    //      ne[0]=W, ne[1]=H, ne[2]=C, ne[3]=B
    //    Memory: [R0,R1,..., G0,G1,..., B0,B1,...]
    //
    //    So we need HWC->CHW conversion when filling inp_raw.
    //    Ref: clip.cpp lines 3645-3665 does HWC->CHW conversion.
    {
        const inp_raw = gf.getTensor("inp_raw") orelse return error.TensorNotFound;
        const W: usize = @intCast(p.image_size);
        const H: usize = @intCast(p.image_size);
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

        try inp_raw.dataSet(f32, dst);
    }

    return gf;
}
