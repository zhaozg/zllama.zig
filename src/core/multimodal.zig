//! Multimodal generation — vision + audio inference pipelines.
//!
//! 引擎层的多模态编排器，通过 mtmd 模块的公共接口集成。
//!
//! ## 设计说明
//!
//! 本模块位于 L6 层（引擎层），负责多模态推理的编排逻辑：
//! - 图像/音频加载与预处理
//! - 编码器图构建与计算
//! - 三阶段 prefill（prefix + media + suffix）
//! - decode 循环
//!
//! 本模块**通过 mtmd 模块的公共 API** 与多模态编码器交互：
//! - `mtmd.MultiModalManager` — 编码器生命周期管理
//! - `mtmd.MtmdContext` — 媒体标记解析、chunk 化输入处理
//! - `mtmd.preprocess` — 图像预处理
//! - `mtmd.audio_mod` — 音频预处理
//! - `mtmd.tokenize` — 按媒体标记分割文本
//! - `mtmd.evalChunks` — 评估所有 chunks
//!
//! 本模块**不直接访问** mtmd 内部子模块（如 mtmd/helper.zig、mtmd/tokenize.zig 等）。
//!
//! Reference: llama.cpp llama_mmd.h / gemma4.cpp / qwen3vl.cpp

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const kv_cache = @import("kv_cache");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const chat_template = @import("chat_template");
const mtmd = @import("mtmd");
const debug = @import("debug");
const preprocess = @import("preprocess");
const engine_common = @import("engine_common");
const prefill_mod = @import("prefill");
const decode_mod = @import("decode");
const verbose_mod = @import("verbose");
const graph_context = @import("graph_context");

const logger = std.log.scoped(.core_multimodal);

/// Engine reference needed by the multimodal generation paths.
pub const EngineContext = struct {
    allocator: std.mem.Allocator,
    ctx_graph: *ggml.Context,
    arch: model_if.Architecture,
    model: model_if.ModelInstance,
    params: model_if.ModelParams,
    tok: *tokenizer.Tokenizer,
    kv_cache_mgr: *kv_cache.KVCache,
    n_threads: i32,
    verbose_prompt: bool,
    benchmark: bool,
    inc_ctx: *graph_context.IncContext,
    mm_manager: ?*mtmd.MultiModalManager,
    mtmd_context: ?*mtmd.MtmdContext,
    capabilities: model_if.ModelCapabilities,
    chat_template_source: ?chat_template.TemplateSource,
    system_prompt: []const u8,
    no_chat_template: bool,
    no_jinja: bool,
    image_max_pixels: u32 = 0,
    gallocr: *ggml.Gallocr,
};

/// 计算动态分辨率的目标尺寸，并在 VisionEncoder params 中设置 user_max_pixels。
/// 返回的实际尺寸仅用于日志记录（实际缩放由 VisionEncoder.encode 内部处理）。
fn computeAndSetTargetSize(
    enc: *mtmd.vision_mod.VisionEncoder,
    src_width: u32,
    src_height: u32,
    user_max_pixels: u32,
) preprocess.Size2D {
    if (user_max_pixels > 0) {
        enc.setUserMaxPixels(user_max_pixels);
    }
    const p = &enc.params;
    const effective_max_pixels = if (p.user_max_pixels > 0) p.user_max_pixels else p.image_max_pixels;
    if (p.image_min_pixels > 0 and effective_max_pixels > 0) {
        const align_size = p.patch_size * p.n_merge;
        if (align_size > 0) {
            const result = preprocess.calcSizePreservedRatio(
                src_width,
                src_height,
                align_size,
                p.image_min_pixels,
                effective_max_pixels,
            );
            logger.info("Dynamic resize: {d}x{d} -> {d}x{d} (align={d}, min_px={d}, max_px={d})", .{
                src_width,  src_height,         result.width,         result.height,
                align_size, p.image_min_pixels, effective_max_pixels,
            });
            return result;
        }
    }
    logger.info("Fixed resize: {d}x{d} -> {d}x{d}", .{ src_width, src_height, p.image_size, p.image_size });
    return .{ .width = p.image_size, .height = p.image_size };
}

/// 加载图像原始数据（不缩放），返回原始尺寸和 RGB 数据
/// 通过 preprocess.loadImageRaw 复用 stb_image 绑定层，避免直接调用 stb_image API。
fn loadImageRaw(allocator: std.mem.Allocator, io: std.Io, file_path: []const u8) !preprocess.ProcessedImage {
    const result = preprocess.loadImageRaw(allocator, io, file_path) catch |err| {
        logger.err("Failed to load image {s}: {}", .{ file_path, err });
        return err;
    };
    logger.info("Loaded raw image: {d}x{d}", .{ result.width, result.height });
    return result;
}

/// Generate text from an image + prompt.
pub fn generateWithImage(ectx: *EngineContext, io: std.Io, prompt: []const u8, image_path: [:0]const u8, max_tokens: u32) !void {
    const mm_mgr = ectx.mm_manager orelse return error.MMProjNotLoaded;
    logger.debug("generateWithImage: START image_path={s}", .{image_path});
    if (!ectx.capabilities.has_vision) return error.VisionNotSupported;
    if (ectx.arch != .gemma4 and ectx.arch != .qwen3vl) {
        logger.err("Vision inference not yet implemented for architecture '{s}'.", .{@tagName(ectx.arch)});
        logger.err("  Currently only Gemma4 and Qwen3VL support vision.", .{});
        return error.VisionNotSupportedForArchitecture;
    }

    if (ectx.mtmd_context) |ctx| {
        logger.info("Image markers from MtmdContext: '{s}' / '{s}'", .{ ctx.img_beg, ctx.img_end });
    }

    // 步骤 1: 加载原始图像（通过 preprocess.loadImageRaw → stb_image 绑定层）
    var raw_img = try loadImageRaw(ectx.allocator, io, image_path);
    defer raw_img.deinit();

    // 步骤 2: 计算目标尺寸（动态分辨率或固定正方形），并传递 user_max_pixels 给 encoder
    logger.debug("generateWithImage: Step 2 - computeAndSetTargetSize", .{});
    var enc = mm_mgr.vision_encoder.?;
    const target_size = computeAndSetTargetSize(&enc, raw_img.width, raw_img.height, ectx.image_max_pixels);
    logger.debug("generateWithImage: target size = {d}x{d}", .{ target_size.width, target_size.height });

    // 步骤 3: 直接传递原始图像给 encodeMedia，由 encode 中的 resizeAndNormalize 处理缩放
    logger.debug("generateWithImage: Step 3 - passing raw image to encodeMedia ({d}x{d})", .{ raw_img.width, raw_img.height });

    if (raw_img.width == 0 or raw_img.height == 0) return error.EmptyImage;

    // 使用 no_alloc = true 模式创建视觉编码器上下文。
    // 使用 no_alloc = true 模式创建视觉编码器上下文。
    // 512 MB 用于元数据（no_alloc 模式下张量数据由 Gallocr 管理）。
    // P0: 从 256MB 增加到 512MB 以解决 Gemma4 E2B 大视觉图的 OutOfMemory 问题。
    var vision_ctx = try ggml.Context.initNoAlloc(512 * 1024 * 1024);
    defer vision_ctx.deinit();
    const vision_graph = try ggml.CGraph.initReserved(vision_ctx, 32768);
    logger.debug("generateWithImage: Step 4 - creating vision_ctx and calling encodeMedia...", .{});
    const vision_embeddings = try mm_mgr.encodeMedia(io, vision_ctx, vision_graph, .{
        .media_type = .image,
        .image_data = raw_img.data,
        .image_width = raw_img.width,
        .image_height = raw_img.height,
    }, ectx.n_threads);
    const buft = ggml.backendCpuBufferType();
    var vis_gallocr = try ggml.Gallocr.init(buft);

    logger.debug("generateWithImage: encodeMedia returned, vision_embeddings ne={any}", .{vision_embeddings.ne()});
    defer vis_gallocr.free();
    _ = try engine_common.computeGraph(vision_graph, ectx.n_threads, vis_gallocr);

    vision_ctx.setNoAlloc(true);

    logger.debug("generateWithImage: Step 5 - computing vision graph...", .{});
    const n_vision_tokens: i32 = @intCast(vision_embeddings.ne()[1]);
    const n_embd_val: usize = @intCast(vision_embeddings.ne()[0]);
    logger.debug("Vision encoder output: shape=[{d}, {d}]", .{ n_embd_val, n_vision_tokens });

    if (ectx.arch == .qwen3vl) {
        logger.debug("  Qwen3VL: allowing encoder dim {d}", .{n_embd_val});
    } else if (n_embd_val != @as(usize, @intCast(ectx.params.n_embd))) {
        logger.err("EMBEDDING DIMENSION MISMATCH: encoder={d} vs model={d}", .{ n_embd_val, ectx.params.n_embd });
        return error.EmbeddingDimensionMismatch;
    }

    const image_token_id: u32 = blk: {
        if (ectx.tok.textToToken("<|image_pad|>")) |id| break :blk @as(u32, @intCast(id));
        if (ectx.tok.textToToken("<|image|>")) |id| break :blk @as(u32, @intCast(id));
        if (ectx.tok.textToToken("<image>")) |id| break :blk @as(u32, @intCast(id));
        if (ectx.tok.textToToken("<|vision_start|>")) |id| break :blk @as(u32, @intCast(id));
        return error.NoImagePlaceholderToken;
    };

    const formatted_prompt = if (ectx.no_chat_template) blk: {
        const with_ph = try chat_template.ensurePlaceholderInContent(prompt, .image, ectx.allocator, null);
        break :blk with_ph;
    } else blk: {
        const model_name: ?[]const u8 = if (ectx.params.model_name.len > 0) ectx.params.model_name else null;
        const system = if (ectx.system_prompt.len > 0) ectx.system_prompt else null;
        break :blk try chat_template.applyWithMedia(
            ectx.allocator,
            ectx.arch,
            model_name,
            ectx.chat_template_source,
            ectx.no_chat_template,
            ectx.no_jinja,
            prompt,
            chat_template.Media.init(.image),
            system,
        );
    };
    defer ectx.allocator.free(formatted_prompt);

    logger.info("formatted_prompt: {s}", .{formatted_prompt});

    var expanded: chat_template.TokenizedSegments = undefined;
    if (ectx.mtmd_context) |mtmd_ctx| {
        // Use mtmd.tokenize when MtmdContext is available.
        const media_marker = mtmd_ctx.media_marker;
        const marker_replaced = try replacePlaceholdersWithMarker(ectx.allocator, formatted_prompt, media_marker);
        defer ectx.allocator.free(marker_replaced);
        const bitmap = mtmd.Bitmap.initPlaceholderImage(@intCast(raw_img.width), @intCast(raw_img.height));
        var chunks = try mtmd.tokenize(mtmd_ctx, io, ectx.allocator, .{ .text = marker_replaced, .add_special = false, .parse_special = true }, &.{bitmap});
        defer chunks.deinit();
        expanded = try inputChunksToTokenizedSegments(ectx.allocator, &chunks, image_token_id, .image);
    } else {
        expanded = try tokenizeWithMediaPlaceholders(ectx, formatted_prompt, image_token_id, @intCast(n_vision_tokens), .image);
    }
    defer expanded.deinit();
    try multimodalPrefillUnified(ectx, io, &expanded, image_token_id, @intCast(n_vision_tokens), vision_embeddings, max_tokens, formatted_prompt, @tagName(ectx.arch));
    logger.debug("generateWithImage: DONE", .{});
}

/// Generate text from an audio file + prompt.
pub fn generateWithAudio(ectx: *EngineContext, io: std.Io, prompt: []const u8, audio_path: [:0]const u8, max_tokens: u32) !void {
    const mm_mgr = ectx.mm_manager orelse return error.MMProjNotLoaded;
    if (!ectx.capabilities.has_audio) return error.AudioNotSupported;
    if (ectx.arch != .gemma4) {
        logger.err("Audio inference not yet implemented for architecture '{s}'.", .{@tagName(ectx.arch)});
        logger.err("  Currently only Gemma4 supports audio.", .{});
        return error.AudioNotSupportedForArchitecture;
    }

    if (ectx.mtmd_context) |ctx| {
        logger.info("Audio markers from MtmdContext: '{s}' / '{s}'", .{ ctx.aud_beg, ctx.aud_end });
    }

    // 通过 mtmd.audio_mod 公共接口加载 WAV 文件
    const wav_result = try mtmd.audio_mod.loadWav(ectx.allocator, io, audio_path);
    defer ectx.allocator.free(wav_result.samples);
    if (wav_result.samples.len == 0) return error.EmptyAudio;

    const is_gemma4ua = mm_mgr.audio_encoder != null and
        std.mem.eql(u8, ectx.capabilities.audio_encoder_type, "gemma4ua");

    var mel: mtmd.audio_mod.types.ProcessedAudio = undefined;
    var mel_owned = false;

    if (is_gemma4ua) {
        // Gemma4UA 预处理：直接把原始 PCM 样本分帧，不做 FFT/Mel 频谱
        const frame_size: u32 = if (mm_mgr.audio_encoder) |enc|
            @intCast(enc.weights.mm_input_proj_w.?.ne()[0])
        else
            return error.AudioEncoderNotAvailable;

        logger.info("Gemma4UA preprocessing: frame_size={d}, samples={d}", .{ frame_size, wav_result.samples.len });
        mel = try mtmd.audio_mod.processRawWaveform(ectx.allocator, wav_result.samples, frame_size);
        mel_owned = true;
    } else {
        const preprocess_params = mtmd.audio_mod.AudioPreprocessParams.fromAudioEncoder(if (mm_mgr.audio_encoder) |enc| enc.params.n_mel_bins else mtmd.audio_mod.AUDIO_N_MEL_BINS);
        mel = try mtmd.audio_mod.computeMelSpectrogram(io, ectx.allocator, wav_result.samples, wav_result.info.sample_rate, preprocess_params);
        mel_owned = true;
    }
    defer if (mel_owned) mel.deinit();

    // 使用独立的 ggml context 构建音频编码器图
    var audio_ctx = try ggml.Context.initNoAlloc(256 * 1024 * 1024);
    defer audio_ctx.deinit();

    logger.debug("Audio encoder: created independent ggml context", .{});

    const audio_graph = try ggml.CGraph.initReserved(audio_ctx, 32768);

    const mel_tensor = try mtmd.audio_mod.melToTensor(audio_ctx, mel.data, mel.n_frames, mel.n_mel_bins);
    logger.debug("Audio encoder: mel tensor created, shape=[{d},{d}]", .{ mel_tensor.ne()[0], mel_tensor.ne()[1] });

    const audio_embeddings = try mm_mgr.encodeMedia(io, audio_ctx, audio_graph, .{
        .media_type = .audio,
        .mel_tensor = mel_tensor,
        .mel_data = mel.data,
        .mel_bins = mel.n_mel_bins,
        .mel_frames = mel.n_frames,
        .audio_length_sec = @as(f32, @floatFromInt(wav_result.info.num_samples)) / @as(f32, @floatFromInt(wav_result.info.sample_rate)),
    }, ectx.n_threads);
    logger.debug("Audio encoder: graph built, embeddings tensor found", .{});

    // Compute the audio graph with a dedicated Gallocr
    logger.debug("Audio encoder: computing graph...", .{});
    const audio_buft = ggml.backendCpuBufferType();
    var audio_gallocr = try ggml.Gallocr.init(audio_buft);
    defer audio_gallocr.free();
    _ = try engine_common.computeGraph(audio_graph, ectx.n_threads, audio_gallocr);

    audio_ctx.setNoAlloc(true);

    const n_audio_tokens: i32 = @intCast(audio_embeddings.ne()[1]);
    const n_embd_val: usize = @intCast(audio_embeddings.ne()[0]);

    logger.debug("Audio encoder output: shape=[{d}, {d}]", .{ n_embd_val, n_audio_tokens });
    if (n_embd_val != @as(usize, @intCast(ectx.params.n_embd))) {
        logger.err("EMBEDDING DIMENSION MISMATCH: encoder={d} vs model={d}", .{ n_embd_val, ectx.params.n_embd });
        return error.EmbeddingDimensionMismatch;
    }

    const audio_token_id: u32 = blk: {
        if (ectx.tok.textToToken("<|audio|>")) |id| break :blk @as(u32, @intCast(id));
        if (ectx.tok.textToToken("<audio>")) |id| break :blk @as(u32, @intCast(id));
        return error.NoAudioPlaceholderToken;
    };

    const formatted_prompt = if (ectx.no_chat_template) blk: {
        const with_ph = try chat_template.ensurePlaceholderInContent(prompt, .audio, ectx.allocator, null);
        break :blk with_ph;
    } else blk: {
        const model_name: ?[]const u8 = if (ectx.params.model_name.len > 0) ectx.params.model_name else null;
        const system = if (ectx.system_prompt.len > 0) ectx.system_prompt else null;
        break :blk try chat_template.applyWithMedia(
            ectx.allocator,
            ectx.arch,
            model_name,
            ectx.chat_template_source,
            ectx.no_chat_template,
            ectx.no_jinja,
            prompt,
            chat_template.Media.init(.audio),
            system,
        );
    };
    defer ectx.allocator.free(formatted_prompt);

    logger.info("formatted_prompt: {s}", .{formatted_prompt});

    var expanded: chat_template.TokenizedSegments = undefined;
    if (ectx.mtmd_context) |mtmd_ctx| {
        // Use mtmd.tokenize when MtmdContext is available.
        const media_marker = mtmd_ctx.media_marker;
        const marker_replaced = try replacePlaceholdersWithMarker(ectx.allocator, formatted_prompt, media_marker);
        defer ectx.allocator.free(marker_replaced);
        const bitmap = mtmd.Bitmap.initPlaceholderAudio(@intCast(wav_result.info.num_samples));
        var chunks = try mtmd.tokenize(mtmd_ctx, io, ectx.allocator, .{ .text = marker_replaced, .add_special = false, .parse_special = true }, &.{bitmap});
        defer chunks.deinit();
        expanded = try inputChunksToTokenizedSegments(ectx.allocator, &chunks, audio_token_id, .audio);
    } else {
        expanded = try tokenizeWithMediaPlaceholders(ectx, formatted_prompt, audio_token_id, @intCast(n_audio_tokens), .audio);
    }
    defer expanded.deinit();
    try multimodalPrefillUnified(ectx, io, &expanded, audio_token_id, @intCast(n_audio_tokens), audio_embeddings, max_tokens, formatted_prompt, "Audio");
}

fn multimodalPrefillUnified(
    ectx: *EngineContext,
    io: std.Io,
    expanded: *chat_template.TokenizedSegments,
    media_token_id: u32,
    n_media_tokens: i32,
    media_embeddings: *ggml.Tensor,
    max_tokens: u32,
    prompt_text: ?[]const u8,
    label: []const u8,
) !void {
    const media_offset: u32 = if (expanded.offsets.len > 0) expanded.offsets[0].token_offset else 0;
    const n_total_tokens: i32 = @intCast(expanded.tokens.items.len);
    const media_count: u32 = if (expanded.offsets.len > 0) expanded.offsets[0].token_count else @intCast(n_media_tokens);
    // For M-RoPE: n_pos (position counter advance) != n_tokens.
    // Mirrors C++ mtmd_helper_decode_image_chunk: n_past += mtmd_input_chunk_get_n_pos(chunk).
    const media_n_pos: i32 = if (expanded.offsets.len > 0 and expanded.offsets[0].n_pos > 0)
        @intCast(expanded.offsets[0].n_pos)
    else
        @intCast(media_count);
    const prefix_tokens = if (media_offset > 0) expanded.tokens.items[0..media_offset] else &[_]u32{};
    const suffix_start: u32 = media_offset + media_count;
    const suffix_tokens = if (suffix_start < n_total_tokens) expanded.tokens.items[suffix_start..@as(usize, @intCast(n_total_tokens))] else &[_]u32{};

    logger.info("=== Multimodal embedding substitution check ({s}) ===", .{label});
    logger.info("  media_offset    = {d}", .{media_offset});
    logger.info("  media_count     = {d}", .{media_count});
    logger.info("  n_media_tokens  = {d}", .{n_media_tokens});
    logger.info("  media_n_pos     = {d}", .{media_n_pos});
    logger.info("  prefix_tokens   = {d}", .{prefix_tokens.len});
    logger.info("  suffix_tokens   = {d}", .{suffix_tokens.len});

    const embd_raw = try media_embeddings.dataGet(f32, ectx.allocator);
    defer ectx.allocator.free(embd_raw);
    const embd_dim: u32 = @intCast(media_embeddings.ne()[0]);
    const embd_heap = try ectx.allocator.dupe(f32, embd_raw);
    defer ectx.allocator.free(embd_heap);

    const pr = try prefill_mod.threeStagePrefill(
        ectx.ctx_graph,
        ectx.model,
        ectx.kv_cache_mgr,
        prefix_tokens,
        media_token_id,
        @intCast(media_count),
        media_n_pos,
        embd_heap,
        embd_dim,
        suffix_tokens,
        &ectx.params,
        ectx.n_threads,
        ectx.allocator,
        ectx.gallocr,
    );

    if (ectx.verbose_prompt) {
        try verbose_mod.printVerbosePrompt(io, ectx.allocator, ectx.tok, prompt_text, expanded.tokens.items, pr.logits, expanded.offsets);
    }

    var best_idx: i32 = 0;
    var best_val: f32 = pr.logits[0];
    for (pr.logits, 0..) |v, j| {
        if (v > best_val) {
            best_val = v;
            best_idx = @intCast(j);
        }
    }
    ectx.allocator.free(pr.logits);

    try decode_mod.reserveDecodeGallocr(ectx.allocator, ectx.kv_cache_mgr, ectx.inc_ctx, ectx.model, &ectx.params);

    _ = try decode_mod.runDecodeLoop(
        ectx.allocator,
        io,
        ectx.model,
        &ectx.params,
        ectx.tok,
        ectx.kv_cache_mgr,
        ectx.inc_ctx,
        ectx.n_threads,
        best_idx,
        pr.pos,
        max_tokens,
        .{
            .sample = struct {
                fn f(_: *anyopaque, l: *ggml.Tensor) i32 {
                    return sampler.Sampler.sampleGreedy(l);
                }
            }.f,
            .skipToken = struct {
                fn f(c: *anyopaque, t: i32) bool {
                    const eng: *EngineContext = @ptrCast(@alignCast(c));
                    return eng.tok.isSkipToken(@intCast(t));
                }
            }.f,
            .onComplete = null,
        },
        @ptrCast(ectx),
        ectx.benchmark,
    );
}

pub fn tokenizeWithMediaPlaceholders(
    ectx: *EngineContext,
    formatted_prompt: []const u8,
    media_token_id: u32,
    media_token_count: u32,
    media_type: chat_template.MediaType,
) !chat_template.TokenizedSegments {
    const all_placeholders = try chat_template.scanPlaceholders(formatted_prompt, ectx.allocator, if (ectx.mtmd_context) |ctx| chat_template.ScanMarkers{
        .img_beg = ctx.img_beg,
        .img_end = ctx.img_end,
        .aud_beg = ctx.aud_beg,
        .aud_end = ctx.aud_end,
    } else null);
    defer ectx.allocator.free(all_placeholders);

    const beg_marker: []const u8 = blk: {
        if (ectx.mtmd_context) |ctx| {
            break :blk switch (media_type) {
                .image => ctx.img_beg,
                .audio => ctx.aud_beg,
            };
        }
        break :blk "";
    };
    const end_marker: []const u8 = blk: {
        if (ectx.mtmd_context) |ctx| {
            break :blk switch (media_type) {
                .image => ctx.img_end,
                .audio => ctx.aud_end,
            };
        }
        break :blk "";
    };

    var beg_tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
    defer beg_tokens.deinit(ectx.allocator);
    if (beg_marker.len > 0) {
        var encoded = try ectx.tok.encode(beg_marker, false, true);
        defer encoded.deinit(ectx.allocator);
        try beg_tokens.appendSlice(ectx.allocator, encoded.items);
    }

    var end_tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
    defer end_tokens.deinit(ectx.allocator);
    if (end_marker.len > 0) {
        var encoded = try ectx.tok.encode(end_marker, false, true);
        defer encoded.deinit(ectx.allocator);
        try end_tokens.appendSlice(ectx.allocator, encoded.items);
    }

    var new_tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
    errdefer new_tokens.deinit(ectx.allocator);
    var offsets = std.ArrayListUnmanaged(chat_template.PlaceholderInfo){ .items = &.{}, .capacity = 0 };
    errdefer offsets.deinit(ectx.allocator);

    var consumed: usize = 0;
    for (all_placeholders) |ph| {
        if (ph.media_type != media_type) continue;
        const text_segment = formatted_prompt[consumed..ph.start];
        if (text_segment.len > 0) {
            var encoded = try ectx.tok.encode(text_segment, false, true);
            defer encoded.deinit(ectx.allocator);
            try new_tokens.appendSlice(ectx.allocator, encoded.items);
        }
        for (beg_tokens.items) |t| {
            try new_tokens.append(ectx.allocator, t);
        }
        try offsets.append(ectx.allocator, .{ .start = 0, .length = 0, .media_type = media_type, .token_count = media_token_count, .token_offset = @intCast(new_tokens.items.len) });
        for (0..media_token_count) |_| {
            try new_tokens.append(ectx.allocator, media_token_id);
        }
        for (end_tokens.items) |t| {
            try new_tokens.append(ectx.allocator, t);
        }
        consumed = ph.start + ph.length;
    }

    if (consumed < formatted_prompt.len) {
        var encoded = try ectx.tok.encode(formatted_prompt[consumed..], false, true);
        defer encoded.deinit(ectx.allocator);
        try new_tokens.appendSlice(ectx.allocator, encoded.items);
    }

    return chat_template.TokenizedSegments{
        .tokens = new_tokens,
        .offsets = try offsets.toOwnedSlice(ectx.allocator),
        .allocator = ectx.allocator,
    };
}

/// Convert InputChunks to TokenizedSegments for use with multimodalPrefillUnified.
fn inputChunksToTokenizedSegments(
    allocator: std.mem.Allocator,
    chunks: *const mtmd.InputChunks,
    media_token_id: u32,
    media_type: chat_template.MediaType,
) !chat_template.TokenizedSegments {
    var tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
    errdefer tokens.deinit(allocator);
    var offsets = std.ArrayListUnmanaged(chat_template.PlaceholderInfo){ .items = &.{}, .capacity = 0 };
    errdefer offsets.deinit(allocator);

    for (chunks.entries.items) |*chunk| {
        switch (chunk.chunk_type) {
            .text => {
                const text_tokens = chunk.tokens_text orelse &.{};
                for (text_tokens) |t| {
                    try tokens.append(allocator, @intCast(t));
                }
            },
            .image => {
                if (media_type == .image) {
                    const n_tokens = chunk.nTokens();
                    // Forward ImageTokens metadata for M-RoPE position computation.
                    // Mirrors C++ mtmd_helper_decode_image_chunk which reads nx/ny/pos
                    // from mtmd_image_tokens for mtmd_helper_image_get_decoder_pos.
                    const img_tok = chunk.tokens_image;
                    const img_pos_type: chat_template.ImagePosType = if (img_tok) |it| switch (it.pos) {
                        .mrope => .mrope,
                        .hunyuanvl => .hunyuanvl,
                        .normal => .normal,
                    } else .normal;
                    const img_n_pos = chunk.nPos();
                    const img_grid_nx = if (img_tok) |it| it.nx else 0;
                    const img_grid_ny = if (img_tok) |it| it.ny else 0;
                    const img_idx = if (img_tok) |it| it.image_idx else 0;

                    try offsets.append(allocator, .{
                        .start = 0,
                        .length = 0,
                        .media_type = .image,
                        .token_count = n_tokens,
                        .token_offset = @intCast(tokens.items.len),
                        .n_pos = img_n_pos,
                        .grid_nx = img_grid_nx,
                        .grid_ny = img_grid_ny,
                        .pos_type = img_pos_type,
                        .image_idx = img_idx,
                    });
                    for (0..n_tokens) |_| {
                        try tokens.append(allocator, media_token_id);
                    }
                }
            },
            .audio => {
                if (media_type == .audio) {
                    const n_tokens = chunk.tokens_audio_n;
                    const n_pos = chunk.nPos();
                    try offsets.append(allocator, .{
                        .start = 0,
                        .length = 0,
                        .media_type = .audio,
                        .token_count = n_tokens,
                        .token_offset = @intCast(tokens.items.len),
                        .n_pos = n_pos,
                    });
                    for (0..n_tokens) |_| {
                        try tokens.append(allocator, media_token_id);
                    }
                }
            },
        }
    }

    return chat_template.TokenizedSegments{
        .tokens = tokens,
        .offsets = try offsets.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

/// Replace image/audio placeholders in formatted_prompt with the given media_marker.
fn replacePlaceholdersWithMarker(allocator: std.mem.Allocator, prompt: []const u8, marker: []const u8) ![]u8 {
    const Span = struct { start: usize, end: usize };
    var positions = std.ArrayListUnmanaged(Span){ .items = &.{}, .capacity = 0 };
    defer positions.deinit(allocator);

    var pos: usize = 0;
    while (pos < prompt.len) {
        if (std.mem.indexOfPos(u8, prompt, pos, "<|image|>")) |idx| {
            try positions.append(allocator, .{ .start = idx, .end = idx + "<|image|>".len });
            pos = idx + "<|image|>".len;
        } else if (std.mem.indexOfPos(u8, prompt, pos, "<image>")) |idx| {
            try positions.append(allocator, .{ .start = idx, .end = idx + "<image>".len });
            pos = idx + "<image>".len;
        } else if (std.mem.indexOfPos(u8, prompt, pos, "<|audio|>")) |idx| {
            try positions.append(allocator, .{ .start = idx, .end = idx + "<|audio|>".len });
            pos = idx + "<|audio|>".len;
        } else if (std.mem.indexOfPos(u8, prompt, pos, "<audio>")) |idx| {
            try positions.append(allocator, .{ .start = idx, .end = idx + "<audio>".len });
            pos = idx + "<audio>".len;
        } else {
            break;
        }
    }

    if (positions.items.len == 0) {
        return allocator.dupe(u8, prompt);
    }

    var total_len: usize = prompt.len;
    for (positions.items) |p| {
        const placeholder_len = p.end - p.start;
        if (marker.len > placeholder_len) {
            total_len += marker.len - placeholder_len;
        } else {
            total_len -= placeholder_len - marker.len;
        }
    }

    var result = try allocator.alloc(u8, total_len);
    var write_pos: usize = 0;
    var read_pos: usize = 0;

    for (positions.items) |p| {
        const segment_len = p.start - read_pos;
        @memcpy(result[write_pos..][0..segment_len], prompt[read_pos..][0..segment_len]);
        write_pos += segment_len;
        @memcpy(result[write_pos..][0..marker.len], marker);
        write_pos += marker.len;
        read_pos = p.end;
    }

    const remaining = prompt.len - read_pos;
    if (remaining > 0) {
        @memcpy(result[write_pos..][0..remaining], prompt[read_pos..]);
    }

    return result;
}
