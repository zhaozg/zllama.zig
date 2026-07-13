//! Multimodal generation — vision + audio inference pipelines.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.
//! Contains generateWithImage, generateWithAudio, multimodalPrefillUnified,
//! and tokenizeWithMediaPlaceholders.
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

const audio_mod = mtmd.audio_mod;

const logger = std.log.scoped(.core_multimodal);

/// Engine reference needed by the multimodal generation paths.
pub const EngineContext = struct {
    allocator: std.mem.Allocator,
    ctx_graph: *ggml.Context,
    arch: model_if.Architecture,
    model: model_if.ModelInstance,
    params: model_if.ModelParams,
    tok: tokenizer.Tokenizer,
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

pub const ImageGenResult = struct {};

/// 计算动态分辨率的目标尺寸。
/// 对于支持动态分辨率的模型（如 Qwen3VL），使用 calcSizePreservedRatio；
/// 对于固定尺寸模型（如 Gemma4），直接使用 image_size。
fn computeTargetSize(
    enc: *const mtmd.vision_mod.VisionEncoder,
    src_width: u32,
    src_height: u32,
) preprocess.Size2D {
    const p = &enc.params;
    // 如果设置了 min_pixels 和 max_pixels，使用动态分辨率
    // 优先使用用户指定的 max_pixels（用于内存控制）
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

    // 回退：固定正方形缩放
    logger.info("Fixed resize: {d}x{d} -> {d}x{d}", .{ src_width, src_height, p.image_size, p.image_size });
    return .{ .width = p.image_size, .height = p.image_size };
}

/// 加载图像原始数据（不缩放），返回原始尺寸和 RGB 数据
fn loadImageRaw(allocator: std.mem.Allocator, io: std.Io, file_path: []const u8) !preprocess.ProcessedImage {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, file_path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const raw = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(raw);
    const total_read = try file.readPositionalAll(io, raw, 0);
    if (total_read != raw.len) return error.FileReadError;

    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const stb = @import("stb_image");
    const pixels = stb.loadFromMemory(raw.ptr, @intCast(raw.len), &w, &h, &comp, 3);
    if (pixels == null) {
        const reason = stb.failureReason();
        logger.err("stb_image failed to load {s}: {s}", .{ file_path, reason });
        return error.ImageDecodeFailed;
    }
    defer stb.free(pixels);

    const src_w: u32 = @intCast(w);
    const src_h: u32 = @intCast(h);
    const pixel_bytes = pixels.?[0..@as(usize, @intCast(src_w * src_h * 3))];

    const owned = try allocator.alloc(u8, pixel_bytes.len);
    @memcpy(owned, pixel_bytes);

    logger.info("Loaded raw image: {d}x{d}", .{ src_w, src_h });

    return preprocess.ProcessedImage{
        .data = owned,
        .width = src_w,
        .height = src_h,
        .allocator = allocator,
    };
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

    // 步骤 1: 先用 stb_image 加载原始图像尺寸（不缩放）
    var raw_img = try loadImageRaw(ectx.allocator, io, image_path);
    defer raw_img.deinit();

    logger.debug("generateWithImage: Step 1 - loadImageRaw done: {d}x{d}", .{ raw_img.width, raw_img.height });
    // 步骤 2: 计算目标尺寸（动态分辨率或固定正方形）
    // 如果用户指定了 image_max_pixels，覆盖 GGUF 默认值
    if (ectx.image_max_pixels > 0) {
        if (mm_mgr.vision_encoder) |*enc| {
            enc.setUserMaxPixels(ectx.image_max_pixels);
        }
    }
    logger.debug("generateWithImage: Step 2 - computeTargetSize", .{});
    const enc = mm_mgr.vision_encoder.?;
    const target = computeTargetSize(&enc, raw_img.width, raw_img.height);

    // 步骤 3: 缩放到目标尺寸
    logger.debug("generateWithImage: Step 3 - resizeRGB to {d}x{d}", .{ target.width, target.height });
    const resized_data = try preprocess.resizeRGB(ectx.allocator, raw_img.data, raw_img.width, raw_img.height, target.width, target.height);
    var img = preprocess.ProcessedImage{
        .data = resized_data,
        .width = target.width,
        .height = target.height,
        .allocator = ectx.allocator,
    };
    defer img.deinit();

    if (img.width == 0 or img.height == 0) return error.EmptyImage;

    // 使用 no_alloc = true 模式创建视觉编码器上下文。
    // 张量数据由 gallocr 在 computeGraph 中分配，context 仅存储元数据。
    // 输入张量的数据由 normalizeToTensor 在 no_alloc 模式下手动分配。
    var vision_ctx = try ggml.Context.initNoAlloc(256 * 1024 * 1024);
    defer vision_ctx.deinit();
    const vision_graph = try ggml.CGraph.initReserved(vision_ctx, 32768);
    logger.debug("generateWithImage: Step 4 - creating vision_ctx and calling encodeMedia...", .{});
    const vision_embeddings = try mm_mgr.encodeMedia(io, vision_ctx, vision_graph, .{
        .media_type = .image,
        .image_data = img.data,
        .image_width = img.width,
        .image_height = img.height,
    }, ectx.n_threads);

    // Compute the vision graph with a dedicated Gallocr (matching helper.zig pattern).
    // The engine's Gallocr is sized for the text model; vision graph may need
    // different memory layout.
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
    } else try applyChatTemplateWithMedia(ectx, prompt, chat_template.Media.init(.image));
    defer ectx.allocator.free(formatted_prompt);

    logger.info("formatted_prompt: {s}", .{formatted_prompt});

    var expanded = try tokenizeWithMediaPlaceholders(ectx, formatted_prompt, image_token_id, @intCast(n_vision_tokens), .image);
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
    const wav_result = try audio_mod.loadWav(ectx.allocator, io, audio_path);
    defer ectx.allocator.free(wav_result.samples);
    if (wav_result.samples.len == 0) return error.EmptyAudio;

    debug.saveData(io, "debug_audio", "zllama_audio_01_samples.json", "audio_mel", wav_result.samples) catch |err| {
        logger.info("Save audio mel data fail: {}", .{err});
    };

    const is_gemma4ua = mm_mgr.audio_encoder != null and
        std.mem.eql(u8, ectx.capabilities.audio_encoder_type, "gemma4ua");

    var mel: audio_mod.types.ProcessedAudio = undefined;
    var mel_owned = false;

    if (is_gemma4ua) {
        // Gemma4UA 预处理：直接把原始 PCM 样本分帧，不做 FFT/Mel 频谱
        // 参考: llama.cpp mtmd_audio_preprocessor_gemma4ua::preprocess()
        // frame_size 从 mm.a.input_projection.weight->ne[0] 推断
        const frame_size: u32 = if (mm_mgr.audio_encoder) |enc|
            @intCast(enc.weights.mm_input_proj_w.?.ne()[0])
        else
            return error.AudioEncoderNotAvailable;

        logger.info("Gemma4UA preprocessing: frame_size={d}, samples={d}", .{ frame_size, wav_result.samples.len });
        mel = try audio_mod.processRawWaveform(ectx.allocator, wav_result.samples, frame_size);
        mel_owned = true;

        debug.saveData(io, "debug_audio", "zllama_audio_02_raw_waveform.json", "audio_raw", mel.data) catch |err| {
            logger.info("Save audio raw waveform data fail: {}", .{err});
        };
    } else {
        const preprocess_params = audio_mod.AudioPreprocessParams.fromAudioEncoder(if (mm_mgr.audio_encoder) |enc| enc.params.n_mel_bins else audio_mod.AUDIO_N_MEL_BINS);
        mel = try audio_mod.computeMelSpectrogram(io, ectx.allocator, wav_result.samples, wav_result.info.sample_rate, preprocess_params);
        mel_owned = true;

        debug.saveData(io, "debug_audio", "zllama_audio_02_mel_conformer.json", "audio_mel", mel.data) catch |err| {
            logger.info("Save audio mel data fail: {}", .{err});
        };
    }
    defer if (mel_owned) mel.deinit();

    // 使用独立的 ggml context 构建音频编码器图，避免与主 LLM context 冲突。
    // 视觉路径 (generateWithImage) 也使用独立 context，这是相同的模式。
    // 张量数据由 gallocr 在 computeGraph 中分配，context 仅存储元数据。
    var audio_ctx = try ggml.Context.initNoAlloc(256 * 1024 * 1024);
    defer audio_ctx.deinit();

    logger.debug("Audio encoder: created independent ggml context", .{});

    const audio_graph = try ggml.CGraph.initReserved(audio_ctx, 32768);

    const mel_tensor = try audio_mod.melToTensor(audio_ctx, mel.data, mel.n_frames, mel.n_mel_bins);
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

    // Compute the audio graph (encoder only builds, caller computes)
    // Compute the audio graph with a dedicated Gallocr (matching helper.zig pattern).
    logger.debug("Audio encoder: computing graph...", .{});
    const audio_buft = ggml.backendCpuBufferType();
    var audio_gallocr = try ggml.Gallocr.init(audio_buft);
    defer audio_gallocr.free();
    _ = try engine_common.computeGraph(audio_graph, ectx.n_threads, audio_gallocr);

    audio_ctx.setNoAlloc(true);

    debug.saveTensorFromGraph(io, ectx.allocator, "debug_audio", "zllama_audio_encoder_input.json", "debug_audio_encoder_input", audio_graph) catch |err| {
        logger.info("Save audio debug_audio_encoder_input data fail: {}", .{err});
    };
    if (mm_mgr.audio_encoder) |enc| {
        enc.saveDebugData(io, ectx.allocator, audio_graph);
    }
    const n_audio_tokens: i32 = @intCast(audio_embeddings.ne()[1]);
    const n_embd_val: usize = @intCast(audio_embeddings.ne()[0]);

    {
        const audio_data = try audio_embeddings.dataGet(f32, ectx.allocator);
        defer ectx.allocator.free(audio_data);
        debug.saveData(io, "debug_audio", "zllama_audio_embeddings.json", "audio_embeddings", audio_data) catch |err| {
            logger.info("Save audio embeddings data fail: {}", .{err});
        };
    }

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
    } else try applyChatTemplateWithMedia(ectx, prompt, chat_template.Media.init(.audio));
    defer ectx.allocator.free(formatted_prompt);

    logger.info("formatted_prompt: {s}", .{formatted_prompt});

    var expanded = try tokenizeWithMediaPlaceholders(ectx, formatted_prompt, audio_token_id, @intCast(n_audio_tokens), .audio);
    defer expanded.deinit();
    try multimodalPrefillUnified(ectx, io, &expanded, audio_token_id, @intCast(n_audio_tokens), audio_embeddings, max_tokens, formatted_prompt, "Audio");
}

/// Unified three-stage multimodal prefill + decode (all architectures via vtable).
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
    const n_total_tokens: i32 = @intCast(expanded.tokens.items.len);
    const media_offset: u32 = if (expanded.offsets.len > 0) expanded.offsets[0].token_offset else 0;
    const media_count: u32 = if (expanded.offsets.len > 0) expanded.offsets[0].token_count else @intCast(n_media_tokens);
    const prefix_tokens = if (media_offset > 0) expanded.tokens.items[0..media_offset] else &[_]u32{};
    const suffix_start: u32 = media_offset + media_count;
    const suffix_tokens = if (suffix_start < n_total_tokens) expanded.tokens.items[suffix_start..@as(usize, @intCast(n_total_tokens))] else &[_]u32{};

    logger.info("=== Multimodal embedding substitution check ({s}) ===", .{label});
    logger.info("  media_offset    = {d}", .{media_offset});
    logger.info("  media_count     = {d}", .{media_count});
    logger.info("  n_media_tokens  = {d}", .{n_media_tokens});
    logger.info("  prefix_tokens   = {d}", .{prefix_tokens.len});
    logger.info("  suffix_tokens   = {d}", .{suffix_tokens.len});

    const embd_raw = try media_embeddings.dataGet(f32, ectx.allocator);
    defer ectx.allocator.free(embd_raw);
    const embd_dim: u32 = @intCast(media_embeddings.ne()[0]);
    const embd_heap = try ectx.allocator.dupe(f32, embd_raw);
    defer ectx.allocator.free(embd_heap);

    // Embedding validation
    {
        const n_total: usize = embd_heap.len;
        const n_preview: usize = @min(n_total, 5);
        var all_zero = true;
        var has_nan = false;
        for (embd_heap) |v| {
            if (v != 0.0) all_zero = false;
            if (std.math.isNan(v)) has_nan = true;
        }
        logger.info("Embedding validation: total={d} preview={d:.4} {d:.4} {d:.4} {d:.4} {d:.4} all_zero={} has_nan={}", .{
            n_total,
            if (n_preview > 0) embd_heap[0] else @as(f32, 0),
            if (n_preview > 1) embd_heap[1] else @as(f32, 0),
            if (n_preview > 2) embd_heap[2] else @as(f32, 0),
            if (n_preview > 3) embd_heap[3] else @as(f32, 0),
            if (n_preview > 4) embd_heap[4] else @as(f32, 0),
            all_zero,
            has_nan,
        });
        if (all_zero) logger.warn("  ⚠ all embedding values are ZERO!", .{});
        if (has_nan) logger.warn("  ⚠ embedding contains NaN!", .{});
    }

    const pr = try prefill_mod.threeStagePrefill(
        ectx.ctx_graph,
        ectx.model,
        ectx.kv_cache_mgr,
        prefix_tokens,
        media_token_id,
        @intCast(media_count),
        embd_heap,
        embd_dim,
        suffix_tokens,
        &ectx.params,
        ectx.n_threads,
        ectx.allocator,
        ectx.gallocr,
    );

    if (ectx.verbose_prompt) {
        try verbose_mod.printVerbosePrompt(io, ectx.allocator, &ectx.tok, prompt_text, expanded.tokens.items, pr.logits, expanded.offsets);
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

    const dr = try decode_mod.runDecodeLoop(
        ectx.allocator,
        io,
        ectx.model,
        &ectx.params,
        &ectx.tok,
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
        },
        @ptrCast(ectx),
        ectx.benchmark,
    );

    if (!ectx.benchmark) {
        const sf = std.Io.File.stdout();
        try sf.writeStreamingAll(io, "\n");
    }
    const total_time_s = pr.pp_time_s + dr.tg_time_s;
    if (dr.gen_count > 0) logger.info("{s} Multimodal: {d} tokens in {d:.2}s ({d:.1} t/s)", .{ label, dr.gen_count, total_time_s, @as(f64, @floatFromInt(dr.gen_count)) / total_time_s });
}

// ============================================================================
// Shared helpers
// ============================================================================

fn applyChatTemplateWithMedia(ectx: *EngineContext, user_prompt: []const u8, media: chat_template.Media) ![]const u8 {
    if (ectx.no_chat_template) return ectx.allocator.dupe(u8, user_prompt);
    const model_name: ?[]const u8 = if (ectx.params.model_name.len > 0) ectx.params.model_name else null;
    const source = ectx.chat_template_source orelse chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(ectx.arch, model_name) };
    var tmpl = try chat_template.resolve(ectx.allocator, source, ectx.arch, model_name, !ectx.no_jinja);
    defer tmpl.deinit(ectx.allocator);

    // Prepend the media placeholder to the content so it's available for
    // both Jinja template rendering (where media info is not passed separately)
    // and preset template rendering (where appendMediaContent will skip adding
    // the marker if it's already present, avoiding double placeholders).
    const effective_prompt = try chat_template.ensurePlaceholderInContent(user_prompt, media.type, ectx.allocator, null);
    const needs_free = effective_prompt.ptr != user_prompt.ptr;
    defer if (needs_free) ectx.allocator.free(effective_prompt);

    const messages = [_]chat_template.ChatMessage{
        chat_template.ChatMessage.withMedia("user", effective_prompt, media),
    };
    const system = if (ectx.system_prompt.len > 0) ectx.system_prompt else null;
    return tmpl.apply(ectx.allocator, &messages, system, true);
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
