//! 多模态视觉输出质量验证工具
//!
//! 使用 zllama.zig 引擎运行多模态图像推理并与 llama.cpp mtmd 参考输出对比。
//! 计算 NMSE、余弦相似度等指标，确保视觉编码器 + LLM 的正确性。
//!
//! 工作流:
//!   1. 用 llama.cpp mtmd 生成参考 logits:
//!      llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image hello.png --jinja -p ":" --logit-binary ref.bin
//!   2. 用本工具对比:
//!      zllama-compare-mtmd-vision --model model.gguf --mmproj mmproj.gguf --image hello.png --prompt "Describe this image" --ref-logits ref.bin
//!
//! 用法:
//!   zllama-compare-mtmd-vision --model <path> --mmproj <path> --image <path> --prompt <text> --ref-logits <file>

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const kv_cache = @import("kv_cache");
const mm = @import("mm");
const preprocess = @import("preprocess");
const chat_template = @import("chat_template");
const engine_common = @import("engine_common");

const log = std.log.scoped(.compare_mtmd_vision);

// ============================================================================
// 配置
// ============================================================================

pub const CompareMtmdVisionConfig = struct {
    model_path: []const u8 = "",
    mmproj_path: []const u8 = "",
    image_path: []const u8 = "",
    prompt: []const u8 = "Describe this image",
    ref_logits_path: []const u8 = "",
    /// NMSE 通过阈值（视觉编码器可能使用量化，放宽阈值）
    nmse_threshold: f64 = 1e-3,
    /// 余弦相似度通过阈值
    cosine_threshold: f64 = 0.999,
    /// 线程数
    n_threads: i32 = 0,
};

// ============================================================================
// Tokenize callback for multimodal placeholder expansion
// ============================================================================

fn tokenizeTextSegment(ctx: ?*anyopaque, text: []const u8, alloc: std.mem.Allocator) ![]u32 {
    const tok: *tokenizer.Tokenizer = @ptrCast(@alignCast(ctx orelse return error.NullCtx));
    var result = try tok.encode(text, false);
    defer result.deinit(alloc);
    return try result.toOwnedSlice(alloc);
}

// ============================================================================
// 视觉对比器
// ============================================================================

pub const MtmdVisionComparator = struct {
    allocator: std.mem.Allocator,
    config: CompareMtmdVisionConfig,

    pub fn init(allocator: std.mem.Allocator, config: CompareMtmdVisionConfig) MtmdVisionComparator {
        return .{ .allocator = allocator, .config = config };
    }

    /// 运行对比：加载模型+mmproj → 图像编码 → 推理 → 与参考对比
    pub fn run(self: *MtmdVisionComparator, io: std.Io) !bool {
        // 1. 加载 GGUF 模型文件
        const dir = std.Io.Dir.cwd();
        const model_file = try dir.openFile(io, self.config.model_path, .{ .mode = .read_only });
        defer model_file.close(io);

        const model_stat = try model_file.stat(io);
        const model_file_size = @as(usize, @intCast(model_stat.size));
        const gguf_data = try self.allocator.alloc(u8, model_file_size);
        defer self.allocator.free(gguf_data);
        {
            var offset: u64 = 0;
            const chunk_size: usize = 64 * 1024 * 1024;
            while (offset < model_file_size) {
                const end = @min(offset + chunk_size, model_file_size);
                const len = end - offset;
                const bytes_read = try model_file.readPositionalAll(io, gguf_data[offset..][0..len], offset);
                if (bytes_read != len) return error.FileReadError;
                offset += bytes_read;
            }
        }

        var gguf_file = try gguf.parse(gguf_data, self.allocator);
        defer gguf_file.deinit();

        // 2. 检测架构（多模态当前仅支持 gemma4）
        const arch = registry.detectArchitecture(&gguf_file) orelse {
            log.err("Could not detect architecture from {s}", .{self.config.model_path});
            return error.UnsupportedArchitecture;
        };
        log.info("Detected architecture: {s}", .{@tagName(arch)});

        if (arch != .gemma4) {
            log.err("Vision comparison only supports gemma4 architecture, got {s}", .{@tagName(arch)});
            return error.UnsupportedArchitecture;
        }

        // 3. 创建模型
        var model = try registry.createModel(self.allocator, &gguf_file, arch, io);
        defer model.deinit(self.allocator);
        const params = model.getParams();

        // 4. 初始化 tokenizer
        var tok = try tokenizer.Tokenizer.init(&gguf_file, self.allocator);
        defer tok.deinit();
        log.info("Tokenizer: {d} tokens", .{tok.vocabSize()});

        // 5. 加载 mmproj
        var capabilities = registry.detectCapabilities(&gguf_file, arch);

        const mmproj_file = try dir.openFile(io, self.config.mmproj_path, .{ .mode = .read_only });
        defer mmproj_file.close(io);

        const mmproj_stat = try mmproj_file.stat(io);
        const mmproj_file_size = @as(usize, @intCast(mmproj_stat.size));
        const mmproj_data = try self.allocator.alloc(u8, mmproj_file_size);
        defer self.allocator.free(mmproj_data);
        {
            var offset: u64 = 0;
            const chunk_size: usize = 64 * 1024 * 1024;
            while (offset < mmproj_file_size) {
                const end = @min(offset + chunk_size, mmproj_file_size);
                const len = end - offset;
                const bytes_read = try mmproj_file.readPositionalAll(io, mmproj_data[offset..][0..len], offset);
                if (bytes_read != len) return error.FileReadError;
                offset += bytes_read;
            }
        }

        var mmproj_gguf = try gguf.parse(mmproj_data, self.allocator);
        defer mmproj_gguf.deinit();

        // Detect capabilities from mmproj
        if (mmproj_gguf.findTensor("v.patch_embd.weight") != null or
            mmproj_gguf.findTensor("mm.input_projection.weight") != null or
            mmproj_gguf.findTensor("mm.soft_emb_norm.weight") != null)
        {
            capabilities.has_vision = true;
            if (capabilities.vision_encoder_type.len == 0) {
                capabilities.vision_encoder_type = "ViT (SigLIP/Gemma4V)";
            }
        }

        if (!capabilities.has_vision) {
            log.err("mmproj file does not contain a vision encoder", .{});
            return error.VisionNotSupported;
        }

        const mm_ctx = try ggml.Context.initNoAlloc(2 * 1024 * 1024 * 1024);
        defer mm_ctx.deinit();

        var mm_mgr = try mm.MultiModalManager.init(self.allocator, &mmproj_gguf, mm_ctx, capabilities);
        defer mm_mgr.deinit();

        log.info("Vision encoder loaded: {s}", .{capabilities.vision_encoder_type});

        // 6. 加载并预处理图像
        const target_size: u32 = if (mm_mgr.vision_encoder) |enc| enc.params.image_size else 896;
        var img = try preprocess.loadImage(self.allocator, io, self.config.image_path, target_size, .auto);
        defer img.deinit();

        if (img.width == 0 or img.height == 0) {
            log.err("Loaded image has zero dimensions", .{});
            return error.EmptyImage;
        }
        log.info("Loaded image: {d}x{d} -> {d}x{d}", .{ img.width, img.height, target_size, target_size });

        // 7. 运行视觉编码器
        const graph_ctx = try ggml.Context.initNoAlloc(512 * 1024 * 1024);
        defer graph_ctx.deinit();

        graph_ctx.setNoAlloc(false);
        var vision_graph = try ggml.CGraph.initReserved(graph_ctx, 32768);
        const vision_embeddings = try mm_mgr.encodeMedia(graph_ctx, vision_graph, .{
            .media_type = .image,
            .image_data = img.data,
            .image_width = img.width,
            .image_height = img.height,
        });
        graph_ctx.setNoAlloc(true);

        const buft = ggml.backendCpuBufferType();
        var v_galloc = try ggml.Gallocr.init(buft);
        defer v_galloc.free();
        if (!v_galloc.allocGraph(vision_graph)) {
            return error.GraphAllocFailed;
        }
        try vision_graph.compute(self.config.n_threads);

        const n_vision_tokens: i32 = @intCast(vision_embeddings.ne()[1]);
        const n_embd_val: usize = @intCast(vision_embeddings.ne()[0]);
        log.info("Vision encoder output: [{d}, {d}]", .{ n_embd_val, n_vision_tokens });

        const model_n_embd: usize = @intCast(params.n_embd);
        if (n_embd_val != model_n_embd) {
            log.err("Vision encoder output dim {d} != model n_embd {d}", .{ n_embd_val, model_n_embd });
            return error.EmbeddingDimensionMismatch;
        }

        // 8. 查找图像占位符 token
        const image_token_id: u32 = blk: {
            if (tok.textToToken("<|image|>")) |id| break :blk @as(u32, @intCast(id));
            if (tok.textToToken("<image>")) |id| break :blk @as(u32, @intCast(id));
            log.err("No <|image|> or <image> token in vocabulary!", .{});
            return error.NoImagePlaceholderToken;
        };
        log.info("Image placeholder token id: {d}", .{image_token_id});

        // 9. 格式化 prompt（使用 chat template）
        const content_with_placeholder = try chat_template.ensurePlaceholderInContent(self.config.prompt, .image, self.allocator);
        defer if (content_with_placeholder.ptr != self.config.prompt.ptr) self.allocator.free(content_with_placeholder);

        const source = chat_template.TemplateSource{ .preset = chat_template.kindForArchitecture(arch, null) };
        var tmpl = try chat_template.resolve(self.allocator, source, arch, null, true);
        defer tmpl.deinit(self.allocator);

        // Use withMedia to pass media info to Jinja template engine
        const media = chat_template.Media{
            .type = .image,
            .data = .{ .image = .{ .data = &.{}, .width = 0, .height = 0 } },
        };
        const messages = [_]chat_template.ChatMessage{
            chat_template.ChatMessage.withMedia("user", content_with_placeholder, media),
        };
        const formatted_prompt = try tmpl.apply(self.allocator, &messages, null, true);
        defer self.allocator.free(formatted_prompt);

        log.info("Formatted prompt ({d} chars):\n{s}", .{ formatted_prompt.len, formatted_prompt });

        // 10. Tokenize 并展开占位符
        var expanded = try chat_template.tokenizeWithPlaceholders(
            self.allocator,
            formatted_prompt,
            @ptrCast(&tok),
            tokenizeTextSegment,
            image_token_id,
            0, // audio_token_id (unused)
            @intCast(n_vision_tokens),
            0, // audio_token_count (unused)
        );
        defer expanded.deinit();

        const n_total_tokens: i32 = @intCast(expanded.tokens.items.len);
        log.info("Total tokens: {d} (including {d} vision tokens)", .{ n_total_tokens, n_vision_tokens });

        // 11. 构建 LLM 推理图（使用 forwardWithEmbdOverride）
        graph_ctx.setNoAlloc(false);
        const input_tensor = try graph_ctx.newTensor1d(.i32, n_total_tokens);
        {
            const data = input_tensor.dataBytes();
            const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_total_tokens))];
            for (expanded.tokens.items, 0..) |token, j| {
                slice[j] = @as(i32, @intCast(token));
            }
        }

        // Setup KV cache context
        const kv_cache_ctx = try ggml.Context.init(2 * 1024 * 1024 * 1024);
        defer kv_cache_ctx.deinit();

        const max_seq_len = @min(params.max_seq_len, 2048);
        const hdim_kv = params.n_head_dim;
        const hdim_k = @max(params.n_head_dim, params.n_head_dim_k);
        const hdim_v = if (params.n_head_dim_v > 0) @max(params.n_head_dim, params.n_head_dim_v) else hdim_kv;
        var kv_cache_mgr = try kv_cache.KVCache.initWithKVDim(kv_cache_ctx, params.n_layer, params.n_kv_head, hdim_k, hdim_v, max_seq_len, self.allocator);
        defer kv_cache_mgr.deinit(self.allocator);

        model.setKVCacheContext(kv_cache_ctx);

        // Access Gemma4Model directly for forwardWithEmbdOverride
        const gemma4_model: *model_if.gemma4.Gemma4Model = @ptrCast(@alignCast(model.ptr));

        // Calculate the token offset for the image placeholder in the token sequence
        const vision_embd_offset: i32 = if (expanded.offsets.len > 0)
            @intCast(expanded.offsets[0].token_offset)
        else
            0;
        log.info("Vision embedding offset in token sequence: {d} (total tokens: {d})", .{ vision_embd_offset, n_total_tokens });

        var graph = try ggml.CGraph.initReserved(graph_ctx, 16384);
        const logits_tensor = try gemma4_model.forwardWithEmbdOverride(
            graph_ctx, graph, input_tensor, n_total_tokens,
            @ptrCast(&kv_cache_mgr), 0, vision_embeddings, vision_embd_offset, false,
        );
        graph_ctx.setNoAlloc(true);

        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();
        if (!galloc.allocGraph(graph)) {
            return error.GraphAllocFailed;
        }

        const n_threads: i32 = if (self.config.n_threads > 0) self.config.n_threads else @as(i32, @intCast(@min(4, @max(1, try std.Thread.getCpuCount() - 1))));
        try graph.compute(n_threads);

        // 12. 读取 logits
        const logits_data = logits_tensor.dataBytes();
        const n_vocab = @as(usize, @intCast(params.n_vocab));
        const our_logits = @as([*]f32, @ptrCast(@alignCast(logits_data.ptr)))[0..n_vocab];

        // 13. 加载参考 logits
        const ref_logits = try self.loadReferenceLogits(io);
        defer self.allocator.free(ref_logits);

        if (ref_logits.len != n_vocab) {
            log.err("Reference logits length mismatch: expected {d}, got {d}", .{ n_vocab, ref_logits.len });
            return error.SizeMismatch;
        }

        // 14. 计算指标
        const nmse = calcNMSE(our_logits, ref_logits);
        const cos_sim = calcCosineSimilarity(our_logits, ref_logits);
        const max_abs_err = calcMaxAbsError(our_logits, ref_logits);
        const argmax_match = calcArgmaxMatch(our_logits, ref_logits);

        // 15. 输出结果
        const stdout_file = std.Io.File.stdout();
        try stdout_file.writeStreamingAll(io, "\n=== zllama.zig Vision vs llama.cpp mtmd Comparison ===\n");
        try printMetric(io, "NMSE", nmse, self.config.nmse_threshold, true);
        try printMetric(io, "Cosine Similarity", cos_sim, self.config.cosine_threshold, false);
        try printMetric(io, "Max Abs Error", max_abs_err, 0.01, true);
        try printArgmaxResult(io, argmax_match);

        const passed = nmse < self.config.nmse_threshold and cos_sim > self.config.cosine_threshold;
        if (passed) {
            try stdout_file.writeStreamingAll(io, "\n✅ PASS: Vision logits match reference within tolerance.\n");
        } else {
            try stdout_file.writeStreamingAll(io, "\n❌ FAIL: Vision logits deviate from reference.\n");
        }
        try stdout_file.writeStreamingAll(io, "=====================================================\n");

        return passed;
    }

    fn loadReferenceLogits(self: *MtmdVisionComparator, io: std.Io) ![]f32 {
        const dir = std.Io.Dir.cwd();
        const ref_file = try dir.openFile(io, self.config.ref_logits_path, .{ .mode = .read_only });
        defer ref_file.close(io);

        const stat = try ref_file.stat(io);
        const size = stat.size;
        if (size % @sizeOf(f32) != 0) {
            log.err("Reference file size ({d}) is not a multiple of f32 size ({d})", .{ size, @sizeOf(f32) });
            return error.InvalidReferenceFile;
        }
        const n = size / @sizeOf(f32);
        const buf = try self.allocator.alloc(f32, n);
        errdefer self.allocator.free(buf);

        const bytes = std.mem.sliceAsBytes(buf);
        const nread = try ref_file.readPositionalAll(io, bytes, 0);
        if (nread != size) return error.UnexpectedEndOfFile;

        return buf;
    }
};

// ============================================================================
// 指标计算
// ============================================================================

fn calcNMSE(a: []const f32, b: []const f32) f64 {
    var sum_sq_err: f64 = 0.0;
    var sum_sq_ref: f64 = 0.0;
    for (a, b) |av, bv| {
        const err: f64 = @as(f64, @floatCast(av)) - @as(f64, @floatCast(bv));
        sum_sq_err += err * err;
        sum_sq_ref += @as(f64, @floatCast(av)) * @as(f64, @floatCast(av));
    }
    return sum_sq_err / (sum_sq_ref + 1e-10);
}

fn calcCosineSimilarity(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0.0;
    var norm_a: f64 = 0.0;
    var norm_b: f64 = 0.0;
    for (a, b) |av, bv| {
        dot += @as(f64, @floatCast(av)) * @as(f64, @floatCast(bv));
        norm_a += @as(f64, @floatCast(av)) * @as(f64, @floatCast(av));
        norm_b += @as(f64, @floatCast(bv)) * @as(f64, @floatCast(bv));
    }
    return dot / (@sqrt(norm_a) * @sqrt(norm_b) + 1e-10);
}

fn calcMaxAbsError(a: []const f32, b: []const f32) f32 {
    var max_err: f32 = 0.0;
    for (a, b) |av, bv| {
        const err = @abs(av - bv);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

const ArgmaxResult = struct { ours: usize, ref: usize, match: bool };

fn calcArgmaxMatch(a: []const f32, b: []const f32) ArgmaxResult {
    var max_ours: f32 = -std.math.inf(f32);
    var max_ref: f32 = -std.math.inf(f32);
    var idx_ours: usize = 0;
    var idx_ref: usize = 0;
    for (a, 0..) |v, i| {
        if (v > max_ours) { max_ours = v; idx_ours = i; }
    }
    for (b, 0..) |v, i| {
        if (v > max_ref) { max_ref = v; idx_ref = i; }
    }
    return .{ .ours = idx_ours, .ref = idx_ref, .match = idx_ours == idx_ref };
}

fn printMetric(io: std.Io, name: []const u8, value: anytype, threshold: anytype, lower_is_better: bool) !void {
    const stdout_file = std.Io.File.stdout();
    var buf: [256]u8 = undefined;
    const pass = if (lower_is_better) value < threshold else value > threshold;
    const status = if (pass) "✅" else "❌";
    const line = try std.fmt.bufPrint(&buf, "  {s} {s}: {e} (threshold: {e})\n", .{ status, name, value, threshold });
    try stdout_file.writeStreamingAll(io, line);
}

fn printArgmaxResult(io: std.Io, argmax: ArgmaxResult) !void {
    const stdout_file = std.Io.File.stdout();
    var buf: [256]u8 = undefined;
    const status = if (argmax.match) "✅" else "❌";
    const line = try std.fmt.bufPrint(&buf, "  {s} Argmax: ours={d}, ref={d}, match={}\n", .{ status, argmax.ours, argmax.ref, argmax.match });
    try stdout_file.writeStreamingAll(io, line);
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    _ = args_iter.next(); // skip argv[0]

    var config = CompareMtmdVisionConfig{};

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            config.model_path = args_iter.next() orelse {
                log.err("--model requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--mmproj")) {
            config.mmproj_path = args_iter.next() orelse {
                log.err("--mmproj requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--image")) {
            config.image_path = args_iter.next() orelse {
                log.err("--image requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            config.prompt = args_iter.next() orelse {
                log.err("--prompt requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--ref-logits")) {
            config.ref_logits_path = args_iter.next() orelse {
                log.err("--ref-logits requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--threads")) {
            const t_str = args_iter.next() orelse {
                log.err("-t requires a value", .{});
                std.process.exit(1);
            };
            config.n_threads = std.fmt.parseInt(i32, t_str, 10) catch {
                log.err("Invalid value for -t: {s}", .{t_str});
                std.process.exit(1);
            };
        }
    }

    if (config.model_path.len == 0 or config.mmproj_path.len == 0 or
        config.image_path.len == 0 or config.ref_logits_path.len == 0)
    {
        log.err("Usage: zllama-compare-mtmd-vision --model <path> --mmproj <path> --image <path> --prompt <text> --ref-logits <file>", .{});
        std.process.exit(1);
    }

    var comparator = MtmdVisionComparator.init(allocator, config);
    const passed = comparator.run(io) catch |err| {
        log.err("Vision comparison failed: {}", .{err});
        std.process.exit(1);
    };

    if (!passed) std.process.exit(1);
}
