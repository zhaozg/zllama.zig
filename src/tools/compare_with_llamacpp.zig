//! 端到端推理数值对比工具
//!
//! 使用 zllama.zig 引擎运行推理并与 llama.cpp 参考输出对比。
//! 计算 NMSE、余弦相似度等指标，确保各模型架构正确性。
//!
//! 工作流:
//!   1. 用 llama.cpp 生成参考 logits:
//!      llama-simple --simple-io --logit-binary -m model.gguf -p "Hello" -n 1 > ref.bin
//!   2. 用本工具对比:
//!      zllama-compare-llamacpp --model model.gguf --prompt "Hello" --ref-logits ref.bin
//!
//! 用法:
//!   zllama-compare-llamacpp --model <path> --prompt <text> --ref-logits <file> [-n tokens]

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const tokenizer = @import("tokenizer");
const kv_cache = @import("kv_cache");
const metrics = @import("metrics");
const log = std.log.scoped(.tool_compare);

// ============================================================================
// 配置
// ============================================================================

pub const CompareWithLlamaConfig = struct {
    model_path: []const u8 = "",
    prompt: []const u8 = "Hello",
    ref_logits_path: []const u8 = "",
    /// 对比的 token 数（只对比前 n 个 token 的 logits）
    n_tokens: usize = 1,
    /// NMSE 通过阈值
    nmse_threshold: f64 = 1e-4,
    /// 余弦相似度通过阈值
    cosine_threshold: f64 = 0.999,
};

// ============================================================================
// 对比器
// ============================================================================

pub const LlamaCppComparator = struct {
    allocator: std.mem.Allocator,
    config: CompareWithLlamaConfig,

    pub fn init(allocator: std.mem.Allocator, config: CompareWithLlamaConfig) LlamaCppComparator {
        return .{ .allocator = allocator, .config = config };
    }

    /// 运行对比：加载模型 → 推理 → 与参考对比
    pub fn run(self: *LlamaCppComparator, io: std.Io) !bool {
        // 1. 加载 GGUF 文件
        const dir = std.Io.Dir.cwd();
        const file = try dir.openFile(io, self.config.model_path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const file_size = @as(usize, @intCast(stat.size));
        const gguf_data = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(gguf_data);
        {
            var offset: u64 = 0;
            const chunk_size: usize = 64 * 1024 * 1024;
            while (offset < file_size) {
                const end = @min(offset + chunk_size, file_size);
                const len = end - offset;
                const bytes_read = try file.readPositionalAll(io, gguf_data[offset..][0..len], offset);
                if (bytes_read != len) return error.FileReadError;
                offset += bytes_read;
            }
        }

        var gguf_file = try gguf.parse(gguf_data, self.allocator);
        defer gguf_file.deinit();

        // 2. 检测架构
        const arch = registry.detectArchitecture(&gguf_file) orelse {
            log.err("Could not detect architecture from {s}", .{self.config.model_path});
            return error.UnsupportedArchitecture;
        };
        log.info("Detected architecture: {s}", .{@tagName(arch)});

        // 3. 创建模型
        var model = try registry.createModel(self.allocator, &gguf_file, arch, io);
        defer model.deinit(self.allocator);
        const params = model.getParams();

        // 4. Tokenize
        var tok = try tokenizer.Tokenizer.init(&gguf_file, self.allocator);
        defer tok.deinit();

        var input_token_list = try tok.encode(self.config.prompt, false, false);
        defer input_token_list.deinit(self.allocator);
        const input_tokens = input_token_list.items;

        log.info("Prompt: \"{s}\" -> {d} tokens", .{ self.config.prompt, input_tokens.len });

        // 5. 推理
        const n_tokens: i32 = @intCast(input_tokens.len);
        const ctx = try ggml.Context.initNoAlloc(512 * 1024 * 1024);
        defer ctx.deinit();

        ctx.setNoAlloc(false);
        const input_tensor = try ctx.newTensor1d(.i32, n_tokens);

        const data = input_tensor.dataBytes();
        const dst = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
        for (input_tokens, 0..) |token, j| {
            dst[j] = @as(i32, @intCast(token));
        }

        const graph = try ggml.CGraph.initReserved(ctx, 32768);
        var builder = graph_builder.GraphBuilder.init(ctx, graph, params, self.allocator);

        // Create KV cache if needed (some models like Gemma4 require it)
        const ctx_kv = try ggml.Context.init(512 * 1024 * 1024);
        defer ctx_kv.deinit();
        const max_seq_len = @min(params.max_seq_len, 2048);
        const hdim_kv = params.n_head_dim;
        const hdim_k = @max(params.n_head_dim, params.n_head_dim_k);
        const hdim_v = if (params.n_head_dim_v > 0) @max(params.n_head_dim, params.n_head_dim_v) else hdim_kv;
        var kv_mgr = try kv_cache.KVCache.initWithKVDim(ctx_kv, params.n_layer, params.n_kv_head, hdim_k, hdim_v, max_seq_len, self.allocator);
        defer kv_mgr.deinit(self.allocator);
        model.setKVCacheContext(ctx_kv);

        const logits_tensor = try model.buildGraph(&builder, input_tensor, n_tokens, @ptrCast(&kv_mgr), 0);
        ctx.setNoAlloc(true);

        const buft = ggml.backendCpuBufferType();
        var galloc = try ggml.Gallocr.init(buft);
        defer galloc.free();
        if (!galloc.allocGraph(graph)) return error.GraphAllocFailed;

        const n_threads = @as(i32, @intCast(@min(4, @max(1, try std.Thread.getCpuCount() - 1))));
        try graph.compute(n_threads);

        // 6. 读取 logits
        const logits_data = logits_tensor.dataBytes();
        const n_vocab = @as(usize, @intCast(params.n_vocab));
        const our_logits = @as([*]f32, @ptrCast(@alignCast(logits_data.ptr)))[0..n_vocab];

        // 7. 加载参考 logits
        const ref_logits = try self.loadReferenceLogits(io);
        defer self.allocator.free(ref_logits);

        if (ref_logits.len != n_vocab) {
            log.err("Reference logits length mismatch: expected {d}, got {d}", .{ n_vocab, ref_logits.len });
            return error.SizeMismatch;
        }

        // 8. 计算指标
        const nmse = metrics.calcNMSE(our_logits, ref_logits);
        const cos_sim = metrics.calcCosineSimilarity(our_logits, ref_logits);
        const max_abs_err = metrics.calcMaxAbsError(our_logits, ref_logits);
        const argmax_match = metrics.calcArgmaxMatch(our_logits, ref_logits);

        const stdout_file = std.Io.File.stdout();
        try stdout_file.writeStreamingAll(io, "\n");
        try stdout_file.writeStreamingAll(io, "=== Logits Comparison ===\n");
        try metrics.printMetric(io, "NMSE", nmse, self.config.nmse_threshold, true);
        try metrics.printMetric(io, "Cosine Similarity", cos_sim, self.config.cosine_threshold, false);
        try metrics.printMetric(io, "Max Abs Error", max_abs_err, 0.01, true);
        try metrics.printArgmaxResult(io, argmax_match);

        const passed = nmse < self.config.nmse_threshold and cos_sim > self.config.cosine_threshold;
        if (passed) {
            try stdout_file.writeStreamingAll(io, "\n✅ PASS: Logits match reference within tolerance.\n");
        } else {
            try stdout_file.writeStreamingAll(io, "\n❌ FAIL: Logits deviate from reference.\n");
        }
        try stdout_file.writeStreamingAll(io, "============================================\n");

        return passed;
    }

    fn loadReferenceLogits(self: *LlamaCppComparator, io: std.Io) ![]f32 {
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
// Main
// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    _ = args_iter.next(); // skip argv[0]

    var config = CompareWithLlamaConfig{};

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            config.model_path = args_iter.next() orelse {
                log.err("--model requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            config.prompt = args_iter.next() orelse {
                log.err("--prompt requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--ref-logits")) {
            config.ref_logits_path = args_iter.next() orelse {
                log.err("--ref-logits requires a value", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "-n")) {
            const n_str = args_iter.next() orelse {
                log.err("-n requires a value", .{});
                std.process.exit(1);
            };
            config.n_tokens = std.fmt.parseInt(usize, n_str, 10) catch {
                log.err("Invalid value for -n: {s}", .{n_str});
                std.process.exit(1);
            };
        }
    }

    if (config.model_path.len == 0 or config.prompt.len == 0 or config.ref_logits_path.len == 0) {
        log.err("Usage: zllama-compare-llamacpp --model <path> --prompt <text> --ref-logits <file> [-n tokens]", .{});
        std.process.exit(1);
    }

    _ = config.n_tokens; // currently we always compare the first token's logits

    var comparator = LlamaCppComparator.init(allocator, config);
    const passed = comparator.run(io) catch |err| {
        log.err("Comparison failed: {}", .{err});
        std.process.exit(1);
    };

    if (!passed) std.process.exit(1);
}
