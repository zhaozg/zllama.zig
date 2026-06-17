//! Logits 比较工具
//!
//! 用于比较两组 logits 的差异，支持 NMSE、余弦相似度、PSNR 等指标。
//! 可用于验证模型推理结果的正确性。
//!
//! 用法：
//!   zllama-compare-logits --ref ref_logits.bin --test test_logits.bin

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const tokenizer = @import("tokenizer");

/// 比较配置
pub const CompareConfig = struct {
    nmse_threshold: f64 = 1e-4,
    max_abs_error_threshold: f32 = 0.01,
    cosine_threshold: f64 = 0.999,
    verbose: bool = false,
    max_diff_positions: usize = 10,
};

/// 比较结果
pub const ComparisonResult = struct {
    nmse: f64,
    max_abs_error: f32,
    mean_abs_error: f32,
    cosine_similarity: f64,
    psnr: f64,
    matched_tokens: usize,
    total_tokens: usize,
    match_rate: f64,
};

/// Logits 比较器
pub const LogitsComparator = struct {
    allocator: std.mem.Allocator,
    config: CompareConfig,

    pub fn init(allocator: std.mem.Allocator, config: CompareConfig) LogitsComparator {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// 比较两组 logits
    pub fn compare(_: *LogitsComparator, ref_logits: []const f32, test_logits: []const f32) !ComparisonResult {
        std.debug.assert(ref_logits.len == test_logits.len);

        const n = ref_logits.len;
        if (n == 0) {
            return ComparisonResult{
                .nmse = 0.0,
                .max_abs_error = 0.0,
                .mean_abs_error = 0.0,
                .cosine_similarity = 1.0,
                .psnr = 100.0,
                .matched_tokens = 0,
                .total_tokens = 0,
                .match_rate = 1.0,
            };
        }

        // 计算 NMSE 和最大绝对误差
        var sum_sq_ref: f64 = 0.0;
        var sum_sq_diff: f64 = 0.0;
        var max_abs_err: f32 = 0.0;
        var sum_abs_err: f64 = 0.0;

        // 计算余弦相似度
        var dot_product: f64 = 0.0;
        var sum_sq_test: f64 = 0.0;

        for (ref_logits, test_logits) |r, t| {
            const rf: f64 = @floatCast(r);
            const tf: f64 = @floatCast(t);
            const diff = r - t;
            const abs_diff = @abs(diff);

            sum_sq_ref += rf * rf;
            sum_sq_test += tf * tf;
            sum_sq_diff += @as(f64, @floatCast(diff)) * @as(f64, @floatCast(diff));
            sum_abs_err += @as(f64, @floatCast(abs_diff));

            if (abs_diff > max_abs_err) {
                max_abs_err = abs_diff;
            }

            dot_product += rf * tf;
        }

        const nmse = if (sum_sq_ref > 0.0) sum_sq_diff / sum_sq_ref else 0.0;
        const mean_abs_error: f32 = @floatCast(sum_abs_err / @as(f64, @floatFromInt(n)));

        // 余弦相似度: dot_product / (||ref|| * ||test||)
        const norm_ref = @sqrt(sum_sq_ref);
        const norm_test = @sqrt(sum_sq_test);
        const cosine_similarity = if (norm_ref > 0.0 and norm_test > 0.0)
            dot_product / (norm_ref * norm_test)
        else
            1.0;

        // PSNR
        const mse = sum_sq_diff / @as(f64, @floatFromInt(n));
        const psnr = if (mse > 0.0) 10.0 * @log10(sum_sq_ref / mse) else 100.0;

        // 计算 argmax 匹配率
        var ref_argmax: usize = 0;
        var ref_max: f32 = -std.math.inf(f32);
        for (ref_logits, 0..) |r, i| {
            if (r > ref_max) {
                ref_max = r;
                ref_argmax = i;
            }
        }

        var test_argmax: usize = 0;
        var test_max: f32 = -std.math.inf(f32);
        for (test_logits, 0..) |t, i| {
            if (t > test_max) {
                test_max = t;
                test_argmax = i;
            }
        }

        const matched: usize = if (ref_argmax == test_argmax) 1 else 0;
        const match_rate = @as(f64, @floatFromInt(matched)) / @as(f64, @floatFromInt(n));

        return ComparisonResult{
            .nmse = nmse,
            .max_abs_error = max_abs_err,
            .mean_abs_error = mean_abs_error,
            .cosine_similarity = cosine_similarity,
            .psnr = psnr,
            .matched_tokens = matched,
            .total_tokens = n,
            .match_rate = match_rate,
        };
    }

    /// 打印比较报告
    pub fn printReport(_: *LogitsComparator, result: ComparisonResult, writer: anytype) !void {
        const pass = result.nmse < 1e-4 and result.max_abs_error < 0.01 and result.cosine_similarity > 0.999;

        try writer.print("=== Logits Comparison Report ===\n", .{});
        try writer.print("Status: {s}\n", .{if (pass) "PASS" else "FAIL"});
        try writer.print("NMSE: {d:.10}\n", .{result.nmse});
        try writer.print("Max Abs Error: {d:.6}\n", .{result.max_abs_error});
        try writer.print("Mean Abs Error: {d:.6}\n", .{result.mean_abs_error});
        try writer.print("Cosine Similarity: {d:.10}\n", .{result.cosine_similarity});
        try writer.print("PSNR: {d:.2} dB\n", .{result.psnr});
        try writer.print("Argmax Match Rate: {d}/{d} ({d:.2}%)\n", .{ result.matched_tokens, result.total_tokens, result.match_rate * 100.0 });
        try writer.print("===============================\n", .{});
    }

    /// 从二进制文件加载 logits
    pub fn loadFromBinary(allocator: std.mem.Allocator, path: []const u8, io: std.Io) ![]f32 {
        const dir = std.Io.Dir.cwd();
        const file = try dir.openFile(io, path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const size = stat.size;
        std.debug.assert(size % @sizeOf(f32) == 0);

        const n = size / @sizeOf(f32);
        const buf = try allocator.alloc(f32, n);
        errdefer allocator.free(buf);

        const bytes = std.mem.sliceAsBytes(buf);
        const nread = try file.readPositionalAll(io, bytes, 0);
        if (nread != size) {
            return error.UnexpectedEndOfFile;
        }

        return buf;
    }
};

fn printReportSimple(result: ComparisonResult) void {
    const pass = result.nmse < 1e-4 and result.max_abs_error < 0.01 and result.cosine_similarity > 0.999;

    std.debug.print("=== Logits Comparison Report ===\n", .{});
    std.debug.print("Status: {s}\n", .{if (pass) "PASS" else "FAIL"});
    std.debug.print("NMSE: {d:.10}\n", .{result.nmse});
    std.debug.print("Max Abs Error: {d:.6}\n", .{result.max_abs_error});
    std.debug.print("Mean Abs Error: {d:.6}\n", .{result.mean_abs_error});
    std.debug.print("Cosine Similarity: {d:.10}\n", .{result.cosine_similarity});
    std.debug.print("PSNR: {d:.2} dB\n", .{result.psnr});
    std.debug.print("Argmax Match Rate: {d}/{d} ({d:.2}%)\n", .{ result.matched_tokens, result.total_tokens, result.match_rate * 100.0 });
    std.debug.print("===============================\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    // 简单参数解析
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    var ref_path: ?[]const u8 = null;
    var test_path: ?[]const u8 = null;

    // 跳过 argv[0]
    _ = args_iter.next();

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--ref")) {
            ref_path = args_iter.next() orelse {
                std.debug.print("Error: --ref requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--test")) {
            test_path = args_iter.next() orelse {
                std.debug.print("Error: --test requires a value\n", .{});
                std.process.exit(1);
            };
        }
    }

    if (ref_path == null or test_path == null) {
        std.debug.print("Usage: zllama-compare-logits --ref <ref_file> --test <test_file>\n", .{});
        std.process.exit(1);
    }

    // 加载 logits
    const ref_logits = try LogitsComparator.loadFromBinary(allocator, ref_path.?, io);
    defer allocator.free(ref_logits);

    const test_logits = try LogitsComparator.loadFromBinary(allocator, test_path.?, io);
    defer allocator.free(test_logits);

    // 比较
    var comp = LogitsComparator.init(allocator, .{});
    const result = try comp.compare(ref_logits, test_logits);
    printReportSimple(result);
}
