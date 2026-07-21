//! 向量对齐比较工具 — 算法对齐验证专用版（工业级严格）
//!
//! 用于比较两个推理引擎对同一数据处理结果的数值对齐程度。
//! 支持多指标综合验证：余弦相似度 + NMSE + RMSE + 相对 MaxErr + Argmax 匹配，
//! 自动检测线性缩放/偏置、离群点分布，并输出通过/警告/失败判决。
//!
//! 判决标准（工业级严格，适合 FP16/BF16 CUDA 内核验证）:
//!   - NMSE < 1e-4  （归一化均方误差）
//!   - 余弦相似度 > 0.9999
//!   - 相对 MaxErr < 1e-4（尺度自适应）
//!   - RMSE < 0.001  （每维度平均误差）
//!   - Argmax 必须匹配
//!   - 平均幅值比值偏离 < 0.001（检测系统性缩放）
//!
//! 用法:
//!   zllama-align-cmp --ref ref.json --test test.json
//!   zllama-align-cmp --ref ref.bin --test test.bin --key data.emb
//!   zllama-align-cmp --ref a.json --test b.json --mode alignment --tol-nmse 1e-4
//!
//! 输入格式:
//!   - JSON: 包含数值数组的对象，自动识别 "vector"/"embedding"/"vec"/"emb" 键，
//!           或通过 --key 指定点分隔路径 (如 "data.emb")
//!   - 二进制: 连续 f32 小端字节序，无元数据
//!   zllama-align-cmp --ref a.json --test b.json --mode alignment --tol-nmse 1e-4
//!
//! 输入格式:
//!   - JSON: 包含数值数组的对象，自动识别 "vector"/"embedding"/"vec"/"emb" 键，
//!           或通过 --key 指定点分隔路径 (如 "data.emb")
//!   - 二进制: 连续 f32 小端字节序，无元数据

const std = @import("std");
const metrics_mod = @import("metrics");
const config_mod = @import("align_cmp_config.zig");
const core_mod = @import("align_cmp_core.zig");

const AlignCmpConfig = config_mod.AlignCmpConfig;
const AlignMetrics = config_mod.AlignMetrics;
const CompareMode = config_mod.CompareMode;
const OutputFormat = config_mod.OutputFormat;
const alignmentVerdict = core_mod.alignmentVerdict;

const ArgmaxResult = config_mod.ArgmaxResult;
const computeMetrics = core_mod.computeMetrics;
const calcArgmaxMatch = core_mod.calcArgmaxMatch;
const log = std.log.scoped(.tool_align_cmp);

/// MaxErr 分级：基于相对最大绝对误差的精度等级（尺度自适应）
/// 使用 rel_max_err = max_abs_err / max(abs(ref), eps) 进行分级，
/// 确保评级在不同数据尺度下具有可比性。
const MaxErrGrade = enum {
    perfect,
    good,
    pass,
    fail,

    fn label(self: MaxErrGrade) []const u8 {
        return switch (self) {
            .perfect => "完美",
            .good => "良好",
            .pass => "需核查",
            .fail => "离群点超标",
        };
    }

    fn emoji(self: MaxErrGrade) []const u8 {
        return switch (self) {
            .perfect => "✅",
            .good => "☑️",
            .pass => "⚠️",
            .fail => "❌",
        };
    }
};

/// 根据相对最大绝对误差返回精度等级（尺度自适应）
/// rel_max_err = max_abs_err / max(abs(ref), eps)
fn gradeMaxErr(rel_max_err: f64) MaxErrGrade {
    if (rel_max_err < 1e-5) return .perfect;
    if (rel_max_err < 1e-4) return .good;
    if (rel_max_err < 1e-3) return .pass;
    return .fail;
}

// ============================================================================
// 比较器
// ============================================================================

pub const AlignComparator = struct {
    allocator: std.mem.Allocator,
    config: AlignCmpConfig,

    pub fn init(allocator: std.mem.Allocator, config: AlignCmpConfig) AlignComparator {
        return .{ .allocator = allocator, .config = config };
    }

    /// 运行比较：加载两个向量 → 计算指标 → 输出报告
    pub fn run(self: *AlignComparator, io: std.Io) !bool {
        // 1. 加载向量
        const v1 = try self.loadVector(io, self.config.ref_path);
        defer self.allocator.free(v1);

        const v2 = try self.loadVector(io, self.config.test_path);
        defer self.allocator.free(v2);

        // 2. 维度检查
        if (v1.len != v2.len) {
            log.err("维度不一致: {d} vs {d}", .{ v1.len, v2.len });
            return error.DimensionMismatch;
        }

        // 3. 计算指标
        const metrics = computeMetrics(v1, v2);
        const argmax = calcArgmaxMatch(v1, v2);

        // 4. 输出报告
        const stdout_file = std.Io.File.stdout();
        switch (self.config.output_format) {
            .human => {
                try stdout_file.writeStreamingAll(io, "============================================================\n");
                try self.printReportHuman(io, metrics, argmax);
                try stdout_file.writeStreamingAll(io, "============================================================\n");
            },
            .ai => {
                try self.printReportAI(io, metrics, argmax);
            },
        }

        // 5. 判决
        var pass: bool = true;
        if (self.config.mode == .alignment) {
            const verdict = alignmentVerdict(metrics, argmax, self.config);
            if (self.config.output_format == .ai) {
                const pass_code: u8 = if (std.mem.indexOf(u8, verdict, "❌") != null) @as(u8, 0) else @as(u8, 1);
                var num_buf: [4]u8 = undefined;
                const pass_str = try std.fmt.bufPrint(&num_buf, "{d}", .{pass_code});
                try stdout_file.writeStreamingAll(io, "PASS=");
                try stdout_file.writeStreamingAll(io, pass_str);
                try stdout_file.writeStreamingAll(io, "\n");
                const verdict_clean = if (std.mem.startsWith(u8, verdict, "✅ "))
                    verdict["✅ ".len..]
                else if (std.mem.startsWith(u8, verdict, "⚠️ "))
                    verdict["⚠️ ".len..]
                else if (std.mem.startsWith(u8, verdict, "❌ "))
                    verdict["❌ ".len..]
                else
                    verdict;
                try stdout_file.writeStreamingAll(io, "VERDICT=");
                try stdout_file.writeStreamingAll(io, verdict_clean);
                try stdout_file.writeStreamingAll(io, "\n");
            } else {
                try stdout_file.writeStreamingAll(io, verdict);
                try stdout_file.writeStreamingAll(io, "\n");
            }

            // 仅 ❌ 表示失败；⚠️ 是信息性警告
            if (std.mem.indexOf(u8, verdict, "❌") != null) {
                pass = false;
            }
        }

        return pass;
    }

    /// 从文件加载向量（自动检测 JSON 或二进制）
    fn loadVector(self: *AlignComparator, io: std.Io, path: []const u8) ![]f32 {
        const is_json = std.mem.endsWith(u8, path, ".json") or
            std.mem.endsWith(u8, path, ".JSON");

        if (is_json) {
            return self.loadVectorFromJson(io, path);
        } else {
            return metrics_mod.loadVectorFromBinary(self.allocator, io, path);
        }
    }

    /// 从 JSON 文件加载向量
    fn loadVectorFromJson(self: *AlignComparator, io: std.Io, path: []const u8) ![]f32 {
        const dir = std.Io.Dir.cwd();
        const file = try dir.openFile(io, path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const size = stat.size;

        const content = try self.allocator.alloc(u8, @intCast(size));
        defer self.allocator.free(content);
        const nread = try file.readPositionalAll(io, content, 0);
        if (nread != size) return error.UnexpectedEndOfFile;

        var parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            content,
            .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
        );
        defer parsed.deinit();

        const value = try self.resolveJsonValue(parsed.value);
        const arr = switch (value) {
            .array => |a| a,
            else => {
                log.err("指定路径的值不是数组", .{});
                return error.JsonTypeError;
            },
        };

        return extractF32Array(self.allocator, arr);
    }

    /// 从 JSON 根节点解析向量值（支持 --key 路径及自动识别）
    fn resolveJsonValue(self: *AlignComparator, root: std.json.Value) !std.json.Value {
        if (self.config.key) |key_path| {
            var parts = std.mem.splitSequence(u8, key_path, ".");
            var current = root;
            while (parts.next()) |part| {
                current = switch (current) {
                    .object => |obj| obj.get(part) orelse {
                        log.err("JSON 路径 '{s}' 中找不到键 '{s}'", .{ key_path, part });
                        return error.JsonKeyNotFound;
                    },
                    else => {
                        log.err("JSON 路径 '{s}' 中途遇到非对象节点", .{key_path});
                        return error.JsonPathError;
                    },
                };
            }
            return current;
        }

        // 自动识别：查找常见向量键名
        if (root != .object) {
            if (root == .array) return root;
            log.err("JSON 根节点不是对象也不是数组，请使用 --key 指定路径", .{});
            return error.JsonFormatError;
        }
        const obj = root.object;
        inline for (.{ "vector", "embedding", "vec", "emb", "data" }) |k| {
            if (obj.get(k)) |v| return v;
        }
        log.err("无法自动识别向量键，请使用 --key 指定路径", .{});
        return error.JsonKeyNotFound;
    }

    // 打印人类可读报告
    fn printReportHuman(self: *AlignComparator, io: std.Io, metrics: AlignMetrics, argmax: ArgmaxResult) !void {
        var buf: [4096]u8 = undefined;
        const stdout_file = std.Io.File.stdout();

        const header = try std.fmt.bufPrint(&buf,
            \\参考文件: {s}
            \\测试文件: {s}
            \\向量维度: {d}
            \\------------------------------------------------------------
            \\归一化均方误差 (NMSE) : {d:.8}
            \\余弦相似度 (Cos)      : {d:.8}
            \\欧氏距离 (L2)         : {d:.8}
            \\均方根误差 (RMSE)     : {d:.8}   (每维度平均误差)
            \\平均绝对误差 (MAE)    : {d:.8}
            \\最大绝对误差 (MaxErr) : {d:.8}
            \\相对最大误差 (Rel)    : {d:.8}  {s} {s}
            \\参考最大幅值          : {d:.4}
            \\(离群点) 数量/占比    : {d} / {d:.4}%
            \\平均幅值比值 (A/B)    : {d:.8}
            \\比值标准差 (Std)      : {d:.8}
            \\疑似线性缩放          : {s}
            \\Argmax 匹配           : {s} (参考索引 {d}, 测试索引 {d})
            \\------------------------------------------------------------
            \\
        , .{
            self.config.ref_path,
            self.config.test_path,
            metrics.dim,
            metrics.nmse,
            metrics.cosine,
            metrics.l2_distance,
            metrics.rmse,
            metrics.mae,
            metrics.max_abs_err,
            metrics.rel_max_err,
            gradeMaxErr(metrics.rel_max_err).emoji(),
            gradeMaxErr(metrics.rel_max_err).label(),
            metrics.ref_max_abs,
            metrics.outlier_count,
            metrics.outlier_ratio,
            metrics.avg_ratio,
            metrics.ratio_std,
            if (metrics.is_scaled) "是 ⚠️" else "否",
            if (argmax.match) "一致" else "不一致",
            argmax.ours,
            argmax.ref,
        });
        try stdout_file.writeStreamingAll(io, buf[0..header.len]);

        if (self.config.mode == .general) {
            const grade = if (metrics.cosine > 0.95)
                "极高相似 (注意检查是否仅存在缩放!)\n"
            else if (metrics.cosine > 0.8)
                "高度相似\n"
            else
                "一般/弱相似\n";
            try stdout_file.writeStreamingAll(io, grade);
        }
    }

    // 打印 AI/机器可读报告（紧凑 KEY=VALUE 格式，适合脚本解析）
    fn printReportAI(self: *AlignComparator, io: std.Io, metrics: AlignMetrics, argmax: ArgmaxResult) !void {
        var buf: [2048]u8 = undefined;
        const stdout_file = std.Io.File.stdout();

        const report = try std.fmt.bufPrint(&buf,
            "FILE_REF={s}\n" ++
                "FILE_TEST={s}\n" ++
                "DIM={d}\n" ++
                "L2={d:.8}\n" ++
                "RMSE={d:.8}\n" ++
                "MAE={d:.8}\n" ++
                "NMSE={d:.8}\n" ++
                "COSINE={d:.8}\n" ++
                "MAX_ABS_ERR={d:.8}\n" ++
                "REL_MAX_ERR={d:.8}\n" ++
                "REF_MAX_ABS={d:.4}\n" ++
                "OUTLIER_COUNT={d}\n" ++
                "OUTLIER_RATIO={d:.8}\n" ++
                "AVG_RATIO={d:.8}\n" ++
                "RATIO_STD={d:.8}\n" ++
                "IS_SCALED={d}\n" ++
                "ARGMAX_MATCH={d}\n" ++
                "ARGMAX_REF={d}\n" ++
                "ARGMAX_TEST={d}\n",
            .{
                self.config.ref_path,
                self.config.test_path,
                metrics.dim,
                metrics.l2_distance,
                metrics.rmse,
                metrics.mae,
                metrics.nmse,
                metrics.cosine,
                metrics.max_abs_err,
                metrics.rel_max_err,
                metrics.ref_max_abs,
                metrics.outlier_count,
                metrics.outlier_ratio,
                metrics.avg_ratio,
                metrics.ratio_std,
                @intFromBool(metrics.is_scaled),
                @intFromBool(argmax.match),
                argmax.ours,
                argmax.ref,
            },
        );
        try stdout_file.writeStreamingAll(io, buf[0..report.len]);
    }

};

/// 从 JSON 数组提取 f32 切片
fn extractF32Array(allocator: std.mem.Allocator, arr: std.json.Array) ![]f32 {
    const n = arr.items.len;
    const result = try allocator.alloc(f32, n);
    errdefer allocator.free(result);

    for (arr.items, 0..) |item, i| {
        result[i] = switch (item) {
            .integer => |num| @as(f32, @floatFromInt(num)),
            .float => |num| @as(f32, @floatCast(num)),
            .number_string => |s| std.fmt.parseFloat(f32, s) catch {
                log.err("数组第 {d} 个元素 number_string 解析失败: {s}", .{ i, s });
                return error.JsonParseError;
            },
            else => {
                log.err("数组第 {d} 个元素不是数值类型", .{i});
                return error.JsonTypeError;
            },
        };
    }
    return result;
}

// ============================================================================
// CLI 参数解析
// ============================================================================

const CliFlag = struct {
    name: []const u8,
    kind: enum { string, f64, bool_flag },
};

const cli_flags = [_]CliFlag{
    .{ .name = "--ref", .kind = .string },
    .{ .name = "--test", .kind = .string },
    .{ .name = "--key", .kind = .string },
    .{ .name = "--mode", .kind = .string },
    .{ .name = "--output", .kind = .string },
    .{ .name = "--tol-nmse", .kind = .f64 },
    .{ .name = "--tol-cosine", .kind = .f64 },
    .{ .name = "--tol-rmse", .kind = .f64 },
    .{ .name = "--tol-max-abs-err", .kind = .f64 },
    .{ .name = "--tol-rel-max-err", .kind = .f64 },
    .{ .name = "--tol-outlier-ratio", .kind = .f64 },
    .{ .name = "--tol-ratio-dev", .kind = .f64 },
};

fn parseArgs(allocator: std.mem.Allocator, args_iter: *std.process.Args.Iterator) !AlignCmpConfig {
    var config = AlignCmpConfig{};

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            return error.HelpRequested;
        }

        const flag = for (cli_flags) |f| {
            if (std.mem.eql(u8, arg, f.name)) break f;
        } else {
            log.err("未知参数: {s}", .{arg});
            return error.InvalidArgument;
        };

        const val = args_iter.next() orelse {
            log.err("{s} 需要参数值", .{arg});
            return error.MissingArgument;
        };

        if (std.mem.eql(u8, flag.name, "--ref")) {
            config.ref_path = allocator.dupe(u8, val) catch return error.OutOfMemory;
        } else if (std.mem.eql(u8, flag.name, "--test")) {
            config.test_path = allocator.dupe(u8, val) catch return error.OutOfMemory;
        } else if (std.mem.eql(u8, flag.name, "--key")) {
            config.key = allocator.dupe(u8, val) catch return error.OutOfMemory;
        } else if (std.mem.eql(u8, flag.name, "--mode")) {
            config.mode = std.meta.stringToEnum(CompareMode, val) orelse {
                log.err("无效的 mode: {s} (可选: general, alignment)", .{val});
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, flag.name, "--output")) {
            config.output_format = std.meta.stringToEnum(OutputFormat, val) orelse {
                log.err("无效的 output: {s} (可选: human, ai)", .{val});
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, flag.name, "--tol-nmse")) {
            config.tol_nmse = try parseF64(val, "tol-nmse");
        } else if (std.mem.eql(u8, flag.name, "--tol-cosine")) {
            config.tol_cosine = try parseF64(val, "tol-cosine");
        } else if (std.mem.eql(u8, flag.name, "--tol-rmse")) {
            config.tol_rmse = try parseF64(val, "tol-rmse");
        } else if (std.mem.eql(u8, flag.name, "--tol-max-abs-err")) {
            config.tol_max_abs_err = try parseF64(val, "tol-max-abs-err");
        } else if (std.mem.eql(u8, flag.name, "--tol-rel-max-err")) {
            config.tol_rel_max_err = try parseF64(val, "tol-rel-max-err");
        } else if (std.mem.eql(u8, flag.name, "--tol-outlier-ratio")) {
            config.tol_outlier_ratio = try parseF64(val, "tol-outlier-ratio");
        } else if (std.mem.eql(u8, flag.name, "--tol-ratio-dev")) {
            config.tol_ratio_deviation = try parseF64(val, "tol-ratio-dev");
        }
    }

    return config;
}

fn parseF64(val: []const u8, name: []const u8) !f64 {
    return std.fmt.parseFloat(f64, val) catch {
        log.err("无效的 {s} 值: {s}", .{ name, val });
        return error.InvalidArgument;
    };
}

fn deinitConfig(allocator: std.mem.Allocator, config: *AlignCmpConfig) void {
    if (config.ref_path.len > 0) allocator.free(config.ref_path);
    if (config.test_path.len > 0) allocator.free(config.test_path);
    if (config.key) |k| allocator.free(k);
}

// ============================================================================
// Help
// ============================================================================

fn printUsage(io: std.Io) !void {
    const stdout_file = std.Io.File.stdout();
    try stdout_file.writeStreamingAll(io, @embedFile("align_cmp_help.txt"));
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();
    _ = args_iter.next(); // 跳过 argv[0]

    var config = parseArgs(allocator, &args_iter) catch |err| {
        if (err == error.HelpRequested) {
            try printUsage(io);
            std.process.exit(0);
        }
        try printUsage(io);
        std.process.exit(1);
    };
    defer deinitConfig(allocator, &config);

    if (config.ref_path.len == 0 or config.test_path.len == 0) {
        try printUsage(io);
        std.process.exit(1);
    }

    var comparator = AlignComparator.init(allocator, config);
    const passed = comparator.run(io) catch |err| {
        log.err("比较失败: {}", .{err});
        std.process.exit(1);
    };

    if (!passed) std.process.exit(1);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "extractF32Array integers" {
    const json_str =
        \\[1, 2, 3, 4, 5]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json_str, .{});
    defer parsed.deinit();
    const arr = parsed.value.array;
    const result = try extractF32Array(testing.allocator, arr);
    defer testing.allocator.free(result);
    try testing.expectEqual(@as(usize, 5), result.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), result[4], 1e-6);
}

test "extractF32Array floats" {
    const json_str =
        \\[1.5, 2.5, 3.5]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json_str, .{});
    defer parsed.deinit();
    const arr = parsed.value.array;
    const result = try extractF32Array(testing.allocator, arr);
    defer testing.allocator.free(result);
    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectApproxEqAbs(@as(f32, 1.5), result[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3.5), result[2], 1e-6);
}

test "loadVectorFromBinary roundtrip" {
    const allocator = testing.allocator;
    const tmp_path = "/tmp/zllama_test_align_cmp.bin";
    {
        const dir = std.Io.Dir.cwd();
        const file = try dir.createFile(testing.allocator, tmp_path, .{ .read = true });
        defer file.close(testing.allocator);
        const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const bytes = std.mem.sliceAsBytes(&data);
        _ = try file.writeAll(testing.allocator, bytes);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    const io = std.Io.null();
    const result = try metrics_mod.loadVectorFromBinary(allocator, io, tmp_path);
    defer allocator.free(result);
    try testing.expectEqual(@as(usize, 4), result.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result[3], 1e-6);
}
