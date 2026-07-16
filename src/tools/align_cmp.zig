//! 向量对齐比较工具 — 算法对齐验证专用版（增强版）
//!
//! 用于比较两个推理引擎对同一数据处理结果的数值对齐程度。
//! 支持多指标综合验证：余弦相似度 + 均方根误差 (RMSE) + 平均绝对误差 (MAE)，
//! 自动检测线性缩放/偏置，并输出 Argmax 匹配结果。
//!
//! 用法:
//!   zllama-align-cmp --ref ref.json --test test.json
//!   zllama-align-cmp --ref ref.bin --test test.bin --key data.emb
//!   zllama-align-cmp --ref a.json --test b.json --mode alignment --tol-cosine 0.9995 --tol-rmse 0.005
//!
//! 输入格式:
//!   - JSON: 包含数值数组的对象，自动识别 "vector"/"embedding"/"vec"/"emb" 键，
//!           或通过 --key 指定点分隔路径 (如 "data.emb")
//!   - 二进制: 连续 f32 小端字节序，无元数据

const std = @import("std");
const metrics_mod = @import("metrics");

const log = std.log.scoped(.tool_align_cmp);

// ============================================================================
// 配置
// ============================================================================

/// 比较模式
pub const CompareMode = enum {
    /// 普通模式：输出相似度等级
    general,
    /// 严格对齐模式：输出通过/失败判决
    alignment,
};

/// 对齐比较配置
pub const AlignCmpConfig = struct {
    /// 参考文件路径
    ref_path: []const u8 = "",
    /// 测试文件路径
    test_path: []const u8 = "",
    /// JSON 向量路径（点分隔，如 "data.emb"）
    key: ?[]const u8 = null,
    /// 比较模式
    mode: CompareMode = .alignment,
    /// 对齐模式下的余弦相似度最低要求（针对高维嵌入建议 0.9995）
    tol_cosine: f64 = 0.9995,
    /// 对齐模式下的均方根误差 (RMSE) 最大容忍度（每维度平均误差）
    tol_rmse: f64 = 0.005,
    /// 对齐模式下的平均比值允许偏离 1.0 的容忍度
    tol_ratio_deviation: f64 = 0.05,
};

// ============================================================================
// 指标结果
// ============================================================================

/// 综合对齐指标
pub const AlignMetrics = struct {
    /// 余弦相似度
    cosine: f64,
    /// 欧几里得距离 (L2) — 仅用于输出参考，不作为判决依据
    l2_distance: f64,
    /// 均方根误差 (RMSE) = L2 / sqrt(dim)
    rmse: f64,
    /// 平均绝对误差 (MAE)
    mae: f64,
    /// 平均幅值比值 (A/B)
    avg_ratio: f64,
    /// 幅值比值的标准差（用于检测是否为严格线性缩放）
    ratio_std: f64,
    /// 是否疑似存在线性缩放（基于比值变异系数）
    is_scaled: bool,
    /// 向量维度
    dim: usize,
};

/// Argmax 匹配结果
/// Argmax 匹配结果（重新导出共享类型以保证 API 兼容）
pub const ArgmaxResult = metrics_mod.ArgmaxResult;
// ============================================================================

pub const AlignComparator = struct {
    allocator: std.mem.Allocator,
    config: AlignCmpConfig,

    pub fn init(allocator: std.mem.Allocator, config: AlignCmpConfig) AlignComparator {
        return .{ .allocator = allocator, .config = config };
    }

    /// 运行比较：加载两个向量 → 计算指标 → 输出报告
    pub fn run(self: *AlignComparator, io: std.Io) !bool {
        const stdout_file = std.Io.File.stdout();
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

        try stdout_file.writeStreamingAll(io, "============================================================\n");
        // 4. 输出报告
        try self.printReport(io, metrics, argmax);

        // 5. 判决
        var pass: bool = true;
        if (self.config.mode == .alignment) {
            const verdict = alignmentVerdict(metrics, self.config);
            try stdout_file.writeStreamingAll(io, verdict);
            try stdout_file.writeStreamingAll(io, "\n");

            // 对齐失败时返回 false
            if (std.mem.indexOf(u8, verdict, "❌") != null or
                std.mem.indexOf(u8, verdict, "⚠️") != null)
            {
                pass = false;
            }
        }
        try stdout_file.writeStreamingAll(io, "============================================================\n");

        return pass;
    }

    /// 从文件加载向量（自动检测 JSON 或二进制）
    fn loadVector(self: *AlignComparator, io: std.Io, path: []const u8) ![]f32 {
        const dir = std.Io.Dir.cwd();
        const file = try dir.openFile(io, path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        const size = stat.size;

        // 根据扩展名判断格式
        const is_json = std.mem.endsWith(u8, path, ".json") or
            std.mem.endsWith(u8, path, ".JSON");

        if (is_json) {
            return self.loadVectorFromJson(io, file, size);
        } else {
            return loadVectorFromBinary(self.allocator, io, file, size);
        }
    }

    /// 从 JSON 文件加载向量
    fn loadVectorFromJson(self: *AlignComparator, io: std.Io, file: std.Io.File, size: u64) ![]f32 {
        // 读取整个文件
        const content = try self.allocator.alloc(u8, @as(usize, @intCast(size)));
        defer self.allocator.free(content);

        const nread = try file.readPositionalAll(io, content, 0);
        if (nread != size) return error.UnexpectedEndOfFile;

        // 解析 JSON 为动态 Value
        var parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            content,
            .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
        );
        defer parsed.deinit();

        const root = parsed.value;

        // 定位向量值
        const value = if (self.config.key) |key_path| blk: {
            // 点分隔路径导航
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
            break :blk current;
        } else blk: {
            // 自动识别：查找常见向量键名
            if (root != .object) {
                // 如果根节点本身就是数组，直接使用
                if (root == .array) break :blk root;
                log.err("JSON 根节点不是对象也不是数组，请使用 --key 指定路径", .{});
                return error.JsonFormatError;
            }
            const obj = root.object;
            inline for (.{ "vector", "embedding", "vec", "emb", "data" }) |k| {
                if (obj.get(k)) |v| {
                    break :blk v;
                }
            }
            log.err("无法自动识别向量键，请使用 --key 指定路径", .{});
            return error.JsonKeyNotFound;
        };

        // 提取数值数组
        const arr = switch (value) {
            .array => |a| a,
            else => {
                log.err("指定路径的值不是数组", .{});
                return error.JsonTypeError;
            },
        };

        const n = arr.items.len;
        const result = try self.allocator.alloc(f32, n);
        errdefer self.allocator.free(result);

        for (arr.items, 0..) |item, i| {
            result[i] = switch (item) {
                .integer => |num| @as(f32, @floatFromInt(num)),
                .float => |num| @as(f32, @floatCast(num)),
                .number_string => |s| blk: {
                    break :blk std.fmt.parseFloat(f32, s) catch {
                        log.err("数组第 {d} 个元素 number_string 解析失败: {s}", .{ i, s });
                        return error.JsonParseError;
                    };
                },
                else => {
                    log.err("数组第 {d} 个元素不是数值类型", .{i});
                    return error.JsonTypeError;
                },
            };
        }

        return result;
    }

    /// 打印详细报告
    fn printReport(self: *AlignComparator, io: std.Io, metrics: AlignMetrics, argmax: ArgmaxResult) !void {
        const stdout_file = std.Io.File.stdout();
        var buf: [8192]u8 = undefined;

        const header = try std.fmt.bufPrint(&buf,
            \\参考文件: {s}
            \\测试文件: {s}
            \\向量维度: {d}
            \\------------------------------------------------------------
            \\余弦相似度 (Cos)    : {d:.10}
            \\欧氏距离 (L2)       : {d:.8}   (仅供参考，不用于判决)
            \\均方根误差 (RMSE)   : {d:.8}   (每维度平均误差)
            \\平均绝对误差 (MAE)  : {d:.8}
            \\平均幅值比值 (A/B)  : {d:.6}
            \\比值标准差 (Std)    : {d:.6}
            \\疑似线性缩放        : {s}
            \\Argmax 匹配         : {s} (参考索引 {d}, 测试索引 {d})
            \\------------------------------------------------------------
            \\
        , .{
            self.config.ref_path,
            self.config.test_path,
            metrics.dim,
            metrics.cosine,
            metrics.l2_distance,
            metrics.rmse,
            metrics.mae,
            metrics.avg_ratio,
            metrics.ratio_std,
            if (metrics.is_scaled) "是 ⚠️" else "否",
            if (argmax.match) "一致" else "不一致",
            argmax.ref,
            argmax.ours,
        });
        try stdout_file.writeStreamingAll(io, buf[0..header.len]);

        if (self.config.mode == .general) {
            const grade = if (metrics.cosine > 0.95)
                "极高相似 (注意检查是否仅存在缩放!)"
            else if (metrics.cosine > 0.8)
                "高度相似"
            else
                "一般/弱相似";
            const grade_line = try std.fmt.bufPrint(&buf, "相似度等级: {s} (注意: 余弦高不代表数值对齐)\n", .{grade});
            try stdout_file.writeStreamingAll(io, grade_line);
        }
    }
};

// ============================================================================
// 指标计算
// ============================================================================

/// 计算综合对齐指标
pub fn computeMetrics(v1: []const f32, v2: []const f32) AlignMetrics {
    std.debug.assert(v1.len == v2.len);

    const n = v1.len;
    if (n == 0) {
        return AlignMetrics{
            .cosine = 1.0,
            .l2_distance = 0.0,
            .rmse = 0.0,
            .mae = 0.0,
            .avg_ratio = 1.0,
            .ratio_std = 0.0,
            .is_scaled = false,
            .dim = 0,
        };
    }

    // 1. 余弦相似度 & 范数
    var dot: f64 = 0.0;
    var norm1: f64 = 0.0;
    var norm2: f64 = 0.0;

    // 2. 欧几里得距离 (L2) 与 RMSE
    var sum_sq_diff: f64 = 0.0;

    // 3. 平均绝对误差 (MAE)
    var sum_abs_err: f64 = 0.0;

    // 4. 比值统计（用于缩放检测）
    var ratio_sum: f64 = 0.0;
    var ratio_sq_sum: f64 = 0.0;
    var valid_ratio_count: usize = 0;

    for (v1, v2) |a, b| {
        const af: f64 = @floatCast(a);
        const bf: f64 = @floatCast(b);

        // 余弦
        dot += af * bf;
        norm1 += af * af;
        norm2 += bf * bf;

        // L2
        const diff = af - bf;
        sum_sq_diff += diff * diff;

        // MAE
        sum_abs_err += @abs(diff);

        // 比值（排除零值）
        if (@abs(bf) > 1e-20 and @abs(af) > 1e-20) {
            const ratio = af / bf;
            ratio_sum += ratio;
            ratio_sq_sum += ratio * ratio;
            valid_ratio_count += 1;
        }
    }

    const cosine = if (norm1 > 0.0 and norm2 > 0.0)
        dot / (@sqrt(norm1) * @sqrt(norm2))
    else
        1.0;

    const l2_distance = @sqrt(sum_sq_diff);
    const rmse = @sqrt(sum_sq_diff / @as(f64, @floatFromInt(n)));
    const mae = sum_abs_err / @as(f64, @floatFromInt(n));

    // 平均比值
    const avg_ratio = if (valid_ratio_count > 0)
        ratio_sum / @as(f64, @floatFromInt(valid_ratio_count))
    else
        1.0;

    // 比值标准差（无偏估计）
    const ratio_std = if (valid_ratio_count > 2) blk: {
        const count_f = @as(f64, @floatFromInt(valid_ratio_count));
        const variance = (ratio_sq_sum - (ratio_sum * ratio_sum) / count_f) / (count_f - 1.0);
        break :blk @sqrt(variance);
    } else 0.0;

    // 缩放检测：基于变异系数 (CV = std/mean) 小于 5% 视为严格线性缩放
    const ratio_deviation = @abs(avg_ratio - 1.0);
    const is_scaled = (ratio_std / avg_ratio) < 0.05 and ratio_deviation > 0.01;

    return AlignMetrics{
        .cosine = cosine,
        .l2_distance = l2_distance,
        .rmse = rmse,
        .mae = mae,
        .avg_ratio = avg_ratio,
        .ratio_std = ratio_std,
        .is_scaled = is_scaled,
        .dim = n,
    };
}

/// 计算 Argmax 匹配
/// 委托给共享 metrics 模块
pub fn calcArgmaxMatch(a: []const f32, b: []const f32) ArgmaxResult {
    return metrics_mod.calcArgmaxMatch(a, b);
}

// ============================================================================
// 判决
// ============================================================================
/// 输出算法对齐的判决结果
pub fn alignmentVerdict(al_metrics: AlignMetrics, config: AlignCmpConfig) []const u8 {
    if (al_metrics.cosine < config.tol_cosine) {
        return "❌ 对齐失败: 余弦相似度不满足要求";
    }
    if (al_metrics.rmse > config.tol_rmse) {
        return "❌ 对齐失败: 均方根误差 (RMSE) 超出允许误差";
    }
    if (al_metrics.is_scaled) {
        return "⚠️ 疑似存在线性缩放，算法未完全对齐！";
    }
    const ratio_deviation = @abs(al_metrics.avg_ratio - 1.0);
    if (ratio_deviation > config.tol_ratio_deviation) {
        return "⚠️ 平均幅值比值偏离 1.0，算法未完全对齐！";
    }
    return "✅ 算法对齐验证通过！";
}

// ============================================================================
// 二进制文件加载
// ============================================================================

/// 从二进制文件加载 f32 向量
pub fn loadVectorFromBinary(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File, size: u64) ![]f32 {
    if (size % @sizeOf(f32) != 0) {
        log.err("二进制文件大小 ({d}) 不是 f32 大小 ({d}) 的整数倍", .{ size, @sizeOf(f32) });
        return error.InvalidBinaryFile;
    }

    const n = @as(usize, @intCast(size / @sizeOf(f32)));
    const buf = try allocator.alloc(f32, n);
    errdefer allocator.free(buf);

    const bytes = std.mem.sliceAsBytes(buf);
    const nread = try file.readPositionalAll(io, bytes, 0);
    if (nread != size) return error.UnexpectedEndOfFile;

    return buf;
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    // 跳过 argv[0]
    _ = args_iter.next();

    var config = AlignCmpConfig{};

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--ref")) {
            config.ref_path = args_iter.next() orelse {
                log.err("--ref 需要参数值", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--test")) {
            config.test_path = args_iter.next() orelse {
                log.err("--test 需要参数值", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--key")) {
            config.key = args_iter.next() orelse {
                log.err("--key 需要参数值", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--mode")) {
            const mode_str = args_iter.next() orelse {
                log.err("--mode 需要参数值 (general|alignment)", .{});
                std.process.exit(1);
            };
            config.mode = std.meta.stringToEnum(CompareMode, mode_str) orelse {
                log.err("无效的 mode: {s} (可选: general, alignment)", .{mode_str});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--tol-cosine")) {
            const val_str = args_iter.next() orelse {
                log.err("--tol-cosine 需要参数值", .{});
                std.process.exit(1);
            };
            config.tol_cosine = std.fmt.parseFloat(f64, val_str) catch {
                log.err("无效的 tol-cosine 值: {s}", .{val_str});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--tol-rmse")) {
            const val_str = args_iter.next() orelse {
                log.err("--tol-rmse 需要参数值", .{});
                std.process.exit(1);
            };
            config.tol_rmse = std.fmt.parseFloat(f64, val_str) catch {
                log.err("无效的 tol-rmse 值: {s}", .{val_str});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--tol-ratio-dev")) {
            const val_str = args_iter.next() orelse {
                log.err("--tol-ratio-dev 需要参数值", .{});
                std.process.exit(1);
            };
            config.tol_ratio_deviation = std.fmt.parseFloat(f64, val_str) catch {
                log.err("无效的 tol-ratio-dev 值: {s}", .{val_str});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printUsage(io);
            std.process.exit(0);
        }
    }

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

fn printUsage(io: std.Io) !void {
    const stdout_file = std.Io.File.stdout();
    try stdout_file.writeStreamingAll(io,
        \\用法: zllama-align-cmp --ref <文件> --test <文件> [选项]
        \\
        \\必需参数:
        \\  --ref <文件>     参考向量文件 (JSON 或二进制)
        \\  --test <文件>    测试向量文件 (JSON 或二进制)
        \\
        \\可选参数:
        \\  --key <路径>     JSON 向量路径，如 "data.emb" (点分隔)
        \\  --mode <模式>    比较模式: general | alignment (默认: alignment)
        \\  --tol-cosine <f> 余弦相似度最低要求 (默认: 0.9995)
        \\  --tol-rmse <f>   均方根误差最大容忍度 (默认: 0.005)
        \\  --tol-ratio-dev <f> 平均比值偏离 1.0 的容忍度 (默认: 0.05)
        \\  --help, -h       显示此帮助信息
        \\
        \\输入格式:
        \\  JSON: 包含数值数组的对象，自动识别 "vector"/"embedding"/"vec"/"emb" 键
        \\  二进制: 连续 f32 小端字节序，无元数据
        \\
        \\示例:
        \\  zllama-align-cmp --ref ref.json --test test.json
        \\  zllama-align-cmp --ref a.bin --test b.bin --mode general
        \\  zllama-align-cmp --ref a.json --test b.json --tol-cosine 0.9995 --tol-rmse 0.005
        \\
    );
}
