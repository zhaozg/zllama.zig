const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");

const log = std.log.scoped(.core_engine_common);

// 运行时日志级别控制
// ============================================================================

var runtime_log_level: std.log.Level = .warn;

pub fn setLogLevel(level: std.log.Level) void {
    runtime_log_level = level;
}

pub fn getLogLevel() std.log.Level {
    return runtime_log_level;
}

pub fn logFilter(comptime level: std.log.Level, comptime scope: @TypeOf(.EnumLiteral), comptime format: []const u8, args: anytype) void {
    // 检查 log_scope_levels 中是否有针对当前 scope 的特定级别设置
    // 如果有，使用 scope 级别（编译期静态定义，覆盖运行时级别）；
    // 否则使用 runtime_log_level（运行时动态控制）
    const scope_level = comptime getScopeLevel(scope);
    if (scope_level) |sl| {
        if (@intFromEnum(level) > @intFromEnum(sl)) return;
    } else {
        if (@intFromEnum(level) > @intFromEnum(runtime_log_level)) return;
    }
    std.log.defaultLog(level, scope, format, args);
}

/// 编译期获取 scope 在 log_scope_levels 中定义的级别
/// 如果未找到，返回 null
fn getScopeLevel(comptime scope: @TypeOf(.EnumLiteral)) ?std.log.Level {
    if (@hasDecl(@import("root"), "std_options")) {
        const opts = @import("root").std_options;
        if (@hasField(@TypeOf(opts), "log_scope_levels")) {
            inline for (opts.log_scope_levels) |sl| {
                if (sl.scope == scope) {
                    return sl.level;
                }
            }
        }
    }
    return null;
}

// ============================================================================
// 时间测量（Zig 0.16 风格：使用 std.Io.Clock）
// ============================================================================

/// 高性能计时器（替代已移除的 std.time.Timer）
/// 使用 std.Io.Clock.now(.awake, io) 获取单调递增时间戳。
pub const WallTimer = struct {
    start_ts: std.Io.Timestamp,
    io: std.Io,

    pub fn start(io: std.Io) !WallTimer {
        return .{
            .start_ts = try std.Io.Clock.now(.awake, io),
            .io = io,
        };
    }

    /// 读取经过的时间（纳秒）
    pub fn read(self: WallTimer) !u64 {
        const now = try std.Io.Clock.now(.awake, self.io);
        const dur = now.since(self.start_ts);
        return dur.nanoseconds;
    }

    /// 读取经过的时间（微秒）
    pub fn readUs(self: WallTimer) !i64 {
        const ns = try self.read();
        return @as(i64, @intCast(ns)) / 1000;
    }

    /// 读取经过的时间（毫秒）
    pub fn readMs(self: WallTimer) !i64 {
        const ns = try self.read();
        return @as(i64, @intCast(ns)) / 1_000_000;
    }
};

/// 获取当前时间（微秒）— 兼容旧 API，但内部使用 std.Io.Clock
/// 注意：此函数需要 io 参数，但为了向后兼容保留无参版本。
/// 新代码应直接使用 WallTimer。
pub fn currentTimeUs() i64 {
    // 回退方案：使用 POSIX clock_gettime（不依赖 io）
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, ts.sec) * 1000000 + @as(i64, @divTrunc(ts.nsec, 1000));
}

/// 获取当前时间（毫秒）— 兼容旧 API
pub fn currentTimeMs() i64 {
    return @divTrunc(currentTimeUs(), 1000);
}

// ============================================================================
// 文件读取（mmap 优先，回退到 read）
// ============================================================================

/// 内存映射文件（mmap）— 零拷贝加载，启动速度提升 2-3 倍。
/// 返回映射的内存切片，调用者负责在不再需要时调用 unmapFile。
/// 对于大文件（>100MB），mmap 显著优于 readFileToMemory。
pub fn mmapFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !MappedFile {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    errdefer file.close(io);

    const stat = try file.stat(io);
    const file_size = @as(usize, @intCast(stat.size));

    // 使用 std.Io.File.createMemoryMap 创建内存映射（Zig 0.16 原生支持）
    const mmap = file.createMemoryMap(io, .{
        .len = file_size,
        .protection = .{ .read = true, .write = false },
        .undefined_contents = false,
        .populate = true,
    }) catch |err| {
        // mmap 失败，回退到 read
        log.warn("mmap failed ({}), falling back to readFileToMemory", .{err});
        const data = try readFileToMemory(io, allocator, path);
        file.close(io);
        return MappedFile{
            .data = data,
            .file = null,
            .mmap = null,
            .is_mmap = false,
            .allocator = allocator,
        };
    };

    return MappedFile{
        .data = mmap.memory,
        .file = file,
        .mmap = mmap,
        .is_mmap = true,
        .allocator = allocator,
    };
}

/// 映射文件结果
pub const MappedFile = struct {
    data: []u8,
    file: ?std.Io.File,
    mmap: ?std.Io.File.MemoryMap,
    is_mmap: bool,
    allocator: std.mem.Allocator,

    /// 释放映射或分配的内存
    pub fn deinit(self: *MappedFile, io: std.Io) void {
        if (self.is_mmap) {
            if (self.mmap) |*m| {
                m.destroy(io);
            }
            if (self.file) |f| f.close(io);
        } else {
            self.allocator.free(self.data);
        }
        self.* = undefined;
    }
};

/// 读取整个文件到内存（传统方式，用于小文件或 mmap 不可用时的回退）
pub fn readFileToMemory(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const file_size = @as(usize, @intCast(stat.size));
    const data = try allocator.alloc(u8, file_size);
    errdefer allocator.free(data);

    var offset: u64 = 0;
    const chunk_size: usize = 64 * 1024 * 1024; // 64MB chunks
    while (offset < file_size) {
        const end = @min(offset + chunk_size, file_size);
        const len = end - offset;
        const bytes_read = try file.readPositionalAll(io, data[offset..][0..len], offset);
        if (bytes_read != len) {
            allocator.free(data);
            return error.FileReadError;
        }
        offset += bytes_read;
    }
    return data;
}

// ============================================================================
// 基准测试输出
// ============================================================================

/// 基准测试结果
pub const BenchmarkResult = struct {
    model_name: []const u8,
    arch_name: []const u8,
    n_threads: i32,
    n_prompt_tokens: i32,
    n_decode: i32,
    pp_time_s: f64,
    tg_time_s: f64,
};

/// 打印基准测试结果
pub fn printBenchmark(result: BenchmarkResult) void {
    const total_time_s = result.pp_time_s + result.tg_time_s;
    const pp_speed = if (result.pp_time_s > 0.0)
        @as(f64, @floatFromInt(result.n_prompt_tokens)) / result.pp_time_s
    else
        0.0;
    const tg_speed = if (result.tg_time_s > 0.0 and result.n_decode > 0)
        @as(f64, @floatFromInt(result.n_decode)) / result.tg_time_s
    else
        0.0;
    const avg_speed = if (total_time_s > 0.0 and result.n_decode > 0)
        @as(f64, @floatFromInt(result.n_decode)) / total_time_s
    else
        0.0;

    std.debug.print(
        \\============ Benchmark Results ============
        \\  Model            : {s}
        \\  Architecture     : {s}
        \\  Threads          : {d}
        \\  Prompt tokens    : {d}
        \\  Output tokens    : {d}
        \\  ------------------------------------------
        \\  PP eval time     : {d:.3} s ({d:.1} tok/s)
        \\  TG time          : {d:.3} s ({d:.1} tok/s)
        \\  Total time       : {d:.3} s ({d:.1} tok/s)
        \\=============================================
        \\
    , .{
        result.model_name,
        result.arch_name,
        result.n_threads,
        result.n_prompt_tokens,
        result.n_decode,
        result.pp_time_s,
        pp_speed,
        result.tg_time_s,
        tg_speed,
        total_time_s,
        avg_speed,
    });
}

/// 打印性能摘要
pub fn printSummary(n_decode: i32, total_time_s: f64) void {
    if (n_decode > 0) {
        const avg_speed = @as(f64, @floatFromInt(n_decode)) / total_time_s;
        std.debug.print("decoded {d} tokens in {d:.2} s, speed: {d:.2} t/s\n", .{ n_decode, total_time_s, avg_speed });
    }
}

// ============================================================================
// 解码并打印 token
// ============================================================================

/// 解码单个 token 并写入 stdout
pub fn decodeAndPrintToken(io: std.Io, tok: anytype, token_id: u32) !void {
    var buf: [128]u8 = undefined;
    const n = try tok.decodeSingle(token_id, &buf);
    if (n > 0) {
        const stdout_file = std.Io.File.stdout();
        try stdout_file.writeStreamingAll(io, buf[0..n]);
    }
}

const testing = std.testing;

// ============================================================================
// 共享计算图执行助手
// ============================================================================

/// 计算图执行结果
pub const ComputeResult = struct {
    /// 是否需要重置 gallocr（例如因图结构变化）
    needs_reset: bool,
};

/// Execute a ggml compute graph on CPU.
///
/// 所有权上移：gallocr 由调用者创建和管理，不再在此函数内泄漏。
/// 调用者负责在 InferenceEngine 生命周期内管理 gallocr。
///
/// 调用者必须在 computeGraph 返回前完成所有必要的张量数据拷贝，
/// 因为 gallocr 可能在后续 reset 中释放内存。
pub fn computeGraph(graph: *ggml.CGraph, n_threads: i32, gallocr: *ggml.Gallocr) !ComputeResult {
    if (!gallocr.allocGraph(graph)) return error.GraphAllocFailed;

    const cpu = try ggml.backendCpuInit();
    defer ggml.backendFree(cpu);
    ggml.backendCpuSetNThreads(cpu, n_threads);
    if (!ggml.backendGraphCompute(cpu, graph)) return error.ComputeFailed;

    return ComputeResult{ .needs_reset = false };
}

/// Execute a ggml compute graph on a specific backend.
pub fn computeGraphOnBackend(graph: *ggml.CGraph, backend: *ggml.Backend, gallocr: *ggml.Gallocr) !ComputeResult {
    if (!gallocr.allocGraph(graph)) return error.GraphAllocFailed;
    if (!ggml.backendGraphCompute(backend, graph)) return error.ComputeFailed;
    return ComputeResult{ .needs_reset = false };
}

test "currentTimeUs monotonic" {
    const t1 = currentTimeUs();
    const t2 = currentTimeUs();
    try testing.expect(t2 >= t1);
}

test "currentTimeMs monotonic" {
    const t1 = currentTimeMs();
    const t2 = currentTimeMs();
    try testing.expect(t2 >= t1);
}
