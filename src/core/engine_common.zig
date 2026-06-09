const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");

const log = std.log.scoped(.engine);

// ============================================================================
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
    if (@intFromEnum(level) > @intFromEnum(runtime_log_level)) return;
    std.log.defaultLog(level, scope, format, args);
}

// ============================================================================
// 时间测量
// ============================================================================

/// 获取当前时间（微秒）
pub fn currentTimeUs() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, ts.sec) * 1000000 + @as(i64, @divTrunc(ts.nsec, 1000));
}

/// 获取当前时间（毫秒）
pub fn currentTimeMs() i64 {
    return @divTrunc(currentTimeUs(), 1000);
}

// ============================================================================
// 文件读取
// ============================================================================

/// 读取整个文件到内存
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
        result.pp_time_s, pp_speed,
        result.tg_time_s, tg_speed,
        total_time_s, avg_speed,
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
