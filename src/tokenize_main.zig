//! zllama-tokenize 工具
//!
//! 对给定 prompt 进行分词，并输出 token ID 和对应的 token 字符串。
//! 与 llama.cpp 的 llama-tokenize 工具功能对齐。

const std = @import("std");
const gguf = @import("gguf.zig");
const tokenizer = @import("tokenizer.zig");

const logger = std.log.scoped(.tokenize);

// 运行时日志级别控制
var runtime_log_level: std.log.Level = .info;

pub const std_options: std.Options = .{
    .log_level = .debug,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .tokenizer, .level = .debug },
        .{ .scope = .tokenize, .level = .debug },
    },
    .logFn = log,
};

pub fn log(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@intFromEnum(level) > @intFromEnum(runtime_log_level)) return;
    std.log.defaultLog(level, scope, format, args);
}

// ============================================================================
// 帮助信息
// ============================================================================

fn printUsage(argv0: []const u8) void {
    std.debug.print(
        \\usage: {s} [options]
        \\
        \\The tokenize program tokenizes a prompt using a given model,
        \\and prints the resulting tokens to standard output.
        \\
        \\It needs a model file, a prompt, and optionally other flags
        \\to control the behavior of the tokenizer.
        \\
        \\    The possible options are:
        \\
        \\    -h, --help                           print this help and exit
        \\    -m MODEL_PATH, --model MODEL_PATH    path to model.
        \\    --ids                                if given, only print numerical token IDs, and not token strings.
        \\                                         The output format looks like [1, 2, 3], i.e. parseable by Python.
        \\    -f PROMPT_FNAME, --file PROMPT_FNAME read prompt from a file.
        \\    -p PROMPT, --prompt PROMPT           read prompt from the argument.
        \\    --stdin                              read prompt from standard input.
        \\    --no-bos                             do not ever add a BOS token to the prompt, even if normally the model uses a BOS token.
        \\    --no-escape                          do not escape input (such as \n, \t, etc.).
        \\    --log-disable                        disable logs. Makes stderr quiet when loading the model.
        \\    --show-count                         print the total number of tokens.
        \\
    , .{argv0});
}

// ============================================================================
// 命令行参数
// ============================================================================

const CliArgs = struct {
    model_path: []const u8 = "",
    printing_ids: bool = false,
    no_bos: bool = false,
    no_escape: bool = false,
    log_disable: bool = false,
    show_token_count: bool = false,
    prompt: ?[]const u8 = null,
    prompt_file: ?[]const u8 = null,
    stdin_mode: bool = false,
    help: bool = false,

    pub fn parse(args_it: *std.process.Args.Iterator) !CliArgs {
        var result = CliArgs{};
        _ = args_it.next();

        var model_path_set = false;
        var prompt_set = false;
        var prompt_file_set = false;
        var stdin_set = false;

        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
                result.help = true;
                return result;
            } else if (std.mem.eql(u8, arg, "--ids")) {
                result.printing_ids = true;
            } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
                if (model_path_set) return error.InvalidArgs;
                result.model_path = args_it.next() orelse return error.InvalidArgs;
                model_path_set = true;
            } else if (std.mem.eql(u8, arg, "--no-bos")) {
                result.no_bos = true;
            } else if (std.mem.eql(u8, arg, "--no-escape")) {
                result.no_escape = true;
            } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
                if (prompt_set) return error.InvalidArgs;
                result.prompt = args_it.next() orelse return error.InvalidArgs;
                prompt_set = true;
            } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--file")) {
                if (prompt_file_set) return error.InvalidArgs;
                result.prompt_file = args_it.next() orelse return error.InvalidArgs;
                prompt_file_set = true;
            } else if (std.mem.eql(u8, arg, "--stdin")) {
                stdin_set = true;
            } else if (std.mem.eql(u8, arg, "--log-disable")) {
                result.log_disable = true;
            } else if (std.mem.eql(u8, arg, "--show-count")) {
                result.show_token_count = true;
            } else {
                return error.InvalidArgs;
            }
        }

        if (!model_path_set) return error.InvalidArgs;
        const prompts_count = @intFromBool(prompt_set) + @intFromBool(prompt_file_set) + @intFromBool(stdin_set);
        if (prompts_count != 1) return error.InvalidArgs;

        result.stdin_mode = stdin_set;
        return result;
    }
};

// ============================================================================
// 读取标准输入
// ============================================================================

fn readStdin(allocator: std.mem.Allocator) ![]u8 {
    const stdin_file = std.Io.File.stdin();
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    var temp_buf: [4096]u8 = undefined;
    while (true) {
        const n = std.posix.read(stdin_file.handle, &temp_buf) catch |err| switch (err) {
            error.WouldBlock => continue,
            else => |e| return e,
        };
        if (n == 0) break;
        try buf.appendSlice(allocator, temp_buf[0..n]);
    }

    return buf.toOwnedSlice(allocator);
}

// ============================================================================
// 转义处理
// ============================================================================

fn processEscapes(text: []const u8, allocator: std.mem.Allocator) ![]u8 {
    var result = try std.ArrayListUnmanaged(u8).initCapacity(allocator, text.len);
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\\' and i + 1 < text.len) {
            switch (text[i + 1]) {
                'n' => try result.append(allocator, '\n'),
                't' => try result.append(allocator, '\t'),
                'r' => try result.append(allocator, '\r'),
                '\\' => try result.append(allocator, '\\'),
                '\'' => try result.append(allocator, '\''),
                '"' => try result.append(allocator, '"'),
                '0' => try result.append(allocator, 0),
                else => {
                    try result.append(allocator, '\\');
                    try result.append(allocator, text[i + 1]);
                },
            }
            i += 2;
        } else {
            try result.append(allocator, text[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// 将 token ID 解码为可读字符串
// ============================================================================

fn tokenToDisplayString(tok: *const tokenizer.Tokenizer, token_id: u32, allocator: std.mem.Allocator) ![]u8 {
    const ids = [_]u32{token_id};
    return tok.decode(&ids, allocator);
}

// ============================================================================
// 主函数
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    const args = CliArgs.parse(&args_iter) catch |err| {
        if (err == error.InvalidArgs) {
            var it = std.process.Args.Iterator.init(init.minimal.args);
            defer it.deinit();
            const argv0 = it.next() orelse "zllama-tokenize";
            printUsage(argv0);
            return;
        }
        return err;
    };

    if (args.help) {
        var it = std.process.Args.Iterator.init(init.minimal.args);
        defer it.deinit();
        const argv0 = it.next() orelse "zllama-tokenize";
        printUsage(argv0);
        return;
    }

    // 禁用日志
    if (args.log_disable) {
        runtime_log_level = .err;
    }

    // 读取模型文件
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, args.model_path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const file_size = @as(usize, @intCast(stat.size));
    const gguf_data = try allocator.alloc(u8, file_size);
    defer allocator.free(gguf_data);

    const bytes_read = try file.readPositionalAll(io, gguf_data, 0);
    if (bytes_read != file_size) return error.FileReadError;

    // 解析 GGUF
    var gguf_file = try gguf.parse(gguf_data, allocator);
    defer gguf_file.deinit();

    // 初始化分词器
    var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
    defer tok.deinit();

    // 获取 prompt
    const prompt_owned = if (args.stdin_mode) blk: {
        break :blk try readStdin(allocator);
    } else if (args.prompt_file) |filepath| blk: {
        const pf = try cwd.openFile(io, filepath, .{ .mode = .read_only });
        defer pf.close(io);
        const pstat = try pf.stat(io);
        const psize = @as(usize, @intCast(pstat.size));
        const pcontent = try allocator.alloc(u8, psize);
        errdefer allocator.free(pcontent);
        const pbytes = try pf.readPositionalAll(io, pcontent, 0);
        if (pbytes != psize) return error.FileReadError;
        break :blk pcontent;
    } else if (args.prompt) |prompt| blk: {
        break :blk try allocator.dupe(u8, prompt);
    } else {
        unreachable;
    };
    defer allocator.free(prompt_owned);

    // 处理转义
    const needs_escape = !args.no_escape;
    const prompt = if (needs_escape) blk: {
        break :blk try processEscapes(prompt_owned, allocator);
    } else prompt_owned;
    defer if (needs_escape) allocator.free(prompt);

    // 确定是否添加 BOS
    const model_wants_add_bos = tok.config.add_bos;
    const add_bos = model_wants_add_bos and !args.no_bos;

    // 编码 prompt
    var tokens = try tok.encode(prompt, add_bos);
    defer tokens.deinit(allocator);

    // 输出结果
    if (args.printing_ids) {
        std.debug.print("[", .{});
        for (tokens.items, 0..) |token_id, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{token_id});
        }
        std.debug.print("]\n", .{});
    } else {
        for (tokens.items) |token_id| {
            const display = try tokenToDisplayString(&tok, token_id, allocator);
            defer allocator.free(display);
            std.debug.print("{d:6} -> '{s}'\n", .{ token_id, display });
        }
    }

    if (args.show_token_count) {
        std.debug.print("Total number of tokens: {d}\n", .{tokens.items.len});
    }
}
