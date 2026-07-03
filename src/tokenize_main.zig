//! zllama-tokenize 工具
//!
//! 对给定 prompt 进行分词，并输出 token ID 和对应的 token 字符串。
//! 与 llama.cpp 的 llama-tokenize 工具功能对齐。
//!
//! 运行时日志级别控制（通过命令行参数）：
//!   --log-level <debug|info|warn|err>  设置日志级别
//!   --verbose, -v                       设置日志级别为 info
//!   --debug, -d                         设置日志级别为 debug
//!   --log-disable                       禁用日志（设置级别为 err）

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const tokenizer = @import("tokenizer");
const utils = @import("utils.zig");
const logger = std.log.scoped(.tokenize);
const c = ggml.c;
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

/// 设置运行时日志级别
pub fn setLogLevel(level: std.log.Level) void {
    runtime_log_level = level;
}

/// 获取当前运行时日志级别
pub fn getLogLevel() std.log.Level {
    return runtime_log_level;
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
        \\    --log-level <level>                  set log level (debug, info, warn, err). Overrides --log-disable, --verbose, --debug.
        \\    -v, --verbose                        set log level to info (more verbose).
        \\    -d, --debug                          set log level to debug (most verbose).
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
    log_level: ?std.log.Level = null,
    verbose: bool = false,
    debug: bool = false,
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
            } else if (std.mem.eql(u8, arg, "--log-level")) {
                const level_str = args_it.next() orelse return error.InvalidArgs;
                result.log_level = std.meta.stringToEnum(std.log.Level, level_str) orelse {
                    logger.err("unknown log level: {s}", .{level_str});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
                result.verbose = true;
            } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--debug")) {
                result.debug = true;
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

    /// 根据命令行参数确定最终的日志级别
    /// 优先级：--log-level > --debug > --verbose > --log-disable > 默认
    pub fn resolveLogLevel(self: *const CliArgs) std.log.Level {
        if (self.log_level) |level| {
            return level;
        }
        if (self.debug) {
            return .debug;
        }
        if (self.verbose) {
            return .info;
        }
        if (self.log_disable) {
            return .err;
        }
        return .info;
    }
};

// ============================================================================
// 主函数
// ============================================================================

/// 检查 token 是否为 EOG token（通过名称匹配）
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

    // 根据命令行参数设置运行时日志级别
    const resolved_level = args.resolveLogLevel();
    setLogLevel(resolved_level);
    logger.debug("Log level set to {s}", .{@tagName(resolved_level)});

    // 使用 mmap 加载模型文件（支持大文件，零拷贝）
    const engine_common = @import("engine_common");
    var mapped_file = try engine_common.mmapFile(io, allocator, args.model_path);
    defer mapped_file.deinit(io);
    const gguf_data = mapped_file.data;
    const cwd = std.Io.Dir.cwd();

    // 解析 GGUF（gguf.parse 借用 gguf_data，所以 gguf_data 必须在 gguf_file 之前释放）
    var gguf_file = try gguf.parse(gguf_data, allocator);
    defer gguf_file.deinit();
    // 初始化分词器
    var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
    defer tok.deinit();

    // 打印 init_tokenizer 信息（与 llama.cpp 保持一致）
    {
        const tokenizer_type: u32 = switch (tok.vocab.getType()) {
            .llama => 1,
            .gpt2 => 2,
            .tiktoken => 2,
            .replit => 2,
            .spm => 1,
            else => 0,
        };
        logger.info("init_tokenizer: initializing tokenizer for type {d}", .{tokenizer_type});
    }
    // 打印模型加载信息
    utils.printModelLoaderInfo(&gguf_file, gguf_data.len, args.model_path);

    // 打印分词器信息
    utils.printTokenizerInfo(&tok, &gguf_file);

    // 打印 llama context 信息（与 llama.cpp 保持一致）
    utils.printLlamaContext();

    // 获取 prompt
    const prompt_owned = if (args.stdin_mode) blk: {
        break :blk try utils.readStdin(allocator);
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
        break :blk try utils.processEscapes(prompt_owned, allocator);
    } else prompt_owned;
    defer if (needs_escape) allocator.free(prompt);

    // 确定是否添加 BOS
    const model_wants_add_bos = tok.vocab.getAddBos();
    const add_bos = model_wants_add_bos and !args.no_bos;

    // 编码 prompt
    var tokens = try tok.encode(prompt, add_bos, true);
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
            const display = try utils.tokenToDisplayString(&tok, token_id, allocator);
            defer allocator.free(display);
            std.debug.print("{d:6} -> '{s}'\n", .{ token_id, display });
        }
    }

    if (args.show_token_count) {
        std.debug.print("Total number of tokens: {d}\n", .{tokens.items.len});
    }
}
