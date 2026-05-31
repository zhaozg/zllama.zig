//! qwen-engine 入口
//!
//! Qwen 3.5 本地推理引擎 - 主入口点
//! 处理 CLI 参数、初始化、推理循环

const std = @import("std");
const ggml = @import("ggml.zig");
const gguf = @import("gguf.zig");

// ============================================================================
// CLI 参数
// ============================================================================

const CliArgs = struct {
    model_path: [:0]const u8 = "",
    prompt: []const u8 = "Hello, how are you?",
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    n_threads: i32 = 0, // 0 = auto
    verbose: bool = false,
    help: bool = false,

    pub fn parse(args_it: *std.process.Args.Iterator) !CliArgs {
        var result = CliArgs{};

        // Skip program name
        _ = args_it.next();

        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                result.help = true;
            } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
                result.model_path = args_it.next() orelse {
                    std.debug.print("Error: --model requires a path argument\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
                result.prompt = args_it.next() orelse {
                    std.debug.print("Error: --prompt requires a string argument\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
                result.max_tokens = std.fmt.parseUnsigned(u32, args_it.next() orelse {
                    std.debug.print("Error: --max-tokens requires a number\n", .{});
                    return error.InvalidArgs;
                }, 10) catch {
                    std.debug.print("Error: invalid --max-tokens value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--temperature") or std.mem.eql(u8, arg, "-t")) {
                result.temperature = std.fmt.parseFloat(f32, args_it.next() orelse {
                    std.debug.print("Error: --temperature requires a number\n", .{});
                    return error.InvalidArgs;
                }) catch {
                    std.debug.print("Error: invalid --temperature value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--top-k") or std.mem.eql(u8, arg, "-k")) {
                result.top_k = std.fmt.parseUnsigned(u32, args_it.next() orelse {
                    std.debug.print("Error: --top-k requires a number\n", .{});
                    return error.InvalidArgs;
                }, 10) catch {
                    std.debug.print("Error: invalid --top-k value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--top-p") or std.mem.eql(u8, arg, "-tp")) {
                result.top_p = std.fmt.parseFloat(f32, args_it.next() orelse {
                    std.debug.print("Error: --top-p requires a number\n", .{});
                    return error.InvalidArgs;
                }) catch {
                    std.debug.print("Error: invalid --top-p value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--threads") or std.mem.eql(u8, arg, "-th")) {
                result.n_threads = std.fmt.parseInt(i32, args_it.next() orelse {
                    std.debug.print("Error: --threads requires a number\n", .{});
                    return error.InvalidArgs;
                }, 10) catch {
                    std.debug.print("Error: invalid --threads value\n", .{});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
                result.verbose = true;
            } else {
                std.debug.print("Warning: unknown argument '{s}'\n", .{arg});
            }
        }

        return result;
    }

    pub fn printHelp() void {
        const help_text =
            \\Qwen 3.5 本地推理引擎
            \\
            \\用法: qwen [选项]
            \\
            \\选项:
            \\  -h, --help            显示此帮助信息
            \\  -m, --model <路径>     模型文件路径 (GGUF格式)
            \\  -p, --prompt <文本>    输入提示词 (默认: "Hello, how are you?")
            \\  -n, --max-tokens <N>  最大生成token数 (默认: 256)
            \\  -t, --temperature <F> 采样温度 (默认: 0.7)
            \\  -k, --top-k <N>       Top-K 采样 (默认: 40)
            \\  -tp, --top-p <F>      Top-P 采样 (默认: 0.9)
            \\  -th, --threads <N>    线程数 (默认: auto)
            \\  -v, --verbose         详细输出
            \\
        ;
        std.debug.print("{s}", .{help_text});
    }
};

// ============================================================================
// 主函数
// Zig 0.16.0: main 可接受 std.process.Init 参数
// 使用 Init 参数可以获取 args 和 io 等系统资源
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    // 解析 CLI 参数
    // Zig 0.16.0: 使用 Args.Iterator.initAllocator 替代 argsWithAllocator
    // 在 macOS/Linux 上，Args 的 Vector 是 []const [*:0]const u8
    // 我们通过 std.process.Args.Iterator.initAllocator 传入空的 Args 来获取迭代器
    // 注意：在 Zig 0.16.0 中，非 Windows 平台使用 Posix 实现，init 不需要 allocator
    var args_iter = std.process.Args.Iterator.init(std.process.Args{ .vector = @as([]const [*:0]const u8, &.{}) });
    defer args_iter.deinit();

    const args = CliArgs.parse(&args_iter) catch |err| {
        if (err == error.InvalidArgs) {
            CliArgs.printHelp();
            return;
        }
        return err;
    };

    if (args.help) {
        CliArgs.printHelp();
        return;
    }

    // 打印版本信息
    std.debug.print("Qwen Engine v0.1.0 (ggml {s})\n", .{ggml.version()});

    // 检测 CPU 特性
    if (args.verbose) {
        std.debug.print("CPU features:\n", .{});
        std.debug.print("  AVX2:  {}\n", .{ggml.CpuFeatures.hasAvx2()});
        std.debug.print("  AVX:   {}\n", .{ggml.CpuFeatures.hasAvx()});
        std.debug.print("  NEON:  {}\n", .{ggml.CpuFeatures.hasNeon()});
        std.debug.print("  Metal: {}\n", .{ggml.CpuFeatures.hasMetal()});
        std.debug.print("  CUDA:  {}\n", .{ggml.CpuFeatures.hasCuda()});
    }

    // 确定线程数
    const n_threads = if (args.n_threads > 0) args.n_threads else ggml.recommendedThreads();
    std.debug.print("Using {d} threads\n", .{n_threads});

    // 如果没有指定模型路径，显示帮助
    if (args.model_path.len == 0) {
        std.debug.print("Error: no model specified. Use --model <path> to specify a GGUF model file.\n", .{});
        std.debug.print("\nTip: You can also run with --help to see all options.\n", .{});
        return;
    }

    // 测试 GGUF 解析
    std.debug.print("\nLoading model: {s}\n", .{args.model_path});

    // 使用 ggml 的 GGUF API 加载
    var gguf_ctx = ggml.GgufContext.initFromFile(args.model_path, false) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        std.debug.print("Trying manual parser...\n", .{});

        // 尝试手动解析
        var parser = gguf.Parser.init(io, allocator);
        defer parser.deinit();

        parser.parseFromFile(args.model_path) catch |parse_err| {
            std.debug.print("Manual parsing also failed: {}\n", .{parse_err});
            return;
        };

        if (args.verbose) {
            parser.dumpMetadata();
            parser.dumpTensors();
        }

        std.debug.print("Model loaded successfully (manual parser).\n", .{});
        std.debug.print("Version: {d}, Tensors: {d}, Metadata KV: {d}\n", .{
            parser.header.version,
            parser.header.tensor_count,
            parser.header.metadata_kv_count,
        });
        return;
    };
    defer gguf_ctx.deinit();

    std.debug.print("Model loaded successfully (ggml API).\n", .{});

    // 打印模型信息
    const version = gguf_ctx.version();
    const n_tensors = gguf_ctx.nTensors();
    const n_kv = gguf_ctx.nKv();
    std.debug.print("GGUF version: {d}\n", .{version});
    std.debug.print("Tensors: {d}\n", .{n_tensors});
    std.debug.print("Metadata KV pairs: {d}\n", .{n_kv});

    // 打印元数据
    if (args.verbose) {
        std.debug.print("\n--- Metadata ---\n", .{});
        var meta_iter = gguf_ctx.initMeta();
        while (meta_iter.next()) |entry| {
            const key = entry.key;
            const val = entry.value;
            switch (val) {
                .string => |s| std.debug.print("  {s}: {s}\n", .{ key, s }),
                .int32 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .uint32 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .float32 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .bool => |v| std.debug.print("  {s}: {}\n", .{ key, v }),
                .int64 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                .uint64 => |v| std.debug.print("  {s}: {d}\n", .{ key, v }),
                else => std.debug.print("  {s}: (other type)\n", .{key}),
            }
        }

        // 打印张量信息
        std.debug.print("\n--- Tensors (first 10) ---\n", .{});
        const n = @min(@as(i64, 10), n_tensors);
        var i: i32 = 0;
        while (i < n) : (i += 1) {
            const info = gguf_ctx.tensorInfo(i);
            std.debug.print("  [{d}] {s}: type={s}, shape=[{d},{d},{d},{d}], offset={d}\n", .{
                i,
                info.name,
                info.typ.name(),
                info.ne[0],
                info.ne[1],
                info.ne[2],
                info.ne[3],
                info.offset,
            });
        }
    }

    // 获取关联的 ggml 上下文
    const ggml_ctx = gguf_ctx.ggmlCtx() orelse {
        std.debug.print("Warning: no ggml context from GGUF\n", .{});
        return;
    };

    _ = ggml_ctx;

    std.debug.print("\nModel loaded. Ready for inference.\n", .{});
    std.debug.print("Prompt: \"{s}\"\n", .{args.prompt});
    std.debug.print("Max tokens: {d}\n", .{args.max_tokens});
    std.debug.print("Temperature: {d}\n", .{args.temperature});
    std.debug.print("Top-K: {d}, Top-P: {d}\n", .{ args.top_k, args.top_p });

    // TODO: 阶段二 - 实现推理循环
    std.debug.print("\nInference not yet implemented (Phase 2).\n", .{});
}

// ============================================================================
// 测试
// ============================================================================

test "CliArgs parse" {
    // Test that CliArgs can be constructed
    const args = CliArgs{};
    try std.testing.expectEqual(@as(u32, 256), args.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.7), args.temperature);
}

test "ggml version available" {
    const v = ggml.version();
    try std.testing.expect(v.len > 0);
}
