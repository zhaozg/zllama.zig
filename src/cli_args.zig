const std = @import("std");

const logger = std.log.scoped(.main);

pub const CliArgs = struct {
    model_path: [:0]const u8 = "",
    prompt: []const u8 = "Hello, how are you?",
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    n_threads: i32 = 0,
    verbose: bool = false,
    debug: bool = false,
    help: bool = false,
    benchmark: bool = false,
    chat: bool = false,
    info: bool = false,
    // Embedding
    embed: bool = false,
    pooling: []const u8 = "mean",
    embed_normalize: bool = true,
    // Multimodal
    mmproj_path: [:0]const u8 = "",
    image_path: [:0]const u8 = "",
    audio_path: [:0]const u8 = "",
    // Chat template
    chat_template_name: []const u8 = "",
    system_prompt: []const u8 = "",
    no_chat_template: bool = false,
    no_jinja: bool = false,
    pub fn parse(args_it: *std.process.Args.Iterator) !CliArgs {
        var result = CliArgs{};
        _ = args_it.next();
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                result.help = true;
            } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
                result.model_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
                result.prompt = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
                result.max_tokens = std.fmt.parseUnsigned(u32, args_it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--temperature") or std.mem.eql(u8, arg, "-t")) {
                result.temperature = std.fmt.parseFloat(f32, args_it.next() orelse return error.InvalidArgs) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--top-k") or std.mem.eql(u8, arg, "-k")) {
                result.top_k = std.fmt.parseUnsigned(u32, args_it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--top-p") or std.mem.eql(u8, arg, "-tp")) {
                result.top_p = std.fmt.parseFloat(f32, args_it.next() orelse return error.InvalidArgs) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--threads") or std.mem.eql(u8, arg, "-th")) {
                result.n_threads = std.fmt.parseInt(i32, args_it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
                result.verbose = true;
            } else if (std.mem.eql(u8, arg, "--debug") or std.mem.eql(u8, arg, "-d")) {
                result.debug = true;
            } else if (std.mem.eql(u8, arg, "--chat") or std.mem.eql(u8, arg, "-c")) {
                result.chat = true;
            } else if (std.mem.eql(u8, arg, "--mmproj")) {
                result.mmproj_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--image")) {
                result.image_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--audio")) {
                result.audio_path = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--embed")) {
                result.embed = true;
            } else if (std.mem.eql(u8, arg, "--pooling")) {
                result.pooling = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--embd-normalize")) {
                const val = args_it.next() orelse return error.InvalidArgs;
                result.embed_normalize = std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
            } else if (std.mem.eql(u8, arg, "--chat-template")) {
                result.chat_template_name = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--system-prompt")) {
                result.system_prompt = args_it.next() orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, arg, "--no-chat-template")) {
                result.no_chat_template = true;
            } else if (std.mem.eql(u8, arg, "--no-jinja")) {
                result.no_jinja = true;
                // --no-jinja only disables Jinja rendering, does NOT disable chat template
            } else {
                logger.warn("unknown argument '{s}'", .{arg});
            }
        }
        return result;
    }

    pub fn printHelp() void {
        std.debug.print(
            \\zllama.zig - 多模型本地推理引擎
            \\
            \\用法: zllama [选项]
            \\
            \\选项:
            \\  -h, --help            显示此帮助信息
            \\  -m, --model <路径>     模型文件路径 (GGUF格式)
            \\  -p, --prompt <文本>    输入提示词
            \\  -n, --max-tokens <N>  最大生成token数 (默认: 256)
            \\  -v, --verbose         详细日志输出 (info 级别)
            \\  -d, --debug           调试日志输出 (debug 级别)
            \\  --benchmark           benchmark 模式
            \\  -c, --chat            交互式聊天模式
            \\
            \\对话模板选项:
            \\  --chat-template <名称|Jinja> 指定对话模板 (预设名称或Jinja模板字符串)
            \\                  预设: chatml | llama3 | llama4 | gemma | gemma4 |
            \\                        mistral-v7 | phi4 | deepseek3 | tinyllama
            \\                  也可直接传入Jinja模板字符串作为自定义模板
            \\  --system-prompt <文本> 指定系统提示词
            \\  --no-chat-template     禁用对话模板，原始 prompt 透传
            \\
            \\嵌入模式选项:
            \\  --embed               启用嵌入向量生成模式
            \\  --pooling <策略>      池化策略: mean | cls | last (默认: mean)
            \\  --embd-normalize 1    是否 L2 归一化 (默认: 1/true)
            \\
            \\多模态选项:
            \\  --mmproj <路径>       多模态投影器文件 (GGUF格式, mmproj)
            \\  --image <路径>        输入图像文件 (PPM/JPEG/PNG/BMP/GIF)
            \\  --audio <路径>        输入音频文件 (WAV 16-bit PCM)
        , .{});
    }
};

const testing = std.testing;

test "CliArgs parse" {
    const test_args = CliArgs{};
    try testing.expectEqual(@as(u32, 256), test_args.max_tokens);
    try testing.expectEqual(@as(f32, 0.7), test_args.temperature);
}
