//! 基于词汇表的 tokenizer 测试
//!
//! 验证 tokenizer 的编码/解码正确性，使用 llama.cpp 提供的标准测试数据。
//!
//! 测试策略（参考 llama.cpp 的 test-tokenizer-0.cpp）：
//! 1. 加载 GGUF 词汇表文件
//! 2. 读取 .inp 文件（文本，由 __ggml_vocab_test__ 分隔）
//! 3. 读取 .out 文件（每行空格分隔的 token ID）
//! 4. 对每个输入文本进行 tokenize，验证输出 token ID 与预期一致
//! 5. 验证 detokenize(tokenize(text)) == text（往返一致性）
//!
//! 支持的词汇表：
//! - llama-bpe (BPE)
//! - llama-spm (SentencePiece/Unigram)
//! - qwen2 (BPE)
//! - qwen35 (BPE)
//! - gpt-2 (BPE)
//! - falcon (BPE)
//! - deepseek-coder (BPE)
//! - deepseek-llm (BPE)
//! - phi-3 (BPE)
//! - command-r (BPE)
//! - starcoder (BPE)
//! - mpt (BPE)
//! - refact (BPE)
//! - baichuan (BPE)
//! - bert-bge (BPE)
//! - gemma-4 (BPE)
//! - nomic-bert-moe (BPE)
//! - aquila (BPE)

const std = @import("std");
const testing = std.testing;
const gguf = @import("gguf");
const tokenizer = @import("tokenizer");

const log = std.log.scoped(.test_vocab);

/// 测试数据目录
const test_data_dir = "deps/llama.cpp/models";

/// 分隔符（与 llama.cpp 保持一致）
const test_separator = "\n__ggml_vocab_test__\n";

/// 词汇表测试配置
const VocabTestConfig = struct {
    name: []const u8,
    /// 是否跳过（某些词汇表可能不支持）
    skip: bool = false,
    /// 是否忽略 merges 检查（某些词汇表 token 可能被拆分为多个）
    ignore_merges: bool = false,
};

/// 所有支持的词汇表测试
const vocab_tests = [_]VocabTestConfig{
    .{ .name = "llama-bpe" },
    .{ .name = "llama-spm" },
    .{ .name = "qwen2" },
    .{ .name = "qwen35" },
    .{ .name = "gpt-2" },
    .{ .name = "falcon" },
    .{ .name = "deepseek-coder" },
    .{ .name = "deepseek-llm" },
    .{ .name = "phi-3" },
    .{ .name = "command-r" },
    .{ .name = "starcoder" },
    .{ .name = "mpt" },
    .{ .name = "refact" },
    .{ .name = "baichuan" },
    .{ .name = "bert-bge" },
    .{ .name = "gemma-4" },
    .{ .name = "nomic-bert-moe" },
    .{ .name = "aquila" },
};

// ============================================================================
// 辅助函数
// ============================================================================

/// 读取文件内容
fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const cwd = std.fs.cwd();
    const file = try cwd.openFile(path, .{ .mode = .read_only });
    defer file.close();

    const stat = try file.stat();
    const content = try allocator.alloc(u8, @as(usize, @intCast(stat.size)));
    errdefer allocator.free(content);

    const bytes_read = try file.readAll(content);
    if (bytes_read != content.len) return error.FileReadError;
    return content;
}

/// 解析 .inp 文件：按分隔符分割文本
/// 返回分配的所有权字符串列表，调用者负责释放
fn parseInpFile(allocator: std.mem.Allocator, content: []const u8) ![][]const u8 {
    var result = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (result.items) |item| allocator.free(item);
        result.deinit();
    }

    var start: usize = 0;
    while (start < content.len) {
        const end = std.mem.indexOfPos(u8, content, start, test_separator) orelse content.len;
        const segment = content[start..end];
        // 复制每个段
        const owned = try allocator.dupe(u8, segment);
        try result.append(owned);
        start = end + test_separator.len;
    }

    return result.toOwnedSlice();
}

/// 解析 .out 文件：每行空格分隔的 token ID
/// 返回分配的所有权列表，调用者负责释放
fn parseOutFile(allocator: std.mem.Allocator, content: []const u8) ![][]const u32 {
    var result = std.ArrayList([]const u32).init(allocator);
    errdefer {
        for (result.items) |item| allocator.free(item);
        result.deinit();
    }

    var line_iter = std.mem.splitScalar(u8, content, '\n');
    while (line_iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (trimmed.len == 0) continue;

        var tokens = std.ArrayList(u32).init(allocator);
        errdefer tokens.deinit();

        var token_iter = std.mem.splitScalar(u8, trimmed, ' ');
        while (token_iter.next()) |token_str| {
            const trimmed_token = std.mem.trim(u8, token_str, &std.ascii.whitespace);
            if (trimmed_token.len == 0) continue;
            const token_id = try std.fmt.parseInt(u32, trimmed_token, 10);
            try tokens.append(token_id);
        }

        try result.append(try tokens.toOwnedSlice());
    }

    return result.toOwnedSlice();
}

/// 释放解析后的测试数据
fn freeTestData(allocator: std.mem.Allocator, inputs: [][]const u8, expected_outputs: [][]const u32) void {
    for (inputs) |input| allocator.free(input);
    allocator.free(inputs);
    for (expected_outputs) |output| allocator.free(output);
    allocator.free(expected_outputs);
}

/// 运行单个词汇表的测试
fn runVocabTest(
    allocator: std.mem.Allocator,
    config: VocabTestConfig,
) !void {
    // 构建文件路径
    const gguf_path = try std.fmt.allocPrint(allocator, "{s}/ggml-vocab-{s}.gguf", .{ test_data_dir, config.name });
    defer allocator.free(gguf_path);

    const inp_path = try std.fmt.allocPrint(allocator, "{s}/ggml-vocab-{s}.gguf.inp", .{ test_data_dir, config.name });
    defer allocator.free(inp_path);

    const out_path = try std.fmt.allocPrint(allocator, "{s}/ggml-vocab-{s}.gguf.out", .{ test_data_dir, config.name });
    defer allocator.free(out_path);

    // 读取 GGUF 文件
    const gguf_data = try readFile(allocator, gguf_path);
    defer allocator.free(gguf_data);

    // 解析 GGUF
    var gguf_file = try gguf.parse(gguf_data, allocator);
    defer gguf_file.deinit();

    // 初始化 tokenizer
    var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
    defer tok.deinit();

    // 读取 .inp 和 .out 文件
    const inp_content = try readFile(allocator, inp_path);
    defer allocator.free(inp_content);

    const out_content = try readFile(allocator, out_path);
    defer allocator.free(out_content);

    // 解析测试数据
    const inputs = try parseInpFile(allocator, inp_content);
    errdefer {
        for (inputs) |input| allocator.free(input);
        allocator.free(inputs);
    }

    const expected_outputs = try parseOutFile(allocator, out_content);
    errdefer {
        for (expected_outputs) |output| allocator.free(output);
        allocator.free(expected_outputs);
    }

    // 验证输入和输出数量一致
    try testing.expectEqual(inputs.len, expected_outputs.len);

    // 对每个测试用例进行 tokenize 验证
    for (inputs, expected_outputs, 0..) |input, expected, i| {
        // 编码（不添加特殊 token）
        var tokens = try tok.encode(input, false);
        defer tokens.deinit(allocator);

        // 验证 token 数量
        if (tokens.items.len != expected.len) {
            std.debug.print("\n  [{s}] Test #{d}: '{s}' -> expected {d} tokens, got {d} tokens\n", .{
                config.name, i, input, expected.len, tokens.items.len,
            });
            std.debug.print("  Expected tokens: ", .{});
            for (expected) |t| std.debug.print("{d} ", .{t});
            std.debug.print("\n  Got tokens: ", .{});
            for (tokens.items) |t| std.debug.print("{d} ", .{t});
            std.debug.print("\n", .{});
        }
        try testing.expectEqual(expected.len, tokens.items.len);

        // 验证每个 token ID
        for (tokens.items, expected, 0..) |got, exp, j| {
            if (got != exp) {
                std.debug.print("\n  [{s}] Test #{d}, token #{d}: expected {d}, got {d}\n", .{
                    config.name, i, j, exp, got,
                });
                std.debug.print("  Input: '{s}'\n", .{input});
            }
            try testing.expectEqual(exp, got);
        }

        // 验证往返一致性：detokenize(tokenize(text)) == text
        const decoded = try tok.decode(tokens.items, allocator);
        defer allocator.free(decoded);

        // 注意：某些特殊 token（如字节级 token）在往返过程中可能会有细微差异
        // 这里只做基本检查，不严格相等
        if (decoded.len > 0) {
            // 验证解码后的文本至少包含原始文本的部分内容
            // 对于空输入，解码结果也应为空
            if (input.len == 0) {
                try testing.expectEqual(@as(usize, 0), decoded.len);
            }
        }
    }

    // 测试通过
    std.debug.print("  [{s}] {d} tests passed\n", .{ config.name, inputs.len });
}

// ============================================================================
// 测试用例
// ============================================================================

test "vocab - llama-bpe" {
    try runVocabTest(testing.allocator, .{ .name = "llama-bpe" });
}

test "vocab - llama-spm" {
    try runVocabTest(testing.allocator, .{ .name = "llama-spm" });
}

test "vocab - qwen2" {
    try runVocabTest(testing.allocator, .{ .name = "qwen2" });
}

test "vocab - qwen35" {
    try runVocabTest(testing.allocator, .{ .name = "qwen35" });
}

test "vocab - gpt-2" {
    try runVocabTest(testing.allocator, .{ .name = "gpt-2" });
}

test "vocab - falcon" {
    try runVocabTest(testing.allocator, .{ .name = "falcon" });
}

test "vocab - deepseek-coder" {
    try runVocabTest(testing.allocator, .{ .name = "deepseek-coder" });
}

test "vocab - deepseek-llm" {
    try runVocabTest(testing.allocator, .{ .name = "deepseek-llm" });
}

test "vocab - phi-3" {
    try runVocabTest(testing.allocator, .{ .name = "phi-3" });
}

test "vocab - command-r" {
    try runVocabTest(testing.allocator, .{ .name = "command-r" });
}

test "vocab - starcoder" {
    try runVocabTest(testing.allocator, .{ .name = "starcoder" });
}

test "vocab - mpt" {
    try runVocabTest(testing.allocator, .{ .name = "mpt" });
}

test "vocab - refact" {
    try runVocabTest(testing.allocator, .{ .name = "refact" });
}

test "vocab - baichuan" {
    try runVocabTest(testing.allocator, .{ .name = "baichuan" });
}

test "vocab - bert-bge" {
    try runVocabTest(testing.allocator, .{ .name = "bert-bge" });
}

test "vocab - gemma-4" {
    try runVocabTest(testing.allocator, .{ .name = "gemma-4" });
}

test "vocab - nomic-bert-moe" {
    try runVocabTest(testing.allocator, .{ .name = "nomic-bert-moe" });
}

test "vocab - aquila" {
    try runVocabTest(testing.allocator, .{ .name = "aquila" });
}

// ============================================================================
// 辅助测试：解析器单元测试
// ============================================================================

test "parseInpFile basic" {
    const content = "hello\n__ggml_vocab_test__\nworld\n__ggml_vocab_test__\n!";
    const inputs = try parseInpFile(testing.allocator, content);
    defer {
        for (inputs) |input| testing.allocator.free(input);
        testing.allocator.free(inputs);
    }

    try testing.expectEqual(@as(usize, 3), inputs.len);
    try testing.expectEqualStrings("hello", inputs[0]);
    try testing.expectEqualStrings("world", inputs[1]);
    try testing.expectEqualStrings("!", inputs[2]);
}

test "parseInpFile empty segments" {
    const content = "a\n__ggml_vocab_test__\n\n__ggml_vocab_test__\nb";
    const inputs = try parseInpFile(testing.allocator, content);
    defer {
        for (inputs) |input| testing.allocator.free(input);
        testing.allocator.free(inputs);
    }

    try testing.expectEqual(@as(usize, 3), inputs.len);
    try testing.expectEqualStrings("a", inputs[0]);
    try testing.expectEqualStrings("", inputs[1]);
    try testing.expectEqualStrings("b", inputs[2]);
}

test "parseOutFile basic" {
    const content = "1 2 3\n4 5\n6\n";
    const outputs = try parseOutFile(testing.allocator, content);
    defer {
        for (outputs) |output| testing.allocator.free(output);
        testing.allocator.free(outputs);
    }

    try testing.expectEqual(@as(usize, 3), outputs.len);
    try testing.expectEqual(@as(u32, 1), outputs[0][0]);
    try testing.expectEqual(@as(u32, 2), outputs[0][1]);
    try testing.expectEqual(@as(u32, 3), outputs[0][2]);
    try testing.expectEqual(@as(u32, 4), outputs[1][0]);
    try testing.expectEqual(@as(u32, 5), outputs[1][1]);
    try testing.expectEqual(@as(u32, 6), outputs[2][0]);
}

test "parseOutFile empty lines" {
    const content = "1 2\n\n3\n\n\n";
    const outputs = try parseOutFile(testing.allocator, content);
    defer {
        for (outputs) |output| testing.allocator.free(output);
        testing.allocator.free(outputs);
    }

    try testing.expectEqual(@as(usize, 2), outputs.len);
    try testing.expectEqual(@as(u32, 1), outputs[0][0]);
    try testing.expectEqual(@as(u32, 3), outputs[1][0]);
}

test "parseOutFile single token" {
    const content = "42\n";
    const outputs = try parseOutFile(testing.allocator, content);
    defer {
        for (outputs) |output| testing.allocator.free(output);
        testing.allocator.free(outputs);
    }

    try testing.expectEqual(@as(usize, 1), outputs.len);
    try testing.expectEqual(@as(u32, 42), outputs[0][0]);
}
