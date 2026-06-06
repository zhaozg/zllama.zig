//! 计算图转储工具
//!
//! 将 ggml 计算图转储为可读的文本格式，用于调试和可视化。
//! 支持输出为：
//! - 文本格式（默认）：人类可读的节点列表
//! - DOT 格式：Graphviz 可视化
//! - JSON 格式：机器可读的结构化数据
//!
//! 用法：
//!   zllama-dump-graph --model model.gguf --prompt "Hello" [--format dot|json|text]
//!
//! 参考 llama.cpp 的 dump_graph 工具设计。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");

const log = std.log.scoped(.dump_graph);

// ============================================================================
// 输出格式
// ============================================================================

pub const OutputFormat = enum {
    text,
    dot,
    json,
};

// ============================================================================
// 图转储器
// ============================================================================

pub const GraphDumper = struct {
    allocator: std.mem.Allocator,
    format: OutputFormat,

    pub fn init(allocator: std.mem.Allocator, format: OutputFormat) GraphDumper {
        return GraphDumper{ .allocator = allocator, .format = format };
    }

    /// 转储计算图到 ArrayList
    pub fn dumpToArrayList(
        self: *GraphDumper,
        graph: *ggml.CGraph,
        out: *std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
    ) !void {
        switch (self.format) {
            .text => try self.dumpText(graph, out, allocator),
            .dot => try self.dumpDot(graph, out, allocator),
            .json => try self.dumpJson(graph, out, allocator),
        }
    }

    /// 文本格式输出
    fn dumpText(
        _: *GraphDumper,
        graph: *ggml.CGraph,
        out: *std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
    ) !void {
        const n_nodes = graph.nNodes();
        try out.appendSlice(allocator, "=== Graph Dump: ");
        try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{n_nodes}));
        try out.appendSlice(allocator, " nodes ===\n\n");

        var i: i32 = 0;
        while (i < n_nodes) : (i += 1) {
            const node = graph.getNode(i);
            const name = node.getName();
            const op = node.getOpName();
            const n_dims = node.nDims();
            const ne = node.ne();

            try out.appendSlice(allocator, "[");
            try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{i}));
            try out.appendSlice(allocator, "] ");
            try out.appendSlice(allocator, op);
            try out.appendSlice(allocator, " (");
            var d: i32 = 0;
            while (d < n_dims) : (d += 1) {
                if (d > 0) try out.appendSlice(allocator, ", ");
                try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{ne[@intCast(d)]}));
            }
            try out.appendSlice(allocator, ")");

            if (name.len > 0) {
                try out.appendSlice(allocator, " \"");
                try out.appendSlice(allocator, name);
                try out.appendSlice(allocator, "\"");
            }

            try out.appendSlice(allocator, "\n");
        }
    }

    /// DOT 格式输出（Graphviz）
    fn dumpDot(
        _: *GraphDumper,
        graph: *ggml.CGraph,
        out: *std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
    ) !void {
        const n_nodes = graph.nNodes();
        try out.appendSlice(allocator, "digraph G {\n");
        try out.appendSlice(allocator, "  rankdir=LR;\n");
        try out.appendSlice(allocator, "  node [shape=box, style=filled, fillcolor=lightyellow];\n\n");

        var i: i32 = 0;
        while (i < n_nodes) : (i += 1) {
            const node = graph.getNode(i);
            const name = node.getName();
            const op = node.getOpName();
            const n_dims = node.nDims();
            const ne = node.ne();

            // 节点标签
            var label_buf: [256]u8 = undefined;
            var label: []const u8 = undefined;
            if (name.len > 0) {
                label = try std.fmt.bufPrint(&label_buf, "{s}\\n{s}\\n({d}", .{ name, op, ne[0] });
            } else {
                label = try std.fmt.bufPrint(&label_buf, "{s}\\n({d}", .{ op, ne[0] });
            }

            // 添加维度信息
            var d: i32 = 1;
            while (d < n_dims) : (d += 1) {
                const new_label = try std.fmt.bufPrint(&label_buf, "{s}, {d}", .{ label, ne[@intCast(d)] });
                label = new_label;
            }
            label = try std.fmt.bufPrint(&label_buf, "{s})", .{label});

            try out.appendSlice(allocator, "  n");
            try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{i}));
            try out.appendSlice(allocator, " [label=\"");
            try out.appendSlice(allocator, label);
            try out.appendSlice(allocator, "\"];\n");
        }

        try out.appendSlice(allocator, "}\n");
    }

    /// JSON 格式输出
    fn dumpJson(
        _: *GraphDumper,
        graph: *ggml.CGraph,
        out: *std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
    ) !void {
        const n_nodes = graph.nNodes();
        try out.appendSlice(allocator, "{\n");
        try out.appendSlice(allocator, "  \"n_nodes\": ");
        try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{n_nodes}));
        try out.appendSlice(allocator, ",\n");
        try out.appendSlice(allocator, "  \"nodes\": [\n");

        var i: i32 = 0;
        while (i < n_nodes) : (i += 1) {
            const node = graph.getNode(i);
            const name = node.getName();
            const op = node.getOpName();
            const n_dims = node.nDims();
            const ne = node.ne();

            try out.appendSlice(allocator, "    {\n");
            try out.appendSlice(allocator, "      \"index\": ");
            try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{i}));
            try out.appendSlice(allocator, ",\n");
            try out.appendSlice(allocator, "      \"op\": \"");
            try out.appendSlice(allocator, op);
            try out.appendSlice(allocator, "\",\n");
            try out.appendSlice(allocator, "      \"name\": \"");
            try out.appendSlice(allocator, name);
            try out.appendSlice(allocator, "\",\n");
            try out.appendSlice(allocator, "      \"n_dims\": ");
            try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{n_dims}));
            try out.appendSlice(allocator, ",\n");
            try out.appendSlice(allocator, "      \"shape\": [");
            var d: i32 = 0;
            while (d < n_dims) : (d += 1) {
                if (d > 0) try out.appendSlice(allocator, ", ");
                try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{ne[@intCast(d)]}));
            }
            try out.appendSlice(allocator, "]\n");

            if (i + 1 < n_nodes) {
                try out.appendSlice(allocator, "    },\n");
            } else {
                try out.appendSlice(allocator, "    }\n");
            }
        }

        try out.appendSlice(allocator, "  ]\n");
        try out.appendSlice(allocator, "}\n");
    }
};

// ============================================================================
// 辅助函数：构建推理图并转储
// ============================================================================

/// 构建模型推理图并返回
pub fn buildInferenceGraph(
    allocator: std.mem.Allocator,
    model: *model_if.ModelInstance,
    params: *const model_if.ModelParams,
    input_tokens: []const i32,
) !*ggml.CGraph {
    const n_tokens: i32 = @intCast(input_tokens.len);

    // 创建 ggml context
    const ctx = try ggml.Context.initNoAlloc(256 * 1024); // 256KB
    errdefer ctx.deinit();

    ctx.setNoAlloc(false);
    const input_tensor = try ctx.newTensor1d(.i32, n_tokens);
    ctx.setNoAlloc(true);

    // 复制输入 token
    const data = input_tensor.dataBytes();
    const dst = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
    @memcpy(dst, input_tokens);

    // 构建计算图
    const graph = try ggml.CGraph.init(ctx);
    var builder = graph_builder.GraphBuilder.init(ctx, graph, params, allocator);
    _ = try model.buildGraph(&builder, input_tensor, n_tokens, null, 0);

    return graph;
}

// ============================================================================
// ============================================================================
// 命令行入口
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    // 解析命令行参数
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    var model_path: ?[]const u8 = null;
    var prompt: []const u8 = "Hello";
    var format: OutputFormat = .text;

    // 跳过 argv[0]
    _ = args_iter.next();

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            model_path = args_iter.next() orelse {
                std.debug.print("Error: --model requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            prompt = args_iter.next() orelse {
                std.debug.print("Error: --prompt requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--format") or std.mem.eql(u8, arg, "-f")) {
            const fmt = args_iter.next() orelse {
                std.debug.print("Error: --format requires a value (text|dot|json)\n", .{});
                std.process.exit(1);
            };
            if (std.mem.eql(u8, fmt, "dot")) {
                format = .dot;
            } else if (std.mem.eql(u8, fmt, "json")) {
                format = .json;
            } else if (!std.mem.eql(u8, fmt, "text")) {
                std.debug.print("Error: unknown format '{s}', expected 'text', 'dot', or 'json'\n", .{fmt});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            std.debug.print("Usage: zllama-dump-graph [options]\n", .{});
            std.debug.print("Options:\n", .{});
            std.debug.print("  --model, -m <path>     Model GGUF file path (required)\n", .{});
            std.debug.print("  --prompt, -p <text>    Input prompt (default: \"Hello\")\n", .{});
            std.debug.print("  --format, -f <fmt>     Output format: text|dot|json (default: text)\n", .{});
            std.debug.print("  --help, -h             Show this help\n", .{});
            return;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            std.debug.print("Warning: unknown argument '{s}'\n", .{arg});
        } else if (model_path == null) {
            // 位置参数作为 model path
            model_path = arg;
        }
    }

    if (model_path == null) {
        std.debug.print("Error: --model is required\n", .{});
        std.debug.print("Usage: zllama-dump-graph --model <model.gguf> [options]\n", .{});
        std.process.exit(1);
    }

    // 1. 加载 GGUF 文件
    log.info("Loading model: {s}", .{model_path.?});
    const dir = std.Io.Dir.cwd();
    const file = try dir.openFile(io, model_path.?, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const file_size = @as(usize, @intCast(stat.size));
    const gguf_data = try allocator.alloc(u8, file_size);
    defer allocator.free(gguf_data);

    const bytes_read = try file.readPositionalAll(io, gguf_data, 0);
    if (bytes_read != file_size) return error.FileReadError;

    var gguf_file = try gguf.parse(gguf_data, allocator);
    defer gguf_file.deinit();

    // 2. 检测架构
    const arch = registry.detectArchitecture(&gguf_file) orelse {
        log.err("Could not detect architecture from {s}", .{model_path.?});
        return error.UnsupportedArchitecture;
    };
    log.info("Detected architecture: {s}", .{@tagName(arch)});

    // 3. 创建模型
    var model = try registry.createModel(allocator, &gguf_file, arch, io);
    defer model.deinit(allocator);

    const params = model.getParams();
    log.info("Model: n_vocab={d}, n_embd={d}, n_layer={d}, n_head={d}", .{
        params.n_vocab, params.n_embd, params.n_layer, params.n_head,
    });

    // 4. Tokenize prompt
    const tokenizer = @import("tokenizer");
    var tok = try tokenizer.Tokenizer.init(&gguf_file, allocator);
    defer tok.deinit();

    var input_token_list = try tok.encode(prompt, false);
    defer input_token_list.deinit(allocator);
    const input_tokens = input_token_list.items;

    log.info("Prompt: \"{s}\" -> {d} tokens", .{ prompt, input_tokens.len });

    // 5. 构建推理图
    const n_tokens: i32 = @intCast(input_tokens.len);
    const ctx = try ggml.Context.initNoAlloc(256 * 1024);
    defer ctx.deinit();

    ctx.setNoAlloc(false);
    const input_tensor = try ctx.newTensor1d(.i32, n_tokens);
    ctx.setNoAlloc(true);

    // 复制输入 token
    const data = input_tensor.dataBytes();
    const dst = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
    for (input_tokens, 0..) |token, j| {
        dst[j] = @as(i32, @intCast(token));
    }

    const graph = try ggml.CGraph.init(ctx);
    var builder = graph_builder.GraphBuilder.init(ctx, graph, params, allocator);
    _ = try model.buildGraph(&builder, input_tensor, n_tokens, null, 0);

    // 6. 输出计算图
    var dumper = GraphDumper.init(allocator, format);
    var out_buf = std.ArrayListUnmanaged(u8).empty;
    defer out_buf.deinit(allocator);
    try dumper.dumpToArrayList(graph, &out_buf, allocator);

    // 写入 stdout
    const stdout_file = std.Io.File.stdout();
    try stdout_file.writeStreamingAll(io, out_buf.items);
}


// 测试
// ============================================================================

const testing = std.testing;

test "GraphDumper init" {
    const dumper = GraphDumper.init(testing.allocator, .text);
    try testing.expectEqual(OutputFormat.text, dumper.format);
}

test "GraphDumper dumpText empty graph" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    var dumper = GraphDumper.init(testing.allocator, .text);

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(testing.allocator);
    try dumper.dumpToArrayList(graph, &buf, testing.allocator);
    const output = buf.items;

    // 验证输出包含节点数
    try testing.expect(std.mem.indexOf(u8, output, "0 nodes") != null);
}

test "GraphDumper dumpDot empty graph" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    var dumper = GraphDumper.init(testing.allocator, .dot);

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(testing.allocator);
    try dumper.dumpToArrayList(graph, &buf, testing.allocator);
    const output = buf.items;

    // 验证 DOT 格式
    try testing.expect(std.mem.indexOf(u8, output, "digraph G") != null);
}

test "GraphDumper dumpJson empty graph" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    var dumper = GraphDumper.init(testing.allocator, .json);

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(testing.allocator);
    try dumper.dumpToArrayList(graph, &buf, testing.allocator);
    const output = buf.items;

    // 验证 JSON 格式
    try testing.expect(std.mem.indexOf(u8, output, "\"n_nodes\"") != null);
}

test "GraphDumper dumpText with nodes" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);

    // 添加一些节点
    ctx.setNoAlloc(false);
    const a = try ctx.newTensor1d(.f32, 4);
    const b = try ctx.newTensor1d(.f32, 4);
    ctx.setNoAlloc(true);

    const c = ggml.add(ctx, a, b);
    graph.buildForwardExpand(c);

    var dumper = GraphDumper.init(testing.allocator, .text);
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(testing.allocator);
    try dumper.dumpToArrayList(graph, &buf, testing.allocator);
    const output = buf.items;

    // 验证输出包含操作名
    try testing.expect(std.mem.indexOf(u8, output, "ADD") != null);
}

test "OutputFormat enum" {
    try testing.expectEqual(@as(u8, 0), @intFromEnum(OutputFormat.text));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(OutputFormat.dot));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(OutputFormat.json));
}
