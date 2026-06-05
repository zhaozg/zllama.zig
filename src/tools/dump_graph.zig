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

    /// 转储计算图到 writer
    pub fn dumpToWriter(
        self: *GraphDumper,
        graph: *ggml.CGraph,
        writer: anytype,
    ) !void {
        switch (self.format) {
            .text => try self.dumpText(graph, writer),
            .dot => try self.dumpDot(graph, writer),
            .json => try self.dumpJson(graph, writer),
        }
    }

    /// 文本格式输出
    fn dumpText(
        _: *GraphDumper,
        graph: *ggml.CGraph,
        writer: anytype,
    ) !void {
        const n_nodes = graph.nNodes();
        try writer.print("=== Graph Dump: {d} nodes ===\n\n", .{n_nodes});

        for (0..@as(usize, @intCast(n_nodes))) |i| {
            const node = graph.getNode(@intCast(i));
            const name = node.getName();
            const op = node.getOpName();
            const n_dims = node.nDims();
            const ne = node.ne();

            try writer.print("[{d:>4}] {s:>20} ", .{ i, op });

            // 输出形状
            try writer.print("(", .{});
            for (0..@as(usize, @intCast(n_dims))) |d| {
                if (d > 0) try writer.print(", ", .{});
                try writer.print("{d}", .{ne[d]});
            }
            try writer.print(")", .{});

            // 输出名称
            if (name.len > 0) {
                try writer.print(" \"{s}\"", .{name});
            }

            try writer.print("\n", .{});
        }
    }

    /// DOT 格式输出（Graphviz）
    fn dumpDot(
        _: *GraphDumper,
        graph: *ggml.CGraph,
        writer: anytype,
    ) !void {
        const n_nodes = graph.nNodes();
        try writer.print("digraph G {\n", .{});
        try writer.print("  rankdir=LR;\n", .{});
        try writer.print("  node [shape=box, style=filled, fillcolor=lightyellow];\n\n", .{});

        for (0..@as(usize, @intCast(n_nodes))) |i| {
            const node = graph.getNode(@intCast(i));
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
            for (1..@as(usize, @intCast(n_dims))) |d| {
                const new_label = try std.fmt.bufPrint(&label_buf, "{s}, {d}", .{ label, ne[d] });
                label = new_label;
            }
            label = try std.fmt.bufPrint(&label_buf, "{s})", .{label});

            try writer.print("  n{d} [label=\"{s}\"];\n", .{ i, label });
        }

        try writer.print("}\n", .{});
    }

    /// JSON 格式输出
    fn dumpJson(
        _: *GraphDumper,
        graph: *ggml.CGraph,
        writer: anytype,
    ) !void {
        const n_nodes = graph.nNodes();
        try writer.print("{\n", .{});
        try writer.print("  \"n_nodes\": {d},\n", .{n_nodes});
        try writer.print("  \"nodes\": [\n", .{});

        for (0..@as(usize, @intCast(n_nodes))) |i| {
            const node = graph.getNode(@intCast(i));
            const name = node.getName();
            const op = node.getOpName();
            const n_dims = node.nDims();
            const ne = node.ne();

            try writer.print("    {{\n", .{});
            try writer.print("      \"index\": {d},\n", .{i});
            try writer.print("      \"op\": \"{s}\",\n", .{op});
            try writer.print("      \"name\": \"{s}\",\n", .{name});
            try writer.print("      \"n_dims\": {d},\n", .{n_dims});
            try writer.print("      \"shape\": [", .{});
            for (0..@as(usize, @intCast(n_dims))) |d| {
                if (d > 0) try writer.print(", ", .{});
                try writer.print("{d}", .{ne[d]});
            }
            try writer.print("]\n", .{});

            if (i + 1 < n_nodes) {
                try writer.print("    }},\n", .{});
            } else {
                try writer.print("    }}\n", .{});
            }
        }

        try writer.print("  ]\n", .{});
        try writer.print("}}\n", .{});
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
    _ = init;
    std.debug.print("zllama-dump-graph: Dump computation graph\n", .{});
    std.debug.print("Usage: zllama-dump-graph <model.gguf>\n", .{});
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

    var buf = std.ArrayList(u8).init(testing.allocator);
    defer buf.deinit();

    try dumper.dumpToWriter(graph, buf.writer());
    const output = buf.items;

    // 验证输出包含节点数
    try testing.expect(std.mem.indexOf(u8, output, "0 nodes") != null);
}

test "GraphDumper dumpDot empty graph" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    var dumper = GraphDumper.init(testing.allocator, .dot);

    var buf = std.ArrayList(u8).init(testing.allocator);
    defer buf.deinit();

    try dumper.dumpToWriter(graph, buf.writer());
    const output = buf.items;

    // 验证 DOT 格式
    try testing.expect(std.mem.indexOf(u8, output, "digraph G") != null);
}

test "GraphDumper dumpJson empty graph" {
    const ctx = try ggml.Context.initNoAlloc(64 * 1024);
    defer ctx.deinit();

    const graph = try ggml.CGraph.init(ctx);
    var dumper = GraphDumper.init(testing.allocator, .json);

    var buf = std.ArrayList(u8).init(testing.allocator);
    defer buf.deinit();

    try dumper.dumpToWriter(graph, buf.writer());
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
    var buf = std.ArrayList(u8).init(testing.allocator);
    defer buf.deinit();

    try dumper.dumpToWriter(graph, buf.writer());
    const output = buf.items;

    // 验证输出包含操作名
    try testing.expect(std.mem.indexOf(u8, output, "ADD") != null);
}

test "OutputFormat enum" {
    try testing.expectEqual(@as(u8, 0), @intFromEnum(OutputFormat.text));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(OutputFormat.dot));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(OutputFormat.json));
}
