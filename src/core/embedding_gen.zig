//! Embedding generation — produces a pooled embedding vector for the input prompt.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.

const std = @import("std");
const ggml = @import("ggml");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const tokenizer = @import("tokenizer");

const logger = std.log.scoped(.embedding);

/// Generate an embedding vector for the given text prompt.
/// Returns a heap-allocated f32 slice owned by the caller.
pub fn generateEmbedding(
    allocator: std.mem.Allocator,
    io: std.Io,
    model: model_if.ModelInstance,
    params: *const model_if.ModelParams,
    tok: *tokenizer.Tokenizer,
    ctx_graph: *ggml.Context,
    n_threads: i32,
    prompt: []const u8,
) ![]f32 {
    var input_tokens = try tok.encode(prompt, true, true);
    defer input_tokens.deinit(allocator);
    const n_tokens: i32 = @intCast(input_tokens.items.len);
    if (n_tokens == 0) return error.EmptyInput;

    ctx_graph.setNoAlloc(false);
    const input_tensor = try ctx_graph.newTensor1d(.i32, n_tokens);
    ctx_graph.setNoAlloc(true);
    {
        const data = input_tensor.dataBytes();
        const slice = @as([*]i32, @ptrCast(@alignCast(data.ptr)))[0..@as(usize, @intCast(n_tokens))];
        for (input_tokens.items, 0..) |token, j| slice[j] = @as(i32, @intCast(token));
    }

    var graph = try ggml.CGraph.initReserved(ctx_graph, 16384);
    var builder = graph_builder.GraphBuilder.init(ctx_graph, graph, params, allocator);
    const embedding_vector = try model.buildGraph(&builder, input_tensor, n_tokens, null, 0);

    const buft = ggml.backendCpuBufferType();
    var galloc = try ggml.Gallocr.init(buft);
    defer galloc.free();
    if (!galloc.allocGraph(graph)) return error.GraphAllocFailed;
    try graph.compute(n_threads);

    const n_embd = @as(usize, @intCast(params.n_embd));
    const result = try allocator.alloc(f32, n_embd);
    {
        const result_data = try embedding_vector.dataGet(f32, allocator);
        defer allocator.free(result_data);
        @memcpy(result, result_data[0..n_embd]);
    }

    const stdout_file = std.Io.File.stdout();
    var buf: [128]u8 = undefined;
    for (result) |v| {
        const line = try std.fmt.bufPrint(&buf, "{d:.6}\n", .{v});
        try stdout_file.writeStreamingAll(io, line);
    }
    return result;
}
