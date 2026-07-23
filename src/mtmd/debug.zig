//! Shared encoder debug utilities
//!
//! Extracts the common saveDebugTensors patterns
//! that are duplicated between VisionEncoder and AudioEncoder.

const std = @import("std");
const ggml = @import("ggml");
const mtmd = @import("mm");
const debug_mod = @import("debug");

/// A single debug tensor entry: tensor name in graph → output filename.
/// `is_input` indicates whether this tensor is an input tensor (data set via setDataPtr)
/// or an output tensor (computed by the graph). Input tensors should not be marked
/// with setOutput, and their data type may differ from f32.
pub const DebugTensorEntry = struct {
    /// Name set via ggml_set_name() in the graph
    tensor_name: []const u8,
    /// Output filename (relative to subdir)
    filename: []const u8,
    /// Whether this is an input tensor (true) or output tensor (false, default)
    is_input: bool = false,
};

/// Save an input tensor's data to a JSON file.
/// Input tensors may have non-f32 types (e.g., i32 for position indices).
/// This function reads the raw data and converts to f32 for JSON output.
fn saveInputTensor(
    io: std.Io,
    allocator: std.mem.Allocator,
    subdir: []const u8,
    fname: []const u8,
    tensor_name: []const u8,
    cgraph: *ggml.CGraph,
    log: anytype,
) void {
    const c = @import("ggml").c;

    var name_buf: [256]u8 = undefined;
    if (tensor_name.len >= name_buf.len) {
        log.warn("saveInputTensor: tensor name too long ({d} >= {d})", .{ tensor_name.len, name_buf.len });
        return;
    }
    @memcpy(name_buf[0..tensor_name.len], tensor_name);
    name_buf[tensor_name.len] = 0;
    const t = c.ggml_graph_get_tensor(@ptrCast(cgraph), &name_buf);
    if (t == null) {
        log.warn("saveInputTensor: tensor '{s}' not found in graph", .{tensor_name});
        return;
    }

    const tensor = @as(*ggml.Tensor, @ptrCast(t));
    const nelements: usize = @intCast(tensor.nElems());
    const dtype = tensor.dataType();

    // Allocate buffer for f32 conversion
    const f32_data = allocator.alloc(f32, nelements) catch |err| {
        log.warn("saveInputTensor: failed to allocate f32 buffer: {}", .{err});
        return;
    };
    defer allocator.free(f32_data);

    // Read data based on type
    if (dtype == .i32) {
        const i32_data = tensor.dataGet(i32, allocator) catch |err| {
            log.warn("saveInputTensor: failed to read i32 data: {}", .{err});
            return;
        };
        defer allocator.free(i32_data);
        for (i32_data, 0..) |val, i| {
            f32_data[i] = @floatFromInt(val);
        }
    } else if (dtype == .f32) {
        const data = tensor.dataGet(f32, allocator) catch |err| {
            log.warn("saveInputTensor: failed to read f32 data: {}", .{err});
            return;
        };
        defer allocator.free(data);
        @memcpy(f32_data, data);
    } else {
        log.warn("saveInputTensor: unsupported type for '{s}': {}", .{ tensor_name, dtype });
        return;
    }

    debug_mod.saveData(io, subdir, fname, tensor_name, f32_data) catch |err| {
        log.warn("saveInputTensor: failed to save data: {}", .{err});
    };
}

/// Save InputChunks to a JSON file.
///
/// Parameters:
///   - io: I/O instance
///   - allocator: allocator for temporary strings
///   - subdir: subdirectory (e.g. "debug_vision"), null for current directory
///   - fname: output filename (e.g. "mtmd_input_chunks.txt")
///   - input_text: the original input text (may be null/empty)
///   - chunks: the parsed InputChunks to serialize
pub fn saveInputChunks(
    io: std.Io,
    allocator: std.mem.Allocator,
    subdir: ?[]const u8,
    fname: []const u8,
    input_text: ?[]const u8,
    chunks: *const mtmd.InputChunks,
) !void {
    const cwd = std.Io.Dir.cwd();

    // Open or create the output file
    const file = if (subdir) |sd| blk: {
        cwd.createDirPath(io, sd) catch {};
        const dir = try cwd.openDir(io, sd, .{});
        defer dir.close(io);
        log.info("Save mtmd_input_chunks to {s}/{s}", .{ sd, fname });
        break :blk try dir.createFile(io, fname, .{});
    } else blk: {
        log.info("Save mtmd_input_chunks to {s}", .{fname});
        break :blk try cwd.createFile(io, fname, .{});
    };
    defer file.close(io);

    // Build the JSON content in memory first for cleaner output
    var buf = std.ArrayList(u8).initCapacity(allocator, 4096) catch @panic("OOM");
    defer buf.deinit(allocator);

    try buf.ensureUnusedCapacity(allocator, 4096);

    try appendAll(&buf, allocator, "{\n");
    try appendAll(&buf, allocator, "  \"version\": 1,\n");

    // Write input_text (JSON-escaped)
    try appendAll(&buf, allocator, "  \"input_text\": ");
    if (input_text) |text| {
        try appendJsonString(&buf, allocator, text);
    } else {
        try appendAll(&buf, allocator, "\"\"");
    }
    try appendAll(&buf, allocator, ",\n");

    try appendAll(&buf, allocator, "  \"chunks\": [\n");

    for (chunks.entries.items, 0..) |*chunk, i| {
        if (i > 0) {
            try appendAll(&buf, allocator, ",\n");
        }
        try appendAll(&buf, allocator, "    {\n");
        try appendAll(&buf, allocator, "      \"type\": ");

        switch (chunk.chunk_type) {
            .text => {
                try appendAll(&buf, allocator, "\"text\",\n");
                const tokens = chunk.tokens_text orelse &.{};
                try appendFmt(&buf, allocator, "      \"n_tokens\": {d},\n", .{tokens.len});
                try appendAll(&buf, allocator, "      \"tokens\": [");
                for (tokens, 0..) |tok, j| {
                    if (j > 0) try appendAll(&buf, allocator, ", ");
                    try appendFmt(&buf, allocator, "{d}", .{tok});
                }
                try appendAll(&buf, allocator, "]\n");
            },
            .image => {
                try appendAll(&buf, allocator, "\"image\",\n");
                if (chunk.tokens_image) |*img| {
                    try appendFmt(&buf, allocator, "      \"n_tokens\": {d},\n", .{img.nTokens()});
                    try appendFmt(&buf, allocator, "      \"nx\": {d},\n", .{img.nx});
                    try appendFmt(&buf, allocator, "      \"ny\": {d},\n", .{img.ny});
                    try appendFmt(&buf, allocator, "      \"n_temporal_merge\": {d},\n", .{img.n_temporal_merge});
                    try appendAll(&buf, allocator, "      \"pos_type\": ");
                    switch (img.pos) {
                        .normal => try appendAll(&buf, allocator, "\"normal\""),
                        .mrope => try appendAll(&buf, allocator, "\"mrope\""),
                        .hunyuanvl => try appendAll(&buf, allocator, "\"hunyuanvl\""),
                    }
                    if (img.id) |id_str| {
                        try appendAll(&buf, allocator, ",\n");
                        try appendAll(&buf, allocator, "      \"id\": ");
                        try appendJsonString(&buf, allocator, id_str);
                    }
                    try appendAll(&buf, allocator, "\n");
                } else {
                    try appendAll(&buf, allocator, "      \"n_tokens\": 0\n");
                }
            },
            .audio => {
                try appendAll(&buf, allocator, "\"audio\",\n");
                try appendFmt(&buf, allocator, "      \"n_tokens\": {d},\n", .{chunk.tokens_audio_n});
                // n_samples: number of Mel spectrogram frames (if mel_data available)
                if (chunk.mel_data) |mel| {
                    try appendFmt(&buf, allocator, "      \"n_samples\": {d}\n", .{mel.len / @max(chunk.mel_bins, 1)});
                } else {
                    try appendAll(&buf, allocator, "      \"n_samples\": 0\n");
                }
                if (chunk.id) |id_str| {
                    try appendAll(&buf, allocator, ",\n");
                    try appendAll(&buf, allocator, "      \"id\": ");
                    try appendJsonString(&buf, allocator, id_str);
                    try appendAll(&buf, allocator, "\n");
                }
            },
        }

        try appendAll(&buf, allocator, "    }");
    }

    try appendAll(&buf, allocator, "\n");
    try appendAll(&buf, allocator, "  ]\n");
    try appendAll(&buf, allocator, "}\n");

    // Write the entire buffer to the file
    try file.writeStreamingAll(io, buf.items);
}

/// Append a string to the ArrayList.
fn appendAll(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try buf.ensureUnusedCapacity(allocator, s.len);
    buf.appendSliceAssumeCapacity(s);
}

/// Append a formatted string to the ArrayList.
fn appendFmt(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const formatted = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(formatted);
    try appendAll(buf, allocator, formatted);
}

/// Append a JSON-escaped string (with quotes) to the ArrayList.
fn appendJsonString(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try buf.ensureUnusedCapacity(allocator, s.len * 2 + 2); // worst case: all chars escaped
    buf.appendAssumeCapacity('"');
    for (s) |c| {
        switch (c) {
            '"' => {
                buf.appendAssumeCapacity('\\');
                buf.appendAssumeCapacity('"');
            },
            '\\' => {
                buf.appendAssumeCapacity('\\');
                buf.appendAssumeCapacity('\\');
            },
            '\n' => {
                buf.appendAssumeCapacity('\\');
                buf.appendAssumeCapacity('n');
            },
            '\r' => {
                buf.appendAssumeCapacity('\\');
                buf.appendAssumeCapacity('r');
            },
            '\t' => {
                buf.appendAssumeCapacity('\\');
                buf.appendAssumeCapacity('t');
            },
            0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0b, 0x0c, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f => {
                // Control characters: \uXXXX
                const escaped = try std.fmt.allocPrint(allocator, "\\u{x:04}", .{@as(u16, c)});
                defer allocator.free(escaped);
                try appendAll(buf, allocator, escaped);
            },
            else => buf.appendAssumeCapacity(c),
        }
    }
    buf.appendAssumeCapacity('"');
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "appendJsonString - basic escaping" {
    var buf = std.ArrayList(u8).initCapacity(testing.allocator, 64) catch @panic("OOM");
    defer buf.deinit(testing.allocator);

    try appendJsonString(&buf, testing.allocator, "hello \"world\"\nline2");
    try testing.expectEqualStrings("\"hello \\\"world\\\"\\nline2\"", buf.items);
}

test "appendJsonString - control characters" {
    var buf = std.ArrayList(u8).initCapacity(testing.allocator, 64) catch @panic("OOM");
    defer buf.deinit(testing.allocator);

    try appendJsonString(&buf, testing.allocator, &[_]u8{ 0x01, 0x02 });
    try testing.expectEqualStrings("\"\\u0001\\u0002\"", buf.items);
}

test "appendJsonString - normal string" {
    var buf = std.ArrayList(u8).initCapacity(testing.allocator, 64) catch @panic("OOM");
    defer buf.deinit(testing.allocator);

    try appendJsonString(&buf, testing.allocator, "simple text");
    try testing.expectEqualStrings("\"simple text\"", buf.items);
}
