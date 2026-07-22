//! Save mtmd InputChunks to a JSON file for debugging/comparison.
//!
//! Mirrors the C++ function `mtmd_debug_save_chunks` in
//! `deps/llama.cpp/tools/mtmd/mtmd.cpp` (line 2308).
//!
//! The output format is identical to the C++ version so that
//! `debug_vision/mtmd_input_chunks.txt` can be compared directly.
//!
//! Usage:
//! ```zig
//! try debug_save_chunks.saveInputChunks(io, allocator, "debug_vision", "mtmd_input_chunks.txt", input_text, &chunks);
//! ```

const std = @import("std");
const mtmd = @import("mm");

const log = std.log.scoped(.mtmd_debug);

/// Save InputChunks to a JSON file, matching the C++ `mtmd_debug_save_chunks` format.
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
