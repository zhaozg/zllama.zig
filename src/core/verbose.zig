//! Verbose prompt printing utilities.
//!
//! Extracted from engine.zig (refact.md §1) to keep files ≤600 lines.
//! Matches llama.cpp's --verbose-prompt output format.

const std = @import("std");
const tokenizer = @import("tokenizer");
const chat_template = @import("chat_template");

const logger = std.log.scoped(.verbose);

/// Print verbose prompt information: prompt text, token IDs, decoded text, and logits preview.
/// Matches llama.cpp's --verbose-prompt output format.
/// If `media_offsets` is provided, media placeholder regions in the token sequence
/// are displayed as `<__media_<type>_<count>tokens__>` instead of repeating the same token.
pub fn printVerbosePrompt(
    io: std.Io,
    allocator: std.mem.Allocator,
    tok: *tokenizer.Tokenizer,
    prompt_text: ?[]const u8,
    input_tokens: []const u32,
    logits: []const f32,
    media_offsets: ?[]const chat_template.PlaceholderInfo,
) !void {
    const stderr_file = std.Io.File.stderr();

    // Use a local buffer for formatted output
    var line_buf: [4096]u8 = undefined;

    // Print header
    const header = "\n========== Verbose Prompt ==========\n";
    _ = try stderr_file.writeStreamingAll(io, header);

    // Print original prompt text (like llama.cpp: LOG_INF("%s: prompt: '%s'\n", ...))
    if (prompt_text) |text| {
        const prompt_line = try std.fmt.bufPrint(&line_buf, "prompt: '{s}'\n", .{text});
        _ = try stderr_file.writeStreamingAll(io, prompt_line);
    }

    // Print token count (like llama.cpp: "number of tokens in prompt = %zu")
    {
        var count_buf: [128]u8 = undefined;
        const count_line = try std.fmt.bufPrint(&count_buf, "number of tokens in prompt = {d}\n", .{input_tokens.len});
        _ = try stderr_file.writeStreamingAll(io, count_line);
    }

    // Print each prompt token with ID and decoded text
    if (media_offsets) |offsets| {
        var off_idx: usize = 0;
        var i: usize = 0;
        while (i < input_tokens.len) : (i += 1) {
            if (off_idx < offsets.len and i == offsets[off_idx].token_offset) {
                const info = offsets[off_idx];
                const media_label = switch (info.media_type) {
                    .image => "image",
                    .audio => "audio",
                };
                const line = try std.fmt.bufPrint(&line_buf, "{d:6}: token {d:6} -> '<__media_{s}_{d}tokens__>'  (placeholder, {d} tokens)\n", .{
                    i, input_tokens[i], media_label, info.token_count, info.token_count,
                });
                _ = try stderr_file.writeStreamingAll(io, line);
                i += info.token_count - 1;
                off_idx += 1;
            } else {
                var dec_buf: [128]u8 = undefined;
                const n = try tok.decodeSingle(input_tokens[i], &dec_buf);
                const display = if (n > 0) dec_buf[0..n] else blk: {
                    if (tok.vocab.tokenText(input_tokens[i])) |text| {
                        break :blk text;
                    }
                    break :blk "(special)";
                };
                const line = try std.fmt.bufPrint(&line_buf, "{d:6} -> '{s}'\n", .{ input_tokens[i], display });
                _ = try stderr_file.writeStreamingAll(io, line);
            }
        }
    } else {
        for (input_tokens) |token_id| {
            var dec_buf: [128]u8 = undefined;
            const n = try tok.decodeSingle(token_id, &dec_buf);
            const display = if (n > 0) dec_buf[0..n] else blk: {
                if (tok.vocab.tokenText(token_id)) |text| {
                    break :blk text;
                }
                break :blk "(special)";
            };
            const line = try std.fmt.bufPrint(&line_buf, "{d:6} -> '{s}'\n", .{ token_id, display });
            _ = try stderr_file.writeStreamingAll(io, line);
        }
    }

    // Print logits preview (top-10 values) — zllama-specific enhancement
    {
        const logits_header = "\n--- Logits preview (last token, top-10) ---\n";
        _ = try stderr_file.writeStreamingAll(io, logits_header);
    }

    // Find top-k logits
    const top_k: usize = @min(10, logits.len);
    var indices = try allocator.alloc(usize, logits.len);
    defer allocator.free(indices);
    for (0..logits.len) |i| indices[i] = i;

    // Partial sort: find top-k
    std.mem.sort(usize, indices, logits, struct {
        fn lessThan(ctx: []const f32, a: usize, b: usize) bool {
            return ctx[a] > ctx[b];
        }
    }.lessThan);

    for (0..top_k) |i| {
        const idx = indices[i];
        var dec_buf: [128]u8 = undefined;
        const n = try tok.decodeSingle(@intCast(idx), &dec_buf);
        const display = if (n > 0) dec_buf[0..n] else blk: {
            if (tok.vocab.tokenText(@intCast(idx))) |text| {
                break :blk text;
            }
            break :blk "(special)";
        };
        const line = try std.fmt.bufPrint(&line_buf, "  {d:6}: token {d:6} -> '{s}' (logit={d:.4})\n", .{ i, idx, display, logits[idx] });
        _ = try stderr_file.writeStreamingAll(io, line);
    }

    const footer = "========================================\n\n";
    _ = try stderr_file.writeStreamingAll(io, footer);
}
