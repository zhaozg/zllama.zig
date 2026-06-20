// Auto-generated from encode.zig - qwen pre-tokenizer
// Original lines 1511-1513

const std = @import("std");
const mod = @import("../mod.zig");
const types = mod.types;
const unicode = mod.unicode;
const PreTokenized = mod.PreTokenized;
const preTokenizeGPT2 = @import("gpt2.zig").preTokenizeGPT2;

pub fn preTokenizeQwen(text: []const u8, result: *PreTokenized) !void {
    try preTokenizeGPT2(text, result);
}
