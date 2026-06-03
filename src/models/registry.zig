//! 模型注册与工厂函数
//!
//! 根据 GGUF 元数据中的 architecture 字段动态选择模型实现。
//! 使用 Zig 的 switch 实现零成本运行时多态分发。

const std = @import("std");
const ggml = @import("../ggml.zig");
const gguf = @import("../gguf.zig");
const kv_cache = @import("../kv_cache.zig");
const model = @import("../model.zig");
const qwen = @import("qwen.zig");
const llama = @import("llama.zig");

const log = std.log.scoped(.registry);

/// 根据架构枚举创建模型实例
/// 返回 opaque 指针，通过 forwardModel / deinitModel 操作
pub fn createModel(
    allocator: std.mem.Allocator,
    gguf_file: *gguf.GGUFFile,
    arch: model.Architecture,
    io: std.Io,
) !*anyopaque {
    return switch (arch) {
        .qwen2 => {
            var m = try allocator.create(qwen.QwenModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return @as(*anyopaque, @ptrCast(m));
        },
        .llama => {
            var m = try allocator.create(llama.LlamaModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return @as(*anyopaque, @ptrCast(m));
        },
    };
}

/// 释放模型实例
pub fn deinitModel(model_ptr: *anyopaque, arch: model.Architecture, allocator: std.mem.Allocator) void {
    switch (arch) {
        .qwen2 => {
            var m = @as(*qwen.QwenModel, @ptrCast(@alignCast(model_ptr)));
            m.deinit(allocator);
            allocator.destroy(m);
        },
        .llama => {
            var m = @as(*llama.LlamaModel, @ptrCast(@alignCast(model_ptr)));
            m.deinit(allocator);
            allocator.destroy(m);
        },
    }
}

/// 执行模型前向计算
pub fn forwardModel(
    model_ptr: *anyopaque,
    arch: model.Architecture,
    ctx: *ggml.Context,
    graph: *ggml.CGraph,
    input_tokens: *ggml.Tensor,
    n_tokens: i32,
    kv_cache_mgr: ?*kv_cache.KVCache,
    start_pos: i32,
) !*ggml.Tensor {
    return switch (arch) {
        .qwen2 => {
            const m = @as(*qwen.QwenModel, @ptrCast(@alignCast(model_ptr)));
            return try m.forward(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos);
        },
        .llama => {
            const m = @as(*llama.LlamaModel, @ptrCast(@alignCast(model_ptr)));
            return try m.forward(ctx, graph, input_tokens, n_tokens, kv_cache_mgr, start_pos);
        },
    };
}

/// 获取模型参数
pub fn modelParams(model_ptr: *anyopaque, arch: model.Architecture) *const model.ModelParams {
    return switch (arch) {
        .qwen2 => @as(*qwen.QwenModel, @ptrCast(@alignCast(model_ptr))).getParams(),
        .llama => @as(*llama.LlamaModel, @ptrCast(@alignCast(model_ptr))).getParams(),
    };
}

/// 从 GGUF 元数据检测架构
pub fn detectArchitecture(gguf_file: *const gguf.GGUFFile) ?model.Architecture {
    // 尝试多个可能的 key
    const arch_names = [_][]const u8{
        "general.architecture",
        "llama.architecture",
        "qwen35.architecture",
        "model.architecture",
    };

    for (arch_names) |key| {
        if (gguf_file.getString(key)) |arch_str| {
            if (model.Architecture.fromString(arch_str)) |arch| {
                log.info("Detected architecture: {s} (from '{s}')", .{ @tagName(arch), arch_str });
                return arch;
            }
            log.warn("Unknown architecture: '{s}' from key '{s}'", .{ arch_str, key });
        }
    }

    // 回退：通过检查特定张量名称来推断
    if (gguf_file.findTensor("blk.0.attn_q.weight") != null) {
        log.info("Fallback: detected qwen2 architecture (found attn_q.weight)", .{});
        return .qwen2;
    }
    if (gguf_file.findTensor("blk.0.attn_qkv.weight") != null) {
        log.info("Fallback: detected qwen2 architecture (found attn_qkv.weight)", .{});
        return .qwen2;
    }

    log.err("Could not detect model architecture from GGUF metadata", .{});
    return null;
}

const testing = std.testing;

test "detectArchitecture" {
    // 单元测试只验证枚举
    try testing.expectEqual(model.Architecture.qwen2, model.Architecture.fromString("qwen2").?);
    try testing.expectEqual(model.Architecture.llama, model.Architecture.fromString("llama3").?);
}
