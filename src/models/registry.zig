//! 模型注册与工厂函数
//!
//! 根据 GGUF 元数据中的 architecture 字段动态选择模型实现。
//! 使用 ModelInstance（虚表模式）实现运行时多态分发。
//!
//! 参考 llama.cpp 的 llama_model_mapping 设计。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const qwen2 = @import("model").qwen2;
const qwen35 = @import("model").qwen35;
const llama = @import("model").llama;

const log = std.log.scoped(.registry);

/// 根据架构枚举创建模型实例
/// 返回 ModelInstance（虚表包装），调用者无需知道具体模型类型
pub fn createModel(
    allocator: std.mem.Allocator,
    gguf_file: *gguf.GGUFFile,
    arch: model_if.Architecture,
    io: std.Io,
) !model_if.ModelInstance {
    return switch (arch) {
        .qwen2 => {
            var m = try allocator.create(qwen2.Qwen2Model);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &qwen2.Qwen2Model.vtable,
                .ptr = @as(*anyopaque, @ptrCast(m)),
            };
        },
        .qwen35 => {
            var m = try allocator.create(qwen35.QwenModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &qwen35.QwenModel.vtable,
                .ptr = @as(*anyopaque, @ptrCast(m)),
            };
        },
        .llama => {
            var m = try allocator.create(llama.LlamaModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &llama.LlamaModel.vtable,
                .ptr = @as(*anyopaque, @ptrCast(m)),
            };
        },
    };
}

/// 从 GGUF 元数据检测架构
pub fn detectArchitecture(gguf_file: *const gguf.GGUFFile) ?model_if.Architecture {
    const arch_names = [_][]const u8{
        "general.architecture",
        "llama.architecture",
        "qwen35.architecture",
        "model.architecture",
    };

    for (arch_names) |key| {
        if (gguf_file.getString(key)) |arch_str| {
            if (model_if.Architecture.fromString(arch_str)) |arch| {
                log.info("Detected architecture: {s} (from '{s}')", .{ @tagName(arch), arch_str });
                return arch;
            }
            log.warn("Unknown architecture: '{s}' from key '{s}'", .{ arch_str, key });
        }
    }

    // Fallback: detect by tensor names
    if (gguf_file.findTensor("blk.0.ssm_conv1d.weight") != null) {
        log.info("Fallback: detected qwen35 architecture (found ssm_conv1d.weight)", .{});
        return .qwen35;
    }
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
    try testing.expectEqual(model_if.Architecture.qwen2, model_if.Architecture.fromString("qwen2").?);
    try testing.expectEqual(model_if.Architecture.qwen35, model_if.Architecture.fromString("qwen35").?);
    try testing.expectEqual(model_if.Architecture.llama, model_if.Architecture.fromString("llama3").?);
}
