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
const qwen3vl = @import("model").qwen3vl;
const llama = @import("model").llama;
const gemma3 = @import("model").gemma3;
const gemma4 = @import("model").gemma4;
const embedding = @import("model").embedding;

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
            var m = try allocator.create(embedding.EmbeddingModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &embedding.EmbeddingModel.vtable,
                .ptr = @as(*anyopaque, @ptrCast(m)),
            };
        },
        .qwen3vl => {
            var m = try allocator.create(qwen3vl.Qwen3VLModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &qwen3vl.Qwen3VLModel.vtable,
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
        .gemma3 => {
            var m = try allocator.create(gemma3.Gemma3Model);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &gemma3.Gemma3Model.vtable,
                .ptr = @as(*anyopaque, @ptrCast(m)),
            };
        },
        .gemma4 => {
            var m = try allocator.create(gemma4.Gemma4Model);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &gemma4.Gemma4Model.vtable,
                .ptr = @as(*anyopaque, @ptrCast(m)),
            };
        },
        .embedding_qwen2 => {
            // Embedding model: reuses Qwen2 architecture with pooling
            var m = try allocator.create(embedding.EmbeddingModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &embedding.EmbeddingModel.vtable,
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

    var arch: ?model_if.Architecture = null;

    for (arch_names) |key| {
        if (gguf_file.getString(key)) |arch_str| {
            if (model_if.Architecture.fromString(arch_str)) |a| {
                arch = a;
                break;
            }
            log.debug("Unknown architecture: '{s}' from key '{s}'", .{ arch_str, key });
        }
    }

    // Fallback: detect by tensor names
    if (arch == null) {
        if (gguf_file.findTensor("blk.0.ssm_conv1d.weight") != null) {
            log.info("Fallback: detected qwen35 architecture (found ssm_conv1d.weight)", .{});
            arch = .qwen35;
        } else if (gguf_file.findTensor("blk.0.attn_q.weight") != null) {
            log.info("Fallback: detected qwen2 architecture (found attn_q.weight)", .{});
            arch = .qwen2;
        } else if (gguf_file.findTensor("blk.0.attn_qkv.weight") != null) {
            log.info("Fallback: detected qwen2 architecture (found attn_qkv.weight)", .{});
            arch = .qwen2;
        }
    }
    // Check if it's an embedding model: pooling_type metadata distinguishes
    // embedding models from generative models.
    // Check multiple possible keys: general.pooling_type (string), qwen3.pooling_type (u32),
    // qwen2.pooling_type (u32)
    if (arch) |a| {
        if (a == .qwen2 and
            (gguf_file.getString("general.pooling_type") != null or
                gguf_file.getU32("qwen3.pooling_type") != null or
                gguf_file.getU32("qwen2.pooling_type") != null))
        {
            log.info("Detected embedding model (pooling_type present), architecture: embedding_qwen2", .{});
            return .embedding_qwen2;
        }
        log.info("Detected architecture: {s}", .{@tagName(a)});
        return a;
    }

    log.debug("Could not detect model architecture from GGUF metadata", .{});
    return null;
}
/// 检测模型的多模态能力
/// 通过检查 GGUF 文件中的张量和元数据来判断模型是否支持视觉/音频输入
pub fn detectCapabilities(gguf_file: *const gguf.GGUFFile, arch: model_if.Architecture) model_if.ModelCapabilities {
    var caps = model_if.ModelCapabilities{};

    switch (arch) {
        .qwen3vl => {
            // Qwen3VL: vision capabilities detected from mmproj file
            // The main model GGUF doesn't have vision tensors
        },
        .gemma4 => {
            // Gemma 4 E2B: 检查音频编码器（Conformer）张量
            // 张量命名遵循 llama.cpp clip-impl.h 的约定:
            // TN_A_CONV1D = "a.conv1d.%d.%s"
            // TN_A_INP_PROJ = "a.input_projection.%s"
            // TN_A_OUT_PROJ = "a.pre_encode.out.%s"
            // TN_A_MM_INP_PROJ = "mm.a.input_projection.%s"
            // TN_A_MM_SOFT_EMB_N = "mm.a.soft_emb_norm.%s"
            if (gguf_file.findTensor("a.conv1d.0.weight") != null or
                gguf_file.findTensor("a.conv1d.1.weight") != null or
                gguf_file.findTensor("a.pre_encode.out.weight") != null or
                gguf_file.findTensor("mm.a.input_projection.weight") != null)
            {
                caps.has_audio = true;
                caps.audio_encoder_type = "Conformer (E2B)";
                caps.audio_sample_rate = 16000;
            }

            // Gemma 4 E2B: 检查视觉编码器张量
            // TN_PATCH_EMBD = "v.patch_embd.weight"
            // TN_POS_EMBD = "%s.position_embd.weight"
            // TN_MM_INP_PROJ = "mm.input_projection.weight"
            // TN_MM_SOFT_EMB_N = "mm.soft_emb_norm.weight"
            if (gguf_file.findTensor("v.patch_embd.weight") != null or
                gguf_file.findTensor("v.position_embd.weight") != null or
                gguf_file.findTensor("mm.input_projection.weight") != null or
                gguf_file.findTensor("mm.soft_emb_norm.weight") != null)
            {
                caps.has_vision = true;
                caps.vision_encoder_type = "ViT (SigLIP/Gemma4V)";
            }

            // 通过 GGUF 元数据进一步确认
            if (gguf_file.getString("gemma4.audio.encoder_type")) |enc_type| {
                caps.has_audio = true;
                caps.audio_encoder_type = enc_type;
            }
            if (gguf_file.getString("gemma4.vision.encoder_type")) |enc_type| {
                caps.has_vision = true;
                caps.vision_encoder_type = enc_type;
            }
        },
        .gemma3 => {
            // Gemma 3 视觉能力
            if (gguf_file.findTensor("mm.input_projection.weight") != null or
                gguf_file.findTensor("mm.soft_emb_norm.weight") != null)
            {
                caps.has_vision = true;
                caps.vision_encoder_type = "SigLIP";
            }
        },
        .llama => {
            // LLaMA 家族多模态变体
            if (gguf_file.findTensor("mm.input_projection.weight") != null or
                gguf_file.findTensor("mm.soft_emb_norm.weight") != null)
            {
                caps.has_vision = true;
                caps.vision_encoder_type = "LLaVA/ViT";
            }
        },
        .qwen2, .qwen35 => {
            // Qwen 家族视觉编码器
            if (gguf_file.findTensor("v.patch_embd.weight") != null or
                gguf_file.findTensor("visual.patch_embeddings.weight") != null)
            {
                caps.has_vision = true;
                caps.vision_encoder_type = "ViT";
            }
            // Qwen 家族音频编码器
            if (gguf_file.findTensor("a.conv1d.0.weight") != null or
                gguf_file.findTensor("a.pre_encode.out.weight") != null)
            {
                caps.has_audio = true;
                caps.audio_encoder_type = "Whisper/Conformer";
                caps.audio_sample_rate = 16000;
            }
        },
        .embedding_qwen2 => {
            // Embedding models: no vision/audio, text-only
        },
    }

    // Detect capabilities through GGUF metadata (separate mmproj files may also contribute)
    // The main model GGUF typically only has text tensors;
    // multimodal encoder weights are loaded from a separate file.
    if (gguf_file.getString("gemma4.audio.encoder_type")) |enc_type| {
        caps.has_audio = true;
        caps.audio_encoder_type = enc_type;
    }
    if (gguf_file.getString("gemma4.vision.encoder_type")) |enc_type| {
        caps.has_vision = true;
        caps.vision_encoder_type = enc_type;
    }

    // Fill special tokens based on architecture
    switch (arch) {
        .gemma4 => {
            caps.special_tokens.img_beg = "<|image>";
            caps.special_tokens.img_end = "<image|>";
            caps.special_tokens.aud_beg = "<|audio>";
            caps.special_tokens.aud_end = "<audio|>";
        },
        .gemma3 => {
            caps.special_tokens.img_beg = "<start_of_image>";
            caps.special_tokens.img_end = "<end_of_image>";
        },
        .qwen3vl => {
            caps.special_tokens.img_beg = "<|vision_start|>";
            caps.special_tokens.img_end = "<|vision_end|>";
        },
        .qwen2 => {
            // Qwen2-VL variant uses qwen2vl vision encoder
            if (std.mem.eql(u8, caps.vision_encoder_type, "qwen2vl")) {
                caps.special_tokens.img_beg = "<|vision_start|>";
                caps.special_tokens.img_end = "<|vision_end|>";
            }
        },
        .llama => {
            caps.special_tokens.img_beg = "<start_of_image>";
            caps.special_tokens.img_end = "<end_of_image>";
        },
        else => {},
    }

    if (caps.has_audio or caps.has_vision) {
        log.info("Multi-modal capabilities detected: audio={}, vision={}", .{ caps.has_audio, caps.has_vision });
    }

    return caps;
}
const testing = std.testing;

test "detectArchitecture" {
    try testing.expectEqual(model_if.Architecture.qwen2, model_if.Architecture.fromString("qwen2").?);
    try testing.expectEqual(model_if.Architecture.qwen35, model_if.Architecture.fromString("qwen35").?);
    try testing.expectEqual(model_if.Architecture.llama, model_if.Architecture.fromString("llama3").?);
}
