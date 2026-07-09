//! zllama.zig 入口
//!
//! zllama.zig - 多模型本地推理引擎 - 主入口点
//! 处理 CLI 参数、初始化、推理循环
//! 支持多模型架构（Qwen / LLaMA 等）
//! 实现首 token 完整图推理 + 增量解码
//!
//! 解码流程：
//! 1. 编码 prompt -> token IDs（不自动添加 BOS，由模型内部处理）
//! 2. 检测模型架构（从 GGUF 元数据）
//! 3. 首 token 完整前向计算（填充 KV Cache）
//! 4. 增量生成后续 tokens（每次 1 个 token）
//! 5. 所有 token 收集后一次性解码，确保 UTF-8 字节序列正确组合
//! 6. 输出完整文本

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model_if = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const graph_context = @import("graph_context");
const memory = @import("memory");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const kv_cache = @import("kv_cache");
const mm = @import("mtmd");
const preprocess = @import("preprocess");
const engine_common = @import("engine_common");
const prefill_mod = @import("prefill");
const chat_template = @import("chat_template");

const CliArgs = @import("cli_args.zig").CliArgs;
const InferenceEngine = @import("core/engine.zig").InferenceEngine;
const loadMMProj = @import("core/loader.zig").loadMMProj;

pub const std_options: std.Options = .{ .log_level = .info, .logFn = engine_common.logFilter, .log_scope_levels = &[_]std.log.ScopeLevel{
    .{ .scope = .tokenizer, .level = .info },
    .{ .scope = .ggml, .level = .info },
    .{ .scope = .qwen, .level = .info },
    .{ .scope = .llama, .level = .info },
    .{ .scope = .model, .level = .info },
    .{ .scope = .main, .level = .info },
    .{ .scope = .engine, .level = .info },
    .{ .scope = .gemma4, .level = .info },
    .{ .scope = .mtmd, .level = .info },
    .{ .scope = .prefill, .level = .debug },
    .{ .scope = .multimodal, .level = .debug },
    .{ .scope = .audio_encoder, .level = .debug },
    .{ .scope = .audio_pipeline, .level = .info },
    .{ .scope = .vision_encoder, .level = .info },
    .{ .scope = .vision_pipeline, .level = .info },

    .{ .scope = .weight_loader, .level = .info },
    .{ .scope = .gemma4a, .level = .debug },
} };

const logger = std.log.scoped(.main);

pub fn main(init: std.process.Init) !void {
    // Set ggml log callback
    ggml.logSet();
    const io = init.io;
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    const args = CliArgs.parse(&args_iter) catch |err| {
        if (err == error.InvalidArgs) {
            CliArgs.printHelp();
            return;
        }
        return err;
    };

    if (args.help) {
        CliArgs.printHelp();
        return;
    }
    if (args.debug) {
        engine_common.setLogLevel(.debug);
    } else if (args.verbose) {
        engine_common.setLogLevel(.info);
    } else {
        engine_common.setLogLevel(.warn);
    }
    logger.info("zllama.zig v0.1.0 (ggml {s})", .{ggml.version()});

    if (args.model_path.len == 0) {
        logger.err("no model specified. Use --model <path>", .{});
        return;
    }

    logger.info("Loading model: {s}", .{args.model_path});
    var engine = InferenceEngine.init(io, allocator, args.model_path, &args) catch |err| {
        logger.err("Failed to initialize inference engine: {}\n", .{err});
        return;
    };
    defer engine.deinit();

    logger.info("Model loaded successfully.", .{});
    logger.info("Prompt: \"{s}\"", .{args.prompt});
    logger.info("Max tokens: {d}", .{args.max_tokens});

    if (args.embed) {
        logger.info("--- Embedding Generation ---", .{});
        const emb = engine.generateEmbedding(io, args.prompt) catch |err| {
            logger.err("Embedding generation failed: {}", .{err});
            return;
        };
        defer allocator.free(emb);

        logger.info("--- Done (dims={d}) ---", .{emb.len});
    } else if (args.chat) {
        try engine.chatLoop(io);
    } else if (args.image_path.len > 0) {
        logger.info("--- Vision Generation ---", .{});
        engine.generateWithImage(io, args.prompt, args.image_path, args.max_tokens) catch |err| {
            logger.err("Vision generation failed: {}", .{err});
            return;
        };
        logger.info("--- Done ---", .{});
    } else if (args.audio_path.len > 0) {
        logger.info("--- Audio Generation ---", .{});
        engine.generateWithAudio(io, args.prompt, args.audio_path, args.max_tokens) catch |err| {
            logger.err("Audio generation failed: {}", .{err});
            return;
        };
        logger.info("--- Done ---", .{});
    } else {
        logger.info("--- Generation ---", .{});
        engine.generate(io, args.prompt, args.max_tokens) catch |err| {
            logger.err("Generation failed: {}", .{err});
            return;
        };
        logger.info("--- Done ---", .{});
    }
}

// 导入所有测试模块（通过 zig build test 运行）

const test_utils = @import("tests/utils.zig");
const test_layers = @import("tests/test_layers.zig");
const test_gguf = @import("tests/test_gguf.zig");
const test_archs = @import("tests/test_archs.zig");
const test_kv_cache = @import("tests/test_kv_cache.zig");
const test_compare_logits = @import("tests/test_compare_logits.zig");
const test_vocab = @import("tests/test_vocab.zig");
const test_embed = @import("tests/test_embed.zig");
