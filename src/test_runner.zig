//! zllama.zig 测试入口
//!
//! 导入所有测试模块，通过 `zig build test` 运行。
//! 所有测试文件放在 src/tests/ 目录下。

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const model = @import("model");
const registry = @import("registry");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler");
const kv_cache = @import("kv_cache");
const graph_context = @import("graph_context");
const mm = @import("mm");
const preprocess = @import("preprocess");
const engine_common = @import("engine_common");
const prefill = @import("prefill");
const chat_template = @import("chat_template");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const attention = @import("attention");
const swiglu = @import("swiglu");
const embed = @import("embed");
const pooling = @import("pooling");
const weight_loader = @import("weight_loader");
const stb_image = @import("stb_image");
const utils = @import("utils");

// 导入所有测试模块
const test_utils = @import("tests/utils.zig");
const test_layers = @import("tests/test_layers.zig");
const test_gguf = @import("tests/test_gguf.zig");
const test_archs = @import("tests/test_archs.zig");
const test_kv_cache = @import("tests/test_kv_cache.zig");
const test_compare_logits = @import("tests/test_compare_logits.zig");
const test_vocab = @import("tests/test_vocab.zig");
const test_embed = @import("tests/test_embed.zig");
const test_mtmd = @import("tests/test_mtmd.zig");
