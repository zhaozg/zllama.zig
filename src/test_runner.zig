//! zllama.zig 测试入口
//!
//! 导入所有测试模块，通过 `zig build test` 运行。
//! 所有测试文件放在 src/tests/ 目录下。

comptime {
    // 导入所有测试模块
    _ = @import("ggml");
    _ = @import("gguf");
    _ = @import("model");
    _ = @import("registry");
    _ = @import("graph_builder");
    _ = @import("memory");
    _ = @import("tokenizer");
    _ = @import("sampler");
    _ = @import("kv_cache");
    _ = @import("graph_context");
    _ = @import("mtmd");
    _ = @import("preprocess");
    _ = @import("engine_common");
    _ = @import("prefill");
    _ = @import("chat_template");
    _ = @import("rms_norm");
    _ = @import("rope");
    _ = @import("attention");
    _ = @import("swiglu");
    _ = @import("embed");
    _ = @import("pooling");
    _ = @import("weight_loader");
    _ = @import("stb_image");
    _ = @import("utils");
    _ = @import("test_utils");
    _ = @import("tests/test_layers.zig");
    _ = @import("tests/test_gguf.zig");
    _ = @import("tests/test_ggml_gguf.zig");
    _ = @import("tests/test_archs.zig");
    _ = @import("tests/test_kv_cache.zig");
    _ = @import("tests/test_compare_logits.zig");
    _ = @import("tests/test_vocab.zig");
    _ = @import("tests/test_embed.zig");
    _ = @import("tests/test_mtmd.zig");
    _ = @import("tests/test_audio.zig");
    _ = @import("tests/test_vision.zig");
    _ = @import("tests/test_permute.zig");
}
