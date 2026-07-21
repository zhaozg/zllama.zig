const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // -Dbundle-ggml: build ggml from source instead of using system-installed libraries
    const bundle_ggml = b.option(bool, "bundle-ggml", "Build ggml from source instead of using system libraries") orelse false;

    // -Dno-galloc-realloc: assert-fail on any gallocr reallocation.
    // Useful for development to detect graph topology changes that would cause
    // runtime reallocations. Not recommended for production.
    const no_galloc_realloc = b.option(bool, "no-galloc-realloc", "Assert-fail on gallocr reallocation (dev only)") orelse false;

    // Build ggml from source when bundling, otherwise nil
    const ggml_lib: ?*std.Build.Step.Compile = if (bundle_ggml)
        buildGgmlFromSource(b, target, optimize)
    else
        null;
    // Build minja from source (C++ bridge around Google minja for Jinja2 templates)
    const minja_lib = buildMinjaFromSource(b, target, optimize);

    // ======================================================================
    // ggml 模块（C 绑定 + 安全封装）
    // ======================================================================
    const ggml_mod = b.createModule(.{
        .root_source_file = b.path("src/ggml/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    if (bundle_ggml) {
        // 从源码构建时：严格使用 deps/ggml 中的头文件，链接自建静态库
        // -I 标志优先级高于系统路径（/usr/local/include），确保头文件不冲突
        ggml_mod.addIncludePath(b.path("deps/ggml/include")); // 公共 API: ggml.h, ggml-cpu.h, ggml-backend.h, ggml-alloc.h, gguf.h
        ggml_mod.addIncludePath(b.path("deps/ggml/src")); // 内部实现: ggml-impl.h, ggml-common.h, ggml-backend-impl.h, ggml-threading.h
        ggml_mod.addIncludePath(b.path("deps/ggml/src/ggml-cpu")); // CPU 后端内部: ggml-cpu-impl.h, traits.h, quants.h 等
        ggml_mod.addCMacro("GGML_USE_CPU", "1");
        if (no_galloc_realloc) {
            ggml_mod.addCMacro("GGML_SCHED_NO_REALLOC", "1");
        }
        ggml_mod.linkLibrary(ggml_lib.?);
    } else {
        // 使用系统安装的 ggml（仅在 -Dbundle-ggml=false 时启用）
        ggml_mod.addIncludePath(.{ .cwd_relative = "/usr/local/include" });
        ggml_mod.linkSystemLibrary("ggml-base", .{});
        ggml_mod.linkSystemLibrary("ggml", .{});
        ggml_mod.linkSystemLibrary("ggml-cpu", .{});
        ggml_mod.linkSystemLibrary("ggml-blas", .{});
        ggml_mod.addRPathSpecial("/usr/local/lib");
        ggml_mod.linkSystemLibrary("omp", .{});
        ggml_mod.linkSystemLibrary("c++", .{});
    }

    // macOS 框架和加速
    if (target.result.os.tag == .macos) {
        ggml_mod.linkFramework("Foundation", .{});
        ggml_mod.linkFramework("Accelerate", .{});
        ggml_mod.addCMacro("GGML_USE_ACCELERATE", "1");
    }

    // ======================================================================
    // 内部模块
    // ======================================================================
    const gguf_mod = b.createModule(.{
        .root_source_file = b.path("src/gguf.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gguf_mod.addImport("ggml", ggml_mod);

    const memory_mod = b.createModule(.{
        .root_source_file = b.path("src/core/memory.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    memory_mod.addImport("ggml", ggml_mod);

    const kv_cache_mod = b.createModule(.{
        .root_source_file = b.path("src/kv_cache.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    kv_cache_mod.addImport("ggml", ggml_mod);
    kv_cache_mod.addImport("memory", memory_mod);

    // --- 层模块（算子库） ---
    const rms_norm_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/rms_norm.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    rms_norm_mod.addImport("ggml", ggml_mod);

    const rope_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/rope.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    rope_mod.addImport("ggml", ggml_mod);

    const swiglu_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/swiglu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    swiglu_mod.addImport("ggml", ggml_mod);

    const attention_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/attention.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    attention_mod.addImport("ggml", ggml_mod);

    const embed_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/embed.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    embed_mod.addImport("ggml", ggml_mod);

    const pooling_mod = b.createModule(.{
        .root_source_file = b.path("src/layers/pooling.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    pooling_mod.addImport("ggml", ggml_mod);

    const weight_loader_mod = b.createModule(.{
        .root_source_file = b.path("src/core/weight_loader.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    weight_loader_mod.addImport("ggml", ggml_mod);
    weight_loader_mod.addImport("gguf", gguf_mod);

    const model_mod = b.createModule(.{
        .root_source_file = b.path("src/model.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    model_mod.addImport("ggml", ggml_mod);
    model_mod.addImport("gguf", gguf_mod);
    model_mod.addImport("memory", memory_mod);
    model_mod.addImport("kv_cache", kv_cache_mod);
    model_mod.addImport("rms_norm", rms_norm_mod);
    model_mod.addImport("rope", rope_mod);
    model_mod.addImport("swiglu", swiglu_mod);
    model_mod.addImport("attention", attention_mod);
    model_mod.addImport("embed", embed_mod);
    model_mod.addImport("pooling", pooling_mod);

    model_mod.addImport("weight_loader", weight_loader_mod);

    const graph_builder_mod = b.createModule(.{
        .root_source_file = b.path("src/core/graph_builder.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    graph_builder_mod.addImport("ggml", ggml_mod);
    graph_builder_mod.addImport("rope", rope_mod);
    graph_builder_mod.addImport("rms_norm", rms_norm_mod);
    graph_builder_mod.addImport("attention", attention_mod);
    graph_builder_mod.addImport("swiglu", swiglu_mod);

    graph_builder_mod.addImport("model", model_mod);
    graph_builder_mod.addImport("memory", memory_mod);

    const graph_context_mod = b.createModule(.{
        .root_source_file = b.path("src/core/graph_context.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    graph_context_mod.addImport("ggml", ggml_mod);
    graph_context_mod.addImport("model", model_mod);
    graph_context_mod.addImport("graph_builder", graph_builder_mod);
    graph_context_mod.addImport("memory", memory_mod);

    model_mod.addImport("graph_builder", graph_builder_mod);

    model_mod.addImport("graph_builder", graph_builder_mod);

    const engine_common_mod = b.createModule(.{
        .root_source_file = b.path("src/core/engine_common.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    engine_common_mod.addImport("ggml", ggml_mod);
    engine_common_mod.addImport("model", model_mod);

    const memory_pool_mod = b.createModule(.{
        .root_source_file = b.path("src/core/memory_pool.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    memory_pool_mod.addImport("ggml", ggml_mod);

    const prefill_mod = b.createModule(.{
        .root_source_file = b.path("src/core/prefill.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    prefill_mod.addImport("ggml", ggml_mod);
    prefill_mod.addImport("graph_builder", graph_builder_mod);
    prefill_mod.addImport("kv_cache", kv_cache_mod);
    prefill_mod.addImport("model", model_mod);
    prefill_mod.addImport("engine_common", engine_common_mod);

    // ===================================================================
    // Shared tool module: metrics
    // ===================================================================
    const metrics_mod = b.createModule(.{
        .root_source_file = b.path("src/tools/metrics.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const registry_mod = b.createModule(.{
        .root_source_file = b.path("src/models/registry.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    registry_mod.addImport("ggml", ggml_mod);
    registry_mod.addImport("gguf", gguf_mod);
    registry_mod.addImport("model", model_mod);
    registry_mod.addImport("graph_builder", graph_builder_mod);
    registry_mod.addImport("memory", memory_mod);

    const pretype_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenizer/pretype.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const vocab_mod = b.createModule(.{
        .root_source_file = b.path("src/vocab.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    vocab_mod.addImport("gguf", gguf_mod);
    vocab_mod.addImport("pretype", pretype_mod);

    // ======================================================================
    // uucode 模块（Unicode 属性查询）
    // ======================================================================
    const uucode_fields: []const []const u8 = &.{
        "is_alphabetic",
        "is_uppercase",
        "is_lowercase",
        "is_emoji",
        "general_category",
        "name",
    };
    const uucode_dep = b.dependency("uucode", .{
        .target = target,
        .optimize = optimize,
        .fields = uucode_fields,
    });
    const uucode_mod = uucode_dep.module("uucode");
    const tokenizer_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenizer/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tokenizer_mod.addImport("ggml", ggml_mod);
    tokenizer_mod.addImport("gguf", gguf_mod);
    tokenizer_mod.addImport("vocab", vocab_mod);
    tokenizer_mod.addImport("uucode", uucode_mod);
    const sampler_mod = b.createModule(.{
        .root_source_file = b.path("src/sampler.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    sampler_mod.addImport("ggml", ggml_mod);

    const tokenize_cb_mod = b.createModule(.{
        .root_source_file = b.path("src/tools/tokenize_cb.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tokenize_cb_mod.addImport("tokenizer", tokenizer_mod);
    const chat_template_mod = b.createModule(.{
        .root_source_file = b.path("src/chat_template/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    chat_template_mod.addImport("model", model_mod);

    // 注册 chat_template 子模块
    const chat_template_types_mod = b.createModule(.{
        .root_source_file = b.path("src/chat_template/types.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const chat_template_multimodal_mod = b.createModule(.{
        .root_source_file = b.path("src/chat_template/multimodal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    chat_template_multimodal_mod.addImport("types", chat_template_types_mod);

    // 让 chat_template 模块可以导入子模块
    chat_template_mod.addImport("types", chat_template_types_mod);
    chat_template_mod.addImport("multimodal", chat_template_multimodal_mod);

    // 注册 chat_template 各模板实现子模块
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/chatml.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("chatml", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/llama3.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("llama3", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/llama4.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("llama4", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/gemma.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("gemma", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/gemma4.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("gemma4", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/mistral_v7.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("mistral_v7", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/phi4.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("phi4", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/deepseek3.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("deepseek3", tmpl);
    }
    {
        const tmpl = b.createModule(.{ .root_source_file = b.path("src/chat_template/tinyllama.zig"), .target = target, .optimize = optimize, .link_libc = true });
        tmpl.addImport("types", chat_template_types_mod);
        chat_template_mod.addImport("tinyllama", tmpl);
    }

    // ======================================================================
    // minja 模块（C++ Jinja2 模板引擎 Zig 封装）
    // ======================================================================
    const minja_mod = b.createModule(.{
        .root_source_file = b.path("src/chat_template/minja.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    minja_mod.addIncludePath(b.path("src/vendor/minja"));
    minja_mod.addIncludePath(b.path("src/vendor/minja/nlohmann"));
    minja_mod.linkLibrary(minja_lib);

    // Let chat_template module import minja for Jinja rendering fallback
    chat_template_mod.addImport("minja", minja_mod);

    // --- 核心引擎子模块（从 engine.zig 拆分 refact.md §1） ---
    const decode_mod = b.createModule(.{
        .root_source_file = b.path("src/core/decode.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    decode_mod.addImport("ggml", ggml_mod);
    decode_mod.addImport("model", model_mod);
    decode_mod.addImport("graph_builder", graph_builder_mod);
    decode_mod.addImport("graph_context", graph_context_mod);
    decode_mod.addImport("kv_cache", kv_cache_mod);
    decode_mod.addImport("tokenizer", tokenizer_mod);
    decode_mod.addImport("sampler", sampler_mod);
    decode_mod.addImport("engine_common", engine_common_mod);

    const verbose_mod = b.createModule(.{
        .root_source_file = b.path("src/core/verbose.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    verbose_mod.addImport("tokenizer", tokenizer_mod);
    verbose_mod.addImport("chat_template", chat_template_mod);

    const embedding_gen_mod = b.createModule(.{
        .root_source_file = b.path("src/core/embedding_gen.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    embedding_gen_mod.addImport("ggml", ggml_mod);
    embedding_gen_mod.addImport("model", model_mod);
    embedding_gen_mod.addImport("graph_builder", graph_builder_mod);
    embedding_gen_mod.addImport("tokenizer", tokenizer_mod);

    // ======================================================================
    // 多模态模块
    // ======================================================================

    const fft_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/fft.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    fft_mod.linkFramework("Accelerate", .{});

    // mtmd debug 模块
    const debug_mod = b.createModule(.{
        .root_source_file = b.path("src/debug.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    debug_mod.addImport("ggml", ggml_mod);

    // mtmd graph 模块（需在 audio/vision 模块之前定义）
    const mm_graph_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/graph/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_graph_mod.addImport("ggml", ggml_mod);
    mm_graph_mod.addImport("gguf", gguf_mod);
    mm_graph_mod.addImport("weight_loader", weight_loader_mod);
    mm_graph_mod.addImport("debug", debug_mod);

    const encoder_debug_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/encoder_debug.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    encoder_debug_mod.addImport("ggml", ggml_mod);
    encoder_debug_mod.addImport("debug", debug_mod);

    const mm_audio_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/audio/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_audio_mod.addImport("ggml", ggml_mod);
    mm_audio_mod.addImport("gguf", gguf_mod);
    mm_audio_mod.addImport("weight_loader", weight_loader_mod);
    mm_audio_mod.addImport("fft", fft_mod);
    mm_audio_mod.addImport("graph", mm_graph_mod);
    mm_audio_mod.addImport("debug", debug_mod);
    mm_audio_mod.addImport("encoder_debug", encoder_debug_mod);

    const mm_vision_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/vision/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_vision_mod.addImport("ggml", ggml_mod);
    mm_vision_mod.addImport("gguf", gguf_mod);
    mm_vision_mod.addImport("weight_loader", weight_loader_mod);
    mm_vision_mod.addImport("graph", mm_graph_mod);
    mm_vision_mod.addImport("debug", debug_mod);
    mm_vision_mod.addImport("encoder_debug", encoder_debug_mod);

    const utils_mod = b.createModule(.{
        .root_source_file = b.path("src/utils.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    utils_mod.addImport("ggml", ggml_mod);
    utils_mod.addImport("gguf", gguf_mod);
    utils_mod.addImport("tokenizer", tokenizer_mod);

    const mm_manager_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_manager_mod.addImport("ggml", ggml_mod);
    mm_manager_mod.addImport("gguf", gguf_mod);
    mm_manager_mod.addImport("model", model_mod);
    mm_manager_mod.addImport("audio", mm_audio_mod);
    mm_manager_mod.addImport("vision", mm_vision_mod);
    mm_manager_mod.addImport("tokenizer", tokenizer_mod);

    mm_manager_mod.addImport("graph", mm_graph_mod);

    const mm_preprocess_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/preprocess.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mm_preprocess_mod.addImport("ggml", ggml_mod);
    mm_preprocess_mod.addImport("fft", fft_mod);

    // stb_image — vendor/stb/stb_image.h wrapper for JPEG/PNG decoding
    const stb_image_mod = b.createModule(.{
        .root_source_file = b.path("src/stb_image.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    stb_image_mod.addIncludePath(b.path("src/vendor/stb"));
    stb_image_mod.addCSourceFile(.{ .file = b.path("src/vendor/stb/stb_image.c") });

    mm_preprocess_mod.addImport("stb_image", stb_image_mod);

    // vision module needs preprocess (was previously using relative @import)
    mm_vision_mod.addImport("preprocess", mm_preprocess_mod);

    // mtmd 子模块（helper 和 tokenize 通过 mm 模块导入 mod.zig）
    {
        const helper_mod = b.createModule(.{
            .root_source_file = b.path("src/mtmd/helper.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        helper_mod.addImport("ggml", ggml_mod);
        helper_mod.addImport("mm", mm_manager_mod);
        helper_mod.addImport("preprocess", mm_preprocess_mod);
        helper_mod.addImport("stb_image", stb_image_mod);
        helper_mod.addImport("engine_common", engine_common_mod);
        mm_manager_mod.addImport("helper", helper_mod);
        mm_audio_mod.addImport("helper", helper_mod);
    }
    {
        const tokenize_mod = b.createModule(.{
            .root_source_file = b.path("src/mtmd/tokenize.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        tokenize_mod.addImport("ggml", ggml_mod);
        tokenize_mod.addImport("gguf", gguf_mod);
        tokenize_mod.addImport("tokenizer", tokenizer_mod);
        tokenize_mod.addImport("mm", mm_manager_mod);
        tokenize_mod.addImport("preprocess", mm_preprocess_mod);
        mm_manager_mod.addImport("tokenize", tokenize_mod);
    }

    // --- 核心引擎多模态子模块（从 engine.zig 拆分 refact.md §1） ---
    const multimodal_mod = b.createModule(.{
        .root_source_file = b.path("src/core/multimodal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    multimodal_mod.addImport("ggml", ggml_mod);
    multimodal_mod.addImport("model", model_mod);
    multimodal_mod.addImport("graph_builder", graph_builder_mod);
    multimodal_mod.addImport("kv_cache", kv_cache_mod);
    multimodal_mod.addImport("tokenizer", tokenizer_mod);
    multimodal_mod.addImport("sampler", sampler_mod);
    multimodal_mod.addImport("chat_template", chat_template_mod);
    multimodal_mod.addImport("mtmd", mm_manager_mod);
    multimodal_mod.addImport("debug", debug_mod);
    multimodal_mod.addImport("preprocess", mm_preprocess_mod);
    multimodal_mod.addImport("engine_common", engine_common_mod);
    multimodal_mod.addImport("prefill", prefill_mod);
    multimodal_mod.addImport("decode", decode_mod);
    multimodal_mod.addImport("verbose", verbose_mod);
    multimodal_mod.addImport("graph_context", graph_context_mod);
    // 主可执行文件 zllama
    multimodal_mod.addImport("memory_pool", memory_pool_mod);

    multimodal_mod.addImport("stb_image", stb_image_mod);

    // ======================================================================
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe_mod.addImport("ggml", ggml_mod);
    exe_mod.addImport("gguf", gguf_mod);
    exe_mod.addImport("model", model_mod);
    exe_mod.addImport("registry", registry_mod);
    exe_mod.addImport("graph_builder", graph_builder_mod);
    exe_mod.addImport("memory", memory_mod);
    exe_mod.addImport("tokenizer", tokenizer_mod);
    exe_mod.addImport("sampler", sampler_mod);
    exe_mod.addImport("graph_context", graph_context_mod);
    exe_mod.addImport("mm", mm_manager_mod);
    exe_mod.addImport("preprocess", mm_preprocess_mod);
    exe_mod.addImport("kv_cache", kv_cache_mod);
    exe_mod.addImport("stb_image", stb_image_mod);
    exe_mod.addImport("pooling", pooling_mod);
    exe_mod.addImport("engine_common", engine_common_mod);
    exe_mod.addImport("chat_template", chat_template_mod);
    exe_mod.addImport("prefill", prefill_mod);
    exe_mod.addImport("mtmd", mm_manager_mod);
    exe_mod.addImport("debug", debug_mod);
    exe_mod.addImport("decode", decode_mod);
    exe_mod.addImport("verbose", verbose_mod);
    exe_mod.addImport("memory_pool", memory_pool_mod);

    exe_mod.addImport("embedding_gen", embedding_gen_mod);
    exe_mod.addImport("multimodal", multimodal_mod);

    const exe = b.addExecutable(.{
        .name = "zllama",
        .root_module = exe_mod,
    });

    // ======================================================================
    // zllama-tokenize
    // ======================================================================
    const tokenize_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenize_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tokenize_mod.addImport("ggml", ggml_mod);
    tokenize_mod.addImport("gguf", gguf_mod);
    tokenize_mod.addImport("tokenizer", tokenizer_mod);
    tokenize_mod.addImport("utils", utils_mod);
    tokenize_mod.addImport("engine_common", engine_common_mod);

    const tokenize_exe = b.addExecutable(.{
        .name = "zllama-tokenize",
        .root_module = tokenize_mod,
    });

    // ======================================================================
    // 测试
    // ======================================================================
    const test_step = b.step("test", "Run all tests");

    const test_root_mod = b.createModule(.{
        .root_source_file = b.path("src/test_runner.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_root_mod.addImport("ggml", ggml_mod);
    test_root_mod.addImport("gguf", gguf_mod);
    test_root_mod.addImport("model", model_mod);
    test_root_mod.addImport("registry", registry_mod);
    test_root_mod.addImport("graph_builder", graph_builder_mod);
    test_root_mod.addImport("memory", memory_mod);
    test_root_mod.addImport("tokenizer", tokenizer_mod);
    test_root_mod.addImport("sampler", sampler_mod);
    test_root_mod.addImport("kv_cache", kv_cache_mod);
    test_root_mod.addImport("graph_context", graph_context_mod);
    test_root_mod.addImport("mm", mm_manager_mod);
    test_root_mod.addImport("audio", mm_audio_mod);
    test_root_mod.addImport("preprocess", mm_preprocess_mod);
    test_root_mod.addImport("engine_common", engine_common_mod);
    test_root_mod.addImport("prefill", prefill_mod);
    test_root_mod.addImport("chat_template", chat_template_mod);
    test_root_mod.addImport("rms_norm", rms_norm_mod);
    test_root_mod.addImport("rope", rope_mod);
    test_root_mod.addImport("attention", attention_mod);
    test_root_mod.addImport("swiglu", swiglu_mod);
    test_root_mod.addImport("embed", embed_mod);
    test_root_mod.addImport("pooling", pooling_mod);
    test_root_mod.addImport("weight_loader", weight_loader_mod);
    test_root_mod.addImport("stb_image", stb_image_mod);
    test_root_mod.addImport("utils", utils_mod);
    test_root_mod.addImport("mtmd", mm_manager_mod);
    test_root_mod.addImport("debug", debug_mod);
    test_root_mod.addImport("decode", decode_mod);
    test_root_mod.addImport("verbose", verbose_mod);
    test_root_mod.addImport("embedding_gen", embedding_gen_mod);
    test_root_mod.addImport("multimodal", multimodal_mod);

    test_root_mod.addImport("graph", mm_graph_mod);
    test_root_mod.addImport("memory_pool", memory_pool_mod);
    test_root_mod.addImport("vision", mm_vision_mod);

    // 测试工具模块（src/tests/utils.zig），与 src/utils.zig 不同
    const test_utils_mod_for_root = b.createModule(.{
        .root_source_file = b.path("src/tests/utils.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_utils_mod_for_root.addImport("ggml", ggml_mod);
    test_utils_mod_for_root.addImport("gguf", gguf_mod);
    test_utils_mod_for_root.addImport("model", model_mod);
    test_root_mod.addImport("test_utils", test_utils_mod_for_root);

    // compare_logits 模块（用于 test_compare_logits.zig）
    const compare_logits_mod_for_root = b.createModule(.{
        .root_source_file = b.path("src/tools/compare_logits.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_root_mod.addImport("compare_logits", compare_logits_mod_for_root);

    // mtmd 模块（用于 test_mtmd.zig）
    test_root_mod.addImport("mtmd", mm_manager_mod);

    const unit_tests = b.addTest(.{
        .root_module = test_root_mod,
    });

    if (!bundle_ggml) {
        unit_tests.root_module.addRPathSpecial("/usr/local/lib");
    }
    const run_test_unit = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_test_unit.step);

    // 辅助函数：为测试步骤添加 rpath（使用系统库时需要）
    const addTestWithRpath: *const fn (*std.Build, []const u8, *std.Build.Module) *std.Build.Step.Compile = if (bundle_ggml)
        struct {
            fn add(b2: *std.Build, name: []const u8, root_module: *std.Build.Module) *std.Build.Step.Compile {
                return b2.addTest(.{ .name = name, .root_module = root_module });
            }
        }.add
    else
        struct {
            fn add(b2: *std.Build, name: []const u8, root_module: *std.Build.Module) *std.Build.Step.Compile {
                const t = b2.addTest(.{ .name = name, .root_module = root_module });
                t.root_module.addRPathSpecial("/usr/local/lib");
                return t;
            }
        }.add;

    // 子测试步骤（用于单独运行特定类别的测试）
    // 测试工具模块（src/tests/utils.zig），与 src/utils.zig 不同
    const test_utils_mod = b.createModule(.{
        .root_source_file = b.path("src/tests/utils.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_utils_mod.addImport("ggml", ggml_mod);
    test_utils_mod.addImport("gguf", gguf_mod);
    test_utils_mod.addImport("model", model_mod);

    const test_layers_step = b.step("test-layers", "Run layer tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_layers.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("rms_norm", rms_norm_mod);
        mod.addImport("rope", rope_mod);
        mod.addImport("attention", attention_mod);
        mod.addImport("swiglu", swiglu_mod);
        mod.addImport("utils", test_utils_mod);
        const t = addTestWithRpath(b, "test-layers", mod);
        const run_t = b.addRunArtifact(t);
        test_layers_step.dependOn(&run_t.step);
    }

    const test_gguf_step = b.step("test-gguf", "Run GGUF tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_gguf.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        const t = addTestWithRpath(b, "test-gguf", mod);
        const run_t = b.addRunArtifact(t);
        test_gguf_step.dependOn(&run_t.step);
    }

    const test_archs_step = b.step("test-archs", "Run architecture tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_archs.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("test_utils", test_utils_mod);
        const t = addTestWithRpath(b, "test-archs", mod);
        const run_t = b.addRunArtifact(t);
        test_archs_step.dependOn(&run_t.step);
    }

    const test_kv_cache_step = b.step("test-kv-cache", "Run KV Cache tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_kv_cache.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("kv_cache", kv_cache_mod);
        mod.addImport("memory", memory_mod);
        const t = addTestWithRpath(b, "test-kv-cache", mod);
        const run_t = b.addRunArtifact(t);
        test_kv_cache_step.dependOn(&run_t.step);
    }

    const test_vocab_step = b.step("test-vocab", "Run vocab-based tokenizer tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_vocab.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("tokenizer", tokenizer_mod);
        const t = addTestWithRpath(b, "test-vocab", mod);
        const run_t = b.addRunArtifact(t);
        test_vocab_step.dependOn(&run_t.step);
    }

    const test_embed_step = b.step("test-embed", "Run embedding tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_embed.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("pooling", pooling_mod);
        const t = addTestWithRpath(b, "test-embed", mod);
        const run_t = b.addRunArtifact(t);
        test_embed_step.dependOn(&run_t.step);
    }

    const test_mtmd_step = b.step("test-mtmd", "Run multimodal tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_mtmd.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("model", model_mod);
        mod.addImport("mm", mm_manager_mod);
        mod.addImport("mtmd", mm_manager_mod);
        const t = addTestWithRpath(b, "test-mtmd", mod);
        const run_t = b.addRunArtifact(t);
        test_mtmd_step.dependOn(&run_t.step);
    }

    const test_audio_step = b.step("test-audio", "Run audio processing tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_audio.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("chat_template", chat_template_mod);
        mod.addImport("preprocess", mm_preprocess_mod);
        mod.addImport("audio", mm_audio_mod);
        const t = addTestWithRpath(b, "test-audio", mod);
        const run_t = b.addRunArtifact(t);
        test_audio_step.dependOn(&run_t.step);
    }

    const test_vision_step = b.step("test-vision", "Run vision processing tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_vision.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("chat_template", chat_template_mod);
        mod.addImport("preprocess", mm_preprocess_mod);
        const t = addTestWithRpath(b, "test-vision", mod);
        const run_t = b.addRunArtifact(t);
        test_vision_step.dependOn(&run_t.step);
    }

    const test_graph_dtypes_step = b.step("test-graph-dtypes", "Run graph dtype (BF16/F16) tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_graph_dtypes.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("graph", mm_graph_mod);
        mod.addImport("mtmd", mm_manager_mod);
        const t = addTestWithRpath(b, "test-graph-dtypes", mod);
        const run_t = b.addRunArtifact(t);
        test_graph_dtypes_step.dependOn(&run_t.step);
    }

    const test_stb_image_step = b.step("test-stb-image", "Run stb_image integration tests only");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_stb_image.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("stb_image", stb_image_mod);
        mod.addImport("preprocess", mm_preprocess_mod);
        mod.addImport("vision", mm_vision_mod);
        const t = addTestWithRpath(b, "test-stb-image", mod);
        const run_t = b.addRunArtifact(t);
        test_stb_image_step.dependOn(&run_t.step);
    }

    const test_compare_logits_step = b.step("test-compare-logits", "Run compare_logits tests only");
    {
        const compare_logits_mod = b.createModule(.{
            .root_source_file = b.path("src/tools/compare_logits.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_compare_logits.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("compare_logits", compare_logits_mod);
        const t = addTestWithRpath(b, "test-compare-logits", mod);
        const run_t = b.addRunArtifact(t);
        test_compare_logits_step.dependOn(&run_t.step);
    }

    const test_ggml_step = b.step("test-ggml", "Run ggml binding tests (arange, cont, dup, conv, pool, roll, etc.)");
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_arange.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-arange", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_cont.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-cont", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_dup.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-dup", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_customop.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-customop", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_interpolate.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-interpolate", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_pad_reflect_1d.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-pad-reflect-1d", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_pool.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-pool", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_rel_pos.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-rel-pos", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_roll.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-roll", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_timestep_embedding.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-timestep-embedding", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_gguf.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-gguf", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_conv.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-conv", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tests/test_ggml_quantize_fns.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        const t = addTestWithRpath(b, "test-ggml-quantize-fns", mod);
        const run_t = b.addRunArtifact(t);
        test_ggml_step.dependOn(&run_t.step);
    }

    // ======================================================================
    // 工具可执行文件
    // ======================================================================
    {

        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/dump_graph.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-dump-graph",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("dump-graph", "Run zllama-dump-graph tool");
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/compare_logits.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-compare-logits",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("compare-logits", "Run zllama-compare-logits tool");
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/generate_reference.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);
        mod.addImport("kv_cache", kv_cache_mod);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-gen-ref",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("gen-ref", "Run zllama-gen-ref tool");
    }

    {
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/compare_with_llamacpp.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);
        mod.addImport("kv_cache", kv_cache_mod);

        mod.addImport("metrics", metrics_mod);
        const exe_tool = b.addExecutable(.{
            .name = "zllama-compare-llamacpp",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("compare-llamacpp", "Run zllama-compare-llamacpp tool");
    }

    {
        // zllama-compare-mtmd-vision: multimodal vision output quality validation
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/compare_mtmd_vision.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);
        mod.addImport("sampler", sampler_mod);
        mod.addImport("kv_cache", kv_cache_mod);
        mod.addImport("mm", mm_manager_mod);
        mod.addImport("mtmd", mm_manager_mod);
        mod.addImport("preprocess", mm_preprocess_mod);
        mod.addImport("chat_template", chat_template_mod);
        mod.addImport("engine_common", engine_common_mod);
        mod.addImport("prefill", prefill_mod);
        mod.addImport("metrics", metrics_mod);
        mod.addImport("tokenize_cb", tokenize_cb_mod);

        const exe_tool = b.addExecutable(.{
            .name = "zllama-compare-mtmd-vision",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("compare-mtmd-vision", "Run zllama-compare-mtmd-vision tool");
    }

    {
        // zllama-compare-mtmd-audio: multimodal audio output quality validation
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/compare_mtmd_audio.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("ggml", ggml_mod);
        mod.addImport("gguf", gguf_mod);
        mod.addImport("model", model_mod);
        mod.addImport("registry", registry_mod);
        mod.addImport("graph_builder", graph_builder_mod);
        mod.addImport("memory", memory_mod);
        mod.addImport("tokenizer", tokenizer_mod);
        mod.addImport("sampler", sampler_mod);
        mod.addImport("kv_cache", kv_cache_mod);
        mod.addImport("mm", mm_manager_mod);
        mod.addImport("mtmd", mm_manager_mod);
        mod.addImport("preprocess", mm_preprocess_mod);
        mod.addImport("chat_template", chat_template_mod);
        mod.addImport("engine_common", engine_common_mod);
        mod.addImport("prefill", prefill_mod);

        mod.addImport("metrics", metrics_mod);
        mod.addImport("tokenize_cb", tokenize_cb_mod);
        const exe_tool = b.addExecutable(.{
            .name = "zllama-compare-mtmd-audio",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("compare-mtmd-audio", "Run zllama-compare-mtmd-audio tool");
    }

    {
        // zllama-align-cmp: vector alignment comparison tool
        const mod = b.createModule(.{
            .root_source_file = b.path("src/tools/align_cmp.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("metrics", metrics_mod);
        const exe_tool = b.addExecutable(.{
            .name = "zllama-align-cmp",
            .root_module = mod,
        });
        b.installArtifact(exe_tool);

        const run_cmd = b.addRunArtifact(exe_tool);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        _ = b.step("align-cmp", "Run zllama-align-cmp tool (vector alignment comparison)");
    }

    // ======================================================================
    // 安装与运行
    // ======================================================================
    b.installArtifact(exe);
    b.installArtifact(tokenize_exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run zllama.zig engine");
    run_step.dependOn(&run_cmd.step);

    const tokenize_run_cmd = b.addRunArtifact(tokenize_exe);
    tokenize_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        tokenize_run_cmd.addArgs(args);
    }
    const tokenize_run_step = b.step("tokenize", "Run zllama-tokenize tool");
    tokenize_run_step.dependOn(&tokenize_run_cmd.step);
}

// ============================================================================
// 从源码构建 ggml 静态库
// ============================================================================

/// 构建 ggml 静态库（CPU 后端）。
/// 返回一个 Step.Compile，可通过 Module.linkLibrary() 链接。
fn buildGgmlFromSource(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const src_dir = "deps/ggml/src";
    const cpu_dir = src_dir ++ "/ggml-cpu";
    const amx_dir = cpu_dir ++ "/amx";
    const cpu_arch = target.result.cpu.arch;
    const os = target.result.os.tag;

    // 创建 ggml 库模块（无 Zig 根文件，仅有 C/C++ link_objects）
    const lib_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    // 头文件路径 — 全部从 deps/ggml/ 解析（-I 优先级 > 系统 /usr/local/include）
    // 绝不使用系统安装的 ggml 头文件，避免版本不一致导致的 ABI 损坏
    lib_mod.addIncludePath(b.path("deps/ggml/include")); // 公共 API: ggml.h, ggml-cpu.h, ggml-backend.h, ggml-alloc.h, gguf.h
    lib_mod.addIncludePath(b.path("deps/ggml/src")); // 内部实现: ggml-impl.h, ggml-common.h, ggml-backend-impl.h, ggml-threading.h, ggml-quants.h
    lib_mod.addIncludePath(b.path("deps/ggml/src/ggml-cpu")); // CPU 后端内部: ggml-cpu-impl.h, traits.h, quants.h, ops.h, vec.h, simd-mappings.h 等
    lib_mod.addIncludePath(b.path("deps/ggml/src/ggml-cpu/amx")); // AMX 内部: amx.h, mmq.h, common.h

    // 宏定义
    lib_mod.addCMacro("GGML_USE_CPU", "1");
    lib_mod.addCMacro("GGML_SCHED_MAX_COPIES", "4");
    lib_mod.addCMacro("GGML_VERSION", "\"0.15.2\"");
    lib_mod.addCMacro("GGML_COMMIT", "\"707321c4\"");

    lib_mod.addCMacro("NDEBUG", "1");
    // x86_64 优化标志（所有 CPU 后端文件共享）
    const x86_opt_flags: []const []const u8 = if (cpu_arch == .x86_64)
        &.{ "-mavx2", "-mfma", "-mf16c", "-mavx", "-msse4.2" }
    else
        &.{};
    // C 文件基础标志
    const c_base_flags = &.{ "-std=c11", "-Wno-unused-function", "-Wno-unused-variable", "-Wno-missing-braces", "-Wno-implicit-function-declaration" };

    // C++ 文件基础标志
    const cpp_base_flags = &.{ "-std=gnu++17", "-Wno-unused-function", "-Wno-unused-variable", "-Wno-missing-braces" };

    // 系统库链接（通过模块）
    switch (os) {
        .windows => {
            lib_mod.linkSystemLibrary("ole32", .{});
            lib_mod.linkSystemLibrary("ws2_32", .{});
        },
        .macos, .ios, .tvos, .watchos => {
            lib_mod.linkFramework("Foundation", .{});
            lib_mod.linkFramework("Accelerate", .{});
            lib_mod.addCMacro("GGML_USE_ACCELERATE", "1");
            lib_mod.addCMacro("ACCELERATE_NEW_LAPACK", "1");
            lib_mod.addCMacro("ACCELERATE_LAPACK_ILP64", "1");
            lib_mod.addCMacro("_DARWIN_C_SOURCE", "1");
            lib_mod.addCMacro("_XOPEN_SOURCE", "600");
        },
        .linux => {
            lib_mod.linkSystemLibrary("pthread", .{});
        },
        else => {},
    }

    // ---- 辅助：连接标志切片 ----
    const ConcatFlags = struct {
        fn concat(alloc: std.mem.Allocator, a: []const []const u8, extra: []const []const u8) []const []const u8 {
            if (extra.len == 0) return a;
            const result = alloc.alloc([]const u8, a.len + extra.len) catch @panic("OOM");
            @memcpy(result[0..a.len], a);
            @memcpy(result[a.len..], extra);
            return result;
        }
    }.concat;

    // ---- 添加 C 源文件 ----
    inline for (.{ "ggml.c", "ggml-quants.c", "ggml-alloc.c" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(src_dir ++ "/" ++ file),
            .flags = c_base_flags,
        });
    }

    // ---- 添加 C++ 核心源文件 ----
    inline for (.{ "ggml.cpp", "ggml-backend.cpp", "ggml-backend-dl.cpp", "ggml-backend-reg.cpp", "ggml-backend-meta.cpp", "ggml-threading.cpp" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(src_dir ++ "/" ++ file),
            .flags = cpp_base_flags,
        });
    }

    // ---- 添加 CPU 后端 C 文件（含架构优化标志） ----
    inline for (.{ "ggml-cpu.c", "quants.c" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(cpu_dir ++ "/" ++ file),
            .flags = ConcatFlags(b.allocator, c_base_flags, x86_opt_flags),
        });
    }

    // ---- 添加 CPU 后端 C++ 文件（含架构优化标志） ----
    inline for (.{ "ggml-cpu.cpp", "repack.cpp", "hbm.cpp", "traits.cpp", "binary-ops.cpp", "unary-ops.cpp", "vec.cpp", "ops.cpp" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(cpu_dir ++ "/" ++ file),
            .flags = ConcatFlags(b.allocator, cpp_base_flags, x86_opt_flags),
        });
    }

    // ---- 添加 AMX C++ 文件（含架构优化标志） ----
    inline for (.{ "amx.cpp", "mmq.cpp" }) |file| {
        lib_mod.addCSourceFile(.{
            .file = b.path(amx_dir ++ "/" ++ file),
            .flags = ConcatFlags(b.allocator, cpp_base_flags, x86_opt_flags),
        });
    }

    // ---- 架构特定源文件 ----
    switch (cpu_arch) {
        .x86_64 => {
            const arch_dir = cpu_dir ++ "/arch/x86";
            lib_mod.addIncludePath(b.path(arch_dir));
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/cpu-feats.cpp"),
                .flags = cpp_base_flags,
            });
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/quants.c"),
                .flags = ConcatFlags(b.allocator, c_base_flags, x86_opt_flags),
            });
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/repack.cpp"),
                .flags = ConcatFlags(b.allocator, cpp_base_flags, x86_opt_flags),
            });
        },
        .aarch64 => {
            const arch_dir = cpu_dir ++ "/arch/arm";
            lib_mod.addIncludePath(b.path(arch_dir));
            lib_mod.addCSourceFile(.{
                .file = b.path(arch_dir ++ "/cpu-feats.cpp"),
                .flags = cpp_base_flags,
            });
        },
        else => {
            // 其他架构：无架构特定优化文件
        },
    }

    // 创建静态库
    const lib = b.addLibrary(.{
        .name = "ggml",
        .root_module = lib_mod,
        .linkage = .static,
    });
    // 链接时优化（macOS 上 LTO 需要 LLD，但 LLD 不支持 Mach-O，故仅在 Linux 上启用）
    if (os == .linux) {
        lib.lto = .full;
        lib.use_lld = true;
    }

    return lib;
}

/// 构建 minja 静态库（C++ Jinja2 模板引擎桥接层）。
/// 返回一个 Step.Compile，可通过 Module.linkLibrary() 链接。
fn buildMinjaFromSource(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const vendor_dir = "src/vendor/minja";

    const lib_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    // Include paths: vendor minja headers and nlohmann/json
    lib_mod.addIncludePath(b.path(vendor_dir));
    lib_mod.addIncludePath(b.path(vendor_dir ++ "/nlohmann"));

    const cpp_flags = &.{ "-std=c++17", "-Wno-unused-function", "-Wno-unused-variable", "-Wno-missing-braces" };

    // Add the bridge C++ source
    lib_mod.addCSourceFile(.{
        .file = b.path(vendor_dir ++ "/bridge.cpp"),
        .flags = cpp_flags,
    });

    const lib = b.addLibrary(.{
        .name = "minja",
        .root_module = lib_mod,
        .linkage = .static,
    });

    return lib;
}
