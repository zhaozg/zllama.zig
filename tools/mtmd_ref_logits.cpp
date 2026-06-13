/**
 * mtmd_ref_logits.cpp
 *
 * 多模态参考 logits 生成工具
 *
 * 使用 llama.cpp 的 C API 加载模型 + mmproj + 媒体文件（图像/音频），
 * 运行多模态推理，并将第一个 token 的 logits 保存为二进制文件。
 *
 * 生成的二进制文件格式：n_vocab 个 float32 值，按行排列。
 *
 * 用法:
 *   # 视觉
 *   mtmd-ref-logits -m model.gguf --mmproj mmproj.gguf --image hello.png -p ":" -o ref_vision.bin
 *
 *   # 音频
 *   mtmd-ref-logits -m model.gguf --mmproj mmproj.gguf --audio hello.wav -p ":" -o ref_audio.bin
 *
 * 编译:
 *   c++ -std=c++17 -O2 tools/mtmd_ref_logits.cpp \
 *       $(pkg-config --cflags --libs llama) \
 *       -lmtmd -o mtmd-ref-logits
 */

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

// ============================================================================
// 配置
// ============================================================================

struct Config {
    std::string model_path;
    std::string mmproj_path;
    std::string media_path;   // 图像或音频文件路径
    std::string prompt = ":"; // 默认 prompt（仅包含媒体标记）
    std::string output_path;
    bool is_audio = false;    // true=音频, false=图像
    int n_threads = 4;
    int n_gpu_layers = 99;
};

static void print_usage(const char * argv0) {
    fprintf(stderr, "Usage: %s -m <model> --mmproj <mmproj> (--image <img>|--audio <audio>) -p <prompt> -o <output.bin> [options]\n", argv0);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>       Path to GGUF model file\n");
    fprintf(stderr, "  --mmproj <path>          Path to multimodal projector file\n");
    fprintf(stderr, "  --image <path>           Path to image file\n");
    fprintf(stderr, "  --audio <path>           Path to audio file\n");
    fprintf(stderr, "  -p, --prompt <text>      Prompt text (default: \":\")\n");
    fprintf(stderr, "  -o, --output <file>      Output binary file for logits\n");
    fprintf(stderr, "  -t, --threads <N>        Number of threads (default: 4)\n");
    fprintf(stderr, "  -ngl, --n-gpu-layers <N> Number of GPU layers (default: 99)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  # Vision\n");
    fprintf(stderr, "  %s -m model.gguf --mmproj mmproj.gguf --image hello.png -p \":\" -o ref_vision.bin\n", argv0);
    fprintf(stderr, "  # Audio\n");
    fprintf(stderr, "  %s -m model.gguf --mmproj mmproj.gguf --audio hello.wav -p \":\" -o ref_audio.bin\n", argv0);
}

static Config parse_args(int argc, char ** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i < argc) cfg.model_path = argv[i];
        } else if (strcmp(argv[i], "--mmproj") == 0) {
            if (++i < argc) cfg.mmproj_path = argv[i];
        } else if (strcmp(argv[i], "--image") == 0) {
            if (++i < argc) { cfg.media_path = argv[i]; cfg.is_audio = false; }
        } else if (strcmp(argv[i], "--audio") == 0) {
            if (++i < argc) { cfg.media_path = argv[i]; cfg.is_audio = true; }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (++i < argc) cfg.prompt = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i < argc) cfg.output_path = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (++i < argc) cfg.n_threads = std::stoi(argv[i]);
        } else if (strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--n-gpu-layers") == 0) {
            if (++i < argc) cfg.n_gpu_layers = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    return cfg;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    ggml_time_init();

    // 1. 解析参数
    Config cfg = parse_args(argc, argv);

    if (cfg.model_path.empty() || cfg.mmproj_path.empty() ||
        cfg.media_path.empty() || cfg.output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "=== mtmd-ref-logits ===\n");
    fprintf(stderr, "  Model:    %s\n", cfg.model_path.c_str());
    fprintf(stderr, "  MMproj:   %s\n", cfg.mmproj_path.c_str());
    fprintf(stderr, "  Media:    %s (%s)\n", cfg.media_path.c_str(), cfg.is_audio ? "audio" : "image");
    fprintf(stderr, "  Prompt:   %s\n", cfg.prompt.c_str());
    fprintf(stderr, "  Output:   %s\n", cfg.output_path.c_str());
    fprintf(stderr, "  Threads:  %d\n", cfg.n_threads);

    // 2. 加载动态后端
    ggml_backend_load_all();

    // 3. 加载 LLM 模型
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from %s\n", cfg.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 4. 初始化上下文
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;  // 足够大的上下文
    ctx_params.n_batch = 512;
    ctx_params.no_perf = true;

    llama_context * lctx = llama_init_from_model(model, ctx_params);
    if (!lctx) {
        fprintf(stderr, "Error: failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }

    // 5. 初始化 mtmd 上下文（加载 mmproj）
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false;
    mparams.print_timings = true;
    mparams.n_threads = cfg.n_threads;
    mparams.warmup = true;

    mtmd_context * mctx = mtmd_init_from_file(cfg.mmproj_path.c_str(), model, mparams);
    if (!mctx) {
        fprintf(stderr, "Error: failed to load mmproj from %s\n", cfg.mmproj_path.c_str());
        llama_free(lctx);
        llama_model_free(model);
        return 1;
    }

    // 6. 加载媒体文件（图像或音频）
    mtmd_bitmap * bitmap = nullptr;

    if (cfg.is_audio) {
        // 音频：使用 mtmd_helper_bitmap_init_from_file
        auto wrapper = mtmd_helper_bitmap_init_from_file(mctx, cfg.media_path.c_str(), false);
        bitmap = wrapper.bitmap;
        if (!bitmap) {
            fprintf(stderr, "Error: failed to load audio from %s\n", cfg.media_path.c_str());
            mtmd_free(mctx);
            llama_free(lctx);
            llama_model_free(model);
            return 1;
        }
        fprintf(stderr, "  Audio loaded: %u samples\n", (unsigned)mtmd_bitmap_get_n_bytes(bitmap) / sizeof(float));
    } else {
        // 图像：使用 mtmd_helper_bitmap_init_from_file
        auto wrapper = mtmd_helper_bitmap_init_from_file(mctx, cfg.media_path.c_str(), false);
        bitmap = wrapper.bitmap;
        if (!bitmap) {
            fprintf(stderr, "Error: failed to load image from %s\n", cfg.media_path.c_str());
            mtmd_free(mctx);
            llama_free(lctx);
            llama_model_free(model);
            return 1;
        }
        fprintf(stderr, "  Image loaded: %ux%u\n",
                (unsigned)mtmd_bitmap_get_nx(bitmap),
                (unsigned)mtmd_bitmap_get_ny(bitmap));
    }

    // 7. 构建 prompt（确保包含媒体标记）
    std::string full_prompt = cfg.prompt;
    // 如果 prompt 中没有媒体标记，在前面添加
    if (full_prompt.find(mtmd_default_marker()) == std::string::npos) {
        full_prompt = mtmd_default_marker() + full_prompt;
    }

    // 8. Tokenize
    mtmd_input_text text;
    text.text = full_prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();

    const mtmd_bitmap * bitmaps[] = { bitmap };
    int32_t ret = mtmd_tokenize(mctx, chunks, &text, bitmaps, 1);
    if (ret != 0) {
        fprintf(stderr, "Error: mtmd_tokenize failed with code %d\n", ret);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        mtmd_free(mctx);
        llama_free(lctx);
        llama_model_free(model);
        return 1;
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    fprintf(stderr, "  Tokenized: %zu chunks\n", n_chunks);

    // 9. 释放 bitmap（tokenize 后不再需要）
    mtmd_bitmap_free(bitmap);

    // 10. 计算总 token 数
    size_t n_total_tokens = mtmd_helper_get_n_tokens(chunks);
    fprintf(stderr, "  Total tokens: %zu\n", n_total_tokens);

    // 11. 执行 eval_chunks（自动处理文本和媒体编码 + llama_decode）
    llama_pos new_n_past = 0;
    ret = mtmd_helper_eval_chunks(
        mctx,
        lctx,
        chunks,
        0,          // n_past
        0,          // seq_id
        512,        // n_batch
        true,       // logits_last: 只保留最后一个 token 的 logits
        &new_n_past
    );

    mtmd_input_chunks_free(chunks);

    if (ret != 0) {
        fprintf(stderr, "Error: mtmd_helper_eval_chunks failed with code %d\n", ret);
        mtmd_free(mctx);
        llama_free(lctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "  n_past after eval: %d\n", new_n_past);

    // 12. 获取 logits
    // 使用 llama_get_logits_ith 获取最后一个 token 的 logits
    float * logits = llama_get_logits_ith(lctx, -1);
    if (!logits) {
        fprintf(stderr, "Error: failed to get logits\n");
        mtmd_free(mctx);
        llama_free(lctx);
        llama_model_free(model);
        return 1;
    }

    int n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "  Vocab size: %d\n", n_vocab);

    // 13. 保存 logits 到二进制文件
    {
        std::ofstream out(cfg.output_path, std::ios::binary);
        if (!out) {
            fprintf(stderr, "Error: failed to open output file %s\n", cfg.output_path.c_str());
            mtmd_free(mctx);
            llama_free(lctx);
            llama_model_free(model);
            return 1;
        }
        out.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));
        out.close();
        fprintf(stderr, "  Logits saved: %s (%d floats, %zu bytes)\n",
                cfg.output_path.c_str(), n_vocab, n_vocab * sizeof(float));
    }

    // 14. 清理
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);

    fprintf(stderr, "=== Done ===\n");
    return 0;
}
