# 文件组织约定

## ✅ 优化目标

- **模块化**：每个子目录有明确的 `mod.zig` 作为公开 API 入口, 每个目录职责明确。
- **扁平化顶层**：顶层只保留真正全局的文件（如 `main.zig`、`model.zig`、`sampler.zig`、`gguf.zig`、`utils.zig`）。
- **语义清晰**：目录名能直观反映其内容职责。
- **符合 Zig 惯例**：使用 `mod.zig`，避免同名文件和目录混淆, 统一导入风格。
- **目录内相对路径**：同一目录内（如 `tokenizer/models/`）可使用 `@import("../mod.zig")` 相对导入；跨目录必须使用模块名。

---

## 📁 当前目录结构

```
src/
├── main.zig                      # 唯一主入口（Juicy Main）
├── model.zig                     # Architecture 枚举、ModelVTable、ModelInstance、SpecialTokens、ModelCapabilities
├── sampler.zig                   # 采样器
├── kv_cache.zig                  # KV 缓存
├── gguf.zig                      # GGUF 解析器（保留顶层，因被多处导入）
├── gguf_parse.zig                # GGUF 字节数据解析（从 gguf.zig 拆分）
├── utils.zig                     # 通用工具（时间、日志过滤等）
├── vocab.zig                     # 词表定义
├── vocab_loader.zig              # 词表加载辅助（从 vocab.zig 拆分）
├── vocab_tests.zig               # 词表导入测试
├── debug.zig                     # 调试输出（张量 dump 等）
├── cli_args.zig                  # CLI 参数解析
├── tokenize_main.zig             # zllama-tokenize 工具入口
├── test_runner.zig               # 测试入口
├── stb_image.zig                 # stb_image 封装（图像加载）
│
├── core/                         # 核心基础设施
│   ├── mod.zig                   # 重新导出所有核心子模块
│   ├── memory.zig                # 内存管理（KV Cache / SSM / Hybrid）
│   ├── engine.zig                # 推理引擎主入口
│   ├── engine_common.zig         # 引擎公共函数（日志、时间、模型加载）
│   ├── graph_builder.zig         # 计算图构建器
│   ├── graph_context.zig         # 计算图上下文（gallocr 封装）
│   ├── loader.zig                # 模型加载编排
│   ├── weight_loader.zig         # 权重加载
│   ├── prefill.zig               # 预填充逻辑
│   ├── decode.zig                # 增量解码循环
│   ├── verbose.zig               # 推理过程输出
│   ├── embedding_gen.zig         # Embedding 模型生成
│   └── multimodal.zig            # 多模态推理入口
│
├── layers/                       # 神经网络层（共享算子）
│   ├── attention.zig             # 缩放点积注意力（含 GQA）
│   ├── embed.zig                 # Token 嵌入
│   ├── linear.zig                # 线性投影
│   ├── pooling.zig               # 池化（mean/cls/last）
│   ├── rms_norm.zig              # RMSNorm 归一化
│   ├── rope.zig                  # RoPE 位置编码
│   └── swiglu.zig                # SwiGLU 前馈网络
│
├── models/                       # 具体模型实现
│   ├── mod.zig                   # 重新导出所有模型实现
│   ├── registry.zig              # 架构检测、工厂创建、能力检测
│   ├── embedding.zig             # Embedding 模型
│   ├── gemma3.zig                # Gemma 3
│   ├── gemma4.zig                # Gemma 4 (E2B)
│   ├── gemma4_graph.zig          # Gemma 4 计算图（从 gemma4.zig 拆分）
│   ├── gemma4_loader.zig         # Gemma 4 权重加载（从 gemma4.zig 拆分）
│   ├── llama.zig                 # LLaMA 3 / TinyLlama
│   ├── qwen2.zig                 # Qwen 2 / 2.5 / 3
│   ├── qwen35.zig                # Qwen 3.5（混合架构）
│   ├── qwen35_loader.zig         # Qwen 3.5 权重加载（从 qwen35.zig 拆分）
│   └── qwen3vl.zig               # Qwen3VL（多模态）
│
├── tokenizer/                    # 分词器
│   ├── mod.zig                   # 分词器入口
│   ├── pretype.zig               # 预处理类型枚举（从 src/ 移入）
│   ├── bpe.zig                   # BPE 算法
│   ├── decode.zig                # Token 解码
│   ├── decode_helpers.zig        # 解码辅助
│   ├── encode.zig                # Token 编码
│   ├── encode_config.zig         # 编码配置
│   ├── encode_spm.zig            # SPM 编码
│   ├── encode_word.zig           # Word 编码
│   ├── trie.zig                  # Trie 数据结构
│   ├── types.zig                 # 分词器类型（兼容层）
│   ├── unicode.zig               # Unicode 处理
│   ├── utils.zig                 # 分词器工具
│   └── models/                   # 预分词器实现（内部使用相对路径）
│       ├── bloom.zig
│       ├── deepseek3.zig
│       ├── deepseek_coder.zig
│       ├── deepseek_llm.zig
│       ├── falcon.zig
│       ├── gpt2.zig
│       ├── gpt2_style.zig
│       ├── gpt2_style_nospace.zig
│       ├── llama3.zig
│       ├── mpt.zig
│       ├── newline_only.zig
│       ├── qwen.zig
│       ├── qwen2.zig
│       ├── starcoder.zig
│       └── tryMatchContractionOrWord.zig
│
├── chat_template/                # 对话模板
│   ├── mod.zig                   # 模板入口（含 Jinja2 渲染）
│   ├── _tests.zig                # 模板单元测试
│   ├── chatml.zig                # ChatML 模板
│   ├── deepseek3.zig             # DeepSeek V3 模板
│   ├── gemma.zig                 # Gemma 模板
│   ├── gemma4.zig                # Gemma 4 模板
│   ├── llama3.zig                # LLaMA 3 模板
│   ├── llama4.zig                # LLaMA 4 模板
│   ├── minja.zig                 # minja C++ 桥接（Jinja2 渲染）
│   ├── mistral_v7.zig            # Mistral V7 模板
│   ├── multimodal.zig            # 多模态模板（占位符扫描/展开，支持动态 ScanMarkers）
│   ├── phi4.zig                  # Phi-4 模板
│   ├── tinyllama.zig             # TinyLlama 模板
│   └── types.zig                 # 模板类型定义（含 ScanMarkers、PlaceholderInfo）
│
├── mtmd/                         # 多模态
│   ├── mod.zig                   # MultiModalManager（编码器管理）
│   ├── fft.zig                   # FFT 实现（Accelerate vDSP）
│   ├── helper.zig                # 辅助函数（chunk 评估、文件加载）
│   ├── preprocess.zig            # 预处理（图像 resize、Mel 频谱）
│   ├── tokenize.zig              # 多模态分词（文本/图像/音频混合）
│   ├── audio/                    # 音频编码器（Conformer）
│   ├── graph/                    # 多模态计算图（ViT/Conformer 图构建）
│   └── vision/                   # 视觉编码器（ViT/SigLIP）
│
├── ggml/                         # ggml 绑定
│   ├── mod.zig                   # 安全封装入口
│   ├── backend.zig               # 后端抽象
│   ├── c.zig                     # C API 声明
│   ├── context.zig               # ggml_context 封装
│   ├── gguf.zig                  # gguf.h 封装
│   ├── graph.zig                 # ggml_cgraph 封装
│   ├── ops.zig                   # 算子封装
│   ├── tensor.zig                # ggml_tensor 封装
│   ├── threadpool.zig            # 线程池封装
│   └── utils.zig                 # 工具函数
│
├── tests/                        # 单元测试
├── tools/                        # 辅助工具（compare_logits, dump_graph, generate_reference 等）
└── vendor/                       # 第三方代码
    ├── minja/                    # minja C++ 桥接（Jinja2 模板引擎）
    └── stb/                      # stb_image（图像加载）
```

## 📌 目录内相对路径约定

同一目录树内可以使用相对路径导入：

| 位置 | 示例 | 说明 |
|------|------|------|
| `tokenizer/models/*.zig` | `@import("../mod.zig")` | 导入父目录 mod.zig |
| `tokenizer/` 内 | 通过 `mod.zig` 重新导出 | 跨目录通过模块名 |

**跨目录必须使用模块名**（如 `@import("vocab")`、`@import("tokenizer")`），禁止 `@import("../other_dir/foo.zig")`。

## 📝 顶层文件说明

以下文件保留在 `src/` 顶层，因为它们被多处模块引用或属于独立入口：

| 文件 | 说明 |
|------|------|
| `main.zig` | 主二进制入口（Juicy Main） |
| `model.zig` | 模型接口 + 所有模型实现的重新导出 |
| `sampler.zig` | 采样器 |
| `kv_cache.zig` | KV Cache |
| `gguf.zig` | GGUF 解析 + 类型定义 |
| `gguf_parse.zig` | GGUF 字节数据解析（从 gguf.zig 拆分） |
| `utils.zig` | 通用工具 |
| `vocab.zig` | 词表定义 + PreType 重导出 |
| `debug.zig` | 调试输出 |
| `cli_args.zig` | CLI 参数 |
| `stb_image.zig` | stb_image 封装 |
