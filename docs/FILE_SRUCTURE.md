# 文件组织约定


## ✅ 优化目标

- **模块化**：每个子目录有明确的 `mod.zig` 作为公开 API 入口, 每个目录职责明确。
- **扁平化顶层**：顶层只保留真正全局的文件（如 `main.zig`、`model.zig`、`sampler.zig`、`gguf.zig`、`utils.zig`）。
- **语义清晰**：目录名能直观反映其内容职责。
- **符合 Zig 惯例**：使用 `mod.zig`，避免同名文件和目录混淆, 统一导入风格。

---

## 📁 建议目录结构

```
src/
├── main.zig                      # 唯一主入口
├── model.zig                     # Model 接口定义
├── sampler.zig                   # 采样器
├── kv_cache.zig                  # KV 缓存
├── gguf.zig                      # GGUF 解析器（保留顶层，因被多处导入）
├── utils.zig                     # 通用工具（时间、日志过滤等）
├── stb_image.zig                 # 可考虑移入 vendor/
│
├── core/                         # 核心基础设施（无业务逻辑）
│   ├── mod.zig                   # 导出所有核心模块
│   ├── memory.zig                # 内存管理
│   ├── engine_common.zig         # 引擎公共函数（日志、时间）
│   └── loader/                   # 加载器子模块
│       ├── mod.zig
│       ├── weight_loader.zig
│       └── graph_loader.zig      # （可选，将 graph 相关移走）
│
├── graph/                        # 计算图相关
│   ├── mod.zig
│   ├── builder.zig               # 原 graph_builder.zig
│   ├── context.zig               # 原 graph_context.zig
│   └── executor.zig              # （如有需要，原 engine_common 的部分）
│
├── layers/                       # 神经网络层（保持原样，仅改命名）
│   ├── mod.zig
│   ├── attention.zig
│   ├── embed.zig
│   ├── linear.zig
│   ├── pooling.zig
│   ├── rms_norm.zig
│   ├── rope.zig
│   └── geglu.zig                 # 原 swiglu.zig
│
├── models/                       # 具体模型实现
│   ├── mod.zig
│   ├── registry.zig
│   ├── gemma3.zig
│   ├── gemma4.zig
│   ├── llama.zig
│   ├── qwen2.zig
│   ├── qwen35.zig
│   └── embedding.zig
│
├── tokenizer/                    # 分词器（使用 mod.zig 入口）
│   ├── mod.zig
│   ├── bpe.zig
│   ├── decode.zig
│   ├── encode.zig
│   ├── trie.zig
│   ├── types.zig
│   └── utils.zig
│
├── chat_template/                # 对话模板（使用 mod.zig 入口）
│   ├── mod.zig
│   ├── jinja.zig
│   ├── multimodal.zig
│   └── types.zig
│
├── mm/                           # 多模态（保持，可选合并 fft）
│   ├── mod.zig
│   ├── manager.zig
│   ├── audio.zig
│   ├── vision.zig
│   ├── preprocess.zig            # 合并 fft.zig 逻辑
│   └── (fft.zig 可删除)
│
├── ggml/                         # ggml 绑定（保持原结构）
│   ├── mod.zig
│   ├── backend.zig
│   ├── c.zig
│   ├── context.zig
│   ├── graph.zig
│   ├── ops.zig
│   ├── tensor.zig
│   ├── threadpool.zig
│   └── utils.zig
│
├── tests/                        # 单元测试
│   └── ...
│
├── tools/                        # 辅助工具（dump_graph, compare_logits 等）
│   └── ...
└── vendor/                       # 存放 stb_image.zig 等第三方单头文件
```

