# 文件组织约定


## ✅ 优化目标

- **模块化**：每个子目录有明确的 `mod.zig` 作为公开 API 入口。
- **扁平化顶层**：顶层只保留真正全局的文件（如 `main.zig`、`model.zig`、`sampler.zig`、`gguf.zig`、`utils.zig`）。
- **语义清晰**：目录名能直观反映其内容职责。
- **符合 Zig 惯例**：使用 `mod.zig`，避免同名文件和目录混淆。

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
│
│
├── models/                       # 模型辅助工具
│   └── templates/                # jinja file for models
│
└── vendor/                       # 存放 stb_image.zig 等第三方单头文件
```

---

## 🔧 具体优化操作建议

| 当前 | 问题 | 优化操作 |
|------|------|----------|
| `src/chat_template.zig` + `src/chat_template/` | 入口与目录重复 | 删除 `chat_template.zig`，创建 `chat_template/mod.zig` 作为公开 API，内部 `pub usingnamespace` 各子模块。 |
| `src/tokenizer.zig` + `src/tokenizer/` | 同上 | 同上。 |
| `src/ggml.zig` + `src/ggml/` | 同上 | 同上（但需注意 `ggml.zig` 可能已被其他文件导入，修改导入路径）。 |
| `src/simple_main.zig` `tokenize_main.zig` | 多个入口 | 将 `simple_main.zig` 移入 `examples/` 或删除（功能可由 `main.zig` 参数覆盖）；`tokenize_main.zig` 移至 `tools/`。 |
| `src/core/` 内容混杂 | 内聚性差 | 将 `graph_builder.zig`、`graph_context.zig` 移至新目录 `graph/`；`engine_common.zig` 移至 `core/`（保留）；`loader.zig` 和 `weight_loader.zig` 移至 `core/loader/`。 |
| `src/layers/swiglu.zig` | 命名不准确 | 重命名为 `geglu.zig`，并修改所有引用。 |
| `src/mm/fft.zig` | 单一用途 | 合并到 `mm/preprocess.zig` 中，删除独立文件。 |
| `src/stb_image.zig` | 第三方依赖 | 创建 `vendor/` 目录，移入。 |
| `src/utils.zig` 内容少 | - | 保留，可将 `engine_common` 中日志、时间函数移入。 |

---

## 📝 导入路径变化示例

优化后，导入将更统一：

```zig
// 旧
const chat_template = @import("chat_template");
const tokenizer = @import("tokenizer");
const ggml = @import("ggml");

// 新
const chat_template = @import("chat_template");
const tokenizer = @import("tokenizer");
const ggml = @import("ggml");
// 注意：现在 chat_template/mod.zig 必须重新导出必要的符号。
// 使用 `mod.zig` 模式后，用户导入路径不变（因为 Zig 允许 `@import("dir")` 自动寻找 `dir/mod.zig`）。
```

因此，只需重命名文件，外部引用**无需改动**（前提是 `mod.zig` 重新导出了原有符号）。这是 Zig 的一大便利之处。

---

## ✅ 优化收益

1. **清晰度提升**：每个目录职责明确，新人能快速定位代码。
2. **可维护性**：减少顶层文件数量，降低认知负担。
3. **模块化**：便于后续拆分到不同编译单元或动态库。
4. **一致性**：所有子模块都遵循 `mod.zig` 惯例，统一导入风格。

---

## 🗂️ 当前结构主要问题

1. **入口文件与子目录并存**
   - `src/chat_template.zig` + `src/chat_template/`
   - `src/tokenizer.zig` + `src/tokenizer/`
   这种模式容易混淆：不知道是应该 `@import("chat_template")` 还是 `@import("chat_template/mod.zig")`。Zig 推荐在子目录下放 `mod.zig` 作为模块入口。

2. **`core/` 语义模糊**
   该目录包含 `engine_common.zig`、`graph_builder.zig`、`graph_context.zig`、`loader.zig`、`memory.zig`、`weight_loader.zig`。其中既有图相关组件，又有加载器和内存工具，耦合度低但内聚性差，可拆分为更明确的子模块。

3. **多个 main 入口**
   `main.zig`、`simple_main.zig`、`tokenize_main.zig` 共存，容易造成维护困惑。通常一个项目只有一个主入口，其他变体应移至 `examples/` 或 `tools/`。

4. **部分文件命名不统一**
   - `swiglu.zig` 实际实现的是 GeGLU（GELU 门控），建议改名 `geglu.zig`。
   - `engine_common.zig` 名称泛泛，且包含日志过滤、时间戳等通用函数，可考虑移至 `utils.zig` 或 `core/utils.zig`。

5. **`tools/` 和 `tests/` 并行**
   测试放在 `tests/`，调试工具放在 `tools/`，清晰合理，无需变动。

---

