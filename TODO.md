# TODO.md — zllama.zig 待办事项

> 优先级：P0（阻塞发布）> P1（关键功能）> P2（扩展与优化）> P3（生态/后端）

## P0 — 核心正确性（必须完成才能发布）

- [x] **端到端推理数值对比**：已创建 `tools/compare_with_llamacpp.zig` 工具，可加载模型运行推理并与 llama.cpp 参考 logits 二进制文件对比 NMSE/余弦相似度（`zig build compare-llamacpp`）
- [x] **Qwen3-Embedding 输出精度**：已在 `qwen2.zig` LayerWeights/forward/loadWeights 中添加 `attn_q_norm_weight`/`attn_k_norm_weight` 可选权重，在 Q/K 投影后、RoPE 前应用 RMSNorm + per-head weight；`embedding.zig` 同步修复
- [x] **多模态视觉输出质量验证**：已创建 `tools/compare_mtmd_vision.zig` 工具（`zig build compare-mtmd-vision`），实现图像加载→视觉编码→forwardWithEmbdOverride→logits 对比的完整管线，需 llama.cpp mtmd 生成参考 logits 后运行验证
- [x] **多模态音频输出质量验证**：已创建 `tools/compare_mtmd_audio.zig` 工具（`zig build compare-mtmd-audio`），实现 WAV 加载→Mel 频谱→Conformer 编码→forwardWithEmbdOverride→logits 对比的完整管线，需真实音频文件 + llama.cpp mtmd 参考输出后运行验证
- [x] **SSM 状态重置完整性**：已在 `/reset`、`/new` 命令、`generate()`、`chatLoop` 文本处理器、`generateWithImage()`、`generateWithAudio()` 所有推理路径中添加 `self.model.resetSSMStates()` 调用

## P1 — 关键功能与性能

- [x] **多模态输出质量与 `llama-mtmd-cli` 交叉验证**：已创建 `tools/mtmd_ref_logits.cpp` 工具（C++，链接 libllama + libmtmd），可加载模型+mmproj+媒体文件，运行 mtmd 推理并输出 logits 到二进制文件。视觉和音频参考 logits 均已成功生成并验证（`zig build compare-mtmd-vision` / `zig build compare-mtmd-audio` 配合 `--ref-logits` 使用）
- [x] **混合内存输入**：已实现 `build_inp_mem_hybrid` 模式 — 在 `memory.zig` 中添加 `HybridMemory` 实现，统一管理 KV Cache + SSM State；`qwen35.zig` 中 `allocateSSMStates()` 方法在 `setKVCacheContext` 时预分配所有持久张量，`resetSSMStates` 改为清零而非置 null
- [x] **graph 类分离**：参考 llama.cpp 设计，将 `forward()` 拆分为独立 graph 类（便于复用和优化）
  - [x] Gemma4: `Gemma4Graph` 结构体（`build()` / `buildWithEmbd()` / `buildMediaOnly()` → `transformerForward()` → `buildLayer()` / `buildAttention()` / `buildOutput()`）
  - [x] Qwen35: `Qwen35Graph` 结构体（`build()` → `buildFullAttnLayer()` / `buildSSMLayer()` / `buildOutput()`）
  - [x] 模型 `forward()` 方法变为 thin wrapper 委托给 graph 类
- [x] **多模态对话模板完整闭环**：使用针对不同模型的内置模板，支持 `<|image|>`/`<|audio|>` 占位符展开 → 嵌入注入
  - [x] tokenizeWithPlaceholders 记录占位符 token 偏移（token_offset）
  - [x] forwardWithEmbdOverride 使用正确的 embd_offset（从 token_offset 计算）
  - [x] forwardMediaOnly 方法（Gemma4 媒体-only 非因果前向，跳过 per-layer embedding）
  - [x] 三阶段 prefill（text prefix causal → media non-causal → text suffix causal）— ✅ 已实现并验证通过
  - [x] per_layer_embd 在媒体位置也需要 override — `forwardMediaOnly` 新增 `input_tokens` 参数，透传至 `transformerForward` 进行 per-layer embedding 查找
  - [x] 音频、图片相关的额外检查:
    - [x] **占位符扩展**：`tokenizeWithPlaceholders` 是否正确将单个 `<|audio|>` token 扩展为 `n_audio_tokens` 个连续 token？, log.debug 打印 `expanded.tokens` 确认。— ✅ 已在 `multimodal.zig` 的 `tokenizeWithPlaceholders` 末尾添加 `log.debug` 打印格式化字符串、每个占位符的类型/位置/token_offset/token_count/token_id、以及展开后的 token 序列预览。
    - [x] **嵌入替换**：三阶段预填充中 `media_offset` 和 `media_count` 是否准确对应音频嵌入的帧数？log.info 打印确保。— ✅ 已在 `engine.zig` 的 `multimodalPrefill` 开头添加 `logger.info` 打印 media_offset/media_count/n_media_tokens/prefix_tokens/suffix_tokens/n_total_tokens，并在不匹配时发出 `logger.warn`；同时在 `prefill.zig` 的 `threeStagePrefill` 开头添加 `log.info` 打印三阶段（prefix/media/suffix）的 token 数量和位置范围。
- [x] **减少 `ggml_cont` 调用**：已审查代码，现有 `ggml_cont` 调用均用于 view/permute/concat 后的必要内存重排，无冗余调用可移除
- [x] **预分配 SSM 状态张量**：已在 `qwen35.zig` 中实现 — `allocateSSMStates()` 方法使用 `ctx_kv_cache` 预分配 conv_state 和 ssm_state 张量，并在 `buildSSMLayer` 中移除 lazy-init 模式，改为前置检查 `SSMStateNotPreallocated` 错误
- [x] **deps/media.md 媒体推理排查系统**：按照 deps/media.md 四项要求，系统性地添加 logger.debug 信息输出以确认正确性：
  - [x] **嵌入维度检查**：`engine.zig` 的 `generateWithImage` / `generateWithAudio` 中编码器输出后打印 shape 并与 model n_embd 对比（期望 1536）；`gemma4.zig` 的 `buildMediaOnly` / `buildWithEmbd` 中打印 override_embd_dim vs model_n_embd，不匹配时 `log.err` 返回错误。
  - [x] **mediaForward 回调审查**：`prefill.zig` Pass 2 前后打印 `causal=false`、`start_pos`、`n_tokens`、`embd_dim`；`gemma4.zig` 的 `transformerForward` 入口打印 `causal` 和 `kv_cache` 状态；`buildMediaOnly` 入口打印参数并注明 non-causal。
  - [x] **KV Cache 状态检查**：`prefill.zig` 的 Pass 1/2/3 各阶段 compute 完成后打印 `kv_cache_mgr.currentLen()`，确认媒体段处理前后 KV Cache 逐段递增。
  - [x] **Token 序列逐步验证**：`engine.zig` 的 `multimodalPrefill` 中打印 prefix/suffix token 的 head+tail 片段及 pos 范围；`prefill.zig` 的三阶段 info 日志已含完整位置范围。

## P2 — 扩展功能与模型支持

- [ ] **Qwen3.5 混合架构完善**：
  - [ ] MTP 解码头（`graph_mtp` 类）
  - [ ] 更多量化格式（Q5_K_M, Q6_K, Q8_0）
  - [ ] 架构改进（`llm_graph_input_rs`、`llm_graph_input_attn_kv` 接口）
- [ ] **与更多多模态模型联调**：Qwen2-VL、LLaVA 等
- [ ] **流式 CLI 增强**：实时输出、交互命令扩展（如 `/image`, `/audio`）
- [ ] **性能基准测试**：自动化，对比参考实现（PP/TG 延迟、吞吐量）

## P3 — 生态工具与后端加速

- [ ] **CI（GitHub Actions）**：编译、单元测试、算子数值测试
- [ ] **预编译发布**：二进制产物（Linux/macOS/Windows）
- [ ] **Metal 后端**（`-Dmetal`，`--backend metal`，macOS GPU 加速）
- [ ] **CUDA 后端**（`-Dcuda`，`--backend cuda`，Linux GPU 加速）
