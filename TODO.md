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
- [ ] **混合内存输入**：实现 `build_inp_mem_hybrid` 模式，统一管理 KV Cache + SSM State（提升 Qwen3.5 混合架构效率）
- [ ] **graph 类分离**：参考 llama.cpp 设计，将 `forward()` 拆分为独立 graph 类（便于复用和优化）
- [ ] **Jinja 模板引擎**：纯 Zig 实现，支持 GGUF 内置 `tokenizer.chat_template` 解析（覆盖 90%+ 模型）
- [ ] **多模态对话模板完整闭环**：Jinja 模板中的 `<|image|>`/`<|audio|>` 占位符展开 → 嵌入注入
- [ ] **线程池**：持久化 `ggml_threadpool`（需 ggml 升级后启用）
- [ ] **减少 `ggml_cont` 调用**：消除不必要的内存重排
- [ ] **预分配 SSM 状态张量**：避免运行时分配

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
