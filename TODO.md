# TODO.md — zllama.zig 待办事项

> 优先级：P0（阻塞发布）> P1（关键功能）> P2（扩展与优化）> P3（生态/后端）

## P0 — 核心正确性（必须完成才能发布）

- [x] **端到端推理数值对比**：`tools/compare_with_llamacpp.zig` 工具，对比 llama.cpp 参考 logits（NMSE/余弦相似度）
- [x] **Qwen3-Embedding 输出精度**：`qwen2.zig` 添加 `attn_q_norm_weight`/`attn_k_norm_weight` 可选权重
- [x] **多模态视觉输出质量验证**：`tools/compare_mtmd_vision.zig` 工具，图像→视觉编码→forwardWithEmbdOverride→logits 对比
- [x] **多模态音频输出质量验证**：`tools/compare_mtmd_audio.zig` 工具，WAV→Mel→Conformer→logits 对比
- [x] **SSM 状态重置完整性**：所有推理路径均已调用 `resetSSMStates()`

## P1 — 关键功能与性能

- [x] **多模态交叉验证**：`tools/mtmd_ref_logits.cpp` 工具，视觉和音频参考 logits 均已生成并验证
- [x] **混合内存输入**：`memory.zig` 添加 `HybridMemory`，统一管理 KV Cache + SSM State
- [x] **graph 类分离**：Gemma4 → `Gemma4Graph`，Qwen35 → `Qwen35Graph`，模型 `forward()` 变为 thin wrapper
- [x] **多模态对话模板闭环**：`<|image|>`/`<|audio|>` 占位符展开 → 嵌入注入 → 三阶段 prefill
- [x] **减少 `ggml_cont` 调用**：审查完成，现有调用均为必要内存重排
- [x] **预分配 SSM 状态张量**：`qwen35.zig` 的 `allocateSSMStates()` 方法
- [x] **媒体推理排查系统**：按 `deps/media.md` 四项要求添加 logger.debug 输出
- [x] **vocab 对齐遗留问题**：修复 test_mtmd.zig 中 4 处内存泄漏，167 测试通过，0 泄漏

## P2 — 扩展功能与模型支持

- [ ] **Qwen3.5 混合架构完善**：
  - [ ] MTP 解码头（`graph_mtp` 类）
  - [ ] 更多量化格式（Q5_K_M, Q6_K, Q8_0）
  - [ ] 架构改进（`llm_graph_input_rs`、`llm_graph_input_attn_kv` 接口）
- [ ] **Qwen3VL 多模态完善**：文本+视觉推理闭环验证
- [ ] **流式 CLI 增强**：实时输出、交互命令扩展（`/image`, `/audio`）
- [ ] **性能基准测试**：自动化，对比参考实现（PP/TG 延迟、吞吐量）

## P3 — 生态工具与后端加速

- [ ] **CI（GitHub Actions）**：编译、单元测试、算子数值测试
- [ ] **预编译发布**：二进制产物（Linux/macOS/Windows）
- [ ] **Metal 后端**（`-Dmetal`，`--backend metal`，macOS GPU 加速）
- [ ] **CUDA 后端**（`-Dcuda`，`--backend cuda`，Linux GPU 加速）
