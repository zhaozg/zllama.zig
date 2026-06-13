# TODO.md — zllama.zig 待办事项

> 优先级：P0（阻塞发布）> P1（关键功能）> P2（扩展与优化）> P3（生态/后端）

## P0 — 核心正确性（必须完成才能发布）

- [ ] **端到端推理数值对比**：与 llama.cpp 输出对比（NMSE/余弦相似度），确保各架构（Qwen3.5、Gemma4、Llama）正确性
- [ ] **Qwen3-Embedding 输出精度**：当前利用 `mulMat` 隐式 GQA 广播实现双向注意力，但未应用 Q/K normalization，输出与参考有偏差
- [ ] **多模态视觉输出质量验证**：与 llama.cpp mtmd 对比 logits，确保三阶段 prefill 产生正确结果
- [ ] **多模态音频输出质量验证**：使用真实音频文件验证转录结果
- [ ] **SSM 状态重置完整性**：确认 `resetModelSSMStates` 在所有推理路径（prefill、增量解码、多轮对话）均正确调用

## P1 — 关键功能与性能

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
