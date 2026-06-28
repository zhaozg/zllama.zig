# 音频推理数据对齐

## zllama 数据对齐分析方法

1. 通过命令: `zig-out\/bin\/zllama -m ~\/.cache\/models\/gemma-4-E2B-it-Q4_K_M.gguf --mmproj ~\/.cache\/models\/mmproj-BF16.gguf --audio ~\/.cache\/models\/hello.wav -p " "` 会在 `debug_audio` 目录下生成 zllama_audio_ 开头的过程处理数据。
2. `debug_audio` 目录下 `llama_audio_*.json` 为对应的 llama.cpp mtmd 生成的中间数据。
3. 运行差异比较工具: `tools\/compare-float.py --file1 debug_audio\/llama_audio_xxx.json --file2 debug_audio\/zllama_audio_xxx.json`
4. 根据差异，对 `src\/mtmd\/audio` 下的程序进行对齐修改。
5. `zig build -Doptimize=ReleaseSafe` 后重新运行 zllama 重新生成数据。
6. 再次进行数据差异对比分析。

## 📊 对齐数据对比表（按处理流程排序）

| 序号 | 阶段说明 | llama 大小 | zllama 大小 | 大小匹配 | 余弦相似度 | 验证结果 |
|:---:|---------|-----------|------------|:--------:|:----------:|:--------:|
| 1 | 原始音频样本输入 | 138,080 | 138,080 | ✅ | 1.000000 | ✅ 完全一致 |
| 2 | Mel 频谱 | 110,883 | 110,883 | ✅ | 1.000000 | ✅ 完全一致 |
| 3 | Conv1d 第0层权重 | 12,346 | 12,346 | ✅ | 1.000000 | ✅ 权重一致 |
| 4 | Conv1d 第1层权重 | 386,451 | 386,451 | ✅ | 0.999999 | ✅ 权重一致 |
| 5 | **Conv2d 第0层输出** | **3,290,186** | **3,277,425** | ❌ | **0.903509** | ⚠️ llama.cpp 数据不可靠 |
| 6 | **Conv2d 第1层输出** | **205,040** | **205,040** | ✅ | **1.000000** | ✅ **完全一致** |
| 7 | **Flatten 输出** | **214,610** | **205,040** | ❌ | **0.046558** | ⚠️ llama.cpp 数据不可靠 |
| 8 | **Input Projection 输出** | **216,705** | **213,634** | ❌ | **-0.068431** | ⚠️ llama.cpp 数据不可靠 |
| 9 | **最终音频嵌入** | **322,907** | **322,559** | ❌ | **0.163728** | ❌ **显著差异** |
| 10 | 注意力掩码 | 9,782 | 9,782 | ✅ | 1.000000 | ✅ 完全一致 |
| 11 | 位置编码 | 134,107 | 134,107 | ✅ | 1.000000 | ✅ 完全一致 |
| 12 | 输入投影权重 | 10,997,223 | 10,997,223 | ✅ | 1.000000 | ✅ 权重一致 |

---

### 🔍 关键结论

#### ✅ 已验证一致的数据（7项）
| 数据项 | 余弦相似度 | 状态 |
|-------|-----------|:----:|
| 原始音频样本输入 | 1.000000 | ✅ |
| Mel 频谱 | 1.000000 | ✅ |
| Conv1d 第0层权重 | 1.000000 | ✅ |
| Conv1d 第1层权重 | 0.999999 | ✅ |
| **Conv2d 第1层输出** | **1.000000** | ✅ **关键验证点** |
| 注意力掩码 | 1.000000 | ✅ |
| 位置编码 | 1.000000 | ✅ |
| 输入投影权重 | 1.000000 | ✅ |

> **Conv2d 第1层输出完全一致**（余弦相似度=1.0）是本次验证的关键发现，它证明：
> - Conv2d 参数顺序正确（kernel 在前，input 在后）
> - Conv2d 的 stride/padding/dilation 参数一致
> - LayerNorm + ReLU 后处理逻辑一致
> - 权重数据加载正确

#### ⚠️ llama.cpp 中间张量数据不可靠（4项）
llama.cpp 在 `gemma4a.cpp` 中对以下中间张量调用了 `ggml_set_input()`，导致 Gallocr 分配器不为这些张量分配内存，因此数据是**未初始化的内存垃圾值**，不可用于对比：

| 数据项 | 余弦相似度 | 原因 |
|-------|:----------:|:----:|
| Conv2d 第0层输出 | 0.903509 | `ggml_set_input` 导致未初始化内存 |
| Flatten 输出 | 0.046558 | `ggml_set_input` 导致未初始化内存 |
| Input Projection 输出 | -0.068431 | `ggml_set_input` 导致未初始化内存 |
| 最终音频嵌入 | 0.163728 | 唯一可靠但仍有差异 |

#### ❌ 最终音频嵌入存在显著差异
最终嵌入输出（`llama_audio_embeddings.json`）是唯一**没有**被 `ggml_set_input` 标记的可靠数据，但余弦相似度仅 0.16，说明 Conformer 编码器内部存在未对齐的计算逻辑。

---

### 📝 详细分析记录

#### 2024-06-28 分析与修复

**已发现并修复的问题：**

1. **encoder_input 保存的是计算图节点的数据（全 0）**
   - 在 `encoder.zig` 的 `encode()` 中，`cur` 是 `cont(transpose(inp_raw))` 的输出，这是一个计算图操作节点
   - 在 `graph.compute()` 之前保存 `cur.dataF32()` 得到的是未初始化的数据（全 0）
   - **修复**：改为保存 `inp_raw.dataF32()`（有实际 mel 数据）
   - 文件：`src/mtmd/audio/encoder.zig` 第 336 行

2. **`norm()` 函数错误地调用了 `ggml_rms_norm` 而不是 `ggml_norm`**
   - Conv2d 后的 LayerNorm 应该使用标准 LayerNorm（`ggml_norm`），而不是 RMSNorm（`ggml_rms_norm`）
   - `Tensor.norm()` 方法之前调用的是 `ggml_rms_norm`，导致 Conv2d 后的归一化错误
   - **修复**：将 `norm()` 改为调用 `ggml_norm`
   - 文件：`src/ggml/tensor.zig` 第 92-94 行

3. **`melToTensor()` 已集成到音频处理流水线**
   - 在 `engine.zig` 的 `generateWithAudio()` 中，在 `computeMelSpectrogram()` 之后调用 `melToTensor()` 创建 ggml 张量
   - 在 `compare_mtmd_audio.zig` 中也同样调用 `melToTensor()`
   - 匹配设计文档 `MTMD_ARCHITECTURE.md` 第5节音频处理流水线的第4步

4. **`AudioEncoder.encode()` 接口已更新**
   - 现在接收 `*ggml.Tensor`（由 `melToTensor()` 创建）而不是原始 `[]const f32`
   - 保留了 `encodeRaw()` 作为向后兼容的回退接口
   - 匹配设计文档第5步

5. **`MediaInput` 已扩展**
   - 添加了 `mel_tensor: ?*ggml.Tensor` 字段
   - `encodeMedia()` 优先使用 `mel_tensor`，回退到 `mel_data`

**待解决的问题：**

6. **最终嵌入输出仍然有显著差异（余弦相似度 ~0.16）**
   - Conv2d 第1层输出已完全一致（余弦相似度=1.0），说明 Conv2d 参数顺序正确
   - 差异出现在 Conformer 编码器内部（12 层 Conformer blocks）
   - 可能的原因：
     a. Conformer 层的权重加载顺序或命名不匹配
     b. 注意力计算中的 RPE 或 mask 逻辑差异
     c. FFN 或卷积模块的实现差异
   - 需要进一步逐层排查

---

### 🔬 下一步排查方向

1. **在 Conformer 层之间添加调试输出**
   - 在每层 Conformer block 之后保存中间张量数据
   - 与 llama.cpp 的对应层输出对比，定位第一个出现差异的层

2. **检查 Conformer 层权重加载**
   - 确认每层权重的命名和形状与 GGUF 文件一致
   - 特别关注 `attn_k_rel.weight`（RPE 投影）和 `per_dim_scale.weight`（Q/K 缩放）

3. **检查注意力计算中的 RPE 逻辑**
   - RPE 的 `matrix_bd` 计算（pad + reshape + view + cont）容易出错
   - 与 llama.cpp 的 `clip_graph_gemma4a::build()` 逐行对比

4. **检查卷积模块（GLU + depthwise conv）**
   - `ssmConv` 的参数顺序和实现
   - GLU gate 的切片逻辑

5. **使用 `compare_mtmd_audio` 工具进行端到端对比**
   - 生成参考 logits 并与 zllama 输出对比
   - 命令：`zllama-compare-mtmd-audio --model model.gguf --mmproj mmproj.gguf --audio hello.wav --prompt " " --ref-logits ref.bin`

---

### 📋 验证命令速查

```bash
# 生成 zllama 调试数据
echo "/exit" | ./zig-out/bin/zllama -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj ~/.cache/models/mmproj-BF16.gguf --audio ~/.cache/models/hello.wav -p " "

# 对比 Mel 频谱
python3 tools/compare-float.py --file1 debug_audio/llama_audio_mel.json --file2 debug_audio/zllama_audio_mel.json --verbose

# 对比 Conv2d 第1层输出（已验证一致）
python3 tools/compare-float.py --file1 debug_audio/llama_audio_conv2d_1_output.json --file2 debug_audio/zllama_audio_conv2d_1_output.json --verbose

# 对比最终嵌入
python3 tools/compare-float.py --file1 debug_audio/llama_audio_embeddings.json --file2 debug_audio/zllama_audio_embeddings.json --verbose

# 批量对比所有数据
for f in llama_audio_samples_input.json llama_audio_mel.json llama_audio_conv1d_0_weight.json llama_audio_conv1d_1_weight.json llama_audio_conv2d_0_output.json llama_audio_conv2d_1_output.json llama_audio_flatten_output.json llama_audio_input_proj_output.json llama_audio_embeddings.json llama_audio_attn_mask.json llama_audio_pos_emb.json llama_audio_input_proj_weight.json; do echo "--- $f ---"; python3 tools/compare-float.py --file1 "debug_audio/$f" --file2 "debug_audio/z$f" 2>&1 | grep -E "^(对比|📉|🔗|📌|长度)"; done
```
