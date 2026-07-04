# MTMD 音频编码对齐分析报告

> 分析 zllama.zig (gemma4a.zig 的 `buildGraphEx`) 与 llama.cpp (`clip_image_batch_encode`) 在音频编码路径上的差异，
> 目标是对齐 `llama_audio_input_proj_output` 数据。
> 请对照关键实现 `./src/mtmd/models/gemma4a.zig` 与 `./deps/llama.cpp/tools/mtmd/models/gemma4a.cpp` 进行仔细比较。

## 测试条件说明

- 模型相同（Gemma 4 E2B）
- mmproj 相同
- 输入音频相同（hello.wav）

## 文件生成顺序

| 序号 | 文件名 | 说明 |
|:---:|:---|:---|
| 1 | `llama_audio_samples_input.json` | **原始音频样本**（输入数据） |
| 2 | `llama_audio_mel.json` | **梅尔频谱特征**（从样本计算得出） |
| 3 | `llama_audio_encoder_input.json` | **转置后的梅尔频谱**（`transpose` + `cont`，送入 Conv2d 前的形状） |
| 4 | `llama_audio_conv1d_0_weight.json` | **Conv2d 第0层权重**（`a.conv1d.0.weight`，用于第1个卷积） |
| 5 | `llama_audio_conv2d_0_output.json` | **Conv2d 第0层输出**（经过 Conv2d + Bias + LayerNorm + ReLU） |
| 6 | `llama_audio_conv1d_1_weight.json` | **Conv2d 第1层权重**（`a.conv1d.1.weight`，用于第2个卷积） |
| 7 | `llama_audio_conv2d_1_output.json` | **Conv2d 第1层输出**（经过 Conv2d + Bias + LayerNorm + ReLU） |
| 8 | `llama_audio_flatten_output.json` | **展平后的特征**（`permute` + `reshape_2d`） |
| 9 | `llama_audio_input_proj_weight.json` | **输入投影权重**（`a.input_projection.weight`，映射到 Conformer 维度） |
| 10 | `llama_audio_input_proj_output.json` | **输入投影输出**（Conformer 编码器的输入） |
| 11 | `llama_audio_pos_emb.json` | **相对位置编码**（RPE，在 Conformer 注意力中使用） |
| 12 | `llama_audio_attn_mask.json` | **分块注意力掩码**（在 Conformer 注意力中使用） |
| 13 | `llama_audio_embeddings.json` | **最终音频嵌入**（Conformer 编码器 + 输出投影） |

---

## 最新对齐状态（2026-06-30 最终确认）

| 数据项 | 余弦相似度 | 状态 |
|-------|-----------|:----:|
| `encoder_input` | 1.0 | ✅ |
| `conv2d_0_output` | 1.0 | ✅ |
| `conv2d_1_output` | 1.0 | ✅ |
| `after_cont`（permute+cont 后） | 1.0 | ✅ |
| `flatten_output` | 1.0 | ✅ |
| `input_proj_output` | **-0.002** | ❌ **几乎正交** |
| `conformer_blocks_output` | 0.749 | ❌ **显著差异（源于 input_proj）** |
| `embeddings` | 0.778 | ❌ **显著差异（源于 input_proj）** |

**所有前置步骤（输入、Mel、卷积、Flatten）均已完美对齐，**

## 任务（已完成 ✅）

> 已通过 commit `dca83df` 修复。`buildMMWithClamp` 现在已正确用于 `w.sscp_inp_proj_w`。

经过排查后发现 `w.sscp_inp_proj_w` 在 Zig 代码中未调用 `buildMMWithClamp`，而 gemma4a.cpp 是进入了 if 分支的。已修复：

1. **Input Projection (`sscp_inp_proj_w`)**：现在使用 `buildMMWithClamp`（带 clamp），与 C++ `build_mm` 一致。
2. **FFN 权重**：新增 `buildFFNWithClamp` 函数，使用 `buildMMWithClamp` 进行所有权重投影（up/gate/down），两个 FFN 调用均已替换。
3. **`attn_k_rel_w`**：C++ 使用纯 `ggml_mul_mat`（无 clamp），Zig 已对齐为 `k_rel.mulMat(ctx, pos_emb)`。

当前代码（`src/mtmd/graph/models/gemma4a.zig` 第 170-171 行）：
```zig
if (w.sscp_inp_proj_w) |proj_w| {
    cur = buildMMWithClamp(ctx, proj_w, cur, clamp_map);
    if (w.sscp_inp_proj_b) |proj_b| {
        cur = cur.add(ctx, proj_b);
    }
}
```

### 调用链说明

`buildGraphEx` 通过以下间接路径被调用：

```
MultiModalManager.encodeMedia()          (src/mtmd/mod.zig:317)
  → AudioEncoder.encode()                (src/mtmd/audio/encoder.zig:123)
    → backend.buildGraph()               (AudioEncoderBackend 函数指针)
      → buildGraphFromWeights()          (src/mtmd/graph/models/gemma4a.zig:578)
        → buildGraphEx()                 (src/mtmd/graph/models/gemma4a.zig:103)
```

`buildGraph`（第 82-90 行，GraphBuilder 版本）目前未被音频编码器调用，它保留用于 vision 管线兼容。

### 一、结构与流程对应（✅ 匹配）

| 步骤 | C++ (`gemma4v.cpp`) | Zig (`encoder.zig`) | 一致性 |
|------|---------------------|----------------------|--------|
| 1. 输入转置 | `ggml_transpose` + `ggml_cont` | `ggml.transpose` + `ggml.cont` | ✅ |
| 2. 两层 Conv2D (stride=2, padding=1) + 后处理 | `ggml_conv_2d` + bias + LayerNorm + ReLU | `cur.conv2d` + add + norm + ReLU | ✅ |
| 3. Flatten + Input Projection | `permute` + `reshape_2d` + `mul_mat` | 相同 | ✅ |
| 4. **Input Projection（mul_mat）** | `build_mm`（带 clamp） | `buildMMWithClamp`（带 clamp） | ✅ **已修复** |
| 5. 12 层 Conformer | FFN → 分块注意力 → 卷积模块 → FFN → Norm | 完全相同的块序列 | ⚠️ 因 Input Projection 错误，后续均受影响 |
| 6. 输出投影 + RMSNorm | `mul_mat` + `rms_norm` + `mul` | 相同 | ⚠️ 受 Input Projection 影响 |

---

### 二、已验证一致的实现细节

| 操作 | C++ | Zig | 一致性 |
|------|-----|-----|:------:|
| `conv2d` 参数顺序 | `ggml_conv_2d(ctx, kernel, input, ...)` | `cur.conv2d(ctx, kernel, ...)` → `ggml_conv_2d(ctx, kernel, cur, ...)` | ✅ |
| `mulMat` 参数顺序 | `ggml_mul_mat(ctx, w, x)` | `w.mulMat(ctx, x)` → `ggml_mul_mat(ctx, w, x)` | ✅ |
| `ssmConv` 参数顺序 | `ggml_ssm_conv(ctx, x, kernel)` | `x.ssmConv(ctx, kernel)` → `ggml_ssm_conv(ctx, x, kernel)` | ✅ |
| `cont2d` | `ggml_cont_2d(ctx, x, ne0, ne1)` | `x.cont2d(ctx, ne0, ne1)` → `ggml_cont_2d(ctx, x, ne0, ne1)` | ✅ |
| `view2d` | `ggml_view_2d(ctx, x, ne0, ne1, nb1, offset)` | `x.view2d(ctx, ne0, ne1, nb1, offset)` | ✅ |
| `view4d` 重叠窗口 | `ggml_view_4d(ctx, t, D, H, S, B, nb1, nb2, C*nb2, 0)` | `t.view4d(ctx, D, H, S, B, nb1, nb2, C*nb2, 0)` | ✅ |
| `pad` | `ggml_pad(ctx, t, 0, 0, pad, 0)` | `t.pad(ctx, 0, 0, pad, 0)` | ✅ |
| `roll` | `ggml_roll(ctx, t, 0, 0, P, 0)` | `t.roll(ctx, 0, 0, P, 0)` | ✅ |
| `permute` | `ggml_permute(ctx, t, 0, 3, 1, 2)` | `t.permute(ctx, 0, 3, 1, 2)` | ✅ |
| `reshape3d` | `ggml_reshape_3d(ctx, t, D, H, N)` | `t.reshape3d(ctx, D, H, N)` | ✅ |
| `reshape4d` | `ggml_reshape_4d(ctx, t, D, H, C, B)` | `t.reshape4d(ctx, D, H, C, B)` | ✅ |
| RMSNorm | `ggml_rms_norm(ctx, x, eps)` + `ggml_mul(ctx, x, w)` | `x.rmsNorm(ctx, eps).mul(ctx, w)` | ✅ |
| LayerNorm | `ggml_norm(ctx, x, eps)` + `ggml_mul(ctx, x, w)` | `x.norm(ctx, eps).mul(ctx, w)` | ✅ |
| GLU gate | `ggml_view_2d(x, d, n, nb1, d*nb0)` + sigmoid | `x.view2d(d, n, nb1, d*sizeof(f32))` + sigmoid | ✅ |
| GLU transpose | `ggml_transpose(x)` (2D) | `x.permute(1, 0, 2, 3)` (2D 等价) | ✅ |
| Softcap | `scale(1/cap)` → `tanh` → `scale(cap)` | 相同 | ✅ |
| `k_scale` | `logf(1+exp(1))/logf(2)` = `log2(1+e)` | `@log2(1+@exp(1))` | ✅ |

---

### 三、当前唯一未对齐环节：Input Projection 的 `buildMMWithClamp`

#### 现象
- 输入 `flatten_output`：形状 `[1024, 20]`，数值与 C++ 完全一致（余弦相似度 1.0）。
- 权重 `input_proj_weight`：形状 `[1024, 1024]`，数值完全一致。
- 输出 `input_proj_output`：形状 `[1024, 20]`，但与 C++ 几乎正交（余弦 -0.002），平均绝对差 3.5，最大差 114。

> **注意**：上述差异是在修复前测量的。修复后（commit `dca83df`）`buildMMWithClamp` 已正确用于 `sscp_inp_proj_w`，需要重新运行测试验证。

---

### 四、验证结果总结

#### ✅ 已验证一致的数据
| 数据项 | 余弦相似度 | 状态 |
|-------|-----------|:----:|
| 原始音频样本输入 | 1.0 | ✅ |
| Mel 频谱 | 1.0 | ✅ |
| Conv1d 第0层权重 | 1.0 | ✅ |
| Conv1d 第1层权重 | 1.0 | ✅ |
| Conv2d 第0层输出 | 1.0 | ✅ |
| Conv2d 第1层输出 | 1.0 | ✅ |
| 输入投影权重 | 1.0 | ✅ |
| 位置编码 | 1.0 | ✅ |
| 注意力掩码 | 1.0 | ✅ |
| `after_cont` | 1.0 | ✅ |
| Flatten 输出 | 1.0 | ✅ |

#### ❌ 未对齐的数据（待重新验证）
| 数据项 | 余弦相似度 | 状态 |
|-------|-----------|:----:|
| `input_proj_output` | -0.002 | ❌ **待重新验证（修复后）** |
| `conformer_blocks_output` | 0.749 | ❌ **待重新验证（修复后）** |
| `embeddings` | 0.778 | ❌ **待重新验证（修复后）** |

---

### 五、已排查但未发现差异的操作

以下操作已逐行对比 C++ 和 Zig 实现，确认一致：

1. **Conv2d 参数顺序** ✅ — `ggml_conv_2d(ctx, kernel, input, ...)`
2. **mulMat 参数顺序** ✅ `ggml_mul_mat(ctx, w, x)`，
3. **ssmConv 参数顺序** ✅ — `ggml_ssm_conv(ctx, input, kernel)`
4. **cont2d 语义** ✅ — `ggml_cont_2d(ctx, x, ne0, ne1)`
5. **view2d/view4d 语义** ✅ — 包括重叠窗口的步长计算
6. **pad/roll 语义** ✅ — 维度顺序和方向一致
7. **permute 语义** ✅ — 所有 permute 模式一致
8. **reshape 语义** ✅ — 所有 reshape 调用一致
9. **RMSNorm 实现** ✅ — `rms_norm` + `mul(weight)`
10. **LayerNorm 实现** ✅ — `norm` + `mul(weight)`（已修复）
11. **SiLU FFN 实现** ✅ — `down(SiLU(up(x)))`
12. **GLU 实现** ✅ — view2d + sigmoid + mul + transpose
13. **Softcap 实现** ✅ — scale(1/cap) → tanh → scale(cap)
14. **k_scale 计算** ✅ — `log2(1 + e)`
15. **q_scale 计算** ✅ — 已修复为 `(1/sqrt(d_head)) / ln(2)`
16. **位置编码填充** ✅ — `fillSinusoidalPosEmb` 与 C++ 一致
17. **注意力掩码填充** ✅ — `fillChunkedAttentionMask` 与 C++ 一致

---

## 📋 验证命令速查

```bash
# 生成 zllama 调试数据
./zig-out/bin/zllama -m ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj ~/.cache/models/gemma-4-E2B-mmproj-BF16.gguf --audio ~/.cache/models/hello.wav -p " "

# 对比 Mel 频谱
python3 tools/compare-float.py --file1 debug_audio/llama_audio_mel.json --file2 debug_audio/zllama_audio_mel.json --verbose

# 对比 Conv2d 第0层输出（现已一致）
python3 tools/compare-float.py --file1 debug_audio/llama_audio_conv2d_0_output.json --file2 debug_audio/zllama_audio_conv2d_0_output.json --verbose

# 对比 Flatten 输出（现已一致）
python3 tools/compare-float.py --file1 debug_audio/llama_audio_flatten_output.json --file2 debug_audio/zllama_audio_flatten_output.json --verbose

# 对比 Input Projection 输出（显著差异）
python3 tools/compare-float.py --file1 debug_audio/llama_audio_input_proj_output.json --file2 debug_audio/zllama_audio_input_proj_output.json --verbose

# 对比最终嵌入
python3 tools/compare-float.py --file1 debug_audio/llama_audio_embeddings.json --file2 debug_audio/zllama_audio_embeddings.json --verbose

# 批量对比所有数据
for f in `cd debug_audio && ls llama_audio_*.json`; do echo "--- $f ---"; python3 tools/compare-float.py --file1 "debug_audio/$f" --file2 "debug_audio/z$f" 2>&1 | grep -E "^(对比|📉|🔗|📌|长度)"; done
```
