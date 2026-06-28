# MTMD 音频编码对齐分析报告

对比 src/mtmd/audio/encoder.zig 的 fn encode() 与 deps/llama.cpp/tools/mtmd/models/gemma4a.cpp 的 clip_graph_gemma4a::build() 中的 Conformer Blocks 的过程.

## 已完成实现

以下 C++ 类接口函数已严格按照 C++ 实现映射到 Zig 中：

### buildNorm（✅ 已完成）
对应 C++ `clip_graph::build_norm()`，位于 `deps/llama.cpp/tools/mtmd/clip.cpp:557-580`

- 支持 RMSNorm（`NORM_TYPE_RMS`）和 LayerNorm（`NORM_TYPE_NORMAL`）
- 可选 weight（`mw`）和 bias（`mb`）参数
- 使用 `buildNorm(ctx, cur, mw, mb, norm_type, norm_eps, il)` 签名

### buildFFN（✅ 已完成）
对应 C++ `clip_graph::build_ffn()`，位于 `deps/llama.cpp/tools/mtmd/clip.cpp:582-620`

- 支持 SiLU、GELU、GELU_ERF、GELU_QUICK、RELU_SQR 激活类型
- 可选 gate、up_bias、gate_bias、down_bias 参数
- 使用 `buildMM` 替代 `build_mm` 以应用 clamp 逻辑
- 使用 `buildFFN(ctx, cur, up, up_b, gate, gate_b, down, down_b, type_op, il, clamp_map)` 签名

### extractBlocks（✅ 已完成）
对应 C++ `gemma4a.cpp` 中的 `extract_blocks` lambda

- 将 `[D, H, N]` 张量通过 pad + roll + overlapping view 转换为 `[D, H, S, B]`
- 使用 `extractBlocks(ctx, t, d_head, n_head, S, B, C, P, n_pos)` 签名

## 调用点更新

所有 `rmsNorm` 和 `ffnSilu` 调用点已更新为使用新的通用函数：

| 位置 | 原函数 | 新函数 |
|------|--------|--------|
| FFN 1 norm | `rmsNorm` | `buildNorm(..., .rms, ...)` |
| FFN 1 | `ffnSilu` | `buildFFN(..., .silu, ...)` |
| Attention norm | `rmsNorm` | `buildNorm(..., .rms, ...)` |
| K/V block extraction | 内联代码 | `extractBlocks` |
| Attention post-norm | `rmsNorm` | `buildNorm(..., .rms, ...)` |
| Conv module norm | `rmsNorm` | `buildNorm(..., .rms, ...)` |
| FFN 2 norm | `rmsNorm` | `buildNorm(..., .rms, ...)` |
| FFN 2 | `ffnSilu` | `buildFFN(..., .silu, ...)` |
| Layer output norm | `rmsNorm` | `buildNorm(..., .rms, ...)` |

## 构建验证

- `zig build -Doptimize=ReleaseSafe` ✅ 编译成功
- `zig build test -Doptimize=ReleaseSafe --summary all` ✅ 160/160 测试通过
