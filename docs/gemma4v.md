# Gemma4V 图形推理过程分析

## 1. 概述

Gemma4V 是 Google Gemma 4 系列的视觉-语言模型（VLM），在 llama.cpp 中通过 `mtmd`（Multi-Modal）框架实现。其视觉编码器基于 ViT (Vision Transformer) 架构，核心特点是使用了**带 2D RoPE 的 SigLIP 风格编码器**、**Gemma4VisionPooler 池化**以及 **Gemma4MultimodalEmbedder 投影**。

## 2. 整体架构流程

```
输入图像
  │
  ▼
[图像预处理] (mtmd_image_preprocessor_dyn_size)
  │  - 缩放到合适尺寸
  │  - 图像标记: <|image> ... <image|>
  ▼
[视觉编码器图形构建] (clip_graph_gemma4v::build)
  │
  ├── 步骤1: Patch Embedding
  │      ├─ build_inp_raw() → [W, H, C, B] 原始像素
  │      ├─ ggml_scale_bias: 归一化 → [-1, 1]
  │      └─ ggml_conv_2d: 卷积提取 patches → [n_embd, n_patches, B]
  │
  ├── 步骤2: 位置编码 (学得的位置嵌入)
  │      ├─ pos_x / pos_y: 每个 patch 的 (col, row) 坐标
  │      └─ position_embeddings 查表得到 emb_x, emb_y 并加到 inp 上
  │
  ├── 步骤3: ViT 编码器 (build_vit)
  │      ├─ add_pos: 2D RoPE (NEOX 排序)
  │      │    ├─ 前半部分用 pos_x 做 RoPE
  │      │    └─ 后半部分用 pos_y 做 RoPE
  │      ├─ V 张量额外 RMS Norm (gemma4v 特有)
  │      └─ N 层 Transformer (RMS Norm + FFN)
  │
  ├── 步骤4: Gemma4VisionPooler
  │      ├─ ggml_pool_2d (AVG): 空间降采样, kernel_size=n_merge(3)
  │      └─ 缩放: × sqrt(n_embd)
  │
  ├── 步骤5: 标准化
  │      └─ (cur - std_bias) * std_scale
  │
  └── 步骤6: Gemma4MultimodalEmbedder
         ├─ ggml_rms_norm
         └─ build_mm: 带 Clipping 的线性投影
              → 输出 [n_mmproj_embd, out_patches, B]
```

## 3. 关键参数配置

从 `clip.cpp` 中 `GEMMA4V` 的配置代码（第1434行）：

```cpp
case PROJECTOR_TYPE_GEMMA4V:
    hparams.rope_theta = 100.0f;       // RoPE 基础频率
    hparams.n_merge = 3;               // 池化核大小 (pooling_kernel_size)
    hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
    hparams.set_limit_image_tokens(40, 280);  // 最小40, 最大280个图像token
    hparams.set_warmup_n_tokens(256);  // 预热token数
```

- **patch_size**: 模型的 patch 大小（通常是 14 或 16）
- **n_embd**: ViT 隐藏维度
- **n_layer**: ViT 层数
- **n_mmproj_embd**: 投影后的嵌入维度（= `mm_input_proj_w->ne[1]`）
- **eps**: RMS Norm epsilon

## 4. 关键函数调用链

### 4.1 入口: `clip_image_batch_encode()`

```
clip_image_batch_encode()          // clip.cpp:3565
  └── clip_get_graph_builder()     // clip.cpp:871
       └── clip_graph_gemma4v 构造函数
            └── clip_graph::clip_graph()  // 初始化公共成员
  └── builder->build()            // gemma4v.cpp:11
       └── clip_graph_gemma4v::build()  // 构建完整计算图
  └── ggml_graph_compute()        // 执行计算图
  └── 提取输出嵌入 (n_mmproj_embd 维)
```

### 4.2 `clip_graph_gemma4v::build()` 详解

位于 `tools/mtmd/models/gemma4v.cpp`，逐段分析：

#### (A) 输入归一化 + Patch Embedding

```cpp
// 1. 获取原始像素输入 [W, H, C, B]
ggml_tensor * inp_raw = build_inp_raw();
// build_inp_raw: ggml_new_tensor_4d(GGML_TYPE_F32, img.nx(), img.ny(), channels, n_batch)

// 2. 归一化到 [-1, 1]: patches = 2 * (patches - 0.5)
inp_raw = ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f);

// 3. 卷积提取 patches: patch_embeddings_0 作为卷积核
//    shape: [n_embd, patch_size*patch_size*C, 1, 1] → 输出 [n_embd, n_patches_w, n_patches_h, B]
ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw,
                                  patch_size, patch_size, 0, 0, 1, 1);
// 4. 重构为 [n_embd, n_patches, B]
inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));  // [n_embd, n_patches, B]
```

#### (B) 学得的位置嵌入 (Learned Positional Embedding)

```cpp
// 创建位置索引输入张量
ggml_tensor * pos_x = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
ggml_tensor * pos_y = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);

// position_embeddings 存储为 [n_embd, pos_size*2] 的查找表
// 前半 pos_size 行是 X 嵌入, 后半是 Y 嵌入
ggml_tensor * tbl_x = ggml_view_2d(ctx0, model.position_embeddings,
                                    n_embd, pos_size, nb1, 0);
ggml_tensor * tbl_y = ggml_view_2d(ctx0, model.position_embeddings,
                                    n_embd, pos_size, nb1, pos_size * nb1);

// 查表获取 [n_embd, n_patches]
ggml_tensor * emb_x = ggml_get_rows(ctx0, tbl_x, pos_x);
ggml_tensor * emb_y = ggml_get_rows(ctx0, tbl_y, pos_y);

// 加到输入上
inp = ggml_add(ctx0, inp, emb_x);
inp = ggml_add(ctx0, inp, emb_y);
```

#### (C) 2D RoPE 位置编码函数

Gemma4V 使用 **NEOX 排序**的 2D RoPE。与常见的 `build_rope_2d` 不同，这里直接将 Q/K 的维度分为前后两半，分别应用 X/Y 方向的 RoPE：

```cpp
auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
    // cur shape: [n_dim, n_head, n_pos, B]

    // 前半: 用 pos_x 做 RoPE
    ggml_tensor * first = ggml_view_4d(ctx0, cur,
        n_dim/2, n_head, n_pos, n_batch, ...);
    first = ggml_rope_ext(ctx0, first, pos_x, nullptr, n_dim/2,
                          GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta, ...);

    // 后半: 用 pos_y 做 RoPE
    ggml_tensor * second = ggml_view_4d(ctx0, cur,
        n_dim/2, n_head, n_pos, n_batch, ..., n_dim/2*element_size);
    second = ggml_rope_ext(ctx0, second, pos_y, nullptr, n_dim/2,
                           GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta, ...);

    // 拼接回来
    cur = ggml_concat(ctx0, first, second, 0);
    return cur;
};
```

#### (D) ViT Transformer (build_vit)

```cpp
ggml_tensor * cur = build_vit(
    inp, n_patches,
    NORM_TYPE_RMS,           // 使用 RMS Norm
    hparams.ffn_op,          // FFN 操作类型
    nullptr,                 // 学得的位置嵌入已在外层处理
    add_pos);                // 2D RoPE 回调
```

`build_vit`（`clip.cpp:314`）内部对每一层执行：

```
for each layer:
    inpL = cur (残差)
    cur = RMS_Norm(cur, ln_1_w)       // pre-attn norm
    Q/K/V = build_mm(qkv_w, cur)      // fused QKV 投影
    Q = add_pos(Q)  → RoPE (X/Y 2D)   // gemma4v 特有: 2D RoPE
    K = add_pos(K)  → RoPE (X/Y 2D)
    V = RMS_Norm(V)                   // gemma4v 特有: V 张量归一化
    cur = FlashAttn / SDPA → [n_embd, n_pos*B]
    cur *= ls_1_w                      // 如果存在
    cur = RMS_Norm(cur, attn_post_norm_w) // 如果存在
    cur = cur + inpL                    // 残差连接

    inpL = cur
    cur = RMS_Norm(cur, ln_2_w)        // pre-ffn norm
    cur = FFN(cur)                      // SwiGLU / GeGLU 等
    cur *= ls_2_w                       // 如果存在
    cur = RMS_Norm(cur, ff_post_norm_w) // 如果存在
    cur = cur + inpL                    // 残差连接
    cur *= ls_out_w                     // 如果存在 (gemma4 的层输出缩放)
```

**gemma4v 特有修改**（`clip.cpp:452`）：
```cpp
if (proj_type == PROJECTOR_TYPE_GEMMA4V) {
    Vcur = ggml_rms_norm(ctx0, Vcur, eps);  // V 张量归一化
}
```

#### (E) Gemma4VisionPooler

```cpp
const int kernel_size = hparams.n_merge;  // 默认 3

// [n_embd, n_patches] → [n_patches_x, n_patches_y, n_embd, B]
cur = ggml_cont_4d(ctx0, ggml_transpose(ctx0, cur),
                   n_patches_x, n_patches_y, n_embd, n_batch);

// 2D 平均池化: kernel_size=3, stride=3, 不填充
cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG,
                   kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);

const int out_x = n_patches_x / kernel_size;
const int out_y = n_patches_y / kernel_size;

// 重塑回 [n_embd, out_x * out_y, n_batch]
cur = ggml_reshape_3d(ctx0, cur, out_x * out_y, n_embd, n_batch);
cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

// 缩放: 乘以 sqrt(n_embd) 以控制幅度
cur = ggml_scale(ctx0, cur, sqrtf((float)n_embd));
```

这个池化操作将 patches 网格从 `(n_patches_x × n_patches_y)` 降采样到 `(n_patches_x/3 × n_patches_y/3)`，大幅减少后续 LLM 需处理的 token 数量。

#### (F) 标准化 + 投影

```cpp
// hidden_states = (hidden_states - self.std_bias) * self.std_scale
if (model.std_bias && model.std_scale) {
    cur = ggml_sub(ctx0, cur, model.std_bias);
    cur = ggml_mul(ctx0, cur, model.std_scale);
}

// Gemma4MultimodalEmbedder
cur = ggml_rms_norm(ctx0, cur, hparams.eps);           // pre-projection norm
cur = build_mm(model.mm_input_proj_w, cur);            // 投影到 LLM 嵌入空间
```

### 4.3 `build_mm()` — Gemma4ClippableLinear

Gemma4V 使用了一种特殊的线性层，在前向传播时可能对输入/输出进行 **Clamp（截断）**：

```cpp
ggml_tensor * clip_graph_gemma4v::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    auto it = model.clamp_info_map.find(w->name);
    if (it == model.clamp_info_map.end()) {
        // 无 clamp 信息 → 普通矩阵乘法
        return ggml_mul_mat(ctx0, w, x);
    } else {
        // 有 clamp 信息 → 输入/输出截断
        const auto & clamp_info = it->second;
        ggml_tensor * clamped = ggml_clamp(ctx0, x, clamp_info.inp_min, clamp_info.inp_max);
        ggml_tensor * out = ggml_mul_mat(ctx0, w, clamped);
        out = ggml_clamp(ctx0, out, clamp_info.out_min, clamp_info.out_max);
        return out;
    }
}
```

`clamp_info_map` 在模型加载时填充（`clip.cpp:2191`），从 GGUF 文件中读取每个 `.weight` 张量对应的 `.input_max`, `.input_min`, `.output_max`, `.output_min` 标量。

### 4.4 位置索引设置

在 `clip.cpp:4115` 处，为 gemma4v 设置位置索引输入：

```cpp
case PROJECTOR_TYPE_GEMMA4V:
    const int n_cols = image_size_width / patch_size;
    std::vector<int> pos_x(num_patches), pos_y(num_patches);
    for (int i = 0; i < num_patches; i++) {
        pos_x[i] = i % n_cols;   // 列坐标
        pos_y[i] = i / n_cols;   // 行坐标
    }
    set_input_i32("pos_x", pos_x);
    set_input_i32("pos_y", pos_y);
```

## 5. LLM 端集成

### 5.1 图像 Token 数量计算

```cpp
// clip.cpp:3382
case PROJECTOR_TYPE_GEMMA4V:
    // X 和 Y 都按 scale_factor 降尺度
    int scale_factor = ctx->model.hparams.n_merge; // 3
    n_patches /= (scale_factor * scale_factor);     // 池化后的 patch 数
```

### 5.2 输出嵌入维度

```cpp
// clip.cpp:4643
case PROJECTOR_TYPE_GEMMA4V:
    return ctx->model.mm_input_proj_w->ne[1];
    // 即投影矩阵的输出维度 = LLM 的 n_embd
```

### 5.3 非因果注意力

Gemma4V 的图像 token 使用**双向（非因果）注意力**：

```cpp
// mtmd.cpp:1673
case PROJECTOR_TYPE_GEMMA4V:
    return true;  // use_non_causal = true
```

这通过 `llama_set_causal_attn(lctx, false)` 在解码图像 token 时临时切换，解码完成后恢复为 `true`（`mtmd-helper.cpp:291-332`）。

### 5.4 图像标记

```cpp
// mtmd.cpp:612
case PROJECTOR_TYPE_GEMMA4V:
    img_beg = "<|image>";
    img_end = "<image|>";
```

## 6. 数据流总结

```
┌─────────────────────────────────────────────────────────────────┐
│ 原始图像 [W, H, 3]                                              │
│   ↓ build_inp_raw()                                             │
│ [W, H, 3, B]  (B=1 for gemma4v since support_batch=true)        │
│   ↓ ggml_scale_bias(x, 2.0f, -1.0f)                             │
│ 归一化到 [-1,1]                                                 │
│   ↓ ggml_conv_2d(patch_embeddings_0, ...)                       │
│ [n_embd, n_patches, B]  ← 卷积提取 patches                      │
│   ↓ + pos_embd_x, pos_embd_y (learned)                          │
│ 加入学得的位置嵌入                                              │
│   ↓ build_vit (N layers)                                        │
│ [n_embd, n_patches, B]                                          │
│   每层: RMS Norm → QKV → 2D RoPE(Q,K) → V Norm → Attn → FFN     │
│   ↓ Gemma4VisionPooler (2D avg pool, kernel=3)                  │
│ [n_embd, n_patches/9, B]                                        │
│   ↓ * sqrt(n_embd)                                              │
│   ↓ (x - std_bias) * std_scale                                  │
│ [n_embd, n_patches/9, B]                                        │
│   ↓ RMS Norm + build_mm(mm_input_proj_w)                        │
│ [n_mmproj_embd, n_patches/9, B]  ← 最终图像嵌入                 │
│   ↓                                                             │
│ LLM decode (non-causal attention)                               │
└─────────────────────────────────────────────────────────────────┘
```

## 7. Gemma4V vs Gemma4UV 差异

| 特性 | Gemma4V | Gemma4UV (Unified Vision) |
|------|---------|---------------------------|
| Patch 提取 | `ggml_conv_2d` | `ggml_im2col` + LayerNorm + `ggml_mul_mat` + LayerNorm |
| Patch 归一化 | 无 | 双层 LayerNorm (`patch_norm_1`, `patch_norm_2`) |
| 位置嵌入后处理 | 无 | LayerNorm (`patch_norm_3`) |
| ViT 编码器 | 有 (完整 Transformer) | 无 (跳过) |
| RoPE | 2D NEOX RoPE | 无 |
| VisionPooler | 有 (avg pool) | 无（patch_size 已放大） |
| std_bias/scale | 支持 | 不支持 |
| Clamp 线性层 | 支持 | 不支持 |

Gemma4UV 是简化变体，将 token merging 直接在 patch 提取阶段完成（`patch_size *= n_merge`），因此不需要 ViT 和 Pooler。

## 8. 实现指南

若要实现新的 Gemma4 视觉推理，需要关注以下几点：

1. **参数加载**：从 GGUF 读取 `patch_embeddings_0`、`position_embeddings`、各层 Transformer 权重、`mm_input_proj_w` 以及可选的 `std_bias`/`std_scale`/`clamp_info`。

2. **Patch 嵌入**：归一化 → 卷积提取 → 形状调整为 `[n_embd, n_patches, B]`。

3. **位置编码**：X/Y 分离的学得位置嵌入 + 2D RoPE（NEOX 排序，半维度 X、半维度 Y）。

4. **Transformer 层**：
   - RMS Norm (pre-norm)
   - QKV fused projection
   - 2D RoPE on Q, K
   - RMS Norm on V (gemma4v 特有)
   - Flash Attention / SDPA
   - 残差 + 后注意归一化
   - RMS Norm → FFN (如 SwiGLU) → 后 FFN 归一化 → 残差

5. **池化投影**：2D 平均池化降采样 → `sqrt(n_embd)` 缩放。

6. **输出投影**：`(x - std_bias) * std_scale` → RMS Norm → Clamp 线性投影到 LLM 嵌入空间。

7. **LLM 集成**：非因果注意力解码图像 token，标记为 `<|image>` / `<image|>`。
