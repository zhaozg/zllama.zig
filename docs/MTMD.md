# MTMD — 多模态推理模块

> 参考: [llama.cpp tools/mtmd](deps/llama.cpp/tools/mtmd/)、[gemma4 模型](deps/llama.cpp/tools/mtmd/models/gemma4*.cpp)

## 目录结构

```
src/
├── mtmd.zig                  # 模块根 — 核心数据模型 + MtmdContext
└── mtmd/
    ├── audio.zig             # AudioEncoder — Gemma4 音频编码器
    ├── vision.zig            # VisionEncoder — Gemma4 视觉编码器
    ├── manager.zig           # MultiModalManager — 协调音视频编码器
    ├── preprocess.zig        # 图像预处理 (Crop/Resize/Normalize/FFT)
    ├── fft.zig               # 2D FFT 实现
    ├── helper.zig            # 辅助函数 (chunk eval, bitmap 加载, pos 计算)
    └── tokenize.zig          # 输入分词 — 文本 + media_marker 分割
```

## 架构分层

```
               ┌─────────────────────┐
               │    main.zig          │  命令行 / 会话循环
               └────────┬────────────┘
                        │
               ┌────────▼────────────┐
               │    mtmd.zig          │  核心数据模型 + MtmdContext
               │  ChunkType/InputChunk│
               │  Bitmap/ImageTokens  │
               └──┬─────┬──────┬─────┘
                  │     │      │
     ┌────────────▼─┐ ┌─▼──────────┐ ┌─▼──────────────┐
     │ tokenize.zig  │ │helper.zig  │ │ manager.zig     │
     │ 输入分词      │ │chunk eval  │ │ 编码器协调      │
     └───────────────┘ │bitmap 加载 │ └─┬───────┬─────┘
                       └────────────┘   │       │
                             ┌──────────▼─┐ ┌───▼──────────┐
                             │ audio.zig   │ │ vision.zig   │
                             │ Gemma4Audio │ │ Gemma4Vision │
                             └──────┬──────┘ └───┬──────────┘
                                    │             │
                             ┌──────▼─────────────▼──────┐
                             │      preprocess.zig        │
                             │  Crop/Resize/Normalize/FFT │
                             └────────────────────────────┘
```

## Chunk 模型 — 输入的多态表示

输入首先通过 `tokenize.zig` 按 `media_marker` 分割为有序的 `InputChunk` 列表：

```
用户输入: "描述这张图<__media__>和这段音频<__media__>"
         ──→ [text: "描述这张图"] [image: bm_0] [text: "和这段音频"] [audio: bm_1]
```

### ChunkType

```zig
pub const ChunkType = enum(u8) { text, image, audio };
```

### InputChunk

```zig
pub const InputChunk = struct {
    chunk_type: ChunkType,            // 块类型
    tokens_text: ?[]const i32 = null, // text 块的 token id 列表
    tokens_image: ?ImageTokens = null,// image 块的形状/位置元数据
    tokens_audio_n: u32 = 0,          // audio 块的预估 token 数
    id: ?[]const u8 = null,           // 块标识 (从 Bitmap.id)
};
```

| 方法 | 返回 | 说明 |
|------|------|------|
| `nTokens()` | `u32` | token 数量（image 含特殊 token） |
| `nPos()` | `u32` | 位置数（M-RoPE 模式下 ≠ nTokens） |

### InputChunks

有序 chunk 集合，提供聚合操作：

```zig
pub const InputChunks = struct {
    // init(allocator) → InputChunks
    // deinit() → void
    // size() → usize
    // get(idx) → *const InputChunk
    // append(chunk) → !void
    // totalTokens() → usize   所有 chunk 的 token 总数
    // totalPos() → usize      所有 chunk 的位置总数
};
```

## Bitmap — 原始媒体数据

```zig
pub const Bitmap = struct {
    nx: u32, ny: u32,           // 宽/高（audio 时 ny=1）
    is_audio: bool = false,
    id: ?[]const u8 = null,     // 可选标识
    data: ?[]const u8 = null,   // 像素/音频样本数据
    allocator: ?std.mem.Allocator = null,
};
```

| 工厂方法 | 说明 |
|----------|------|
| `initImage(nx, ny, data)` | 创建图像 Bitmap |
| `initAudio(n_samples, data)` | 创建音频 Bitmap |
| `initPlaceholderImage(nx, ny)` | 占位图像（无 data，用于 token 估算） |
| `initPlaceholderAudio(n_samples)` | 占位音频 |

## ImageTokens — 图像位置元数据

```zig
pub const ImageTokens = struct {
    nx: u32, ny: u32,           // patch 网格尺寸
    pos: PosType = .normal,     // 位置编码类型
    image_idx: u32 = 0,         // 图像索引
    // ...
};
```

### PosType

| 枚举 | 适用模型 | 行为 |
|------|----------|------|
| `.normal` | LLaVA 等 | 顺序位置，nTokens = nx × ny |
| `.mrope` | Qwen2-VL | M-RoPE 多维位置编码 |
| `.hunyuanvl` | Hunyuan | 特殊格式，nTokens = (nx+1)×ny+2 |

## MtmdContext — 多模态会话上下文

```zig
pub const MtmdContext = struct {
    allocator: std.mem.Allocator,
    mm_manager: *mm.MultiModalManager,  // 编码器管理
    caps: model.ModelCapabilities,      // 能力标志
    params: ContextParams,              // 运行时参数
    n_embd_text: i32,                   // 文本嵌入维度
    tok: ?*tokenizer.Tokenizer = null,  // 分词器引用
    media_marker: []const u8,           // 媒体标记字符串
    img_beg: []const u8,                // 图像开始 token
    img_end: []const u8,                // 图像结束 token
    aud_beg: []const u8,                // 音频开始 token
    aud_end: []const u8,                // 音频结束 token
    pos_type: PosType = .normal,        // 位置编码类型
    output_embd: ?[]f32 = null,         // 编码器输出缓存
};
```

### ContextParams

```zig
pub const ContextParams = struct {
    use_gpu: bool = true,
    print_timings: bool = true,
    n_threads: u32 = 4,
    media_marker: []const u8 = "<__media__>",
    warmup: bool = true,
    image_min_tokens: i32 = -1,
    image_max_tokens: i32 = -1,
};
```

### 主要方法

| 方法 | 说明 |
|------|------|
| `init(allocator, mmproj_path, io, text_n_embd, params, tok)` | 加载 mmproj 文件，初始化编码器 |
| `deinit()` | 释放所有资源 |
| `supportVision()` → `bool` | 模型是否支持视觉 |
| `supportAudio()` → `bool` | 模型是否支持音频 |
| `decodeUseNonCausal(?)` → `bool` | 是否需要非因果注意力（Gemma4） |
| `decodeUseMRope()` → `bool` | 是否使用 M-RoPE 位置编码 |

### 典型使用流程

```zig
// 1. 加载 mmproj 文件创建上下文
var mtmd_ctx = try mtmd.MtmdContext.init(
    allocator, mmproj_path, io, text_n_embd, params, &tokenizer,
);
defer mtmd_ctx.deinit();

// 2. 准备输入
const bitmaps = [_]mtmd.Bitmap{
    mtmd.Bitmap.initImage(224, 224, image_rgb_data),
};
const input = mtmd.InputText{ .text = "描述这张图<__media__>", .parse_special = true };

// 3. 分词 → chunk 序列
var chunks = try mtmd.tokenize.tokenize(mtmd_ctx, allocator, input, &bitmaps);
defer chunks.deinit();

// 4. 对 image/audio chunk 调用编码器获取嵌入
for (0..chunks.size()) |i| {
    const chunk = chunks.get(i);
    switch (chunk.chunk_type) {
        .image => {
            // 使用 helper 加载 bitmap 并编码
            const bm_w = try mtmd.helper.bitmapInitFromFile(allocator, "image.jpg");
            defer bm_w.deinit();
            // ...
        },
        .audio => { /* 类似 */ },
        .text => { /* 直接使用 tokens_text */ },
    }
}

// 5. interleave text + media embeddings for LLM decoder
```

## manager — MultiModalManager

协调音频/视觉编码器的加载与推理。

```zig
pub const MultiModalManager = struct {
    allocator: std.mem.Allocator,
    capabilities: model.ModelCapabilities,
    audio_encoder: ?audio.AudioEncoder,
    vision_encoder: ?vision.VisionEncoder,

    // init(allocator, gguf_file, ctx, caps) → !MultiModalManager
    // deinit() → void
    // encodeMedia(ctx, graph, input) → !*ggml.Tensor
    // estimateTokenCount(input) → u32
    // supportsMediaType(media_type) → bool
    // formatCapabilities(writer) → !void
};
```

### MediaInput

```zig
pub const MediaInput = struct {
    media_type: MediaType,             // .text | .image | .audio
    text: ?[]const u8 = null,
    image_data: ?[]const u8 = null,    // RGB [h][w][3]
    image_width: u32 = 0,
    image_height: u32 = 0,
    mel_data: ?[]const f32 = null,     // Mel 频谱
    mel_bins: u32 = 0,
    mel_frames: u32 = 0,
    audio_length_sec: f32 = 0,
};
```

## audio / vision — 编码器

两个编码器遵循相同接口模式：

```zig
// AudioEncoder
pub const AudioEncoder = struct {
    // init(gguf_file, ctx, allocator) → !AudioEncoder
    // deinit(allocator) → void
    // encode(ctx, graph, mel_data, n_mel_bins, n_frames) → !*ggml.Tensor
    // estimateOutputTokens(duration_sec: f32) → u32
    // isAvailable() → bool
};

// VisionEncoder
pub const VisionEncoder = struct {
    // init(gguf_file, ctx, allocator) → !VisionEncoder
    // deinit(allocator) → void
    // encode(ctx, graph, image_data, width, height) → !*ggml.Tensor
    // estimateOutputTokens(width, height) → u32
    // isAvailable() → bool
};
```

`encode()` 返回一个 `[n_tokens, n_embd]` 形状的 ggml Tensor，可直接送入 LLM decoder。

## helper — 辅助函数

```zig
// 对 chunk 序列进行编码器评估（核心流程）
fn evalChunks(
    mtmd_ctx: *mtmd.MtmdContext,
    ggml_ctx: *ggml.Context,
    graph: *ggml.CGraph,
    chunks: *mtmd.InputChunks,
    bitmaps: []const mtmd.Bitmap,
    n_tokens: *usize,
    output_embd: *[]f32,
) !void;

// 计算 image chunk 的 decoder 位置
fn imageGetDecoderPos(img: ImageTokens, start_t: u32, dst: []DecoderPos) void;

// Bitmap 加载
fn bitmapInitFromFile(allocator, path) !BitmapWrapper;
fn bitmapInitFromBuf(allocator, data, format) !BitmapWrapper;
```

## tokenize — 输入分词

```zig
pub fn tokenize(
    ctx: *MtmdContext,
    allocator: std.mem.Allocator,
    text: InputText,
    bitmaps: []const Bitmap,
) !InputChunks;
```

1. 按 `ctx.media_marker` 分割文本
2. text 段通过 `tok.encode()` 转为 token id
3. 相邻 text 段自动合并
4. image/audio 段自动插入 `img_beg`/`img_end` 或 `aud_beg`/`aud_end`
5. 验证 marker 数量与 bitmap 数量一致，否则返回 `error.MarkerBitmapMismatch`

## 能力检测

```zig
pub fn getCapFromFile(allocator, mmproj_path, io) !Caps;

pub const Caps = struct {
    inp_vision: bool,  // 是否支持视觉输入
    inp_audio: bool,   // 是否支持音频输入
};
```

内部通过 `detectCapabilities()` 扫描 GGUF tensor 名称自动识别：

| Tensor | 推断 |
|--------|------|
| `v.patch_embd.weight` / `v.position_embd.weight` | 视觉编码器 |
| `patch_norm_1.weight` | → `vision_encoder_type = "gemma4uv"` |
| 否则 | → `vision_encoder_type = "gemma4v"` |
| `a.conv1d.0.weight` / `a.input_projection.weight` | 音频编码器 |

## 测试

```bash
zig build test-mtmd
```

测试覆盖 (`src/tests/test_mtmd.zig`)：

| 测试 | 覆盖 |
|------|------|
| `Bitmap: image creation` | Bitmap.initImage / isPlaceholder |
| `ImageTokens: normal` | nTokens 计算 |
| `InputChunk: text` | text chunk nTokens/nPos |
| `InputChunks: total tokens` | append/deinit/size/totalTokens |
| `Caps: default` | 零值 |
| `DecoderPos: normal` | helper.imageGetDecoderPos |
| `tokenize: text only` | 纯文本分词 |
| `tokenize: marker mismatch` | marker/bitmap 不匹配错误 |
| `tokenize: image marker` | 图像 marker 分词 |
| `tokenize: audio marker` | 音频 marker 分词 |

## 设计决策

1. **无动态分发**：`ChunkType` 是 enum，chunk 处理使用 `switch`，编译期零成本。
2. **ArrayList 替代 vector**：`InputChunks.entries` 用 `std.ArrayList(InputChunk)`，符合 Zig 习惯。
3. **编码器为可选**：`audio_encoder: ?AudioEncoder`，模型无相关能力时自动跳过。
4. **GGUF 自动检测**：无需手动指定多模态类型，扫描 tensor 名推断。
5. **位置编码策略隔离**：`PosType` enum + `ImageTokens.nTokens()` 封闭了位置编码差异。
