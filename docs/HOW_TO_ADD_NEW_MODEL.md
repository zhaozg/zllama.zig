# 新增模型指南

> 本文档指导如何在 zllama.zig 中添加对新模型架构的支持。

---

## 📋 概览

新增一个模型需要修改以下文件：

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `src/models/your_model.zig` | **新建** — 模型实现（参数、权重、buildGraph） |
| 2 | `src/model.zig` | **修改** — 添加架构枚举值、模型模块导入和重新导出 |
| 3 | `src/models/registry.zig` | **修改** — 添加架构检测、工厂创建、能力检测（含 special_tokens 配置） |
| 4 | `src/chat_template/mod.zig` | **修改** — 添加对话模板（如需要） |
| 5 | `build.zig` | **无需修改** — 模型通过 `model.zig` 间接导入，无需单独注册模块 |

---

## 第 1 步：创建模型实现文件

在 `src/models/` 下创建新文件，例如 `src/models/mistral.zig`。

### 文件结构模板

```zig
//! Mistral 模型实现
//!
//! 参考：llama.cpp src/models/model_mistral.cpp

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const kv_cache = @import("kv_cache");
const rms_norm = @import("rms_norm");
const rope = @import("rope");
const swiglu = @import("swiglu");
const graph_builder = @import("graph_builder");
const memory = @import("memory");
const attention = @import("attention");
const embed = @import("embed");
const weight_loader = @import("weight_loader");

// 导入 model 模块（通过模块名，因跨目录）
const model = @import("model");

const log = std.log.scoped(.mistral);

// ============================================================================
// 超参数
// ============================================================================

/// Mistral 模型超参数（从 GGUF 元数据读取）
pub const MistralParams = struct {
    base: model.ModelParams = .{},
};

// ============================================================================
// 权重
// ============================================================================

/// 单层权重
pub const LayerWeights = struct {
    attn_q: *ggml.Tensor,
    attn_k: *ggml.Tensor,
    attn_v: *ggml.Tensor,
    attn_o: *ggml.Tensor,
    ffn_gate: *ggml.Tensor,
    ffn_down: *ggml.Tensor,
    ffn_up: *ggml.Tensor,
    attn_norm: *ggml.Tensor,
    ffn_norm: *ggml.Tensor,
};

/// 所有权重
pub const MistralWeights = struct {
    params: MistralParams,
    token_embd: *ggml.Tensor,
    output_weight: ?*ggml.Tensor,
    output_norm: *ggml.Tensor,
    layers: []LayerWeights,
};

// ============================================================================
// 模型结构体 + 虚表
// ============================================================================

pub const MistralModel = struct {
    allocator: std.mem.Allocator,
    weights: MistralWeights,
    ctx: *ggml.Context,

    /// 虚表 — 必须导出为 pub const
    pub const vtable: model.ModelVTable = .{
        .getParams = getParams,
        .buildGraph = buildGraph,
        .deinit = deinit,
    };

    fn getParams(ptr: *anyopaque) *const model.ModelParams {
        const self: *MistralModel = @ptrCast(@alignCast(ptr));
        return &self.weights.params.base;
    }

    pub fn init(
        self: *MistralModel,
        allocator: std.mem.Allocator,
        gguf_file: *gguf.GGUFFile,
        io: std.Io,
    ) !void {
        // 1. 解析超参数
        const params = try parseParams(gguf_file, allocator);

        // 2. 估算内存并分配 ggml context
        const mem_size = try estimateMemSize(gguf_file);
        const ctx = try ggml.Context.initNoAlloc(mem_size, io);
        errdefer ctx.deinit();

        // 3. 加载权重
        const weights = try loadWeights(gguf_file, ctx, &params, allocator);

        self.* = .{
            .allocator = allocator,
            .weights = weights,
            .ctx = ctx,
        };
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *MistralModel = @ptrCast(@alignCast(ptr));
        allocator.free(self.weights.layers);
        self.ctx.deinit();
        allocator.destroy(self);
    }

    pub fn buildGraph(
        ptr: *anyopaque,
        builder: *graph_builder.GraphBuilder,
        input: *ggml.Tensor,
        n_tokens: i32,
        cache: ?*anyopaque,
        pos: i32,
    ) !*ggml.Tensor {
        const self: *MistralModel = @ptrCast(@alignCast(ptr));
        const w = &self.weights;
        const p = &w.params.base;

        const kv: *kv_cache.KVCache = @ptrCast(@alignCast(cache.?));

        // --- 构建计算图 ---
        // 1. Token 嵌入
        var cur = try embed.forward(builder, input, w.token_embd, p.n_embd);

        // 2. 逐层处理
        for (w.layers, 0..) |*layer, i| {
            // RMSNorm
            cur = try rms_norm.forward(builder, cur, layer.attn_norm, p.norm_eps);

            // Q/K/V 投影 + RoPE
            const n_heads: i32 = @intCast(p.n_head);
            const n_kv_heads: i32 = @intCast(p.n_kv_head);
            const head_dim: i32 = @intCast(p.n_head_dim);
            const rope_dim: i32 = @intCast(p.rope_dim);
            const layer_id: i32 = @intCast(i);

            const q = try rope.forward(builder, cur, layer.attn_q, n_heads, head_dim, rope_dim, pos, p.rope_theta);
            const k = try rope.forward(builder, cur, layer.attn_k, n_kv_heads, head_dim, rope_dim, pos, p.rope_theta);

            // KV Cache 写入（零拷贝视图）
            kv.setKv(builder.ctx, builder.graph, i, k, layer.attn_v, @intCast(n_tokens));

            // 注意力计算
            const k_cache = kv.getKView(builder.ctx, i);
            const v_cache = kv.getVView(builder.ctx, i);
            cur = try attention.forward(builder, q, k_cache, v_cache, layer.attn_o, n_heads, n_kv_heads, head_dim, n_tokens);

            // FFN (SwiGLU)
            cur = try rms_norm.forward(builder, cur, layer.ffn_norm, p.norm_eps);
            cur = try swiglu.forward(builder, cur, layer.ffn_gate, layer.ffn_up, layer.ffn_down);
        }

        // 3. 输出层归一化
        cur = try rms_norm.forward(builder, cur, w.output_norm, p.norm_eps);

        // 4. 输出投影
        const output = w.output_weight orelse w.token_embd;
        return try builder.mulMat(cur, output);
    }
};

// ============================================================================
// 参数解析
// ============================================================================

pub fn parseParams(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !MistralParams {
    _ = allocator;
    var params: MistralParams = .{};
    const b = &params.base;

    // 键名前缀需与实际架构名一致（如 mistral、qwen2、gemma 等）
    b.n_vocab = gguf_file.getU32("mistral.vocab_size") orelse
        gguf_file.getU32("tokenizer.ggml.vocab_size") orelse 32000;
    b.n_embd = gguf_file.getU32("mistral.embedding_length") orelse 4096;
    b.n_head = gguf_file.getU32("mistral.head_count") orelse 32;
    b.n_head_dim = gguf_file.getU32("mistral.head_dim") orelse
        @divExact(b.n_embd, b.n_head);
    b.n_kv_head = gguf_file.getU32("mistral.head_count_kv") orelse b.n_head;
    b.n_layer = gguf_file.getU32("mistral.block_count") orelse 32;
    b.n_ff = gguf_file.getU32("mistral.feed_forward_length") orelse 14336;
    b.max_seq_len = gguf_file.getU32("mistral.context_length") orelse 32768;
    b.rope_theta = gguf_file.getF32("mistral.rope.theta") orelse 10000000.0;
    b.rope_dim = gguf_file.getU32("mistral.rope.dimension_count") orelse
        @divExact(b.n_embd, b.n_head);
    b.norm_eps = gguf_file.getF32("mistral.attention.layer_norm_rms_epsilon") orelse 1e-6;
    b.model_name = gguf_file.getString("general.name") orelse "";
    b.tokenizer_name = gguf_file.getString("tokenizer.ggml.model") orelse "";

    log.info("Mistral params: vocab={d}, embd={d}, heads={d}, kv_heads={d}, layers={d}, ff={d}, max_seq={d}",
        .{ b.n_vocab, b.n_embd, b.n_head, b.n_kv_head, b.n_layer, b.n_ff, b.max_seq_len });

    return params;
}

// ============================================================================
// 权重加载
// ============================================================================

pub fn loadWeights(
    gguf_file: *const gguf.GGUFFile,
    ctx: *ggml.Context,
    params: *const MistralParams,
    allocator: std.mem.Allocator,
) !MistralWeights {
    const b = &params.base;
    const n_layers = b.n_layer;

    const token_embd = try weight_loader.load(ctx, gguf_file, "token_embd.weight");
    const output_weight = weight_loader.load(ctx, gguf_file, "output.weight") catch null;
    const output_norm = try weight_loader.load(ctx, gguf_file, "output_norm.weight");

    const layers = try allocator.alloc(LayerWeights, n_layers);
    errdefer allocator.free(layers);

    for (0..n_layers) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}", .{i});
        defer allocator.free(prefix);

        layers[i] = .{
            .attn_q = try weight_loader.load(ctx, gguf_file, prefix ++ ".attn_q.weight"),
            .attn_k = try weight_loader.load(ctx, gguf_file, prefix ++ ".attn_k.weight"),
            .attn_v = try weight_loader.load(ctx, gguf_file, prefix ++ ".attn_v.weight"),
            .attn_o = try weight_loader.load(ctx, gguf_file, prefix ++ ".attn_o.weight"),
            .ffn_gate = try weight_loader.load(ctx, gguf_file, prefix ++ ".ffn_gate.weight"),
            .ffn_down = try weight_loader.load(ctx, gguf_file, prefix ++ ".ffn_down.weight"),
            .ffn_up = try weight_loader.load(ctx, gguf_file, prefix ++ ".ffn_up.weight"),
            .attn_norm = try weight_loader.load(ctx, gguf_file, prefix ++ ".attn_norm.weight"),
            .ffn_norm = try weight_loader.load(ctx, gguf_file, prefix ++ ".ffn_norm.weight"),
        };
    }

    return .{
        .params = params.*,
        .token_embd = token_embd,
        .output_weight = output_weight,
        .output_norm = output_norm,
        .layers = layers,
    };
}

// ============================================================================
// 内存估算
// ============================================================================

fn estimateMemSize(gguf_file: *const gguf.GGUFFile) !usize {
    const total = gguf_file.totalTensorDataSize();
    const extra = total / 3;
    return total + extra + 1024 * 1024; // 至少 1MB 余量
}
```

### 关键要点

1. **`vtable` 必须导出**：`pub const vtable: model.ModelVTable = .{...}` 供 registry 使用
2. **`init` 签名固定**：`(self: *T, allocator, gguf_file: *gguf.GGUFFile, io: std.Io) !void`
3. **`buildGraph` 签名固定**：`(ptr: *anyopaque, builder: *graph_builder.GraphBuilder, input: *ggml.Tensor, n_tokens: i32, cache: ?*anyopaque, pos: i32) !*ggml.Tensor`
4. **`deinit` 签名固定**：`(ptr: *anyopaque, allocator: std.mem.Allocator) void`
5. **参数键名**：使用 `{architecture}.{key}` 格式（如 `mistral.embedding_length`），务必查阅对应架构的 GGUF 元数据（参考 `llama.cpp` 的 `src/models/models.h`）。
6. **权重张量名**：使用 `blk.{N}.{name}` 格式（如 `blk.0.attn_q.weight`），通常跨架构一致。
7. **错误处理**：所有可能失败的操作使用 `try` 向上传播，禁止 `@panic`。
8. **导入规范**：跨目录使用模块名（`@import("model")`、`@import("kv_cache")`），禁止 `@import("../model.zig")`。

---

## 第 2 步：注册架构枚举

编辑 `src/model.zig`：

### 2a. 在 Architecture 枚举中添加新值

```zig
pub const Architecture = enum {
    qwen2,
    qwen35,
    qwen3vl,
    llama,
    gemma3,
    gemma4,
    embedding_qwen2,
    mistral,          // <-- 新增
};
```

### 2b. 在 fromString() 中添加映射

```zig
pub fn fromString(s: []const u8) ?Architecture {
    // ... 现有代码 ...
    if (std.mem.eql(u8, s, "mistral") or
        std.mem.eql(u8, s, "mistral-v0.1"))
    {
        return .mistral;
    }
    return null;
}
```

### 2c. 导入模型模块

```zig
// 导入并重新导出模型实现（使 registry 可通过 @import("model").mistral 访问）
pub const qwen2 = @import("models/qwen2.zig");
pub const llama = @import("models/llama.zig");
// ... 现有 ...
pub const mistral = @import("models/mistral.zig");  // <-- 新增
```

---

## 第 3 步：更新注册表

编辑 `src/models/registry.zig`：

### 3a. 导入新模型

```zig
const mistral = @import("model").mistral;  // <-- 通过 model 模块访问
```

### 3b. 在 createModel() 中添加工厂分支

```zig
pub fn createModel(
    allocator: std.mem.Allocator,
    gguf_file: *gguf.GGUFFile,
    arch: model_if.Architecture,
    io: std.Io,
) !model_if.ModelInstance {
    return switch (arch) {
        // ... 现有分支 ...
        .mistral => {
            var m = try allocator.create(mistral.MistralModel);
            errdefer allocator.destroy(m);
            try m.init(allocator, gguf_file, io);
            return model_if.ModelInstance{
                .vtable = &mistral.MistralModel.vtable,
                .ptr = @as(*anyopaque, @ptrCast(m)),
            };
        },
    };
}
```

### 3c. 添加架构检测（可选回退逻辑）

若 GGUF 缺少 `general.architecture` 元数据，可通过张量名进行启发式检测：

```zig
pub fn detectArchitecture(gguf_file: *const gguf.GGUFFile) ?model_if.Architecture {
    // 优先从元数据读取
    if (gguf_file.getString("general.architecture")) |arch_str| {
        if (model.Architecture.fromString(arch_str)) |arch| {
            return arch;
        }
    }

    // 回退检测：检查张量模式
    if (gguf_file.findTensor("blk.0.attn_q.weight") != null and
        gguf_file.findTensor("blk.0.attn_k.weight") != null and
        gguf_file.findTensor("blk.0.attn_qkv.weight") == null)  // Mistral 使用分离 Q/K/V
    {
        if (gguf_file.findTensor("blk.0.attn_norm.weight") != null and
            gguf_file.findTensor("blk.0.ffn_norm.weight") != null)
        {
            log.info("Fallback: detected mistral architecture by tensor pattern", .{});
            return .mistral;
        }
    }
    return null;
}
```

### 3d. 在 detectCapabilities() 中添加能力检测

添加对应架构的能力描述，**必须设置 `special_tokens`** 以动态配置多模态媒体标记：

```zig
pub fn detectCapabilities(gguf_file, arch) model_if.ModelCapabilities {
    var caps = model_if.ModelCapabilities{};
    switch (arch) {
        // ... 其它分支 ...
        .mistral => {
            // 纯文本模型，无需额外设置
        },
        .gemma4 => {
            caps.special_tokens.img_beg = "<|image>";
            caps.special_tokens.img_end = "<image|>";
            caps.special_tokens.aud_beg = "<|audio>";
            caps.special_tokens.aud_end = "<audio|>";
        },
        .qwen3vl => {
            caps.special_tokens.img_beg = "<|vision_start|>";
            caps.special_tokens.img_end = "<|vision_end|>";
        },
        // 纯文本模型：留空或使用默认值
        else => {},
    }
    return caps;
}
```

> **重要**：`special_tokens` 使用 `ModelCapabilities.SpecialTokens` 类型（定义在 `src/model.zig`），
> 动态驱动 `chat_template/multimodal.zig` 的占位符扫描和 `mtmd/mod.zig` 的媒体标记解析。
> 新模型须在 `detectCapabilities()` 中根据架构填充对应的 `img_beg`/`img_end`/`aud_beg`/`aud_end`，
> 以消除模型特定标记的硬编码。
```

---

## 第 4 步：添加对话模板

如果新模型使用已有的模板格式（如 ChatML、Llama 3），无需修改 `chat_template/mod.zig`。

如果需要新模板：

1. 在 `src/chat_template/` 下新建模板文件（如 `mistral_v7.zig`）
2. 在 `chat_template/mod.zig` 中注册 Kind 到实现的映射
3. 在 `build.zig` 的 chat_template 模块组中注册新子模块

---

## 第 5 步：build.zig（通常无需修改）

模型文件通过 `src/model.zig` 的 `@import("models/your_model.zig")` 间接导入，`model` 模块已在 `build.zig` 中注册，因此**大多数情况下无需修改 build.zig**。

只有在模型需要额外的 C 库或系统依赖时才需修改。

---

## ✅ 验证清单

完成上述步骤后，运行以下验证：

```bash
# 1. 编译
zig build

# 2. 运行所有测试
zig build test -Doptimize=ReleaseSafe --summary all

# 3. 测试模型信息
zig-out/bin/zllama -m path/to/model.gguf --info

# 4. 测试推理
zig-out/bin/zllama -m path/to/model.gguf -p "Hello" -n 5

# 5. 对比 Logits（强制）
# 要求：NMSE < 1e-5 或余弦相似度 > 0.999
# 如差异过大，检查注意力实现、RoPE 参数、Norm 精度等
```

---

## 🔍 调试技巧

### 查看 GGUF 元数据

```bash
python3 -c "
import gguf
r = gguf.GGUFReader('path/to/model.gguf')
for k, v in r.fields.items():
    print(f'{k}: {v.value}')
"
```

### 常见 GGUF 元数据键名参考

| 参数 | 键名格式（以 `mistral` 为例） | 示例值 |
|------|-------------------------------|--------|
| 词表大小 | `mistral.vocab_size` 或 `tokenizer.ggml.vocab_size` | 32000 |
| 嵌入维度 | `mistral.embedding_length` | 4096 |
| 注意力头数 | `mistral.head_count` | 32 |
| KV 头数 | `mistral.head_count_kv` | 8 |
| 层数 | `mistral.block_count` | 32 |
| FFN 维度 | `mistral.feed_forward_length` | 14336 |
| 上下文长度 | `mistral.context_length` | 32768 |
| RoPE theta | `mistral.rope.theta` | 10000000.0 |
| RoPE 维度 | `mistral.rope.dimension_count` | 64 |
| Norm epsilon | `mistral.attention.layer_norm_rms_epsilon` | 1e-6 |
| 模型名称 | `general.name` | Mistral-7B-v0.1 |

> **注意**：不同架构的键名前缀可能不同，务必根据 `general.architecture` 或模型文档确认。

### 常见权重张量名（通常跨架构通用）

| 张量 | 名称格式 |
|------|----------|
| Token 嵌入 | `token_embd.weight` |
| 输出权重 | `output.weight` |
| 输出归一化 | `output_norm.weight` |
| 注意力 Q | `blk.{N}.attn_q.weight` |
| 注意力 K | `blk.{N}.attn_k.weight` |
| 注意力 V | `blk.{N}.attn_v.weight` |
| 注意力 O | `blk.{N}.attn_o.weight` |
| FFN gate | `blk.{N}.ffn_gate.weight` |
| FFN down | `blk.{N}.ffn_down.weight` |
| FFN up | `blk.{N}.ffn_up.weight` |
| Attn norm | `blk.{N}.attn_norm.weight` |
| FFN norm | `blk.{N}.ffn_norm.weight` |

---

## 📚 参考实现

- llama.cpp: `src/models/model_mistral.cpp`, `src/models/model_qwen2.cpp`
- 本项目已有模型: `src/models/llama.zig`, `src/models/qwen2.zig`, `src/models/gemma4.zig`

---

## 🔄 多模态扩展（如适用）

如果新模型支持图像或音频输入，需要在 `ModelCapabilities` 中声明 `has_vision`/`has_audio` 并在 `special_tokens` 中设置媒体标记，同时实现对应的编码器函数。参考 `src/mtmd/` 目录下的现有实现（如 `audio/`、`vision/`），并在 `detectCapabilities()` 中根据编码器类型配置 `special_tokens` 字段（详见步骤 3d）。

具体步骤请参考 `src/mtmd/graph.md` 或询问维护者。

---

## 💡 最后提醒

- 所有代码必须通过 `zig fmt` 格式化。
- 提交前运行完整测试套件 `zig build test -Doptimize=ReleaseSafe --summary all`。
- 若修改了公共接口（如 `model.zig` 中的枚举），请同步更新 `docs/ARCHITECTURE.md`。
- 确保新增模型的 logits 与 llama.cpp 参考一致，这是保证正确性的底线。

---

**祝新增模型顺利！** 🚀
