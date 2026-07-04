# 新增模型指南

> 本文档指导如何在 zllama.zig 中添加对新模型架构的支持。

---

## 📋 概览

新增一个模型需要修改以下文件：

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `src/models/your_model.zig` | **新建** — 模型实现（参数、权重、前向计算） |
| 2 | `src/model.zig` | **修改** — 添加架构枚举值、导入模型模块 |
| 3 | `src/models/registry.zig` | **修改** — 添加架构检测、工厂创建、能力检测 |
| 4 | `src/chat_template/mod.zig` | **修改** — 添加对话模板（如需要） |
| 5 | `build.zig` | **修改** — 注册新模块（如需要） |

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

const model = @import("../model.zig");

const log = std.log.scoped(.mistral);

// ============================================================================
// 超参数
// ============================================================================

/// Mistral 模型超参数（从 GGUF 元数据读取）
pub const MistralParams = struct {
    // 复用 model.ModelParams 作为基础
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
// 模型结构体
// ============================================================================

pub const MistralModel = struct {
    allocator: std.mem.Allocator,
    weights: MistralWeights,
    ctx: *ggml.Context,

    pub const vtable: model.ModelVTable = .{
        .getParams = getParams,
        .buildGraph = buildGraph,
        .deinit = deinit,
    };

    fn getParams(ptr: *anyopaque) *const model.ModelParams {
        const self = @as(*MistralModel, @ptrCast(@alignCast(ptr)));
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
        const ctx = try ggml.Context.initNoAlloc(mem_size);
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
        const self = @as(*MistralModel, @ptrCast(@alignCast(ptr)));
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
        const self = @as(*MistralModel, @ptrCast(@alignCast(ptr)));
        const w = &self.weights;
        const params = &w.params.base;

        // 获取 KV cache
        const kv = @as(*kv_cache.KVCache, @ptrCast(@alignCast(cache.?)));

        // --- 构建计算图 ---

        // 1. Token 嵌入
        var cur = try embed.forward(builder, input, w.token_embd, params.n_embd);

        // 2. 逐层处理
        for (w.layers, 0..) |*layer, i| {
            const layer_id = @as(i32, @intCast(i));

            // RMSNorm
            cur = try rms_norm.forward(builder, cur, layer.attn_norm, params.norm_eps);

            // RoPE + 注意力
            const n_heads = @as(i32, @intCast(params.n_head));
            const n_kv_heads = @as(i32, @intCast(params.n_kv_head));
            const head_dim = @as(i32, @intCast(params.n_head_dim));
            const rope_dim = @as(i32, @intCast(params.rope_dim));

            const q = try rope.forward(builder, cur, layer.attn_q, n_heads, head_dim, rope_dim, pos, params.rope_theta);
            const k = try rope.forward(builder, cur, layer.attn_k, n_kv_heads, head_dim, rope_dim, pos, params.rope_theta);
            const v = layer.attn_v; // V 不做 RoPE

            // KV Cache 写入（零拷贝视图）
            try kv.setK(io, layer_id, pos, k);
            try kv.setV(io, layer_id, pos, v);

            // 注意力计算
            const k_cache = try kv.getK(io, builder, layer_id, n_tokens);
            const v_cache = try kv.getV(io, builder, layer_id, n_tokens);
            cur = try attention.forward(builder, q, k_cache, v_cache, layer.attn_o, n_heads, n_kv_heads, head_dim, n_tokens);

            // FFN (SwiGLU)
            cur = try rms_norm.forward(builder, cur, layer.ffn_norm, params.norm_eps);
            cur = try swiglu.forward(builder, cur, layer.ffn_gate, layer.ffn_up, layer.ffn_down);
        }

        // 3. 输出层归一化
        cur = try rms_norm.forward(builder, cur, w.output_norm, params.norm_eps);

        // 4. 输出投影
        const output = w.output_weight orelse w.token_embd;
        return try builder.forwardOutput(cur, output, params.n_vocab);
    }
};

// ============================================================================
// 参数解析
// ============================================================================

pub fn parseParams(gguf_file: *const gguf.GGUFFile, allocator: std.mem.Allocator) !MistralParams {
    _ = allocator;
    var params: MistralParams = .{};

    const b = &params.base;
    // 注意：键名前缀需根据实际架构名称替换（如 mistral、qwen2、gemma 等）
    // 可从 GGUF 的 general.architecture 获取架构名，但此处直接按已知架构名读取
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

    // 加载非分层权重
    const token_embd = try weight_loader.load(ctx, gguf_file, "token_embd.weight");
    const output_weight = weight_loader.load(ctx, gguf_file, "output.weight") catch null;
    const output_norm = try weight_loader.load(ctx, gguf_file, "output_norm.weight");

    // 加载分层权重
    const layers = try allocator.alloc(LayerWeights, n_layers);
    errdefer allocator.free(layers);

    for (0..n_layers) |i| {
        // 使用 try 传播 OOM 错误，而非 panic
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
    // 基础张量数据大小
    const total = gguf_file.totalTensorDataSize();
    // 根据 ggml 的上下文开销（节点、中间张量等）适当增加
    // 参考 llama.cpp 中 llama_model_quantize 的估算方式
    // 这里给予 30% 余量，对于超大模型可能需要动态调整
    const extra = total / 3;
    // 确保至少 1MB
    return total + extra + 1024 * 1024;
}
```

### 关键要点

1. **`vtable` 必须导出**：`pub const vtable: model.ModelVTable = .{...}` 供 registry 使用
2. **`init` 签名固定**：`(self, allocator, gguf_file, io) !void`
3. **`buildGraph` 签名固定**：`(ptr, builder, input, n_tokens, cache, pos) !*ggml.Tensor`
4. **参数键名**：使用 `{architecture}.{key}` 格式（如 `mistral.embedding_length`），**务必**查阅对应架构的 GGUF 元数据（参考 `llama.cpp` 的 `src/models/models.h` 或 `src/gguf.cpp` 中的键名定义）。如果架构名包含连字符（如 `gemma-4`），建议在 `model.Architecture.fromString` 中统一转换为枚举值后使用标准小写名称作为键前缀。
5. **权重张量名**：使用 `blk.{N}.{name}` 格式（如 `blk.0.attn_q.weight`），通常跨架构一致。
6. **错误处理**：所有可能失败的操作（包括内存分配）均使用 `try` 向上传播，禁止使用 `@panic`。
7. **I/O 传递**：在需要文件访问的地方（如 KV Cache 持久化）显式传递 `io: std.Io` 参数。

---

## 第 2 步：注册架构枚举

编辑 `src/model.zig`：

### 2a. 添加架构枚举值

```zig
pub const Architecture = enum {
    qwen2,
    qwen35,
    llama,
    gemma3,
    gemma4,
    embedding_qwen2,
    mistral,          // <-- 新增
    // ...
};
```

### 2b. 添加 `fromString` 映射

在 `fromString` 函数中增加一个分支。**建议先进行归一化**（小写、去连字符），再与已知名称比较：

```zig
pub fn fromString(s: []const u8) ?Architecture {
    // 归一化：转小写，去除连字符
    var buf: [64]u8 = undefined;
    const normalized = normalizeArchName(s, &buf) catch return null;

    // ... 现有代码 ...
    if (std.mem.eql(u8, normalized, "mistral")) {
        return .mistral;
    }
    // 其它映射...
    return null;
}

// 辅助函数（也可直接内联）
fn normalizeArchName(name: []const u8, buf: []u8) ![]u8 {
    var i: usize = 0;
    for (name) |c| {
        if (c == '-') continue;
        if (i >= buf.len) return error.BufferTooSmall;
        buf[i] = std.ascii.toLower(c);
        i += 1;
    }
    return buf[0..i];
}
```

### 2c. 导入模型模块

```zig
pub const qwen2 = @import("models/qwen2.zig");
pub const qwen35 = @import("models/qwen35.zig");
pub const llama = @import("models/llama.zig");
pub const gemma3 = @import("models/gemma3.zig");
pub const gemma4 = @import("models/gemma4.zig");
pub const embedding = @import("models/embedding.zig");
pub const mistral = @import("models/mistral.zig");  // <-- 新增
```

---

## 第 3 步：更新注册表

编辑 `src/models/registry.zig`：

### 3a. 导入新模型

```zig
const mistral = @import("model").mistral;  // <-- 新增
```

### 3b. 添加工厂分支

```zig
pub fn createModel(allocator, gguf_file, arch, io) !model_if.ModelInstance {
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
    // 注意：务必确保不会误判，最好结合多个特征
    if (gguf_file.findTensor("blk.0.attn_q.weight") != null and
        gguf_file.findTensor("blk.0.attn_k.weight") != null and
        gguf_file.findTensor("blk.0.attn_qkv.weight") == null)  // Mistral 使用分离的 Q/K/V
    {
        // 进一步区分：检查是否有 `blk.0.attn_norm.weight`（Mistral 有）
        if (gguf_file.findTensor("blk.0.attn_norm.weight") != null) {
            log.info("Fallback: detected mistral architecture by tensor pattern", .{});
            return .mistral;
        }
    }
    return null;
}
```

### 3d. 添加能力检测（多模态支持）

如果新模型支持图像或音频输入，在 `detectCapabilities` 中设置相应的标志：

```zig
pub fn detectCapabilities(gguf_file, arch) model_if.ModelCapabilities {
    var caps = model_if.ModelCapabilities{};
    // 基础文本能力默认开启
    caps.text = true;
    switch (arch) {
        // ... 其它分支 ...
        .mistral => {
            // 纯文本，无需额外设置
        },
        // 如果模型支持视觉，可设置：
        // .my_multimodal => {
        //     caps.vision = true;
        //     caps.audio = false;
        // },
    }
    return caps;
}
```

---

## 第 4 步：添加对话模板

如果新模型使用已有的模板格式（如 ChatML、Llama 3），无需修改 `chat_template/mod.zig`。

如果需要新模板，编辑 `src/chat_template/mod.zig`：

### 4a. 添加模板类型（如果需要）

```zig
pub const TemplateKind = enum {
    chatml,
    llama3,
    // ... 现有 ...
    mistral_v7,       // 可能已存在，若没有则新增
    // ...
};
```

### 4b. 添加格式化函数（如果需要）

```zig
fn applyMistralV7(allocator, messages, system_prompt, add_generation_prompt) ![]const u8 {
    // 实现 Mistral 的 [INST] 格式
    // 参考：https://docs.mistral.ai/chat/templates/
}
```

### 4c. 注册架构到模板映射

```zig
pub fn kindForArchitecture(arch: model.Architecture, model_name: ?[]const u8) TemplateKind {
    // ... 现有代码 ...
    return switch (arch) {
        // ... 现有 ...
        .mistral => .mistral_v7,
        // 若已有对应模板则使用之
    };
}
```

---

## 第 5 步：更新 `build.zig`（如需额外依赖）

大多数情况下，新模型只需使用现有 `layers` 和 `core` 模块，无需修改 `build.zig`。

若模型需要额外的 C 库或系统依赖（如 OpenCV 用于多模态预处理），在 `build.zig` 的 `exe` 或 `staticLibrary` 中添加链接：

```zig
exe.linkSystemLibrary("opencv_core");
exe.linkSystemLibrary("opencv_imgproc");
```

但通常不建议引入外部依赖，除非必须。

---

## ✅ 验证清单

完成上述步骤后，运行以下验证：

```bash
# 1. 编译
zig build

# 2. 运行所有测试
zig build test

# 3. 测试模型信息（检查参数读取正确）
zig-out/bin/zllama -m path/to/mistral.gguf --info

# 4. 测试推理（生成简短文本）
zig-out/bin/zllama -m path/to/mistral.gguf -p "Hello" -n 20

# 5. 测试对话模板（如适用）
zig-out/bin/zllama -m path/to/mistral.gguf -p "What is AI?" -n 50

# 6. **对比 Logits（强制）**
# 使用 compare_logits 工具与 llama.cpp 参考输出对比
tools/compare_logits.zig --model path/to/mistral.gguf --prompt "Once upon a time" --n-tokens 5
# 要求：NMSE < 1e-5 或余弦相似度 > 0.999
# 如果差异过大，检查注意力实现、RoPE 参数、Norm 精度等
```

---

## 🔍 调试技巧

### 查看 GGUF 元数据

```bash
# 使用 Python gguf 库查看所有键值对
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

如果新模型支持图像或音频输入，需要在 `ModelCapabilities` 中声明，并实现对应的编码器函数。请参考 `src/mtmd/` 目录下的现有实现（如 `vision.zig`、`audio.zig`），并在模型结构体中增加 `encode_image` 和 `encode_audio` 方法，然后通过 `ModelInstance` 的虚表暴露。

具体步骤可参考 `HOW_TO_ADD_MULTIMODAL.md`（若有）或询问维护者。

---

## 💡 最后提醒

- 所有代码必须通过 `zig fmt` 格式化。
- 提交前运行完整测试套件 `zig build test -Doptimize=ReleaseSafe --summary all`。
- 若修改了公共接口（如 `model.zig` 中的枚举），请同步更新 `docs/ARCHITECTURE.md`。
- 确保新增模型的 logits 与 llama.cpp 参考一致，这是保证正确性的底线。

---

**祝新增模型顺利！** 🚀
