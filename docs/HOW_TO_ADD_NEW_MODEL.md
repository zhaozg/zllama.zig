# 新增模型指南

> 本文档指导如何在 zllama.zig 中添加对新模型架构的支持。

## 📋 概览

新增一个模型需要修改以下文件：

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `src/models/your_model.zig` | **新建** — 模型实现（参数、权重、前向计算） |
| 2 | `src/model.zig` | **修改** — 添加架构枚举值、导入模型模块 |
| 3 | `src/models/registry.zig` | **修改** — 添加架构检测、工厂创建、能力检测 |
| 4 | `src/chat_template.zig` | **修改** — 添加对话模板（如需要） |
| 5 | `build.zig` | **修改** — 注册新模块（如需要） |

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
        const mem_size = estimateMemSize(gguf_file);
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

            // KV Cache 写入
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
        const prefix = std.fmt.allocPrint(allocator, "blk.{d}", .{i}) catch @panic("OOM");
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

fn estimateMemSize(gguf_file: *const gguf.GGUFFile) usize {
    // 计算所有权重的总大小 + 20% 余量
    const total = gguf_file.totalTensorDataSize();
    return total + total / 5;
}
```

### 关键要点

1. **`vtable` 必须导出**：`pub const vtable: model.ModelVTable = .{...}` 供 registry 使用
2. **`init` 签名固定**：`(self, allocator, gguf_file, io) !void`
3. **`buildGraph` 签名固定**：`(ptr, builder, input, n_tokens, cache, pos) !*ggml.Tensor`
4. **参数键名**：使用 `{architecture}.{key}` 格式（如 `mistral.embedding_length`），参考 llama.cpp 的 GGUF 元数据命名
5. **权重张量名**：使用 `blk.{N}.{name}` 格式（如 `blk.0.attn_q.weight`）

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
```

### 2b. 添加 `fromString` 映射

```zig
pub fn fromString(s: []const u8) ?Architecture {
    // ... 现有代码 ...
    if (std.mem.eql(u8, s, "mistral")) {
        return .mistral;
    }
    return null;
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

### 3c. 添加架构检测

```zig
pub fn detectArchitecture(gguf_file: *const gguf.GGUFFile) ?model_if.Architecture {
    // ... 现有代码 ...
    // 在 for 循环中，fromString 会自动处理 "mistral"
    // 如果 GGUF 元数据中没有 general.architecture，可以添加张量名回退检测：
    if (arch == null) {
        if (gguf_file.findTensor("blk.0.attn_q.weight") != null and
            gguf_file.findTensor("blk.0.attn_k.weight") != null)
        {
            // 进一步区分：检查是否有特定张量
            if (gguf_file.findTensor("blk.0.attn_qkv.weight") == null) {
                log.info("Fallback: detected mistral architecture", .{});
                arch = .mistral;
            }
        }
    }
}
```

### 3d. 添加能力检测（可选）

```zig
pub fn detectCapabilities(gguf_file, arch) model_if.ModelCapabilities {
    // ... 现有代码 ...
    switch (arch) {
        // ... 现有分支 ...
        .mistral => {
            // Mistral 目前是纯文本模型
        },
    }
}
```

## 第 4 步：添加对话模板

如果新模型使用已有的模板格式（如 ChatML、Llama 3），无需修改 `chat_template.zig`。

如果需要新模板，编辑 `src/chat_template.zig`：

### 4a. 添加模板类型

```zig
pub const TemplateKind = enum {
    chatml,
    llama3,
    // ... 现有 ...
    mistral_v7,       // 可能已存在
    // ...
};
```

### 4b. 添加格式化函数

```zig
fn applyMistralV7(allocator, messages, system_prompt, add_generation_prompt) ![]const u8 {
    // 实现 Mistral 的 [INST] 格式
}
```

### 4c. 注册架构到模板映射

```zig
pub fn kindForArchitecture(arch: model.Architecture, model_name: ?[]const u8) TemplateKind {
    // ... 现有代码 ...
    return switch (arch) {
        // ... 现有 ...
        .mistral => .mistral_v7,
    };
}
```

## 第 5 步：更新 build.zig（如需要）

如果新模型需要额外的依赖模块，在 `build.zig` 中注册：

```zig
// 通常不需要，因为模型通过 model_mod 导入 layers 模块
// 只有当模型有特殊依赖时才需要
```

## ✅ 验证清单

完成上述步骤后，运行以下验证：

```bash
# 1. 编译
zig build

# 2. 运行测试
zig build test

# 3. 测试模型信息
zig-out/bin/zllama -m path/to/mistral.gguf --info

# 4. 测试推理
zig-out/bin/zllama -m path/to/mistral.gguf -p "Hello" -n 20

# 5. 测试对话模板（如适用）
zig-out/bin/zllama -m path/to/mistral.gguf -p "What is AI?" -n 50
```

## 🔍 调试技巧

### 查看 GGUF 元数据

```bash
# 使用 gguf-dump 查看模型元数据
python3 -c "
import gguf
r = gguf.GGUFReader('path/to/model.gguf')
for k, v in r.fields.items():
    print(f'{k}: {v.value}')
"
```

### 常见 GGUF 元数据键名

| 参数 | 键名格式 | 示例值 |
|------|----------|--------|
| 词表大小 | `{arch}.vocab_size` | 32000 |
| 嵌入维度 | `{arch}.embedding_length` | 4096 |
| 注意力头数 | `{arch}.head_count` | 32 |
| KV 头数 | `{arch}.head_count_kv` | 8 |
| 层数 | `{arch}.block_count` | 32 |
| FFN 维度 | `{arch}.feed_forward_length` | 14336 |
| 上下文长度 | `{arch}.context_length` | 32768 |
| RoPE theta | `{arch}.rope.theta` | 10000000.0 |
| RoPE 维度 | `{arch}.rope.dimension_count` | 64 |
| Norm epsilon | `{arch}.attention.layer_norm_rms_epsilon` | 1e-6 |
| 模型名称 | `general.name` | Mistral-7B-v0.1 |

### 常见权重张量名

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

## 📚 参考实现

- llama.cpp: `src/models/model_mistral.cpp`, `src/models/model_qwen2.cpp`
- 本项目已有模型: `src/models/llama.zig`, `src/models/qwen2.zig`
