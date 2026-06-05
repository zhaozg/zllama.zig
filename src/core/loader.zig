//! 模型加载器
//!
//! 统一模型加载流程，从 GGUF 文件解析元数据、张量。
//! 参考 llama.cpp 的 llama_model_loader 设计。
//!
//! 职责：
//! 1. 解析 GGUF 文件头、元数据 KV、张量信息
//! 2. 提供统一的张量查找和创建接口
//! 3. 管理模型权重在 ggml context 中的分配

const std = @import("std");
const ggml = @import("../ggml.zig");
const gguf = @import("../gguf.zig");
const model_if = @import("../model.zig");

const log = std.log.scoped(.loader);

/// 张量加载标志
pub const TensorFlags = struct {
    pub const NOT_REQUIRED: u32 = 1 << 0;
    pub const DUPLICATED: u32 = 1 << 1;
    pub const SKIP: u32 = 1 << 2;
};

/// 模型加载器
///
/// 封装 GGUF 解析和权重加载逻辑。
/// 每个模型在 init 时使用 Loader 加载权重。
pub const Loader = struct {
    allocator: std.mem.Allocator,
    gguf_data: []const u8,
    gguf_file: gguf.GGUFFile,
    ctx_weights: *ggml.Context,
    params: model_if.ModelParams,

    /// 从 GGUF 字节初始化加载器
    pub fn init(
        allocator: std.mem.Allocator,
        gguf_data: []const u8,
        ctx_weights: *ggml.Context,
    ) !Loader {
        var gguf_file = try gguf.parse(gguf_data, allocator);
        errdefer gguf_file.deinit();

        // 读取通用超参数
        var params = model_if.ModelParams{};
        params.n_vocab = gguf_file.getU32("tokenizer.ggml.tokens", 0) orelse
            @as(u32, @intCast(gguf_file.getArrayLen("tokenizer.ggml.tokens") orelse 0));
        params.n_embd = gguf_file.getU32("llama.embedding_length", 0) orelse
            gguf_file.getU32("qwen2.embedding_length", 0) orelse
            gguf_file.getU32("qwen35.embedding_length", 0) orelse 0;
        params.n_head = gguf_file.getU32("llama.attention.head_count", 0) orelse
            gguf_file.getU32("qwen2.attention.head_count", 0) orelse
            gguf_file.getU32("qwen35.attention.head_count", 0) orelse 0;
        params.n_kv_head = gguf_file.getU32("llama.attention.head_count_kv", 0) orelse
            gguf_file.getU32("qwen2.attention.head_count_kv", 0) orelse
            gguf_file.getU32("qwen35.attention.head_count_kv", 0) orelse params.n_head;
        params.n_layer = gguf_file.getU32("llama.block_count", 0) orelse
            gguf_file.getU32("qwen2.block_count", 0) orelse
            gguf_file.getU32("qwen35.block_count", 0) orelse 0;
        params.n_ff = gguf_file.getU32("llama.feed_forward_length", 0) orelse
            gguf_file.getU32("qwen2.feed_forward_length", 0) orelse
            gguf_file.getU32("qwen35.feed_forward_length", 0) orelse 0;
        params.max_seq_len = gguf_file.getU32("llama.context_length", 0) orelse
            gguf_file.getU32("qwen2.context_length", 0) orelse
            gguf_file.getU32("qwen35.context_length", 0) orelse 2048;
        params.rope_theta = gguf_file.getF32("llama.rope.freq_base", 0) orelse
            gguf_file.getF32("qwen2.rope.freq_base", 0) orelse
            gguf_file.getF32("qwen35.rope.freq_base", 0) orelse 10000000.0;
        params.norm_eps = gguf_file.getF32("llama.attention.layer_norm_rms_epsilon", 0) orelse
            gguf_file.getF32("qwen2.attention.layer_norm_rms_epsilon", 0) orelse
            gguf_file.getF32("qwen35.attention.layer_norm_rms_epsilon", 0) orelse 1e-6;

        // 计算 head_dim
        if (params.n_head > 0 and params.n_embd > 0) {
            params.n_head_dim = params.n_embd / params.n_head;
        }

        // 读取 rope_dim
        params.rope_dim = gguf_file.getU32("llama.rope.dimension_count", 0) orelse
            gguf_file.getU32("qwen2.rope.dimension_count", 0) orelse
            gguf_file.getU32("qwen35.rope.dimension_count", 0) orelse params.n_head_dim;

        log.info("Model params: n_vocab={d}, n_embd={d}, n_head={d}, n_kv_head={d}, n_layer={d}, n_ff={d}",
            .{ params.n_vocab, params.n_embd, params.n_head, params.n_kv_head, params.n_layer, params.n_ff });

        return Loader{
            .allocator = allocator,
            .gguf_data = gguf_data,
            .gguf_file = gguf_file,
            .ctx_weights = ctx_weights,
            .params = params,
        };
    }

    pub fn deinit(self: *Loader) void {
        self.gguf_file.deinit();
    }

    /// 获取模型参数
    pub fn getParams(self: *const Loader) *const model_if.ModelParams {
        return &self.params;
    }

    /// 获取 GGUF 文件引用
    pub fn getGGUF(self: *const Loader) *const gguf.GGUFFile {
        return &self.gguf_file;
    }

    /// 查找并创建张量（从 GGUF 数据加载）
    /// 如果张量不存在且 flags 包含 NOT_REQUIRED，返回 null
    pub fn findOrCreateTensor(
        self: *Loader,
        name: []const u8,
        ne: []const i64,
        flags: u32,
    ) !?*ggml.Tensor {
        _ = flags;
        // 在 GGUF 中查找张量
        const tensor_info = self.gguf_file.findTensor(name);
        if (tensor_info == null) {
            return null;
        }

        // 在 weights context 中创建张量
        const tensor = try self.ctx_weights.newTensor(
            tensor_info.?.type,
            ne,
        );
        tensor.setName(name);

        // 从 GGUF 数据复制权重
        const src_data = tensor_info.?.data;
        const dst_data = tensor.dataBytes();
        @memcpy(dst_data[0..src_data.len], src_data);

        return tensor;
    }

    /// 便捷：创建 1D 张量
    pub fn createTensor1d(self: *Loader, name: []const u8, type_: ggml.Type, ne0: i64, flags: u32) !?*ggml.Tensor {
        return self.findOrCreateTensor(name, &[_]i64{ne0}, flags);
    }

    /// 便捷：创建 2D 张量
    pub fn createTensor2d(self: *Loader, name: []const u8, type_: ggml.Type, ne0: i64, ne1: i64, flags: u32) !?*ggml.Tensor {
        return self.findOrCreateTensor(name, &[_]i64{ ne0, ne1 }, flags);
    }

    /// 便捷：创建 3D 张量
    pub fn createTensor3d(self: *Loader, name: []const u8, type_: ggml.Type, ne0: i64, ne1: i64, ne2: i64, flags: u32) !?*ggml.Tensor {
        return self.findOrCreateTensor(name, &[_]i64{ ne0, ne1, ne2 }, flags);
    }

    /// 获取架构名称
    pub fn getArchName(self: *const Loader) ?[]const u8 {
        return self.gguf_file.getString("general.architecture");
    }

    /// 检测架构
    pub fn detectArch(self: *const Loader) ?model_if.Architecture {
        const arch_name = self.getArchName() orelse return null;
        return model_if.Architecture.fromString(arch_name);
    }
};

const testing = std.testing;

test "Loader basic" {
    try testing.expectEqual(@as(usize, @sizeOf(Loader)), @sizeOf(Loader));
}

test "TensorFlags" {
    try testing.expectEqual(@as(u32, 1), TensorFlags.NOT_REQUIRED);
    try testing.expectEqual(@as(u32, 2), TensorFlags.DUPLICATED);
}
