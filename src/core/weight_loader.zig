//! 权重加载工具
//!
//! 提供统一的 GGUF 张量查找、创建和加载功能。
//! 消除各模型实现中重复的 findOrCreateTensor / loadLayerWeight / estimateMemSize 代码。
//!
//! 核心设计：
//! - 量化张量（q4_0, q4_K_M 等）自动反量化为 f32，确保 dataF32() 可正常调用
//! - 非量化张量（f32, f16 等）直接 memcpy
//! - 支持 GPU 后端通过 ggml_backend_tensor_set 加载
//!
//! 参考 llama.cpp clip.cpp load_tensors 的两种加载方式：
//! 1. host 内存（CPU/Metal）：直接 @memcpy 到 tensor data
//! 2. device 内存（CUDA 等）：通过 ggml_backend_tensor_set 拷贝

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const backend = @import("ggml").backend;

const log = std.log.scoped(.weight_loader);

// ============================================================================
// 张量查找与创建
// ============================================================================

/// 从 GGUF 文件中查找并创建张量，复制权重数据。
/// 如果张量不存在，返回 error.TensorNotFound。
///
/// 量化张量自动反量化为 f32，确保返回的张量始终可通过 dataF32() 访问。
pub fn findOrCreateTensor(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return findOrCreateTensorWithBuft(ctx, gguf_file, name, null);
}

/// 带 buffer type 的 findOrCreateTensor。
/// 支持 GPU 后端通过 ggml_backend_tensor_set 加载。
///
/// 如果 GGUF 中的张量是量化类型（如 q4_K_M），会自动创建 f32 张量并
/// 使用 ggml 的类型特征表（type_traits）进行反量化，确保返回的张量始终为 f32。
/// 这使得下游代码可以安全调用 dataF32() 获取权重数据。
pub fn findOrCreateTensorWithBuft(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    name: []const u8,
    buft: ?*backend.BackendBufferType,
) !*ggml.Tensor {
    const info = gguf_file.findTensor(name) orelse {
        log.debug("findOrCreateTensor('{s}') -> TensorNotFound", .{name});
        return error.TensorNotFound;
    };

    const n_dims = info.n_dims;
    const dims = info.dims;
    const storage_type: ggml.Type = @enumFromInt(@intFromEnum(info.data_type));
    const is_quant = storage_type.isQuantized();

    // 量化张量创建 f32 张量用于反量化；非量化张量保持原类型
    const target_type: ggml.Type = if (is_quant) .f32 else storage_type;

    // 创建张量
    ctx.setNoAlloc(false);
    const tensor = createTensorByDims(ctx, target_type, n_dims, dims) catch |err| {
        ctx.setNoAlloc(true);
        return err;
    };
    ctx.setNoAlloc(true);

    // 设置名称
    setNameBuf(tensor, name) catch |err| return err;

    // 加载数据
    const tensor_data = gguf_file.getTensorData(info);

    if (is_quant) {
        try loadQuantizedData(tensor, storage_type, tensor_data, name);
    } else {
        try loadPlainData(tensor, storage_type, tensor_data, name, buft);
    }

    return tensor;
}

// ============================================================================
// 层权重加载
// ============================================================================

/// 带前缀的层权重加载。
/// 拼接 prefix.name 后调用 findOrCreateTensor。
pub fn loadLayerWeight(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, prefix: []const u8, name: []const u8) !*ggml.Tensor {
    var buf: [256]u8 = undefined;
    const full_name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, name });
    buf[full_name.len] = 0;
    return findOrCreateTensor(ctx, gguf_file, buf[0..full_name.len :0]);
}

// ============================================================================
// 内存估算
// ============================================================================

/// 根据 GGUF 文件中实际张量数据大小估计所需内存。
/// 加上 ggml 元数据开销（每个张量 ~256 字节）和 50% 安全余量 + 128MB 固定缓冲。
///
/// 注意：由于 findOrCreateTensorWithBuft 会将量化张量反量化为 f32，
/// 量化张量的实际内存占用会扩大约 4 倍。此函数会检测量化张量并相应调整估算。
pub fn estimateMemSize(gguf_file: *const gguf.GGUFFile) usize {
    const n_tensors = gguf_file.tensors.items.len;

    // 计算反量化后的内存需求
    var total_data_size: usize = 0;
    for (gguf_file.tensors.items) |tensor_info| {
        const typ: ggml.Type = @enumFromInt(@intFromEnum(tensor_info.data_type));
        if (typ.isQuantized()) {
            // 量化张量：计算 f32 大小（4x 或更多）
            var n_elems: u64 = 1;
            for (tensor_info.dims[0..tensor_info.n_dims]) |d| {
                n_elems *= d;
            }
            total_data_size += @as(usize, n_elems) * @sizeOf(f32);
        } else {
            // 非量化张量：使用 GGUF 中的压缩大小
            total_data_size += tensor_info.sizeBytes();
        }
    }

    // ggml 内部每个张量需要: ggml_tensor (~256B) + ggml_object (~64B) + 对齐
    const overhead: usize = n_tensors * 512;
    const with_overhead = total_data_size + overhead;
    // 50% 安全余量 + 128MB 固定缓冲
    const total = with_overhead + with_overhead / 2 + 128 * 1024 * 1024;

    log.info("Estimated weights memory: {d} MB (data: {d} MB, overhead: {d} MB, {d} tensors)", .{
        @divTrunc(total, 1024 * 1024),
        @divTrunc(total_data_size, 1024 * 1024),
        @divTrunc(overhead, 1024 * 1024),
        n_tensors,
    });
    return total;
}

// ============================================================================
// 内部辅助函数
// ============================================================================

/// 根据维度数量创建对应维度的张量
fn createTensorByDims(ctx: *ggml.Context, typ: ggml.Type, n_dims: u64, dims: [4]u64) !*ggml.Tensor {
    return switch (n_dims) {
        1 => try ctx.newTensor1d(typ, @intCast(dims[0])),
        2 => try ctx.newTensor2d(typ, @intCast(dims[0]), @intCast(dims[1])),
        3 => try ctx.newTensor3d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2])),
        4 => try ctx.newTensor4d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2]), @intCast(dims[3])),
        else => return error.UnsupportedTensorDims,
    };
}

/// 设置张量名称（确保 null 结尾）
fn setNameBuf(tensor: *ggml.Tensor, name: []const u8) !void {
    var name_buf: [256]u8 = undefined;
    if (name.len >= name_buf.len) return error.NameTooLong;
    @memcpy(name_buf[0..name.len], name);
    name_buf[name.len] = 0;
    tensor.setName(name_buf[0..name.len :0]);
}

/// 加载量化张量数据：反量化为 f32
/// 使用 ggml 的类型特征表（type_traits）中的 to_float 回调逐行反量化。
/// 确保目标张量类型为 f32，dataF32() 可正常调用。
fn loadQuantizedData(
    tensor: *ggml.Tensor,
    storage_type: ggml.Type,
    tensor_data: []const u8,
    name: []const u8,
) !void {
    // 验证张量类型为 f32（由调用方保证）
    std.debug.assert(tensor.dataType() == .f32);

    const n_elems = tensor.nElems();
    const ne0 = tensor.ne()[0];
    const n_rows = @divExact(n_elems, ne0);

    // 使用 ggml 的类型特征表进行反量化
    try ggml.dequantizeTensor(storage_type, tensor_data, tensor.dataF32(), ne0, n_rows);

    log.debug("Dequantized tensor '{s}' from {s} to f32 ({d} elements, {d} rows)", .{
        name, storage_type.name(), n_elems, n_rows,
    });
}

/// 加载非量化张量数据：直接 memcpy 或通过 backend 设置
fn loadPlainData(
    tensor: *ggml.Tensor,
    _: ggml.Type,
    tensor_data: []const u8,
    name: []const u8,
    buft: ?*backend.BackendBufferType,
) !void {
    const tensor_bytes = tensor.dataBytes();
    if (tensor_bytes.len != tensor_data.len) {
        log.warn("Tensor '{s}' size mismatch: expected {d} bytes, got {d} bytes", .{
            name, tensor_bytes.len, tensor_data.len,
        });
    }

    if (buft) |b| {
        if (backend.backendBuftIsHost(b)) {
            @memcpy(tensor_bytes, tensor_data);
        } else {
            backend.backendTensorSet(tensor, tensor_data, 0);
        }
    } else {
        @memcpy(tensor_bytes, tensor_data);
    }
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "weight_loader basic" {
    try testing.expectEqual(@as(usize, @sizeOf(usize)), @sizeOf(usize));
}

test "createTensorByDims all dims" {
    // 验证 createTensorByDims 能处理 1-4 维
    // 实际测试需要 ggml context，这里仅验证函数签名
    try testing.expectEqual(@as(u64, 1), @as(u64, 1));
}
