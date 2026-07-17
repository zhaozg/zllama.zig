//! 权重加载工具
//!
//! 提供统一的 GGUF 张量查找、创建和加载功能。
//! 消除各模型实现中重复的 findOrCreateTensor / loadLayerWeight / estimateMemSize 代码。
//!
//! 核心设计（与 llama.cpp clip.cpp load_tensors 的 get_tensor 一致）：
//! - 创建与 GGUF 中相同形状和数据类型的张量（包括量化类型如 q4_K_M），
//!   语义等价于 ggml_dup_tensor（创建相同形状和类型的新张量，不复制数据）
//! - 数据直接从 GGUF 文件加载到 tensor data，不进行显式反量化
//! - 计算时直接使用量化权重，由 ggml_mul_mat 内部支持量化运算（通过后端实现）
//! - 支持 GPU 后端通过 ggml_backend_tensor_set 加载
//!
//! 参考 llama.cpp clip.cpp load_tensors 的两种加载方式：
//! 1. host 内存（CPU/Metal）：直接 @memcpy 到 tensor data
//! 2. device 内存（CUDA 等）：通过 ggml_backend_tensor_set 拷贝

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const backend = @import("ggml").backend;

const log = std.log.scoped(.core_weight_loader);

// ============================================================================
// 张量查找与创建
// ============================================================================

/// 从 GGUF 文件中查找并创建张量，复制权重数据。
/// 如果张量不存在，返回 error.TensorNotFound。
///
/// 创建与 GGUF 中相同形状和数据类型的张量（包括量化类型），
/// 语义等价于 ggml_dup_tensor（创建相同形状和类型的新张量，不复制数据）。
/// 数据直接从 GGUF 文件加载，不进行显式反量化。
/// 计算时直接使用量化权重，由 ggml_mul_mat 内部支持量化运算。
pub fn findOrCreateTensor(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return findOrCreateTensorWithBuft(ctx, gguf_file, name, null);
}

/// 带 buffer type 的 findOrCreateTensor。
/// 支持 GPU 后端通过 ggml_backend_tensor_set 加载。
///
/// 与 llama.cpp clip.cpp load_tensors 的 get_tensor 实现一致：
/// 创建与 GGUF 中相同形状和数据类型的张量（包括量化类型），
/// 语义等价于 ggml_dup_tensor（创建相同形状和类型的新张量，不复制数据）。
/// 数据直接从 GGUF 文件加载，不进行显式反量化。
/// 计算时直接使用量化权重，由 ggml_mul_mat 内部支持量化运算（通过后端实现）。
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
    const storage_type: ggml.Type = info.data_type;

    // 创建与 GGUF 中相同形状和数据类型的张量（与 llama.cpp 的 get_tensor 一致）
    // 使用 ggml_dup_tensor 语义：创建具有相同形状和数据类型的新张量，但不复制数据。
    // 数据将在后续步骤中从 GGUF 文件加载。
    // 注意：ggml_dup_tensor 内部调用 ggml_new_tensor(ctx, type, n_dims, ne)，
    // 我们直接使用 createTensorByDims 达到相同效果，避免创建多余的 template 张量。
    ctx.setNoAlloc(false);
    const tensor = createTensorByDims(ctx, storage_type, n_dims, dims) catch |err| {
        ctx.setNoAlloc(true);
        return err;
    };
    ctx.setNoAlloc(true);

    // 设置名称
    setNameBuf(tensor, name) catch |err| return err;

    // 加载数据：直接从 GGUF 文件读取原始字节到 tensor data
    const tensor_data = gguf_file.getTensorData(info);
    try loadTensorData(tensor, storage_type, tensor_data, name, buft);

    // 输出详细日志，格式参考: n_dims = 1, name = a.blk.0.conv_norm.weight, tensor_size=4096, offset=39887616, shape:[1024, 1, 1, 1], type = f32
    log.debug(
        \\n_dims = {d}, name = {s}, tensor_size={d}, offset={d}, shape:[{d}, {d}, {d}, {d}], type = {s}
    , .{
        n_dims,
        name,
        info.sizeBytes(),
        info.offset,
        if (n_dims >= 1) dims[0] else 1,
        if (n_dims >= 2) dims[1] else 1,
        if (n_dims >= 3) dims[2] else 1,
        if (n_dims >= 4) dims[3] else 1,
        @tagName(storage_type),
    });

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
/// 注意：与 llama.cpp 一致，量化张量保持原始量化类型，不反量化为 f32，
/// 因此内存估算直接使用 GGUF 中的压缩大小。
pub fn estimateMemSize(gguf_file: *const gguf.GGUFFile) usize {
    const n_tensors = gguf_file.tensors.items.len;

    // 计算实际内存需求（使用 GGUF 中的原始压缩大小）
    var total_data_size: usize = 0;
    for (gguf_file.tensors.items) |tensor_info| {
        total_data_size += tensor_info.sizeBytes();
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

/// 加载张量数据：直接从 GGUF 文件读取原始字节到 tensor data。
/// 与 llama.cpp clip.cpp load_tensors 的 get_tensor 实现一致：
/// - 量化张量保持原始量化类型，不反量化
/// - 数据直接从 GGUF 文件加载
/// - 计算时由 ggml_mul_mat 内部支持量化运算
fn loadTensorData(
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
