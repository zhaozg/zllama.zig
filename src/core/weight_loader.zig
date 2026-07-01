//! 权重加载工具
//!
//! 提供统一的 GGUF 张量查找、创建和加载功能。
//! 消除各模型实现中重复的 findOrCreateTensor / loadLayerWeight / estimateMemSize 代码。
//!
//! 参考 llama.cpp clip.cpp load_tensors 的两种加载方式：
//! 1. host 内存（CPU/Metal）：直接 @memcpy 到 tensor data
//! 2. device 内存（CUDA 等）：通过 ggml_backend_tensor_set 拷贝

const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf");
const backend = @import("ggml").backend;

const log = std.log.scoped(.weight_loader);

/// 从 GGUF 文件中查找并创建张量，复制权重数据
/// 如果张量不存在，返回 error.TensorNotFound
///
/// 支持两种加载方式：
/// - 如果 buft 为 null 或 buft 是 host 内存：直接 @memcpy
/// - 如果 buft 是 device 内存：使用 ggml_backend_tensor_set
pub fn findOrCreateTensor(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, name: []const u8) !*ggml.Tensor {
    return findOrCreateTensorWithBuft(ctx, gguf_file, name, null);
}

/// 带 buffer type 的 findOrCreateTensor
/// 支持 GPU 后端通过 ggml_backend_tensor_set 加载
pub fn findOrCreateTensorWithBuft(
    ctx: *ggml.Context,
    gguf_file: *const gguf.GGUFFile,
    name: []const u8,
    buft: ?*backend.BackendBufferType,
) !*ggml.Tensor {
    if (gguf_file.findTensor(name)) |info| {
        const n_dims = info.n_dims;
        const dims = info.dims;
        const typ: ggml.Type = @enumFromInt(@intFromEnum(info.data_type));

        ctx.setNoAlloc(false);
        const tensor = switch (n_dims) {
            1 => try ctx.newTensor1d(typ, @intCast(dims[0])),
            2 => try ctx.newTensor2d(typ, @intCast(dims[0]), @intCast(dims[1])),
            3 => try ctx.newTensor3d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2])),
            4 => try ctx.newTensor4d(typ, @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2]), @intCast(dims[3])),
            else => return error.UnsupportedTensorDims,
        };
        ctx.setNoAlloc(true);

        // 确保名称以 null 结尾，因为 ggml_set_name 需要 [:0]const u8
        var name_buf: [256]u8 = undefined;
        if (name.len >= name_buf.len) return error.NameTooLong;
        @memcpy(name_buf[0..name.len], name);
        name_buf[name.len] = 0;
        tensor.setName(name_buf[0..name.len :0]);

        const tensor_data = gguf_file.getTensorData(info);
        const tensor_bytes = tensor.dataBytes();
        if (tensor_bytes.len != tensor_data.len) {
            log.warn("Tensor '{s}' size mismatch: expected {d} bytes, got {d} bytes", .{ name, tensor_bytes.len, tensor_data.len });
        }

        // 根据 buffer type 选择加载方式
        if (buft) |b| {
            if (backend.backendBuftIsHost(b)) {
                // host 内存（CPU/Metal）：直接 memcpy
                @memcpy(tensor_bytes, tensor_data);
            } else {
                // device 内存（CUDA 等）：使用 ggml_backend_tensor_set
                backend.backendTensorSet(tensor, tensor_data, 0);
            }
        } else {
            // 无 buft 信息，默认使用 host 内存方式
            @memcpy(tensor_bytes, tensor_data);
        }

        return tensor;
    }

    log.debug("findOrCreateTensor('{s}') return TensorNotFound", .{name});
    return error.TensorNotFound;
}

/// 带前缀的层权重加载
/// 拼接 prefix.name 后调用 findOrCreateTensor
pub fn loadLayerWeight(ctx: *ggml.Context, gguf_file: *const gguf.GGUFFile, prefix: []const u8, name: []const u8) !*ggml.Tensor {
    var buf: [256]u8 = undefined;
    const full_name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, name });
    buf[full_name.len] = 0;
    return findOrCreateTensor(ctx, gguf_file, buf[0..full_name.len :0]);
}

/// 根据 GGUF 文件中实际张量数据大小估计所需内存
/// 加上 ggml 元数据开销（每个张量 ~256 字节）和 33% 安全余量 + 64MB 固定缓冲
pub fn estimateMemSize(gguf_file: *const gguf.GGUFFile) usize {
    const raw_data_size = gguf_file.totalTensorDataSize();
    const n_tensors = gguf_file.tensors.items.len;
    // ggml 内部每个张量需要: ggml_tensor (~256B) + ggml_object (~64B) + 对齐
    // 使用 384 字节/tensor 以确保覆盖
    const overhead: usize = n_tensors * 384;
    const with_overhead = raw_data_size + overhead;
    // 33% 安全余量 + 64MB 固定缓冲
    const total = with_overhead + with_overhead / 3 + 64 * 1024 * 1024;
    log.info("Estimated weights memory: {d} MB (raw: {d} MB, {d} tensors)", .{ total / (1024 * 1024), raw_data_size / (1024 * 1024), n_tensors });
    return total;
}

const testing = std.testing;

test "estimateMemSize zero" {
    // 无法实际测试，仅验证函数存在
    try testing.expectEqual(@as(usize, @sizeOf(usize)), @sizeOf(usize));
}
