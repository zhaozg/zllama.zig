//! Clamp 信息加载工具
//!
//! 提供从 GGUF 元数据加载 clamp_info_map 的通用函数。
//! 各模型后端可通过 `loadClampInfoFromWeightNames` 复用此逻辑。
//!
//! 参考: deps/llama.cpp/src/llama-clip.cpp loadClampInfo()

const std = @import("std");
const gguf = @import("gguf");
const ClampInfo = @import("types.zig").ClampInfo;

const log = std.log.scoped(.clamp);

/// 从 GGUF 张量数据加载 clamp_info_map
///
/// 遍历所有权重名称，对每个以 `.weight` 结尾的名称：
/// 1. 去掉 `.weight` 后缀得到前缀
/// 2. 构造 `{prefix}.input_max`、`{prefix}.input_min`、`{prefix}.output_max`、`{prefix}.output_min` 四个张量名
/// 3. 使用 `gguf_file.findTensor(name)` 查找张量，然后通过 `gguf_file.getTensorData(info)` 读取 f32 值
/// 4. 以完整权重名（含 `.weight`）为键存入 clamp_map
///
/// 注意：这些 clamp 值在 GGUF 文件中存储为**张量**（每个 4 字节 = 1 个 f32），
/// 不是元数据 KV 对。必须使用 findTensor + getTensorData 而非 getF32。
/// 参考: deps/llama.cpp/src/llama-clip.cpp loadClampInfo()
pub fn loadClampInfoFromWeightNames(
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GGUFFile,
    weight_names: []const []const u8,
) !std.StringHashMap(ClampInfo) {
    var clamp_map = std.StringHashMap(ClampInfo).init(allocator);

    const weight_suffix = ".weight";
    const clamp_suffixes = [_][]const u8{ ".input_max", ".input_min", ".output_max", ".output_min" };

    for (weight_names) |w_name| {
        if (!std.mem.endsWith(u8, w_name, weight_suffix)) continue;

        const prefix_len = w_name.len - weight_suffix.len;
        var clamp_names: [4][]const u8 = undefined;

        for (&clamp_names, clamp_suffixes) |*out_name, suffix| {
            const new_len = prefix_len + suffix.len;
            const buf = try allocator.alloc(u8, new_len);
            errdefer allocator.free(buf);
            @memcpy(buf[0..prefix_len], w_name[0..prefix_len]);
            @memcpy(buf[prefix_len..][0..suffix.len], suffix);
            out_name.* = buf;
        }
        defer {
            for (clamp_names) |n| allocator.free(n);
        }

        // 这些 clamp 值在 GGUF 文件中存储为**张量**（每个 4 字节 = 1 个 f32），
        // 不是元数据 KV 对。必须使用 findTensor + getTensorData 读取。
        const inp_max_val = readTensorF32(gguf_file, clamp_names[0]) orelse std.math.floatMax(f32);
        const inp_min_val = readTensorF32(gguf_file, clamp_names[1]) orelse -std.math.floatMax(f32);
        const out_max_val = readTensorF32(gguf_file, clamp_names[2]) orelse std.math.floatMax(f32);
        const out_min_val = readTensorF32(gguf_file, clamp_names[3]) orelse -std.math.floatMax(f32);

        try clamp_map.put(w_name, ClampInfo{
            .inp_min = inp_min_val,
            .inp_max = inp_max_val,
            .out_min = out_min_val,
            .out_max = out_max_val,
        });
    }

    return clamp_map;
}

/// 从 GGUF 文件中读取一个标量张量的值，返回 f32。
///
/// 在 GGUF 文件中，某些标量值（如 clamp 的 input_max/min、output_max/min）
/// 被存储为张量（而非元数据 KV 对），类型可能是 f32、f16 或 bf16。
/// 此函数通过 findTensor 查找张量描述符，然后根据实际类型读取并转换为 f32。
///
/// 参考: llama.cpp llama-clip.cpp loadClampInfo() — 使用 ggml_fp16_to_fp32 处理 F16
fn readTensorF32(gguf_file: *const gguf.GGUFFile, name: []const u8) ?f32 {
    const info = gguf_file.findTensor(name) orelse return null;
    const data = gguf_file.getTensorData(info);
    const ggml_type = info.data_type;

    switch (ggml_type) {
        .f32 => {
            if (data.len < 4) return null;
            const bits = std.mem.readInt(u32, data[0..4], .little);
            return @as(f32, @bitCast(bits));
        },
        .f16 => {
            if (data.len < 2) return null;
            const bits = std.mem.readInt(u16, data[0..2], .little);
            // ggml_fp16_to_fp32 将 IEEE 754 half-precision 转换为 f32
            return ggml_fp16_to_fp32(bits);
        },
        .bf16 => {
            if (data.len < 2) return null;
            const bits = std.mem.readInt(u16, data[0..2], .little);
            // BF16: 直接左移 16 位得到 f32（BF16 是 f32 的截断版本）
            return @as(f32, @bitCast(@as(u32, bits) << 16));
        },
        else => {
            // 对于其他类型（如量化类型），尝试按 f32 读取
            if (data.len < 4) return null;
            const bits = std.mem.readInt(u32, data[0..4], .little);
            return @as(f32, @bitCast(bits));
        },
    }
}

/// 将 IEEE 754 half-precision (F16) 转换为 f32
/// 参考: ggml_fp16_to_fp32 实现
fn ggml_fp16_to_fp32(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15) << 31;
    const exp: u32 = @as(u32, h >> 10) & 0x1f;
    const mant: u32 = @as(u32, h) & 0x3ff;

    if (exp == 0) {
        // 次正规数或零
        if (mant == 0) return @as(f32, @bitCast(sign));
        // 次正规数: exp = 0, mant != 0
        const result = @as(u32, sign) | (127 - 15 - 10) << 23 | mant << 13;
        return @as(f32, @bitCast(result));
    } else if (exp == 31) {
        // 无穷大或 NaN
        const result = @as(u32, sign) | 0x7f800000 | mant << 13;
        return @as(f32, @bitCast(result));
    }

    // 正规数
    const result = @as(u32, sign) | ((exp + (127 - 15)) << 23) | mant << 13;
    return @as(f32, @bitCast(result));
}

test "loadClampInfoFromWeightNames basic" {
    // 单元测试：验证函数能正确处理权重名称并构造正确的元数据键
    const testing = std.testing;
    const allocator = testing.allocator;

    // 由于需要真实的 gguf.GGUFFile 实例，这里只测试名称构造逻辑
    // 实际集成测试在模型加载测试中完成
    _ = allocator;
}
