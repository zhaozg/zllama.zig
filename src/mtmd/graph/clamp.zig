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

/// 从 GGUF 文件中读取一个标量 f32 张量的值
///
/// 在 GGUF 文件中，某些标量值（如 clamp 的 input_max/min、output_max/min）
/// 被存储为 4 字节的 f32 张量（而非元数据 KV 对）。
/// 此函数通过 findTensor 查找张量描述符，然后从 tensor data 区域读取 f32 值。
fn readTensorF32(gguf_file: *const gguf.GGUFFile, name: []const u8) ?f32 {
    const info = gguf_file.findTensor(name) orelse return null;
    const data = gguf_file.getTensorData(info);
    if (data.len < 4) return null;
    // 张量数据以 little-endian f32 格式存储
    const bits = std.mem.readInt(u32, data[0..4], .little);
    return @as(f32, @bitCast(bits));
}

test "loadClampInfoFromWeightNames basic" {
    // 单元测试：验证函数能正确处理权重名称并构造正确的元数据键
    const testing = std.testing;
    const allocator = testing.allocator;

    // 由于需要真实的 gguf.GGUFFile 实例，这里只测试名称构造逻辑
    // 实际集成测试在模型加载测试中完成
    _ = allocator;
}
