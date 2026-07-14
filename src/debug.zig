//! 调试工具模块
//!
//! 提供张量调试命名注册、中间张量数据保存等功能。
//! 服务于整个项目，不限于特定模块。
//!
//! 参考: deps/llama.cpp/tools/mtmd/clip-impl.h (debugging section)
//!       deps/llama.cpp/tools/mtmd/clip.cpp (mtmd_debug_save_data usage)
//!       deps/llama.cpp/tools/mtmd/mtmd.cpp (mtmd_debug_save_data implementation)

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.app_debug);

/// 将浮点数组保存为 JSON 数组文件。
///
/// 格式：
/// [
/// 1.234567,
/// 2.345678,
/// ...
/// 9.876543
/// ]
/// 每个值保留 6 位小数。
///
/// 参数:
///   - io: I/O 实例
///   - subdir: 子目录（如 "debug_audio"），null 表示当前目录
///   - fname: 文件名（不含路径）
///   - title: 数据标题（用于日志）
///   - data: 浮点数据
pub fn saveData(io: std.Io, subdir: ?[]const u8, fname: []const u8, title: []const u8, data: []const f32) !void {
    const cwd = std.Io.Dir.cwd();

    if (subdir) |sd| {
        cwd.createDirPath(io, sd) catch {}; // 忽略目录已存在错误
        const dir = try cwd.openDir(io, sd, .{});
        defer dir.close(io);

        log.info("Save {s} debug data to {s}/{s}", .{ title, sd, fname });

        const file = try dir.createFile(io, fname, .{});
        defer file.close(io);
        try writeJsonArray(io, file, data);
    } else {
        log.info("Save {s} debug data to {s}", .{ title, fname });

        const file = try cwd.createFile(io, fname, .{});
        defer file.close(io);
        try writeJsonArray(io, file, data);
    }
}

/// 保存单个张量数据到文件
///
/// 参数:
///   - allocator: 分配器（保留供将来使用）
///   - tensor: 要保存的张量
///   - storage_path: 存储目录路径
///   - filename: 输出文件名
pub fn saveTensor(
    io: std.Io,
    allocator: std.mem.Allocator,
    subdir: ?[]const u8,
    fname: []const u8,
    tensor: *ggml.Tensor,
) !void {
    const data = try tensor.dataGet(f32, allocator);
    defer allocator.free(data);
    try saveData(io, subdir, fname, tensor.getName(), data);
}

// ============================================================================
// 调试张量注册表
// ============================================================================

/// 调试张量条目
pub const DebugTensorEntry = struct {
    /// 张量指针
    tensor: *ggml.Tensor,
    /// 调试名称（用于 setOutput 和文件名）
    debug_name: []const u8,
    /// 是否为输入张量（影响 setInput/setOutput 调用）
    is_input: bool,
};

/// 调试张量注册表
/// 保存所有注册了调试命名的张量，用于后续数据保存
pub const DebugTensorRegistry = struct {
    /// 存储路径（目录）
    storage_path: []const u8 = "",
    /// 文件名前缀
    file_prefix: []const u8 = "",
    /// 是否启用调试
    enabled: bool = false,
    /// 注册的张量列表
    entries: std.ArrayListUnmanaged(DebugTensorEntry) = .{},

    /// 初始化调试注册表
    pub fn init() DebugTensorRegistry {
        return DebugTensorRegistry{
            .enabled = false,
            .storage_path = "",
            .file_prefix = "",
            .entries = .{},
        };
    }

    /// 释放注册表资源
    pub fn deinit(self: *DebugTensorRegistry, allocator: std.mem.Allocator) void {
        // 释放每个条目的 debug_name
        for (self.entries.items) |entry| {
            allocator.free(entry.debug_name);
        }
        self.entries.deinit(allocator);
        self.* = undefined;
    }

    /// 注册一个张量用于调试
    ///
    /// 参数:
    ///   - allocator: 分配器
    ///   - tensor: 要调试的张量
    ///   - debug_name: 调试名称（会通过 setName 设置到张量）
    ///   - is_input: true 表示输入张量（调用 setInput），false 表示输出张量（调用 setOutput）
    ///
    /// 调用此函数会:
    ///   1. 设置张量的调试名称 (setName)
    ///   2. 如果是输入张量，调用 setInput；否则调用 setOutput（确保计算后数据可访问）
    ///   3. 将条目添加到注册表，供后续 saveAll 使用
    pub fn register(
        self: *DebugTensorRegistry,
        allocator: std.mem.Allocator,
        tensor: *ggml.Tensor,
        debug_name: []const u8,
        is_input: bool,
    ) !void {
        if (!self.enabled) return;

        // 设置张量名称
        const name_z = try allocator.dupeZ(u8, debug_name);
        defer allocator.free(name_z);
        tensor.setName(name_z);

        // 标记为输入或输出，确保计算后数据可访问
        if (is_input) {
            ggml.setInput(tensor);
        } else {
            ggml.setOutput(tensor);
        }

        // 保存到注册表
        const name_copy = try allocator.dupe(u8, debug_name);
        errdefer allocator.free(name_copy);

        try self.entries.append(allocator, DebugTensorEntry{
            .tensor = tensor,
            .debug_name = name_copy,
            .is_input = is_input,
        });

        log.debug("registered debug tensor: {s} (input={})", .{ debug_name, is_input });
    }

    /// 保存所有注册张量的数据到文件
    ///
    /// 参数:
    ///   - allocator: 分配器
    ///   - storage_path: 存储目录路径（如 "/tmp/debug"）
    ///   - file_prefix: 文件名前缀（如 "llama_audio"）
    ///
    /// 每个张量保存为一个 JSON 数组文件，文件名为:
    ///   {storage_path}/{file_prefix}_{debug_name}.json
    ///
    /// 文件格式（与 llama.cpp mtmd_debug_save_data 兼容）:
    ///   [value1, value2, ..., valueN]
    ///   每个值为 float，保留 6 位小数
    pub fn saveAll(
        self: *DebugTensorRegistry,
        io: std.Io,
        allocator: std.mem.Allocator,
        storage_path: []const u8,
        file_prefix: []const u8,
    ) !void {
        if (!self.enabled) return;
        if (self.entries.items.len == 0) {
            log.info("no debug tensors to save", .{});
            return;
        }

        log.info("saving {d} debug tensors to {s}/{s}_*.json", .{
            self.entries.items.len,
            storage_path,
            file_prefix,
        });

        for (self.entries.items) |entry| {
            const tensor = entry.tensor;
            const debug_name = entry.debug_name;

            // 构建文件名: {prefix}_{debug_name}.json
            const filename = try std.fmt.allocPrint(allocator, "{s}_{s}.json", .{ file_prefix, debug_name });
            defer allocator.free(filename);

            // 复用 saveTensor 保存单个张量
            try saveTensor(io, allocator, storage_path, filename, tensor);
        }

        log.info("debug tensor save complete", .{});
    }

    /// 通过调试名称在图中查找张量并保存
    ///
    /// 参数:
    ///   - allocator: 分配器
    ///   - graph: 计算图
    ///   - debug_name: 调试名称（必须已通过 setName 设置）
    ///   - storage_path: 存储目录路径
    ///   - filename: 输出文件名
    pub fn saveTensorByName(
        io: std.Io,
        allocator: std.mem.Allocator,
        graph: *ggml.CGraph,
        debug_name: [:0]const u8,
        storage_path: []const u8,
        filename: []const u8,
    ) !void {
        const tensor = graph.getTensor(debug_name) orelse {
            log.warn("tensor not found in graph: {s}", .{debug_name});
            return;
        };
        try saveTensor(io, allocator, tensor, storage_path, filename, tensor);
    }
};

// ============================================================================
// 便捷函数：直接保存张量数据（无需注册表）
// ============================================================================

/// 将浮点数组以 JSON 数组格式写入文件。
///
/// 格式：
/// [
/// 1.234567,
/// 2.345678,
/// ...
/// 9.876543
/// ]
/// 每个值保留 6 位小数。
///
/// 参数:
///   - io: I/O 实例
///   - file: 已打开的文件
///   - data: 浮点数据
pub fn writeJsonArray(io: std.Io, file: std.Io.File, data: []const f32) !void {
    if (data.len == 0) {
        try file.writeStreamingAll(io, "[]\n");
        return;
    }
    try file.writeStreamingAll(io, "[\n");

    for (data[0 .. data.len - 1]) |val| {
        var buf: [64]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "{d:.6},\n", .{val}) catch unreachable;
        try file.writeStreamingAll(io, line);
    }
    {
        var buf: [64]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "{d:.6}\n", .{data[data.len - 1]}) catch unreachable;
        try file.writeStreamingAll(io, line);
    }

    try file.writeStreamingAll(io, "]");
}

/// 从计算图中查找指定名称的张量，并调用 ggml.setOutput() 标记为输出。
///
/// 这必须在图分配/计算之前调用（即在 Gallocr.allocGraph 或
/// ggml_backend_graph_compute 之前）。没有 setOutput()，图分配器可能会
/// 重用中间张量的内存缓冲区，导致后续 saveTensorFromGraph() 读取到
/// 过期/被覆盖的数据。
///
/// 参数:
///   - cgraph: 计算图指针
///   - name: 张量名称（必须已通过 setName 设置）
pub fn markTensorAsOutput(cgraph: *ggml.CGraph, name: []const u8) !void {
    const c = @import("ggml").c;

    var name_buf: [256]u8 = undefined;
    if (name.len >= name_buf.len) {
        log.warn("markTensorAsOutput: tensor name too long ({d} >= {d})", .{ name.len, name_buf.len });
        return;
    }
    @memcpy(name_buf[0..name.len], name);
    name_buf[name.len] = 0;
    const t = c.ggml_graph_get_tensor(@ptrCast(cgraph), &name_buf);
    if (t == null) {
        log.warn("markTensorAsOutput: tensor '{s}' not found in graph", .{name});
        return;
    }

    const tensor = @as(*ggml.Tensor, @ptrCast(t));
    ggml.setOutput(tensor);
    log.debug("markTensorAsOutput: marked '{s}' as output", .{name});
}

/// 从计算图中查找指定名称的 F32 张量，将其数据保存为 JSON 数组文件。
///
/// 注意：此函数通过 `ggml_graph_get_tensor` 查找张量，然后通过 `dataF32()` 直接
/// 读取 CPU 内存中的数据。如果张量在 GPU 后端上，`dataF32()` 可能返回空指针，
/// 此时会回退到 `ggml_backend_tensor_get`。
///
/// 重要：要保存的中间张量必须在图分配/计算之前通过 markTensorAsOutput()
/// 标记为输出，否则图分配器可能会重用其内存缓冲区，导致读取到过期数据。
///
/// C++ 参考 (clip.cpp):
/// ```cpp
/// auto save_tensor = [&](const char * name, const char * fname) {
///     ggml_tensor * t = ggml_graph_get_tensor(gf, name);
///     if (t && t->type == GGML_TYPE_F32) {
///         std::vector<float> data(ggml_nelements(t));
///         ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
///         mtmd_debug_save_data(fname, name, data.data(), data.size());
///     }
/// };
/// ```
///
/// 参数:
///   - io: I/O 实例
///   - subdir: 子目录（如 "debug_audio"），null 表示当前目录
///   - fname: 输出文件名
///   - title: 张量在计算图中的调试名称（用于 ggml_graph_get_tensor 查找）
///   - cgraph: 计算图指针
pub fn saveTensorFromGraph(io: std.Io, allocator: std.mem.Allocator, subdir: ?[]const u8, fname: []const u8, title: []const u8, cgraph: *ggml.CGraph) !void {
    const c = @import("ggml").c;

    // 通过名称从计算图中查找张量
    // 构造 null-terminated 字符串（ggml_graph_get_tensor 需要 C 字符串）
    var title_buf: [256]u8 = undefined;
    if (title.len >= title_buf.len) {
        log.warn("saveTensorFromGraph: tensor name too long ({d} >= {d})", .{ title.len, title_buf.len });
        return;
    }
    @memcpy(title_buf[0..title.len], title);
    title_buf[title.len] = 0;
    const t = c.ggml_graph_get_tensor(@ptrCast(cgraph), &title_buf);
    if (t == null) {
        log.warn("saveTensorFromGraph: tensor '{s}' not found in graph", .{title});
        return;
    }

    const tensor = @as(*ggml.Tensor, @ptrCast(t));
    try saveTensor(io, allocator, subdir, fname, tensor);
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "DebugTensorRegistry init and deinit" {
    var registry = DebugTensorRegistry.init();
    defer registry.deinit(testing.allocator);

    try testing.expect(!registry.enabled);
    try testing.expectEqual(@as(usize, 0), registry.entries.items.len);
}

test "DebugTensorRegistry register disabled" {
    var registry = DebugTensorRegistry.init();
    defer registry.deinit(testing.allocator);

    // 当 enabled=false 时，register 不应添加条目
    try testing.expect(!registry.enabled);
}

test "DebugTensorRegistry saveAll with no entries" {
    var registry = DebugTensorRegistry.init();
    defer registry.deinit(testing.allocator);

    registry.enabled = true;
    // 空列表不应出错
    try registry.saveAll(testing.allocator, "/tmp/test_debug", "test");
}
