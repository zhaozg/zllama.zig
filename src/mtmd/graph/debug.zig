//! 计算图调试支持模块
//!
//! 提供张量调试命名注册、中间张量数据保存等功能。
//! 底层实现委托给 src/debug.zig 中的通用调试工具。
//! 参考: deps/llama.cpp/tools/mtmd/clip-impl.h (debugging section)
//!       deps/llama.cpp/tools/mtmd/clip.cpp (mtmd_debug_save_data usage)
//!       deps/llama.cpp/tools/mtmd/mtmd.cpp (mtmd_debug_save_data implementation)

const std = @import("std");
const ggml = @import("ggml");
const debug_mod = @import("debug");

const log = std.log.scoped(.graph_debug);

// ============================================================================
// 调试张量注册表
// ============================================================================

/// 调试张量条目
pub const DebugTensorEntry = debug_mod.DebugTensorEntry;

/// 调试张量注册表
/// 保存所有注册了调试命名的张量，用于后续数据保存
/// 底层使用 src/debug.zig 中的 saveTensor 保存数据
pub const DebugTensorRegistry = struct {
    /// 存储路径（目录）
    storage_path: []const u8 = "",
    /// 文件名前缀
    file_prefix: []const u8 = "",
    /// 是否启用调试
    enabled: bool = false,
    /// 注册的张量列表
    entries: std.ArrayListUnmanaged(DebugTensorEntry) = .{ .items = &.{}, .capacity = 0 },

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

            // 委托给 src/debug.zig 中的 saveTensor
            try debug_mod.DebugTensorRegistry.saveTensor(allocator, tensor, storage_path, filename);
        }

        log.info("debug tensor save complete", .{});
    }

    /// 保存单个张量数据到文件
    ///
    /// 参数:
    ///   - allocator: 分配器（保留供将来使用）
    ///   - tensor: 要保存的张量
    ///   - storage_path: 存储目录路径
    ///   - filename: 输出文件名
    pub fn saveTensor(
        allocator: std.mem.Allocator,
        tensor: *ggml.Tensor,
        storage_path: []const u8,
        filename: []const u8,
    ) !void {
        // 委托给 src/debug.zig 中的实现
        try debug_mod.DebugTensorRegistry.saveTensor(allocator, tensor, storage_path, filename);
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
        try debug_mod.DebugTensorRegistry.saveTensor(allocator, tensor, storage_path, filename);
    }
};

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
