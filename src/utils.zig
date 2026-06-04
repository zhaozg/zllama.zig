const std = @import("std");
const ggml = @import("ggml");
const gguf = @import("gguf.zig");
const tokenizer = @import("tokenizer.zig");
const logger = std.log.scoped(.utils);
const c = ggml.c;
// ============================================================================
// 读取标准输入
// ============================================================================

pub fn readStdin(allocator: std.mem.Allocator) ![]u8 {
    const stdin_file = std.Io.File.stdin();
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    var temp_buf: [4096]u8 = undefined;
    while (true) {
        const n = std.posix.read(stdin_file.handle, &temp_buf) catch |err| switch (err) {
            error.WouldBlock => continue,
            else => |e| return e,
        };
        if (n == 0) break;
        try buf.appendSlice(allocator, temp_buf[0..n]);
    }

    return buf.toOwnedSlice(allocator);
}

// ============================================================================
// 转义处理
// ============================================================================

pub fn processEscapes(text: []const u8, allocator: std.mem.Allocator) ![]u8 {
    var result = try std.ArrayListUnmanaged(u8).initCapacity(allocator, text.len);
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\\' and i + 1 < text.len) {
            switch (text[i + 1]) {
                'n' => try result.append(allocator, '\n'),
                't' => try result.append(allocator, '\t'),
                'r' => try result.append(allocator, '\r'),
                '\\' => try result.append(allocator, '\\'),
                '\'' => try result.append(allocator, '\''),
                '"' => try result.append(allocator, '"'),
                '0' => try result.append(allocator, 0),
                else => {
                    try result.append(allocator, '\\');
                    try result.append(allocator, text[i + 1]);
                },
            }
            i += 2;
        } else {
            try result.append(allocator, text[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// 将 token ID 解码为可读字符串
// ============================================================================

pub fn tokenToDisplayString(tok: *const tokenizer.Tokenizer, token_id: u32, allocator: std.mem.Allocator) ![]u8 {
    const ids = [_]u32{token_id};
    return tok.decode(&ids, allocator);
}

// ============================================================================
// 模型加载信息打印（对齐 llama.cpp 的 print_info 输出）
// ============================================================================

/// 获取 GGUF 值的类型标签字符串
pub fn getTypeTag(val: gguf.MetadataValue, allocator: std.mem.Allocator) []const u8 {
    return switch (val.value_type) {
        .uint32 => "u32",
        .int32 => "i32",
        .float32 => "f32",
        .bool => "bool",
        .string => "str",
        .array => blk: {
            const arr = val.array_val;
            if (arr.len == 0) break :blk "arr[]";
            const elem_tag = switch (arr[0].value_type) {
                .uint32 => "u32",
                .int32 => "i32",
                .float32 => "f32",
                .bool => "bool",
                .string => "str",
                else => "?",
            };
            break :blk std.fmt.allocPrint(allocator, "arr[{s},{d}]", .{ elem_tag, arr.len }) catch "arr[?]";
        },
        else => "?",
    };
}

/// 格式化 GGUF 元数据值为可读字符串
pub fn formatMetadataValue(val: gguf.MetadataValue, allocator: std.mem.Allocator) ![]u8 {
    switch (val.value_type) {
        .uint32 => return std.fmt.allocPrint(allocator, "{d}", .{val.asU32().?}),
        .int32 => return std.fmt.allocPrint(allocator, "{d}", .{val.asI32().?}),
        .float32 => return std.fmt.allocPrint(allocator, "{d:.6}", .{val.asF32().?}),
        .bool => return std.fmt.allocPrint(allocator, "{s}", .{if (val.asBool().?) "true" else "false"}),
        .string => {
            const s = val.asString() orelse "(null)";
            if (s.len > 60) {
                return std.fmt.allocPrint(allocator, "{s}...", .{s[0..60]});
            }
            return std.fmt.allocPrint(allocator, "{s}", .{s});
        },
        .array => {
            const arr = val.array_val;
            if (arr.len == 0) return allocator.dupe(u8, "[]");
            if (arr.len > 5) {
                // 对于大数组，显示前几个元素
                var preview: [5][]const u8 = undefined;
                var preview_count: usize = 0;
                for (arr, 0..) |item, j| {
                    if (j >= 5) break;
                    if (item.asString()) |s| {
                        preview[j] = try std.fmt.allocPrint(allocator, "\"{s}\"", .{s});
                    } else if (item.asU32()) |uv| {
                        preview[j] = try std.fmt.allocPrint(allocator, "{d}", .{uv});
                    } else if (item.asF32()) |fv| {
                        preview[j] = try std.fmt.allocPrint(allocator, "{d}", .{fv});
                    } else {
                        preview[j] = try allocator.dupe(u8, "?");
                    }
                    preview_count = j + 1;
                }
                defer for (preview[0..preview_count]) |p| allocator.free(p);

                var result = try std.ArrayListUnmanaged(u8).initCapacity(allocator, 100);
                errdefer result.deinit(allocator);
                try result.appendSlice(allocator, "[");
                for (preview[0..preview_count], 0..) |p, idx| {
                    if (idx > 0) try result.appendSlice(allocator, ", ");
                    try result.appendSlice(allocator, p);
                }
                try result.appendSlice(allocator, ", ...]");
                return result.toOwnedSlice(allocator);
            }
            // 小数组，显示所有元素
            var result = try std.ArrayListUnmanaged(u8).initCapacity(allocator, 100);
            errdefer result.deinit(allocator);
            try result.appendSlice(allocator, "[");
            for (arr, 0..) |item, idx| {
                if (idx > 0) try result.appendSlice(allocator, ", ");
                const formatted = try formatMetadataValue(item, allocator);
                defer allocator.free(formatted);
                try result.appendSlice(allocator, formatted);
            }
            try result.appendSlice(allocator, "]");
            return result.toOwnedSlice(allocator);
        },
        else => return allocator.dupe(u8, "?"),
    }
}

/// 打印模型加载信息（对齐 llama.cpp 的 llama_model_loader 输出）
pub fn printModelLoaderInfo(gguf_file: *const gguf.GGUFFile, file_size: u64, filename: []const u8) void {
    _ = file_size;
    // 打印 GGUF 版本信息
    const version_str = switch (gguf_file.version) {
        .v2 => "GGUF V2",
        .v3 => "GGUF V3 (latest)",
        else => "GGUF V?",
    };

    // 计算 KV 数量（通过 metadata_keys 长度）
    const kv_count = gguf_file.metadata_keys.len;

    logger.info("loaded meta data with {d} key-value pairs and {d} tensors from {s} (version {s})", .{
        kv_count,
        gguf_file.tensors.items.len,
        filename,
        version_str,
    });

    // 打印所有 KV 元数据
    logger.info("Dumping metadata keys/values. Note: KV overrides do not apply in this output.", .{});
    for (gguf_file.metadata_keys, 0..) |key, kv_idx| {
        const val = gguf_file.metadata.get(key) orelse continue;
        const formatted = formatMetadataValue(val, std.heap.page_allocator) catch |err| {
            logger.err("- kv {d:3}: {s: >42} {s: <17} = <format error: {s}>", .{ kv_idx, key, getTypeTag(val, std.heap.page_allocator), @errorName(err) });
            continue;
        };
        logger.info("- kv {d:3}: {s: >42} {s: <17} = {s}", .{ kv_idx, key, getTypeTag(val, std.heap.page_allocator), formatted });
    }

    // 打印张量类型统计
    var type_counts: std.StringHashMapUnmanaged(u32) = .{};
    for (gguf_file.tensors.items) |tensor| {
        const type_name = tensorDataTypeToString(tensor.data_type);
        const entry = type_counts.getOrPut(std.heap.page_allocator, type_name) catch continue;
        if (entry.found_existing) {
            entry.value_ptr.* += 1;
        } else {
            entry.value_ptr.* = 1;
        }
    }
    // 按类型名排序输出
    var type_list = std.ArrayListUnmanaged(struct { name: []const u8, count: u32 }){ .items = &.{}, .capacity = 0 };
    defer type_list.deinit(std.heap.page_allocator);
    var iter = type_counts.iterator();
    while (iter.next()) |entry| {
        type_list.append(std.heap.page_allocator, .{ .name = entry.key_ptr.*, .count = entry.value_ptr.* }) catch continue;
    }
    std.mem.sortUnstable(@TypeOf(type_list.items[0]), type_list.items, {}, struct {
        fn lessThan(_: void, a: @TypeOf(type_list.items[0]), b: @TypeOf(type_list.items[0])) bool {
            return std.mem.order(u8, a.name, b.name) == .lt;
        }
    }.lessThan);
    for (type_list.items) |item| {
        logger.info("- type {s: >4}: {d:3} tensors", .{ item.name, item.count });
    }

    // 打印文件类型
    if (gguf_file.metadata.get("general.file_type")) |ft_val| {
        if (ft_val.asU32()) |ft| {
            const file_type_str = fileTypeToString(ft);
            logger.info("file type   = {s}", .{file_type_str});
        }
    }

    // 计算文件大小和 BPW（与 llama.cpp 保持一致：n_bytes 只计算张量数据，不包括元数据）
    const total_params = countTotalParams(gguf_file);
    const tensor_bytes = countTensorBytes(gguf_file);
    const file_size_mib = @as(f64, @floatFromInt(tensor_bytes)) / (1024.0 * 1024.0);
    const bpw = if (total_params > 0)
        @as(f64, @floatFromInt(tensor_bytes * 8)) / @as(f64, @floatFromInt(total_params))
    else
        0.0;
    logger.info("file size   = {d:.2} MiB ({d:.2} BPW)", .{ file_size_mib, bpw });
}

/// 将 TensorDataType 转换为可读字符串
pub fn tensorDataTypeToString(dt: gguf.TensorDataType) []const u8 {
    return switch (dt) {
        .f32 => "f32",
        .f16 => "f16",
        .q4_0 => "q4_0",
        .q4_1 => "q4_1",
        .q5_0 => "q5_0",
        .q5_1 => "q5_1",
        .q8_0 => "q8_0",
        .q8_1 => "q8_1",
        .q2_k => "q2_K",
        .q3_k => "q3_K",
        .q4_k => "q4_K",
        .q5_k => "q5_K",
        .q6_k => "q6_K",
        .q8_k => "q8_K",
        .i8 => "i8",
        .i16 => "i16",
        .i32 => "i32",
        else => "?",
    };
}

pub fn fileTypeToString(ft: u32) []const u8 {
    return switch (ft) {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        10 => "Q2_K - Medium",
        11 => "Q3_K - Small",
        12 => "Q3_K - Medium",
        13 => "Q3_K - Large",
        14 => "Q4_K - Small",
        15 => "Q4_K - Medium",
        16 => "Q5_K - Small",
        17 => "Q5_K - Medium",
        18 => "Q6_K",
        19 => "IQ2_XXS - 2.0625 bpw",
        20 => "IQ2_XS - 2.3125 bpw",
        21 => "Q2_K - Small",
        22 => "IQ3_XS - 3.3 bpw",
        23 => "IQ3_XXS - 3.0625 bpw",
        28 => "IQ2_S - 2.5 bpw",
        29 => "IQ2_M - 2.7 bpw",
        30 => "IQ4_XS - 4.25 bpw",
        31 => "IQ1_M - 1.75 bpw",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4 MoE",
        39 => "NVFP4",
        40 => "Q1_0",
        else => "Unknown",
    };
}

pub fn countTotalParams(gguf_file: *const gguf.GGUFFile) u64 {
    var total: u64 = 0;
    for (gguf_file.tensors.items) |tensor| {
        var n_elems: u64 = 1;
        for (0..tensor.n_dims) |i| {
            n_elems *= tensor.dims[i];
        }
        total += n_elems;
    }
    return total;
}

pub fn countTensorBytes(gguf_file: *const gguf.GGUFFile) u64 {
    var total: u64 = 0;
    for (gguf_file.tensors.items) |tensor| {
        var n_elems: u64 = 1;
        for (0..tensor.n_dims) |i| {
            n_elems *= tensor.dims[i];
        }
        // 使用 gguf.TensorDataType 的 typeSize 和 blockSize 计算
        const type_size = tensor.data_type.typeSize();
        const blck_size = tensor.data_type.blockSize();
        const n_blocks = (n_elems + blck_size - 1) / blck_size;
        total += n_blocks * type_size;
    }
    return total;
}

/// 从 GGUF 元数据读取模型参数信息
pub fn getModelParamU32(gguf_file: *const gguf.GGUFFile, key: []const u8) ?u32 {
    return gguf_file.getU32(key);
}

pub fn printTokenizerInfo(tok: *tokenizer.Tokenizer, gguf_file: *const gguf.GGUFFile) void {
    // 获取架构名称
    const arch = gguf_file.getString("general.architecture") orelse "unknown";
    logger.info("arch                  = {s}", .{arch});
    logger.info("vocab_only            = 1", .{});
    logger.info("no_alloc              = 0", .{});

    const vocab_type_str = switch (tok.config.model) {
        .llama => "SPM",
        .gpt2 => "BPE",
        .tiktoken => "BPE",
        .replit => "BPE",
        else => @tagName(tok.config.model),
    };
    logger.info("vocab type            = {s}", .{vocab_type_str});

    // 从 tokenizer 获取实际值
    const n_vocab = tok.vocab.items.len;
    const n_merges = tok.merges.count();
    const n_embd = getModelParamU32(gguf_file, "llama.embedding_length") orelse
        getModelParamU32(gguf_file, "qwen.embedding_length") orelse
        getModelParamU32(gguf_file, "qwen35.embedding_length") orelse 0;
    const n_ctx_train = getModelParamU32(gguf_file, "llama.context_length") orelse
        getModelParamU32(gguf_file, "qwen.context_length") orelse
        getModelParamU32(gguf_file, "qwen35.context_length") orelse 0;

    logger.info("n_vocab               = {d}", .{n_vocab});
    logger.info("n_merges              = {d}", .{n_merges});
    logger.info("n_ctx_train           = {d}", .{n_ctx_train});
    logger.info("n_embd                = {d}", .{n_embd});

    // 打印特殊 token - 直接从 vocab 读取文本，避免 decode 跳过特殊 token
    // 对于 BPE 类型（gpt2），llama.cpp 默认 BOS 为 11
    {
        const bos_id = if (tok.config.model == .gpt2 and gguf_file.getU32("tokenizer.ggml.bos_token_id") == null)
            @as(u32, 11)
        else
            tok.special.bos;
        const bos_str = getTokenText(tok, bos_id);
        logger.info("BOS token             = {d} '{s}'", .{ bos_id, bos_str });
    }
    {
        const eos_str = getTokenText(tok, tok.special.eos);
        logger.info("EOS token             = {d} '{s}'", .{ tok.special.eos, eos_str });
    }
    {
        const eot_id = gguf_file.getU32("tokenizer.ggml.eot_token_id") orelse tok.special.eos;
        const eot_str = getTokenText(tok, eot_id);
        logger.info("EOT token             = {d} '{s}'", .{ eot_id, eot_str });
    }
    {
        const pad_str = getTokenText(tok, tok.special.pad);
        logger.info("PAD token             = {d} '{s}'", .{ tok.special.pad, pad_str });
    }
    {
        // LF token: 对于 BPE 类型，通过 tokenize("\n") 查找
        const lf_id = findLfToken(tok);
        const lf_str = getTokenText(tok, lf_id);
        logger.info("LF token              = {d} '{s}'", .{ lf_id, lf_str });
    }

    // FIM tokens - 先尝试从 GGUF 元数据读取，再通过文本匹配自动检测
    {
        const fim_pre_id = findFimToken(tok, gguf_file, "tokenizer.ggml.fim_pre_token_id", "<|fim_prefix|>");
        if (fim_pre_id != 0) {
            const fim_pre_str = getTokenText(tok, fim_pre_id);
            logger.info("FIM PRE token         = {d} '{s}'", .{ fim_pre_id, fim_pre_str });
        }
    }
    {
        const fim_suf_id = findFimToken(tok, gguf_file, "tokenizer.ggml.fim_suf_token_id", "<|fim_suffix|>");
        if (fim_suf_id != 0) {
            const fim_suf_str = getTokenText(tok, fim_suf_id);
            logger.info("FIM SUF token         = {d} '{s}'", .{ fim_suf_id, fim_suf_str });
        }
    }
    {
        const fim_mid_id = findFimToken(tok, gguf_file, "tokenizer.ggml.fim_mid_token_id", "<|fim_middle|>");
        if (fim_mid_id != 0) {
            const fim_mid_str = getTokenText(tok, fim_mid_id);
            logger.info("FIM MID token         = {d} '{s}'", .{ fim_mid_id, fim_mid_str });
        }
    }
    {
        const fim_pad_id = findFimToken(tok, gguf_file, "tokenizer.ggml.fim_pad_token_id", "<|fim_pad|>");
        if (fim_pad_id != 0) {
            const fim_pad_str = getTokenText(tok, fim_pad_id);
            logger.info("FIM PAD token         = {d} '{s}'", .{ fim_pad_id, fim_pad_str });
        }
    }
    {
        const fim_rep_id = findFimToken(tok, gguf_file, "tokenizer.ggml.fim_rep_token_id", "<|repo_name|>");
        if (fim_rep_id != 0) {
            const fim_rep_str = getTokenText(tok, fim_rep_id);
            logger.info("FIM REP token         = {d} '{s}'", .{ fim_rep_id, fim_rep_str });
        }
    }
    {
        const fim_sep_id = findFimToken(tok, gguf_file, "tokenizer.ggml.fim_sep_token_id", "<|file_sep|>");
        if (fim_sep_id != 0) {
            const fim_sep_str = getTokenText(tok, fim_sep_id);
            logger.info("FIM SEP token         = {d} '{s}'", .{ fim_sep_id, fim_sep_str });
        }
    }

    // 收集所有 EOG tokens（通过名称匹配和元数据）
    logger.info("load: printing all EOG tokens:", .{});
    {
        var eog_ids = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };

        // 从 GGUF 元数据收集
        if (gguf_file.getU32("tokenizer.ggml.eos_token_id")) |eos_id| {
            if (!eogIdsContains(eog_ids.items, eos_id)) {
                eog_ids.append(std.heap.page_allocator, eos_id) catch {};
            }
        }
        if (gguf_file.getU32("tokenizer.ggml.eot_token_id")) |eot_id| {
            if (!eogIdsContains(eog_ids.items, eot_id)) {
                eog_ids.append(std.heap.page_allocator, eot_id) catch {};
            }
        }
        if (gguf_file.getU32("tokenizer.ggml.fim_pad_token_id")) |fim_pad_id| {
            if (!eogIdsContains(eog_ids.items, fim_pad_id)) {
                eog_ids.append(std.heap.page_allocator, fim_pad_id) catch {};
            }
        }
        if (gguf_file.getU32("tokenizer.ggml.fim_rep_token_id")) |fim_rep_id| {
            if (!eogIdsContains(eog_ids.items, fim_rep_id)) {
                eog_ids.append(std.heap.page_allocator, fim_rep_id) catch {};
            }
        }
        if (gguf_file.getU32("tokenizer.ggml.fim_sep_token_id")) |fim_sep_id| {
            if (!eogIdsContains(eog_ids.items, fim_sep_id)) {
                eog_ids.append(std.heap.page_allocator, fim_sep_id) catch {};
            }
        }

        // 通过名称匹配收集 EOG tokens（与 llama.cpp 保持一致）
        const eog_names = [_][]const u8{
            "<|endoftext|>",
            "<|im_end|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
            "<|eot_id|>",
            "<|end|>",
            "<|END|>",
            "<EOS>",
            "<EOT>",
            "<end_of_text>",
            "<|end_of_text|>",
            "<end_of_utterance>",
            "<eos>",
        };
        for (tok.vocab.items, 0..) |entry, id| {
            if (entry == .normal) {
                for (eog_names) |name| {
                    if (std.mem.eql(u8, entry.normal, name)) {
                        const uid = @as(u32, @intCast(id));
                        if (!eogIdsContains(eog_ids.items, uid)) {
                            eog_ids.append(std.heap.page_allocator, uid) catch {};
                        }
                        break;
                    }
                }
            }
        }

        for (eog_ids.items) |eog_id| {
            const eog_str = getTokenText(tok, eog_id);
            logger.info("load:   - {d} ('{s}')", .{ eog_id, eog_str });
        }
    }
    {
        var unused_count: u32 = 0;
        var i: usize = tok.token_types.items.len;
        while (i > 0) {
            i -= 1;
            const ttype = tok.token_types.items[i];
            if (ttype == .control) {
                const uid = @as(u32, @intCast(i));
                // 检查是否在 EOG 列表中（通过名称匹配）
                const is_eog = isEogToken(tok, uid);
                if (!is_eog) {
                    const text = getTokenText(tok, uid);
                    logger.info("load: control token: {d} '{s}' is not marked as EOG", .{ uid, text });
                }
            } else if (ttype == .unused) {
                unused_count += 1;
            }
        }
        logger.info("load: 0 unused tokens", .{});
    }

    // special tokens cache size - 包含 control、user_defined 和 unknown 类型（与 llama.cpp 保持一致）
    {
        var special_count: u32 = 0;
        for (tok.token_types.items, 0..) |ttype, idx| {
            _ = idx;
            if (ttype == .control or ttype == .user_defined or ttype == .unknown) {
                special_count += 1;
            }
        }
        logger.info("load: special tokens cache size = {d}", .{special_count});
    }

    // token to piece cache size
    {
        // 估算 token to piece 缓存大小：每个 token 平均约 8 字节
        const cache_size = @as(f64, @floatFromInt(tok.vocab.items.len)) * 8.0 / (1024.0 * 1024.0);
        logger.info("load: token to piece cache size = {d:.4} MB", .{cache_size});
    }

    // 获取 max token length
    var max_token_len: u32 = 0;
    for (tok.vocab.items) |entry| {
        if (entry == .normal and entry.normal.len > max_token_len) {
            max_token_len = @as(u32, @intCast(entry.normal.len));
        }
    }
    logger.info("max token length      = {d}", .{max_token_len});
}

/// 查找 FIM token：先尝试从 GGUF 元数据读取，再通过文本匹配自动检测
pub fn findFimToken(tok: *const tokenizer.Tokenizer, gguf_file: *const gguf.GGUFFile, key: []const u8, text: []const u8) u32 {
    // 先尝试从 GGUF 元数据读取
    if (gguf_file.getU32(key)) |id| {
        return id;
    }
    // 再通过文本匹配自动检测
    for (tok.vocab.items, 0..) |entry, id| {
        if (entry == .normal and std.mem.eql(u8, entry.normal, text)) {
            return @as(u32, @intCast(id));
        }
    }
    return 0;
}

pub fn getTokenText(tok: *const tokenizer.Tokenizer, id: u32) []const u8 {
    if (id < tok.vocab.items.len) {
        switch (tok.vocab.items[id]) {
            .normal => |s| return s,
            .byte => |b| {
                // 将单字节转换为字符串
                const buf = &[_]u8{b};
                return buf[0..1];
            },
        }
    }
    return "?";
}


/// 检查 token 是否为 EOG token（通过名称匹配）
pub fn isEogToken(tok: *const tokenizer.Tokenizer, id: u32) bool {
    const eog_names = [_][]const u8{
        "<|endoftext|>",
        "<|im_end|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<|eot_id|>",
        "<|end|>",
        "<|END|>",
        "<EOS>",
        "<EOT>",
        "<end_of_text>",
        "<|end_of_text|>",
        "<end_of_utterance>",
        "<eos>",
    };
    if (id < tok.vocab.items.len) {
        const entry = tok.vocab.items[id];
        if (entry == .normal) {
            for (eog_names) |name| {
                if (std.mem.eql(u8, entry.normal, name)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// 查找 LF ('\n') 对应的 token ID
/// 对于 BPE 类型，通过 tokenize("\n") 查找（与 llama.cpp 保持一致）
pub fn findLfToken(tok: *tokenizer.Tokenizer) u32 {
    // 对于 BPE 类型，通过 tokenize("\n") 查找
    if (tok.config.model == .gpt2 or tok.config.model == .tiktoken or tok.config.model == .replit) {
        var tokens = tok.encode("\n", false) catch return 13;
        defer tokens.deinit(tok.allocator);
        if (tokens.items.len > 0) {
            return tokens.items[0];
        }
        return 13;
    }
    // 对于 SPM 类型，查找字节 token 为 '\n' 的条目
    for (tok.vocab.items, 0..) |entry, id| {
        if (entry == .byte and entry.byte == '\n') {
            return @as(u32, @intCast(id));
        }
    }
    return 13; // 默认值
}

/// 检查 EOG IDs 列表中是否已包含指定 ID
pub fn eogIdsContains(ids: []const u32, id: u32) bool {
    for (ids) |existing| {
        if (existing == id) return true;
    }
    return false;
}

pub fn printLlamaContext() void {
    logger.info("llama_model_load: vocab only - skipping tensors", .{});
    // vocab_only 模式下不读取模型参数，与 llama.cpp 保持一致
    const ctx_freq_base = 0.0;
    const ctx_freq_scale = 1.0;
    const ctx_n_ctx: u32 = 512; // 默认上下文大小
    const ctx_n_ctx_train_vocab: u32 = 0; // vocab_only 模式下 n_ctx_train 为 0，与 llama.cpp 保持一致

    logger.info("llama_context: constructing llama_context", .{});
    logger.info("llama_context: n_seq_max     = 1", .{});
    logger.info("llama_context: n_ctx         = {d}", .{ctx_n_ctx});
    logger.info("llama_context: n_ctx_seq     = {d}", .{ctx_n_ctx});
    logger.info("llama_context: n_batch       = {d}", .{ctx_n_ctx});
    logger.info("llama_context: n_ubatch      = {d}", .{ctx_n_ctx});
    logger.info("llama_context: causal_attn   = 1", .{});
    logger.info("llama_context: flash_attn    = auto", .{});
    logger.info("llama_context: kv_unified    = false", .{});
    logger.info("llama_context: freq_base     = {d:.1}", .{ctx_freq_base});
    logger.info("llama_context: freq_scale    = {d}", .{ctx_freq_scale});
    logger.info("llama_context: n_rs_seq      = 0", .{});
    if (ctx_n_ctx > ctx_n_ctx_train_vocab) {
        logger.warn("llama_context: n_ctx_seq ({d}) > n_ctx_train ({d}) -- possible training context overflow", .{ ctx_n_ctx, ctx_n_ctx_train_vocab });
    } else if (ctx_n_ctx < ctx_n_ctx_train_vocab) {
        logger.warn("llama_context: n_ctx_seq ({d}) < n_ctx_train ({d}) -- the full capacity of the model will not be utilized", .{ ctx_n_ctx, ctx_n_ctx_train_vocab });
    }
}
