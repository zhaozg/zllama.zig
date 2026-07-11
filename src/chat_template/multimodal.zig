//! 多模态占位符处理
//!
//! 提供占位符识别、展开等功能，支持图像和音频媒体在对话模板中的嵌入。
//!
//! 处理流程（参考 docs/DIALOG_TEMPLATE.md §4.3）：
//!   1. 模板渲染 → 字符串（含 <|image|> / <|audio|> 标记）
//!   2. 占位符扫描 → 位置信息列表
//!   3. 分段 Tokenization → 文本段编码 + 占位符展开
//!   4. 嵌入注入 → forwardWithEmbdOverride
//!
//! 设计原则：
//!   - 只有通过 --image / --audio 显式传入的媒体才会触发占位符替换
//!   - prompt 中的占位符字符串原样保留（输入标记安全）

const std = @import("std");
const types = @import("types");

const log = std.log.scoped(.multimodal);

// ============================================================================
// 占位符扫描
// ============================================================================

/// 扫描字符串中的媒体占位符，返回所有占位符的位置信息
///
/// 返回的 PlaceholderInfo.start 是相对于输入字符串开头的绝对位置，
/// 按占位符在字符串中的出现顺序排列。
///
/// 支持的占位符：
///   - 默认: <|image|> 或 <image>（图像）
///   - 默认: <|audio|> 或 <audio>（音频）
///   - custom_markers: 额外的模型特定标记（如 <|vision_start|> for Qwen）
///
/// @param text 要扫描的字符串
/// @param allocator 分配器
/// @param custom_markers 模型特定的媒体标记（img_beg, img_end, aud_beg, aud_end）
/// @returns 占位符信息列表
pub fn scanPlaceholders(
    text: []const u8,
    allocator: std.mem.Allocator,
    custom_markers: ?types.ScanMarkers,
) ![]types.PlaceholderInfo {
    var result = std.ArrayListUnmanaged(types.PlaceholderInfo){ .items = &.{}, .capacity = 0 };
    errdefer result.deinit(allocator);

    // Build the search list: default markers + custom markers
    const SearchEntry = struct { str: []const u8, media_type: types.MediaType };
    var searches: [8]SearchEntry = undefined;
    var n_searches: usize = 0;
    searches[n_searches] = .{ .str = types.IMAGE_PLACEHOLDER, .media_type = .image };
    n_searches += 1;
    searches[n_searches] = .{ .str = types.IMAGE_PLACEHOLDER_ALT, .media_type = .image };
    n_searches += 1;
    searches[n_searches] = .{ .str = types.AUDIO_PLACEHOLDER, .media_type = .audio };
    n_searches += 1;
    searches[n_searches] = .{ .str = types.AUDIO_PLACEHOLDER_ALT, .media_type = .audio };
    n_searches += 1;
    if (custom_markers) |cm| {
        if (cm.img_beg.len > 0) {
            searches[n_searches] = .{ .str = cm.img_beg, .media_type = .image };
            n_searches += 1;
        }
        if (cm.img_end.len > 0) {
            searches[n_searches] = .{ .str = cm.img_end, .media_type = .image };
            n_searches += 1;
        }
        if (cm.aud_beg.len > 0) {
            searches[n_searches] = .{ .str = cm.aud_beg, .media_type = .audio };
            n_searches += 1;
        }
        if (cm.aud_end.len > 0) {
            searches[n_searches] = .{ .str = cm.aud_end, .media_type = .audio };
            n_searches += 1;
        }
    }

    var pos: usize = 0;
    while (pos < text.len) {
        var best_pos: ?usize = null;
        var best_type: ?types.MediaType = null;
        var best_len: usize = 0;
        for (searches[0..n_searches]) |entry| {
            if (entry.str.len == 0) continue;
            if (std.mem.indexOf(u8, text[pos..], entry.str)) |p| {
                if (best_pos == null or p < best_pos.?) {
                    best_pos = p;
                    best_type = entry.media_type;
                    best_len = entry.str.len;
                }
            }
        }

        if (best_pos) |p| {
            try result.append(allocator, .{
                .start = pos + p,
                .length = best_len,
                .media_type = best_type.?,
                .token_count = 0,
            });
            pos += p + best_len;
        } else {
            break;
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// 分段 Tokenization + 占位符展开
// ============================================================================

/// 分段 Tokenization 结果
pub const TokenizedSegments = struct {
    /// 完整的 token ID 序列（文本 token + 占位符填充 token 交错排列）
    tokens: std.ArrayListUnmanaged(u32),
    /// 每个占位符在 tokens 中的起始位置和 token 数量
    offsets: []types.PlaceholderInfo,
    /// 分配器
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TokenizedSegments) void {
        self.tokens.deinit(self.allocator);
        self.allocator.free(self.offsets);
    }
};

/// 将模板渲染后的字符串分段 tokenize，同时展开占位符。
///
/// 处理流程：
///   1. 扫描字符串中的占位符
///   2. 将字符串按占位符位置分段
///   3. 对每个文本段调用 tokenizer 编码
///   4. 将占位符展开为指定数量的填充 token
///   5. 交错排列文本 token 和占位符 token
///
/// 注意：scanPlaceholders 返回的 start 是绝对位置。
/// 本函数使用 consumed 指针追踪已处理的字符数来正确切片。
///
/// @param allocator 分配器
/// @param formatted 模板渲染后的字符串（可能包含占位符）
/// @param tokenizer_fn 文本段 tokenize 回调：fn([]const u8) ![]u32
/// @param image_token_id 图像占位符对应的 token ID
/// @param audio_token_id 音频占位符对应的 token ID
/// @param image_token_count 每个图像占位符展开后的 token 数量
/// @param audio_token_count 每个音频占位符展开后的 token 数量
/// @returns 交错排列后的完整 token 序列 + 占位符偏移信息
pub fn tokenizeWithPlaceholders(
    allocator: std.mem.Allocator,
    formatted: []const u8,
    ctx: ?*anyopaque,
    tokenizer_fn: *const fn (ctx: ?*anyopaque, text: []const u8, alloc: std.mem.Allocator) anyerror![]u32,
    image_token_id: u32,
    audio_token_id: u32,
    image_token_count: u32,
    audio_token_count: u32,
) !TokenizedSegments {
    var tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
    errdefer tokens.deinit(allocator);

    const placeholders = try scanPlaceholders(formatted, allocator, null);
    errdefer allocator.free(placeholders);

    for (placeholders) |*ph| {
        ph.token_count = switch (ph.media_type) {
            .image => image_token_count,
            .audio => audio_token_count,
        };
    }

    // 分段处理：文本段 tokenize + 占位符展开
    // 使用 consumed 追踪已处理的原始字符串偏移量，
    // 因为 scanPlaceholders 返回的 start 是相对于 formatted 开头的绝对位置。
    var consumed: usize = 0;
    var current_token_offset: u32 = 0;
    for (placeholders) |*ph| {
        // 占位符前的文本段：从 consumed 到 ph.start（绝对位置）
        const text_segment = formatted[consumed..ph.start];
        if (text_segment.len > 0) {
            const text_tokens = try tokenizer_fn(ctx, text_segment, allocator);
            defer allocator.free(text_tokens);
            try tokens.appendSlice(allocator, text_tokens);
            current_token_offset += @as(u32, @intCast(text_tokens.len));
        }

        // Record the token offset for this placeholder (before inserting placeholder tokens)
        ph.token_offset = current_token_offset;

        // 占位符展开为多个填充 token
        const placeholder_token_id = switch (ph.media_type) {
            .image => image_token_id,
            .audio => audio_token_id,
        };
        for (0..ph.token_count) |_| {
            try tokens.append(allocator, placeholder_token_id);
        }
        current_token_offset += ph.token_count;

        consumed = ph.start + ph.length;
    }

    // 最后一个占位符后的文本段
    if (consumed < formatted.len) {
        const text_tokens = try tokenizer_fn(ctx, formatted[consumed..], allocator);
        defer allocator.free(text_tokens);
        try tokens.appendSlice(allocator, text_tokens);
    }

    // —— 占位符扩展调试日志 ——
    log.debug("tokenizeWithPlaceholders: formatted=\"{s}\"", .{formatted});
    log.debug("  total tokens after expansion: {d}", .{tokens.items.len});
    if (placeholders.len == 0) {
        log.debug("  no placeholders found — pure text tokenization", .{});
    }
    for (placeholders, 0..) |ph, i| {
        const media_label = switch (ph.media_type) {
            .image => "image",
            .audio => "audio",
        };
        log.debug("  placeholder[{d}]: type={s} str_pos={d}..{d} token_offset={d} token_count={d} token_id={d}", .{
            i,
            media_label,
            ph.start,
            ph.start + ph.length,
            ph.token_offset,
            ph.token_count,
            if (ph.media_type == .image) image_token_id else if (ph.media_type == .audio) audio_token_id else @as(u32, 0),
        });
        // 打印展开后的 token 序列片段（前 8 个 + 后 4 个，避免刷屏）
        const start_idx = ph.token_offset;
        const end_idx = ph.token_offset + ph.token_count;
        const max_preview: u32 = 8;
        const max_tail: u32 = 4;
        if (ph.token_count <= max_preview + max_tail + 2) {
            log.debug("    expanded tokens[{d}..{d}): {any}", .{ start_idx, end_idx, tokens.items[start_idx..end_idx] });
        } else {
            log.debug("    expanded tokens[{d}..{d}+{d}..{d}): head={any} ... tail={any}", .{
                start_idx,                                          start_idx + max_preview,
                end_idx - max_tail,                                 end_idx,
                tokens.items[start_idx .. start_idx + max_preview], tokens.items[end_idx - max_tail .. end_idx],
            });
        }
    }

    return TokenizedSegments{
        .tokens = tokens,
        .offsets = placeholders,
        .allocator = allocator,
    };
}

/// 展开占位符为 token 序列（旧版，仅生成占位符 token，不包含文本段）
///
/// 注意：此函数仅生成占位符对应的填充 token，不包含文本段的 tokenization。
/// 推荐使用 tokenizeWithPlaceholders 进行完整的处理。
pub fn expandPlaceholders(
    allocator: std.mem.Allocator,
    formatted: []const u8,
    image_token_id: u32,
    audio_token_id: u32,
    image_token_count: u32,
    audio_token_count: u32,
) !types.ExpandedPlaceholders {
    var tokens = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
    errdefer tokens.deinit(allocator);
    const placeholders = try scanPlaceholders(formatted, allocator, null);
    // 为每个占位符填充 token_count
    for (placeholders, 0..) |*ph, i| {
        _ = i;
        ph.token_count = switch (ph.media_type) {
            .image => image_token_count,
            .audio => audio_token_count,
        };
    }

    // 构建 token 序列（仅占位符 token，跳过文本段）
    var consumed: usize = 0;
    for (placeholders) |*ph| {
        // 跳过占位符前的文本段（expandPlaceholders 不处理文本 token）
        // Mark consumed up to this placeholder
        ph.token_offset = @intCast(tokens.items.len);

        // 占位符展开为多个 token
        const token_count = ph.token_count;
        for (0..token_count) |_| {
            const placeholder_token_id = switch (ph.media_type) {
                .image => image_token_id,
                .audio => audio_token_id,
            };
            try tokens.append(allocator, placeholder_token_id);
        }
        consumed = ph.start + ph.length;
    }
    // consumed tracks the end of processed text; unused beyond this point
    return types.ExpandedPlaceholders{
        .tokens = tokens,
        .offsets = placeholders,
        .allocator = allocator,
    };
}

/// 检查字符串中是否包含媒体占位符
pub fn containsPlaceholder(text: []const u8) bool {
    return std.mem.indexOf(u8, text, types.IMAGE_PLACEHOLDER) != null or
        std.mem.indexOf(u8, text, types.IMAGE_PLACEHOLDER_ALT) != null or
        std.mem.indexOf(u8, text, types.AUDIO_PLACEHOLDER) != null or
        std.mem.indexOf(u8, text, types.AUDIO_PLACEHOLDER_ALT) != null;
}

/// 在消息内容前插入媒体占位符（如果消息包含媒体但内容中没有占位符）
///
/// 对于预设模板，采用固定策略：占位符始终放在文本内容之前。
/// 这与 llama.cpp 的行为一致。
///
/// 当提供 custom_markers 时，优先使用模型特定的标记（如 <|vision_start|>），
/// 否则使用默认标记（<|image|> 或 <|audio|>）。
pub fn ensurePlaceholderInContent(content: []const u8, media_type: types.MediaType, allocator: std.mem.Allocator, custom_markers: ?types.ScanMarkers) ![]const u8 {
    if (containsPlaceholderEx(content, custom_markers)) {
        return content;
    }

    const placeholder = switch (media_type) {
        .image => if (custom_markers) |cm| blk: {
            if (cm.img_beg.len > 0) break :blk cm.img_beg;
            break :blk types.IMAGE_PLACEHOLDER;
        } else types.IMAGE_PLACEHOLDER,
        .audio => if (custom_markers) |cm| blk: {
            if (cm.aud_beg.len > 0) break :blk cm.aud_beg;
            break :blk types.AUDIO_PLACEHOLDER;
        } else types.AUDIO_PLACEHOLDER,
    };

    return try std.fmt.allocPrint(allocator, "{s}{s}", .{ placeholder, content });
}

/// 检查字符串中是否包含媒体占位符（含自定义标记）
pub fn containsPlaceholderEx(text: []const u8, custom_markers: ?types.ScanMarkers) bool {
    if (containsPlaceholder(text)) return true;
    if (custom_markers) |cm| {
        if (cm.img_beg.len > 0 and std.mem.indexOf(u8, text, cm.img_beg) != null) return true;
        if (cm.img_end.len > 0 and std.mem.indexOf(u8, text, cm.img_end) != null) return true;
        if (cm.aud_beg.len > 0 and std.mem.indexOf(u8, text, cm.aud_beg) != null) return true;
        if (cm.aud_end.len > 0 and std.mem.indexOf(u8, text, cm.aud_end) != null) return true;
    }
    return false;
}

/// 计算占位符在 token 序列中的偏移量
///
/// @param offsets 占位符信息列表
/// @param placeholder_index 占位符索引
/// @returns 该占位符在 token 序列中的起始位置
pub fn placeholderTokenOffset(offsets: []const types.PlaceholderInfo, placeholder_index: usize) u32 {
    var offset: u32 = 0;
    for (offsets, 0..) |ph, i| {
        if (i == placeholder_index) break;
        offset += ph.token_count;
    }
    return offset;
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "scanPlaceholders: image" {
    const text = "Describe <|image|> this";
    const placeholders = try scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 1), placeholders.len);
    try testing.expectEqual(@as(usize, 10), placeholders[0].start);
    try testing.expectEqual(types.IMAGE_PLACEHOLDER.len, placeholders[0].length);
    try testing.expectEqual(types.MediaType.image, placeholders[0].media_type);
}

test "scanPlaceholders: audio" {
    const text = "Transcribe <|audio|> please";
    const placeholders = try scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 1), placeholders.len);
    try testing.expectEqual(types.MediaType.audio, placeholders[0].media_type);
}

test "scanPlaceholders: multiple" {
    const text = "<|image|>First<|audio|>Second<image>";
    const placeholders = try scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 3), placeholders.len);
    try testing.expectEqual(types.MediaType.image, placeholders[0].media_type);
    try testing.expectEqual(types.MediaType.audio, placeholders[1].media_type);
    try testing.expectEqual(types.MediaType.image, placeholders[2].media_type);
}

test "scanPlaceholders: none" {
    const text = "Just plain text";
    const placeholders = try scanPlaceholders(text, testing.allocator);
    defer testing.allocator.free(placeholders);

    try testing.expectEqual(@as(usize, 0), placeholders.len);
}

test "expandPlaceholders: image" {
    const text = "Describe <|image|> this";
    var expanded = try expandPlaceholders(testing.allocator, text, 258880, 258881, 784, 20);
    defer expanded.deinit();

    try testing.expectEqual(@as(usize, 784), expanded.tokens.items.len);
    try testing.expectEqual(@as(usize, 1), expanded.offsets.len);
    try testing.expectEqual(@as(u32, 258880), expanded.tokens.items[0]);
    try testing.expectEqual(@as(u32, 784), expanded.offsets[0].token_count);
}

test "expandPlaceholders: audio" {
    const text = "Transcribe <|audio|> please";
    var expanded = try expandPlaceholders(testing.allocator, text, 258880, 258881, 784, 20);
    defer expanded.deinit();

    try testing.expectEqual(@as(usize, 20), expanded.tokens.items.len);
    try testing.expectEqual(@as(usize, 1), expanded.offsets.len);
    try testing.expectEqual(@as(u32, 258881), expanded.tokens.items[0]);
}

test "expandPlaceholders: multiple" {
    const text = "<|image|>A<|audio|>B";
    var expanded = try expandPlaceholders(testing.allocator, text, 258880, 258881, 784, 20);
    defer expanded.deinit();

    try testing.expectEqual(@as(usize, 804), expanded.tokens.items.len); // 784 + 20
    try testing.expectEqual(@as(usize, 2), expanded.offsets.len);
}

test "containsPlaceholder" {
    try testing.expect(containsPlaceholder("<|image|>"));
    try testing.expect(containsPlaceholder("<image>"));
    try testing.expect(containsPlaceholder("<|audio|>"));
    try testing.expect(containsPlaceholder("<audio>"));
    try testing.expect(!containsPlaceholder("plain text"));
}

test "ensurePlaceholderInContent: already has" {
    const content = "Describe <|image|> this";
    const result = try ensurePlaceholderInContent(content, .image, testing.allocator, null);
    // 如果已有占位符，返回原内容（指针相同）
    try testing.expect(result.ptr == content.ptr);
}

test "ensurePlaceholderInContent: add image" {
    const content = "Describe this";
    const result = try ensurePlaceholderInContent(content, .image, testing.allocator, null);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("<|image|>Describe this", result);
}

test "ensurePlaceholderInContent: add audio" {
    const content = "Transcribe";
    const result = try ensurePlaceholderInContent(content, .audio, testing.allocator, null);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("<|audio|>Transcribe", result);
}

test "placeholderTokenOffset: single" {
    const offsets = [_]types.PlaceholderInfo{
        .{ .start = 0, .length = 9, .media_type = .image, .token_count = 784 },
    };
    try testing.expectEqual(@as(u32, 0), placeholderTokenOffset(&offsets, 0));
}

test "placeholderTokenOffset: second" {
    const offsets = [_]types.PlaceholderInfo{
        .{ .start = 0, .length = 9, .media_type = .image, .token_count = 784 },
        .{ .start = 10, .length = 9, .media_type = .audio, .token_count = 20 },
    };
    try testing.expectEqual(@as(u32, 784), placeholderTokenOffset(&offsets, 1));
}

test "tokenizeWithPlaceholders: image only" {
    const text = "<|image|>";
    // 模拟 tokenizer：将文本按空格分割为 token
    const tokenizer_fn = struct {
        fn tokenize(_ctx: ?*anyopaque, text_seg: []const u8, alloc: std.mem.Allocator) ![]u32 {
            _ = _ctx;
            _ = text_seg;
            const result = try alloc.alloc(u32, 0);
            return result;
        }
    }.tokenize;

    var result = try tokenizeWithPlaceholders(
        testing.allocator,
        text,
        null,
        &tokenizer_fn,
        258880,
        258881,
        784,
        20,
    );
    defer result.deinit();

    try testing.expectEqual(@as(usize, 784), result.tokens.items.len);
    try testing.expectEqual(@as(usize, 1), result.offsets.len);
    try testing.expectEqual(@as(u32, 258880), result.tokens.items[0]);
}

test "tokenizeWithPlaceholders: text before and after" {
    const text = "Describe <|image|> this";
    // 模拟 tokenizer：每个字符作为一个 token
    const tokenizer_fn = struct {
        fn tokenize(_ctx: ?*anyopaque, text_seg: []const u8, alloc: std.mem.Allocator) ![]u32 {
            _ = _ctx;
            var tokens = try alloc.alloc(u32, text_seg.len);
            for (text_seg, 0..) |c, i| {
                tokens[i] = @intCast(c);
            }
            return tokens;
        }
    }.tokenize;

    var result = try tokenizeWithPlaceholders(
        testing.allocator,
        text,
        null,
        &tokenizer_fn,
        258880,
        258881,
        784,
        20,
    );
    defer result.deinit();

    // "Describe " (9 chars) + 784 placeholder tokens + " this" (5 chars) = 798
    try testing.expectEqual(@as(usize, 9 + 784 + 5), result.tokens.items.len);
    try testing.expectEqual(@as(usize, 1), result.offsets.len);
    // 验证文本段 token 正确
    try testing.expectEqual(@as(u32, 'D'), result.tokens.items[0]);
    try testing.expectEqual(@as(u32, ' '), result.tokens.items[8]);
    // 验证占位符 token
    try testing.expectEqual(@as(u32, 258880), result.tokens.items[9]);
    // 验证占位符后的文本
    try testing.expectEqual(@as(u32, ' '), result.tokens.items[9 + 784]);
    try testing.expectEqual(@as(u32, 't'), result.tokens.items[9 + 784 + 1]);
}

test "tokenizeWithPlaceholders: two placeholders with text" {
    const text = "T1<|image|>T2<|audio|>T3";
    // tokenizer: each char is a token
    const tokenizer_fn = struct {
        fn tokenize(_ctx: ?*anyopaque, text_seg: []const u8, alloc: std.mem.Allocator) ![]u32 {
            _ = _ctx;
            var tokens = try alloc.alloc(u32, text_seg.len);
            for (text_seg, 0..) |c, i| {
                tokens[i] = @intCast(c);
            }
            return tokens;
        }
    }.tokenize;

    var result = try tokenizeWithPlaceholders(
        testing.allocator,
        text,
        null,
        &tokenizer_fn,
        258880,
        258881,
        3,
        2,
    );
    defer result.deinit();

    // Expected: "T1" (2) + IMG×3 + "T2" (2) + AUD×2 + "T3" (2) = 11
    try testing.expectEqual(@as(usize, 2 + 3 + 2 + 2 + 2), result.tokens.items.len);
    try testing.expectEqual(@as(usize, 2), result.offsets.len);

    // Verify offsets: image placeholder
    try testing.expectEqual(@as(u32, 2), result.offsets[0].token_offset);
    try testing.expectEqual(@as(u32, 3), result.offsets[0].token_count);
    // Audio placeholder comes after "T1" + 3 image tokens + "T2"
    try testing.expectEqual(@as(u32, 2 + 3 + 2), result.offsets[1].token_offset);
    try testing.expectEqual(@as(u32, 2), result.offsets[1].token_count);

    // Verify token values
    try testing.expectEqual(@as(u32, 'T'), result.tokens.items[0]);
    try testing.expectEqual(@as(u32, '1'), result.tokens.items[1]);
    try testing.expectEqual(@as(u32, 258880), result.tokens.items[2]); // image placeholder
    try testing.expectEqual(@as(u32, 258880), result.tokens.items[4]); // last image token
    try testing.expectEqual(@as(u32, 'T'), result.tokens.items[5]); // T2 starts
    try testing.expectEqual(@as(u32, 258881), result.tokens.items[7]); // audio placeholder
    try testing.expectEqual(@as(u32, 'T'), result.tokens.items[9]); // T3 starts
}
