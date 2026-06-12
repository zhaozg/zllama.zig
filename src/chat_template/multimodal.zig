//! 多模态占位符处理
//!
//! 提供占位符识别、展开等功能，支持图像和音频媒体在对话模板中的嵌入。
//!
//! 处理流程（参考 docs/DIALOG_TEMPLATE.md §4.3）：
//!   1. 模板渲染 → 字符串（含 <|image|> / <|audio|> 标记）
//!   2. 占位符识别与展开 → token 序列 + 偏移信息
//!   3. Tokenization → token IDs
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
/// 支持的占位符：
///   - <|image|> 或 <image>（图像）
///   - <|audio|> 或 <audio>（音频）
///
/// @param text 要扫描的字符串
/// @param allocator 分配器
/// @returns 占位符信息列表
pub fn scanPlaceholders(
    text: []const u8,
    allocator: std.mem.Allocator,
) ![]types.PlaceholderInfo {
    var result = std.ArrayListUnmanaged(types.PlaceholderInfo){ .items = &.{}, .capacity = 0 };
    errdefer result.deinit(allocator);

    var pos: usize = 0;
    while (pos < text.len) {
        // 查找下一个占位符
        const next_image = std.mem.indexOf(u8, text[pos..], types.IMAGE_PLACEHOLDER);
        const next_image_alt = std.mem.indexOf(u8, text[pos..], types.IMAGE_PLACEHOLDER_ALT);
        const next_audio = std.mem.indexOf(u8, text[pos..], types.AUDIO_PLACEHOLDER);
        const next_audio_alt = std.mem.indexOf(u8, text[pos..], types.AUDIO_PLACEHOLDER_ALT);

        // 找到最近的下一个占位符
        var best_pos: ?usize = null;
        var best_type: types.MediaType = .none;
        var best_len: usize = 0;

        if (next_image) |p| {
            best_pos = p;
            best_type = .image;
            best_len = types.IMAGE_PLACEHOLDER.len;
        }
        if (next_image_alt) |p| {
            if (best_pos == null or p < best_pos.?) {
                best_pos = p;
                best_type = .image;
                best_len = types.IMAGE_PLACEHOLDER_ALT.len;
            }
        }
        if (next_audio) |p| {
            if (best_pos == null or p < best_pos.?) {
                best_pos = p;
                best_type = .audio;
                best_len = types.AUDIO_PLACEHOLDER.len;
            }
        }
        if (next_audio_alt) |p| {
            if (best_pos == null or p < best_pos.?) {
                best_pos = p;
                best_type = .audio;
                best_len = types.AUDIO_PLACEHOLDER_ALT.len;
            }
        }

        if (best_pos) |p| {
            try result.append(allocator, .{
                .start = pos + p,
                .length = best_len,
                .media_type = best_type,
                .token_count = 0, // 由调用者填充
            });
            pos += p + best_len;
        } else {
            break;
        }
    }

    return result.toOwnedSlice(allocator);
}

/// 展开占位符为 token 序列
/// 将字符串中的占位符替换为指定数量的填充 token（0），
/// 并记录每个占位符的偏移信息供嵌入注入使用。
///
/// @param allocator 分配器
/// @param formatted 模板渲染后的字符串（可能包含占位符）
/// @param image_token_id 图像占位符对应的 token ID
/// @param audio_token_id 音频占位符对应的 token ID
/// @param image_token_count 每个图像占位符展开后的 token 数量
/// @param audio_token_count 每个音频占位符展开后的 token 数量
/// @returns 展开后的 token 序列和偏移信息
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
    const placeholders = try scanPlaceholders(formatted, allocator);
    // 为每个占位符填充 token_count
    for (placeholders, 0..) |*ph, i| {
        _ = i;
        ph.token_count = switch (ph.media_type) {
            .image => image_token_count,
            .audio => audio_token_count,
            .none => 0,
        };
    }

    // 构建 token 序列
    var remaining = formatted;
    var is_first_segment = true;

    for (placeholders) |ph| {
        // 占位符前的文本段（当前实现中跳过 tokenization，由调用者处理）
        _ = remaining[0..ph.start];

        // 占位符展开为多个 token
        const token_count = ph.token_count;
        for (0..token_count) |_| {
            const placeholder_token_id = switch (ph.media_type) {
                .image => image_token_id,
                .audio => audio_token_id,
                .none => 0,
            };
            try tokens.append(allocator, placeholder_token_id);
        }

        remaining = remaining[ph.start + ph.length ..];
        is_first_segment = false;
    }

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
pub fn ensurePlaceholderInContent(content: []const u8, media_type: types.MediaType, allocator: std.mem.Allocator) ![]const u8 {
    if (containsPlaceholder(content)) {
        return content;
    }

    const placeholder = switch (media_type) {
        .image => types.IMAGE_PLACEHOLDER,
        .audio => types.AUDIO_PLACEHOLDER,
        .none => return content,
    };

    return try std.fmt.allocPrint(allocator, "{s}{s}", .{ placeholder, content });
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
    const result = try ensurePlaceholderInContent(content, .image, testing.allocator);
    // 如果已有占位符，返回原内容（指针相同）
    try testing.expect(result.ptr == content.ptr);
}

test "ensurePlaceholderInContent: add image" {
    const content = "Describe this";
    const result = try ensurePlaceholderInContent(content, .image, testing.allocator);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("<|image|>Describe this", result);
}

test "ensurePlaceholderInContent: add audio" {
    const content = "Transcribe";
    const result = try ensurePlaceholderInContent(content, .audio, testing.allocator);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("<|audio|>Transcribe", result);
}
