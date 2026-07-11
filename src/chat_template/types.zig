//! 对话模板类型定义
//!
//! 定义 ChatMessage、Media 等核心类型，支持多模态消息。
//!
//! 参考: docs/DIALOG_TEMPLATE.md §4.4

const std = @import("std");

// ============================================================================
// 媒体类型
// ============================================================================

/// 媒体类型 — ChatMessage 附件类型，仅标记占位符类型，不携带二进制数据。
/// 原始图像/音频数据流经 mtmd.Bitmap → mtmd.tokenize 管道。
/// MediaType.none 已移除：媒体缺席由 ChatMessage.media: ?Media 表示。
pub const MediaType = enum {
    image,
    audio,
};

/// 媒体描述符 — 轻量标记，仅表示消息附带的媒体类型。
/// 二进制负载（像素/音频样本）归属于 mtmd.Bitmap 层。
pub const Media = struct {
    type: MediaType,

    pub fn init(t: MediaType) Media {
        return .{ .type = t };
    }
};

// ============================================================================
// 对话消息
// ============================================================================

/// 单条对话消息，支持关联媒体类型标记
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
    media: ?Media = null,

    pub fn init(role: []const u8, content: []const u8) ChatMessage {
        return .{ .role = role, .content = content, .media = null };
    }

    pub fn withMedia(role: []const u8, content: []const u8, media: Media) ChatMessage {
        return .{ .role = role, .content = content, .media = media };
    }

    /// 检查消息是否包含指定类型的媒体
    pub fn hasMediaType(self: *const ChatMessage, media_type: MediaType) bool {
        if (self.media) |m| return m.type == media_type;
        return false;
    }
};

// ============================================================================
// 占位符常量
// ============================================================================

/// 图像占位符 token 字符串
pub const IMAGE_PLACEHOLDER = "<|image|>";
/// 图像占位符备选（部分模型使用）
pub const IMAGE_PLACEHOLDER_ALT = "<image>";
/// 音频占位符 token 字符串
pub const AUDIO_PLACEHOLDER = "<|audio|>";
/// 音频占位符备选（部分模型使用）
pub const AUDIO_PLACEHOLDER_ALT = "<audio>";

/// 占位符信息
pub const PlaceholderInfo = struct {
    /// 占位符在字符串中的起始位置
    start: usize,
    /// 占位符长度（字符数）
    length: usize,
    /// 媒体类型
    media_type: MediaType,
    /// 展开后的 token 数量（由编码器决定）
    token_count: u32,
    /// 占位符在 token 序列中的起始偏移（由 tokenizeWithPlaceholders 填充）
    token_offset: u32 = 0,
};

/// 检查字符串中是否包含媒体占位符
pub fn containsPlaceholder(text: []const u8) bool {
    return std.mem.indexOf(u8, text, IMAGE_PLACEHOLDER) != null or
        std.mem.indexOf(u8, text, IMAGE_PLACEHOLDER_ALT) != null or
        std.mem.indexOf(u8, text, AUDIO_PLACEHOLDER) != null or
        std.mem.indexOf(u8, text, AUDIO_PLACEHOLDER_ALT) != null;
}

/// 占位符展开结果
pub const ExpandedPlaceholders = struct {
    /// 展开后的 token ID 序列（占位符位置用 0 填充）
    tokens: std.ArrayListUnmanaged(u32),
    /// 每个占位符的展开信息
    offsets: []PlaceholderInfo,
    /// 分配器
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ExpandedPlaceholders) void {
        self.tokens.deinit(self.allocator);
        self.allocator.free(self.offsets);
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "ChatMessage init" {
    const msg = ChatMessage.init("user", "Hello");
    try testing.expectEqualStrings("user", msg.role);
    try testing.expectEqualStrings("Hello", msg.content);
    try testing.expect(msg.media == null);
}

test "ChatMessage withMedia" {
    const media = Media.init(.image);
    const msg = ChatMessage.withMedia("user", "Describe this", media);
    try testing.expectEqualStrings("user", msg.role);
    try testing.expect(msg.media != null);
    try testing.expect(msg.hasMediaType(.image));
    try testing.expect(!msg.hasMediaType(.audio));
}

test "ChatMessage hasMediaType" {
    const msg = ChatMessage.init("user", "Hello");
    try testing.expect(!msg.hasMediaType(.image));
    try testing.expect(!msg.hasMediaType(.audio));
}

test "containsPlaceholder" {
    try testing.expect(containsPlaceholder("<|image|>"));
    try testing.expect(containsPlaceholder("<image>"));
    try testing.expect(containsPlaceholder("<|audio|>"));
    try testing.expect(containsPlaceholder("<audio>"));
    try testing.expect(!containsPlaceholder("plain text"));
    try testing.expect(containsPlaceholder("Describe <|image|> this"));
    try testing.expect(!containsPlaceholder(""));
}
