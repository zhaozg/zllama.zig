//! 模型实现入口
//! 重新导出 `src/models/` 下所有模型实现，统一导入风格。
//!
//! 注意：多数使用者应通过 `@import("model")` 导入，它已重新导出以下模块：
//!   `model.qwen2`, `model.qwen35`, `model.llama`, `model.gemma3`,
//!   `model.gemma4`, `model.embedding`, `model.qwen3vl`,
//!   `model.gemma4_graph`

pub const registry = @import("registry");
pub const qwen2 = @import("qwen2.zig");
pub const qwen35 = @import("qwen35.zig");
pub const qwen3vl = @import("qwen3vl.zig");
pub const llama = @import("llama.zig");
pub const gemma3 = @import("gemma3.zig");
pub const gemma4 = @import("gemma4.zig");
pub const gemma4_graph = @import("gemma4_graph.zig");
pub const gemma4_loader = @import("gemma4_loader.zig");
pub const qwen35_loader = @import("qwen35_loader.zig");
pub const embedding = @import("embedding.zig");
