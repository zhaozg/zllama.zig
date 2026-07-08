//! ggml C API 原始绑定
//!
//! 提供 ggml C API 的原始 @cImport 和类型枚举。
//! 所有其他模块通过此模块间接访问 C API，避免业务代码直接 @cImport。

const std = @import("std");

// ============================================================================
// 原始 C API
// ============================================================================

pub const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-cpu.h");
    @cInclude("ggml-backend.h");
    @cInclude("ggml-alloc.h");
    @cInclude("gguf.h");
});

// ============================================================================
// ggml 数据类型枚举
// ============================================================================

/// ggml 数据类型枚举
pub const Type = enum(c.ggml_type) {
    f32 = c.GGML_TYPE_F32,
    f16 = c.GGML_TYPE_F16,
    q4_0 = c.GGML_TYPE_Q4_0,
    q4_1 = c.GGML_TYPE_Q4_1,
    q5_0 = c.GGML_TYPE_Q5_0,
    q5_1 = c.GGML_TYPE_Q5_1,
    q8_0 = c.GGML_TYPE_Q8_0,
    q8_1 = c.GGML_TYPE_Q8_1,
    q2_K = c.GGML_TYPE_Q2_K,
    q3_K = c.GGML_TYPE_Q3_K,
    q4_K = c.GGML_TYPE_Q4_K,
    q5_K = c.GGML_TYPE_Q5_K,
    q6_K = c.GGML_TYPE_Q6_K,
    q8_K = c.GGML_TYPE_Q8_K,
    iq2_xxs = c.GGML_TYPE_IQ2_XXS,
    iq2_xs = c.GGML_TYPE_IQ2_XS,
    iq3_xxs = c.GGML_TYPE_IQ3_XXS,
    iq1_s = c.GGML_TYPE_IQ1_S,
    iq4_nl = c.GGML_TYPE_IQ4_NL,
    iq3_s = c.GGML_TYPE_IQ3_S,
    iq2_s = c.GGML_TYPE_IQ2_S,
    iq4_xs = c.GGML_TYPE_IQ4_XS,
    i8 = c.GGML_TYPE_I8,
    i16 = c.GGML_TYPE_I16,
    i32 = c.GGML_TYPE_I32,
    i64 = c.GGML_TYPE_I64,
    f64 = c.GGML_TYPE_F64,
    iq1_m = c.GGML_TYPE_IQ1_M,
    bf16 = c.GGML_TYPE_BF16,

    pub fn sizeOf(t: Type) usize {
        return c.ggml_type_size(@intFromEnum(t));
    }

    pub fn blockSize(t: Type) i64 {
        return c.ggml_blck_size(@intFromEnum(t));
    }

    pub fn rowSize(t: Type, ne: i64) usize {
        return c.ggml_row_size(@intFromEnum(t), ne);
    }

    pub fn isQuantized(t: Type) bool {
        return c.ggml_is_quantized(@intFromEnum(t));
    }

    pub fn name(t: Type) [:0]const u8 {
        return std.mem.sliceTo(c.ggml_type_name(@intFromEnum(t)), 0);
    }
};

// ============================================================================
// GGUF 值类型枚举
// ============================================================================

/// GGUF 值类型枚举
pub const GgufValueType = enum(c.gguf_type) {
    uint8 = c.GGUF_TYPE_UINT8,
    int8 = c.GGUF_TYPE_INT8,
    uint16 = c.GGUF_TYPE_UINT16,
    int16 = c.GGUF_TYPE_INT16,
    uint32 = c.GGUF_TYPE_UINT32,
    int32 = c.GGUF_TYPE_INT32,
    float32 = c.GGUF_TYPE_FLOAT32,
    bool = c.GGUF_TYPE_BOOL,
    string = c.GGUF_TYPE_STRING,
    array = c.GGUF_TYPE_ARRAY,
    uint64 = c.GGUF_TYPE_UINT64,
    int64 = c.GGUF_TYPE_INT64,
    float64 = c.GGUF_TYPE_FLOAT64,

    pub fn name(t: GgufValueType) [:0]const u8 {
        return std.mem.sliceTo(c.gguf_type_name(@intFromEnum(t)), 0);
    }
};

// ============================================================================
// ggml 精度枚举
// ============================================================================

/// ggml 精度枚举
pub const Prec = enum(c_uint) {
    default = c.GGML_PREC_DEFAULT,
    f32 = c.GGML_PREC_F32,
};

// ============================================================================
// 池化操作类型枚举
// ============================================================================

/// ggml 池化操作类型
pub const PoolOp = enum(c_uint) {
    max = c.GGML_OP_POOL_MAX,
    avg = c.GGML_OP_POOL_AVG,
    count = c.GGML_OP_POOL_COUNT,
};

// ============================================================================
// 缩放模式枚举
// ============================================================================

/// ggml 缩放模式
pub const ScaleMode = enum(c_uint) {
    nearest = c.GGML_SCALE_MODE_NEAREST,
    bilinear = c.GGML_SCALE_MODE_BILINEAR,
    bicubic = c.GGML_SCALE_MODE_BICUBIC,
    count = c.GGML_SCALE_MODE_COUNT,
};

/// ggml 缩放标志
pub const ScaleFlag = enum(c_uint) {
    align_corners = c.GGML_SCALE_FLAG_ALIGN_CORNERS,
    antialias = c.GGML_SCALE_FLAG_ANTIALIAS,
};

/// 最大任务数（-1 表示使用最大可用任务数）
pub const n_tasks_max: i32 = -1;

// ============================================================================
// GGUF 值联合体
// ============================================================================

/// GGUF 值联合体
pub const GgufValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: [:0]const u8,
    array: struct { typ: GgufValueType, items: []const u8 },
    uint64: u64,
    int64: i64,
    float64: f64,

    pub fn asString(self: GgufValue) ![]const u8 {
        return switch (self) {
            .string => |s| s,
            else => error.TypeMismatch,
        };
    }

    pub fn asInt(self: GgufValue) !i64 {
        return switch (self) {
            .int32 => |v| @as(i64, v),
            .int64 => |v| v,
            .uint32 => |v| @as(i64, v),
            .uint64 => |v| @as(i64, @intCast(v)),
            else => error.TypeMismatch,
        };
    }

    pub fn asFloat(self: GgufValue) !f64 {
        return switch (self) {
            .float32 => |v| @as(f64, v),
            .float64 => |v| v,
            else => error.TypeMismatch,
        };
    }

    pub fn asBool(self: GgufValue) !bool {
        return switch (self) {
            .bool => |v| v,
            else => error.TypeMismatch,
        };
    }
};

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "Type enum values" {
    try testing.expectEqual(Type.f32, @as(Type, @enumFromInt(c.GGML_TYPE_F32)));
    try testing.expect(Type.isQuantized(.q4_K));
    try testing.expect(!Type.isQuantized(.f32));
}

test "Type name" {
    try testing.expectEqualStrings("f32", Type.name(.f32));
    try testing.expectEqualStrings("q4_K", Type.name(.q4_K));
}

test "GgufValueType name" {
    try testing.expectEqualStrings("uint32", GgufValueType.name(.uint32));
    try testing.expectEqualStrings("string", GgufValueType.name(.string));
}

test "GgufValue conversions" {
    const v = GgufValue{ .int32 = 42 };
    try testing.expectEqual(@as(i64, 42), try v.asInt());
    try testing.expectError(error.TypeMismatch, v.asString());
}
