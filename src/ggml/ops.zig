//! 计算图操作函数
//!
//! 提供 ggml 计算图操作的封装，包括矩阵乘法、激活函数、归一化等。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Context = @import("context.zig").Context;
const Tensor = @import("tensor.zig").Tensor;

// ============================================================================
// 矩阵运算
// ============================================================================

/// 矩阵乘法: result = a * b
/// a: [M, K], b: [K, N], result: [M, N]
pub fn mulMat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_mul_mat(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
    )));
}

/// 逐元素乘法
pub fn mul(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_mul(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
    )));
}

/// 逐元素加法
pub fn add(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_add(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
    )));
}

/// 取负
pub fn neg(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_neg(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

/// 指数运算
pub fn exp(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_exp(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

/// 张量拷贝
pub fn cpy(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_cpy(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
    )));
}

// ============================================================================
// 归一化与激活函数
// ============================================================================

/// RMS 归一化
pub fn rmsNorm(ctx: *Context, a: *Tensor, eps: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_rms_norm(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        eps,
    )));
}

/// RoPE 位置编码（扩展版本）
/// 注意：参数顺序与 ggml_rope_ext C API 一致
pub fn ropeExt(
    ctx: *Context,
    a: *Tensor,
    pos: *Tensor,
    mode: i32,
    n_dims: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_rope_ext(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(pos)),
        null, // c: second position tensor (NULL = not used)
        n_dims,
        mode,
        n_ctx_orig,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow,
    )));
}

/// 缩放
pub fn scale(ctx: *Context, a: *Tensor, s: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_scale(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        s,
    )));
}

/// Softmax
pub fn softMax(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_soft_max(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

/// 对角线 mask 无穷
pub fn diagMaskInf(ctx: *Context, a: *Tensor, n_past: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_diag_mask_inf(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        n_past,
    )));
}

/// SiLU 激活函数
pub fn silu(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_silu(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

/// Sigmoid 激活函数
pub fn sigmoid(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_sigmoid(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

/// Softplus 激活函数
pub fn softplus(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_softplus(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

// ============================================================================
// 张量操作
// ============================================================================

/// 置换
pub fn permute(ctx: *Context, a: *Tensor, axis0: i32, axis1: i32, axis2: i32, axis3: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_permute(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        axis0,
        axis1,
        axis2,
        axis3,
    )));
}

/// 连续化
pub fn cont(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_cont(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

/// 重塑为 2D
pub fn reshape2d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_reshape_2d(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        ne0,
        ne1,
    )));
}

/// 重塑为 3D
pub fn reshape3d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_reshape_3d(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        ne0,
        ne1,
        ne2,
    )));
}

/// 重塑为 4D
pub fn reshape4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_reshape_4d(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        ne0,
        ne1,
        ne2,
        ne3,
    )));
}

/// 重复张量以匹配目标形状
pub fn repeat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_repeat(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
    )));
}

/// 转置张量
pub fn transpose(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_transpose(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

/// 沿指定轴拼接两个张量
pub fn concat(ctx: *Context, a: *Tensor, b: *Tensor, axis: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_concat(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
        axis,
    )));
}

/// 行查找（embedding lookup）
/// a: [n_embd, n_rows] — 数据矩阵
/// b: [n_indices] — 行索引（i32 类型）
/// 返回: [n_embd, n_indices]
pub fn getRows(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_get_rows(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
    )));
}

// ============================================================================
// 卷积与 SSM 操作
// ============================================================================

/// 1D 卷积
pub fn conv1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: i32, p0: i32, d0: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_conv_1d(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
        @ptrCast(@alignCast(b)),
        s0,
        p0,
        d0,
    )));
}

/// SSM 卷积（Mamba-2 风格）
/// sx: [d_conv-1+n_t, d_inner, n_s] — 3D 输入（包含历史状态）
/// c:  [d_conv, d_inner] — 2D 卷积核
/// 返回: [d_inner, n_t, n_s] — 3D 输出
pub fn ssmConv(ctx: *Context, sx: *Tensor, kernel: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_ssm_conv(
        @ptrCast(ctx),
        @ptrCast(@alignCast(sx)),
        @ptrCast(@alignCast(kernel)),
    )));
}

/// SSM Scan 操作
pub fn ssmScan(ctx: *Context, sx: *Tensor, B: *Tensor, C: *Tensor, dt: *Tensor, A: *Tensor, state: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_ssm_scan(
        @ptrCast(ctx),
        @ptrCast(@alignCast(sx)),
        @ptrCast(@alignCast(B)),
        @ptrCast(@alignCast(C)),
        @ptrCast(@alignCast(dt)),
        @ptrCast(@alignCast(A)),
        @ptrCast(@alignCast(state)),
    )));
}

// ============================================================================
// 输出设置
// ============================================================================

/// 设置输出张量
pub fn setOutput(tensor: *Tensor) void {
    c.ggml_set_output(@ptrCast(@alignCast(tensor)));
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "ops basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*Tensor)), @sizeOf(*Tensor));
}
