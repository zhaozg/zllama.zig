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

pub fn mulMat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_mul_mat(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

pub fn mul(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_mul(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

pub fn add(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_add(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

pub fn neg(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_neg(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn exp(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_exp(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn cpy(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_cpy(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

// ============================================================================
// 归一化与激活函数
// ============================================================================

pub fn rmsNorm(ctx: *Context, a: *Tensor, eps: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_rms_norm(@ptrCast(ctx), @ptrCast(@alignCast(a)), eps)));
}

pub fn l2Norm(ctx: *Context, a: *Tensor, eps: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_l2_norm(@ptrCast(ctx), @ptrCast(@alignCast(a)), eps)));
}

pub fn ropeExt(ctx: *Context, a: *Tensor, pos: *Tensor, freq_factors: ?*Tensor, n_dims: i32, mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) *Tensor {
    const factors_ptr = if (freq_factors) |ff| @as(?*c.struct_ggml_tensor, @ptrCast(@alignCast(ff))) else null;
    return @as(*Tensor, @ptrCast(c.ggml_rope_ext(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(pos)), factors_ptr, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)));
}

pub fn ropeMulti(ctx: *Context, a: *Tensor, pos: *Tensor, n_dims: i32, sections: *const [4]i32, mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_rope_multi(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(pos)), null, n_dims, @ptrCast(@constCast(sections)), mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)));
}

pub fn scale(ctx: *Context, a: *Tensor, s: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_scale(@ptrCast(ctx), @ptrCast(@alignCast(a)), s)));
}

pub fn softMax(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_soft_max(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn diagMaskInf(ctx: *Context, a: *Tensor, n_past: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_diag_mask_inf(@ptrCast(ctx), @ptrCast(@alignCast(a)), n_past)));
}

pub fn silu(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_silu(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn sigmoid(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_sigmoid(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn softplus(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_softplus(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn gelu(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_gelu(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn tanh(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_tanh(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn relu(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_relu(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

// ============================================================================
// 张量操作
// ============================================================================

pub fn permute(ctx: *Context, a: *Tensor, axis0: i32, axis1: i32, axis2: i32, axis3: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_permute(@ptrCast(ctx), @ptrCast(@alignCast(a)), axis0, axis1, axis2, axis3)));
}

pub fn cont(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_cont(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn cont2d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_cont_2d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1)));
}

pub fn cont4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_cont_4d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1, ne2, ne3)));
}

pub fn reshape2d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_reshape_2d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1)));
}

pub fn reshape3d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_reshape_3d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1, ne2)));
}

pub fn reshape4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_reshape_4d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1, ne2, ne3)));
}

pub fn repeat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_repeat(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

pub fn repeat4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_repeat_4d(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1, ne2, ne3)));
}

pub fn transpose(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_transpose(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn concat(ctx: *Context, a: *Tensor, b: *Tensor, axis: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_concat(@ptrCast(ctx), @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(a))), @as(*c.struct_ggml_tensor, @ptrCast(@alignCast(b))), axis)));
}

pub fn getRows(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_get_rows(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

// ============================================================================
// 卷积与 SSM 操作
// ============================================================================

pub fn conv1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: i32, p0: i32, d0: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_conv_1d(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)), s0, p0, d0)));
}

pub fn ssmConv(ctx: *Context, sx: *Tensor, kernel: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_ssm_conv(@ptrCast(ctx), @ptrCast(@alignCast(sx)), @ptrCast(@alignCast(kernel)))));
}

pub fn ssmScan(ctx: *Context, s: *Tensor, x: *Tensor, dt: *Tensor, A: *Tensor, B: *Tensor, C: *Tensor, ids: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_ssm_scan(@ptrCast(ctx), @ptrCast(@alignCast(s)), @ptrCast(@alignCast(x)), @ptrCast(@alignCast(dt)), @ptrCast(@alignCast(A)), @ptrCast(@alignCast(B)), @ptrCast(@alignCast(C)), @ptrCast(@alignCast(ids)))));
}

pub fn gatedDeltaNet(ctx: *Context, q: *Tensor, k: *Tensor, v: *Tensor, g: *Tensor, beta: *Tensor, state: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_gated_delta_net(@ptrCast(ctx), @ptrCast(@alignCast(q)), @ptrCast(@alignCast(k)), @ptrCast(@alignCast(v)), @ptrCast(@alignCast(g)), @ptrCast(@alignCast(beta)), @ptrCast(@alignCast(state)))));
}

// ============================================================================
// 输出设置
// ============================================================================

pub fn setOutput(tensor: *Tensor) void {
    c.ggml_set_output(@ptrCast(@alignCast(tensor)));
}

const testing = std.testing;

test "ops basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*Tensor)), @sizeOf(*Tensor));
}
