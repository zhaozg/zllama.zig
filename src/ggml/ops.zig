//! 计算图操作函数
//!
//! 提供 ggml 计算图操作的封装，包括矩阵乘法、激活函数、归一化等。

const std = @import("std");
const cmod = @import("c.zig");
const c = cmod.c;
const Type = cmod.Type;
const Context = @import("context.zig").Context;
const Tensor = @import("tensor.zig").Tensor;

/// 反量化一行数据：将量化数据的一行（ne0 个元素）反量化为 f32。
/// 使用 ggml 的类型特征表（type_traits）中的 to_float 回调。
/// 如果该类型不支持反量化，返回 error.DequantizeNotSupported。
pub fn dequantizeRow(typ: Type, quant_data: *const anyopaque, f32_data: []f32, ne0: i64) !void {
    const traits = c.ggml_get_type_traits(@intFromEnum(typ));
    const to_float = traits.*.to_float orelse return error.DequantizeNotSupported;
    to_float(quant_data, f32_data.ptr, ne0);
}

/// 反量化整个张量：将量化数据按行反量化为 f32 目标缓冲区。
/// f32_dst 的长度必须 >= n_rows * ne0。
pub fn dequantizeTensor(typ: Type, quant_src: []const u8, f32_dst: []f32, ne0: i64, n_rows: i64) !void {
    const traits = c.ggml_get_type_traits(@intFromEnum(typ));
    const to_float = traits.*.to_float orelse return error.DequantizeNotSupported;
    var offset: usize = 0;
    var row: i64 = 0;
    while (row < n_rows) : (row += 1) {
        const row_quant = quant_src[offset..];
        to_float(row_quant.ptr, &f32_dst[@as(usize, @intCast(row * ne0))], ne0);
        offset += Type.rowSize(typ, ne0);
    }
}

// ============================================================================
// 矩阵运算
// ============================================================================

pub fn mulMat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_mul_mat(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

pub fn mulMatSetPrec(a: *Tensor, prec: cmod.Prec) void {
    c.ggml_mul_mat_set_prec(@ptrCast(@alignCast(a)), @intFromEnum(prec));
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

pub fn cast(ctx: *Context, a: *Tensor, typ: Type) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_cast(@ptrCast(ctx), @ptrCast(@alignCast(a)), @intFromEnum(typ))));
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

/// Fused softmax with optional mask, scaling, and ALiBi bias
/// mask can be null. scaling applied before softmax. max_bias=0.0 for no ALiBi.
pub fn softMaxExt(ctx: *Context, a: *Tensor, mask: ?*Tensor, scaling: f32, max_bias: f32) *Tensor {
    const mask_ptr = if (mask) |m| @as(?*c.struct_ggml_tensor, @ptrCast(@alignCast(m))) else null;
    return @as(*Tensor, @ptrCast(c.ggml_soft_max_ext(@ptrCast(ctx), @ptrCast(@alignCast(a)), mask_ptr, scaling, max_bias)));
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

pub fn geluErf(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_gelu_erf(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn geluQuick(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_gelu_quick(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn sqr(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_sqr(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn tanh(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_tanh(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn relu(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_relu(@ptrCast(ctx), @ptrCast(@alignCast(a)))));
}

pub fn clamp(ctx: *Context, a: *Tensor, min: f32, max: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_clamp(@ptrCast(ctx), @ptrCast(@alignCast(a)), min, max)));
}

// ============================================================================
// GLU 变体操作 (split variants: gate * activation(up))
// ============================================================================

/// swiglu_split(a, b) = silu(a) * b
pub fn swigluSplit(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_swiglu_split(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

/// geglu_split(a, b) = gelu(a) * b
pub fn gegluSplit(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_geglu_split(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

/// geglu_erf_split(a, b) = gelu_erf(a) * b
pub fn gegluErfSplit(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_geglu_erf_split(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
}

/// geglu_quick_split(a, b) = gelu_quick(a) * b
pub fn gegluQuickSplit(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_geglu_quick_split(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)))));
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

/// 创建一个与 src 具有相同形状和数据类型的张量（不复制数据）。
/// 等价于 ggml_dup_tensor。
pub fn dupTensor(ctx: *Context, src: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_dup_tensor(@ptrCast(ctx), @ptrCast(@alignCast(src)))));
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

pub fn gatedDeltaNet(ctx: *Context, q: *Tensor, k: *Tensor, v: *Tensor, g: *Tensor, beta: *Tensor, state: *Tensor, K: i64) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_gated_delta_net(@ptrCast(ctx), @ptrCast(@alignCast(q)), @ptrCast(@alignCast(k)), @ptrCast(@alignCast(v)), @ptrCast(@alignCast(g)), @ptrCast(@alignCast(beta)), @ptrCast(@alignCast(state)), K)));
}

// ============================================================================
// Flash Attention
// ============================================================================

/// Flash attention with optional mask, scaling, ALiBi, and logit softcap.
/// Q, K, V should be permuted to [d_head, n_patches, n_head, n_batch].
/// Result shape: [d_head, n_patches, n_head, n_batch].
pub fn flashAttnExt(ctx: *Context, q: *Tensor, k: *Tensor, v: *Tensor, mask: ?*Tensor, kq_scale: f32, max_bias: f32, logit_softcap: f32) *Tensor {
    const mask_ptr = if (mask) |m| @as(?*c.struct_ggml_tensor, @ptrCast(@alignCast(m))) else null;
    return @as(*Tensor, @ptrCast(c.ggml_flash_attn_ext(
        @ptrCast(ctx),
        @ptrCast(@alignCast(q)),
        @ptrCast(@alignCast(k)),
        @ptrCast(@alignCast(v)),
        mask_ptr,
        kq_scale,
        max_bias,
        logit_softcap,
    )));
}

pub fn flashAttnExtSetPrec(a: *Tensor, prec: cmod.Prec) void {
    c.ggml_flash_attn_ext_set_prec(@ptrCast(@alignCast(a)), @intFromEnum(prec));
}

// ============================================================================
// 输出设置
// ============================================================================

pub fn setOutput(tensor: *Tensor) void {
    c.ggml_set_output(@ptrCast(@alignCast(tensor)));
}

// ============================================================================
// 归约操作
// ============================================================================

/// 沿第 1 维求和：输入 [ne0, ne1, ne2, ne3] → 输出 [ne0, 1, ne2, ne3]
pub fn sumRows(ctx: *Context, a: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_sum_rows(
        @ptrCast(ctx),
        @ptrCast(@alignCast(a)),
    )));
}

// ============================================================================
// 张量生成与填充
// ============================================================================

/// 创建等差数列张量：start, start+step, ..., stop (exclusive)
pub fn arange(ctx: *Context, start: f32, stop: f32, step: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_arange(@ptrCast(ctx), start, stop, step)));
}

/// 用常量值填充张量
pub fn fill(ctx: *Context, a: *Tensor, val: f32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_fill(@ptrCast(ctx), @ptrCast(@alignCast(a)), val)));
}

// ============================================================================
// 插值与缩放
// ============================================================================

/// 插值缩放张量到指定尺寸
pub fn interpolate(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, mode: u32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_interpolate(@ptrCast(ctx), @ptrCast(@alignCast(a)), ne0, ne1, ne2, ne3, mode)));
}

// ============================================================================
// 填充操作
// ============================================================================

/// 反射填充 1D（沿第 0 维左右填充）
pub fn padReflect1d(ctx: *Context, a: *Tensor, p0: i32, p1: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_pad_reflect_1d(@ptrCast(ctx), @ptrCast(@alignCast(a)), p0, p1)));
}

// ============================================================================
// 滚动操作
// ============================================================================

/// 沿各维度滚动张量元素（超出边界的元素循环到开头）
pub fn roll(ctx: *Context, a: *Tensor, shift0: i32, shift1: i32, shift2: i32, shift3: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_roll(@ptrCast(ctx), @ptrCast(@alignCast(a)), shift0, shift1, shift2, shift3)));
}

// ============================================================================
// 时间步嵌入
// ============================================================================

/// 时间步嵌入（用于扩散模型）
pub fn timestepEmbedding(ctx: *Context, timesteps: *Tensor, dim: i32, max_period: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_timestep_embedding(@ptrCast(ctx), @ptrCast(@alignCast(timesteps)), dim, max_period)));
}

// ============================================================================
// 池化操作
// ============================================================================

/// 1D 池化
pub fn pool1d(ctx: *Context, a: *Tensor, op: c_uint, k0: i32, s0: i32, p0: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_pool_1d(@ptrCast(ctx), @ptrCast(@alignCast(a)), @intCast(op), k0, s0, p0)));
}

// ============================================================================
// 相对位置编码
// ============================================================================

/// 获取相对位置编码
pub fn getRelPos(ctx: *Context, a: *Tensor, qh: i32, kh: i32) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_get_rel_pos(@ptrCast(ctx), @ptrCast(@alignCast(a)), qh, kh)));
}

/// 添加相对位置编码
pub fn addRelPos(ctx: *Context, a: *Tensor, pw: *Tensor, ph: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_add_rel_pos(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(pw)), @ptrCast(@alignCast(ph)))));
}

/// 添加相对位置编码（原地）
pub fn addRelPosInplace(ctx: *Context, a: *Tensor, pw: *Tensor, ph: *Tensor) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_add_rel_pos_inplace(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(pw)), @ptrCast(@alignCast(ph)))));
}

// ============================================================================
// 自定义算子
// ============================================================================

/// 自定义一元算子
pub fn mapCustom1(ctx: *Context, a: *Tensor, fun: c.ggml_custom1_op_t, n_tasks: i32, userdata: ?*anyopaque) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_map_custom1(@ptrCast(ctx), @ptrCast(@alignCast(a)), fun, n_tasks, if (userdata) |ud| @ptrCast(ud) else null)));
}

/// 自定义二元算子
pub fn mapCustom2(ctx: *Context, a: *Tensor, b: *Tensor, fun: c.ggml_custom2_op_t, n_tasks: i32, userdata: ?*anyopaque) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_map_custom2(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)), fun, n_tasks, if (userdata) |ud| @ptrCast(ud) else null)));
}

/// 自定义三元算子
pub fn mapCustom3(ctx: *Context, a: *Tensor, b: *Tensor, c2: *Tensor, fun: c.ggml_custom3_op_t, n_tasks: i32, userdata: ?*anyopaque) *Tensor {
    return @as(*Tensor, @ptrCast(c.ggml_map_custom3(@ptrCast(ctx), @ptrCast(@alignCast(a)), @ptrCast(@alignCast(b)), @ptrCast(@alignCast(c2)), fun, n_tasks, if (userdata) |ud| @ptrCast(ud) else null)));
}

/// 自定义 4D 算子（多输入）
pub fn custom4d(ctx: *Context, typ: Type, ne0: i64, ne1: i64, ne2: i64, ne3: i64, args: []const *Tensor, fun: c.ggml_custom_op_t, n_tasks: i32, userdata: ?*anyopaque) *Tensor {
    var c_args: [16][*c]c.struct_ggml_tensor = undefined;
    for (args, 0..) |arg, i| {
        c_args[i] = @ptrCast(@alignCast(arg));
    }
    return @as(*Tensor, @ptrCast(c.ggml_custom_4d(
        @ptrCast(ctx),
        @intFromEnum(typ),
        ne0,
        ne1,
        ne2,
        ne3,
        @ptrCast(&c_args),
        @intCast(args.len),
        fun,
        n_tasks,
        if (userdata) |ud| @ptrCast(ud) else null,
    )));
}

const testing = std.testing;

test "ops basic" {
    try testing.expectEqual(@as(usize, @sizeOf(*Tensor)), @sizeOf(*Tensor));
}
