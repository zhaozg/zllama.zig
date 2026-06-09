//! Apple Accelerate vDSP FFT 封装
//!
//! 使用 Apple 原生 Accelerate 框架进行高性能 FFT 运算。
//! 通过 vDSP 模块的 SIMD 加速，在 M 系列芯片上实现极高的能效比。
//!
//! 主要 API：
//! - AccelFFT.init() / deinit(): 管理 FFT 配置和窗函数
//! - AccelFFT.powerSpectrum(): 计算实数帧的功率谱

const std = @import("std");
const math = std.math;

const vdsp = @cImport({
    @cInclude("vecLib/vDSP.h");
});

const log = std.log.scoped(.fft);

/// 使用 Accelerate vDSP 的 FFT 引擎
///
/// 一次性分配 FFT 配置、Hann 窗和重用缓冲区，
/// 避免逐帧分配内存，实现零分配 FFT 计算。
pub const AccelFFT = struct {
    allocator: std.mem.Allocator,
    setup: vdsp.FFTSetup,
    log2n: u32,
    n: u32,
    /// Hann 窗函数 [n]
    window: []f32,
    /// 窗函数能量补偿因子
    window_scale: f32,
    /// 分割复数缓冲区 [n/2]，重用避免逐帧分配
    split_buf: []f32,
    /// 帧填充缓冲区 [n]，重用避免逐帧分配
    frame_buf: []f32,

    /// 初始化 FFT 引擎
    ///
    /// @param fft_size FFT 点数，必须为 2 的幂（如 256、512、1024）
    pub fn init(allocator: std.mem.Allocator, fft_size: u32) !AccelFFT {
        // 验证 fft_size 为 2 的幂
        const log2n = math.log2_int(u32, fft_size);
        const n: u32 = @as(u32, 1) << @as(u5, @intCast(log2n));
        if (n != fft_size) return error.FftSizeNotPowerOfTwo;

        // 创建 FFT 配置对象
        const setup = vdsp.vDSP_create_fftsetup(log2n, vdsp.kFFTRadix2);
        if (setup == null) return error.FftSetupFailed;
        errdefer vdsp.vDSP_destroy_fftsetup(setup);
        // 生成 Hann 窗函数
        const window = try allocator.alloc(f32, n);
        errdefer allocator.free(window);
        vdsp.vDSP_hann_window(window.ptr, n, vdsp.vDSP_HANN_DENORM);

        // 计算窗函数能量补偿因子
        // window_scale = 2.0 / sum(window)  (amplitude-preserving for Hann)
        var window_scale: f32 = undefined;
        {
            var s: f32 = 0;
            for (window) |w| s += w;
            window_scale = 2.0 / s;
        }
        const split_buf = try allocator.alloc(f32, n); // n floats for interleaved complex storage
        errdefer allocator.free(split_buf);
        const frame_buf = try allocator.alloc(f32, n);

        log.info("AccelFFT: n={d} log2n={d} window_scale={d:.4}", .{ n, log2n, window_scale });

        return AccelFFT{
            .allocator = allocator,
            .setup = setup,
            .log2n = @intCast(log2n),
            .n = n,
            .window = window,
            .window_scale = window_scale,
            .split_buf = split_buf,
            .frame_buf = frame_buf,
        };
    }

    /// 释放 FFT 引擎所有资源
    pub fn deinit(self: *AccelFFT) void {
        vdsp.vDSP_destroy_fftsetup(self.setup);
        self.allocator.free(self.window);
        self.allocator.free(self.split_buf);
        self.allocator.free(self.frame_buf);
        self.* = undefined;
    }

    /// 计算实数帧的功率谱
    ///
    /// 处理流程：
    /// 1. 零填充 + 加窗 + 缩放
    /// 2. 转换到分割复数格式
    /// 3. 执行前向实数 FFT (vDSP_fft_zrip)
    /// 4. 计算幅度平方 → 功率谱
    ///
    /// @param frame 输入音频帧（长度 ≤ self.n，自动零填充）
    /// @param spectrum 输出功率谱 [n/2 + 1]
    pub fn powerSpectrum(self: *AccelFFT, frame: []const f32, spectrum: []f32) void {
        const n = self.n;
        std.debug.assert(spectrum.len >= n / 2 + 1);

        // 1. 零填充并加窗
        const buf = self.frame_buf;
        const copy_len = @min(frame.len, n);
        @memcpy(buf[0..copy_len], frame[0..copy_len]);
        @memset(buf[copy_len..n], 0);

        // 应用窗函数缩放因子，再乘以窗函数
        vdsp.vDSP_vsmul(buf.ptr, 1, &self.window_scale, buf.ptr, 1, n);
        vdsp.vDSP_vmul(buf.ptr, 1, self.window.ptr, 1, buf.ptr, 1, n);

        // 2. 转换到分割复数格式（in-place）
        //    将 buf 作为交错复数数组转换为 DSPSplitComplex
        var split = vdsp.DSPSplitComplex{
            .realp = self.split_buf.ptr,
            .imagp = self.split_buf.ptr + n / 2,
        };
        // ctoz: 将交错复数 (buf) 转换到分割复数 (split)
        // stride=2 因为 buf 中两两一组为实部和虚部（全为实数时虚部=0）
        vdsp.vDSP_ctoz(@as([*]vdsp.DSPComplex, @ptrCast(@alignCast(buf.ptr))), 2, &split, 1, n / 2);

        // 3. 前向 FFT
        vdsp.vDSP_fft_zrip(self.setup, &split, 1, self.log2n, vdsp.kFFTDirection_Forward);

        // 4. 计算幅度平方（功率谱）
        //    vDSP_zvmags: 直接输出 |real+j*imag|² 到 spectrum
        vdsp.vDSP_zvmags(&split, 1, spectrum.ptr, 1, n / 2);

        // FFT 的 DC 分量和 Nyquist 分量需要特殊处理：
        // - DC 分量：realp[0]^2（imagp[0] 总是 0）
        // - Nyquist 分量：realp[n/2-1]^2 / imagp[0]（存储在 split buffer 特殊位置）
        spectrum[0] = split.realp[0] * split.realp[0];
        spectrum[n / 2] = split.imagp[0] * split.imagp[0];

        // 归一化到每 bin 的平均功率
        const norm = 1.0 / @as(f32, @floatFromInt(n));
        for (spectrum[0 .. n / 2 + 1]) |*s| {
            s.* *= norm;
        }
    }
};

test "AccelFFT: init and deinit" {
    var fft = try AccelFFT.init(std.testing.allocator, 512);
    defer fft.deinit();
    try std.testing.expectEqual(@as(u32, 512), fft.n);
    try std.testing.expectEqual(@as(u32, 9), fft.log2n);
}

test "AccelFFT: power spectrum of DC signal" {
    const n = 512;
    var fft = try AccelFFT.init(std.testing.allocator, n);
    defer fft.deinit();

    var frame: [400]f32 = [_]f32{1.0} ** 400;
    var spectrum: [257]f32 = undefined; // n/2 + 1 = 257

    fft.powerSpectrum(&frame, &spectrum);

    // DC component should dominate
    const dc_power = spectrum[0];
    var total: f32 = 0;
    for (spectrum[1..]) |s| {
        total += s;
    }
    // DC power should be much larger than sum of all other bins
    try std.testing.expect(dc_power > total * 100);
}

test "AccelFFT: power spectrum of sine wave" {
    const n = 512;
    var fft = try AccelFFT.init(std.testing.allocator, n);
    defer fft.deinit();

    var frame: [400]f32 = undefined;
    // 1 kHz sine at 16 kHz sample rate → bin = 1000 * 512 / 16000 = 32
    const freq_hz: f32 = 1000;
    const sample_rate: f32 = 16000;
    for (&frame, 0..) |*s, i| {
        s.* = @sin(2.0 * math.pi * freq_hz * @as(f32, @floatFromInt(i)) / sample_rate);
    }

    var spectrum: [257]f32 = undefined;
    fft.powerSpectrum(&frame, &spectrum);

    // Bin 32 should have the peak
    const peak_bin = 32;
    for (spectrum[1..], 1..) |s, bin| {
        if (bin == peak_bin) continue;
        try std.testing.expect(s < spectrum[peak_bin] * 0.5);
    }
}

test "AccelFFT: rejects non-power-of-two" {
    const result = AccelFFT.init(std.testing.allocator, 500);
    try std.testing.expectEqual(error.FftSizeNotPowerOfTwo, result);
}

test "AccelFFT: multiple frames reuse buffers" {
    const n = 512;
    var fft = try AccelFFT.init(std.testing.allocator, n);
    defer fft.deinit();

    var frame: [400]f32 = undefined;
    var spectrum: [257]f32 = undefined;

    // Process many frames — should not leak or crash
    for (0..1000) |_| {
        fft.powerSpectrum(&frame, &spectrum);
    }
}
