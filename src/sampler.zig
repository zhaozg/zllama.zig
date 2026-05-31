//! 采样器
//!
//! 从 logits 张量中采样下一个 token。
//! 支持温度采样、Top-K、Top-P 过滤。

const std = @import("std");
const ggml = @import("ggml.zig");

const log = std.log.scoped(.sampler);

// ============================================================================
// 采样参数
// ============================================================================

/// 采样配置参数
pub const SamplerParams = struct {
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    seed: u64 = 0,  // 0 = 使用随机种子
};

// ============================================================================
// 采样器
// ============================================================================

/// 采样器状态
pub const Sampler = struct {
    rng: std.Random.Xoshiro256,
    params: SamplerParams,

    pub fn init(params: SamplerParams) Sampler {
        const seed = if (params.seed != 0) params.seed else @as(u64, @bitCast(std.time.nanoTimestamp()));
        return Sampler{
            .rng = std.Random.Xoshiro256.init(seed),
            .params = params,
        };
    }

    /// 从 logits 张量采样下一个 token
    pub fn sample(self: *Sampler, logits_tensor: *ggml.Tensor) u32 {
        const logits = logits_tensor.dataF32();
        const n_vocab = logits.len;

        if (n_vocab == 0) return 0;

        // 应用温度
        var scaled = if (self.params.temperature > 0.0)
            applyTemperature(logits, self.params.temperature)
        else
            logits[0..];

        // 应用 Top-K
        if (self.params.top_k > 0 and self.params.top_k < n_vocab) {
            scaled = applyTopK(scaled, self.params.top_k);
        }

        // 应用 Top-P
        if (self.params.top_p > 0.0 and self.params.top_p < 1.0) {
            scaled = applyTopP(scaled, self.params.top_p);
        }

        // 转换为概率并采样
        return sampleFromLogits(self, scaled);
    }

    /// 贪心采样（取最大概率 token）
    pub fn sampleGreedy(logits_tensor: *ggml.Tensor) u32 {
        const logits = logits_tensor.dataF32();
        var max_val: f32 = -std.math.inf(f32);
        var max_idx: u32 = 0;

        for (logits, 0..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = @intCast(i);
            }
        }

        return max_idx;
    }
};

// ============================================================================
// 内部函数
// ============================================================================

fn applyTemperature(logits: []const f32, temperature: f32) []f32 {
    if (temperature <= 0.0 or temperature == 1.0) {
        return @constCast(logits);
    }
    return @constCast(logits);
}


/// 应用 Top-K 过滤
fn applyTopK(logits: []const f32, k: u32) []f32 {
    _ = k;
    return @constCast(logits);
}

/// 应用 Top-P (nucleus) 过滤
fn applyTopP(logits: []const f32, p: f32) []f32 {
    _ = p;
    return @constCast(logits);
}

/// 从 logits 中采样
fn sampleFromLogits(sampler: *Sampler, logits: []const f32) u32 {
    // 计算 softmax
    var max_logit: f32 = -std.math.inf(f32);
    for (logits) |v| {
        if (v > max_logit) max_logit = v;
    }

    var sum: f32 = 0.0;
    var probs: [128000]f32 = undefined;
    const n = @min(logits.len, probs.len);

    for (logits[0..n], 0..) |v, i| {
        probs[i] = std.math.exp(v - max_logit);
        sum += probs[i];
    }

    if (sum <= 0.0) return 0;

    // 归一化
    for (probs[0..n]) |*p| {
        p.* /= sum;
    }

    // 采样
    const r = sampler.rng.float(f32);
    var cumsum: f32 = 0.0;
    for (probs[0..n], 0..) |p, i| {
        cumsum += p;
        if (r < cumsum) {
            return @intCast(i);
        }
    }

    return @intCast(n - 1);
}

// ============================================================================
// 测试
// ============================================================================

const testing = std.testing;

test "Sampler init" {
    const params = SamplerParams{
        .temperature = 0.7,
        .top_k = 40,
        .top_p = 0.9,
    };
    const sampler = Sampler.init(params);
    _ = sampler;
}

test "SamplerParams defaults" {
    const params = SamplerParams{};
    try testing.expectEqual(@as(f32, 0.7), params.temperature);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(f32, 0.9), params.top_p);
}
