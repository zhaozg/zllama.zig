//! 采样器
//!
//! 实现 Top-p / Top-k 采样策略。

const std = @import("std");
const ggml = @import("ggml.zig");

const log = std.log.scoped(.sampler);

// ============================================================================
// 采样参数
// ============================================================================

pub const SamplerParams = struct {
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    seed: u64 = 0,
};

// ============================================================================
// 采样器
// ============================================================================

pub const Sampler = struct {
    rng: std.Random.Xoshiro256,
    params: SamplerParams,

    pub fn init(params: SamplerParams) Sampler {
        const seed = if (params.seed != 0) params.seed else @as(u64, @bitCast(@as(i64, -1)));
        return Sampler{
            .rng = std.Random.Xoshiro256.init(seed),
            .params = params,
        };
    }

    /// 从 logits 张量采样下一个 token
    pub fn sample(self: *Sampler, logits: *ggml.Tensor) i32 {
        _ = self;
        _ = logits;
        // TODO: 实现采样逻辑
        return 0;
    }

    /// 贪心采样：选择 logits 中概率最大的 token
    /// logits: [n_vocab] f32 张量
    pub fn sampleGreedy(logits: *ggml.Tensor) i32 {
        const data = logits.dataBytes();
        const ne = logits.ne();
        // logits 形状为 [n_vocab, n_tokens]，取最后一个 token
        const n_vocab = @as(usize, @intCast(ne[0]));
        const n_tokens = @max(@as(usize, @intCast(ne[1])), 1);
        const stride = n_vocab;
        const scores = @as([*]f32, @ptrCast(@alignCast(data.ptr)))[(n_tokens - 1) * stride .. (n_tokens - 1) * stride + n_vocab];

        var best_idx: i32 = 0;
        var best_val: f32 = scores[0];
        for (scores, 0..) |val, i| {
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }

};
