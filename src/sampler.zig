//! 采样器
//!
//! 实现 Top-p / Top-k 采样策略。

const std = @import("std");
const ggml = @import("ggml");

const log = std.log.scoped(.core_sampler);

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

    /// 从堆上的 logits 数组贪心采样
    pub fn sampleGreedyFromLogits(logits: []const f32) i32 {
        var best_idx: i32 = 0;
        var best_val: f32 = logits[0];
        for (logits, 0..) |val, i| {
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }

    /// 贪心采样：选择 logits 中概率最大的 token
    /// logits: [n_vocab] f32 张量
    pub fn sampleGreedy(logits: *ggml.Tensor) i32 {
        const data = logits.dataBytes();
        const ne = logits.ne();
        log.debug("sampleGreedy: ne[0]={d}, ne[1]={d}, ne[2]={d}", .{ ne[0], ne[1], ne[2] });
        // logits 形状可能为 [n_vocab, n_tokens] 或 [n_vocab, n_tokens, n_seqs]
        // 在列主序中，ne[0] 是最内层维度
        // 数据布局：[vocab0..n_vocab for token0, vocab0..n_vocab for token1, ...]
        // 取最后一个 token 的 logits
        const n_vocab = @as(usize, @intCast(ne[0]));
        const n_tokens = @max(@as(usize, @intCast(ne[1])), 1);
        const stride = n_vocab;
        const scores = @as([*]f32, @ptrCast(@alignCast(data.ptr)))[(n_tokens - 1) * stride .. (n_tokens - 1) * stride + n_vocab];
        log.debug("sampleGreedy: first 5 scores: {d} {d} {d} {d} {d}", .{ scores[0], scores[1], scores[2], scores[3], scores[4] });

        var best_idx: i32 = 0;
        var best_val: f32 = scores[0];
        for (scores, 0..) |val, i| {
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(i);
            }
        }
        log.debug("sampleGreedy: best_idx={d}, best_val={d}", .{ best_idx, best_val });
        return best_idx;
    }
};
