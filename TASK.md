# 语音识别问题修复

运行: `./zig-out/bin/zllama --model ~/.cache/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj ~/.cache/models/mmproj-F16.gguf --audio ~/.cache/models/hello.wav -n 5 -p "Transcribe the audio:" -v`

输出 `Outsપણે Œ Strattonヨーク` 与预期 `hello world` 不符合.

## 已完成 (2026-06-10)

### 1. ✅ 使用 std.log 的运行时调整日志级别，实现 -v(info) -d(debug)
- commit 418092a

### 2. 🚧 从底层到高层，逐层验证 audio 处理过程
- WAV 加载 (16kHz, 1ch, 0.8s) → 验证通过
- Mel 频谱 (79帧×128bins, sr=16000Hz) → 验证通过
- Conv2D 子采样 → 待验证
- Conformer 编码 → 正在验证

### 3. ✅ 杜绝 hardcode，从模型/权重加载参数
- `pos_emb` 张量用正弦位置编码填充（原未初始化 — 关键修复）
- `n_mel_bins` 从 GGUF 元数据读取

### 4. ✅ kq_mask 不再使用未初始化内存
- commit: current
- `kq_mask` 张量原被分配但从未填充，导致注意力使用随机内存
- 新增 `fillChunkedAttentionMask()` 函数，按因果+padding规则正确填充

## 输出变化
| 版本 | 输出 |
|------|------|
| 修复前 (pos_emb+kq_mask未初始化) | `مور subroutine conval SajMình` |
| pos_emb修复后, kq_mask未初始化 | `town kulaTUB Chick Bie` |
| pos_emb+kq_mask都修复 | `Outsપણે Œ Strattonヨーク` |

## 待进一步调查
- Conformer 分块注意力中的张量布局（permute 顺序 + mulMat 维度）
- 卷积模块（GLU gate + depthwise conv）实现正确性
- 输出中仍含 Unicode 字符 → 可能 tokenizer/decoder 也有问题
- Mel 频谱参数可能需要与模型训练时匹配
