# test-vocab 修复

## 当前状态 (2024-06-20)

```
❯ zig build test-vocab -Doptimize=ReleaseSafe
Build Summary: 4/6 steps succeeded (1 failed); 17/23 tests passed (6 failed)
```

## 已修复的问题

### ✅ deepseek-coder / deepseek-llm Test #18 (`🦙.cpp`)
- 添加 Emoji 优先分支，确保 Emoji 被识别为独立单词

### ✅ deepseek-coder / deepseek-llm Test #21（高棉文）
- 添加 `isLetterStrict()` 函数（使用 L* 类别），字母分支改用严格字母判断

### ✅ llama-bpe / qwen2 / qwen35 Test #21（高棉文）
- 保留 `isLetter()` 使用 `is_alphabetic` 属性，确保 GPT-2 风格预分词器行为不变

### ✅ deepseek-llm 全部 46 个测试通过

## 当前失败（6 个）

### falcon Test #32
- 输入包含 emoji、CJK、数字混合
- token #20 期望 `17419`，实际 `1313`
- 可能是数字预分词或 BPE 编码问题

### deepseek-coder Test #30
- 输入 `'\n        ='`，期望 `[185, 405]`，实际 `[185, 207, 28]`
- 测试数据可能过时（当前 llama.cpp 输出 `[185, 294, 28]`）

### mpt Test #28
- 输入 `'\n    Hello\n    Hello\n'`，期望 5 tokens，实际 4 tokens
- 测试数据可能过时（当前 llama.cpp 输出 7 tokens）

### llama-bpe / qwen2 / qwen35 Test #45
- 长文本测试，差 1 个 token
- 可能是预分词器对某些边界情况的处理与 llama.cpp 不一致

## 测试状态

| 测试 | 状态 | 说明 |
|------|------|------|
| llama-spm | ✅ 通过 | 46 tests passed |
| llama-bpe | ❌ 失败 | Test #45 长文本 |
| qwen2 | ❌ 失败 | Test #45 长文本 |
| qwen35 | ❌ 失败 | Test #45 长文本 |
| gpt-2 | ✅ 通过 | 46 tests passed |
| gemma-4 | ✅ 通过 | 46 tests passed |
| falcon | ❌ 失败 | Test #32 token #20 |
| deepseek-coder | ❌ 失败 | Test #30 |
| deepseek-llm | ✅ 通过 | **全部 46 个测试通过！** |
| phi-3 | ✅ 通过 | 46 tests passed |
| command-r | ✅ 通过 | 46 tests passed |
| starcoder | ✅ 通过 | 46 tests passed |
| mpt | ❌ 失败 | Test #28 |
| refact | ✅ 通过 | 46 tests passed |
| baichuan | ⏭️ 跳过 | 缺少测试数据 |
| bert-bge | ⏭️ 跳过 | WPM 未实现 |
| nomic-bert-moe | ⏭️ 跳过 | 缺少测试数据 |
| aquila | ⏭️ 跳过 | 缺少测试数据 |

**总计：17/23 通过（6 失败），相比修复前 9/23 通过有显著提升。**
