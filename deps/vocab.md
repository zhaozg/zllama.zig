# test-vocab 修复

## 当前状态

```
❯ zig build test-vocab -Doptimize=ReleaseSafe
Build Summary: 4/6 steps succeeded (1 failed); 19/23 tests passed (4 failed)
```

## falcon 修复完成 ✅

修复前：
```
error: 'test_vocab.test.vocab - falcon' failed:
         [falcon] Test #32, token #20: expected 17419, got 1313
         Input: 'Hello, y'all! How are you 😁 ?我想在apple工作1314151天～'
       expected 17419, found 1313
```

修复后：**46 tests passed** ✅

### 问题分析

llama.cpp 的 falcon 预分词器使用三个正则表达式按顺序应用：
1. `[\p{P}\$+<=>\^~\|`]+` - 标点符号
2. `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)` - GPT-2 主模式
3. `[0-9][0-9][0-9]` - 三位数字分组

第三个正则 `[0-9][0-9][0-9]` 在 GPT-2 模式之后应用，将已分组的数字进一步拆分为最多3位一组。

### 修复方法

在 `src/tokenizer/models/falcon.zig` 中：
- 修改步骤2的数字匹配逻辑：ASCII 数字（0-9）每次最多匹配3位
- 非 ASCII 数字（Unicode 数字如 ½ ² ³）保持原样（全部匹配）
- 删除步骤8（三位数字匹配），因为步骤2已处理
- 添加 `countAsciiDigits` 辅助函数

## 剩余失败（预存在问题）

1. **llama-bpe** - Test #45: 167 vs 168 tokens（差1个token）
2. **qwen2** - Test #45: 181 vs 182 tokens（差1个token）
3. **qwen35** - Test #45: 181 vs 182 tokens（差1个token）
4. **mpt** - Test #28: 测试数据过时，当前 llama.cpp 输出与测试数据不一致

### mpt 测试数据过时说明

通过对比当前 llama.cpp 的输出与测试数据，发现 mpt 的测试数据（`ggml-vocab-mpt.gguf.out`）已过时。
当前 llama.cpp 对多个测试用例的输出与测试数据不一致，包括：
- Test #7-9: 空行序列的输出不同
- Test #10: `\t` 输出 `[186]` 而非 `[186 187]`
- Test #28: `\n    Hello\n    Hello\n` 输出 7 tokens 而非 5 tokens
- Test #30: `\n =` 输出 `[426]` 而非 `[187 426]`
- Test #45: 首 token 不同

这些差异表明测试数据需要重新生成。
