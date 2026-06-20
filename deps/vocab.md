# test-vocab 修复

## 当前状态

```
❯ zig build test-vocab -Doptimize=ReleaseSafe
Build Summary: 4/6 steps succeeded (1 failed); 20/23 tests passed (3 failed)
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
1. `[\\p{P}\\$+<=>\\^~\\|`]+` - 标点符号
2. `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)` - GPT-2 主模式
3. `[0-9][0-9][0-9]` - 三位数字分组

第三个正则 `[0-9][0-9][0-9]` 在 GPT-2 模式之后应用，将已分组的数字进一步拆分为最多3位一组。

### 修复方法

在 `src/tokenizer/models/falcon.zig` 中：
- 修改步骤2的数字匹配逻辑：ASCII 数字（0-9）每次最多匹配3位
- 非 ASCII 数字（Unicode 数字如 ½ ² ³）保持原样（全部匹配）
- 删除步骤8（三位数字匹配），因为步骤2已处理
- 添加 `countAsciiDigits` 辅助函数

## mpt 修复完成 ✅

修复前：
```
[mpt] Test #28: '    Hello\n    Hello' -> expected 5 tokens, got 4 tokens
Expected tokens: 50274 12092 187 50274 12092 
Got tokens: 50274 12092 1760 12092 
```

修复后：**46 tests passed** ✅

### 问题分析

MPT 使用与 GPT-2 相同的正则表达式，但预分词行为不同：
- GPT-2 的 `\s+(?!\S)` 规则会取 n-1 个空白字符，留下最后一个给下一次迭代
- MPT 需要将连续的同类型空白分组，但不同类型空白分开处理

### 修复方法

在 `src/tokenizer/models/mpt.zig` 中重写了空白处理逻辑：
- 连续换行符合并，如果后面恰好有一个空格且该空格后面是空白，则包含该空格
- 连续空格合并
- 单个空格后面是制表符时，空格和制表符合并
- 连续制表符合并
- 其他空白按单个字符处理

## 剩余失败（预存在问题）

1. **llama-bpe** - Test #45: 167 vs 168 tokens（差1个token）
2. **qwen2** - Test #45: 181 vs 182 tokens（差1个token）
3. **qwen35** - Test #45: 181 vs 182 tokens（差1个token）
