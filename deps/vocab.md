# test-vocab 修复

## 当前状态 (2024-06-20)

经过修复，测试结果从 16 pass, 7 fail 变为 16 pass, 7 fail（失败数量未变，但具体失败项有变化）。

### 已修复的问题

1. **llama-bpe Test #32** (`😁` 等): **已修复** ✅
   - 原因：`1314151` 数字序列的 BPE 合并顺序与预期不同。
   - 修复：为 `llama3` pre_type 创建独立的 `preTokenizeLlama3` 函数，使用 `\\p{N}{1,3}`（1-3位数字分组）替代 `\\p{N}+`（无限数字）。

2. **deepseek-llm Test #0** (`½`): **已修复** ✅
   - 原因：`½` (U+00BD) 被 `isAsciiLatinLetter` 错误识别为 Latin 字母（因为其 UTF-8 首字节 0xC2 >= 0xC0）。
   - 修复：在 `preTokenizeDeepseekLlm` 中使用 `unicode.decodeCodepoint` + `isLatinLetter` 替代 `isAsciiLatinLetter`，并在 `isLatinLetter` 中排除 emoji。

3. **falcon Test #0** (`ied 4 ½ months`): **已修复** ✅
   - 原因：` ?\\p{N}+` 模式在 `tryMatchContractionOrWord` 的空白处理之后才被检查，导致 ` ½` 被拆分为 ` ` + `½`。
   - 修复：在 `preTokenizeFalcon` 中将 ` ?\\p{N}+` 检查移到空白处理之前，并添加独立的 ` ?\\p{L}+` 和 ` ?[^\\s\\p{L}\\p{N}]+` 检查。

4. **falcon Test #25** (`'  Hello'`): **已修复** ✅
   - 原因：`\\s+(?!\\S)` 回溯逻辑导致 `  Hello` 中的空格处理错误。
   - 修复：在空白处理中添加条件回溯，仅当空白后的字符可以被 ` ?` 模式匹配时才应用回溯。

5. **falcon Test #18** (` this is 🦙.cpp`): **已修复** ✅
   - 原因：` ?[^\\s\\p{L}\\p{N}]+` 模式在空白处理之后才被检查，导致 ` 🦙` 被拆分为 ` ` + `🦙`。
   - 修复：在空白处理之前添加独立的 ` ?[^\\s\\p{L}\\p{N}]+` 检查。

### 仍存在的问题

| 模型 | 失败测试 | 问题本质 |
|------|---------|---------|
| `llama-bpe` | Test #45（长文本） | 多 emoji 及特殊字符序列的预分词不一致（167 vs 168 tokens） |
| `qwen2` / `qwen35` | Test #45（长文本） | 多 emoji 及特殊字符序列的预分词不一致（181 vs 182 tokens） |
| `falcon` | Test #30 (`\n        =`) | 换行+空格未能合并为一个 token |
| `deepseek-coder` | Test #18 (`🦙.cpp`) | `.` 被合并到 `🦙`，期望分离 |
| `deepseek-llm` | Test #18 (`🦙.cpp`) | `.` 被合并到 `🦙`，期望分离 |
| `mpt` | Test #25 (`  Hello`) | 空格处理与预期不同 |

### 待修复问题分析

#### llama-bpe / qwen2 / qwen35 Test #45（长文本）
- 问题：`😶‍🌫️` ZWJ emoji 序列的预分词不一致。
- 预期：`10947` 作为一个 token。
- 实际：`414 198` 作为两个 token。
- 原因：ZWJ emoji 序列在 BPE 编码阶段的合并顺序与预期不同。预分词器正确地将 ZWJ 序列保持为一个词，但 BPE 合并算法产生了不同的结果。
- 修复方向：需要深入分析 BPE 合并算法的优先级队列行为，确保合并顺序与 llama.cpp 一致。

#### falcon Test #30 (`\n        =`)
- 问题：`\n        =` 应产生 2 个 token（空白 + `=`），但产生了 3 个 token。
- 原因：`\\s+(?!\\S)` 回溯逻辑将 `\n        `（9个空白字符）拆分为 `\n       `（8个）+ ` `（1个），因为 `=` 是标点符号，条件回溯判断 `=` 可以被 ` ?` 模式匹配。
- 修复方向：需要更精确地判断何时应用回溯。当空白后的字符是标点符号时，不应应用回溯（因为标点符号由独立的 `[\\p{P}...]+` 模式匹配，不需要前导空格）。

#### deepseek-coder / deepseek-llm Test #18 (`🦙.cpp`)
- 问题：`🦙.cpp` 应产生 5 个 token（`🦙` + `.` + `cpp`），但产生了 3 个 token（`🦙.` + `cpp`）。
- 原因：`🦙`（emoji）被 `isLetterAt` 识别为字母（`is_alphabetic` 属性为 true），导致 ` ?\\p{L}+` 模式匹配了 ` 🦙` 作为一个词。然后 `.` 被包含在 `🦙` 的字母匹配中。
- 修复方向：在 deepseek-coder 和 deepseek-llm 的字母匹配中排除 emoji。已在 `isLatinLetter` 中添加了 emoji 排除，但 deepseek-coder 的 `isLetterAt` 调用尚未排除 emoji。

#### mpt Test #25 (`  Hello`)
- 问题：`  Hello` 应产生 2 个 token（`  ` + `Hello`），但产生了不同的结果。
- 原因：MPT 的空白处理逻辑与 GPT-2 不同。
- 修复方向：需要参考 llama.cpp 中 MPT 的 regex 实现，调整空白处理逻辑。

## 修复策略总结

1. **为每个模型编写独立的预分词函数**，复制 `llama.cpp` 对应的正则表达式（将 C++ 的正则翻译为 Zig 的状态机或循环）。
2. **使用 `uucode` 提供的类别判断函数**，精确匹配 `\\p{L}`、`\\p{N}`、`\\p{P}`、`\\p{S}` 等。
3. **排除 emoji 的字母匹配**：在 deepseek-coder 等模型的字母匹配中，使用 `isEmoji` 排除 emoji 字符。
4. **精确控制空白回溯**：`\\s+(?!\\S)` 回溯仅应在空白后的字符可以被 ` ?` 模式匹配时应用。
