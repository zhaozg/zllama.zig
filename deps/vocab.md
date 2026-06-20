# test-vocab 修复

## 当前状态 (2024-06-20)

```
❯ zig build test-vocab -Doptimize=ReleaseSafe
test-vocab
└─ run test test-vocab 17 pass, 6 fail (23 total)
  [refact] 46 tests passed
  [baichuan] SKIPPED
  [bert-bge] SKIPPED
  [nomic-bert-moe] SKIPPED
  [aquila] SKIPPED
error: 'test_vocab.test.vocab - llama-bpe' failed:
         [llama-spm] 46 tests passed

         [llama-bpe] Test #45: '










       🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````""""......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL' -> expected 167 tokens, got 168 tokens
         Expected tokens: 198 4815 15073 66597 8004 1602 2355 79772 11187 9468 248 222 320 8416 8 27623 114 102470 9468 234 104 31643 320 36773 100166 98634 8 26602 227 11410 99 247 9468 99 247 220 18 220 1644 220 8765 220 8765 18 220 8765 1644 220 8765 8765 220 8765 8765 18 220 8765 8765 1644 220 18 13 18 220 18 497 18 220 18 1131 18 220 21549 222 98629 241 45358 233 21549 237 45358 224 21549 244 21549 115 21549 253 45358 223 21549 253 21549 95 98629 227 76460 223 949 37046 101067 19000 23182 102301 9263 18136 16 36827 21909 56560 54337 19175 102118 13373 64571 34694 3114 112203 80112 3436 106451 14196 14196 74694 3089 3089 29249 17523 3001 27708 7801 358 3077 1027 364 83 820 568 596 1070 11 364 793 499 2771 30 364 44 539 2771 358 3358 1304 433 11 364 35 499 1093 1063 15600 30 1226 6 43712 264 64966 43
         Got tokens: 198 4815 15073 66597 8004 1602 2355 79772 415 198 9468 248 222 320 8416 8 27623 114 102470 9468 234 104 31643 320 36773 100166 98634 8 26602 227 11410 99 247 9468 99 247 220 18 220 1644 220 8765 220 8765 18 220 8765 1644 220 8765 8765 220 8765 8765 18 220 8765 8765 1644 220 18 13 18 220 18 497 18 220 18 1131 18 220 21549 222 98629 241 45358 233 21549 237 45358 224 21549 244 21549 115 21549 253 45358 223 21549 253 21549 95 98629 227 76460 223 949 37046 101067 19000 23182 102301 9263 18136 16 36827 21909 56560 54337 19175 102118 13373 64571 34694 3114 112203 80112 3436 106451 14196 14196 74694 3089 3089 29249 17523 3001 27708 7801 358 3077 1027 364 83 820 568 596 1070 11 364 793 499 2771 30 364 44 539 2771 358 3358 1304 433 11 364 35 499 1093 1063 15600 30 1226 6 43712 264 64966 43
       expected 167, found 168
error: 'test_vocab.test.vocab - qwen2' failed:
         [qwen2] Test #45: '










       🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````""""......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL' -> expected 181 tokens, got 182 tokens
         Expected tokens: 198 4710 14731 65497 7847 1572 2303 78672 10947 145836 320 8252 8 26525 114 378 235 149921 30543 320 35673 99066 97534 8 25521 227 11162 99 247 149955 220 18 220 18 18 220 18 18 18 220 18 18 18 18 220 18 18 18 18 18 220 18 18 18 18 18 18 220 18 18 18 18 18 18 18 220 18 18 18 18 18 18 18 18 220 18 13 18 220 18 496 18 220 18 1112 18 220 146394 97529 241 44258 233 146568 44258 224 147603 20879 115 146280 44258 223 146280 147272 97529 227 144534 937 104100 18493 22377 99257 16 18 16 19 16 20 16 35727 21216 55460 53237 18658 14144 1456 13073 63471 33594 3038 133178 79012 3355 4605 4605 13874 13874 73594 3014 3014 28149 17085 2928 26610 7646 358 3003 1012 364 83 813 566 594 1052 11 364 787 498 2704 30 364 44 537 2704 358 3278 1281 432 11 364 35 498 1075 1045 15243 30 1205 6 42612 264 63866 43
         Got tokens: 198 4710 14731 65497 7847 1572 2303 78672 414 198 145836 320 8252 8 26525 114 378 235 149921 30543 320 35673 99066 97534 8 25521 227 11162 99 247 149955 220 18 220 18 18 220 18 18 18 220 18 18 18 18 220 18 18 18 18 18 220 18 18 18 18 18 18 220 18 18 18 18 18 18 18 220 18 18 18 18 18 18 18 18 220 18 13 18 220 18 496 18 220 18 1112 18 220 146394 97529 241 44258 233 146568 44258 224 147603 20879 115 146280 44258 223 146280 147272 97529 227 144534 937 104100 18493 22377 99257 16 18 16 19 16 20 16 35727 21216 55460 53237 18658 14144 1456 13073 63471 33594 3038 133178 79012 3355 4605 4605 13874 13874 73594 3014 3014 28149 17085 2928 26610 7646 358 3003 1012 364 83 813 566 594 1052 11 364 787 498 2704 30 364 44 537 2704 358 3278 1281 432 11 364 35 498 1075 1045 15243 30 1205 6 42612 264 63866 43
       expected 181, found 182
error: 'test_vocab.test.vocab - qwen35' failed:
         [qwen35] Test #45: '










       🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````""""......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL' -> expected 181 tokens, got 182 tokens
         Expected tokens: 198 4710 14731 65497 7847 1572 2303 78672 10947 145836 320 8252 8 26525 114 378 235 149921 30543 320 35673 99066 97534 8 25521 227 11162 99 247 149955 220 18 220 18 18 220 18 18 18 220 18 18 18 18 220 18 18 18 18 18 220 18 18 18 18 18 18 220 18 18 18 18 18 18 18 220 18 18 18 18 18 18 18 18 220 18 13 18 220 18 496 18 220 18 1112 18 220 146394 97529 241 44258 233 146568 44258 224 147603 20879 115 146280 44258 223 146280 147272 97529 227 144534 937 104100 18493 22377 99257 16 18 16 19 16 20 16 35727 21216 55460 53237 18658 14144 1456 13073 63471 33594 3038 133178 79012 3355 4605 4605 13874 13874 73594 3014 3014 28149 17085 2928 26610 7646 358 3003 1012 364 83 813 566 594 1052 11 364 787 498 2704 30 364 44 537 2704 358 3278 1281 432 11 364 35 498 1075 1045 15243 30 1205 6 42612 264 63866 43
         Got tokens: 198 4710 14731 65497 7847 1572 2303 78672 414 198 145836 320 8252 8 26525 114 378 235 149921 30543 320 35673 99066 97534 8 25521 227 11162 99 247 149955 220 18 220 18 18 220 18 18 18 220 18 18 18 18 220 18 18 18 18 18 220 18 18 18 18 18 18 220 18 18 18 18 18 18 18 220 18 18 18 18 18 18 18 18 220 18 13 18 220 18 496 18 220 18 1112 18 220 146394 97529 241 44258 233 146568 44258 224 147603 20879 115 146280 44258 223 146280 147272 97529 227 144534 937 104100 18493 22377 99257 16 18 16 19 16 20 16 35727 21216 55460 53237 18658 14144 1456 13073 63471 33594 3038 133178 79012 3355 4605 4605 13874 13874 73594 3014 3014 28149 17085 2928 26610 7646 358 3003 1012 364 83 813 566 594 1052 11 364 787 498 2704 30 364 44 537 2704 358 3278 1281 432 11 364 35 498 1075 1045 15243 30 1205 6 42612 264 63866 43
       expected 181, found 182
error: 'test_vocab.test.vocab - falcon' failed:
         [gpt-2] 46 tests passed
         [gemma-4] 46 tests passed

         [falcon] Test #30: '
        =' -> expected 2 tokens, got 3 tokens
         Expected tokens: 1212 40
         Got tokens: 193 204 40
       expected 2, found 3
error: 'test_vocab.test.vocab - deepseek-coder' failed:
         [deepseek-coder] Test #25: '  Hello' -> expected 3 tokens, got 2 tokens
         Expected tokens: 207 414 9489
         Got tokens: 243 17535
       expected 3, found 2
error: 'test_vocab.test.vocab - mpt' failed:
         [deepseek-llm] 46 tests passed
         [phi-3] 46 tests passed
         [command-r] 46 tests passed
         [starcoder] 46 tests passed

         [mpt] Test #25, token #0: expected 50276, got 209
         Input: '  Hello'
       expected 50276, found 209
failed command: ./.zig-cache/o/a70c3b9cc101de0ffb901f68f8e8e211/test-vocab --cache-dir=./.zig-cache --seed=0xd0b3904c --listen=-

Build Summary: 4/6 steps succeeded (1 failed); 17/23 tests passed (6 failed)
test-vocab transitive failure
└─ run test test-vocab 17 pass, 6 fail (23 total)

error: the following build command failed with exit code 1:
.zig-cache/o/09ab2f9ec9d24fa003b7512be1f6d130/build /Users/zhaozg/opt/zig-x86_64-macos-0.16.0/zig /Users/zhaozg/opt/zig-x86_64-macos-0.16.0/lib /Users/zhaozg/work/ai/zllama.zig .zig-cache /Users/zhaozg/.cache/zig --seed 0xd0b3904c -Z14b077599ba58855 test-vocab -Doptimize=ReleaseSafe
```

## 存在的问题

我们集中处理剩余 6 个失败中的 3 个简单问题（`falcon`、`deepseek-coder`、`mpt`），这些失败的原因已经明确，修复方法清晰。


### 📌 修复 `falcon` Test #30

**问题**：`'\n        ='` 期望 `[1212, 40]`，实际 `[193, 204, 40]`。预分词将换行和空格拆分为两个 token，而官方将它们合并为一个。

**修复**：在 `preTokenizeFalcon` 函数中，**优先处理空白序列**，将连续空白（包括换行）合并为一个单词。

**代码修改**（`src/tokenizer/models/falcon.zig`）：
```zig
pub fn preTokenizeFalcon(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 优先处理空白（包括换行和空格）
        if (unicode.isAsciiWhitespace(text[i])) {
            const start = i;
            while (i < text.len and unicode.isAsciiWhitespace(text[i])) i += 1;
            const word = try result.allocator.dupe(u8, text[start..i]);
            try result.words.append(result.allocator, word);
            continue;
        }
        // 其余逻辑（标点、字母、数字等）保持不变...
        // 注意：原有的 tryMatchContractionOrWord 中的空白处理会被跳过，因为外层已处理。
    }
}
```

---

### 📌 修复 `deepseek-coder` Test #25

**问题**：`'  Hello'` 期望 `[207, 414, 9489]`（两个空格 + Hello），实际 `[243, 17535]`（空格被合并）。

**修复**：在 `preTokenizeDeepseekCoder` 中，**将空格独立分支移到最前面**（紧接换行处理之后），确保每个空格作为独立的单词。

参考：
```
zllama.zig on  main [!⇡] via ↯ v0.16.0
❯ llama-tokenize -m deps/llama.cpp/models/ggml-vocab-deepseek-coder.gguf --ids -p ' Hello' --log-disable --no-bos
[414, 9489]

zllama.zig on  main [!⇡] via ↯ v0.16.0
❯ llama-tokenize -m deps/llama.cpp/models/ggml-vocab-deepseek-coder.gguf --ids -p '  Hello' --log-disable --no-bos
[207, 414, 9489]

❯ llama-tokenize -m deps/llama.cpp/models/ggml-vocab-deepseek-coder.gguf --ids -p '   Hello' --log-disable --no-bos
[243, 414, 9489]

```

**代码修改**（`src/tokenizer/models/deepseek_coder.zig`）：
```zig
pub fn preTokenizeDeepseekCoder(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        // 1. 换行处理（不变）
        if (text[i] == '\r' or text[i] == '\n') { ... }

        // 2. 空格独立分支（提前）
        if (text[i] == ' ') {
            const word = try result.allocator.dupe(u8, text[i..i+1]);
            try result.words.append(result.allocator, word);
            i += 1;
            continue;
        }

        // 3. 其他分支（字母、标点、CJK、数字等）不变...
    }
}
```

---

## 📌 修复 `mpt` Test #25

**问题**：`'  Hello'` 期望 `[50276, 12092]`（双空格 + Hello），实际 `[209, 12092]`（单空格 + Hello），因为双空格被拆分为两个单空格。

**修复**：`preTokenizeMpt` 应将连续空格合并为一个单词，且 `encodeWord` 的整词匹配能命中 `'  '`。

**代码修改**（`src/tokenizer/models/mpt.zig`）：
```zig
pub fn preTokenizeMpt(text: []const u8, result: *PreTokenized) !void {
    var i: usize = 0;
    while (i < text.len) {
        const start = i;
        if (text[i] == ' ') {
            // 连续空格合并为一个 token
            while (i < text.len and text[i] == ' ') i += 1;
        } else if (text[i] == '\n') {
            // 换行符单独作为一个 token
            i += 1;
        } else if (unicode.isAsciiWhitespace(text[i])) {
            // 其他空白（制表符等）合并
            while (i < text.len and unicode.isAsciiWhitespace(text[i])) i += 1;
        } else {
            // 非空白序列
            while (i < text.len and !unicode.isAsciiWhitespace(text[i])) i += 1;
        }
        const word = try result.allocator.dupe(u8, text[start..i]);
        try result.words.append(result.allocator, word);
    }
}
```

同时，在 `encodeWord` 中，确保 `'  '` 能被 `textToTokenFn` 命中。如果仍未命中，检查词表中是否存在 `'  '`，以及查询时使用的字符串是否相同（注意空格编码，可能应为 `ĠĠ`）。如果是 `ĠĠ`，则需在预分词时将空格映射为 `Ġ`，但根据官方输出，`50276` 就是双空格的 token ID，说明词表中存在原始双空格字符串。因此，整词匹配应使用原始单词 `'  '`。

---

## ✅ 验证

修改上述三个文件后，重新编译并运行测试：

```bash
zig build clean
zig build test-vocab -Doptimize=ReleaseSafe --summary all
```

预期 `falcon`、`deepseek-coder`、`mpt` 通过，剩下 `llama-bpe`、`qwen2`、`qwen35` 的长文本测试（Test #45）可能仍失败，需要后续分析。

---

## 📌 长文本测试分析（llama-bpe / qwen2 / qwen35）

这些失败在开头出现差异（`415` 或 `414` 替换了 `10947`），可能由于预分词器对 `'  '`（双空格）或 `' '` 的处理与官方不同，导致 token 数量多 1。可暂时搁置，优先确保其他测试通过。

