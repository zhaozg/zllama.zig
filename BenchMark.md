# 性能记录

重源码构建，飞一般的感觉。

## zig + native ggml

```
zllama.zig on  build/ggml via ↯ v0.16.0
❯ zig-out/bin/zllama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf "3+3=?" --benchmark

============ Benchmark Results ============
  Model            : gpt2
  Architecture     : qwen35
  Threads          : 6
  Prompt tokens    : 5
  Output tokens    : 32
  ------------------------------------------
  PP eval time     : 0.316 s (15.8 tok/s)
  TG time          : 2.209 s (14.5 tok/s)
  Total time       : 2.525 s (12.7 tok/s)
=============================================
```

## zig + Dbundle-ggml with ReleaseFast

```
❯ zig build -Dbundle-ggml -Doptimize=ReleaseFast

zllama.zig on  build/ggml via ↯ v0.16.0 took 57s
❯ zig-out/bin/zllama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf "3+3=?" --benchmark

============ Benchmark Results ============
  Model            : gpt2
  Architecture     : qwen35
  Threads          : 6
  Prompt tokens    : 5
  Output tokens    : 32
  ------------------------------------------
  PP eval time     : 0.060 s (82.9 tok/s)
  TG time          : 0.748 s (42.8 tok/s)
  Total time       : 0.808 s (39.6 tok/s)
=============================================
```


## zig + native ggml with ReleaseFast

```
❯ zig build -Doptimize=ReleaseFast

zllama.zig on  build/ggml via ↯ v0.16.0 took 33s
❯ zig-out/bin/zllama-simple -m ~/.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf "3+3=?" --benchmark

============ Benchmark Results ============
  Model            : ����
  Architecture     : qwen35
  Threads          : 6
  Prompt tokens    : 5
  Output tokens    : 32
  ------------------------------------------
  PP eval time     : 0.464 s (10.8 tok/s)
  TG time          : 2.354 s (13.6 tok/s)
  Total time       : 2.819 s (11.4 tok/s)
=============================================
```
