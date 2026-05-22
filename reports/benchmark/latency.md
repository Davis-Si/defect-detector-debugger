# CPU latency benchmark

- run: `baseline`
- platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- torch: 2.5.1+cpu, intra-op threads: 8
- model: ResNet18 (frozen backbone) + linear head, 6 classes

| batch | p50 (ms) | p95 (ms) | per-image p50 (ms) | throughput (FPS) |
|------:|---------:|---------:|-------------------:|-----------------:|
|     1 |    47.92 |   555.41 |              47.92 |             20.9 |
|     4 |    90.85 |   301.09 |              22.71 |             44.0 |
|    16 |   309.11 |   424.32 |              19.32 |             51.8 |
|    32 |   476.25 |   654.50 |              14.88 |             67.2 |

Methodology: warmup of 5 forward passes, then 30 timed iterations per batch size.
Timing via `time.perf_counter()` around `model(x)` only — preprocessing is
not measured. p50 / p95 are over the 30-iteration sample.