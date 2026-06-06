# benchmarks/ — Qwen3-Next-80B-A3B (NVFP4) on the GX10

Measured 2026-06-06, NVFP4 / FP8 KV / 32k ctx / Marlin fallback. Raw:
[`results/qwen3-next-80b-a3b-nvfp4.json`](results/qwen3-next-80b-a3b-nvfp4.json).

## Single-stream

| tok/s | TTFT | ITL | GPU util | Power |
|---:|---:|---:|---:|---:|
| **35.5** | 91 ms | 27 ms | **60.6%** | 27 W |

Predicted realistic ceiling 129 → **~27% efficiency — the matrix minimum.** The low GPU util (60%
vs 89–96% for everything else) says it's *not* compute/Marlin-bound; the Gated DeltaNet recurrence
appears latency-bound per token.

## Concurrency sweep (aggregate tok/s)

| c=1 | c=4 | c=8 | c=16 | c=32 | c=64 | c=96 | c=128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36.5 | 133 | 250 | 426 | 639 | 790 | **1021** | 1002 |

Peak **1021 tok/s** @ c96 (90% GPU, 51 W) — concurrency hides the single-stream latency. See
[FINDINGS](../../FINDINGS.md). *(This model benchmarked successfully before the GPT-OSS-120B deploy
wedged the box — it's the salvaged round-2 data point.)*
