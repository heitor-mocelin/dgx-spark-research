# benchmarks/ — Nemotron-3-Nano-30B-A3B (NVFP4) on the GX10

Measured 2026-06-06, NVFP4 / FP8 KV / 32k ctx / Marlin fallback. Raw:
[`results/nemotron-nano-30b-a3b-nvfp4.json`](results/nemotron-nano-30b-a3b-nvfp4.json).

## Single-stream

| tok/s | TTFT | ITL | GPU util | Power |
|---:|---:|---:|---:|---:|
| **54.1** | 80 ms | 17 ms | 96% | 45 W |

Predicted realistic ceiling 129 → **42% efficiency** (matrix minimum — the hybrid Mamba-2 layers
don't match the attention-weight-read roofline).

## Concurrency sweep (aggregate tok/s)

| c=1 | c=4 | c=8 | c=16 | c=32 | c=64 | c=96 | c=128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.1 | 187.8 | 344.4 | 572.2 | 861.0 | 982.4 | **1215.3** | 1214.1 |

Peak **1215 tok/s (matrix max)** @ 96% GPU util, **44 W (matrix min power)** — best
throughput-per-watt of any model tested. See [FINDINGS](../../FINDINGS.md).
