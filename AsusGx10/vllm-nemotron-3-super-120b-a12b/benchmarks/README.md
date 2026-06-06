# benchmarks/ — Nemotron-3-Super-120B-A12B (NVFP4) on the GX10

Measured 2026-06-06, NVFP4 / FP8 KV / 32k ctx / Marlin fallback. Raw:
[`results/nemotron-super-120b-a12b-nvfp4.json`](results/nemotron-super-120b-a12b-nvfp4.json).

## Single-stream

| tok/s | TTFT | ITL | GPU util | Power |
|---:|---:|---:|---:|---:|
| **14.6** | 265 ms | 67 ms | 89% | 36 W |

Predicted realistic ceiling 32.4 → **45% efficiency** (hybrid Mamba-2/LatentMoE discount, like
Nemotron-Nano). A 120 B *dense* model would be ~4 tok/s — the 12 B-active MoE makes it usable.

## Concurrency sweep (aggregate tok/s)

| c=1 | c=4 | c=8 | c=16 | c=32 | c=64 | c=96 | c=128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 14.7 | 48.4 | 90.2 | 151.7 | 227.4 | 245.5 | **327.0** | 326.2 |

Peak **327 tok/s** @ 96% GPU util / 56 W. Modest aggregate — 12 B-active is real compute/token under
batching. **MTP heads enable native speculative decoding** (latency, not measured here). See
[FINDINGS](../../FINDINGS.md).
