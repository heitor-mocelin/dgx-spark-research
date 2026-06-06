# benchmarks/ — Llama-3.3-70B (NVFP4) on the GX10

Measured 2026-06-06, NVFP4 / FP8 KV / 32k ctx / Marlin fallback. Raw:
[`results/llama33-70b-nvfp4.json`](results/llama33-70b-nvfp4.json).

## Single-stream

| tok/s | TTFT | ITL | GPU util | Power |
|---:|---:|---:|---:|---:|
| **5.4** | 198 ms | 186 ms | 96% | 32 W |

Predicted realistic ceiling 5.5 → **98% efficiency** — the matrix maximum (39 GB/token weight-read
dwarfs all fixed overhead).

## Concurrency sweep (aggregate tok/s)

| c=1 | c=4 | c=8 | c=16 | c=32 | c=64 | c=96 | c=128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.4 | 21.2 | 42.2 | 83.0 | 159.1 | 234.6 | **431.8** | 431.5 |

Peak **432 tok/s** @ c96, 96% GPU util, **71 W (highest power in the matrix)** — most energy for the
least throughput. See [FINDINGS](../../FINDINGS.md).
