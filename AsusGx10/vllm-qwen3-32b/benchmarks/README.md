# benchmarks/ — Qwen3-32B (NVFP4) on the GX10

Measured 2026-06-06, NVFP4 / FP8 KV / 32k ctx / Marlin fallback. Single-stream (5 reps) +
concurrency sweep (each point ×3) + dcgm telemetry. Raw: [`results/qwen3-32b-nvfp4.json`](results/qwen3-32b-nvfp4.json).

## Single-stream

| tok/s | TTFT | ITL | GPU util | Power |
|---:|---:|---:|---:|---:|
| **11.0** (±0.08) | 102 ms | 90 ms | 89% | 37 W |

Predicted realistic ceiling 12.1 tok/s → **91% efficiency** (one of the highest in the matrix —
the dense weight-read dominates fixed overhead).

## Concurrency sweep (aggregate tok/s)

| c=1 | c=4 | c=8 | c=16 | c=32 | c=64 | c=96 | c=128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11.0 | 44.6 | 88.2 | 173.7 | 333.9 | 483.9 | 880.7 | **883.8** |

Near-linear to ~c96, then flat (power/compute cap). Peak **884 tok/s** @ 96% GPU util / 62 W.

## Context

See [FINDINGS](../../FINDINGS.md) for cross-model analysis. Reproduce with the shared client:
`benchmark_sweep.py` (point `MODEL=qwen3-32b` at this endpoint).
