# vllm-nemotron-3-super-120b-a12b

**NVIDIA Nemotron-3-Super-120B-A12B** (NVFP4) on the GX10 — a **120 B** hybrid Mamba-2 + LatentMoE
model that runs *interactively* on a desktop because only **12 B activate per token**. The
[matrix's](../FINDINGS.md) "big total, small active" extreme. Benchmark-centric subproject.

## The model

| | |
|---|---|
| Params | **120.6 B total / 12.7 B active** |
| Architecture | **LatentMoE**: Mamba-2 + latent-dimension MoE routing + Attention + **MTP heads** (native speculative decoding) |
| Context | up to **1 M tokens** (RULER@1M = 91.75 — class-leading) |
| Vendor / License | NVIDIA (open) |
| NVFP4 footprint | ~75 GB (still fits the 128 GB box) |

## Headline result (measured 2026-06-06)

| Single-stream | Peak aggregate | Efficiency vs roofline |
|---:|---:|---:|
| **14.6 tok/s** | **327 tok/s** | 45% |

**Why it matters:** this is the **MoE thesis at the extreme.** A 120 B *dense* model would read
~67 GB/token → ~**4 tok/s** and barely fit. By activating only 12 B, Nemotron-Super reads ~6.75 GB
and runs **~15 tok/s** — *usable* — while holding a 120 B-parameter brain and a 1 M context in the
GX10's unified memory. Plus **MTP heads give native speculative decoding** (a latency upside we
didn't exercise in greedy benchmarking). See [FINDINGS](../FINDINGS.md).

## Contents

- **[guides/](guides/00-overview-deployment-and-results.md)** · **[scripts/](scripts/launch.sh)** ·
  **[benchmarks/](benchmarks/README.md)** · **[sources/](sources/INDEX.md)**

[MIT](../../LICENSE) © 2026 Heitor Mocelin
