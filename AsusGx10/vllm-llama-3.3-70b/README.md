# vllm-llama-3.3-70b

**Llama-3.3-70B-Instruct** (NVFP4) on the GX10 — the **largest dense** model in the
[7-model test matrix](../FINDINGS.md), and the one that runs **closest to its theoretical ceiling
(98%)**. Benchmark-centric subproject.

## The model

| | |
|---|---|
| Params | **70.6 B, dense** |
| Architecture | 80-layer transformer, hidden 8192, GQA | 
| Context | 128K |
| Languages | 8 (EN, DE, FR, IT, PT, HI, ES, TH) |
| Training | SFT + RLHF |
| Released / License | Meta, Dec 2024 / Llama 3.3 Community License |
| NVFP4 footprint | ~39 GB |

## Headline result (measured 2026-06-06)

| Single-stream | Peak aggregate | Efficiency vs roofline |
|---:|---:|---:|
| **5.4 tok/s** | **432 tok/s** @ c96 | **98%** |

**Why it matters:** Llama-3.3-70B is the matrix's **roofline anchor** — at 98% it *is* its
bandwidth ceiling. The huge ~39 GB weight-read per token dwarfs every fixed overhead, so there's
almost nothing left to lose. It also draws the **most power (71 W)** for the **least throughput** —
the clearest illustration that, on the GX10, a big dense model is a slow ceiling, not a broken one.
See [FINDINGS](../FINDINGS.md).

## Contents

- **[guides/](guides/00-overview-deployment-and-results.md)** · **[scripts/](scripts/launch.sh)** ·
  **[benchmarks/](benchmarks/README.md)** · **[sources/](sources/INDEX.md)**

[MIT](../../LICENSE) © 2026 Heitor Mocelin
