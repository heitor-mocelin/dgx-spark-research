# vllm-qwen3-32b

**Qwen3-32B** (NVFP4) on the GX10 — a **dense** 32 B model, included in the
[7-model test matrix](../FINDINGS.md) as a *dense reference point*. This is a focused,
benchmark-centric subproject: what the model is, how it's served, and what it measured.

> Not a full local-inference guide series (those are reserved for the flagship MoE models
> [Qwen3.6](../vllm-qwen3.6-35b-a3b/) and [Gemma 4](../vllm-gemma4-26b-a4b/)). Here the focus is
> the **measured roofline result**.

## The model

| | |
|---|---|
| Params | **32.8 B, dense** (all active per token) |
| Architecture | 64-layer transformer, GQA (64 query / 8 KV heads), SwiGLU, RoPE, RMSNorm |
| Context | 32K native, **131K with YaRN** |
| Modes | unified **thinking / non-thinking** |
| Languages | 119 |
| License | Apache-2.0 (Qwen Team) |
| NVFP4 footprint | ~18 GB (fits the 128 GB box trivially) |

## Headline result (measured 2026-06-06)

| Single-stream | Peak aggregate | Efficiency vs roofline |
|---:|---:|---:|
| **11.0 tok/s** | **884 tok/s** @ c128 | **91%** |

**Why it matters:** Qwen3-32B is one of two models that nearly *hit* their roofline (91%), because
a dense 32 B weight-read dominates the fixed per-token overheads. It's the clean counterpoint to
the small-active MoEs (42–58%) — and the foil that exposes [Gemma-4-31B as an
outlier](../vllm-gemma4-26b-a4b/) (54% at the same size). Full analysis:
[FINDINGS](../FINDINGS.md).

## Contents

- **[guides/](guides/00-overview-deployment-and-results.md)** — overview, architecture, the NVFP4
  serving recipe, and the measured roofline placement.
- **[scripts/launch.sh](scripts/launch.sh)** — the vLLM NVFP4 launch used in the matrix.
- **[benchmarks/](benchmarks/README.md)** — single-stream + concurrency sweep + telemetry.
- **[sources/](sources/INDEX.md)** — cited references.

[MIT](../../LICENSE) © 2026 Heitor Mocelin
