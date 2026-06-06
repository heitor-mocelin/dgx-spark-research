# vllm-nemotron-3-nano-30b-a3b

**NVIDIA Nemotron-3-Nano-30B-A3B** (NVFP4) on the GX10 — a **hybrid Mamba-2 + MoE + Attention**
model, and the most surprising entry in the [7-model matrix](../FINDINGS.md): **lowest
single-stream efficiency yet highest aggregate throughput and lowest power.** Benchmark-centric
subproject.

## The model

| | |
|---|---|
| Params | **31.6 B total / ~3.2 B active** (3.6 B w/ embeddings) |
| Architecture | **Hybrid**: 23 Mamba-2 & MoE layers + 6 Attention layers; **128 routed experts, top-6 + 2 shared** |
| Context | up to **1 M tokens** |
| Training | ~25 T tokens; reasoning / agentic focus |
| Vendor / License | NVIDIA (open) |
| NVFP4 footprint | ~19 GB |

## Headline result (measured 2026-06-06)

| Single-stream | Peak aggregate | Efficiency vs roofline | Power@peak |
|---:|---:|---:|---:|
| **54.1 tok/s** | **1215 tok/s** (matrix max) | 42% | **44 W** (matrix min) |

**Why it matters:** same active-param count as Qwen3.6-35B-A3B (3 B), same predicted ceiling — but
it measures **54 vs 75 tok/s** single-stream. The difference is **architecture**: the Mamba-2
state-space layers don't read weights the way the roofline (built for attention) assumes, so its
efficiency is lower *but* it scales to the **highest aggregate throughput (1215 tok/s) at the
lowest power (44 W)** of any model tested. A vivid case of [FINDINGS Discovery 3](../FINDINGS.md):
architecture matters beyond the roofline.

## Contents

- **[guides/](guides/00-overview-deployment-and-results.md)** · **[scripts/](scripts/launch.sh)** ·
  **[benchmarks/](benchmarks/README.md)** · **[sources/](sources/INDEX.md)**

[MIT](../../LICENSE) © 2026 Heitor Mocelin
