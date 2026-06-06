# 00 · Nemotron-3-Super-120B-A12B on the GX10 — overview, deployment & measured results

> The headline is almost a magic trick: a **120-billion-parameter** model running at a usable
> **~15 tok/s** on a desktop. This guide explains how (it activates only 12 B per token), what it
> measured, and why it's the clearest demonstration of the whole project's thesis — *active
> parameters, not total, decide local speed.*

## Plain-language on-ramp

A 120 B *dense* model is a non-starter on the GX10: you'd read ~67 GB of weights per token and crawl
at ~4 tok/s. Nemotron-Super is a **Mixture-of-Experts** — of its 120 B parameters, only ~12 B fire
for any given token. So it reads ~6.75 GB/token and runs ~15 tok/s, while still *holding all 120 B
in memory* (75 GB in NVFP4) and a **1-million-token context**. It's the biggest "brain" that still
behaves interactively on this box — and it carries the **highest long-context score** of any open
model on the RULER@1M benchmark.

## Architecture (the deep dive)

NVIDIA calls it **LatentMoE** [[s1]](../sources/INDEX.md):

- **Hybrid Mamba-2 + MoE + Attention**, like Nemotron-Nano but scaled up — Mamba-2 state-space layers
  keep long-context cost bounded.
- **MoE routing in a *projected latent* dimension** (not the full model dimension) — cheaper routing
  per token, part of why 12 B-active is so efficient.
- **Multi-Token Prediction (MTP) heads** with shared weights → **native speculative decoding without
  a separate draft model.** (Our greedy benchmark didn't use it; with MTP on, real-world latency
  should improve — see [Qwen3.6 guide 03](../../vllm-qwen3.6-35b-a3b/guides/03-latency-and-speculative-decoding.md)
  for the MTP pattern.)
- **1 M context**; RULER@1M = **91.75** (vs ~22 for GPT-OSS-120B) — genuinely usable long context.

## NVFP4 deployment

~75 GB in NVFP4 — the largest footprint we deployed, still leaving ~40 GB for KV at
`--gpu-memory-utilization 0.85`. Standard recipe ([`scripts/launch.sh`](../scripts/launch.sh)) +
`--trust-remote-code`. Deployed cleanly from `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`.

## Measured results (2026-06-06)

| Concurrency | 1 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate tok/s | 14.7 | 48.4 | 90.2 | 151.7 | 227.4 | 245.5 | **327.0** | 326.2 |

- **Single-stream: 14.6 tok/s** (TTFT 265 ms, ITL 67 ms).
- **Predicted ceiling:** 40 peak, 32 realistic → **45% efficiency** (hybrid-architecture discount,
  like Nemotron-Nano).
- **Telemetry:** single-stream GPU 89% / 36 W; saturated GPU 96% / 56 W.

### Reading the result (the "why")

- **The MoE thesis, proven at the top end.** 12 B active → ~15 tok/s for a 120 B model; a 120 B dense
  would be ~4× slower *and* tighter on memory. **Active parameters decide local speed** — the entire
  project's claim, at its most dramatic.
- **Same hybrid discount as Nemotron-Nano** (45% vs 42%) — the Mamba-2 layers sit below the
  attention-roofline, consistent across both Nemotrons ([FINDINGS Discovery 3](../../FINDINGS.md)).
- **Aggregate is modest (327)** because 12 B-active is real compute per token under batching — more
  than a 3 B MoE, so it saturates the power/compute budget sooner.
- **Untapped upside:** MTP speculative decoding (latency) + 1 M context (capability) are reasons to
  pick this model that our throughput benchmark didn't capture.

## Where it fits on the GX10

The pick when you want **maximum capability** — frontier-class reasoning + class-leading 1 M context
— and can accept ~15 tok/s (or batch to ~327). It's the "smartest model that still fits and runs"
on the box. For pure speed, the 3 B-active MoEs win. See [FINDINGS](../../FINDINGS.md).

## Sources

[sources/INDEX.md](../sources/INDEX.md) — Nemotron-3 Super technical report + model card.
