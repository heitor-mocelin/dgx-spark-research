# 00 · Qwen3-32B on the GX10 — overview, deployment & measured results

> A dense 32 B model on a bandwidth-bound box is a *deliberate* choice: you trade speed for the
> simplicity and quality of a dense model. This guide explains the trade, how to serve it in NVFP4,
> and exactly what it measured — including why it lands at **91% of its theoretical ceiling**.

## Plain-language on-ramp

Qwen3-32B is a **dense** model: every one of its 32.8 B parameters participates in every token.
That's the opposite of a Mixture-of-Experts model (which fires only a slice). On the GX10, where
decode speed is set by *how many weight-bytes you read per token*, "dense" means "slow but
predictable" — you read all 32 B every step. The upside is a simpler, uniformly-strong model; the
downside is ~11 tok/s, not the ~75 of a 3 B-active MoE.

## Architecture (the deep dive)

A conventional, well-tuned transformer [[s1]](../sources/INDEX.md):

- **64 layers**, **Grouped-Query Attention** (64 query heads sharing **8 KV heads**) — GQA shrinks
  the KV cache ~8× vs full multi-head, which is why long context is affordable.
- **SwiGLU** FFN, **RoPE** positions, **RMSNorm** pre-norm — the standard modern stack.
- **Unified thinking / non-thinking**: one model, switch reasoning on for hard problems, off for
  speed (the AIME-class scores are *with* thinking).
- **Context**: 32K native, 131K with YaRN RoPE scaling.

Because it's dense and transformer-standard, Qwen3-32B is the **cleanest test of the roofline** —
no MoE routing, no exotic layers to muddy the weight-read accounting.

## NVFP4 deployment on the GX10

NVFP4 shrinks the 32.8 B weights to **~18 GB**, leaving ~100 GB for KV cache. The vLLM serve
command (see [`scripts/launch.sh`](../scripts/launch.sh)) mirrors the production Qwen3.6 recipe —
`--quantization modelopt`, FP8 KV, the `sm_121` env flags — just pointed at the Qwen3-32B NVFP4
checkpoint. It deploys cleanly (unlike the Gemma MoE checkpoint), no patch needed.

## Measured results (2026-06-06)

| Concurrency | 1 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate tok/s | 11.0 | 44.6 | 88.2 | 173.7 | 333.9 | 483.9 | 880.7 | **883.8** |

- **Single-stream: 11.0 tok/s** (TTFT 102 ms, ITL 90 ms), std 0.08 — deterministic.
- **Predicted ceiling:** 15.2 tok/s peak, **12.1 realistic** → measured **91%** of realistic.
- **Telemetry:** single-stream GPU 89% / 37 W; saturated GPU 96% / 62 W.

### Reading the result (the "why")

- **91% efficiency is near the roofline.** The ~18 GB weight-read per token is so large that the
  fixed per-token overheads (KV, attention, scheduler, Marlin decompress) are a *small* fraction —
  so the measured number almost *is* the bandwidth ceiling. This is [FINDINGS Discovery
  2](../../FINDINGS.md) in action: **efficiency rises with active params.**
- **Near-linear batching up to ~c96**, then it flattens (880 → 884) — the power/compute cap
  ([FINDINGS Discovery 4](../../FINDINGS.md)). Peak ~884 tok/s aggregate at 62 W.
- **The dense penalty is the point:** 11 tok/s single-stream vs Qwen3.6-35B-A3B's 75 — same vendor,
  ~same total size, but the MoE's 3 B active makes it ~7× faster. **On the GX10, prefer the MoE
  unless you specifically need the dense model's uniformity.**

## Where it fits on the GX10

Use Qwen3-32B when you want a **dense, single-mode, Apache-licensed** model and can tolerate
~11 tok/s interactive (or serve it batched at ~880 tok/s aggregate). For latency-sensitive local
use, the MoE models win decisively. Cross-model context: [FINDINGS](../../FINDINGS.md).

## Sources

See [sources/INDEX.md](../sources/INDEX.md) — Qwen3 technical report + model card.
