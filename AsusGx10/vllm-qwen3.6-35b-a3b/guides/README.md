# Guides — optimizing vLLM on the ASUS Ascent GX10 (GB10)

A layered series: each guide opens with a **plain-language on-ramp**, then a **deep dive**, then
an **"On your GX10"** section tying it to the real device, and a **cited sources** list. Read
top-to-bottom for a full mental model, or jump to the topic you're tuning.

## Reading order

| # | Guide | What you'll get |
|---|-------|-----------------|
| 00 | [Overview & hardware](00-overview-and-hardware.md) | The mental model: 273 GB/s, unified memory, why generation is bandwidth-bound, the `sm_121` caveat. **Start here.** |
| 01 | [Throughput & batching](01-throughput-and-batching.md) | The biggest lever — continuous batching, chunked prefill, prefix caching, `max-num-seqs` / `max-num-batched-tokens`, KV-budget math. |
| 02 | [Quantization — NVFP4 & FP8](02-quantization-nvfp4-and-fp8.md) | How NVFP4 works, FP8/NVFP4 KV cache, and the full *"GPU lacks native FP4 → Marlin"* story for GB10. |
| 03 | [Latency & speculative decoding](03-latency-and-speculative-decoding.md) | TTFT vs ITL, and MTP speculative decoding for low-concurrency snappiness. |
| 04 | [Tool-calling & context](04-tool-calling-and-context.md) | The `qwen3_coder` parser, reasoning mode, long-context KV math and YaRN. |
| 05 | [Benchmarking & the tuning loop](05-benchmarking-and-the-tuning-loop.md) | `vllm bench serve`, a 10-experiment plan, and how to measure everything above. |

## The through-line

Two levers explain the whole series:

1. **Shrink the bytes per token** → quantization (guide `02`).
2. **Amortize the weight read across requests** → batching (guide `01`).

Latency (`03`), tool-calling/context (`04`), and measurement (`05`) are how you apply, balance,
and verify those two on a single GB10 serving **Qwen3.6-35B-A3B** in NVFP4.

## Status & honesty notes

- **Grounded in 20 cited sources** ([`../sources/`](../sources/INDEX.md)) plus this device's real
  config. Every claim links to a source or is flagged as an estimate.
- **The GX10 is read-only** during drafting, so device-specific performance numbers are marked
  **"to measure in Phase 3."** Each becomes a concrete `vllm bench` experiment in guide `05` and a
  parameter in the (forthcoming) [`../scripts/`](../scripts/) (`launch` / `benchmark` / `tune` /
  `rollback`).
- **Headline open question:** how much throughput the `sm_121` Marlin fallback costs vs the native
  FP4-MoE kernels landing in vLLM (guides `02`, `05`).
