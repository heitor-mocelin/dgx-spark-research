# Guides — running Gemma 4 locally (GX10 / GB10 reference)

A layered series (plain-language on-ramp → deep dive → cited sources), focused on **local
inference**. Read top-to-bottom, or jump to your task.

| # | Guide | What you'll get |
|---|-------|-----------------|
| 00 | [Overview & hardware](00-overview-gemma4-and-hardware.md) | Gemma 4 variants, the MoE/hybrid-attention/PLE/shared-KV architecture, why 26B-A4B wins locally, who built it. **Start here.** |
| 01 | [Choosing a runtime](01-choosing-a-runtime.md) | Ollama vs llama.cpp vs vLLM — pick by use case + hardware. |
| 02 | [GX10 deployment (NVFP4 on vLLM)](02-running-on-the-gx10-vllm-nvfp4.md) | The concrete 26B-A4B NVFP4 recipe, ~52 tok/s, and the `sm_121`/Marlin caveat. |
| 03 | [Quantization & memory](03-quantization-and-memory.md) | GGUF vs NVFP4/FP8, footprints per variant, KV-cache math. |
| 04 | [Multimodal, thinking & tool-calling](04-multimodal-thinking-and-tool-calling.md) | Vision token budget, thinking mode, the custom tool protocol (+ runtime caveats). |
| 05 | [Benchmarks](05-benchmarks.md) | Published GB10 speed + quality numbers, and how to reproduce. |

## The through-line

On a bandwidth-bound box, **active parameters and bytes-per-weight decide local speed.** So:

1. **Pick the MoE** (26B-A4B, 3.8B active) → ~8× faster than the 31B dense on the GB10.
2. **Quantize it** (NVFP4 on vLLM / Q4_K_M on llama.cpp) → ~14–16 GB, leaving the 128 GB box's
   memory for long-context KV cache.
3. **Serve it on vLLM** for throughput, reliable tool-calling, and full multimodal.

It's the same lesson as the [Qwen3.6 subproject](../../vllm-qwen3.6-35b-a3b/) — Gemma 4
26B-A4B is its close analog, down to the shared `sm_121` Marlin-vs-native FP4 question.

## Honesty notes

- Grounded in [14 cited sources](../sources/INDEX.md) + the GX10's measured Qwen baseline. Community
  blogs are flagged; verify commands against the official model card (g01) and vLLM recipe (g05).
- GB10 Gemma 4 numbers are **cited, not run here** (the GX10 currently serves Qwen3.6, and deploying
  Gemma needs docker access on the DGX). Guide `05` has the reproduce path.
- Gemma 4 weights are **gated** (accept the license) and **Apache-2.0** licensed for use.
