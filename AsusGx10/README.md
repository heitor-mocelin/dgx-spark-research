# AsusGx10

The **ASUS Ascent GX10** — NVIDIA **GB10 Grace Blackwell**, **128 GB** unified LPDDR5x @
**273 GB/s**, 20-core Arm (10× Cortex-X925 + 10× A725), Blackwell 5th-gen Tensor Cores
(`sm_121`), DGX OS (Ubuntu ARM64). All work here targets this one device.

Decode on this box is **memory-bandwidth-bound**, so the recurring theme across every subproject
is the same: **fewer active parameters (MoE) + fewer bytes per weight (NVFP4/FP8) → more tok/s**,
then amortize the weight read across requests (batching).

## Subprojects

**Flagship local-inference guides** (full corpus + 6-guide series + scripts):

| Folder | What | Highlight |
|---|---|---|
| **[vllm-qwen3.6-35b-a3b/](vllm-qwen3.6-35b-a3b/)** | Optimizing vLLM serving of Qwen3.6-35B-A3B (NVFP4/FP8) | **measured** ~627 tok/s @ c32, ~951 peak; full guide series |
| **[vllm-gemma4-26b-a4b/](vllm-gemma4-26b-a4b/)** | Running Google Gemma 4 locally (vLLM/Ollama/llama.cpp) | 26B-A4B NVFP4 ~52 tok/s; runtime choice + deployment |

**Benchmark-tested models** (overview + NVFP4 recipe + measured results, from the [test matrix](FINDINGS.md)):

| Folder | Type | Single-stream | Note |
|---|---|---:|---|
| **[vllm-qwen3-32b/](vllm-qwen3-32b/)** | dense 32B | 11 tok/s | 91% of roofline — dense reference |
| **[vllm-llama-3.3-70b/](vllm-llama-3.3-70b/)** | dense 70B | 5.4 tok/s | **98%** — the roofline anchor |
| **[vllm-nemotron-3-nano-30b-a3b/](vllm-nemotron-3-nano-30b-a3b/)** | hybrid Mamba-MoE, 3B active | 54 tok/s | **1215 tok/s** peak @ 44 W — throughput champion |
| **[vllm-nemotron-3-super-120b-a12b/](vllm-nemotron-3-super-120b-a12b/)** | hybrid LatentMoE, 12B active | 15 tok/s | 120B that runs interactively; 1M ctx |
| **[vllm-qwen3-next-80b-a3b/](vllm-qwen3-next-80b-a3b/)** | hybrid Gated-DeltaNet MoE, 3B active | 35.5 tok/s | **27%** — matrix-min efficiency; latency-bound recurrence |

**Other:**

| Folder | What | Highlight |
|---|---|---|
| **[research-digests/](research-digests/)** | On-device, model-generated literature digests | major discoveries in efficient LLM inference (48 arXiv papers) |

> 🚀 **New here?** → **[Newcomer's Guide: Local Inference on a DGX Spark](getting-started.md)** —
> which model / runtime / quantization to pick, and the gotchas.
> ⚡ **Just want it running?** → **[`bootstrap/`](bootstrap/README.md)** — one command, bare device →
> serving endpoint (env-adaptive, fully logged). Then dive into the study below.

## 🔬 The test program — predict, measure, learn

We benchmarked **7 NVFP4 models** against a from-first-principles roofline. Read these in order:

1. **[testing-plan.md](testing-plan.md)** — the theory + the per-model prediction (the roofline).
2. **[FINDINGS.md](FINDINGS.md)** — what the measurements discovered (start here for conclusions).
3. **[benchmarks/](benchmarks/README.md)** — every number + reproduction scripts.

### Measured single-stream (NVFP4, 2026-06-06)

| Model | Active | Measured | Eff. vs ceiling |
|---|---:|---:|---:|
| Qwen3-Next-80B-A3B (MoE, DeltaNet) | 3.0B | 35.5 tok/s | **27%** (min) |
| Nemotron-3-Nano-30B-A3B (MoE) | 3.0B | 54 tok/s | 42% |
| Qwen3.6-35B-A3B (MoE) | 3.0B | **75 tok/s** | 58% |
| Nemotron-3-Super-120B-A12B (MoE) | 12B | 15 tok/s | 45% |
| Qwen3-32B (dense) | 32B | 11 tok/s | 91% |
| Gemma-4-31B (dense) | 31B | 7 tok/s | 54% |
| Llama-3.3-70B (dense) | 70B | 5 tok/s | **98%** |

## Shared findings across models (measured)

- **MoE beats dense, hard.** 3B-active MoEs run at **54–75 tok/s** single-stream; dense 31–70B models
  crawl at **5–11 tok/s**. On a bandwidth-bound box, *active* parameters decide speed.
- **Efficiency rises with active params** (42% → 98%) — the roofline is tight for big/dense models, a
  loose upper bound for small-active MoEs. (Our "constant ~55%" guess was refuted — see FINDINGS.)
- **Aggregate is power-capped** (96% GPU-util, 44–71 W), not bandwidth-capped.
- **The `sm_121` Marlin fallback is visible compute** (89–96% GPU-util even single-stream) — native
  FP4-MoE kernels (FlashInfer b12x / vLLM PR #40082) are the shared upside, per each subproject's guide 02.

## License

[MIT](../LICENSE) © 2026 Heitor Mocelin
