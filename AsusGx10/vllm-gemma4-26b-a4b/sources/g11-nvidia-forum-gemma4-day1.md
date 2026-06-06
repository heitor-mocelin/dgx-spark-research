---
id: g11
title: "Gemma 4 Day-1 Inference on DGX Spark — Preliminary Benchmarks"
url: "https://forums.developer.nvidia.com/t/gemma-4-day-1-inference-on-nvidia-dgx-spark-preliminary-benchmarks/365503"
publisher: "NVIDIA Developer Forums"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [dgx-spark, gb10, benchmark]
---

Hello all, this is just basic result made with llm-benchy


⚠️ Preliminary results.These benchmarks were captured on April 2, 2026 — the same day Gemma 4 was released. Consider this a day-1 snapshot. Numbers will improve as vLLM kernels mature, quantization recipes are refined, and serving parameters are tuned.

## Hardware

**NVIDIA DGX Spark (GB10 Grace Blackwell)**

| Spec | Value |
|---|---|
| Architecture | Grace Blackwell Superchip (GB10) |
| Unified memory | 122 GB LPDDR5X |
| Memory bandwidth | ~273 GB/s |
| Platform | Ubuntu 24.04, aarch64 |
| CUDA | 13.0 (driver 580.142) |

The DGX Spark uses a fully unified memory architecture — CPU and GPU share the same LPDDR5X pool. This gives exceptional capacity (122 GB) at a fraction of the power of a datacenter GPU, but at lower bandwidth than HBM-based cards (~273 GB/s vs 3.35 TB/s on an H100 SXM). This has direct implications for decode throughput, discussed below.

## Docker Image

The vLLM team published an official Gemma 4 image **on the same day as the model release**:

```
vllm/vllm-openai:gemma4-cu130
```


This image ships native Gemma 4 support out of the box — no backporting required. It targets CUDA 13.0, which matches the DGX Spark driver stack perfectly, and is released under Apache 2.0.

Before this image was available, we had manually backported Gemma 4 support into

`vllm 0.18.1rc1`

by patching the model registry, reasoning parser registry, and rotary embedding module with files from vLLM main (PR #38826), plus upgrading`transformers`

to 5.5.0. The official image makes all of that unnecessary.

## Fixes Required

Even with the official image, two small adjustments were needed:

**1. --load-format fastsafetensors not available**

The

`vllm[fastsafetensors]`

optional dependency is not included in the `gemma4-cu130`

image. Replace with `--load-format safetensors`

.**2. --quantization awq conflicts with compressed-tensors format**


The

`cyankiwi`

AWQ quantized models use the `compressed-tensors`

format (llm-compressor). Specifying `--quantization awq`

explicitly causes a validation error:```
Quantization method specified in the model config (compressed-tensors) does not match
the quantization method specified in the `quantization` argument (awq).
```


The fix is to simply **omit the --quantization flag** — vLLM auto-detects

`compressed-tensors`

from the model’s `config.json`

.**3. Attention backend**

The official image automatically forces `TRITON_ATTN`

for all Gemma 4 models due to their heterogeneous head dimensions (local layers: `head_dim=256`

, global layers: `head_dim=512`

). No manual flag needed.

## Serving Configuration

All models were served with the following common parameters:

```
vllm serve <model> \
--enable-auto-tool-choice \
--tool-call-parser pythonic \
--reasoning-parser gemma4 \
--gpu-memory-utilization 0.70 \
--host 0.0.0.0 \
--port 30000 \
--kv-cache-dtype fp8 \
--load-format safetensors \
--enable-prefix-caching \
--enable-chunked-prefill \
--max-model-len 262144 \
--max-num-seqs 4 \
--max-num-batched-tokens 8192
```


**Context window:** 256K tokens (262144) — maximum supported by Gemma 4.

## Models Tested

| Model | Quantization | Format | Disk size | Notes |
|---|---|---|---|---|
`google/gemma-4-31B-it` |
bf16 | safetensors | ~62 GB | Dense, reference baseline |
`cyankiwi/gemma-4-31B-it-AWQ-8bit` |
int8 | compressed-tensors | ~33 GB | Community quant |
`cyankiwi/gemma-4-31B-it-AWQ-4bit` |
int4 | compressed-tensors | ~20 GB | Community quant |
`google/gemma-4-26B-A4B-it` |
bf16 | safetensors | ~49 GB | MoE: 26B total / 4B active |

## Benchmark Results

Benchmarked with llama-benchy v0.3.5.

**Conditions:** 3 runs per config, concurrency 1, depth 0, no prior context.

Raw result files: `results/`


### Prompt Processing throughput (t/s — higher is better)

| Model | pp128 | pp512 | pp2048 |
|---|---|---|---|
| 31B bf16 | 244 ± 46 | 757 ± 67 | 1066 ± 48 |
| 31B AWQ int8 | 267 ± 26 | 399 ± 33 | 430 ± 0 |
| 31B AWQ int4 | 545 ± 104 | 778 ± 39 | 810 ± 2 |
26B-A4B MoE |
429 ± 165 |
1299 ± 441 |
3105 ± 372 |

### Token Generation / Decode (t/s — higher is better)

| Model | tg128 | peak |
|---|---|---|
| 31B bf16 | 3.7 ± 0.1 | 4.0 |
| 31B AWQ int8 | 6.5 ± 0.1 | 7.0 |
| 31B AWQ int4 | 10.6 ± 0.0 | 11.0 |
26B-A4B MoE |
23.7 ± 0.0 |
24.0 |

### Time To First Response (ms — lower is better)

| Model | TTFR pp128 | TTFR pp512 | TTFR pp2048 |
|---|---|---|---|
| 31B bf16 | 547 ± 91 | 686 ± 64 | 1929 ± 89 |
| 31B AWQ int8 | 490 ± 51 | 1297 ± 108 | 4761 ± 2 |
| 31B AWQ int4 | 247 ± 46 |
664 ± 33 | 2533 ± 8 |
26B-A4B MoE |
371 ± 176 | 464 ± 197 |
672 ± 82 |

## Analysis

### Decode is bandwidth-bound on LPDDR5X

On the DGX Spark, single-user token generation is limited by memory bandwidth. Each generated token requires streaming all active model weights through memory once:

```
theoretical decode = memory_bandwidth / active_model_size_in_memory
31B bf16 : 273 GB/s ÷ 62 GB ≈ 4.4 t/s (measured: 3.7 t/s — 84% efficiency)
31B int8 : 273 GB/s ÷ 31 GB ≈ 8.8 t/s (measured: 6.5 t/s — 74% efficiency)
31B int4 : 273 GB/s ÷ 16 GB ≈ 17.0 t/s (measured: 10.6 t/s — 62% efficiency)
26B-A4B : 273 GB/s ÷ 8 GB ≈ 34.0 t/s (measured: 23.7 t/s — 70% efficiency)
```


The gap between int8/int4 theory and measurement reflects the overhead of `compressed-tensors`

dequantization on the TRITON_ATTN path — an area that will improve as vLLM’s kernel support for this format matures on aarch64/Blackwell.

### MoE structural advantage

The MoE model’s decode advantage is structural: even though all 49 GB of expert weights reside in GPU memory, only the **4B active parameters** are read per token — giving it **6.4× better decode throughput** than the dense bf16 baseline, and **2.2× better than AWQ int4**.

At longer contexts, the MoE prompt processing advantage is equally striking: 3105 t/s at pp2048 vs 1066 t/s for dense bf16 — nearly **3× faster prefill**, for the same reason.

### AWQ int8 prompt processing regression

The `compressed-tensors`

int8 path dequantizes weights before the matmul, which eliminates the compute-density advantage of native bf16. Prompt processing drops to 430 t/s at pp2048 vs 1066 t/s for bf16 — a 60% regression. For prompt-heavy workloads, int8 is the worst choice of the four.

### AWQ int4 is the best short-prompt dense model

At pp128, AWQ int4 delivers the lowest TTFR (247 ms) — less than half of bf16. For short conversational turns where the dense 31B quality is preferred over the MoE, int4 is the most responsive option.

## GPU Memory Usage

| Model | GPU memory used | Notes |
|---|---|---|
| 31B bf16 | ~63 GB | Weights ~62 GB + small KV cache |
| 31B AWQ int8 | ~85 GB | Weights ~31 GB + large KV cache |
| 31B AWQ int4 | ~85 GB | Weights ~16 GB + very large KV cache |
| 26B-A4B MoE | ~86 GB | All expert weights + large KV cache |

With `--gpu-memory-utilization 0.70`

(70% of 122 GB ≈ 85 GB), quantized and MoE models benefit from a proportionally much larger KV cache budget — up to ~70 GB for fp8 KV vs ~23 GB for bf16. This matters significantly for 256K context workloads.

## Conclusion

For Gemma 4 inference on a DGX Spark, the **26B-A4B MoE model is the clear winner** for interactive and agentic workloads: fastest decode (23.7 t/s), best prompt processing at long contexts (3105 t/s at pp2048), and competitive TTFR. The LPDDR5X unified memory architecture that constrains dense models actually favors the MoE design — only 4B active parameters need to be streamed per token.

For use cases where the full 31B dense model quality is required, **AWQ int4** is the best choice: 10.6 t/s decode (nearly 3× the bf16 baseline), lowest TTFR on short prompts (247 ms), and fits comfortably in 20 GB leaving ample room for KV cache at 256K context.

The `vllm/vllm-openai:gemma4-cu130`

image provides a clean, zero-patch deployment path on the DGX Spark — day-1 availability with CUDA 13.0 support is a strong signal for the community.

*Hardware: NVIDIA DGX Spark GB10 | Image: vllm/vllm-openai:gemma4-cu130 | Tool: llama-benchy v0.3.5 | Date: April 2, 2026*
