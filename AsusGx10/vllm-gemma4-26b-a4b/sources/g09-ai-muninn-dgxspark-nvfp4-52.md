---
id: g09
title: "Gemma 4 26B in 16 GB at 52 tok/s — DGX Spark NVFP4"
url: "https://ai-muninn.com/en/blog/dgx-spark-gemma4-26b-nvfp4-52-toks"
publisher: "ai-muninn (blog)"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [dgx-spark, gb10, nvfp4, benchmark, moe]
---

DGX Spark · part 7

# Gemma 4 on DGX Spark: 52 tok/s in 16 GB with NVFP4 (Benchmark)

## ❯ cat --toc

- Plain-Language Version: What Is Gemma 4? What Does 52 tok/s Mean?
- Preface
- Phase 0: Why Not the 31B Dense?
- The Model: bg-digitalservices NVFP4
- Deployment: One Docker Command
- Benchmark Results
- vLLM vs Ollama on the Same Model
- Takeaways
- Where the time went
- Reusable diagnostics
- The general principle
- Deployment Checklist

TL;DR

Gemma 4 26B-A4B NVFP4 runs at **52 tok/s** on DGX Spark (GB10) via vLLM 0.19, using only 16.5 GB of model memory and leaving 82 GB for KV cache. The 31B dense variant is 7.5x slower — don't bother.

## Plain-Language Version: What Is Gemma 4? What Does 52 tok/s Mean?

Gemma 4 is an open-source AI model released by Google in April 2026. Open-source means anyone can download it and run it on their own computer — no payment to Google, no internet connection required. It can chat, write code, and answer questions like ChatGPT, but it lives on your machine.

The model comes in two variants: 31B (31 billion parameters, the full version) and 26B-A4B (26 billion parameters total, but only 3.8 billion active at a time). The latter is called MoE (Mixture of Experts) — think of a company with 260 employees where only 38 show up to handle each task while the rest stay idle. This means it moves far less data per inference step, so it runs much faster.

What does 52 tok/s (tokens per second) feel like? One token is roughly one Chinese character or half an English word. 52 tok/s means the AI generates about 52 characters per second — close to your reading speed, so conversations feel instant. By comparison, the 31B full version manages 7 tok/s on the same hardware. Seven characters per second. It types slower than you read.

I benchmarked both variants on an NVIDIA DGX Spark (a ~$3,000 desktop AI workstation). This article covers the full decision process, deployment steps, and every gotcha along the way.

## Preface

The wrong model at the right quantization is still the wrong model. A 31B dense model on a 273 GB/s memory bus will always lose to a 26B MoE with 4B active parameters, regardless of how cleverly you pack the weights.

This picks up where Part 6: 30W Power Safety Mode left off. The GX10 was stable on Qwen3.5-35B FP8 at 47 tok/s. Google released Gemma 4 on April 2, and vLLM 0.19 shipped the same day with SM121 NVFP4 fixes that had been broken since March. Time to test.

## Phase 0: Why Not the 31B Dense?

The natural first instinct was to try nvidia/Gemma-4-31B-IT-NVFP4 — the official NVIDIA quantized checkpoint. Community benchmarks on the NVIDIA Developer Forums killed that idea fast:

| Model | Format | tok/s on GB10 |
|---|---|---|
| Gemma 4 31B | BF16 | 3.7 |
| Gemma 4 31B | AWQ int4 | 10.6 |
| Gemma 4 31B | NVFP4 | 6.9 |
Gemma 4 26B-A4B | NVFP4 | ~48 (reported) |

The 31B is dense — all 31 billion parameters are active per token. On GB10's 273 GB/s memory bandwidth, that translates to roughly 7 tok/s regardless of quantization level. The quantization shrinks the model, but the bandwidth cost of reading all weights per token stays proportional.

The 26B-A4B is MoE: 26 billion total parameters, 3.8 billion active. That is the difference between a 7x memory read and a 1x memory read per token.

## The Model: bg-digitalservices NVFP4

The official NVIDIA NVFP4 checkpoint only exists for the 31B dense variant. For the 26B-A4B MoE, bg-digitalservices built a community quantization using a custom modelopt plugin — standard NVIDIA tooling doesn't support Gemma 4's fused 3D expert tensor format.

The numbers:

| Metric | BF16 | NVFP4 |
|---|---|---|
| Size on disk | 49 GB | 16.5 GB |
| Tokens/sec | 23.3 | 48.2 |
| TTFT | 97 ms | 53 ms |
| Quality retained | — | 97.6% |

The model ships with a `gemma4_patched.py`

that fixes vLLM's `expert_params_mapping`

— without it, NVFP4 scale keys (`.weight_scale`

, `.weight_scale_2`

, `.input_scale`

) fail to map to FusedMoE parameter names. This is tracked in vLLM issue #38912.

## Deployment: One Docker Command

Download the model:

```
huggingface-cli download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 \
--local-dir ~/models/gemma4-26b-a4b-nvfp4
```


Start the container:

```
docker run -d \
--name gemma4-nvfp4 \
--gpus all --ipc host --shm-size 64gb \
-p 8002:8000 \
-v ~/models/gemma4-26b-a4b-nvfp4:/models/gemma4 \
-v ~/models/gemma4-26b-a4b-nvfp4/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py \
vllm/vllm-openai:gemma4-cu130 \
--model /models/gemma4 \
--served-model-name gemma-4-26b \
--host 0.0.0.0 --port 8000 \
--quantization modelopt \
--kv-cache-dtype fp8 \
--max-model-len 131072 \
--gpu-memory-utilization 0.85 \
--moe-backend marlin \
--reasoning-parser gemma4 \
--enable-auto-tool-choice --tool-call-parser pythonic
```


The critical flags:

— SM121 has no native FP4 compute. Without this, CUTLASS MoE runs and produces garbage (NaN scale factors,`--moe-backend marlin`

`!!!!!`

output). Marlin decompresses FP4 weights to BF16 at runtime — slower than native W4A4 but correct.— the NVFP4 checkpoint was quantized with NVIDIA modelopt.`--quantization modelopt`

— maps into vLLM's model directory to fix the scale key mapping bug.`gemma4_patched.py`

mount— this is the correct image. The`vllm/vllm-openai:gemma4-cu130`

`gemma4`

tag (without`-cu130`

) is actually v0.18.2-dev and crashes with`RuntimeError: [FP4 gemm Runner] Failed to run cutlass FP4 gemm on sm120/sm121`

.

Startup takes about 90 seconds — 84 seconds for weight loading, then torch.compile warmup. The startup log should show:

```
Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM
Using 'MARLIN' NvFp4 MoE backend
Model loading took 15.76 GiB memory
Available KV cache memory: 81.8 GiB
GPU KV cache size: 714,768 tokens
```


If the log says `CUTLASS_FP4`

instead of `MARLIN`

for MoE, the `--moe-backend marlin`

flag was not picked up. Stop and fix.

## Benchmark Results

Five sequential runs at 800 tokens each:

| Run | Tokens | Time | tok/s |
|---|---|---|---|
| 1 | 800 | 15.48s | 51.6 |
| 2 | 800 | 15.52s | 51.5 |
| 3 | 800 | 15.51s | 51.5 |
| 4 | 800 | 15.48s | 51.6 |
| 5 | 800 | 15.48s | 51.6 |

Variance: ±0.1 tok/s. Rock solid.

Long output test (1633 tokens): **51.0 tok/s** — no degradation at length.

Concurrent load (3 parallel requests, 500 tokens each): **114.6 tok/s aggregate**, each request at ~38 tok/s.

### vLLM vs Ollama on the Same Model

Ollama has a `gemma4:26b`

GGUF (Q4_K_M, 17 GB). Same architecture, different runtime:

| vLLM NVFP4 | Ollama Q4_K_M | |
|---|---|---|
| tok/s | 52 | 40 |
| Model size | 16.5 GB | 17 GB |
| KV cache available | 82 GB | N/A |
| Concurrent requests | Yes (OpenAI API) | No |
| Tool calling | Yes | No |

vLLM wins by 30%. Both support vision. For the full head-to-head — latency, memory behavior, and when Ollama is actually the better call — see vLLM vs Ollama on the Same Model.

One Ollama gotcha worth documenting: if a previous vLLM container used the GPU (even if stopped), Ollama may load the model with only partial GPU allocation — `66% CPU / 34% GPU`

in `ollama ps`

. The fix is to fully unload before loading:

```
curl -s http://localhost:11434/api/generate \
-d '{"model":"gemma4:26b","keep_alive":0}'
```


## Takeaways

### Where the time went

The Phase 0 research. Not the deployment. The vLLM 0.19 SM121 fixes (#37725 for NVFP4 NaN, #38126 for DGX Spark) meant the deployment itself was straightforward. The time went into establishing that the 31B dense variant was not worth attempting, and that the community NVFP4 checkpoint existed and worked.

### Reusable diagnostics

- On bandwidth-constrained hardware (GB10's 273 GB/s), always pick MoE over dense. The total parameter count is irrelevant — active parameters determine speed.
`vllm/vllm-openai:gemma4`

and`vllm/vllm-openai:gemma4-cu130`

are different images with different vLLM versions. Tag naming does not imply one is a superset of the other. Always check`docker images`

to verify.- Ollama's CPU/GPU split is silent. A model reporting 40 tok/s in one session and 16 tok/s in the next is probably a split issue, not a model issue.

### The general principle

Do the arithmetic before the experiment. 31B parameters × 2 bytes (BF16) ÷ 273 GB/s = 227 ms per token = 4.4 tok/s theoretical max. No amount of quantization tricks changes the memory bandwidth equation for a dense model on a bandwidth-limited chip.

## Deployment Checklist

- Download
`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`

(~16.5 GB) - Pull
`vllm/vllm-openai:gemma4-cu130`

(not`gemma4`

) - Unload all Ollama models before starting vLLM (
`keep_alive:0`

) - Mount
`gemma4_patched.py`

into the container - Use
`--moe-backend marlin`

and`--quantization modelopt`

- Verify startup log shows
`MARLIN`

for MoE,`FLASHINFER_CUTLASS`

for dense - Test:
`curl http://<your-gx10-ip>:8002/v1/chat/completions`


Also in this series: Part 1: Ollama Benchmark — 8 Models · Part 2: vLLM + Qwen3.5 Setup · Part 5: FP8 KV Cache Repetition Bug · Part 6: 30W Power Safety Mode

## FAQ

- How fast is Gemma 4 26B-A4B NVFP4 on DGX Spark?
- 52 tok/s decode, stable across 5 sequential runs (±0.1 tok/s). Long outputs (1600+ tokens) show no speed degradation. Three concurrent requests achieve 114.6 tok/s aggregate throughput.
- Should I run Gemma 4 31B or 26B-A4B on DGX Spark?
- 26B-A4B, without question. The 31B dense variant runs at 6.9 tok/s on GB10 — bandwidth-bound at 273 GB/s. The 26B-A4B MoE (4B active parameters) runs at 52 tok/s with NVFP4 quantization. Same model family, 7.5x faster.
- Does Gemma 4 NVFP4 work on SM121 (GB10) with vLLM 0.19?
- Yes, but only with --moe-backend marlin. SM121 lacks native FP4 compute, so MoE layers must use the Marlin W4A16 backend. Dense layers use FLASHINFER_CUTLASS. The official vllm/vllm-openai:gemma4-cu130 image handles this correctly.
- What is the gemma4_patched.py file and do I need it?
- Yes. The community NVFP4 checkpoint (bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4) requires a patched gemma4.py to correctly map NVFP4 scale keys (.weight_scale, .weight_scale_2, .input_scale) in FusedMoE. Without it, weight loading fails. The patch ships with the model repo.
