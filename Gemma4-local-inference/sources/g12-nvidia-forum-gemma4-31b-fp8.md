---
id: g12
title: "Gemma 4 31B on DGX Spark: Runtime FP8 Benchmarks (Single & Dual Node TP=2)"
url: "https://forums.developer.nvidia.com/t/gemma-4-31b-on-dgx-spark-runtime-fp8-benchmarks-single-dual-node-tp-2/365814"
publisher: "NVIDIA Developer Forums"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [dgx-spark, fp8, dense, benchmark, tensor-parallel]
---

Gemma 4 31B dense benchmarks on DGX Spark with runtime FP8 quantization. Includes single-node (TP=1), dual-node (TP=2), multi-user concurrency, tool calling validation, and multimodal vision. Getting Gemma 4 running on vLLM required working around several issues — documenting everything here for anyone else attempting this.

## Setup

**Hardware:**NVIDIA DGX Spark (1x GB10, 128GB unified memory)**Model:**`google/gemma-4-31B-it`

(BF16 base, runtime dynamic FP8)**vLLM:**0.19.1rc1.dev31+ga88ce94bb (April 4 build, includes PR #38826)**Container:**`vllm-node-tf5`

(transformers 5 baked in)**Quantization:**`--quantization fp8 --kv-cache-dtype fp8`

**Context:**65,536 tokens |**GPU mem utilization:**0.85 |**TP:**1 and 2**Attention:**Triton (forced — Gemma 4 has heterogeneous head dims: 256 local / 512 global)**Compilation:**torch.compile enabled,`VLLM_DISABLE_COMPILE_CACHE=1`

(see workarounds below)**Mods:**`fix-gemma4-reasoning`

(#38855),`fix-gemma4-tool-parser`

(PR #38909)**Benchmark:**llama-benchy 0.3.5, 3 runs per config

### Workarounds required

-
**transformers >= 5.0**— The`gemma4`

architecture isn’t in transformers 4.x. Using`vllm-node-tf5`

container which has transformers 5 baked in (avoids pip dependency conflicts from runtime upgrade). -
**Runtime FP8 instead of pre-quantized checkpoints**— Pre-quantized FP8 models (e.g.`protoLabsAI/gemma-4-31B-it-FP8`

) fail with`KeyError: 'layers.0.mlp.down_proj.weight_scale'`

in`gemma4.py`

load_weights. Same class of bug as vllm #38912. Using the BF16 base model with`--quantization fp8`

works fine — the GB10’s 128GB unified memory handles the ~59GB BF16 load before online quantization kicks in. -
— torch.compile crashes with`VLLM_DISABLE_COMPILE_CACHE=1`

`_pickle.PicklingError: Can't pickle <function launcher>`

during AOTAutogradCache save in the multiprocess EngineCore. Disabling the compile cache bypasses this while keeping torch.compile + CUDA graphs active. Cold start goes from ~1 min to ~3 min. -
**Reasoning parser patch**— vllm #38855:`reasoning_content`

is always null because`skip_special_tokens=True`

strips the`<|channel>`

delimiters before the parser sees them. Patched via a mod that re-decodes with`skip_special_tokens=False`

when reasoning boundary tokens are present.

## Benchmarks — Single User (c=1)

```
llama-benchy --base-url http://localhost:8000/v1 --model google/gemma-4-31B-it \
--pp 128 512 2048 --tg 32 128 --depth 0 --runs 3 --concurrency 1
```


| model | test | t/s | peak t/s | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|---|---|---|---|---|---|---|
| google/gemma-4-31B-it | pp128 | 593.84 ± 95.22 | 228.45 ± 40.44 | 223.71 ± 40.44 | 228.53 ± 40.46 | |
| google/gemma-4-31B-it | tg32 | 6.93 ± 0.01 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp128 | 616.91 ± 98.45 | 219.81 ± 37.38 | 215.07 ± 37.38 | 219.88 ± 37.37 | |
| google/gemma-4-31B-it | tg128 | 6.91 ± 0.01 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp512 | 1569.74 ± 84.61 | 332.28 ± 17.59 | 327.54 ± 17.59 | 332.35 ± 17.59 | |
| google/gemma-4-31B-it | tg32 | 6.86 ± 0.01 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp512 | 1558.11 ± 10.33 | 334.00 ± 2.19 | 329.26 ± 2.19 | 334.08 ± 2.18 | |
| google/gemma-4-31B-it | tg128 | 6.84 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp2048 | 1466.99 ± 2.91 | 1401.48 ± 2.78 | 1396.74 ± 2.78 | 1401.57 ± 2.79 | |
| google/gemma-4-31B-it | tg32 | 6.79 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp2048 | 1478.21 ± 8.49 | 1391.15 ± 8.31 | 1386.41 ± 8.31 | 1391.24 ± 8.31 | |
| google/gemma-4-31B-it | tg128 | 6.79 ± 0.00 | 7.00 ± 0.00 |

## Benchmarks — Multi User (c=2, c=4)

```
llama-benchy --base-url http://localhost:8000/v1 --model google/gemma-4-31B-it \
--pp 128 512 2048 --tg 32 128 --depth 0 --runs 3 --concurrency 2 4
```


| model | test | t/s (total) | t/s (req) | peak t/s | peak t/s (req) | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|---|---|---|---|---|---|---|---|---|
| google/gemma-4-31B-it | pp128 (c2) | 1142.43 ± 15.71 | 575.81 ± 8.37 | 225.78 ± 3.14 | 224.07 ± 3.14 | 225.81 ± 3.14 | ||
| google/gemma-4-31B-it | tg32 (c2) | 13.93 ± 0.33 | 7.04 ± 0.09 | 16.00 ± 0.00 | 8.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp128 (c4) | 1189.79 ± 2.08 | 427.42 ± 129.01 | 333.62 ± 99.76 | 331.92 ± 99.76 | 333.66 ± 99.76 | ||
| google/gemma-4-31B-it | tg32 (c4) | 26.82 ± 0.01 | 6.96 ± 0.05 | 32.00 ± 0.00 | 8.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp128 (c2) | 1142.08 ± 5.12 | 575.70 ± 2.86 | 225.49 ± 1.23 | 223.79 ± 1.23 | 225.54 ± 1.22 | ||
| google/gemma-4-31B-it | tg128 (c2) | 12.95 ± 1.57 | 7.01 ± 0.03 | 16.00 ± 0.00 | 8.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp128 (c4) | 1504.25 ± 227.97 | 420.35 ± 72.59 | 318.28 ± 58.91 | 316.58 ± 58.91 | 318.31 ± 58.92 | ||
| google/gemma-4-31B-it | tg128 (c4) | 27.43 ± 0.31 | 6.94 ± 0.02 | 29.33 ± 1.89 | 7.33 ± 0.47 | |||
| google/gemma-4-31B-it | pp512 (c2) | 1904.91 ± 13.54 | 955.71 ± 6.80 | 538.50 ± 3.84 | 536.80 ± 3.84 | 538.55 ± 3.84 | ||
| google/gemma-4-31B-it | tg32 (c2) | 13.90 ± 0.01 | 6.95 ± 0.00 | 14.00 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp512 (c4) | 1770.15 ± 86.47 | 535.22 ± 82.74 | 983.92 ± 154.95 | 982.22 ± 154.95 | 983.94 ± 154.95 | ||
| google/gemma-4-31B-it | tg32 (c4) | 23.95 ± 1.81 | 6.63 ± 0.27 | 28.00 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp512 (c2) | 1906.87 ± 7.96 | 956.67 ± 4.06 | 537.60 ± 2.16 | 535.90 ± 2.16 | 537.64 ± 2.17 | ||
| google/gemma-4-31B-it | tg128 (c2) | 11.92 ± 2.78 | 6.88 ± 0.11 | 14.00 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp512 (c4) | 1846.54 ± 45.65 | 708.08 ± 387.71 | 895.61 ± 319.23 | 893.91 ± 319.23 | 895.63 ± 319.22 | ||
| google/gemma-4-31B-it | tg128 (c4) | 24.89 ± 1.06 | 6.75 ± 0.08 | 28.00 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp2048 (c2) | 1573.33 ± 5.19 | 943.33 ± 309.04 | 2332.16 ± 485.54 | 2330.46 ± 485.54 | 2332.21 ± 485.54 | ||
| google/gemma-4-31B-it | tg32 (c2) | 12.31 ± 1.26 | 6.60 ± 0.47 | 14.00 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp2048 (c4) | 1613.51 ± 10.79 | 549.32 ± 169.50 | 4050.59 ± 1049.58 | 4048.89 ± 1049.58 | 4050.62 ± 1049.56 | ||
| google/gemma-4-31B-it | tg32 (c4) | 15.75 ± 1.79 | 5.62 ± 0.92 | 26.33 ± 2.36 | 7.26 ± 0.85 | |||
| google/gemma-4-31B-it | pp2048 (c2) | 1574.70 ± 3.60 | 811.10 ± 23.39 | 2529.78 ± 72.68 | 2528.08 ± 72.68 | 2529.86 ± 72.67 | ||
| google/gemma-4-31B-it | tg128 (c2) | 12.67 ± 1.25 | 6.81 ± 0.03 | 14.00 ± 0.00 | 7.00 ± 0.00 | |||
| google/gemma-4-31B-it | pp2048 (c4) | 1617.20 ± 8.55 | 615.15 ± 336.70 | 3946.83 ± 1229.94 | 3945.13 ± 1229.94 | 3946.86 ± 1229.94 | ||
| google/gemma-4-31B-it | tg128 (c4) | 22.90 ± 0.57 | 6.33 ± 0.32 | 28.00 ± 0.00 | 7.00 ± 0.00 |

## torch.compile vs enforce-eager (c=1)

Out of the box, torch.compile crashes with a pickling error in AOTAutogradCache. Setting `VLLM_DISABLE_COMPILE_CACHE=1`

bypasses the cache serialization while keeping compilation + CUDA graphs active. The performance difference is meaningful:

| Metric | enforce-eager | compiled | Delta |
|---|---|---|---|
| tg32 decode (t/s) | 6.88 | 6.93 | +0.7% |
| tg128 decode (t/s) | 6.24 | 6.91 | +10.7% |
| pp128 prefill (t/s) | 496 | 605 | +22.0% |
| pp512 prefill (t/s) | 1,321 | 1,564 | +18.4% |
| pp2048 prefill (t/s) | 1,416 | 1,473 | +4.0% |
| TTFT pp128 (ms) | 265 | 224 | -15.5% |
| TTFT pp512 (ms) | 391 | 333 | -14.8% |

Compilation gives +10-22% on prefill and longer decode. Short token generation (tg32) is fully memory-bandwidth bound and doesn’t benefit. The tradeoff is ~3 min cold start vs ~1 min (compilation happens fresh each launch since the cache is disabled).

## Summary — Single Node (TP=1)

| Single user | 4 concurrent users | |
|---|---|---|
Decode (tg128) |
~6.9 tok/s | ~27 tok/s aggregate |
Decode (tg32) |
~6.9 tok/s | ~27 tok/s aggregate |
Prefill (pp512) |
~1,560 tok/s | ~1,850 tok/s |
TTFT (pp512) |
~333 ms | ~940 ms |

The model is solidly memory-bandwidth bound on the GB10 — per-request decode throughput stays flat (~6.8-7.0 t/s) regardless of concurrency. Aggregate scales near-linearly up to c=4. Prefill peaks at ~1,900 t/s at c=2, limited by the forced Triton attention backend (Gemma 4’s mixed head dimensions: 256 local / 512 global prevent FlashInfer).

## Dual Spark — TP=2 Benchmarks (NEW)

Using `vllm-node-tf5`

container with `--distributed-executor-backend ray`

and TP=2 across two DGX Sparks connected via InfiniBand.

```
llama-benchy --base-url http://localhost:8000/v1 --model google/gemma-4-31B-it \
--pp 2048 --tg 32 128 --depth 0 4096 8192 16384 --runs 3 --concurrency 1
```


### Token Generation (single user)

| Context depth | TP=1 (1 Spark) | TP=2 (2 Sparks) | Speedup |
|---|---|---|---|
| d=0, tg128 | 6.5 ± 0.0 t/s | 11.2 ± 0.7 t/s | 1.7× |
| d=4096, tg128 | 6.5 ± 0.0 t/s | 10.3 ± 0.9 t/s | 1.6× |
| d=8192, tg128 | 6.4 ± 0.0 t/s | 10.1 ± 1.0 t/s | 1.6× |
| d=16384, tg128 | 6.3 ± 0.0 t/s | 11.0 ± 0.3 t/s | 1.7× |

### Prompt Processing

| Context depth | TP=1 | TP=2 | Speedup |
|---|---|---|---|
| pp2048, d=0 | 1,890 ± 132 t/s | 2,608 ± 69 t/s | 1.4× |
| pp2048, d=8192 | 1,255 ± 1 t/s | 2,010 ± 3 t/s | 1.6× |
| pp2048, d=16384 | 886 ± 4 t/s | 1,488 ± 2 t/s | 1.7× |

### Time to First Response (lower = better)

| Context depth | TP=1 | TP=2 | Speedup |
|---|---|---|---|
| d=0 | 1,300 ms | 906 ms | 1.4× |
| d=8192 | 8,926 ms | 5,561 ms | 1.6× |
| d=16384 | 22,669 ms | 13,432 ms | 1.7× |

### TP=2 VRAM usage

| Node | VRAM used |
|---|---|
| Head (Spark 1) | 105,721 MB |
| Worker (Spark 2) | 103,161 MB |

**Takeaway:** TP=2 gives a consistent ~1.7× speedup across the board. Not a full 2× due to inter-node all-reduce overhead per layer via InfiniBand, but the improvement is meaningful — especially at longer contexts where TTFR drops from 22.7s to 13.4s at 16K depth. On GB10’s shared memory, `nvidia-smi --query-compute-apps=used_gpu_memory`

is required for accurate VRAM readings (standard `memory.used`

returns N/A).

### Comparison across all Gemma 4 variants on Spark

| Model | Quant | Nodes | tg t/s (c=1) |
|---|---|---|---|
| Gemma 4 31B dense | BF16 | 1 | ~3.7 |
| Gemma 4 31B dense | FP8 runtime | 1 | ~6.5 |
| Gemma 4 31B dense | NVFP4 | 1 | ~6.9 |
Gemma 4 31B dense |
FP8 runtime |
2 (TP=2) |
~11.2 |
| Gemma 4 26B-A4B MoE | FP8 | 1 | ~38–40 |

Runtime FP8 single-node is within 6% of NVFP4. Dual-node TP=2 pushes past the single-node ceiling. The 26B-A4B MoE remains ~3.5× faster than dense 31B at TP=2 — expected since only 4B of 26B params are active per token.

## Multimodal Vision Test

Gemma 4 is a multimodal model (text + image + video). Vision works out of the box via the standard OpenAI image_url content type — no extra flags needed.

Tested with a synthetic scene (house, tree, sun, sky):

```
"content": [
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
{"type": "text", "text": "What is this a picture of? List everything you see."}
]
```


This is a simple, stylized illustration of a house in a landscape.


A house:It has a red body and a red triangular roof.Windows:Two light-blue square windows on the house.A door:A black rectangular door.A tree:A brown trunk with a green circular top.The ground:A solid green field.The sky:A light blue background.The sun:A yellow circle in the upper right corner.

Correctly identified all elements including shapes, colors, and spatial relationships. TTFT for vision requests is comparable to text-only (~300ms for small images).

## Tool Calling & Agentic Workflows

With `fix-gemma4-reasoning`

+ `fix-gemma4-tool-parser`

mods applied (`vllm-node-tf5`

container):

| Test | Streaming | Result |
|---|---|---|
| Single tool call | No | ✅ Valid JSON arguments |
| Single tool call | Yes | ✅ Valid JSON streamed |
| Multi-tool call (parallel) | No | ✅ Both tools called correctly |
| Agentic multi-step (tool → result → synthesis) | No | ✅ Correct answer from tool data |
| Agentic multi-step | Yes | ✅ Streamed correctly |
| HTML in streaming tool args | Yes | ✅ No `<<` duplication (PR #38909 fix) |
| Tool selection from 3 options | No | ✅ Correct tool chosen |
| Full agentic loop (2 tools → results → answer) | Yes | ✅ Complete workflow |

**Known limitation:** Non-streaming `maxSteps`

agent loops (e.g. Vercel AI SDK `generateText`

with `maxSteps`

) may exit after 1 step — vLLM returns `stop`

instead of continuing tool calls (#39043). **Streaming mode ( streamText) is recommended for agentic workflows.**

## Open vLLM Issues Affecting Gemma 4

| Issue | Impact | Status/Workaround |
|---|---|---|
| #38912 | Pre-quantized FP8 checkpoints fail to load | Use BF16 + `--quantization fp8` |
| #38946 | Streaming tool calls produce invalid JSON | Fixed via PR #38992 (merged) |
| #38855 | `reasoning_content` always null |
Fixed via `fix-gemma4-reasoning` mod |
| #38909 | Streaming HTML duplication in tool args | Fixed via `fix-gemma4-tool-parser` mod |
| #39043 | Tool call tags leak in non-streaming multi-turn | Use streaming for agentic workflows |
| AOTAutogradCache pickle | torch.compile cache crashes on aarch64 | `VLLM_DISABLE_COMPILE_CACHE=1` |

## Reproduction

Recipe and mods available at spark-vllm-docker (PR #165 to upstream):

```
# Single node (TP=1)
HF_HOME=/mnt/tank/models ./run-recipe.sh gemma-4-31b-fp8 --setup --solo
# Dual node (TP=2)
HF_HOME=/mnt/tank/models ./run-recipe.sh gemma-4-31b-fp8 --setup -n spark1,spark2 --tp 2
# Benchmark
uvx llama-benchy --base-url http://localhost:8000/v1 \
--model google/gemma-4-31B-it \
--pp 128 512 2048 --tg 32 128 --depth 0 4096 8192 16384 \
--runs 3 --concurrency 1 2 4 --latency-mode generation
```
