# 05 · Benchmarks — what Gemma 4 actually does locally

> Published GB10 numbers, how to read them, and how to reproduce them on your own endpoint. (We
> didn't run these on this GX10 — it currently serves Qwen3.6, and deploying Gemma 4 needs docker
> access there — so these are cited measurements; reproduce them with the client below once you
> stand Gemma 4 up.)

## Speed on the GB10 (DGX Spark)

| Model | Format | tok/s (GB10) | Notes | Source |
|---|---|---:|---|---|
| **26B-A4B MoE** | **NVFP4** | **~48–52** | 16.5 GB weights, ~82 GB KV; the recipe (guide `02`) | [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) |
| 31B dense | NVFP4 | ~6.9 | bandwidth-bound; ~7.5× slower | [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) |
| 31B dense | Q8 | ~6.4 | quantization doesn't help a dense model here | [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) |
| 31B dense | FP8 (single & dual-node TP2) | see thread | multi-Spark scaling | [[g12]](../sources/g12-nvidia-forum-gemma4-31b-fp8.md) |
| family | llama.cpp (ARM64/CUDA13) | suite | single-seq, context-scaling, multi-user, CoT | [[g10]](../sources/g10-shamily-gemma4-llama-dgx-spark.md) |

NVIDIA's own day-1 DGX Spark preliminary benchmarks add more datapoints across the family [[g11]](../sources/g11-nvidia-forum-gemma4-day1.md).

**The headline:** on this hardware, **26B-A4B NVFP4 is ~8× faster than the 31B dense** for
comparable quality — the MoE is the only sensible choice for interactive local use. For reference,
the sibling Qwen3.6-35B-A3B (also MoE, ~3B active) measured **~75 tok/s single-stream / ~627 tok/s
aggregate @ c32** on the same box ([Qwen benchmarks](../../AsusGx10-vllm-optimization/benchmarks/README.md))
— Gemma 4's 26B-A4B should land in a similar regime, modulo the Marlin caveat below.

## Quality

| Benchmark | 31B dense | 26B-A4B MoE | Source |
|---|---:|---:|---|
| AIME 2026 (math) | 89.2% | 88.3% | [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md)[[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md) |
| GPQA Diamond (science) | 84.3% | 82.3% | [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md)[[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md) |
| LiveCodeBench v6 (coding) | 80.0% | 77.1% | [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md)[[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md) |
| MMLU-Pro | 85.2% | ~ | [[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md) |

The 26B-A4B gives up only ~1–3 points to the dense 31B while running ~8× faster — **the efficiency
story in one row.** (Numbers are *with* thinking mode — guide `04`.)

## The open question: Marlin vs native FP4

The 52 tok/s is measured **with the Marlin fallback** (FP4→BF16 decompress at runtime), because the
GB10's `sm_121` native FP4 path wasn't fully engaged [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) — the same gap as the
[Qwen subproject](../../AsusGx10-vllm-optimization/guides/02-quantization-nvfp4-and-fp8.md). So there is
likely **headroom above 52 tok/s** once the native FP4-MoE kernels (FlashInfer b12x, vLLM PR #40082)
land for SM121. That delta is the single most valuable thing to measure.

## Reproduce on your endpoint

The benchmark client from the Qwen subproject is **runtime-agnostic** — it speaks the OpenAI API, so
it works against any Gemma 4 vLLM/Ollama/llama.cpp server:

```bash
# point it at your Gemma 4 endpoint + served model name
#   (edit URL/MODEL at the top of the script, or parameterize)
python3 ../../AsusGx10-vllm-optimization/benchmarks/benchmark_sweep.py
```

It reports the throughput-vs-latency curve (single-stream → saturation), which is exactly how to find
your operating point and to quantify thinking-mode / vision-token costs. Drop results into
[`../benchmarks/`](../benchmarks/) as you collect them.

## Sources cited

- [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) Run locally (quality numbers) • [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) NVFP4 on DGX Spark (speed)
- [[g10]](../sources/g10-shamily-gemma4-llama-dgx-spark.md) GB10 llama.cpp suite • [[g11]](../sources/g11-nvidia-forum-gemma4-day1.md) NVIDIA day-1 benchmarks
- [[g12]](../sources/g12-nvidia-forum-gemma4-31b-fp8.md) 31B FP8 single/dual-node • [[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md) Benchmarks (Lushbinary)
