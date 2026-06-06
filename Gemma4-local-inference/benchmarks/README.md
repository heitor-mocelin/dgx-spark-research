# benchmarks/ — Gemma 4 on the GB10

**Cited** measurements (not run on this GX10 — it currently serves Qwen3.6, and deploying Gemma 4
needs docker access on the DGX). See [guide 05](../guides/05-benchmarks.md) for full context.

## Speed (DGX Spark / GB10)

| Model | Format | tok/s | Source |
|---|---|---:|---|
| 26B-A4B MoE | NVFP4 | **~48–52** | [g09](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) |
| 31B dense | NVFP4 | ~6.9 | [g09](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) |
| 31B dense | Q8 | ~6.4 | [g09](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) |

MoE ≈ **8× faster** than dense here. The 52 tok/s is **with** the `sm_121` Marlin fallback — likely
headroom once native FP4-MoE kernels land (guide 05).

## Quality

26B-A4B (3.8B active): AIME-2026 **88.3%**, GPQA-Diamond **82.3%**, LiveCodeBench-v6 **77.1%** —
within ~1–3 pts of the 31B dense ([g08](../sources/g08-run-locally-ollama-llamacpp-vllm.md),
[g14](../sources/g14-lushbinary-gemma4-benchmarks.md)).

## Reproduce

```bash
# runtime-agnostic OpenAI-API client (from the Qwen subproject); point URL/MODEL at your Gemma endpoint
python3 ../../AsusGx10-vllm-optimization/benchmarks/benchmark_sweep.py
```

Drop measured JSON/tables here as you collect them.
