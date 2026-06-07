# Qwen3-4B-Instruct-2507 on DGX Spark / GB10 — vLLM research digest

## Checkpoint
- **Best vLLM-native NVFP4:** `kaitchup/Qwen3-4B-Instruct-2507-NVFP4` (or identical `llmat/...`) —
  NVFP4 via **LLM Compressor**, **~3.6 GB**, `compressed-tensors` format, stated vLLM v0.11+ compatible
  (tested on RTX 5090). Safer for vLLM than the ModelOpt/TRT-LLM-oriented repo. [2][3]
- Smaller alt: `OPENZEKA/Qwen3-4B-Instruct-2507-NVFP4` — NVFP4 W4A4 via ModelOpt, **~2.82 GB**, but
  benchmarked via TensorRT-LLM, not vLLM. [1]
- Fallbacks: `Qwen/Qwen3-4B-Instruct-2507-FP8` (~5.19 GB — **reported `!!!!` corruption on vLLM 0.10.x**,
  verify first) [4]; BF16 base `Qwen/Qwen3-4B-Instruct-2507` (~8 GB, always works). [5]
- On 128 GB, quant is about decode speed, not capacity.

## Architecture
- 4.0B total (3.6B non-embed), 36 layers, GQA 32 query / 8 KV heads. [5]
- **Context 262,144 native** (no YaRN needed). [5]
- **Non-thinking instruct refresh:** does NOT emit `<think>`; **`enable_thinking` is a no-op** (no
  reasoning mode to disable). The reasoning variant is the separate `Qwen3-4B-Thinking-2507`. [5][6]
- Sampling: temp 0.7, top_p 0.8, top_k 20, min_p 0. [5]
- **Tool parser: `hermes`** (`--enable-auto-tool-choice --tool-call-parser hermes`). qwen3_coder /
  qwen3_xml are for Qwen3.5/3.6 and Coder, NOT the 2507 dense models. [7]

## Serving (vLLM)
```bash
# BF16 baseline (any GPU):
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 262144   # drop to 32768 if memory-pressured
```
NVFP4 on GB10 (sm_121):
- Use `vllm/vllm-openai:cu130-nightly` (carries Blackwell NVFP4 + FlashInfer/CUTLASS path). [8][9]
- `--quantization compressed-tensors` for the LLM-Compressor repos; `--gpu-memory-utilization 0.85`,
  `--max-num-seqs 4` (single-user), optional `--kv-cache-dtype fp8_e4m3`. [8][9]
- **sm_121 FP4 gotcha:** the SM80 Marlin fallback produces wrong logits / illegal-instruction crashes
  for FP4 on Blackwell — must route through FlashInfer/CUTLASS. Native FP4 for SM120/121 landed in vLLM
  main 2026-05-20 (PR #40082); needs `CUTE_DSL_ARCH=sm_121a` + `nvidia-cutlass-dsl==4.4.2` (4.5.0 emits
  bad PTX). `--moe-backend` flags are MoE-only — for this **dense** 4B the requirement is simply that
  the FP4 GEMM uses FlashInfer/CUTLASS, not Marlin. [10][11][12]

## Measured performance
- **No vLLM-on-Spark measurement for this exact model — not found.** Closest:
  - This model NVFP4 on Spark via **TensorRT-LLM** (not vLLM), concurrency 2: **TTFT 43.9 ms, decode
    50.4 tok/s**; vs BF16 90.7 ms / 23.8 tok/s. Upper-bound proxy. [1]
  - Qwen3.6-35B-A3B-NVFP4 (MoE) vLLM on Spark: ~55.9 tok/s w/ MTP, TTFT 166 ms. [8]
- A dense 4B is far smaller than all measured neighbors → single-stream should comfortably exceed them,
  but **measure directly** (this benchmark fills the gap).

## Known issues
- FP4 Marlin fallback broken on sm_121 (wrong logits). [10][11]
- NVFP4 illegal-instruction crashes seen on ARM64 GB10 on older vLLM (0.16.1rc1) — cu130-nightly +
  PR#40082 b12x path is the intended fix. [11][12]
- `Qwen3-4B-Instruct-2507-FP8` `!!!!` corruption on vLLM 0.10.x. [4]
- Pin `nvidia-cutlass-dsl==4.4.2`. [12]

## Sources
1. https://huggingface.co/OPENZEKA/Qwen3-4B-Instruct-2507-NVFP4
2. https://huggingface.co/kaitchup/Qwen3-4B-Instruct-2507-NVFP4
3. https://huggingface.co/llmat/Qwen3-4B-Instruct-2507-NVFP4
4. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8/discussions/2
5. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
6. https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
7. https://qwen.readthedocs.io/en/latest/deployment/vllm.html
8. https://stevescargall.com/blog/2026/04/vllm-recipe-redhatai/qwen3.6-35b-a3b-nvfp4-on-dgx-spark/
9. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
10. https://github.com/vllm-project/vllm/issues/37030
11. https://github.com/vllm-project/vllm/issues/35519
12. https://github.com/vllm-project/vllm/pull/40082
13. https://medium.com/@stablehigashi/vllm-installation-on-dgx-spark-gb10-sm-121-and-qwen-3-5-serving-guide-9eba91e448f8
