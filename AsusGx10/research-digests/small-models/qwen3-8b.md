# Qwen3-8B on DGX Spark / GB10 — vLLM research digest

## Checkpoint
- **`nvidia/Qwen3-8B-NVFP4` confirmed real** — base `Qwen/Qwen3-8B`, 8.2B params, NVFP4 via TensorRT
  Model Optimizer v0.35.0, 131K ctx, Apache 2.0. On-disk **~5–6 GB** (exact not stated). [1]
- **vLLM-ready? Mixed.** NVIDIA documents it for **TensorRT-LLM**, but vLLM loads ModelOpt NVFP4 via
  `--quantization modelopt`. No published vLLM-on-GB10 validation for this dense 8B, and NVFP4 on ARM64
  GB10 currently risks an illegal-instruction crash (#35519). [2][3][4]
- Safer alternatives on GB10: **FP8** (broadly works, needs the sm_121 FP8 guard patch) or a
  cu130-nightly build validated for sm_121. AWQ INT4 also community-used (NVFP4 ~20% faster than AWQ). [5][6][7][8]

## Architecture
- 8.2B dense; native ctx 32,768 (`max_position_embeddings` 40,960), extendable to 131K via YaRN. [1][11]
- **Reasoning ON by default.** Disable: per-request `chat_template_kwargs={"enable_thinking": false}`
  or `/no_think`; server-wide `--reasoning-parser qwen3 --default-chat-template-kwargs
  '{"enable_thinking": false}'`. [11][12]
- **Tool parser: `hermes`** (`--enable-auto-tool-choice --tool-call-parser hermes`). With thinking ON,
  Qwen3 often plans but fails to emit tool calls (~60% fail) → run non-thinking for reliable tools. [11][13][14]

## Serving (vLLM)
- Use `vllm/vllm-openai:cu130-nightly` (stock NVIDIA container lacks NVFP4 / `flashinfer_cutlass`). Stock
  wheels compile only ≤ sm_120 → can crash at init on Spark; use an sm_121-targeted/community image. [10][15][16]
```bash
vllm serve <checkpoint> --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser hermes
# add --quantization modelopt for the nvidia NVFP4 checkpoint
```
- **KV cache:** official Spark guidance warns against `--kv-cache-dtype fp8` unless memory-constrained
  (an 8B has ample room) — keep default. [10]
- FP4 native vs Marlin: avoid Marlin (buggy on SM12.x); native NVFP4 works in the cu130-nightly/recipe
  path. Cold start ~25 s (JIT). [17][18][10]

## Measured performance
- **No Qwen3-8B-dense-on-Spark numbers published — not found.** Nearest neighbors:
  - Qwen3.6-35B-A3B-FP8 (MoE) Spark vLLM: ~28–30 tok/s single-user, TTFT ~363 ms. [6]
  - Qwen3.6-35B-A3B-NVFP4 (MoE) Spark + MTP: 55.9 tok/s, TTFT 166 ms. [15]
  - Qwen2.5-3B BF16 Spark: 26 tok/s single-user (a dense 8B will be below this per-token). [20]
- Decode is bandwidth-bound (~273 GB/s). Capture our own numbers.

## Known issues
1. **NVFP4 illegal-instruction crash on ARM64 GB10 (open, chief risk for the NVFP4 path).** [4]
2. No sm_121 in stock aarch64 builds → init crash without cu130-nightly/community image. [16]
3. **FP8 SM12.1 kernel-guard trap** (`enable_sm120_only`) — patch to `enable_sm120_family`. [7]
4. BF16↔NVFP4 Marlin dequant garbles output (SM<100; reason to avoid Marlin FP4). [21]
5. Marlin MXFP4 wrong first token on SM121. [22]

## Sources
1. https://huggingface.co/nvidia/Qwen3-8B-NVFP4
2. https://docs.vllm.ai/en/stable/features/quantization/modelopt/
3. https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
4. https://github.com/vllm-project/vllm/issues/35519
5. https://rikkarth.com/blog/2026-04-23-benchmark-results-for-qwen-qwen3-6-35b-a3b-fp8-nvidia-dgx-spark-gb10-serving-via-vllm
6. https://rikkarth.com/blog/2026-04-23-benchmark-results-for-qwen-qwen3-6-35b-a3b-fp8-nvidia-dgx-spark-gb10-serving-via-vllm
7. https://github.com/eugr/spark-vllm-docker/issues/143
8. https://github.com/Avarok-Cybersecurity/dgx-vllm
9. https://forums.developer.nvidia.com/t/qwen3-6-27b-awq-int4-on-dgx-spark-gb10-only-1-8-4-9-tok-s-decode-with-285k-token-prompt-how-to-improve/371529
10. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
11. https://qwen.readthedocs.io/en/latest/deployment/vllm.html
12. https://docs.vllm.ai/en/latest/features/reasoning_outputs/
13. https://docs.vllm.ai/en/stable/features/tool_calling/
14. https://github.com/QwenLM/Qwen3/issues/1817
15. https://stevescargall.com/blog/2026/04/vllm-recipe-redhatai/qwen3.6-35b-a3b-nvfp4-on-dgx-spark/
16. https://github.com/vllm-project/vllm/issues/36821
17. https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069/131
18. https://github.com/vllm-project/vllm/issues/31085
19. https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-up-to-51-tok-s-v2-1-patches-quick-start-benchmark/365639
20. https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
21. https://github.com/vllm-project/vllm/issues/34694
22. https://github.com/vllm-project/vllm/issues/37030
