# Ministral-8B-Instruct-2410 on DGX Spark / GB10 — vLLM research digest

> ⚠️ Two friction points up front: (1) **no NVFP4 exists** for this model; (2) the base repo is
> **gated** on HF and under the **Mistral Research License (non-commercial)** — fine for personal lab
> research, but the download needs an HF token with accepted terms.

## Checkpoint
- Base: `mistralai/Ministral-8B-Instruct-2410` — BF16, ~16 GB, **gated + MRL (non-commercial)**. [1]
- **NVFP4: NOT FOUND** for the 2410 model (only the newer Mistral 3 / Ministral 3 family has NVFP4). To
  get NVFP4 you'd self-quantize with `llm-compressor` / `nvidia-modelopt`. [6][9]
- Best available for Blackwell + vLLM (recommended order):

| Repo | Scheme | On-disk | Notes |
|---|---|---|---|
| `mistralai/Ministral-8B-Instruct-2410` (BF16) | BF16 | ~16 GB | Safest; **recommended default** given quant fragility on sm_121 |
| `PyrTools/Ministral-8B-Instruct-2410-FP8-Dynamic` | FP8 W8A8 | 9.1 GB | Needs the sm_121 FP8 guard fix (see below) |
| `warshanks/Ministral-8B-Instruct-2410-AWQ` | W4A16 AWQ | ~5–6 GB | W4A16 well-supported on Blackwell |
| `shuyuej/Ministral-8B-Instruct-2410-GPTQ` | GPTQ 4-bit | ~5–6 GB | Standard GPTQ |

(GGUF variants are for llama.cpp, not vLLM. AWQ/GPTQ sizes are estimates — verify file listings.)

## Architecture
- 8.02B dense; 36 layers, dim 4096, 32 attn / **8 KV heads (GQA)**, head dim 128. [1]
- Trained to **128k** but uses **interleaved sliding-window attention**; **vLLM caps at 32k** (SWA paged
  kernel not implemented) → `--max-model-len 32768`. Full 128k needs `mistral-inference`. [1][7]
- Tokenizer **V3-Tekken** (vocab 131,072) → `--tokenizer-mode mistral`.
- Tool calling supported → `--tool-call-parser mistral --enable-auto-tool-choice`. [1]

## Serving (vLLM) — official BF16 [1]
```bash
vllm serve mistralai/Ministral-8B-Instruct-2410 \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
  --max-model-len 32768 --tool-call-parser mistral --enable-auto-tool-choice \
  --gpu-memory-utilization 0.85
```
Needs vLLM ≥ 0.6.4, `mistral_common` ≥ 1.4.4. The three `mistral` flags are for the Mistral-native repo;
HF-format mirrors / AWQ-GPTQ repos may use standard `MistralForCausalLM` config and not need them — match
flags to the checkpoint. DGX Spark: cu130-nightly image; harmless "capability 12.1 vs 12.0" warning;
`--attention-backend triton_attn` as FlashInfer fallback.

## Measured performance
- **No Ministral-8B-2410 measurement on DGX Spark — not found.** Proxy = Llama-3.1-8B NVFP4 TRT-LLM on
  Spark: **38.65 tok/s decode / 10,257 prefill** (BS1). [4] General Spark 8B-class single-stream ≈ 30–40
  tok/s, bandwidth-bound. Measure directly.

## Known issues
1. **vLLM 32k context cap** (interleaved SWA). [1][7]
2. **FP8 GEMM crash on sm_121** (`enable_sm120_only` guard) → patch to `enable_sm120_family`, or use a
   build that already has it. Verify FP8 works before relying on it. [8]
3. FlashInfer has no sm_120/121 cubins on some versions → `--attention-backend triton_attn`. [3]
4. FlashInfer + MTP spec-decode crash on SM121. [3]
5. **MRL non-commercial license**; gated download. [1]

## Sources
1. https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
2. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
3. https://github.com/vllm-project/vllm/issues/37754
4. https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
5. https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/
6. https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4
7. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
8. https://github.com/eugr/spark-vllm-docker/issues/143
9. https://mistral.ai/news/mistral-3/
10. https://github.com/eelbaz/dgx-spark-vllm-setup
11. https://huggingface.co/PyrTools/Ministral-8B-Instruct-2410-FP8-Dynamic
12. https://huggingface.co/warshanks/Ministral-8B-Instruct-2410-AWQ
13. https://huggingface.co/shuyuej/Ministral-8B-Instruct-2410-GPTQ
