# AsusGx10

The **ASUS Ascent GX10** — NVIDIA **GB10 Grace Blackwell**, **128 GB** unified LPDDR5x @
**273 GB/s**, 20-core Arm (10× Cortex-X925 + 10× A725), Blackwell 5th-gen Tensor Cores
(`sm_121`), DGX OS (Ubuntu ARM64). All work here targets this one device.

Decode on this box is **memory-bandwidth-bound**, so the recurring theme across every subproject
is the same: **fewer active parameters (MoE) + fewer bytes per weight (NVFP4/FP8) → more tok/s**,
then amortize the weight read across requests (batching).

## Subprojects

| Folder | What | Highlight |
|---|---|---|
| **[vllm-qwen3.6-35b-a3b/](vllm-qwen3.6-35b-a3b/)** | Optimizing vLLM serving of Qwen3.6-35B-A3B (NVFP4/FP8) | **measured** ~627 tok/s @ c32, ~951 peak; full guide series + scripts + benchmarks |
| **[vllm-gemma4-26b-a4b/](vllm-gemma4-26b-a4b/)** | Running Google Gemma 4 locally (vLLM/Ollama/llama.cpp) | 26B-A4B NVFP4 ~52 tok/s; runtime choice + deployment recipe |
| **[research-digests/](research-digests/)** | On-device, model-generated literature digests | e.g. major discoveries in efficient LLM inference (48 arXiv papers) |

## Shared findings across models

- **MoE beats dense, hard.** Qwen3.6-35B-A3B and Gemma-4-26B-A4B (both ~3–4B active) run at tens of
  tok/s; comparable dense models crawl (Gemma 4 31B dense ≈ 7 tok/s).
- **The `sm_121` Marlin caveat is cross-model.** NVFP4 weights currently fall back to the Marlin
  kernel (FP4→BF16 decompress) on the GB10 for both models — native FP4-MoE kernels (FlashInfer
  b12x / vLLM PR #40082) are the shared upside, documented in each subproject's guide 02.
- **128 GB is plenty; bandwidth is the wall.** Capacity lets you hold big models and long-context
  KV cache; the 273 GB/s ceiling is what quantization + MoE are fighting.

## License

[MIT](../LICENSE) © 2026 Heitor Mocelin
