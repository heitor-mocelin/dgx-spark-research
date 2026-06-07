# Gemma-4-E4B-it on DGX Spark / GB10 — vLLM research digest

> "Gemma-4-E4B-it" = the `E4B` ("effective 4B") tier of **Gemma 4** (Google, released 2026-03-31,
> Apache 2.0), successor to Gemma 3n's E-series. Not Gemma 3n.

## Checkpoint
- **Recommended NVFP4 (best for Blackwell):** `coolthor/Gemma-4-E4B-it-NVFP4A16` — the only community
  NVFP4 E4B quant with *measured DGX Spark numbers*. Scheme **NVFP4A16 (W4A16, weight-only)** via
  `llm-compressor`. On-disk ≈ 9–10 GB; runtime ≈ **9.8 GB**. Weight-only → does **not** use FP4
  activation GEMMs, so it sidesteps the broken SM120/121 FP4-GEMM path. [1][7]
- Alt NVFP4: `cosmicproc/gemma-4-E4B-it-NVFP4` (full W4A4 via ModelOpt) — **requires** working Blackwell
  FP4 kernels, risky on GB10. [2]
- **No official `nvidia/*-NVFP4` for E4B** — NVIDIA published NVFP4 only for `Gemma-4-31B-IT` and
  `Gemma-4-26B-A4B`. [3][8]
- Base BF16: `google/gemma-4-E4B-it` (~15 GB runtime). [5]

## Architecture
- 8B total / **4.5B effective**; large **Per-Layer Embeddings (PLE)** stay BF16 even when quantized
  (~5.4 GB), hit every token → reduce effective active params. [5][6][7]
- 42 layers, hidden 2560, vocab 262144, 8 attn / 2 KV heads (GQA), **head_dim 256 (sliding) / 512
  (global)**, interleaved local/global attention, sliding window 512. The heterogeneous head dims are
  the source of the perf bug below. [6]
- Context 128K (out up to 32K). Text + image + **native audio**. [5]
- **Thinking off by default**; enable via `enable_thinking=True`. Disable server-wide:
  `--default-chat-template-kwargs '{"enable_thinking": false}'`. [9][10]
- Tool calling: `--reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice` +
  `tool_chat_template_gemma4.jinja`. [4][10]

## Serving (vLLM) — DGX Spark validated [1]
```bash
docker run -d --name gemma4-e4b --gpus all --ipc host --shm-size 32gb -p 8003:8000 \
  -v ~/models/gemma4-e4b-nvfp4:/models/gemma4-e4b \
  vllm/vllm-openai:gemma4-cu130 \
  --model /models/gemma4-e4b --served-model-name gemma-4-e4b \
  --quantization compressed-tensors --kv-cache-dtype fp8 \
  --max-model-len 16384 --gpu-memory-utilization 0.75
```
- **Use a CUDA-13 image** on Blackwell. **Do NOT force FlashInfer** — head_size 256/512 → unsupported;
  vLLM auto-falls-back to `TRITON_ATTN`, which works (and is the perf cap). [11][4]
- Prefer **NVFP4A16 weight-only** (avoids the broken FP4 activation GEMM on sm_121). [2][13]

## Measured performance — DGX Spark (GB10), single-stream, 500-tok decode [1]
| Variant | Decode tok/s | Footprint |
|---|---|---|
| BF16 | 19.2 | 15 GB |
| FP8 online | 36.0 | 11.4 GB |
| **NVFP4A16** | **49.9** | **9.8 GB** |

- ~2.6× over BF16; **~50 tok/s** single-stream. TTFT/prefill on DGX Spark: **not found**.
- Reference (96 GB *desktop* Blackwell, NOT Spark): ≈149 tok/s decode, 17 ms TTFT — do not use for Spark
  planning (much higher bandwidth). [15]

## Known issues
- **Triton-attention fallback penalty (biggest):** heterogeneous head dims force `TRITON_ATTN`; caps
  throughput everywhere, GB10 included. [16][11]
- **NVFP4 W4A4 GEMM broken on SM120/121** (CUTLASS FP4 fails) → use NVFP4A16 weight-only. [13][2]
- Reasoning/tool-call tag leakage with thinking toggles. [17][19]
- TP/PP NVFP4 failures (multi-node only). [18]

## Sources
1. https://ai-muninn.com/en/blog/dgx-spark-gemma4-e4b-nvfp4-50-toks
2. https://huggingface.co/cosmicproc/gemma-4-E4B-it-NVFP4
3. https://developer.nvidia.com/blog/bringing-ai-closer-to-the-edge-and-on-device-with-gemma-4/
4. https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it
5. https://huggingface.co/google/gemma-4-E4B-it
6. https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json
7. https://huggingface.co/coolthor/Gemma-4-E4B-it-NVFP4A16
8. https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
9. https://github.com/vllm-project/vllm/issues/39130
10. https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
11. https://github.com/vllm-project/vllm/issues/40677
12. https://allenkuo.medium.com/finishing-what-we-started-gemma-4-nvfp4-on-vllm-desktop-blackwell-wsl2-b2088c34815a
13. https://github.com/flashinfer-ai/flashinfer/issues/2577
14. https://forums.developer.nvidia.com/t/how-to-run-gemma-4-nvfp4-in-vllm-docker/365513
15. https://allenkuo.medium.com/gemma-4-on-vllm-vs-ollama-benchmarks-on-a-96-gb-blackwell-gpu-804ca4845a21
16. https://github.com/vllm-project/vllm/issues/38887
17. https://github.com/vllm-project/vllm/issues/39043
18. https://github.com/vllm-project/vllm/issues/42516
19. https://github.com/vllm-project/vllm/issues/38855
