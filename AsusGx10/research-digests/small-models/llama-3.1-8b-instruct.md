# Llama-3.1-8B-Instruct on DGX Spark / GB10 — vLLM research digest

## Checkpoint
- **`nvidia/Llama-3.1-8B-Instruct-NVFP4` confirmed** — base `meta-llama/Llama-3.1-8B-Instruct`,
  quantized with nvidia-modelopt v0.35.0 (weights+activations of transformer linear ops to FP4).
  On-disk **≈ 6.0 GB** (4.98 + 1.05 GB safetensors). **vLLM-ready — leave `--quantization` unset**
  (auto-detect). [1]
- FP8 alt: `nvidia/Llama-3.1-8B-Instruct-FP8`. No official NVIDIA AWQ found. [1]

## Architecture
- 8B dense, **128K (131,072) context**. [1]
- **No thinking mode** — do not set `--reasoning-parser`. [1]
- **Tool parser: `llama3_json`** (not pythonic). [1]

## Serving (vLLM)
```bash
docker run -d --name llama31-8b --gpus all --ipc host --shm-size 16g -p 8001:8000 \
  -v ~/models/llama31-8b-nvfp4:/models/llama31-8b \
  vllm/vllm-openai:cu130-nightly \
  --model /models/llama31-8b --served-model-name llama31-8b \
  --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --enable-auto-tool-choice --tool-call-parser llama3_json
```
- Use sm_121-validated image (`cu130-nightly`); `--gpu-memory-utilization 0.85`, `--max-num-seqs 4`,
  keep default KV-cache dtype (avoid fp8 KV), backends on `auto`.
- **Dense Llama NVFP4 routes through FlashInfer/CUTLASS, NOT the broken Marlin path** — the Marlin
  correctness bug is a MoE/MXFP4 problem, so this dense model is comparatively safe.
- Watch the FP4 build-target trap (`sm_121f` / `FLASHINFER_CUDA_ARCH_LIST="12.1f"`).

## Measured performance
- **No vLLM+NVFP4 8B number on DGX Spark — not found.** Best reference: **38.65 tok/s decode,
  10,256.9 tok/s prefill** (NVIDIA dev blog, **TRT-LLM** NVFP4, BS1, 2048/128). [4]
- LMSYS SGLang FP8: 20.5 tok/s decode / 7,991 prefill at BS1.
- A "924 tok/s FP4 8B" snippet **could not be verified** — discard.

## Known issues
- sm_121 Marlin MoE/MXFP4 first-token corruption (a MoE problem; dense Llama unaffected). [22 in cross-refs]
- FP4 build-target trap (`12.1f`). Historically limited backend override — improved by vLLM PR #40082.
- JIT cold-start ~25 s.
- One honest source conflict: DeepWiki says GB10 has native FP4 via `mma.kind::mxf4.block_scale`
  (needs `sm_121f`); a vLLM issue reports "no native FP4" forcing Marlin for MoE — verify against the
  actual container at run time.

## Sources
1. https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4
2. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
3. https://www.lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/
4. https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
5. https://github.com/vllm-project/vllm/pull/40082
6. https://github.com/vllm-project/vllm/issues/35519
7. https://github.com/vllm-project/vllm/issues/37030
8. https://docs.vllm.ai/en/stable/features/tool_calling/
