# gpt-oss-20b on DGX Spark / GB10 — vLLM research digest

## Checkpoint
- `openai/gpt-oss-20b` ships **native MXFP4** (MoE weights post-trained at ~4.25 bits/param). On-disk
  **~13.76 GB** (3 shards). Runs in ~16 GB. [1][2]
- **No official NVFP4 repackage** found (NVIDIA's NIM serves the MXFP4 weights). [9]

## Architecture
- ~20.9–21B total, **3.6B active**/token; **32 experts, top-4**; 24 layers; GQA 64/8; alternating
  full + sliding-128 attention; RoPE+YaRN; attention sinks. **128K ctx**. [1][4][8]
- **Harmony response format required** ("will not work correctly otherwise"). [1]
- **Reasoning effort low/medium/high** via system prompt (`Reasoning: low` = fast). Higher = more decode
  tokens = more latency. [1]
- Tool calling native → vLLM `--tool-call-parser openai --enable-auto-tool-choice`. [5]

## Serving (vLLM)
Canonical recipe [5]:
```bash
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
vllm serve openai/gpt-oss-20b --config GPT-OSS_Blackwell.yaml --tensor-parallel-size 1
# GPT-OSS_Blackwell.yaml: kv-cache-dtype: fp8 / no-enable-prefix-caching: true /
#   max-cudagraph-capture-size: 2048 / max-num-batched-tokens: 8192 / stream-interval: 20
```
- ⚠️ That YAML targets **datacenter Blackwell (sm_100)**. DGX Spark is **sm_121**. A community GB10
  working config (for 120b, flags apply) [10]:
```bash
vllm serve openai/gpt-oss-20b --quantization mxfp4 --mxfp4-backend CUTLASS \
  --attention-backend FLASHINFER --kv-cache-dtype fp8 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 --max-model-len 131072 --max-num-seqs 2 \
  --max-num-batched-tokens 8192 --enable-prefix-caching --load-format fastsafetensors
```
  That author found **CUDA 12.1f** (not 12.1a) was required to enable FP4 tensor-core paths. A separate
  patched effort used vLLM 0.17.0 + `VLLM_MXFP4_BACKEND=marlin` + SM121 fixes. [10][11]
- Single-user low latency: `--max-num-seqs 1-2`, `Reasoning: low` in the system prompt.

## Measured performance — single-stream, gpt-oss-20b MXFP4 on DGX Spark
| Engine | Decode tok/s | Prefill tok/s | Source |
|---|---|---|---|
| Ollama | **58.27** | 3,224 | [3] |
| Ollama (other) | 49.7 | 2,053 | [4][12] |
| llama.cpp | **60.85** (→48.3 @32K) | 2,008 (→1,193 @32K) | [12] |
| SGLang (LMSYS tuned) | **~70** | — | [6] |
| **vLLM** | **not found** (only 120b vLLM numbers published) | — | — |
- Expect ~50–70 tok/s single-stream on GB10; **our vLLM-20b number fills the gap.** TTFT on Spark: not
  found. Footprint ~13.8 GB + KV, comfortable in 128 GB. [1][2]

## Known issues
- **MXFP4 broken on SM121 with Marlin fallback (critical):** vLLM #37030 — wrong logits for the first
  Harmony token → chat returns `content: null`. Documented for 120b; same kernel path → **verify a
  single 20b completion before benchmarking.** [13]
- SM121 has **no native FP4 hardware compute**; needs CUTLASS/FlashInfer FP4 correctly built (community
  needed CUDA 12.1f). [12][13]
- FlashInfer + MTP spec-decode crash on SM121 → `--attention-backend triton_attn`. [14]
- FlashInfer attention-sinks `NotImplementedError` on some versions → recent FlashInfer build. [7]
- The LMSYS "~70 tok/s" guide is **SGLang, not vLLM** — don't copy its config into vLLM. [6]

## Sources
1. https://huggingface.co/openai/gpt-oss-20b
2. https://huggingface.co/openai/gpt-oss-20b/tree/main
3. https://ollama.com/blog/nvidia-spark-performance
4. https://arxiv.org/html/2508.16700v1
5. https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
6. https://www.lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/
7. https://blog.vllm.ai/2025/08/05/gpt-oss.html
8. https://arxiv.org/pdf/2508.10925
9. https://build.nvidia.com/openai/gpt-oss-20b/modelcard
10. https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651
11. https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824
12. https://github.com/ggml-org/llama.cpp/discussions/16578
13. https://github.com/vllm-project/vllm/issues/37030
14. https://github.com/vllm-project/vllm/issues/37754
