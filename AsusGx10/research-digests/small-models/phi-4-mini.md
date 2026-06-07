# Phi-4-mini-instruct on DGX Spark / GB10 — vLLM research digest

> Headline: **no NVFP4 exists for the plain-text `Phi-4-mini-instruct`.** The `nvidia/*-NVFP4` Phi-4
> repos are for *different* models (`Phi-4-multimodal-instruct`, `Phi-4-reasoning-plus`).

## Checkpoint
| Repo | Quant | ~On-disk | vLLM | Notes |
|---|---|---|---|---|
| **`pytorch/Phi-4-mini-instruct-FP8`** | FP8 W8A8 (torchao) | ~4 GB (~5.7 GB peak) | ✅ card has `vllm serve` | Best quant for the *text* model; ~0.4% benchmark delta vs BF16 [1] |
| **`microsoft/Phi-4-mini-instruct`** | BF16 base | ~7.7 GB | ✅ first-class | Safest on Spark [3][4] |
| `nvidia/Phi-4-multimodal-instruct-NVFP4` | NVFP4 | — | TRT-LLM | **different model** [6] |
| `nvidia/Phi-4-reasoning-plus-NVFP4` | NVFP4 | — | — | **different model** [6] |

No NVFP4/AWQ/GPTQ for plain `Phi-4-mini-instruct` as of 2026-06. GGUF exists (llama.cpp only). [2][6]

## Architecture
- 3.8B dense, **128K ctx**, RoPE, GQA, shared embeddings, vocab 200,064, BF16. [3]
- **Not a reasoning model** — plain instruct, answers are already direct; nothing to suppress. (The
  reasoning sibling is `Phi-4-reasoning-plus`.) [3]
- Tool calling: vLLM has a dedicated parser → `--enable-auto-tool-choice --tool-call-parser
  phi4_mini_json` + `tool_chat_template_phi4_mini.jinja`. [7][8]

## Serving (vLLM) — BF16, single-user, with tools
```bash
vllm serve microsoft/Phi-4-mini-instruct --host 0.0.0.0 --port 8001 \
  --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --enable-auto-tool-choice --tool-call-parser phi4_mini_json
```
- **FP8 variant:** model `pytorch/Phi-4-mini-instruct-FP8`, add `--tokenizer microsoft/Phi-4-mini-instruct -O3`,
  set `VLLM_DISABLE_COMPILE_CACHE=1` (torchao). [1]
- DGX Spark: recent `cu130-nightly` image (older NGC `vllm:25.11-py3` lacks the sm_121 FP8 fix);
  `--gpu-memory-utilization 0.85`, `--max-num-seqs 4`, KV cache default (f16), backend `auto`. ~25 s JIT
  warmup. [9]

## Measured performance
- **No Phi-4-mini measurement on DGX Spark — not found.** Closest analog: **Qwen2.5-3B BF16 ≈ 26 tok/s**
  single-user on Spark → expect Phi-4-mini BF16 ~20–26 tok/s; FP8 (~half the bytes/token) likely
  faster. Extrapolation, not measured. [11]

## Known issues
1. **FP8 CUTLASS crash on sm_121** (`enable_sm120_only` guard) → patch to `enable_sm120_family` or use a
   recent cu130 image. Relevant if you pick FP8. [12]
2. NVFP4 silent slow fallback on sm_121 (not applicable here — no NVFP4 checkpoint). [10][13]
3. torchao FP8 compile-cache conflict → `VLLM_DISABLE_COMPILE_CACHE=1`. [1]
4. Old NGC container ships vLLM ≤ 0.11 — prefer newer cu130. [9][12]

## Sources
1. https://huggingface.co/pytorch/Phi-4-mini-instruct-FP8
2. https://huggingface.co/models?search=Phi-4-mini-instruct
3. https://huggingface.co/microsoft/Phi-4-mini-instruct
4. https://docs.vllm.ai/projects/recipes/en/latest/Microsoft/Phi-4.html
5. https://github.com/vllm-project/recipes/blob/main/Microsoft/Phi-4.md
6. https://huggingface.co/nvidia/Phi-4-multimodal-instruct-NVFP4
7. https://docs.vllm.ai/en/stable/api/vllm/tool_parsers/phi4mini_tool_parser/
8. https://docs.vllm.ai/en/stable/features/tool_calling/
9. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
10. https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119
11. https://blog.kubesimplify.com/day-2-anatomy-of-an-llm-inference-request-from-prompt-to-answer-step-by-step
12. https://github.com/eugr/spark-vllm-docker/issues/143
13. https://github.com/vllm-project/vllm/issues/35519
