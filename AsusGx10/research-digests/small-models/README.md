# Small "reflex" models — research digests & finalized benchmark configs

Community-grounded research (Hugging Face cards, vLLM/NVIDIA docs, GitHub issues, DGX Spark blogs &
forums — ~8–22 sources each) for 7 small, instruction-tuned, quantized models evaluated as fast
single-user companions to the production **Qwen3.6-35B-A3B** on the GX10. One digest per model:

- [gemma-4-e4b.md](gemma-4-e4b.md)
- [qwen3-4b-instruct.md](qwen3-4b-instruct.md)
- [qwen3-8b.md](qwen3-8b.md)
- [llama-3.1-8b-instruct.md](llama-3.1-8b-instruct.md)
- [ministral-8b.md](ministral-8b.md)
- [phi-4-mini.md](phi-4-mini.md)
- [gpt-oss-20b.md](gpt-oss-20b.md)

---

## ⚠️ The cross-cutting discovery: FP4 on sm_121 (GB10) is fragile

Independently confirmed across nearly every model's sources — this **reshapes the benchmark**:

1. **The SM80 "Marlin" FP4 fallback produces wrong logits / illegal-instruction crashes on GB10**
   (vLLM [#35519](https://github.com/vllm-project/vllm/issues/35519),
   [#37030](https://github.com/vllm-project/vllm/issues/37030)). FP4 must route through
   **FlashInfer/CUTLASS**, not Marlin. Native FP4 for sm_121 landed in vLLM PR
   [#40082](https://github.com/vllm-project/vllm/pull/40082) (2026-05-20).
2. **FP8 CUTLASS has an sm_121 guard bug** (`enable_sm120_only` rejects ARCH 1210) — needs the
   `enable_sm120_family` patch ([eugr #143](https://github.com/eugr/spark-vllm-docker/issues/143)) or a
   build that already has it.
3. **Several "expected" NVFP4 checkpoints don't exist** for the small models (Ministral, Phi-4-mini),
   and the dense `nvidia/*-NVFP4` ones are documented for **TensorRT-LLM**, not vLLM.
4. The production Qwen already runs a *working NVFP4-MoE* path on the box's nightly image — but these
   **dense / MXFP4** models exercise **different kernels**, so each needs its own verification.

**Consequence → a hard gate added to the methodology:** before any model is benchmarked, it must pass a
**correctness smoke test** (a real chat completion that returns coherent, non-null content). A model
that returns `null`/garbage on the current image is recorded as "blocked on sm_121 (reason)" rather than
given a fake number. This honesty is the point of the study.

---

## Finalized per-model benchmark configs (what we'll actually run)

| Model | Checkpoint we'll use | Format | Port | Key flags / notes |
|---|---|---|---|---|
| **Gemma-4-E4B-it** | `coolthor/Gemma-4-E4B-it-NVFP4A16` | NVFP4 (W4A16, weight-only) | 8001 | `--quantization compressed-tensors --kv-cache-dtype fp8 --max-model-len 16384`; **don't force FlashInfer** (Triton attn). Has a **real ~50 tok/s Spark reference**. |
| **Qwen3-4B-Instruct-2507** | `kaitchup/Qwen3-4B-Instruct-2507-NVFP4` | NVFP4 (compressed-tensors) | 8001 | `--quantization compressed-tensors`; non-thinking (no-op); `--tool-call-parser hermes`. Likely speed champ. |
| **Qwen3-8B** | `nvidia/Qwen3-8B-NVFP4` (FP8 fallback) | NVFP4 modelopt | 8001 | `--quantization modelopt`, disable thinking, `hermes`. **Watch #35519 crash** → fall back to FP8 if it crashes. |
| **Llama-3.1-8B-Instruct** | `nvidia/Llama-3.1-8B-Instruct-NVFP4` | NVFP4 modelopt | 8001 | leave `--quantization` unset (auto); dense → CUTLASS not Marlin (safer); `--tool-call-parser llama3_json`. |
| **Ministral-8B-Instruct-2410** | base BF16 (gated) → FP8 alt | BF16 / FP8 | 8001 | **No NVFP4.** `--tokenizer-mode mistral --config-format mistral --load-format mistral --max-model-len 32768`. **Gated download (HF token) + non-commercial license.** |
| **Phi-4-mini-instruct** | `pytorch/Phi-4-mini-instruct-FP8` (BF16 fallback) | FP8 torchao | 8001 | `VLLM_DISABLE_COMPILE_CACHE=1`, `--tool-call-parser phi4_mini_json`. **No NVFP4** for the text model. |
| **gpt-oss-20b** | `openai/gpt-oss-20b` | MXFP4 (native) | 8001 | `--quantization mxfp4 --mxfp4-backend CUTLASS --attention-backend FLASHINFER`; harmony format; `Reasoning: low`. **Verify non-null output first (#37030).** |

> Quant formats deliberately differ per model (NVFP4 weight-only / NVFP4 W4A4 / FP8 / BF16 / MXFP4) —
> this is **honest "what's actually fastest on *my* box," not a synthetic apples-to-apples.** Every
> result records the exact checkpoint + format + image + flags.

**Two known frictions to handle during execution (not blockers):**
- **Ministral-8B** is HF-gated + non-commercial (MRL). Needs an HF token with accepted terms on the DGX;
  if unavailable, it's skipped and noted (or swapped for a non-gated 8B).
- **gpt-oss-20b** MXFP4 may return null on the current image; if it can't be made correct without a
  custom build, it's recorded as "blocked on sm_121" rather than benchmarked.

See [../../testing-plan-small-models.md](../../testing-plan-small-models.md) for the full methodology
(roofline predictions, TTFT-first metrics, 3× averaging, telemetry, co-residency protocol).
