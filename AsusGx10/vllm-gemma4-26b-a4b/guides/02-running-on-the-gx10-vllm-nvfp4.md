# 02 · Running Gemma 4 on the GX10 — NVFP4 on vLLM

> The reference local deployment: **Gemma 4 26B-A4B in NVFP4, ~52 tok/s on the GB10**, serving an
> OpenAI-compatible endpoint. This guide is the concrete recipe — and the `sm_121` kernel caveat
> that decides how fast it actually goes.

## Plain-language on-ramp

You want Gemma 4 as a fast, private API on the GX10. The winning combination is the **26B-A4B MoE**
(few active params) in **NVFP4** (4-bit weights) on **vLLM** (serving engine). Published result:
**~52 tok/s, 16.5 GB of weights, ~82 GB left for KV cache** — vs ~7 tok/s for the 31B dense, which
you should *not* run here [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md).

Two gotchas to know before you start:
1. There's **no official NVFP4 checkpoint for the 26B-A4B MoE** — only for the 31B dense. You use a
   **community quantization** plus a small model-code patch.
2. The GB10 (`sm_121`) doesn't yet run the *native* FP4 path, so vLLM uses the **Marlin** kernel
   (correct, but leaves speed on the table) — the same story as the Qwen subproject.

## Why not the 31B dense

It's bandwidth. A dense model reads **all** its weights per token; on the GB10's 273 GB/s that's
~7 tok/s **regardless of quantization** — NVFP4 shrinks the bytes but the per-token read stays
proportional [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md). Measured: 31B NVFP4 ≈ **6.9 tok/s**, 26B-A4B NVFP4 ≈ **48–52 tok/s** [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md).
The MoE only fires ~3.8B params/token, so it moves ~8× fewer weight-bytes — hence ~7.5× faster.

> 📊 **From our own 7-model matrix ([FINDINGS](../../FINDINGS.md)):** Gemma-4-31B measured **6.8 tok/s**
> single-stream — confirming the dense wall. But note it ran at only **54% of its roofline**, vs **91%**
> for the similarly-sized Qwen3-32B — Gemma-4-31B is a measured **efficiency outlier**, its hybrid
> attention + Per-Layer-Embeddings costing it on the `sm_121` kernels. The MoE is doubly the right call
> here. ⚠️ The **26B-A4B NVFP4 community checkpoint failed to deploy** on our vLLM nightly (engine init —
> patch/version mismatch); the 52 tok/s above is from [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md), pending a retry with a matched build.

## The checkpoint situation

NVIDIA's official `nvidia/Gemma-4-31B-IT-NVFP4` exists only for the **dense 31B**. For the MoE,
standard NVIDIA modelopt tooling **can't quantize Gemma 4's fused 3D expert tensor format**, so the
community **`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`** was built with a custom modelopt plugin
and ships a small `gemma4_patched.py` that vLLM needs (without it the NVFP4 scale keys fail to load)
[[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md). Expect this to be temporary — upstream support will land — but as of this writing it's the
working path.

## The deployment (grounded in [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md))

```bash
# 1. download the community NVFP4 MoE checkpoint
huggingface-cli download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 \
  --local-dir ~/models/gemma4-26b-a4b-nvfp4

# 2. serve via vLLM (note the patched model file mounted over vLLM's gemma4.py)
sudo docker run -d --name gemma4-nvfp4 --runtime=nvidia --gpus=all \
  -p 8000:8000 \
  -v ~/models/gemma4-26b-a4b-nvfp4:/models/gemma4 \
  -v ~/models/gemma4-26b-a4b-nvfp4/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py \
  vllm/vllm-openai:cu130-nightly \
  --model /models/gemma4 --served-model-name gemma-4-26b \
  --quantization modelopt --max-model-len 131072 --gpu-memory-utilization 0.85
```

Key flags: `--quantization modelopt` (the checkpoint is modelopt-NVFP4), `--max-model-len 131072`
(128K — plenty of KV headroom in the ~82 GB free), `--gpu-memory-utilization 0.85` [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md). Pin a
**recent nightly** — Gemma 4 needs it, and **vLLM 0.19 shipped day-one with the `sm_121` NVFP4
fixes** that had been broken since March [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md)[[g05]](../sources/g05-vllm-recipe-gemma4.md).

> This mirrors your existing Qwen3.6 launch exactly — same `--quantization modelopt`, same NVFP4,
> same Marlin caveat below. The [`scripts/`](../scripts/) here parameterize it.

## The `sm_121` / Marlin caveat (cross-ref Qwen guide 02)

On the GB10, **Marlin decompresses the FP4 weights to BF16 at runtime — correct, but slower than the
native W4A4 FP4 path** [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md). That's the *same* `FLASHINFER_CUTLASS`/`sm_121` gap documented in the
[Qwen subproject's guide 02](../../vllm-qwen3.6-35b-a3b/guides/02-quantization-nvfp4-and-fp8.md):
the FP4 MoE kernels lacked SM120/121 coverage, so vLLM falls back. vLLM 0.19's day-one SM121 fixes
got NVFP4 *working* on the GB10; whether it's the *fast* native path or Marlin depends on your build
— track the FlashInfer b12x kernels (vLLM PR #40082) and re-measure (guide `05`). The 52 tok/s figure
is **with** the fallback, so there may be headroom.

## Memory budget on 128 GB

- Weights (26B-A4B NVFP4): **~16.5 GB** → leaves **~82 GB for KV cache** at `util 0.85` [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md).
- That KV budget easily covers 128K context for many concurrent sessions — Gemma 4's **shared KV
  cache** + dual-RoPE design stretch it further (guide `03`).
- You can run Gemma 4 *alongside* the existing Qwen3.6 server on a different port, memory permitting,
  or swap between them.

## Multimodal, thinking, tools (preview → guide `04`)

vLLM exposes Gemma 4's image input (configurable vision token budget via
`--mm-processor-kwargs '{"max_soft_tokens": 560}'`), structured **thinking mode**, and the **custom
tool-call protocol** through the OpenAI API [[g05]](../sources/g05-vllm-recipe-gemma4.md). Details and flags in guide `04`.

## Where to go next

- **Guide `03`** — quantization formats + the memory/KV math.
- **Guide `05`** — measure your GX10's actual Gemma 4 numbers (and the Marlin-vs-native delta).

## Sources cited

- [[g05]](../sources/g05-vllm-recipe-gemma4.md) Gemma 4 vLLM recipe
- [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) 52 tok/s NVFP4 on DGX Spark (the recipe + numbers)
