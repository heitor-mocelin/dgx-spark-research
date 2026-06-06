# 00 · Gemma 4 & the local-inference picture

> **This series in one line:** how to run **Google Gemma 4** *on your own hardware* — fast,
> cheap, private — with the **GX10 / NVIDIA GB10** as the reference device. Each guide has a
> plain-language on-ramp and a deep dive.

## Plain-language on-ramp

**Gemma 4** is Google DeepMind's open model family (released **2 April 2026**, **Apache-2.0** —
no usage restrictions [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md)). "Open" means you download the weights and run them yourself: no
API bill, no data leaving your machine. The models are **multimodal** (text + images; audio on
the small ones), speak **140+ languages**, and come in five sizes so you can match the model to
your hardware [[g01]](../sources/g01-gemma4-model-card.md)[[g03]](../sources/g03-hf-blog-welcome-gemma4.md).

The single most important choice for local use is **dense vs MoE**. A *dense* model uses all its
weights for every token; a **Mixture-of-Experts (MoE)** model only activates a slice. On a
memory-bandwidth-limited box (like the GB10), that slice is what determines speed — so the
**26B-A4B MoE is the local sweet spot**: ~30B-class quality at ~4B-class speed.

If you remember one thing: **on local hardware, pick the 26B-A4B MoE and quantize it.**

## The lineup

| Variant | Type | Total / active | Context | VRAM (Q4) | Best for |
|---|---|---|---|---|---|
| E2B | dense (effective) | ~2.3B | 128K | ~1.5 GB | phones, Raspberry Pi |
| E4B | dense (effective) | ~4.5B | 128K | ~3 GB | quick local tasks |
| 12B | dense (unified) | 12B | 256K | — | mid-range GPUs, +audio |
| **26B-A4B** | **MoE** | 25.2B / **3.8B** | 256K | **~14 GB** | **local sweet spot** |
| 31B | dense | 31B | 256K | ~18 GB | maximum quality |

Sources: sizes/active params [[g03]](../sources/g03-hf-blog-welcome-gemma4.md)[[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md); context windows [[g01]](../sources/g01-gemma4-model-card.md). Each ships **base** and
**instruction-tuned (`-it`)**.

## The architecture that makes it efficient

Gemma 4 packs several tricks aimed squarely at local/long-context use [[g01]](../sources/g01-gemma4-model-card.md)[[g03]](../sources/g03-hf-blog-welcome-gemma4.md)[[g05]](../sources/g05-vllm-recipe-gemma4.md):

- **MoE (26B-A4B):** 128 fine-grained experts, **top-8 routing + 1 shared expert**, custom
  GELU-activated FFN. Only ~3.8B params fire per token.
- **Hybrid attention:** alternating **local sliding-window** and **global** layers (the final
  layer is always global), with **dual RoPE** — standard for sliding layers, *pruned* for global
  layers — to stretch context cheaply.
- **Per-Layer Embeddings (PLE):** a second embedding table injects a small residual into every
  decoder layer (quality without proportional compute).
- **Shared KV cache:** late layers reuse earlier layers' key/value states, cutting KV memory —
  important when you want long context on a fixed memory budget (guide `03`).
- **Encoder-free 12B:** the 12B "unified" variant has *no* vision/audio tower — raw pixel patches
  and audio frames are projected straight into the LM [[g05]](../sources/g05-vllm-recipe-gemma4.md).
- **Dynamic vision resolution:** per-request image token budget (70 / 140 / 280 / 560 / 1120) —
  trade detail for speed (guide `04`).

## Why this matters on the GX10 (GB10)

The GB10 has **128 GB unified memory at 273 GB/s** — generous capacity, modest bandwidth. So:

- **MoE crushes dense here.** Published DGX Spark numbers: **26B-A4B NVFP4 ≈ 52 tok/s** (16.5 GB
  weights, **~82 GB left for KV cache**) versus the **31B dense ≈ 6.4 tok/s — ~7.5× slower**
  [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md). Bandwidth, not capacity, is the wall, and MoE moves far fewer bytes per token.
- **Memory is not the constraint — bandwidth is.** With 128 GB you can hold even the 31B dense
  comfortably; it's just slow. The whole game is *fewer active bytes per token* (MoE) + *fewer
  bytes per weight* (quantization, guide `03`).
- This is the **same lesson** as the Qwen3.6-35B-A3B work in the
  [sibling subproject](../../vllm-qwen3.6-35b-a3b/) — Gemma 4 26B-A4B is its close analog,
  and the measured Qwen baseline (~627 tok/s aggregate, ~75 single-stream) is a useful yardstick.

## Quality — it's not a toy

Gemma 4's reasoning is frontier-adjacent. Reported scores: **31B** — AIME-2026 89.2%,
LiveCodeBench-v6 80.0%, GPQA-Diamond 84.3%, MMLU-Pro 85.2% [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md)[[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md); **26B-A4B MoE** —
AIME 88.3%, GPQA 82.3%, LiveCodeBench 77.1% with only **3.8B active** [[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md). The 31B hit **#3 on
the Arena leaderboard** at launch [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md).

## Who built it

The **Gemma Team, Google DeepMind** — collective authorship, as with Gemma 3 (the Gemma 4 model
card credits the team rather than individual authors; no standalone arXiv technical report had
appeared as of this writing) [[g01]](../sources/g01-gemma4-model-card.md)[[g02]](../sources/g02-google-blog-gemma4.md).

## Where to go next

- **Guide `01`** — pick a runtime: vLLM vs Ollama vs llama.cpp.
- **Guide `02`** — the reference GX10 deployment (NVFP4 on vLLM, the 52 tok/s recipe).

## Sources cited

- [[g01]](../sources/g01-gemma4-model-card.md) Gemma 4 model card (Google)
- [[g02]](../sources/g02-google-blog-gemma4.md) Gemma 4 launch (Google blog)
- [[g03]](../sources/g03-hf-blog-welcome-gemma4.md) Welcome Gemma 4 (Hugging Face)
- [[g05]](../sources/g05-vllm-recipe-gemma4.md) Gemma 4 vLLM recipe
- [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) Run Gemma 4 locally (DEV)
- [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) 52 tok/s NVFP4 on DGX Spark
- [[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md) Gemma 4 benchmarks (Lushbinary)
