# vllm-gemma4-26b-a4b

Running **Google Gemma 4** locally — with the **ASUS Ascent GX10 / NVIDIA GB10** as the
reference device — across vLLM, Ollama, and llama.cpp. Same method as the
[Qwen3.6 subproject](../vllm-qwen3.6-35b-a3b/): a cited corpus, layered guides
(newcomer on-ramp → advanced deep-dive), and reproducible scripts.

> **Status:** draft. 14-source cited corpus complete; guides + scripts in progress.

## What is Gemma 4?

Open multimodal models from the **Gemma Team, Google DeepMind**, released **2026-04-02**.
Five sizes, dense **and** MoE, multimodal (text + image; audio on E2B/E4B/12B), 140+ languages.

| Variant | Type | Params (total / active) | Context | Notes |
|---|---|---|---|---|
| **E2B** | dense (effective) | ~2.3B eff. | 128K | edge / on-device, +audio |
| **E4B** | dense (effective) | ~4.5B eff. | 128K | edge, +audio |
| **12B** | dense | 12B | 256K | +audio |
| **26B-A4B** | **MoE** | 25.2B / **3.8B active** (128 experts, 8+1) | 256K | **local sweet spot** |
| **31B** | dense | 31B | 256K | highest quality, heaviest |

Each ships base + instruction-tuned (`-it`). Architecture highlights: **hybrid attention**
(interleaved local sliding-window + global, final layer global), **dual RoPE** (standard for
sliding, pruned for global), **Per-Layer Embeddings (PLE)**, and a **Shared KV Cache** (late
layers reuse earlier KV). Configurable visual token budget (70–1120 tokens/image).

## Why MoE wins on the GX10 (and the local angle)

On the GB10's **273 GB/s** memory, decode is bandwidth-bound — so active-parameter count
dominates. Published DGX Spark numbers: **26B-A4B NVFP4 ≈ 52 tok/s** (16.5 GB weights, ~82 GB
free for KV) vs the **31B dense ≈ 6.4 tok/s** (~7.5× slower) [[g09]](sources/g09-ai-muninn-dgxspark-nvfp4-52.md). The MoE is the model
to run locally here — directly analogous to Qwen3.6-35B-A3B.

## Contents

- **[sources/](sources/INDEX.md)** — 14 cited sources (official model card, DeepMind/Google &
  HF announcements, vLLM/Ollama/llama.cpp local guides, GB10 benchmarks).
- **guides/** — layered local-inference series (in progress).
- **scripts/** — launch/benchmark/rollback for local Gemma 4 (in progress).
- **benchmarks/** — measured/collected numbers.

## License

[MIT](../../LICENSE) © 2026 Heitor Mocelin
