# Source corpus index — Gemma 4 local inference

14 sources, retrieved 2026-06-06, fetched on-device (OpenClaw LXC). Weighted toward
**local inference** (vLLM / Ollama / llama.cpp) and the **GX10 / GB10** reference device.
See each file's front matter for provenance, and [README.md](README.md) for conventions.

## Official / model

| ID | Title | Publisher | Topics |
|----|-------|-----------|--------|
| [g01](g01-gemma4-model-card.md) | Gemma 4 model card | Google AI for Developers | architecture, variants, multimodal, context |
| [g02](g02-google-blog-gemma4.md) | Gemma 4: Byte for byte, the most capable open models | Google blog | announcement, team, benchmarks |
| [g03](g03-hf-blog-welcome-gemma4.md) | Welcome Gemma 4: Frontier multimodal intelligence on device | Hugging Face blog | architecture, on-device, variants |
| [g04](g04-hf-card-26b-a4b-it.md) | Gemma 4 26B-A4B-it model card | Hugging Face (Google) | moe, model, local, quantization |

## Local inference (vLLM / Ollama / llama.cpp)

| ID | Title | Publisher | Topics |
|----|-------|-----------|--------|
| [g05](g05-vllm-recipe-gemma4.md) | Gemma 4 Usage Guide (vLLM Recipes) | vLLM Recipes | vllm, serving, multimodal |
| [g06](g06-unsloth-gemma4.md) | Gemma 4 — How to Run Locally (Unsloth) | Unsloth docs | gguf, llamacpp, quantization |
| [g07](g07-aimadetools-run-locally.md) | How to Run Gemma 4 Locally — Complete Setup Guide | aimadetools blog | ollama, llamacpp, vllm |
| [g08](g08-run-locally-ollama-llamacpp-vllm.md) | Run Gemma 4 Locally with Ollama, llama.cpp, and vLLM | DEV Community | ollama, llamacpp, vllm |

## GX10 / DGX Spark (GB10) benchmarks

| ID | Title | Publisher | Topics |
|----|-------|-----------|--------|
| [g09](g09-ai-muninn-dgxspark-nvfp4-52.md) | Gemma 4 26B in 16 GB at 52 tok/s — DGX Spark NVFP4 | ai-muninn | gb10, nvfp4, benchmark, moe |
| [g10](g10-shamily-gemma4-llama-dgx-spark.md) | gemma4-llama-dgx-spark: Dockerized GB10 inference + benchmarks | GitHub (community) | gb10, llamacpp, benchmark, arm64 |
| [g11](g11-nvidia-forum-gemma4-day1.md) | Gemma 4 Day-1 Inference on DGX Spark — Preliminary Benchmarks | NVIDIA Dev Forums | gb10, benchmark |
| [g12](g12-nvidia-forum-gemma4-31b-fp8.md) | Gemma 4 31B on DGX Spark: Runtime FP8 (Single & Dual Node TP=2) | NVIDIA Dev Forums | fp8, dense, tensor-parallel, benchmark |

## Architecture & benchmarks (analysis)

| ID | Title | Publisher | Topics |
|----|-------|-----------|--------|
| [g13](g13-labellerr-gemma4-overview.md) | Google Gemma 4: A Technical Overview | Labellerr | architecture, team, benchmarks |
| [g14](g14-lushbinary-gemma4-benchmarks.md) | Gemma 4 Developer Guide: Benchmarks, Architecture, Local Deployment | Lushbinary | benchmarks, architecture, local |

## Coverage

- **Model/architecture:** g01, g03, g04, g13 (hybrid attention, PLE, shared KV, MoE 26B-A4B)
- **Researchers / provenance:** g01, g02, g03 (Gemma Team, Google DeepMind; 2026-04-02)
- **Local — vLLM:** g05, g07, g08, g09  • **Ollama:** g07, g08  • **llama.cpp/GGUF:** g06, g08, g10
- **GB10 benchmarks:** g09 (52 tok/s NVFP4 MoE), g10 (llama.cpp suite), g11/g12 (forum runs)
- **Quality benchmarks:** g13, g14 (MMLU-Pro 85.2% / 31B; 26B-A4B: AIME 88.3%, GPQA 82.3%, LiveCodeBench 77.1%)

## Caveats

Weights are gated; some local-guide sources are community blogs (verify commands against the
official model card g01 and vLLM recipe g05). arXiv has no standalone Gemma 4 technical report
yet — authorship is the collective **Gemma Team, Google DeepMind** per the model card.
