# dgx-spark-research

Research, guides, and tooling for getting the most out of **NVIDIA DGX Spark / GB10
(Grace Blackwell)** class hardware for local LLM inference.

Each subproject is self-contained: a cited research corpus, layered guides (newcomer
on-ramp → advanced deep-dive), and reproducible scripts, grounded in both published
sources and real measurements on the author's hardware.

## Subprojects

- **[AsusGx10-vllm-optimization/](AsusGx10-vllm-optimization/)** — Optimizing
  [vLLM](https://docs.vllm.ai/) serving Qwen3-class MoE models (NVFP4 / FP8) on an
  **ASUS Ascent GX10** (GB10 Grace Blackwell, 128 GB LPDDR5). Throughput, quantization,
  latency, and tool-calling — from bare device to a tuned OpenAI-compatible endpoint.

## Conventions

- **Cite everything.** Every non-trivial claim traces to a saved source (see each
  subproject's `sources/`) or to a measurement on the author's hardware.
- **Layered docs.** Guides open with a plain-language on-ramp and build into advanced
  detail in the same flow, so newcomers and operators can both use them.
- **Reproducible scripts.** Ops scripts mirror the exact commands used on the hardware.

## License

[MIT](LICENSE) © 2026 Heitor Mocelin
