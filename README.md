# dgx-spark-research

Research, guides, and tooling for getting the most out of **NVIDIA DGX Spark / GB10
(Grace Blackwell)** class hardware for local LLM inference.

Each subproject is self-contained: a cited research corpus, layered guides (newcomer
on-ramp → advanced deep-dive), and reproducible scripts, grounded in both published
sources and real measurements on the author's hardware.

## Subprojects

Grouped by **device**. The first (and currently only) device is the ASUS Ascent GX10:

- **[AsusGx10/](AsusGx10/)** — NVIDIA **GB10 Grace Blackwell**, 128 GB unified LPDDR5x @ 273 GB/s.
  - **[vllm-qwen3.6-35b-a3b/](AsusGx10/vllm-qwen3.6-35b-a3b/)** — optimizing
    [vLLM](https://docs.vllm.ai/) serving of Qwen3.6-35B-A3B (NVFP4 / FP8): throughput,
    quantization, latency, tool-calling — bare device → tuned endpoint, with measured benchmarks.
  - **[vllm-gemma4-26b-a4b/](AsusGx10/vllm-gemma4-26b-a4b/)** — running **Google Gemma 4** locally
    (vLLM / Ollama / llama.cpp): variants & architecture, the 26B-A4B NVFP4 recipe (~52 tok/s),
    quantization, multimodal / thinking / tool-calling.
  - **[research-digests/](AsusGx10/research-digests/)** — auto-generated, on-device literature
    digests built by the local model (e.g. major discoveries in efficient LLM inference).

## Conventions

- **Cite everything.** Every non-trivial claim traces to a saved source (see each
  subproject's `sources/`) or to a measurement on the author's hardware.
- **Layered docs.** Guides open with a plain-language on-ramp and build into advanced
  detail in the same flow, so newcomers and operators can both use them.
- **Reproducible scripts.** Ops scripts mirror the exact commands used on the hardware.

## License

[MIT](LICENSE) © 2026 Heitor Mocelin
