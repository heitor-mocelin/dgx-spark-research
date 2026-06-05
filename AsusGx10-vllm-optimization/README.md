# AsusGx10-vllm-optimization

Optimizing **vLLM** serving of **Qwen3-class MoE models** on an **ASUS Ascent GX10**
(NVIDIA **GB10 Grace Blackwell**, 128 GB LPDDR5, DGX OS / Ubuntu ARM64) — from a bare
device to a tuned, OpenAI-compatible inference endpoint.

> **Status:** work in progress. Research corpus and guides are being assembled.

## Hardware & baseline (this device)

| | |
|---|---|
| Device | ASUS Ascent GX10 (DGX Spark class) |
| SoC | NVIDIA GB10 Grace Blackwell |
| Memory | 128 GB LPDDR5 (unified) |
| Server | vLLM (OpenAI-compatible), Qwen3-class ~35B MoE, NVFP4 weights, FP8 KV cache |
| Context | 32k | 
| Baseline throughput | ~600 tok/s (to be re-measured under controlled benchmarks) |

## What's inside

- **[guides/](guides/)** — a layered series, divided by step/discovery. Each guide starts
  with a plain-language on-ramp and builds into advanced detail (kernels, quantization
  internals, batching math). Priority order: **throughput → NVFP4/FP8 quantization →
  latency → tool-calling & long context**.
- **[scripts/](scripts/)** — reproducible shell scripts: launch, benchmark, tune, rollback.
- **[sources/](sources/)** — the cited research corpus with provenance (see
  [sources/README.md](sources/README.md)).

## Scope

Focused on **this hardware** (ASUS Ascent GX10 / GB10). Findings may generalize to other
DGX Spark / GB10 systems, but only claims validated here or traced to a cited source are
presented as fact.

## License

[MIT](../LICENSE) © 2026 Heitor Mocelin
