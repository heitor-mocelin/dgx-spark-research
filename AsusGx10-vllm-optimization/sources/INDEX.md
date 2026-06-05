# Source corpus index

Batch 1 — 10 sources, retrieved 2026-06-05 (on-device via OpenClaw LXC: trafilatura
for HTML, raw markdown for GitHub). Weighted to the project priority order
(throughput → quantization → latency → tool-calling/context → platform). See each file's
YAML front matter for full provenance, and [README.md](README.md) for conventions.

| ID | Title | Publisher | Primary topics |
|----|-------|-----------|----------------|
| [001](001-vllm-optimization-tuning.md) | Optimization and Tuning | vLLM docs | throughput, latency, batching |
| [002](002-vllm-quantized-kv-cache.md) | Quantized KV Cache | vLLM docs | quantization, fp8, memory, context |
| [003](003-nvidia-introducing-nvfp4.md) | Introducing NVFP4 | NVIDIA Technical Blog | quantization, nvfp4, blackwell |
| [004](004-nvidia-spark-nvfp4-playbook.md) | DGX Spark Playbook: NVFP4 Quantization | NVIDIA (GitHub) | quantization, nvfp4, platform |
| [005](005-nvidia-spark-vllm-playbook.md) | DGX Spark Playbook: vLLM for Inference | NVIDIA (GitHub) | throughput, platform, serving |
| [006](006-vllm-env-vars.md) | Environment Variables | vLLM docs | backends, moe, flashinfer, quantization |
| [007](007-vllm-benchmark-cli.md) | Benchmark CLI | vLLM docs | latency, throughput, ttft, itl |
| [008](008-vllm-tool-calling.md) | Tool Calling | vLLM docs | tool-calling, parsers, qwen3 |
| [009](009-vllm-recipes-qwen35-qwen36.md) | Qwen3.5 & Qwen3.6 Usage Guide | vLLM Recipes | moe, quantization, throughput, tool-calling, context |
| [010](010-nvidia-spark-sw-optimizations.md) | Software/Model Optimizations Supercharge DGX Spark | NVIDIA Technical Blog | platform, throughput, quantization, nvfp4 |

## Priority coverage (batch 1)

- **Throughput:** 001, 005, 009, 010
- **Quantization (NVFP4/FP8):** 002, 003, 004, 006, 009, 010
- **Latency (TTFT/ITL):** 001, 007
- **Tool-calling & context:** 008, 009, 002
- **Platform (GB10 / DGX Spark):** 004, 005, 010

## Known gaps to fill in later batches

- Dedicated **SM120/SM121 (GB10) FP4 MoE backend** thread / "GPU lacks native FP4 → Marlin
  kernel" specifics (vLLM issues #33333/#33416, PR #40082).
- **`vllm bench` throughput** examples on Grace-Blackwell and community **DGX Spark NVFP4
  benchmark** numbers (e.g. mark-ramsey-ri/vllm-dgx-spark presets, adadrag/qwen3.5-dgx-spark).
- **FlashInfer / CUTLASS MoE** backend internals and `--moe-backend` selection details.
