# Source corpus index

20 cited sources, retrieved 2026-06-05, fetched on-device via the OpenClaw LXC
(trafilatura for HTML, raw markdown for GitHub READMEs, `gh` CLI for issue/PR threads).
Weighted to the project priority order (throughput → quantization → latency →
tool-calling/context → platform). See each file's YAML front matter for full provenance,
and [README.md](README.md) for conventions. Research is considered complete at this batch.

## Batch 1 — core docs (vLLM + NVIDIA)

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
| [010](010-nvidia-spark-sw-optimizations.md) | SW/Model Optimizations Supercharge DGX Spark | NVIDIA Technical Blog | platform, throughput, quantization, nvfp4 |

## Batch 2 — GB10/SM121 specifics, backends, benchmarks, hardware

| ID | Title | Publisher | Primary topics |
|----|-------|-----------|----------------|
| [011](011-vllm-pr40082-flashinfer-b12x-sm121.md) | PR #40082: FlashInfer b12x MoE + FP4 GEMM for SM120/121 | GitHub (vLLM) | backends, moe, nvfp4, sm121, flashinfer |
| [012](012-vllm-issue33333-flashinfer-cutlass-sm120.md) | Issue #33333: FLASHINFER_CUTLASS NVFP4 MoE unsupported on SM120 | GitHub (vLLM) | backends, moe, nvfp4, sm121, marlin |
| [013](013-nvidia-dgx-spark-hardware-overview.md) | DGX Spark Hardware Overview (273 GB/s, specs) | NVIDIA DGX docs | platform, memory, bandwidth, hardware |
| [014](014-community-adadrag-qwen35-dgx-spark.md) | Qwen3.5-35B-A3B on DGX Spark — single-box benchmarks | GitHub (community) | benchmarking, platform, throughput, nvfp4 |
| [015](015-community-markramsey-vllm-dgx-spark.md) | vllm-dgx-spark preset matrix (FP8/NVFP4/MXFP4) | GitHub (community) | platform, quantization, serving, benchmarking |
| [016](016-community-bjk110-spark-nvfp4-mtp.md) | Single-GPU NVFP4 W4A4 + MTP, FlashInfer-from-source SM121 | GitHub (community) | nvfp4, backends, sm121, throughput, speculative |
| [017](017-flashinfer-readme.md) | FlashInfer: kernel library for LLM serving | GitHub (FlashInfer) | backends, kernels, attention, moe |
| [018](018-awesome-dgx-spark-index.md) | Awesome DGX Spark — curated GB10 ecosystem index | GitHub (community) | platform, ecosystem, benchmarking |
| [019](019-nvidia-nvfp4-kv-cache-long-context.md) | NVFP4 KV Cache for Long Context & Large Batch | NVIDIA Technical Blog | quantization, nvfp4, context, memory, kv-cache |
| [020](020-nvidia-ptq-performance-accuracy.md) | Optimizing LLMs with Post-Training Quantization | NVIDIA Technical Blog | quantization, ptq, accuracy, calibration |

## Priority coverage

- **Throughput:** 001, 005, 009, 010, 014, 015, 018
- **Quantization (NVFP4/FP8):** 002, 003, 004, 006, 009, 010, 019, 020
- **Latency (TTFT/ITL, spec-decode):** 001, 007, 009, 016
- **Tool-calling & context:** 008, 009, 002, 019
- **Platform / GB10-SM121:** 004, 005, 010, 011, 012, 013, 016, 017, 018
- **Empirical single-box benchmarks:** 014, 015, 016

## Gaps closed in batch 2

- **SM121/GB10 FP4-MoE → Marlin** ("GPU lacks native FP4" log): 011, 012, 016 — the
  FLASHINFER_CUTLASS device-capability gap, the breaking commit / FlashInfer issue #2077 /
  PR #33417, and the b12x kernel fix (PR #40082).
- **Hardware/bandwidth ceiling:** 013 — 128 GB LPDDR5x, 256-bit @ 4266 MHz, **273 GB/s**.
- **Single-GB10 benchmark numbers:** 014, 015, 016.
- **FlashInfer / CUTLASS MoE backend:** 017 (+ 011/012 for `--moe-backend` selection on SM121).

> Note: the canonical `vllm serve` flag reference and the spec-decode feature page are
> JS-rendered and don't extract cleanly to Markdown; the relevant flags are instead captured
> across 001, 006, 009, and the serve examples in 005/014/015. Use `vllm serve --help=all`
> on the device for the exhaustive list.
