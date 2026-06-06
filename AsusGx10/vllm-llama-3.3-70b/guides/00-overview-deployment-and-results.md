# 00 · Llama-3.3-70B on the GX10 — overview, deployment & measured results

> The biggest dense model we tested, and the most instructive: it runs at **98% of its theoretical
> ceiling**. That sounds great until you see the ceiling is **5.4 tok/s**. This guide is the story
> of "efficient, but slow" — and why that's exactly what the roofline predicts.

## Plain-language on-ramp

Llama-3.3-70B is a **dense** 70.6 B model — all 70 B parameters fire every token. On the GX10 that
means reading ~39 GB of weights *per token* from 273 GB/s memory, which caps you around 5–6 tok/s no
matter what. The good news: there's so little overhead relative to that giant read that the box
delivers **98% of the theoretical max**. The bad news: 98% of a small number is still a small
number. This is the model to reach for when you want **Llama-class quality** and can live with
slow, batched serving — not snappy chat.

## Architecture

A mature, conventional transformer [[s1]](../sources/INDEX.md):

- **80 layers**, hidden size **8192**, **Grouped-Query Attention** (keeps the KV cache manageable
  at 128K context).
- Instruction-tuned via **SFT + RLHF**; text-only; 8 languages.
- No MoE, no exotic layers — like Qwen3-32B, a *clean* roofline test, just bigger.

## NVFP4 deployment

NVFP4 brings the 70.6 B weights to **~39 GB** — comfortably inside 128 GB, with ~70 GB left for KV.
The launch ([`scripts/launch.sh`](../scripts/launch.sh)) is the standard NVFP4 recipe
(`--quantization modelopt`, FP8 KV, `sm_121` env). Deployed cleanly from `nvidia/Llama-3.3-70B-Instruct-NVFP4`.

## Measured results (2026-06-06)

| Concurrency | 1 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate tok/s | 5.4 | 21.2 | 42.2 | 83.0 | 159.1 | 234.6 | **431.8** | 431.5 |

- **Single-stream: 5.4 tok/s** (TTFT 198 ms, ITL 186 ms).
- **Predicted ceiling:** 6.9 peak, **5.5 realistic** → measured **98%** — the highest in the matrix.
- **Telemetry:** single-stream GPU 96% / 32 W; saturated GPU 96% / **71 W (matrix-max power)**.

### Reading the result (the "why")

- **98% = the roofline, basically reached.** A 39 GB/token read leaves the fixed overheads
  (KV, attention, scheduler, Marlin decompress) as a rounding error → the measurement sits right on
  the bandwidth ceiling. This is the **clean confirmation** of [FINDINGS Discovery 1 & 2](../../FINDINGS.md):
  the roofline is *most* predictive for large/dense models.
- **Most power for least speed.** 71 W at saturation for ~432 tok/s aggregate — vs Nemotron-Nano's
  44 W for **1215** tok/s. Dense bigness costs energy *and* throughput.
- **Linear batching to ~c96**, then the power cap. Even batched, the aggregate (432) is the lowest
  of the matrix's working models — a 70 B dense model is simply expensive per token.

## Where it fits on the GX10

Pick Llama-3.3-70B when you specifically want **70 B dense Llama quality** and your workload is
**latency-tolerant / batch-oriented** (offline generation, eval harnesses). For anything interactive
or high-throughput on this box, a 3 B-active MoE is **10–14× faster** — see [FINDINGS](../../FINDINGS.md).

## Sources

[sources/INDEX.md](../sources/INDEX.md) — Llama 3.3 model card + Meta materials.
