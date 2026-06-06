# 00 · Qwen3-Next-80B-A3B on the GX10 — overview, deployment & measured results

> Another 3B-active MoE — so the roofline predicts the *same* 129 tok/s realistic ceiling as
> Qwen3.6 and Nemotron-Nano. It measured **35.5**, the lowest fraction of any model (27%). This
> guide is about *why* a linear-attention hybrid breaks the roofline harder than anything else.

## Plain-language on-ramp

Qwen3-Next is an 80-billion-parameter model that activates only **3 billion** per token — extreme
sparsity. But its bigger novelty is **how it does attention**: instead of standard attention on
every layer, three out of every four layers use **Gated DeltaNet**, a *linear-attention* mechanism
that carries a small recurrent state rather than a growing KV cache. That makes long context cheap
— but, as the GX10 numbers show, it also changes the performance profile in a way the
weight-bandwidth roofline doesn't capture.

## Architecture (the deep dive)

[[s1]](../sources/INDEX.md) — a hybrid, ultra-sparse MoE:

- **Gated DeltaNet (linear attention) + Gated Attention in a 3:1 ratio** — most layers are linear,
  with periodic full-attention layers to retain precision.
- **MoE at ~1:50 activation** — 3B of 80B active per token.
- **Context**: 256K native, 1M with YaRN.
- Reaches dense-Qwen3-32B quality at <10% of the training cost.

## NVFP4 deployment

~48 GB in NVFP4 (`--quantization modelopt`, FP8 KV, `--trust-remote-code` for the Qwen3-Next custom
architecture — vLLM has supported it since Sept 2025). It **deployed cleanly** and at
`--gpu-memory-utilization 0.85` was safe (48 GB is well within the 128 GB box — unlike the 120B
models, see the note below). Recipe: [`scripts/launch.sh`](../scripts/launch.sh).

> ⚠️ **Memory lesson from this matrix:** 0.85 util is fine for a ~48 GB model but **wedged the box
> on GPT-OSS-120B** (~61 GB) — vLLM reserves that fraction of the *shared* 128 GB without accounting
> for the host + other containers. For 60 GB+ models, use ~0.55–0.6. (Details in [FINDINGS](../../FINDINGS.md).)

## Measured results (2026-06-06)

| Concurrency | 1 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate tok/s | 36.5 | 133 | 250 | 426 | 639 | 790 | **1021** | 1002 |

- **Single-stream: 35.5 tok/s** (TTFT 91 ms, ITL 27 ms).
- **Predicted ceiling:** 162 peak, 129 realistic → **~27% efficiency — the matrix minimum.**
- **Telemetry:** single-stream **GPU 60.6% / 27 W** (also the matrix minimum util); saturated GPU
  90% / 51 W.

### Reading the result (the "why")

- **Lowest efficiency *and* lowest single-stream GPU util.** Every other model pegged 89–96% GPU
  util at batch 1 (Marlin keeping the SMs busy). Qwen3-Next sits at **60%** — so it's *not*
  compute-bound and *not* Marlin-bound. The most likely explanation: **the Gated DeltaNet recurrent
  state update is serial per token** (a true recurrence, unlike parallel attention), so each token
  has latency the GPU can't fill → the chip idles → low single-stream throughput. The roofline,
  which only counts weight reads, can't see this.
- **Same 3B active, very different speed:** Qwen3.6 (transformer) 75 → Nemotron-Nano (Mamba hybrid)
  54 → Qwen3-Next (DeltaNet hybrid) 35.5. A clean ordering: **the more "linear/recurrent" the
  attention, the further below the attention-roofline it falls** ([FINDINGS Discovery 3](../../FINDINGS.md)).
- **Batches well anyway:** 1021 tok/s peak @ c96 — concurrency hides the per-token latency, so for
  *serving* it's strong; it's *single-user* latency where the recurrence shows.

## Where it fits on the GX10

Compelling for **long-context, high-concurrency** serving (1M context via cheap linear-attention
state + 1021 tok/s aggregate). For single-user snappiness, the pure-transformer MoE (Qwen3.6) is
2× faster. See [FINDINGS](../../FINDINGS.md).

## Sources

[sources/INDEX.md](../sources/INDEX.md) — Qwen3-Next blog + model card + vLLM support post.
