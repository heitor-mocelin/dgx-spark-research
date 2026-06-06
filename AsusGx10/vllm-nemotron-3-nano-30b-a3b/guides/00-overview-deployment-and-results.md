# 00 · Nemotron-3-Nano-30B-A3B on the GX10 — overview, deployment & measured results

> This model breaks the tidy "active params → speed" story in an interesting way. It has the *same*
> 3 B active parameters as Qwen3.6, so the roofline predicts the same ceiling — yet it measures
> slower single-stream but **faster in aggregate** and at **lower power**. The reason is that it
> isn't a pure transformer: it's a **hybrid Mamba-2 + MoE**.

## Plain-language on-ramp

Most LLMs are transformers — every token attends to every previous token, and the KV cache grows
with context. **Mamba** layers are different: they're *state-space models* that carry a fixed-size
recurrent state instead of a growing KV cache. Nemotron-3-Nano **mixes both** — mostly Mamba-2 + MoE
layers with a few attention layers sprinkled in. The result is a model that's cheap to run at long
context (small, fixed state) and very throughput-friendly, but whose per-token cost doesn't match
the attention-weight-read roofline we used to predict it.

## Architecture (the deep dive)

A **hybrid Mixture-of-Experts** [[s1]](../sources/INDEX.md):

- **23 Mamba-2 & MoE layers + 6 Attention layers.** The Mamba-2 state-space layers replace most of
  the attention, giving near-constant per-token cost regardless of context length.
- **MoE:** 128 routed experts, **top-6 routing + 2 shared** experts — ~3.2 B active of 31.6 B total.
- **Context up to 1 M tokens** — the Mamba state, not a growing KV cache, is what makes that
  affordable.
- Reasoning/agentic-tuned, ~25 T training tokens.

**Why this bends the roofline:** our `bandwidth ÷ (active × bytes)` ceiling assumes the per-token
cost is *reading attention weights*. Mamba layers do recurrent state updates with a different
compute/memory profile, so the "active parameter" count is a looser proxy for the actual per-token
work — hence the 42% efficiency (vs Qwen3.6's 58% at the same active count).

## NVFP4 deployment

~19 GB in NVFP4. Standard vLLM recipe ([`scripts/launch.sh`](../scripts/launch.sh)) with
`--quantization modelopt` + `--trust-remote-code` (NVIDIA custom architecture). Deployed cleanly
from `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`.

## Measured results (2026-06-06)

| Concurrency | 1 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate tok/s | 56.1 | 187.8 | 344.4 | 572.2 | 861.0 | 982.4 | **1215.3** | 1214.1 |

- **Single-stream: 54.1 tok/s** (TTFT 80 ms, **ITL 17 ms** — fast per-token).
- **Predicted ceiling:** 162 peak, 129 realistic → **42%** (the matrix minimum efficiency).
- **Telemetry:** GPU 96% throughout; **44 W at peak — the lowest power in the matrix.**

### Reading the result (the "why")

- **Lowest efficiency, highest aggregate, lowest power — all at once.** The hybrid architecture
  leaves more single-stream performance "on the table" vs the roofline (42%), but its lean per-token
  compute lets it **batch to 1215 tok/s at just 44 W** — the most throughput-per-watt of anything
  tested. For a busy agent gateway, this is arguably the *best* model on the box.
- **Architecture > active-param count.** Same 3 B active as Qwen3.6, 54 vs 75 single-stream — proof
  the roofline predicts *order*, not exact magnitude, once you leave pure-transformer territory
  ([FINDINGS Discovery 3](../../FINDINGS.md)).
- **Mamba's long-context promise:** 1 M context without a ballooning KV cache — untested here but a
  strong reason to pick this model for long-document/agentic work.

## Where it fits on the GX10

The **throughput-and-efficiency champion** for concurrent serving (1215 tok/s @ 44 W), and a strong
long-context option. For raw single-user latency, Qwen3.6 edges it (75 vs 54). See
[FINDINGS](../../FINDINGS.md).

## Sources

[sources/INDEX.md](../sources/INDEX.md) — Nemotron-3 Nano technical report + model card.
