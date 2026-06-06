# Findings — what measuring 7 NVFP4 models on the GX10 actually taught us

> **The method, in one line:** for each NVFP4 model we *predicted* a decode-throughput ceiling from
> first principles, *measured* the real GB10, and *compared*. The value isn't the raw tok/s — it's
> where the prediction held, where it broke, and **why**. This page is the synthesis; the numbers
> live in [benchmarks/](benchmarks/README.md), the theory in [testing-plan.md](testing-plan.md).

---

## The one-paragraph version

On a bandwidth-limited box, **active parameters set single-stream speed** — that part of the roofline
held beautifully (70 → 5 tok/s as models grow; a 3B-active MoE is ~10× faster than a 31B dense). But
the *efficiency* of hitting that ceiling is **not constant** — it climbs from ~42% for tiny-active
MoEs to ~98% for big dense models, because a large weight-read drowns out fixed per-token overhead.
Architecture adds ±15% scatter, aggregate throughput is **power-capped** (not bandwidth-capped), and
the Marlin FP4 fallback shows up as *visible compute*, not idle waiting. Net: **for local interactive
use on the GX10, a 3B-active MoE is the only sensible choice** — and now we can say so with data.

![Predicted vs measured, and efficiency vs active params](assets/roofline-measured-gx10.png)

---

## The measured table

| Model (NVFP4) | Type | Active | Predicted (realistic) | **Measured single** | **Eff.** | Peak agg | Power |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.3-70B | dense | 70B | 5.5 | **5.4** | **98%** | 432 | 71 W |
| Qwen3-32B | dense | 32B | 12.1 | **11.0** | **91%** | 884 | 62 W |
| Qwen3.6-35B-A3B | MoE | 3.0B | 129 | **75** | 58% | 951 | — |
| Gemma-4-31B | dense | 31B | 12.5 | **6.8** | 54% | 513 | 61 W |
| Nemotron-3-Super-120B-A12B | MoE | 12B | 32.4 | **14.6** | 45% | 327 | 56 W |
| Nemotron-3-Nano-30B-A3B | MoE | 3.0B | 129 | **54.1** | 42% | 1215 | 44 W |
| Gemma-4-26B-A4B | MoE | 3.8B | 102 | *deploy failed* | — | — | — |

*3× repeated, std ≤ 0.1 tok/s — essentially deterministic. Efficiency = measured ÷ realistic ceiling.*

---

## Discovery 1 — the roofline predicts *order* perfectly; *magnitude* only for big models

Single-stream speed tracks active parameters exactly as the `bandwidth ÷ (active × bytes)` model says:
the ranking from 70B-dense (5.4) up to 3B-MoE (75) is monotonic, and the ~10× active-param gap between
a 31B dense and a 3B MoE produces the predicted ~7–10× speed gap. **The roofline is the right mental
model.** What it *can't* do is nail the absolute number for small models — which is Discovery 2.

## Discovery 2 — efficiency **rises with active params** (our headline hypothesis, refuted)

We pre-registered a guess: a *model-independent* ~55% efficiency factor. **Six data points refuted it**
(span 42%→98%). The replacement is sharper and mechanistic:

```
 efficiency (measured ÷ realistic ceiling)
 100% ┤                                          ● Llama-70B (98%)
      │                                  ● Qwen3-32B (91%)
  ~55%┤·········· the guess that failed ··················
      │   ● Qwen3.6 (58%)    ● Gemma-31B (54%, outlier ↓)
  40% ┤ ● Nemotron-Nano(42%)  ● Nemotron-Super(45%)
      └────────────────────────────────────────────────────
        3B            12B          31–32B            70B
```

**Why (the mechanism):** the roofline counts *only* the active-weight read. Every other per-token cost
— KV reads, attention, sampling, scheduler/Python, **Marlin decompress setup** — is roughly fixed. For
a 70B dense model the ~39 GB weight-read **dwarfs** those, so it runs at **98% of its ceiling**. For a
3B-active MoE the weight-read is ~1.7 GB, so the same fixed overhead is a large fraction → **~42–58%**.
**Takeaway: trust the roofline as a tight estimate for big/dense models, and as a loose upper bound
(×~0.5) for small-active MoEs.**

## Discovery 3 — architecture matters beyond the roofline (±15%, and Gemma is an outlier)

- **Same active params, different speed:** Qwen3.6 and Nemotron-Nano are *both* 3B-active MoEs with an
  identical predicted ceiling, yet measured **75 vs 54 tok/s** — a 40% spread from architecture/kernel
  quality alone.
- **Gemma-4-31B underperforms its class:** **54%** efficiency vs Qwen3-32B's **91%** at nearly identical
  size. The same hybrid sliding/global attention + Per-Layer-Embeddings that make Gemma memory-thrifty
  make it kernel-heavier on `sm_121`. *A model's efficiency is not separable from its kernel support.*

## Discovery 4 — peak aggregate is **power/compute-capped, not bandwidth-capped**

Single-stream is bandwidth-bound; **saturation is not**. Every model pegs **96% GPU-util** and tops out
at **44–71 W** under load — so the aggregate ceiling is the device's power/compute budget:
- 3B-active MoEs win aggregate (**Nemotron-Nano 1215**, **Qwen3.6 951 tok/s**) — least compute/token.
- Power rises with density: **Llama-70B draws 71 W for the least throughput**.
- Practical consequence: batching helps until ~96% util / the power cap, then stops — the
  [Qwen concurrency sweep](vllm-qwen3.6-35b-a3b/benchmarks/README.md) (peak ~951 @ c≈96, regressing at
  128) is this wall, measured.

## Discovery 5 — the "Marlin tax" is visible as compute, not idle-wait

We expected single-stream to show *low* GPU-util (bandwidth-bound = math units waiting on memory).
Instead it's **89–96% even at batch 1** — because Marlin **decompresses FP4→BF16 on-chip every token**,
keeping the SMs busy. So on the GB10 the `sm_121` FP4 fallback isn't free headroom sitting idle; it's
*active compute overhead*. This is the strongest evidence yet that a **native FP4-MoE kernel
(FlashInfer b12x / vLLM PR #40082) should recover real throughput** — especially on the small-active
MoEs where overhead dominates (Discovery 2). *That experiment is the top open item.*

> **Telemetry honesty:** `DCGM_FI_DEV_MEM_COPY_UTIL` reads **0% on the GB10** for every model — Grace-
> Blackwell's unified memory doesn't populate that discrete-GPU counter, so we can't read bandwidth
> saturation directly. GPU-util + power carried the analysis instead.

---

## What this means in practice — the GX10 model-selection map

| If you want… | Run | Because (from the data) |
|---|---|---|
| **Snappy single-user chat/agent** | a **3B-active MoE** (Qwen3.6-35B-A3B = 75 tok/s) | fastest single-stream; MoE beats every dense model 5–14× |
| **Max aggregate (many users)** | Nemotron-Nano (1215) or Qwen3.6 (951 tok/s) | least compute/token → highest under batching |
| **Best dense quality, latency-tolerant** | Llama-3.3-70B (5.4 tok/s, but 98% efficient) | runs at its ceiling; just a slow ceiling |
| **Avoid for interactive** | any dense 31B+ (5–11 tok/s) | bandwidth wall; quantization can't fix dense |
| **Watch out** | Gemma-4-31B | underperforms its size on `sm_121` today |

---

## Honest status

- ✅ **7 models predicted; 6 measured** (3× averaged + telemetry), published with raw JSON.
- ⚠️ **Gemma-4-26B-A4B deploy failed** — community NVFP4 MoE checkpoint patch mismatched this vLLM
  nightly; cited 52 tok/s used as a placeholder. Retry with a matched build.
- ⏳ **Marlin-vs-native FP4** — the experiment Discovery 5 motivates; not yet run.
- 🔁 **We were wrong once, on purpose:** the constant-efficiency hypothesis failed, and that failure
  produced Discovery 2. Pre-register, measure, update.

See [testing-plan.md](testing-plan.md) for the methodology and [benchmarks/](benchmarks/README.md) for
every number and the reproduction scripts.
