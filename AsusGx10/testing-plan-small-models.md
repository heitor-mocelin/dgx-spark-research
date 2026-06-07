# Testing plan — small "reflex" models: who answers fastest on the GX10

> **Why this study exists.** The main [study](testing-plan.md) chased *aggregate throughput* and
> *roofline efficiency* across 30B–120B models. This one asks a different question: **which small,
> instruction-tuned, quantized model gives the snappiest single-user answer** — a "reflex" model to
> sit *alongside* the production **Qwen3.6-35B-A3B** on the same 128 GB box. For interactive Q&A the
> metric that matters isn't peak tok/s, it's **how fast the first token lands (TTFT)** and how fast it
> sustains after. We predict first (roofline), then measure, then attribute the gap — same discipline
> as the parent study.

---

## 1. The 7 candidates (all instruction-tuned, all quantized)

| # | Model | Arch | Active params | Quant | Why it's here |
|---|---|---|---:|---|---|
| 1 | **Gemma-4-E4B-it** | dense + PLE ("effective 4B") | ~4B (less via PLE) | NVFP4 | Edge-tuned, 128K ctx, multimodal; community's fastest small on this box |
| 2 | **Qwen3-4B-Instruct-2507** | dense | 4.0B | NVFP4 | Same family as the main model → consistent behavior; likely speed champion |
| 3 | **Qwen3-8B** | dense | 8.2B | NVFP4 (nvidia) | Quality/speed sweet spot, same family |
| 4 | **Llama-3.1-8B-Instruct** | dense | 8.0B | NVFP4 (nvidia) | The community baseline everyone benchmarks against |
| 5 | **Ministral-8B-Instruct-2410** | dense | 8.0B | NVFP4 if avail, else FP8/AWQ | Praised for speed + instruction-following; 3rd 8B family for cross-check |
| 6 | **Phi-4-mini-instruct** | dense | 3.8B | NVFP4/FP8 | Best reasoning-per-param in the tiny class — "small but smart" |
| 7 | **gpt-oss-20b** | MoE | 3.6B of 20B | **MXFP4** (native) | Different arch + format; strong agentic/tool model; small-active = fast |

**Spread by design:** an effective-MoE (Gemma), a dense-4B (Qwen), three dense-8B across families
(Qwen / Llama / Ministral — isolates *family* from *size*), a reasoning-dense tiny (Phi), and a
small-active MoE in a second quant format (gpt-oss). One quant note worth flagging up front:
#5–#7 may not have a clean `nvidia/*-NVFP4` checkpoint — I'll use NVFP4 where it exists and fall back
to FP8 / a reputable community quant, **recording the format per model** so comparisons stay honest.

---

## 2. Predict first — the roofline ceiling per model

Same model as the parent study: single-stream decode is bandwidth-bound, so

```
                    realistic bandwidth (≈ 0.8 × 273 GB/s ≈ 218 GB/s)
 ceiling (tok/s) ≈ ───────────────────────────────────────────────────
                            active_params × bytes_per_param
```

NVFP4 = **0.5625 B/param**; MXFP4 ≈ **0.53 B/param** (4-bit weights + block scales, attention in BF16).

| Model | Active | B/param | Bytes/token | **Peak ceiling** | **Realistic (×0.8)** | Expected efficiency¹ | **Predicted measured** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 4.0B | 0.5625 | 2.25 GB | 121 tok/s | 97 | 55–70% | **~65–85 tok/s** |
| Gemma-4-E4B² | ~3B eff | 0.5625 | ~1.7 GB | 162 | 130 | high (PLE) | **~120–150** (community: 149) |
| Phi-4-mini | 3.8B | 0.5625 | 2.14 GB | 128 | 102 | 55–65% | **~60–80** |
| Qwen3-8B | 8.2B | 0.5625 | 4.6 GB | 59 | 47 | 60–75% | **~35–45** |
| Llama-3.1-8B | 8.0B | 0.5625 | 4.5 GB | 61 | 49 | 60–75% | **~37–46** |
| Ministral-8B | 8.0B | 0.5625 | 4.5 GB | 61 | 49 | 60–75% | **~37–46** |
| gpt-oss-20b | 3.6B | 0.53 | 1.9 GB | 144 | 115 | 40–50% (small MoE) | **~50–70** (community: 50–70) |

¹ From the parent study's measured curve: efficiency *rises* with active params; small-active models
leave more on the table to fixed overhead. ² Gemma's Per-Layer Embeddings reduce *effective* active
params, so it can **beat the naive-4B ceiling** — a hypothesis this study will test directly.

**Falsifiable predictions before we run:** (a) the 4B-class beats the 8B-class on TTFT and tok/s;
(b) Gemma-E4B exceeds its naive roofline thanks to PLE; (c) the three 8B dense models land within
~15% of each other (size dominates family); (d) gpt-oss is fast-to-first-token but mid-pack on
sustained decode due to small-MoE overhead.

---

## 3. What we measure (interactive-first metric set)

| Metric | What it captures | Why it matters here | Primary? |
|---|---|---|:--:|
| **TTFT** (time-to-first-token) | latency from request → first token | *the* felt-speed metric for short answers | ⭐ |
| **Single-stream decode tok/s** | sustained generation, 1 user | how fast it "types" the answer | ⭐ |
| **End-to-end latency** | TTFT + decode for a ~256-tok answer | the real wait for a typical reply | ⭐ |
| **Prefill throughput** (prompt tok/s) | prompt ingestion speed | matters for long prompts / RAG | ◦ |
| **Concurrency sweep** (c=1,4,16,32) | aggregate tok/s + per-req latency | finds the knee if the reflex model gets reused | ◦ |
| **tok/s-per-watt** | efficiency under the power cap | the GX10 is power-capped, not bandwidth-capped at load | ◦ |
| **Memory footprint** | resident unified memory | gates co-residency with Qwen (§6) | ⭐ |

---

## 4. Methodology (matches the parent study's rigor)

- **3× per measurement, averaged** (report mean ± stdev); a warm-up pass is discarded.
- **Fixed token budgets** so models are compared on equal work: prefill prompts at ~256 / ~2048
  input tokens; decode capped at 256 output tokens; TTFT measured on a short prompt.
- **Thinking mode OFF** for the headline latency numbers (`enable_thinking:false` / non-reasoning) —
  Qwen3, Phi, and gpt-oss can emit long reasoning traces that destroy TTFT. Thinking-on is measured
  *separately* and labeled, never mixed in.
- **Harness:** vLLM's `benchmark_serving.py` + a small custom TTFT probe, hitting the OpenAI endpoint
  on `:8000`. Same container image and flags family as production (`--quantization modelopt`,
  `--kv-cache-dtype fp8`, `--enable-chunked-prefill`); gpt-oss uses its MXFP4 path + harmony parser.
- **One model loaded at a time** for the standalone runs (no memory contention).
- **Deviations logged:** quant format per model, parser, any flag that differs, the `sm_121` Marlin
  FP4→BF16 fallback note.

---

## 5. Telemetry (humans + agents)

Reuse the existing observability stack: **Prometheus** (`172.27.27.212:9090`) scraping **DCGM**
(`DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_UTIL`) → **Grafana** (`172.27.27.215:3000`,
"Homelab/DGX Spark" dashboard). Each run window is timestamped so power/util can be joined to the
tok/s numbers and a real **tok/s-per-watt** computed. (`MEM_COPY_UTIL` reads 0 on GB10 — unified
memory; noted, not used.)

---

## 6. Co-residency protocol — can a reflex model live next to Qwen3.6?

The actual goal: a permanent "fast lane" beside the production model on the 128 GB box. For each
candidate, after the standalone run:
1. Start production **Qwen3.6** (`:8000`, the frozen production `docker run`, util ~0.55 to leave headroom).
2. Start the **candidate** as a *second* vLLM container on **`:8001`** with a conservative
   `--gpu-memory-utilization` sized to its footprint (§3) + margin.
3. Record **combined resident memory** (must stay well under 128 GB — the OOM-wedge lesson), then
   re-measure the candidate's **TTFT + decode** (a) while Qwen is idle and (b) while Qwen serves a
   concurrent load → quantify interference.
4. **Hard safety rails:** never exceed ~0.55 util on the big model, watch `free`/`nvidia-smi`, abort
   on any NVRM pressure. One candidate co-resident at a time. (We hard-crashed the box once at 0.85
   on a 60 GB model — not repeating it.)

A candidate "passes" co-residency if both models stay healthy, combined memory has margin, and the
candidate keeps near-standalone TTFT while Qwen is idle (the expected steady state).

---

## 7. Light quality gate (so we don't ship fast-but-dumb)

A fixed **~15-prompt** set — strict format-following, JSON-only output, a short multi-step reasoning
item, a tool/function-call, a refusal/safety check, a couple of factual + summarization tasks. Scored
by a rubric (pass/fail per criterion); optionally the production Qwen3.6 acts as a judge. **Not** a
full benchmark — a smoke gate so the recommendation balances speed against basic competence.

---

## 8. Deliverables & success criteria

- **This plan** (predictions) → **scripts** (`benchmarks/small-models/`) → **results JSON** per model
  → a **`FINDINGS-small-models.md`** with the measured-vs-predicted table, a TTFT/tok-s chart, and
  **the recommendation: the 1–2 keepers** to run resident alongside Qwen, with the deploy command.
- Per-model subproject folders only for the winner(s), following the established convention.
- **Success =** every prediction in §2 confirmed or explained, a clear "fastest snappy answer" winner,
  and a validated co-residency config the bootstrap script could later adopt as a second "fast lane."

---

## 9. Open items to confirm before running
- Quant availability for Ministral-8B / Phi-4-mini in NVFP4 (else FP8/AWQ — will record).
- gpt-oss-20b needs its MXFP4 + harmony-format serving path (separate from the modelopt NVFP4 path).
- Estimated hardware time: ~7 models × (standalone + co-resident, 3× each) ≈ a few hours; one model
  at a time, production Qwen restored between heavy runs.
