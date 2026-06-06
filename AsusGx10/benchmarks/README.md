# Benchmark matrix — measured NVFP4 ceilings (2026-06-06)

> **What this is.** The execution of the [testing plan](../testing-plan.md): for each NVFP4 model we
> predicted a roofline ceiling, then measured the GX10, then compared. The headline is that the
> *prediction was partly wrong in an instructive way* — which is exactly why you measure.

Method: dedicated streaming HTTP client, **single-stream (5 reps)** + **concurrency sweep 1→128
(each point ×3, mean reported)** + **Prometheus dcgm telemetry** per phase. Each model deployed
solo (`--gpu-memory-utilization 0.85`, 32k ctx, Marlin FP4 fallback active). Raw JSON in
[`results/`](results/). Production Qwen restored after via the runner's trap.

## Results

| Model (NVFP4) | Type | Active | Predicted (realistic) | **Measured single** | **Eff.** | Peak agg | Power@peak |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.3-70B | dense | 70B | 5.5 | **5.4** | **98%** | 432 | 71 W |
| Qwen3-32B | dense | 32B | 12.1 | **11.0** | **91%** | 884 | 62 W |
| Qwen3.6-35B-A3B | MoE | 3.0B | 129 | **75** | 58% | 951 | — |
| Gemma-4-31B | dense | 31B | 12.5 | **6.8** | 54% | 513 | 61 W |
| Nemotron-3-Super-120B-A12B | MoE | 12B | 32.4 | **14.6** | 45% | 327 | 56 W |
| Nemotron-3-Nano-30B-A3B | MoE | 3.0B | 129 | **54.1** | 42% | 1215 | 44 W |
| Gemma-4-26B-A4B | MoE | 3.8B | 102 | *deploy failed* (cited 52) | ~51% | — | — |

*Efficiency = measured single-stream ÷ realistic ceiling (0.8 × peak). All single-stream values
had std ≤ 0.1 tok/s across reps — essentially deterministic.*

![Predicted vs measured + efficiency](../assets/roofline-measured-gx10.png)

## Finding 1 — the "constant ~55% efficiency" hypothesis is **refuted**

The [plan](../testing-plan.md#3-the-central-hypothesis-a-near-constant-efficiency-factor) guessed a
model-independent ~55% factor (from 3 early points). With 6 points it spans **42%→98%**. *Why it
looked constant:* the first three samples happened to cluster; more coverage broke it. Good — a
prediction met data and lost. The *replacement* finding is sharper:

## Finding 2 — efficiency **rises with active-parameter count**

```
 98% ┤                                              ● Llama-70B
     │                                       ● Qwen3-32B (91%)
 ~55%┤········· original guess band ·················
     │   ● Qwen3.6 (58%)   ● Gemma-31B (54%, outlier)
 42% ┤ ● Nemotron-Nano   ● Nemotron-Super (45%)
     └──────────────────────────────────────────────
       3B           12B        31-32B          70B   (active params)
```

**The mechanism (the "why"):** the roofline counts only the active-weight read. Every *other*
per-token cost — KV reads, attention, sampling, scheduler/Python, and the **Marlin FP4→BF16
decompress setup** — is roughly fixed. For a 70B dense model the weight read (~39 GB/token)
**dwarfs** those fixed costs, so it runs at **98% of its roofline**. For a 3B-active MoE the weight
read is tiny (~1.7 GB), so the same fixed overhead is a *large fraction* → only ~42–58%. **The
roofline is an excellent predictor for large/dense models and a loose upper bound for small-active
MoEs.**

## Finding 3 — architecture adds ±15% scatter (and Gemma is an outlier)

- Two **3B-active MoEs** with identical predicted ceilings measured very differently: **Qwen3.6 = 75**
  vs **Nemotron-Nano = 54 tok/s**. Same active params, ~40% spread → architecture/kernel quality
  matters beyond the roofline.
- **Gemma-4-31B (54%)** sits far below **Qwen3-32B (91%)** at nearly identical size. Its hybrid
  sliding/global attention + Per-Layer-Embeddings (or weaker `sm_121` kernel coverage) costs it —
  the same architecture that makes Gemma memory-thrifty makes it kernel-heavier here.

## Finding 4 — aggregate throughput is **power/compute-capped, not bandwidth-capped**

At saturation every model pegs **96% GPU-util** and tops out at **44–71 W**. So the *peak aggregate*
ceiling is set by the device's power/compute budget, not memory bandwidth:

- The 3B-active MoEs win aggregate (**Nemotron-Nano 1215**, **Qwen3.6 951 tok/s**) — least compute
  per token.
- Power climbs with model "density": Llama-70B draws the most (**71 W**) for the least throughput.

> **Telemetry caveat:** `DCGM_FI_DEV_MEM_COPY_UTIL` reads **0% on the GB10** for every model —
> Grace-Blackwell unified memory doesn't populate that discrete-GPU counter, so we can't read
> bandwidth saturation directly. Notably, **GPU-util is ~89–96% even at single-stream**, because
> Marlin's FP4→BF16 decompress keeps the SMs busy — the "Marlin tax" shows up as *visible compute*,
> not idle-wait. We classify the bound via tok/s-vs-roofline + power instead.

## Open items

- **Gemma-4-26B-A4B deploy failed** — the community `bg-digitalservices` NVFP4 MoE checkpoint hit
  *"Engine core initialization failed"* on this vLLM nightly (patch likely version-mismatched to the
  fused-3D-expert format). Cited value (52 tok/s, [g09](../vllm-gemma4-26b-a4b/sources/g09-ai-muninn-dgxspark-nvfp4-52.md)) used as a placeholder; retry with a matched build.
- **Marlin-vs-native FP4** still outstanding — Finding 4's high single-stream GPU-util suggests the
  Marlin decompress is a real compute cost; a b12x/PR-#40082 build should lift the small-active MoEs
  most (where overhead dominates). This is the next experiment.

## Reproduce

Scripts on the device (`/home/user/matrix/`): `bench_model.py` (3× + telemetry), `run_bench.sh`
(deploy+bench+restore-trap), `download_all.sh`. Chart: [`../assets/plot_results.py`](../assets/plot_results.py).
