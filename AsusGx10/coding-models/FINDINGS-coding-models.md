# Coding models on the DGX Spark (GB10) — performance + HumanEval + context findings

Measured on a single **ASUS GX10 / NVIDIA GB10** (128 GB unified memory, `sm_121`), vLLM nightly.
Each model served **standalone** on `:8001` at `--gpu-memory-utilization 0.85` with the proven sm_121
FP4 env, one model at a time (OOM-safe). **Ready-made quants only** — NVFP4 where a usable one exists,
otherwise a reputable FP8. Battery per model: **HumanEval pass@1** (greedy, instruct-style, 16-way
parallel gen) · single-stream **decode tok/s** + **TTFT** · **concurrency sweep** (c=1/4/16/32) ·
**context-length sweep**. Harness in [`coding-models/`](./); raw JSON in [`coding-models/results/`](./results/).

## TL;DR

1. **On a bandwidth-bound single Spark, MoE coders dominate.** Qwen3-Coder-30B-**A3B** (3 B active)
   decodes at **73 tok/s** — **~10× faster than the 32 B *dense* coder (6.9 tok/s)** — *and* scores higher
   on HumanEval. If you want one interactive coding model on a Spark, it's **Qwen3-Coder-30B-A3B (NVFP4)**.
2. **FP8-dynamic (W8A8) dense is slow here.** The 32 B FP8 manages only **6.9 tok/s** single-stream — fine
   for *batch* (209 tok/s @ c=32) but painful interactively. NVFP4 dense (Devstral 24 B) is better (16 tok/s)
   but still far behind MoE. Prefer NVFP4 over FP8-dynamic when both exist.
3. **Long context is prefill-bound.** A 130 k-token prompt costs **~99 s TTFT** on the flagship; decode only
   sags from 73 → 26 tok/s. DeepSeek-V2-Lite's **MLA** makes long context cheaper (45 k tok: 9.6 s TTFT vs
   the flagship's 13.9 s at 42 k).
4. **Everything scales with concurrency** — even the slow 32 B reaches 209 tok/s aggregate at c=32, so these
   are viable as *batched* backends regardless of single-stream speed.

## Summary

| Model | Params | Quant | HumanEval pass@1 | Decode tok/s | TTFT | Agg tok/s @c=32 |
|---|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B-Instruct** | 30B / 3B MoE | NVFP4 (ig1) | **93.3%** (153/164) | **73.5** | 46.7 ms | **1398.5** |
| Qwen2.5-Coder-32B-Instruct | 32B dense | FP8 (RedHatAI) | **92.1%** (151/164) | 6.9 | 295.6 ms | 208.6 |
| Qwen2.5-Coder-14B-Instruct | 14B dense | FP8 (RedHatAI) | **91.5%** (150/164) | 14.8 | 138.6 ms | 443.7 |
| Qwen2.5-Coder-7B-Instruct | 7B dense | FP8 (RedHatAI) | 48.8%* (80/164) | 28.3 | 73.6 ms | 850.5 |
| DeepSeek-Coder-V2-Lite-Instruct | 16B / 2.4B MoE | FP8 (RedHatAI) | **78.7%** (129/164) | 70.5 | 85.7 ms | 644.9 |
| Devstral-Small-2-24B | 24B dense | NVFP4 (Firworks) | **79.3%** (130/164) | 16.0 | 127.8 ms | 484.3 |
| Codestral-22B-v0.1 | 22B dense | FP8 (TechxGenus) | **77.4%** (127/164) | 10.3 | 198.5 ms | 309.2 |

\* **7B caveat** — see *Methodology & caveats*. 48.8% reflects our **chat-style** harness (the small model
often *renames/rewrites* the target function, failing `check()`); it is **not** comparable to the
base-completion HumanEval (~85%) usually quoted for this model. The other models follow the instruction reliably.

## Why MoE wins here (decode is memory-bandwidth-bound)

GB10 decode speed tracks **active** parameters × memory bandwidth, not total parameters. Per generated token:

| Model | Active params | Decode tok/s |
|---|---|---|
| Qwen3-Coder-30B-A3B (MoE) | ~3 B | 73.5 |
| DeepSeek-Coder-V2-Lite (MoE) | ~2.4 B | 70.5 |
| Qwen2.5-Coder-7B (dense) | 7 B | 28.3 |
| Qwen2.5-Coder-14B (dense) | 14 B | 14.8 |
| Codestral-22B (dense) | 22 B | 10.3 |
| Qwen2.5-Coder-32B (dense) | 32 B | 6.9 |

A clean inverse relationship: **dense models pay full-parameter bandwidth every token**; MoE models pay only
for the few active experts. On a single Spark this is decisive for interactive use.

## Context-length sweep (TTFT / decode tok/s by real prompt tokens)

The full per-model tables are below. The pattern is universal: **decode degrades gently with context, TTFT
explodes** (prefill is compute-heavy). For coding agents stuffing large repos into context, **TTFT is the cost
to watch**, and **MLA models (DeepSeek) are the cheapest at long context**.

### Qwen3-Coder-30B-A3B-Instruct
| Prompt tokens | TTFT | Decode tok/s |
|---|---|---|
| 1252 | 225.4 ms | 73.3 |
| 10304 | 1874.2 ms | 64.1 |
| 42264 | 13867.0 ms | 45.3 |
| 129624 | 99401.1 ms | 26.0 |

### DeepSeek-Coder-V2-Lite-Instruct (MLA — note the lower long-context TTFT)
| Prompt tokens | TTFT | Decode tok/s |
|---|---|---|
| 1351 | 187.7 ms | 71.4 |
| 11057 | 1352.4 ms | 66.2 |
| 45231 | 9575.3 ms | 52.3 |

### Qwen2.5-Coder-32B-Instruct
| Prompt tokens | TTFT | Decode tok/s |
|---|---|---|
| 1273 | 614.6 ms | 7.2 |
| 10325 | 5485.6 ms | 6.9 |
| 31335 | 19016.3 ms | 6.4 |

### Qwen2.5-Coder-14B-Instruct
| Prompt tokens | TTFT | Decode tok/s |
|---|---|---|
| 1273 | 291.8 ms | 16.1 |
| 10325 | 2820.4 ms | 15.3 |
| 31335 | 11182.2 ms | 13.8 |

### Qwen2.5-Coder-7B-Instruct
| Prompt tokens | TTFT | Decode tok/s |
|---|---|---|
| 1273 | 156.7 ms | 29.3 |
| 10325 | 1398.2 ms | 27.4 |
| 31335 | 5100.4 ms | 25.6 |

### Devstral-Small-2-24B
| Prompt tokens | TTFT | Decode tok/s |
|---|---|---|
| 1247 | 257.5 ms | 16.2 |
| 10299 | 2578.9 ms | 15.5 |
| 42259 | 14630.6 ms | 13.5 |
| 129619 | 92460.3 ms | 10.3 |

### Codestral-22B-v0.1
| Prompt tokens | TTFT | Decode tok/s |
|---|---|---|
| 1448 | 515.8 ms | 10.5 |
| 11808 | 4663.8 ms | 10.0 |

## Compatibility findings (serving these on GB10 / sm_121, vLLM nightly)

These cost real debugging time and are the kind of thing worth writing down:

- **NVFP4 from `llm-compressor` declares `compressed-tensors`, not `modelopt`.** Forcing
  `--quantization modelopt` errors with *"Quantization method ... does not match"*. **Fix:** omit
  `--quantization` and let vLLM auto-detect (works for compressed-tensors / modelopt / fp8 alike).
- **DeepSeek-V2 (MLA) + `--kv-cache-dtype fp8` ⇒ engine-core init crash.** **Fix:** drop fp8 KV for
  MLA models (MLA's latent KV is already tiny, so you lose little). With that removed, DeepSeek-Coder-V2-Lite
  serves cleanly at 128 k.
- **Codestral-22B-v0.1 is a base/FIM model with no chat template** → HTTP 400 on `/v1/chat/completions`.
  **Fix:** pass a Mistral `[INST]` chat template (`--chat-template`, see `coding-models/codestral-chat.jinja`).
- **FP8-dynamic (W8A8) dense weights hit a slow path on sm_121** (the 32 B's 6.9 tok/s). NVFP4 weight-only
  is materially faster where a quant exists.
- Cold start ≈ 3–6 min (weights + `torch.compile`); poll `/health` up to ~10 min before declaring failure.

## Recommendations

- **One interactive coding model on a single Spark → Qwen3-Coder-30B-A3B-Instruct (NVFP4).** Best accuracy
  *and* best speed; nothing else is close on this hardware.
- **Need a small/fast agent backend →** DeepSeek-Coder-V2-Lite (MoE, 70 tok/s, cheap long context) or
  Qwen2.5-Coder-7B (fast, but verify quality on your tasks).
- **Want dense 32 B accuracy →** Qwen2.5-Coder-32B FP8 is excellent on HumanEval but only viable **batched**
  (single-stream is ~7 tok/s); serve it behind a queue, not as a live assistant.
- **Agentic / multi-file editing →** Devstral-Small-24B (purpose-built for it; HumanEval understates it).

## Methodology & caveats

- **HumanEval here is instruct-style chat pass@1, greedy, our prompt + fenced-code extraction.** It is
  **internally consistent across these models on this box**, but absolute numbers are **not** 1:1 with the
  official base-completion HumanEval leaderboard. The 7 B's low score is the clearest example (it rewrites the
  function signature under chat instructions).
- Each model uses **whatever ready-made quant exists** (NVFP4 / FP8) — format is recorded per row. This is an
  honest "what runs, and how fast, on *this* box today," not a same-precision apples-to-apples accuracy study.
- Standalone serving (one model, util 0.85). Co-resident / under-load behavior not measured here.
- DCGM power (`DCGM_FI_DEV_POWER_USAGE`, Prometheus `172.27.27.212:9090`) is available per run window for a
  power-efficiency follow-up.
