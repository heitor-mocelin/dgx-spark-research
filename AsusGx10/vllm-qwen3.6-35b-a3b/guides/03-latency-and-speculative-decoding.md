# 03 · Latency & speculative decoding

> **When this guide matters:** throughput (guide `01`) is about *aggregate* tokens/sec across
> many users. Latency is about *one* user's experience — how fast the first word appears and
> how smoothly the rest stream. They pull on the same dials in opposite directions, so you
> tune for a *profile*, not a universal "fast."

---

## Plain-language on-ramp

Two numbers describe latency, and users feel both:

- **TTFT — time to first token.** The pause after you hit enter before *anything* appears.
  Dominated by **prefill**: the model reading your whole prompt before it can answer.
- **ITL / TPOT — inter-token latency / time per output token.** How fast words stream once
  they start. Dominated by **decode**: one memory-bandwidth-bound step per token (guide `00`).

A chat feels snappy when TTFT is low; it feels *smooth* when ITL is low and steady. The levers:

- **For TTFT:** make prefill cheaper — a bigger per-step token budget, and **prefix caching**
  so a repeated system prompt isn't re-read every time.
- **For ITL:** make each decode step cheaper or do *fewer* of them — a smaller per-step budget
  reduces contention, and **speculative decoding** produces several tokens per model step.

The big idea unique to this guide is **speculative decoding**: let the model guess a few tokens
ahead and verify them in one pass, so you take fewer slow sequential steps. Qwen3.6 has this
built in (**MTP**), and it's the main latency win on this box at low concurrency.

---

## TTFT in depth

TTFT ≈ time to prefill your prompt + scheduling delay. Levers, strongest first:

- **Prefix caching** — if the prompt's prefix (system prompt, tool schema, few-shot examples)
  was seen before, its KV cache is reused and that part of prefill is ~free [[001]](../sources/001-vllm-optimization-tuning.md)[[009]](../sources/009-vllm-recipes-qwen35-qwen36.md). For agentic
  traffic with a fixed preamble, this is the single biggest TTFT win — and it's why it's *on*
  in your throughput profile.
- **`--max-num-batched-tokens` ↑** — a larger token budget lets the scheduler push more prefill
  tokens through per step, improving TTFT (the same change that *hurts* ITL — see the tension
  below) [[001]](../sources/001-vllm-optimization-tuning.md).
- **Chunked prefill** — splits a huge prompt into chunks so it doesn't block, *but* the chunking
  itself means a very long prompt's TTFT is paid in pieces; it's a smoothness/fairness win more
  than a raw-TTFT win [[001]](../sources/001-vllm-optimization-tuning.md).
- **CUDA graphs / `-O` levels** — `-O2`+ captures the execution graph and fuses ops, cutting
  per-call launch overhead that otherwise inflates both TTFT and ITL [[001]](../sources/001-vllm-optimization-tuning.md).

## ITL / TPOT in depth

Each decode step reads the active weights once (≈1.7 GB/token at NVFP4 on your model — guide
`00`) and runs attention over the KV cache. Drivers:

- **Bandwidth** — fixed by the format (NVFP4) and the silicon (273 GB/s). This is the floor;
  quantization (guide `02`) is how you lower it.
- **Batch contention** — a larger per-step token budget means each decode token waits behind
  more work, raising ITL. vLLM's guidance: **`max-num-batched-tokens ≈ 2048` for better ITL**,
  higher for better TTFT/throughput [[001]](../sources/001-vllm-optimization-tuning.md).
- **Number of steps** — this is what speculative decoding attacks.

---

## Speculative decoding (the headline latency win)

Normal decoding is strictly sequential: one model forward pass per output token. **Speculative
decoding** proposes *k* candidate tokens cheaply, then verifies them in a single forward pass;
accepted guesses mean several tokens emerged from one step → lower TPOT.

### MTP — built into Qwen3.5/3.6
Qwen3.6 ships **Multi-Token Prediction (MTP)**: the model itself drafts the next tokens, so you
don't need a separate draft model. Enable it via the speculative config [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md):

```bash
vllm serve <model> \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser qwen3
```

Key behaviors from the official recipe [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md):

- **MTP-1 lowers TPOT (per-token latency) with a high acceptance rate** — great for interactive,
  low-concurrency use.
- **It *reduces* throughput under load**, because speculative tokens consume KV-cache capacity,
  shrinking the effective batch size. So it's a **latency-profile** feature, not a throughput one.
- **Disable prefix caching when optimizing for this latency path** — the recipe pairs MTP with
  prefix caching *off* [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md).
- **`num_speculative_tokens` is tunable (1–5).** Higher *k* can cut latency further but
  acceptance rate and throughput trade-offs vary — measure per workload [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md).

### EAGLE-3 — the other approach
NVIDIA's DGX Spark playbooks include an **EAGLE-3** speculative-decoding example (with
GPT-OSS-120B) that uses a built-in drafting head instead of a separate draft model, simplifying
deployment and raising token acceptance [[010]](../sources/010-nvidia-spark-sw-optimizations.md). For Qwen3.6, MTP is the native path; EAGLE
is worth knowing as the general technique and for other models.

> NVIDIA's own headline result pairs NVFP4 **with** speculative decoding for up to **2.6× vs
> FP8** on a dual-Spark Qwen-235B run [[010]](../sources/010-nvidia-spark-sw-optimizations.md) — i.e. quantization and spec-decode compound.

---

## The central tension: latency vs throughput

| Dial | Latency-friendly | Throughput-friendly | Source |
|---|---|---|---|
| `--max-num-batched-tokens` | low (~2048) → better ITL | high (8192+) → better TTFT/throughput | [[001]](../sources/001-vllm-optimization-tuning.md) |
| `--max-num-seqs` | low (less contention) | high (more amortization) | [[001]](../sources/001-vllm-optimization-tuning.md) |
| Speculative decoding (MTP) | **on** (k=1–2) | **off** (frees KV/batch) | [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) |
| Prefix caching | off (per recipe, with MTP) | on (reuse shared prefix) | [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) |

The practical conclusion: **run the profile that matches your traffic.** A single interactive
user → latency profile (MTP-1, smaller batch). Many concurrent agent requests → throughput
profile (big batch, spec off, prefix caching on). You can't maximize both at once on one server;
if you need both, run two configurations or accept a balanced midpoint.

---

## On *your* GX10

Your current config is a **throughput profile** (`--max-num-seqs 128`, `--max-num-batched-tokens
4096`, prefix caching on, no speculative config). That's the right default for serving an agent
gateway with concurrent tool calls.

A **latency profile** worth A/B-testing in Phase 3 (read-only until you green-light), for when a
single user is waiting on a long generation:

```bash
# latency-leaning variant (illustrative — measure before adopting)
--speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
--max-num-batched-tokens 2048 \
--max-num-seqs <smaller> \
# (drop --enable-prefix-caching per the recipe's latency guidance)
```

Phase-3 experiments:

1. **MTP on/off at concurrency = 1**: measure TTFT, TPOT, and the **acceptance rate** (how often
   speculative tokens are kept). Sweep `num_speculative_tokens` 1→2→3.
2. **`max-num-batched-tokens` 2048 vs 4096** at low concurrency: quantify the ITL improvement.
3. **MTP under load**: confirm the throughput regression the recipe warns about, so you know the
   crossover concurrency where MTP stops being worth it.
4. **Compound effect**: NVFP4 (native, once the `sm_121` kernels land — guide `02`) + MTP, to see
   if the GX10 sees a fraction of NVIDIA's 2.6× headline [[010]](../sources/010-nvidia-spark-sw-optimizations.md).

> Caveat: MTP consumes KV-cache capacity. On your 32k context with FP8 KV there's headroom, but
> watch for preemption warnings (guide `01`) if you combine MTP with a wide batch.

---

## Where to go next

- **Guide `04` — Tool-calling & context:** the `qwen3_coder` parser and long-context KV math.
- **Guide `05` — Benchmarking:** how to measure TTFT/ITL/acceptance-rate properly.

## Sources cited

- [[001]](../sources/001-vllm-optimization-tuning.md) Optimization and Tuning (vLLM)
- [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) Qwen3.5/3.6 usage guide (vLLM Recipes)
- [[010]](../sources/010-nvidia-spark-sw-optimizations.md) SW/model optimizations supercharge DGX Spark (NVIDIA)
