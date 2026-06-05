# 01 · Throughput & batching — the biggest lever

> **Goal:** maximize useful tokens/second on the GX10. On a bandwidth-bound box, that means
> **reading the model's weights once and serving as many sequences as possible from that
> read.** Everything here is in service of that.

---

## Plain-language on-ramp

Imagine the GPU has to fetch a heavy book (the model weights) from a slow shelf (memory) to
answer each question. If it fetches the book *per person*, the line is slow. If it gathers a
**batch** of people and answers them all from one fetch, everyone's served faster. That's
**batching**, and it's why a server like vLLM gets far more total throughput than running one
prompt at a time.

vLLM does this automatically with a few mechanisms you mostly just *enable*:

- **Continuous batching** — new requests hop into the running batch instead of waiting for it
  to finish, keeping the GPU busy [[005]](../sources/005-nvidia-spark-vllm-playbook.md).
- **PagedAttention** — stores each sequence's KV cache in small pages so memory isn't wasted,
  letting more sequences fit at once [[005]](../sources/005-nvidia-spark-vllm-playbook.md).
- **Chunked prefill** — splits long prompts into chunks so reading a big new prompt doesn't
  stall everyone else's token generation [[001]](../sources/001-vllm-optimization-tuning.md).
- **Prefix caching** — if many requests share the same beginning (a system prompt, a tool
  schema), compute it once and reuse it [[001]](../sources/001-vllm-optimization-tuning.md).

The two dials you actually turn are **how many sequences** run at once (`--max-num-seqs`) and
**how many tokens per scheduler step** (`--max-num-batched-tokens`). The rest is about giving
those dials enough memory to work with.

---

## The mechanisms, in depth

### Continuous batching + PagedAttention
vLLM keeps the GPU saturated by admitting new requests into the in-flight batch every step,
and by paging KV cache so fragmentation doesn't strand memory [[005]](../sources/005-nvidia-spark-vllm-playbook.md). You don't configure these —
they're the engine. What you configure is how *wide* and *deep* the batch is allowed to get.

### `--max-num-seqs` — batch **width**
The maximum number of sequences resident in the running batch. Wider batch → more sequences
share each weight read → higher aggregate throughput, until you run out of KV-cache memory or
saturate compute. Your current value is **128**.

### `--max-num-batched-tokens` — batch **depth** (per step)
The token budget the scheduler can pack into one step (prefill chunks + decode tokens). vLLM's
tuning guidance is explicit [[001]](../sources/001-vllm-optimization-tuning.md):

> *Smaller values (e.g. 2048) achieve better inter-token latency (ITL)… higher values achieve
> better time-to-first-token (TTFT)… for optimal throughput, set `max_num_batched_tokens > 8192`,
> especially for smaller models on large GPUs.*

Your current value is **4096** — a balanced midpoint. On a throughput-first profile there may
be headroom to raise it (toward 8192+); on a latency-first profile, lower it. **This is a
prime Phase-3 sweep** (guide `05`).

### Chunked prefill — keeping decodes alive
With chunked prefill on, the scheduler **prioritizes decode requests**, batches all pending
decodes, then fills the remaining `max_num_batched_tokens` budget with prefill chunks; an
oversized prompt is automatically chunked rather than monopolizing a step [[001]](../sources/001-vllm-optimization-tuning.md). The net effect is
that one user pasting a 30k-token prompt doesn't freeze everyone else's token stream. It's on
in your config (`--enable-chunked-prefill`) and is default-on in vLLM V1 where possible [[001]](../sources/001-vllm-optimization-tuning.md).

### Prefix caching — free wins for shared context
`--enable-prefix-caching` reuses the KV cache for identical prompt prefixes [[001]](../sources/001-vllm-optimization-tuning.md)[[009]](../sources/009-vllm-recipes-qwen35-qwen36.md). For agentic
/ tool-calling workloads (yours) where every request carries the same long system prompt and
tool schema, this is a large, cheap saving. **Caveat:** for latency-sensitive single-stream
work with speculative decoding, the Qwen recipe actually recommends *disabling* it (see guide
`03`) [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) — so it's workload-dependent, not always-on.

### `--gpu-memory-utilization` — the memory you hand to KV cache
vLLM pre-allocates this fraction of memory for weights + KV cache. Higher → more KV cache →
more concurrent sequences and longer contexts → more batching headroom. Too high and you risk
OOM (worse on UMA, see guide `00`). You run **0.90**, a sensible aggressive-but-safe value
[[001]](../sources/001-vllm-optimization-tuning.md).

### Compilation levels `-O0`…`-O3`
vLLM trades startup time for runtime performance via optimization levels: `-O0` none (fastest
start), `-O2` default (fusions + full/piecewise CUDA graphs), `-O3` aggressive [[001]](../sources/001-vllm-optimization-tuning.md). Your cold
start of ~5–6 min is dominated by weight load + `torch.compile`; the payoff is the fused,
graph-captured hot path. Don't drop below the default to "speed up startup" on a server you
launch rarely — you'd pay for it on every token.

### Watch for preemption
If KV cache runs short, vLLM **preempts and later recomputes** sequences, and logs a
`PreemptionMode.RECOMPUTE` warning — a sign you're memory-starved [[001]](../sources/001-vllm-optimization-tuning.md). Fixes: raise
`--gpu-memory-utilization`, lower `--max-num-seqs`, or shorten `--max-model-len`. Frequent
preemption silently tanks end-to-end latency.

### A note on parallelism (single GB10)
The official Qwen recipes lead with `-dp 8 --enable-expert-parallel` or `--tensor-parallel-size 8`
[[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) — those target **multi-GPU clusters or dual-Spark** rigs. On a **single GB10** you run
**TP=1**, no expert/data parallelism; your throughput comes entirely from batching + quantization
on one device. (Multi-Spark via Ray is a separate path [[005]](../sources/005-nvidia-spark-vllm-playbook.md), out of scope here.)

---

## A tuning decision table

| You want… | Turn this way | Source |
|---|---|---|
| **Max aggregate throughput** | `max-num-batched-tokens` ↑ (8192+), `max-num-seqs` ↑, prefix-caching on, `gpu-memory-utilization` ↑ | [[001]](../sources/001-vllm-optimization-tuning.md) |
| **Low inter-token latency (ITL)** | `max-num-batched-tokens` ↓ (~2048), smaller batch | [[001]](../sources/001-vllm-optimization-tuning.md) |
| **Low time-to-first-token (TTFT)** | `max-num-batched-tokens` ↑ | [[001]](../sources/001-vllm-optimization-tuning.md) |
| **Stop OOM / preemption** | `gpu-memory-utilization` ↑ *or* `max-num-seqs` ↓ *or* `max-model-len` ↓ | [[001]](../sources/001-vllm-optimization-tuning.md) |

There is no single "fast" setting — throughput and latency trade against each other through
the same dials. Pick a profile, then measure (guide `05`).

---

## On *your* GX10

Current throughput-relevant flags:

```
--max-num-seqs 128 --max-num-batched-tokens 4096 \
--enable-chunked-prefill --enable-prefix-caching \
--gpu-memory-utilization 0.90 --max-model-len 32768
```

This is a solid balanced baseline. The **Phase-3 experiments** worth running (read-only until
you green-light):

1. **Sweep `--max-num-batched-tokens`** 2048 → 4096 → 8192 → 16384, recording TTFT, ITL, and
   tok/s at fixed concurrency. The docs predict 4096→8192 trades ITL for throughput/TTFT [[001]](../sources/001-vllm-optimization-tuning.md);
   confirm where the GX10's 273 GB/s actually tips over.
2. **Sweep `--max-num-seqs`** (e.g. 64 / 128 / 256) to find where KV-cache pressure starts
   causing preemption warnings on a 32k context.
3. **Prefix-caching A/B** on your real agentic traffic (long shared system prompt + tool
   schema) — measure the cache-hit throughput gain.
4. **`--gpu-memory-utilization`** 0.90 → 0.92 → 0.95, watching for UMA OOM vs extra KV
   headroom.

Each becomes a row in guide `05`'s benchmark table, and a parameter in the `tune.sh` script.

---

## Sources cited

- [[001]](../sources/001-vllm-optimization-tuning.md) Optimization and Tuning (vLLM)
- [[005]](../sources/005-nvidia-spark-vllm-playbook.md) DGX Spark vLLM playbook (NVIDIA)
- [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) Qwen3.5/3.6 usage guide (vLLM Recipes)
