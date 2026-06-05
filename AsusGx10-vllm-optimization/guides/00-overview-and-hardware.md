# 00 · Overview & the GB10 hardware you're tuning for

> **This series in one line:** how to make vLLM serve a Qwen3-class MoE model *fast* on an
> ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell), and *why* each knob matters.
>
> Each guide has two layers: a **plain-language on-ramp** (read this if you're new) and a
> **deep dive** (read this if you're tuning in anger). Skip between them freely.

---

## Plain-language on-ramp

A **DGX Spark / GB10** is a small desktop box with a Grace-Blackwell chip and **128 GB of
memory that the CPU and GPU share** ("unified memory"). That shared pool is the headline
feature: you can load a fairly large model entirely in memory without a data-center GPU.

**vLLM** is the server that runs the model and exposes an OpenAI-compatible API
(`/v1/chat/completions`). Your job when "optimizing" is to get more useful tokens per second
out of the box without breaking accuracy or running out of memory.

Two ideas explain almost everything that follows:

1. **The model is quantized.** Instead of storing each weight as a 16-bit number, you store
   it in a 4-bit format (**NVFP4**) or keep the KV cache in 8-bit (**FP8**). Smaller numbers
   = less memory moved = faster, with very little accuracy loss. (Details in guide `02`.)
2. **Generation is memory-bandwidth-bound.** To produce one token, the GPU must read the
   model's active weights out of memory. The GX10's memory delivers **273 GB/s** [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md).
   So "how fast can I generate?" is mostly "how many bytes must I read per token, and how
   well can I share that read across many requests?" That's the whole game — and it's why
   **batching** (guide `01`) is the single biggest throughput lever.

If you remember nothing else: **shrink the bytes per token (quantization), then amortize the
read across requests (batching).**

---

## The hardware, precisely

| Component | Spec (GX10 / DGX Spark) | Source |
|---|---|---|
| Architecture | NVIDIA GB10 Grace Blackwell | [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md) |
| GPU | Blackwell, **5th-gen Tensor Cores** (native FP4/FP8/FP16), 4th-gen RT | [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md) |
| CUDA capability | **12.1 → `sm_121` / `sm_121a`** | [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md) |
| CPU | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) | [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md) |
| Memory | **128 GB LPDDR5x unified**, 256-bit @ 4266 MHz | [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md) |
| **Memory bandwidth** | **273 GB/s** | [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md) |
| Compute (FP4) | ~1 PFLOP FP4 (sparse) | [[010]](../sources/010-nvidia-spark-sw-optimizations.md) |
| OS | DGX OS (Ubuntu ARM64) | [[005]](../sources/005-nvidia-spark-vllm-playbook.md) |

Two consequences of this spec sheet drive the entire series:

- **273 GB/s is modest** compared with a data-center GPU (an H200 moves ~5 TB/s). On a
  bandwidth-bound workload that makes the GX10 a *throughput-via-batching and
  quantization* machine, not a brute-force one. This is a feature, not a complaint — it's
  why NVFP4 matters so much here.
- **`sm_121` is new-ish silicon.** Several vLLM kernels were written for data-center
  Blackwell (`sm_100`) first, and GX10 (`sm_121`) support arrived later. This is the root
  of the "GPU lacks native FP4 → Marlin kernel" message you see in your logs — covered in
  detail in guide `02` and sourced to [[011]](../sources/011-vllm-pr40082-flashinfer-b12x-sm121.md) / [[012]](../sources/012-vllm-issue33333-flashinfer-cutlass-sm120.md).

### Unified memory (UMA): the thing that bites you

CPU and GPU share one 128 GB pool. Great for fitting big models; occasionally surprising:
applications that don't yet understand UMA can report out-of-memory even when you're within
capacity. NVIDIA's own playbook documents the fix — drop the buffer cache:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

— from the DGX Spark vLLM playbook [[005]](../sources/005-nvidia-spark-vllm-playbook.md).

---

## Deep dive: why generation is bandwidth-bound (with a worked estimate)

LLM decoding produces one token at a time. For each token, every weight that participates
must be read from memory at least once. For a **Mixture-of-Experts (MoE)** model only a
subset of weights ("active parameters") fire per token, which is exactly why MoE suits this
box.

Your model — **Qwen3.6-35B-A3B** — is **35B total but only ~3B active** per token (256
experts, 8 routed + 1 shared) [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md). At NVFP4 (~4.5 bits ≈ **0.5625 bytes/param**
[[003]](../sources/003-nvidia-introducing-nvfp4.md)):

- **Total weights in memory:** 35e9 × 0.5625 B ≈ **~20 GB** → fits trivially in 128 GB,
  leaving the rest for KV cache and other work. (Compare: the same model in BF16 is ~70 GB,
  per the near-identical community build [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md).)
- **Bytes read per token (active weights only):** 3e9 × 0.5625 B ≈ **~1.7 GB/token**.
- **Single-stream decode ceiling (first principles):** 273 GB/s ÷ 1.7 GB ≈ **~160 tok/s**,
  *before* counting KV-cache reads, activations, and kernel overhead — so real single-stream
  will be lower.

> ⚠️ **Estimate, not a measurement.** This is a back-of-envelope upper bound to build
> intuition. The GX10 is read-only until Phase 3; we'll replace these with measured
> `vllm bench` numbers then (guide `05`).

This estimate explains two real observations:

- The near-twin BF16 build reports **~31 tok/s single-stream** [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md) — BF16 reads ~2 bytes/param
  (≈3.5× more than NVFP4), so a single stream is much slower. NVFP4 is the difference
  between "usable" and "sluggish" on this box.
- Your setup reportedly sustains **~600 tok/s aggregate** at `--max-num-seqs 128`
  *(to confirm in Phase 3)*. That's far above the ~160 tok/s single-stream ceiling because
  **batching amortizes the weight read across many sequences** — read the experts once,
  serve many tokens. This is the crux of guide `01`.

---

## On *your* GX10 (current baseline)

Your live `docker run` already encodes most best practices from this corpus:

| Setting | Your value | Why it's right (forward ref) |
|---|---|---|
| Weights | NVFP4 (`--quantization modelopt`) | guide `02` — ~20 GB footprint, bandwidth win |
| KV cache | `--kv-cache-dtype fp8` | guide `02` — more tokens in memory |
| Batching | `--max-num-seqs 128`, `--max-num-batched-tokens 4096` | guide `01` — the throughput lever |
| Scheduling | `--enable-chunked-prefill`, `--enable-prefix-caching` | guide `01` |
| Memory | `--gpu-memory-utilization 0.90` | guide `01` — KV headroom vs OOM |
| Tools | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | guide `04` — matches the official recipe [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) |

**Open question we'll chase:** your logs warn the GPU falls back to the **Marlin** kernel for
FP4 because native FP4-MoE kernels didn't yet cover `sm_121`. How much throughput that costs
on the GX10 is the most interesting thing to measure in Phase 3 → guide `02`.

---

## Where to go next

- **Guide `01` — Throughput & batching:** the single biggest lever, and the knobs behind your
  `max-num-seqs` / `max-num-batched-tokens` / chunked-prefill settings.
- **Guide `02` — Quantization (NVFP4 & FP8):** what NVFP4 actually is, and the `sm_121`
  Marlin-fallback story in full.

## Sources cited

- [[003]](../sources/003-nvidia-introducing-nvfp4.md) Introducing NVFP4 (NVIDIA)
- [[005]](../sources/005-nvidia-spark-vllm-playbook.md) DGX Spark vLLM playbook (NVIDIA)
- [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) Qwen3.5/3.6 usage guide (vLLM Recipes)
- [[010]](../sources/010-nvidia-spark-sw-optimizations.md) SW/model optimizations supercharge DGX Spark (NVIDIA)
- [[011]](../sources/011-vllm-pr40082-flashinfer-b12x-sm121.md) PR #40082 — FlashInfer b12x MoE/FP4 for SM120/121
- [[012]](../sources/012-vllm-issue33333-flashinfer-cutlass-sm120.md) Issue #33333 — FLASHINFER_CUTLASS unsupported on SM120
- [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md) DGX Spark hardware overview (NVIDIA)
- [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md) Qwen3.5-35B-A3B on DGX Spark (community)
