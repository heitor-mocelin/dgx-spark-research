# 02 · Quantization — NVFP4 weights & FP8 KV cache (and the SM121 Marlin story)

> **Why this guide is the heart of the series:** on a 273 GB/s box, the bytes you move per
> token *are* your throughput. Quantization is how you cut those bytes. This guide explains
> what NVFP4 and FP8 actually do, how to use them in vLLM, and — critically for the GX10 —
> why your logs say *"GPU lacks native FP4 → Marlin kernel"* and what that costs you.

---

## Plain-language on-ramp

A model is billions of numbers (weights). Store each as a big 16-bit number and you move a
lot of bytes; store each as a tiny 4-bit number and you move ~4× fewer. Fewer bytes moved =
faster generation and a smaller memory footprint. The catch is precision: squeeze a number
into 4 bits naively and the model gets dumber. The whole art of a *good* 4-bit format is
losing as little accuracy as possible.

There are three precisions you'll meet on this box:

- **BF16** (16-bit) — the "full" weights. Accurate, but ~70 GB for your model and slow to read.
- **FP8** (8-bit) — half the bytes, tiny accuracy loss. The current safe default everywhere.
- **NVFP4** (4-bit) — NVIDIA's 4-bit format, *purpose-built* to keep accuracy. ~20 GB for
  your model, fastest to read. This is what you run.

Separately, the **KV cache** (the model's running memory of the conversation) can *also* be
quantized — you run it at **FP8**. That's a different knob from the weights, and it buys you
longer contexts and bigger batches.

The one-sentence takeaway: **NVFP4 weights make the GX10 fast; FP8 KV cache makes it hold more
at once; and on `sm_121` silicon there's a kernel-support wrinkle that means you may not yet
be getting the *full* NVFP4 speedup.**

---

## NVFP4, precisely

### The format
NVFP4 stores each value in **4 bits as E2M1** (1 sign, 2 exponent, 1 mantissa), representing
roughly the range −6…+6 (values like 0, 0.5, 1, 1.5, 2, 3, 4, 6) [[003]](../sources/003-nvidia-introducing-nvfp4.md). Four bits alone is far
too coarse for model weights, so NVFP4 adds **two levels of scaling**:

1. A **per-16-value micro-block** scale stored in **E4M3 FP8**.
2. A **per-tensor** second-level scale stored in **FP32**.

So a reconstructed weight is `x = x_q × s`, where `s` is the block's FP8 scale, and the
per-tensor FP32 scale keeps the whole distribution in range so the E4M3 block scales can be
used effectively [[003]](../sources/003-nvidia-introducing-nvfp4.md). Effective storage is **~4.5 bits/value** (4 data bits + the amortized
FP8 block scale over 16 values) [[003]](../sources/003-nvidia-introducing-nvfp4.md).

### Why it beats MXFP4
NVFP4's predecessor, **MXFP4**, uses **32-value blocks** and a **power-of-two (E8M0)** block
scale. NVFP4 improves on both axes [[003]](../sources/003-nvidia-introducing-nvfp4.md):

- **Smaller blocks (16 vs 32)** → twice as many chances to match the local dynamic range of
  the data, so large and small weights in the same tensor are both represented well.
- **E4M3 (fractional) scales instead of E8M0 (power-of-two)** → the scale can be chosen to
  minimize the block's *total* squared error, rather than snapping to the nearest 2ⁿ. NVIDIA's
  worked example shows a lower mean-squared error for E4M3 scaling (≈0.08 avg) vs E8M0 [[003]](../sources/003-nvidia-introducing-nvfp4.md).

The payoff, per NVIDIA's measurements on DeepSeek-R1-0528: **<1% accuracy loss vs FP8** across
seven evals (and *better* on AIME-2024), while cutting memory **~3.5× vs FP16 and ~1.8× vs
FP8** [[003]](../sources/003-nvidia-introducing-nvfp4.md). On Blackwell, the 5th-gen Tensor Cores handle the micro-scaling, grouping, and
4-bit matmul natively [[003]](../sources/003-nvidia-introducing-nvfp4.md)[[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md).

### What "native FP4" means (and why it matters for the next section)
The *native* path does **W4A4** — 4-bit **weights and activations** through the FP4 Tensor
Core path — so you get both the memory saving (4-bit weights) **and** the compute saving
(4-bit math). When that path isn't available for your GPU, frameworks fall back to a kernel
that keeps 4-bit *weights* but runs the math in higher precision (W4A16-style). You keep the
bandwidth win; you lose part of the compute win. **Hold that thought.**

---

## The `sm_121` / Marlin fallback — your log line, explained

Your vLLM logs warn that the GPU lacks native FP4 and falls back to the **Marlin** kernel.
Here's the full chain, sourced from the vLLM trackers:

1. **GB10 is `sm_121` (CUDA capability 12.1), and `sm_121a`** [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md). Data-center Blackwell
   (B200/GB200) is `sm_100`. Many FP4 kernels were written and validated for `sm_100`/`sm_90`
   first.
2. **vLLM's FP4 MoE backend (`FLASHINFER_CUTLASS`) didn't recognize `sm_120`/`sm_121`** as a
   supported device family. Users on 5090 (`sm_120`), RTX Pro 6000, and GB10 hit *"NvFp4 MoE
   backend 'FLASHINFER_CUTLASS' does not support the deployment configuration since kernel
   does not support current device"* [[012]](../sources/012-vllm-issue33333-flashinfer-cutlass-sm120.md). The thread fingers a specific breaking commit and
   links the upstream **FlashInfer issue #2077** and **PR #33417** [[012]](../sources/012-vllm-issue33333-flashinfer-cutlass-sm120.md).
3. **So vLLM falls back to the Marlin kernel** for the FP4 weights — which runs, but via the
   mixed-precision path rather than the native W4A4 FP4-MoE path. That's the *"GPU lacks
   native FP4 → Marlin kernel"* message.
4. **The fix is landing via FlashInfer "b12x" kernels for `sm_120`/`sm_121`** — vLLM **PR
   #40082** integrates FlashInfer b12x MoE and FP4 GEMM kernels specifically for these
   architectures [[011]](../sources/011-vllm-pr40082-flashinfer-b12x-sm121.md). Until that's in your image, you're on the fallback.

### What it costs you — and the three ways out
- **Cost:** you retain NVFP4's *memory/bandwidth* savings (4-bit weights still move ~4× fewer
  bytes), but likely forfeit some of the *compute* speedup of the native FP4-MoE path. The
  exact tok/s delta on the GX10 is **not yet measured** — it's the single most valuable Phase-3
  experiment (compare Marlin fallback vs a b12x-enabled build).
- **Option A — track nightlies.** Run a vLLM nightly that includes PR #40082's b12x kernels
  and re-check whether the Marlin warning disappears and throughput rises. (You already run
  `vllm/vllm-openai:nightly`, so this is mostly a date-watch.)
- **Option B — build FlashInfer from source for `sm_121`.** The community
  `SPARK_Qwen3.5-122B-A10B-NVFP4` build compiles FlashInfer from source for SM121 and applies
  NVFP4-specific patches precisely to get native FP4 MoE on GB10 [[016]](../sources/016-community-bjk110-spark-nvfp4-mtp.md). This is the
  highest-effort, highest-control path.
- **Option C — accept the fallback.** It works today and still beats FP8 on memory. If your
  workload is throughput-via-batching (it is), the fallback may be "good enough" until the
  kernels land upstream. Measure before investing in B.

> Your env vars are already part of this story: `VLLM_USE_FLASHINFER_MOE_FP4=0` (now deprecated
> in favor of `--moe-backend` — see guide on env/backends and [[006]](../sources/006-vllm-env-vars.md)),
> `VLLM_FP8_MOE_BACKEND=flashinfer_cutlass`, and `CUTE_DSL_ARCH=sm_121a` — that last one is you
> *telling* the CUTLASS DSL to target GB10's architecture explicitly.

---

## Getting & making NVFP4 checkpoints

You have three supply routes for NVFP4 weights:

1. **Pre-quantized from NVIDIA / community.** NVIDIA publishes NVFP4 checkpoints (Llama, Qwen,
   Phi, Nemotron, Gemma) validated for Spark in the playbook's support matrix [[005]](../sources/005-nvidia-spark-vllm-playbook.md), and the Qwen
   recipe recommends the `nvidia/...-NVFP4` checkpoints for serving efficiency [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md).
2. **Quantize it yourself with TensorRT-Model-Optimizer.** The DGX Spark NVFP4 playbook walks
   the full PTQ flow inside a TensorRT-LLM container, producing a unified HF checkpoint you
   load in vLLM [[004]](../sources/004-nvidia-spark-nvfp4-playbook.md). Note: *not every architecture is supported for NVFP4 quantization* [[005]](../sources/005-nvidia-spark-vllm-playbook.md).
3. **LLM Compressor** is the other supported toolchain for producing FP4/FP8 checkpoints for
   vLLM [[003]](../sources/003-nvidia-introducing-nvfp4.md).

Whichever route, **validate accuracy** before trusting it — PTQ can degrade specific tasks, so
run evals on the quantized model [[004]](../sources/004-nvidia-spark-nvfp4-playbook.md)[[020]](../sources/020-nvidia-ptq-performance-accuracy.md). NVIDIA's PTQ guide covers the calibration techniques
(SmoothQuant, AWQ, AutoQuantize) that recover accuracy when a naive PTQ pass falls short [[020]](../sources/020-nvidia-ptq-performance-accuracy.md).

In vLLM you select the path with `--quantization modelopt` (your setting) for ModelOpt-format
NVFP4 checkpoints.

---

## FP8 KV cache — your second quantization knob

Weights are only half the bytes. The **KV cache** grows with every token of context and every
concurrent sequence; on long contexts it can rival the weights in size. Quantizing it to FP8
roughly halves it, letting you **store more tokens → longer context and bigger batches** [[002]](../sources/002-vllm-quantized-kv-cache.md).
You already run `--kv-cache-dtype fp8`.

Things worth knowing [[002]](../sources/002-vllm-quantized-kv-cache.md):

- **`fp8_e4m3` vs `fp8_e5m2`:** E4M3 has more mantissa (more precision, smaller range) and is
  the usual choice; E5M2 trades precision for range. Both need CUDA 11.8+.
- **Scaling granularity:** per-tensor (one scale per K/V tensor) or per-attention-head (one
  scale per head — more accurate, but currently Flash-Attention-only and needs llm-compressor
  calibration).
- **Calibration:** default scales of 1.0 (zero-cost, least accurate) → on-the-fly estimation
  from a warmup batch → **dataset calibration via llm-compressor** (most accurate, recommended).
- **FA3 interaction:** with the Flash Attention 3 backend + FP8 KV, attention runs in FP8 and
  *queries* are also quantized — relevant if you tune attention backends.

### The future knob: NVFP4 KV cache
NVIDIA now offers an **NVFP4 KV cache** that halves the KV footprint *again* vs FP8 —
~50% smaller, enabling roughly **2× the context length or batch size**, with <1% accuracy loss
on long-context/code evals [[019]](../sources/019-nvidia-nvfp4-kv-cache-long-context.md). It's worth watching for your 32k (and beyond) context goals,
though FP8 KV remains the safe, broadly-supported default today.

---

## On *your* GX10

Your quantization stack, annotated:

```
--quantization modelopt        # NVFP4 weights (ModelOpt checkpoint) -> ~20 GB, bandwidth win
--kv-cache-dtype fp8           # FP8 KV cache -> more tokens in memory, longer ctx/bigger batch
-e VLLM_FP8_MOE_BACKEND=flashinfer_cutlass
-e VLLM_USE_FLASHINFER_MOE_FP4=0   # deprecated -> migrate to --moe-backend [[006]]
-e CUTE_DSL_ARCH=sm_121a       # target GB10 explicitly for the CUTLASS DSL
```

This is the right stack for the box. The **one open performance question** is the Marlin
fallback. Concrete Phase-3 plan:

1. **Quantify the fallback.** Benchmark current image (Marlin fallback) vs a nightly that
   includes PR #40082's b12x kernels — same model, same `vllm bench` config — and record the
   tok/s and TTFT delta. This tells you whether Option B (source build) is worth it.
2. **Check the warning's gone** after upgrading: if native FP4-MoE engages, the *"lacks native
   FP4"* line should disappear.
3. **Accuracy guardrail.** Run a small eval set on your NVFP4 model so you can prove the
   quantization (and any KV-cache changes) didn't regress your real tasks [[020]](../sources/020-nvidia-ptq-performance-accuracy.md).
4. **(Stretch) NVFP4 KV cache** trial once supported in your stack, measuring the context/batch
   headroom gain vs FP8 KV [[019]](../sources/019-nvidia-nvfp4-kv-cache-long-context.md).

### Decision table

| Situation | Do this | Source |
|---|---|---|
| Want max memory/bandwidth saving, accuracy matters | NVFP4 weights + **validate with evals** | [[003]](../sources/003-nvidia-introducing-nvfp4.md)[[020]](../sources/020-nvidia-ptq-performance-accuracy.md) |
| Seeing *"lacks native FP4 → Marlin"* | Track PR #40082 nightly **or** build FlashInfer for `sm_121` | [[011]](../sources/011-vllm-pr40082-flashinfer-b12x-sm121.md)[[016]](../sources/016-community-bjk110-spark-nvfp4-mtp.md) |
| KV cache pressure on long context | FP8 KV now; watch NVFP4 KV cache | [[002]](../sources/002-vllm-quantized-kv-cache.md)[[019]](../sources/019-nvidia-nvfp4-kv-cache-long-context.md) |
| Need a checkpoint for an unpublished model | Quantize via TensorRT-Model-Optimizer (playbook) | [[004]](../sources/004-nvidia-spark-nvfp4-playbook.md) |

---

## Where to go next

- **Guide `03` — Latency & speculative decoding:** trimming TTFT/ITL, and the MTP trick the
  Qwen recipe recommends for low-concurrency latency.
- **Guide `05` — Benchmarking:** how to actually measure the Marlin-vs-native delta above.

## Sources cited

- [[002]](../sources/002-vllm-quantized-kv-cache.md) Quantized KV Cache (vLLM)
- [[003]](../sources/003-nvidia-introducing-nvfp4.md) Introducing NVFP4 (NVIDIA)
- [[004]](../sources/004-nvidia-spark-nvfp4-playbook.md) DGX Spark NVFP4 quantization playbook (NVIDIA)
- [[005]](../sources/005-nvidia-spark-vllm-playbook.md) DGX Spark vLLM playbook (NVIDIA)
- [[006]](../sources/006-vllm-env-vars.md) vLLM environment variables
- [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) Qwen3.5/3.6 usage guide (vLLM Recipes)
- [[011]](../sources/011-vllm-pr40082-flashinfer-b12x-sm121.md) PR #40082 — FlashInfer b12x MoE/FP4 for SM120/121
- [[012]](../sources/012-vllm-issue33333-flashinfer-cutlass-sm120.md) Issue #33333 — FLASHINFER_CUTLASS unsupported on SM120
- [[013]](../sources/013-nvidia-dgx-spark-hardware-overview.md) DGX Spark hardware overview (NVIDIA)
- [[016]](../sources/016-community-bjk110-spark-nvfp4-mtp.md) Single-GPU NVFP4 + MTP, FlashInfer-from-source SM121 (community)
- [[019]](../sources/019-nvidia-nvfp4-kv-cache-long-context.md) NVFP4 KV cache for long context (NVIDIA)
- [[020]](../sources/020-nvidia-ptq-performance-accuracy.md) Post-training quantization for performance & accuracy (NVIDIA)
