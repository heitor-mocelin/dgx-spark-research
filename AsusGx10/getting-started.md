# Local Inference on a DGX Spark (GB10): A Newcomer's Guide

> **How this was written:** drafted by the **local Qwen3.6-35B-A3B model itself** (running on this
> GX10) evaluating this project's [measured findings](FINDINGS.md), then fact-checked by a human —
> two of its recommendations were corrected (noted ⚠️ below). A fitting test: the box's own model
> summarizing what the box is good at. Every number traces to [the benchmarks](benchmarks/README.md).

> ⚡ **Fastest path — one command:** [`bootstrap/dgx-spark-bootstrap.sh`](bootstrap/README.md) takes a
> fresh box → a serving endpoint (detects your stack, lets you pick a model, deploys, verifies, logs
> everything). The sections below explain the *choices* it makes.

## 1. What to expect

The NVIDIA GB10 Grace Blackwell (128 GB unified memory) makes local inference about **memory
bandwidth (273 GB/s), not raw compute.** The practical upshot: **Mixture-of-Experts (MoE) models
with few active parameters are far faster than dense models** of similar size. Efficiency is
nuanced — big dense models nearly hit their (low) theoretical ceiling, while small MoEs leave more
on the table to overhead — and under heavy load you hit a **power cap (~44–71 W)**, not a bandwidth
wall. Capacity is generous; speed is what you budget for.

## 2. Pick your model

| Goal | Recommended model | Why |
| :--- | :--- | :--- |
| **Snappy single-user chat** | **Qwen3.6-35B-A3B** (MoE) | Fastest single-stream — **75 tok/s** |
| **Serve many users** | **Nemotron-3-Nano-30B-A3B** (MoE) | Highest aggregate **1215 tok/s** at lowest power (**44 W**) |
| **Long context / RAG** | **Qwen3-Next-80B-A3B** (MoE) | 1M-token context via linear attention; 1021 tok/s aggregate ⚠️ |
| **Maximum capability** | **Nemotron-3-Super-120B-A12B** (MoE) | A 120B brain that's still usable — 15 tok/s, 1M ctx |
| **Highest single-model quality** (latency-tolerant) | **Llama-3.3-70B** (dense) | Frontier dense quality; slow at **5.4 tok/s** but runs at 98% of its ceiling ⚠️ |
| **Tiny / edge** | *not benchmarked here* — use a small model (e.g. Gemma-4-E4B, Qwen3-4B) | Every model in this study is 30B+; for phones/Pi-class go far smaller ⚠️ |

> ⚠️ *Human corrections to the model's draft:* it had put Qwen3-Next under "highest quality" (it's
> actually our **lowest-efficiency** model — its real edge is long context) and listed 31–70 B dense
> models as "tiny/edge" (they're large). Fixed above.

**The one rule that explains the table:** single-stream speed ≈ `bandwidth ÷ (active_params ×
bytes)`. MoE beats dense **5–14×** here. Unless you specifically need a dense model, **pick an MoE.**

## 3. Pick your runtime
- **vLLM** — for serving: best throughput, an OpenAI API, reliable tool-calling. (What this study used.)
- **Ollama** — the easiest start: two commands, auto-fits a quantization to your memory.
- **llama.cpp** — for CPU, Apple/Metal, edge, or custom GGUF quantization.

## 4. Pick your quantization
- **NVFP4** — best on Blackwell (4-bit); note it currently falls back to the Marlin FP4→BF16 kernel on `sm_121`.
- **FP8** — the safe, broadly-supported 8-bit option.
- **GGUF Q4_K_M** — the default for Ollama / llama.cpp.

## 5. Top 5 mistakes to avoid
1. **`--gpu-memory-utilization` too high on a big model.** On a 60 GB+ model, 0.85 OOM-**wedges the
   whole box** — unified memory is shared CPU+GPU. Use **~0.55** and load one big model at a time.
   *(We learned this the hard way — it hard-crashed the machine.)*
2. **Expecting dense models to be fast.** Llama-3.3-70B is 5.4 tok/s here; an MoE is 5–14× quicker.
3. **Ignoring the power cap.** Aggregate throughput tops out at ~44–71 W / 96% GPU util — past the
   knee, more concurrency just adds latency.
4. **Overestimating linear-attention hybrids.** Mamba-2 / Gated-DeltaNet models (Nemotron-Nano,
   Qwen3-Next) run *below* the attention-roofline single-stream — great for batched serving, slower
   for one user.
5. **Assuming the roofline is exact.** It's tight for big/dense models (~90–98%) but only a loose
   upper bound for small-active MoEs (~40–58%).

## 6. Your first hour
1. **Easiest start — Ollama:** install it, then `ollama pull gemma4:26b-moe` (or a model from §2) and
   `ollama run …`. You're chatting in minutes.
2. **Watch the speed:** a 3B-active MoE should feel snappy (tens of tok/s); if you tried a dense 30B+
   it'll crawl — that's expected (§5.2).
3. **Step up to a served endpoint — vLLM:** pull `vllm/vllm-openai:nightly`, `vllm serve <model>
   --quantization modelopt --kv-cache-dtype fp8 --gpu-memory-utilization 0.85` (use **0.55** for
   60 GB+ models). You get an OpenAI-compatible API on `:8000`.
4. **Verify:** `curl localhost:8000/v1/models`, then a `/v1/chat/completions` request.
5. **Benchmark before tuning:** measure single-stream *and* a concurrency sweep (see
   [benchmarks/](benchmarks/README.md)); find where throughput plateaus and latency climbs — that's
   your operating point.

---
*Deeper dives: [FINDINGS.md](FINDINGS.md) (the measured study) · per-model subprojects under
[AsusGx10/](README.md) · methodology in [testing-plan.md](testing-plan.md).*
