# Transport study, blocked models, and cluster-stability findings

Captured 2026-06-13 on the 2× DGX Spark (GB10) cluster, vLLM 0.22.1rc1 nightly + Ray, TP=2 over RoCE.

## Transport study — RDMA vs TCP (the empirical "latency vs bandwidth" test)

Same model (`Qwen3-Coder-30B-A3B`, NVFP4), same RoCE wire, only the NCCL transport changed (`NCCL_IB_DISABLE=0` vs `=1`, verified in NCCL logs: `Using network IB` vs `Using network Socket`). Single-stream, 6 reps, 200 tokens.

| Transport | Decode tok/s | TTFT |
|---|---|---|
| **IB / RDMA** | **36.0** | **52 ms** |
| **TCP / Socket** | **35.7** | **111 ms** |

**Finding (corrects an earlier overstatement):** decode throughput is *identical* — forcing TCP did **not** cripple decode. NCCL overlaps the per-layer all-reduces with compute, hiding the transport latency during steady-state decode of a low-active (3B-active) model. RDMA's advantage is in **TTFT (~2×, prefill is communication-heavier and less overlappable)** and would grow under high concurrency (bigger all-reduce payloads). So for single-stream decode, the link matters even less than the latency framing implied — GPU compute/memory is the bottleneck, not the interconnect (RDMA *or* TCP). This reinforces the core thesis: **for serving big models, the interconnect's headline specs — bandwidth and even RDMA latency — are largely irrelevant to decode.**

## Blocked models (negative findings — not every NVFP4 model runs distributed on Spark)

GLM-4.7 (plain `modelopt` NVFP4 + standard all-reduce) ran fine. Two others did not, for structural reasons:

- **DeepSeek-V4-Flash** (`nvidia/DeepSeek-V4-Flash-NVFP4`, `deepseek_v4`, NVFP4-MoE): architecture *is* supported by vLLM, but **no NVFP4-MoE kernel works on sm_121/GB10**. Progression: needs `--kv-cache-dtype fp8` (ok) → needs explicit `--moe-backend` → `flashinfer_cutlass` rejects it (`Model sets swiglu_limit=10.0` but cutlass doesn't apply the SwiGLU clamp) → `flashinfer_trtllm` rejects it (`NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support the deployment configuration since kernel does not support current device cuda`). **Verdict: won't run on DGX Spark with this stack.**
- **MiniMax-M2.7** (`nvidia/MiniMax-M2.7-NVFP4`, `minimax_m2`): loads weights but **fails at its custom fused-RMSNorm all-reduce** (`minimax_rms_norm/lamport_workspace.py` → `cudaErrorInvalidResourceHandle`). The "Lamport" one-shot all-reduce uses CUDA IPC/peer handles designed for **NVLink-connected GPUs in one box** — invalid across two physically separate Sparks. `--disable-custom-all-reduce` doesn't bypass it (it's hardcoded in the model). **Verdict: MiniMax-M2's TP optimization is single-node-only; won't run multi-node.**

**Lesson:** models that use exotic TP optimizations (special NVFP4-MoE kernels, single-node IPC all-reduce) may not survive multi-node deployment on GB10. The models that *work* are the ones using plain `modelopt` NVFP4 + standard NCCL all-reduce (GLM-4.7, Qwen3.x).

## Cluster-stability finding

Repeated failed engine inits (the DeepSeek ×4 and MiniMax ×3 attempts) **degraded the cluster** — a subsequent clean model (`qwen3-coder-30b`) deployed but then crashed during inference with `RuntimeError: RPC call to sample_tokens timed out` (the same cross-node RPC timeout that kills high-concurrency runs). **A fresh `docker rm` + recreate of the Ray containers fully restored stability** (3/3 clean probe requests after). So: on this multi-node setup, fail-and-retry cycles accumulate bad GPU/Ray state; recreate the containers between failed deploys rather than re-`exec`-ing into the same ones.

## TP=2 vs PP=2 (Qwen3-Coder-30B-A3B, fits one box, distributed two ways)

Single-box reference ≈ 73 tok/s (from the single-Spark study). Distributed two ways over the same IB/RDMA link:

| Mode | Single-stream decode | TTFT | c8 aggregate | % of single-box |
|---|---|---|---|---|
| **TP=2** (tensor-parallel) | 36 tok/s | 52 ms | — | 49% |
| **PP=2** (pipeline-parallel) | **57 tok/s** | 202 ms | ~304 tok/s | **78%** |

**Finding:** PP=2 is **58% faster single-stream** than TP=2. TP all-reduces twice per layer (96 syncs/token for this 48-layer model); PP hands off the activation **once per token** at the stage boundary. With cross-node sync being the dominant distribution overhead, PP's communication-light design recovers most of the single-box speed, while TP loses ~half to sync barriers. The trade-off flips for TTFT (TP 52 ms vs PP 202 ms — PP's prefill traverses both stages sequentially through the bubble) and PP needs concurrency to fill the pipeline (c8 ≈ 304 tok/s). **Takeaway: for a model that fits one box but is run distributed, prefer PP for decode-heavy / low-concurrency; this is also why llama.cpp's RPC layer-split can be competitive.**

## Security: specialist vs generalist (single-box)

Both single-box (security models are small — not a cluster workload). Same vuln-analysis prompts to each:

| Model | Size | Prompt: `strcpy` overflow | Prompt: format-string `%x/%n` |
|---|---|---|---|
| **WhiteRabbitNeo-V3-7B** (specialist, Qwen2.5-Coder base, uncensored) | 7B / 15 GB | correct, detailed | correct, detailed |
| **Qwen3-Coder-30B-A3B** (generalist) | 30B-A3B | correct, well-structured | correct, well-structured |

**Finding:** comparable quality — the 7B specialist held its own against the 30B generalist, and **neither refused**. The specialist's differentiation (uncensored offensive operations) wasn't strongly exposed by these prompts because a code-tuned generalist isn't safety-gated either. For vuln analysis, a small specialist on **one** Spark is sufficient; this is not a cluster job.
