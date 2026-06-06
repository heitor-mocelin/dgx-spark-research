# vllm-qwen3-next-80b-a3b

**Qwen3-Next-80B-A3B** (NVFP4) on the GX10 — an **80B-total / 3B-active** MoE built on a
**linear-attention hybrid** (Gated DeltaNet + Gated Attention). In the [test matrix](../FINDINGS.md)
it's the **lowest-efficiency model measured (~27% of roofline)** — the strongest evidence that
linear-attention hybrids sit far below the attention-roofline. Benchmark-centric subproject.

## The model

| | |
|---|---|
| Params | **80B total / 3B active** MoE (ultra-sparse, ~1:50 activation) |
| Architecture | **Hybrid**: Gated DeltaNet (linear attention) + Gated Attention, **3:1** ratio |
| Context | 256K native, **1M with YaRN** |
| Quality | ≈ dense Qwen3-32B at <10% of its training cost |
| License | Apache-2.0 (Qwen) |
| NVFP4 footprint | ~48 GB |

## Headline result (measured 2026-06-06)

| Single-stream | Peak aggregate | Efficiency vs roofline | Single-stream GPU util |
|---:|---:|---:|---:|
| **35.5 tok/s** | **1021 tok/s** | **~27%** (matrix min) | **60.6%** (matrix min) |

**Why it matters:** same 3B active as Qwen3.6 (75 tok/s) and Nemotron-Nano (54), but it measured
**only 35.5** — *and* with the **lowest single-stream GPU utilization (60%)** of any model. Low
util + low throughput means it's **neither compute- nor (Marlin-)bound** single-stream — the Gated
DeltaNet recurrence appears **latency-bound per token** (the state update serializes), idling the
GPU. Yet it still batches to **1021 tok/s** aggregate. See [FINDINGS](../FINDINGS.md).

## Contents

- **[guides/](guides/00-overview-deployment-and-results.md)** · **[scripts/](scripts/launch.sh)** ·
  **[benchmarks/](benchmarks/README.md)** · **[sources/](sources/INDEX.md)**

[MIT](../../LICENSE) © 2026 Heitor Mocelin
