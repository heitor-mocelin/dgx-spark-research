# benchmarks/ — measured results (Phase 3)

Real measurements on the GX10, **2026-06-06**, against the live production config
(Qwen3.6-35B-A3B NVFP4, FP8 KV, 32k ctx, `--max-num-seqs 128 --max-num-batched-tokens 4096`,
`vllm/vllm-openai:nightly`, **Marlin FP4 fallback active**).

Measured with a dependency-free streaming HTTP client ([`benchmark_http.py`](benchmark_http.py),
[`benchmark_sweep.py`](benchmark_sweep.py)) hitting the OpenAI endpoint directly — chosen because
`docker exec`/`vllm bench serve` need `sudo` on the DGX (password-gated), so the in-container path
in [`../scripts/`](../scripts/) couldn't run unattended. Output: 192 tokens/request, `ignore_eos`,
greedy, thinking disabled.

## Baseline ([baseline_bench.json](baseline_bench.json))

| Regime | Result |
|---|---|
| **Single-stream** (c=1) | TTFT p50 **58 ms**, ITL **13 ms** → **~75 tok/s/stream** |
| **Concurrent** (c=32) | **627.6 tok/s** aggregate, TTFT p50 351 ms / p99 504 ms |

Confirms the project's "~600 tok/s" figure, and an **~8.3× batching amplification** at c=32.

## Concurrency sweep ([concurrency_sweep.json](concurrency_sweep.json))

| Concurrency | Aggregate tok/s | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 75.2 | 54.6 | 58.4 | 13.1 |
| 2 | 123.8 | 61.6 | 63.8 | 16.2 |
| 4 | 199.4 | 111.4 | 126.6 | 20.2 |
| 8 | 338.9 | 124.4 | 127.3 | 23.8 |
| 16 | 467.4 | 210.0 | 282.3 | 33.4 |
| 32 | 671.1 | 312.7 | 321.7 | 46.8 |
| 48 | 753.3 | 484.1 | 510.9 | 61.5 |
| 64 | 774.5 | 547.8 | 735.3 | 73.2 |
| **96** | **951.4** (peak) | 953.0 | 1162.3 | 96.2 |
| 128 | 931.1 | 880.5 | 1094.1 | 98.8 |

### Reading the curve

- **Peak aggregate ≈ 951 tok/s at c≈96**, then it *regresses* at 128 — the server is saturated;
  extra concurrency past the knee buys no throughput and much worse latency.
- **Efficient operating band ≈ c32–48** (670–750 tok/s) where latency is still moderate
  (ITL ≤ ~62 ms, TTFT ≤ ~0.5 s). This is the throughput/latency sweet spot for agentic serving.
- **~12.6× batching amplification** (75 → 951 tok/s) — the guide-01 "amortize the weight read"
  thesis, measured.
- **Single-stream ~75 tok/s is ~half** the weights-only bandwidth ceiling (~160 tok/s est., guide
  00). Plausible contributors: KV/activation reads + the **Marlin FP4 fallback** (not the native
  W4A4 path). Disentangling that needs the marquee experiment below.

## What's NOT measured yet — and why

The restart-based flag sweeps and the **marquee Marlin-vs-native FP4 experiment** (guide 02/05)
require restarting the container / swapping the image — i.e. `sudo docker` on the DGX, which is
**password-gated** for the `user` account. To unblock unattended runs, one-time on the DGX:

```bash
sudo usermod -aG docker user && newgrp docker   # then `docker …` needs no sudo
```

With that, [`../scripts/launch.sh`](../scripts/launch.sh) + [`tune.sh`](../scripts/tune.sh) can run
the `max-num-batched-tokens` / `max-num-seqs` / `gpu-memory-utilization` sweeps and the
Marlin-vs-b12x comparison, appending rows here.

## Method & caveats

- Single warm server, one config; the sweep loads the **production** endpoint (briefly saturating
  it at high concurrency). Numbers are wall-clock client-side, not vLLM-internal counters.
- `ignore_eos` forces fixed 192-token outputs for comparability; real traffic varies.
- Re-run: `python3 benchmark_sweep.py` (from anywhere that reaches `http://172.27.27.210:8000`).
