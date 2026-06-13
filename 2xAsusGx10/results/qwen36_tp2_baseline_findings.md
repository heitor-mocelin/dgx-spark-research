# Baseline: Qwen3.6-35B-A3B, TP=2 across 2× DGX Spark (RDMA/RoCE)

Date: 2026-06-13. Model fits one box (22 GB NVFP4) — run split only to measure the distribution tax.
Config: `vllm-ray:local`, TP=2, `--distributed-executor-backend ray --enforce-eager --kv-cache-dtype fp8 --max-model-len 32768 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85`. NCCL on `NET/IB` (both x4 rails). Head = DGX1 (.210), worker = DGX2 (.211).

## Performance — the distribution tax
| Metric | Cluster TP=2 | Single-box (prior) | Ratio |
|---|---|---|---|
| Single-stream decode | **25.6 tok/s** | ~75 tok/s | **2.9× slower** |
| TTFT | 0.10 s | 0.058 s | — |
| Peak aggregate | **346 tok/s** @ c=16 | ~951 tok/s @ c=96 | 2.7× lower |
| Sweep | c1=24.5, c4=95.4, c16=346.4 tok/s | — | — |

## Interconnect — bandwidth is a non-issue
- Under sustained c=8 load: DGX1 link **TX 1.46 / RX 1.46 Gb/s** (bidir 2.92) = **0.73% of the 200 Gb/s ceiling**.
- Per-rail balanced: `rocep1s0f0` 0.74 + `roceP2p1s0f0` 0.72 Gb/s → NCCL stripes evenly across both x4 lanes.
- Whole-run cumulative: ~21 GB over ~7 min ≈ 0.4 Gb/s average.
- **Conclusion: the tax is latency (per-token cross-node sync), not bandwidth.** Dual-rail vs single-rail is irrelevant for inference; the one-cable decision loses nothing.

## Thermals & power
- Idle: 43 °C, ~4 W, 208 MHz (SM idle clock), both GPUs.
- Under load: GPU die **56–58 °C**, board ~48 °C, total power **36 W**, util **29%**. Never thermally throttled; GPUs starved (waiting on sync), not compute- or bandwidth-bound.
- GB10 does not expose `DCGM_FI_DEV_MEMORY_TEMP` (reads 0) — GPU die temp is the thermal signal.

## Memory pooling (the actual reason to cluster)
- KV pool **13.2M tokens → 403× concurrency at 32K**, **88.76 GiB** KV available — two boxes' memory combined. This is the payoff, not speed.

## Stability / gotchas (article-worthy)
1. **c=32 crashes the engine**: `TimeoutError: RPC call to sample_tokens timed out` — the cross-node worker RPC exceeds vLLM's timeout under high concurrency. Safe envelope **c ≤ 16**.
2. **RDMA is invisible to `node_network_*`** counters (kernel bypass) — must use `node_infiniband_port_data_{transmitted,received}_bytes_total`.
3. **Docker needs `--device=/dev/infiniband/uverbs*`** (not a bind-mount) or NCCL silently falls back to `NET/Socket` (TCP).
4. **Qwen3.6 hybrid needs `--max-num-batched-tokens ≥ 2096`** (Mamba block_size) or engine-core crashes at init.

## Takeaway
Splitting a model that fits one box is a ~3× throughput loss, bounded entirely by interconnect latency. The cluster's value is exclusively the pooled memory for models that DON'T fit 128 GB — which is the hero-model test (GLM-4.6) next.
