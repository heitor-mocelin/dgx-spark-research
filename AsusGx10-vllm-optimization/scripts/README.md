# scripts/

Reproducible shell scripts that take the GX10 from a bare device to a tuned vLLM endpoint and
back. They mirror this device's validated `docker run` exactly, and every knob is an env var so
the same scripts drive the [guide-05](../guides/05-benchmarking-and-the-tuning-loop.md) tuning
sweeps.

> ⚠️ **These restart the inference server.** A cold start is **~5–6 min** (weight load +
> `torch.compile`) and the server is unavailable during it. Run `tune.sh` only inside a
> maintenance window. `launch.sh` backs up the running container so `rollback.sh` can restore it.

## The four scripts

| Script | What it does | Safety |
|---|---|---|
| [`launch.sh`](launch.sh) | Start/restart vLLM with the production defaults (or overrides). Backs up the running container, waits on `/health`. | `DRY_RUN=1` prints the command only |
| [`benchmark.sh`](benchmark.sh) | `vllm bench serve` against the live endpoint — a single-stream (latency) pass and a saturated (throughput) pass — saving JSON + a metadata sidecar. | read-only against the server |
| [`tune.sh`](tune.sh) | Sweep one knob: relaunch + benchmark at each value, collect a summary CSV. | interactive confirm; `DRY_RUN=1` shows the plan |
| [`rollback.sh`](rollback.sh) | Restore the `vllm_prebackup` container after a bad experiment. | `DRY_RUN=1` supported |

## Quick start

```bash
# 1. Launch with current production config (defaults match the live server)
./launch.sh

# 2. Preview a change without touching anything
MAX_NUM_BATCHED_TOKENS=8192 DRY_RUN=1 ./launch.sh

# 3. Benchmark the running server (writes ./results/*.json)
./benchmark.sh

# 4. Sweep a knob across a maintenance window
SWEEP_VAR=MAX_NUM_BATCHED_TOKENS SWEEP_VALUES="2048 4096 8192 16384" ./tune.sh

# 5. Something regressed — restore last-known-good
./rollback.sh
```

## Key variables (full list in each script's header)

| Variable | Default | Guide |
|---|---|---|
| `IMAGE` | `vllm/vllm-openai:nightly` | `00`, `02` (sm_121 kernels) |
| `MODEL_PATH` | `/models/qwen36-35b-moe-nvfp4` | `00` |
| `QUANTIZATION` / `KV_CACHE_DTYPE` | `modelopt` / `fp8` | `02` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | `01`, `03` |
| `MAX_NUM_SEQS` | `128` | `01` |
| `GPU_MEM_UTIL` | `0.90` | `01` |
| `MAX_MODEL_LEN` | `32768` | `04` |
| `TOOL_CALL_PARSER` / `REASONING_PARSER` | `qwen3_coder` / `qwen3` | `04` |
| `EXTRA_ARGS` | _(empty)_ | e.g. MTP `--speculative-config '{...}'` (`03`) |
| `DOCKER` | `sudo docker` | — (set to `docker` if in the `docker` group) |

## Notes & caveats

- **Validate before trusting.** These are written against this device's config and the documented
  vLLM CLI, but **have not yet been executed on the GX10** (it's read-only until Phase 3). Flag
  names like `vllm bench serve --save-result/--result-filename/--percentile-metrics` should be
  reconciled with your installed vLLM version (`vllm bench serve --help`) on first run.
- **The marquee experiment** (`tune.sh` can't fully automate it) is swapping `IMAGE` to a nightly
  with the PR-#40082 `sm_121` FP4-MoE kernels and re-benchmarking vs the Marlin fallback — see
  [guide 02](../guides/02-quantization-nvfp4-and-fp8.md).
- **Telemetry:** pair runs with the device's dcgm-exporter → Prometheus → Grafana to see whether
  you're bandwidth-, compute-, or KV-bound (guide 05).
