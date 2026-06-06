# 05 · Benchmarking & the tuning loop

> **The guide that makes the others true.** Every "to measure in Phase 3" flag in this series
> resolves here. You can't optimize what you don't measure, and on a bandwidth-bound box the
> only way to know where a knob tips is to sweep it and watch the numbers.

---

## Plain-language on-ramp

Benchmarking an inference server means: send it a controlled stream of requests, and record how
fast it answers. Four numbers matter:

- **TTFT** — time to first token (responsiveness).
- **ITL / TPOT** — inter-token latency / time per output token (streaming smoothness).
- **Output throughput** — total generated tokens/sec across all requests (capacity).
- **Request throughput** — completed requests/sec at your latency budget.

There's a fundamental shape to every inference benchmark: as you push more concurrent load,
**throughput rises but latency rises too**, until throughput plateaus (you're bandwidth/compute
bound) and latency keeps climbing (requests queue). Your job is to find the load where you're
getting most of the throughput while latency is still acceptable — and to compare *configurations*
at that operating point.

vLLM ships the tool for this: **`vllm bench serve`**.

---

## `vllm bench serve` in depth

A representative run (from the Qwen recipe) [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md):

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model qwen36-moe \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 1000 \
  --request-rate 20
```

The knobs that shape the experiment [[007]](../sources/007-vllm-benchmark-cli.md):

- **`--request-rate`** — target requests/sec; set to **`inf`** to find max throughput (fire
  everything at once), or a finite rate to test a latency budget.
- **`--max-concurrency`** — cap simultaneous in-flight requests (the cleanest way to define an
  operating point).
- **`--burstiness`** — Gamma-distributed arrival variability; low = bursty, high = uniform.
  Bursty traffic stresses the scheduler differently than steady traffic.
- **`--random-input-len` / `--random-output-len`** — fix prompt/response sizes so runs are
  comparable; choose values that resemble your real traffic.
- **`--metric-percentiles`** — which percentiles to report (default p99); report **p50 and p99**.
- **`--output-json`** — save results for diffing across configs. **Always use this** — it's how
  the sweeps below become a table.
- vLLM also supports **ramping** the request rate over a run to find the saturation knee [[007]](../sources/007-vllm-benchmark-cli.md).

### Methodology that produces trustworthy numbers

1. **Warm up first.** Discard the first run after a (5–6 min) cold start — `torch.compile` and
   cache warmup distort early numbers.
2. **Change one variable at a time.** Hold input/output lengths and concurrency fixed; sweep the
   single knob under test.
3. **Separate the two regimes.** Run a **single-stream** pass (`--max-concurrency 1`) for *latency*
   truth, and a **saturated** pass (`--request-rate inf`) for *throughput* truth. They answer
   different questions; don't average them.
4. **Report p50 *and* p99.** A good mean can hide an ugly tail.
5. **Pin everything.** Record the exact image tag, model, all serve flags, and env vars per run —
   results are only comparable against a fixed stack (the community Spark repos follow this
   single-stream + multi-user-concurrency + stress-test structure [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md)[[015]](../sources/015-community-markramsey-vllm-dgx-spark.md)).

---

## The experiment plan (everything this series deferred)

Each row is a `vllm bench serve` sweep; together they're the `benchmark.sh` + `tune.sh` backbone:

| # | Experiment | Vary | Measure | From guide |
|---|---|---|---|---|
| 1 | Batch depth | `--max-num-batched-tokens` 2048/4096/8192/16384 | tok/s, TTFT, ITL | `01` |
| 2 | Batch width | `--max-num-seqs` 64/128/256 | tok/s, preemption warnings | `01` |
| 3 | **FP4 kernel** | Marlin fallback vs PR-#40082 b12x build | tok/s, TTFT, "lacks native FP4" gone? | `02` |
| 4 | KV dtype | FP8 vs (later) NVFP4 KV | max context/batch, accuracy | `02`/`04` |
| 5 | Spec decode | MTP off vs k=1/2/3 @ concurrency 1 | TPOT, acceptance rate | `03` |
| 6 | Spec under load | MTP on/off across concurrency | throughput crossover | `03` |
| 7 | Prefix caching | on vs off on real agent traffic | TTFT, tok/s | `01`/`04` |
| 8 | Context budget | `--max-model-len` 32k/64k | concurrent sessions before preemption | `04` |
| 9 | Memory util | `--gpu-memory-utilization` 0.90/0.92/0.95 | KV headroom vs UMA OOM | `01` |
| 10 | Accuracy guardrail | quantized vs reference on a small eval set | task accuracy delta | `02` |

Experiment **#3 is the marquee result** for this hardware — it's the dollar value of solving the
`sm_121` Marlin fallback.

---

## Correlate with hardware telemetry

Token-rate numbers are more meaningful next to what the GPU was doing. Your homelab already has
the stack for this: **dcgm-exporter → Prometheus → Grafana** on the GX10. For each benchmark run,
capture GPU utilization, memory used, and power draw, and line them up against tok/s. That tells
you *which* ceiling you hit:

- **High tok/s, GPU util < 100%, memory bandwidth saturated** → bandwidth-bound (expected on this
  box; quantization is your lever).
- **GPU util pegged at 100%** → compute-bound (the regime where fixing the FP4 kernel, experiment
  #3, pays off).
- **Memory near `gpu-memory-utilization` cap + preemption warnings** → KV-starved (widen memory or
  narrow batch).

*(Telemetry stack is your existing infrastructure, not a corpus source — see your Grafana/Prometheus
setup.)*

---

## On *your* GX10 — the Phase-3 protocol

> **Read-only until you green-light Phase 3.** These runs require restarting the `vllm` container
> (~5–6 min cold start each) and brief downtime — schedule a maintenance window.

A pragmatic order:

1. **Baseline.** Pin the current image + flags, warm up, run single-stream + saturated passes,
   `--output-json`. This is the number every change is measured against.
2. **Cheap wins first** (no image change): experiments #1, #2, #7, #9 — pure flag sweeps.
3. **Latency profile**: #5, #6 (MTP).
4. **The big one**: #3 — stand up a PR-#40082 nightly (or a FlashInfer-from-source `sm_121` build
   per [[016]](../sources/016-community-bjk110-spark-nvfp4-mtp.md)) and compare to baseline. Confirm the *"lacks native FP4"* warning is gone.
5. **Guardrail**: #10 after any quantization/KV change.

Each run appends a row to a results table (template below) and, where a winner emerges, updates the
defaults in `launch.sh`. Mind the safety nets you already have: rollback script
`/home/user/vllm-rollback.sh` and the preserved `vllm_prebackup` container.

### Results — measured 2026-06-06 (see [`../benchmarks/`](../benchmarks/README.md))

Baseline = production config (NVFP4, FP8 KV, 32k, `max-num-seqs 128`, Marlin FP4 fallback),
measured via a dependency-free streaming HTTP client (`vllm bench`/`docker exec` need password
`sudo` on the DGX — see the blocker note below).

| Concurrency | Aggregate tok/s | TTFT p50 | ITL p50 |
|---:|---:|---:|---:|
| 1 (single-stream) | 75 | 55 ms | 13 ms |
| 32 | 671 | 313 ms | 47 ms |
| 48 | 753 | 484 ms | 62 ms |
| **96 (peak)** | **951** | 953 ms | 96 ms |
| 128 | 931 | 881 ms | 99 ms |

**Findings:** peak ~951 tok/s at c≈96 then saturation-regression at 128; efficient band c≈32–48;
~12.6× batching amplification; single-stream ~½ the weights-only ceiling (Marlin fallback suspected).

**Still blocked (need non-interactive docker on the DGX):** the restart-based flag sweeps
(experiments 1, 2, 9) and the **marquee experiment 3** (Marlin vs PR-#40082 native FP4). One-time
fix: `sudo usermod -aG docker user`, then `scripts/tune.sh` can run them.

---

## Where to go next

- **`scripts/`** — `launch.sh`, `benchmark.sh`, `tune.sh`, `rollback.sh` operationalize this plan
  (Phase 4).
- Back to **guide `00`** for the map, or any topic guide to revisit a specific knob.

## Sources cited

- [[007]](../sources/007-vllm-benchmark-cli.md) Benchmark CLI (vLLM)
- [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) Qwen3.5/3.6 usage guide (vLLM Recipes)
- [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md) Qwen3.5-35B-A3B on DGX Spark (community)
- [[015]](../sources/015-community-markramsey-vllm-dgx-spark.md) vllm-dgx-spark preset matrix (community)
- [[016]](../sources/016-community-bjk110-spark-nvfp4-mtp.md) Single-GPU NVFP4 + MTP, FlashInfer-from-source SM121 (community)
