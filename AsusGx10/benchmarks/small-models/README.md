# benchmarks/small-models — raw data + harness

Measured 2026-06 on the ASUS GX10 / NVIDIA GB10, image `vllm/vllm-openai:cu130-nightly`, small models
served on `:8001`. See [../../FINDINGS-small-models.md](../../FINDINGS-small-models.md) for analysis.

## Harness
- **`serve_small.sh KEY [PORT] [UTIL]`** — launch one model with its research-derived flags, wait `/health`.
- **`bench_small.py --base --model --label --out`** — correctness smoke → TTFT (3×) → single-stream decode
  tok/s (3×) → concurrency sweep (c=1,4,16,32). Streams the OpenAI API; TTFT = time to first content token.
- **`run_sweep.sh`** — standalone sweep, one model at a time (OOM-safe), Phi FP8→BF16 fallback.
- **`run_coresident.sh`** — each model on `:8001` (util 0.25) while production Qwen runs on `:8000` (util
  0.40); plus an under-load interference test on the winner.

## Results files
`<model>.json` = standalone · `<model>.coresident.json` = co-resident with Qwen (idle) ·
`qwen3-4b.coresident-underload.json` = winner measured while Qwen is hammered ·
`gpt-oss-20b.json` = blocked record (harmony tokenizer init).

## Method notes
- 3× averaged, warm-up discarded, fixed prompts, `temperature 0`, thinking-mode **off** for latency.
- Quant formats differ per model (NVFP4 / BF16 / MXFP4) by what actually exists + runs on sm_121 — every
  file records its `format`/`extra_body`. This is honest "what's fastest on *this* box," not synthetic.
- Power: Prometheus DCGM `DCGM_FI_DEV_POWER_USAGE` (`172.27.27.212:9090`), peak over each run window.
