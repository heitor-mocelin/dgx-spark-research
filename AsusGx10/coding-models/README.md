# coding-models — coding-specialized LLMs on the DGX Spark (GB10)

Performance + correctness benchmarks for **coding-specialized** models on a single ASUS GX10 / NVIDIA GB10,
vLLM nightly. Extends the repo's small-models bench-loop with two extra dimensions: **HumanEval pass@1** and a
**context-length sweep**.

**Read the results:** [`FINDINGS-coding-models.md`](./FINDINGS-coding-models.md) · raw JSON in [`results/`](./results/).

## Models (ready-made quants only)

| Key | Model | Quant (HF) |
|---|---|---|
| `qwen3-coder-30b` | Qwen3-Coder-30B-A3B-Instruct (MoE) | NVFP4 — `ig1/Qwen3-Coder-30B-A3B-Instruct-NVFP4` |
| `qwen25-coder-32b` | Qwen2.5-Coder-32B-Instruct | FP8 — `RedHatAI/Qwen2.5-Coder-32B-Instruct-FP8-dynamic` |
| `qwen25-coder-14b` | Qwen2.5-Coder-14B-Instruct | FP8 — `RedHatAI/…-14B-…-FP8-dynamic` |
| `qwen25-coder-7b` | Qwen2.5-Coder-7B-Instruct | FP8 — `RedHatAI/…-7B-…-FP8-dynamic` |
| `deepseek-coder-v2-lite` | DeepSeek-Coder-V2-Lite-Instruct (MoE/MLA) | FP8 — `RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8` |
| `devstral-24b` | Devstral-Small-2-24B (agentic) | NVFP4 — `Firworks/Devstral-Small-2-24B-Instruct-2512-nvfp4` |
| `codestral-22b` | Codestral-22B-v0.1 (FIM) | FP8 — `TechxGenus/Codestral-22B-v0.1-FP8` |

## Harness

- **`serve_coding.sh KEY [PORT] [UTIL]`** — launch one model with its GB10/sm_121-proven flags, wait `/health`.
  Encodes the per-model compatibility fixes (auto-detect quant; DeepSeek MLA without fp8 KV; Codestral chat template).
- **`bench_small.py`** — correctness smoke → TTFT (3×) → single-stream decode tok/s (3×) → concurrency sweep
  (c=1/4/16/32). (From `../benchmarks/small-models/`.)
- **`humaneval.py {gen,score}`** — HumanEval pass@1, instruct-style, **16-way parallel** generation; scoring runs
  the model's code in a `--network none` sandbox.
- **`bench_ctx.py`** — TTFT + decode tok/s at increasing real prompt-token counts.
- **`run_coding_sweep.sh`** — the full unattended bench-loop over all models (download → serve → perf → HumanEval
  → context → teardown), resume-safe. `run_one.sh KEY [CTX]` does a single model.
- **`aggregate.py`** — JSON → `results/combined.json` + `FINDINGS-coding-models.md`.
- **`codestral-chat.jinja`** — Mistral `[INST]` template so the base/FIM Codestral can be chat-benchmarked.

## Reproduce

```bash
# on the DGX (user in docker group), with models dir at /home/user/models
bash run_coding_sweep.sh           # downloads + benchmarks all models into /home/user/results-coding
python3 aggregate.py /home/user/results-coding FINDINGS-coding-models.md
```

Method notes, caveats (incl. the 7B chat-harness caveat), and the GB10 compatibility fixes are in the FINDINGS.
