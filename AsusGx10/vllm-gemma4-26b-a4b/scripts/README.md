# scripts/ — run Gemma 4 locally

| Script | What it does |
|---|---|
| [`launch-gemma4-vllm.sh`](launch-gemma4-vllm.sh) | Serve **26B-A4B NVFP4** on the GX10/GB10 via vLLM (the [guide-02](../guides/02-running-on-the-gx10-vllm-nvfp4.md) recipe): downloads the community NVFP4 checkpoint, mounts the `gemma4` model patch, serves an OpenAI endpoint on `:8001`, health-gated. |
| [`quickstart-ollama.sh`](quickstart-ollama.sh) | Fastest path anywhere — `ollama pull/run gemma4:26b-moe`. |

**Benchmarking** reuses the runtime-agnostic OpenAI-API client from the sibling subproject —
[`../../vllm-qwen3.6-35b-a3b/benchmarks/benchmark_sweep.py`](../../vllm-qwen3.6-35b-a3b/benchmarks/benchmark_sweep.py)
(point its `URL`/`MODEL` at your Gemma 4 endpoint).

## Notes & caveats

- **Not yet run on this GX10.** The DGX currently serves Qwen3.6, and `sudo docker` there is
  password-gated — so these scripts are validated by `bash -n` and the cited recipe, not an on-device
  run. Reconcile flags with your installed vLLM (`vllm serve --help`) on first use.
- **Port 8001** by default so Gemma 4 can run **alongside** the Qwen3.6 server on 8000 (memory
  permitting on the 128 GB box).
- **Gated weights:** accept the Gemma license on Hugging Face and set `HF_TOKEN` before downloading.
- **Tool-calling:** use the vLLM path (not Ollama) for reliable function calling at launch — see
  [guide 04](../guides/04-multimodal-thinking-and-tool-calling.md).
- **Marlin caveat:** the ~52 tok/s is with the `sm_121` FP4→BF16 Marlin fallback; native FP4-MoE
  kernels (FlashInfer b12x / vLLM PR #40082) should raise it — [guide 02](../guides/02-running-on-the-gx10-vllm-nvfp4.md).
