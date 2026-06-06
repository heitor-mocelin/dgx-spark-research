# 01 · Choosing a runtime — Ollama vs llama.cpp vs vLLM

> Three ways to run Gemma 4 locally. They trade **ease** against **control** against
> **throughput**. Pick by what you're doing, not by hype.

## Plain-language on-ramp

- **Ollama** — easiest. Two commands, it picks a quantization that fits your memory, you chat.
- **llama.cpp** — most control and the widest hardware reach (CPU, Apple Metal, edge, ARM). You
  pick the GGUF quantization yourself.
- **vLLM** — the serving engine. An OpenAI-compatible HTTP endpoint, best throughput under
  concurrency, full multimodal, and the most reliable tool-calling. This is what you run to make
  Gemma 4 a *service* on the GX10.

Rule of thumb: **trying it → Ollama; a Mac/CPU/edge box → llama.cpp; a homelab endpoint or many
users → vLLM.**

## Ollama — fastest to "hello"

```bash
# needs Ollama v0.20.0+
ollama pull gemma4:26b-moe      # the MoE sweet spot (~14 GB Q4)
ollama run  gemma4:26b-moe
```

Ollama downloads the GGUF, auto-selects quantization to fit your RAM/VRAM, and manages memory
[[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md). Other tags: `gemma4:31b` (max quality), `gemma4:4b` / `gemma4:2b` (edge) [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md).

> ⚠️ **Tool-calling caveat:** Gemma 4's hybrid-attention architecture exposed bugs in Ollama's
> tool-call parser at launch — if you need function calling, prefer vLLM or raw llama.cpp until
> fixed [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md). (Apple Silicon also had early bugs.)

## llama.cpp — control, CPU/Metal, edge, ARM

Grab a pre-quantized GGUF (Unsloth publishes well-tested ones — 31B `Q4_K_M` ~18 GB, 26B-MoE
`Q4_K_M` ~14 GB) and run with `llama-server` / `llama-cli` [[g06]](../sources/g06-unsloth-gemma4.md)[[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md). Build with `-DGGML_CUDA=ON`
for NVIDIA, `OFF` for CPU; Metal is on by default on Macs [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md). For the **GB10 specifically**,
there's a community **ARM64 + CUDA 13** build with a full benchmark suite (single-seq throughput,
context scaling, multi-user, CoT timing) [[g10]](../sources/g10-shamily-gemma4-llama-dgx-spark.md). Best when you want a specific quantization,
CPU/Apple deployment, or no server.

## vLLM — the serving endpoint (and the GX10 path)

vLLM gives an OpenAI-compatible API, continuous batching, NVFP4/FP8 quantization, full multimodal,
**thinking mode**, and Gemma 4's **custom tool-call protocol** done right [[g05]](../sources/g05-vllm-recipe-gemma4.md). Gemma 4 support
needs a **recent nightly** (or `vllm/vllm-openai:latest-cu130` for CUDA 13) [[g05]](../sources/g05-vllm-recipe-gemma4.md). Minimum for
26B-A4B at BF16 is one 80 GB GPU — the GB10's 128 GB unified memory clears that easily, and NVFP4
shrinks it to ~16 GB (guide `02`) [[g05]](../sources/g05-vllm-recipe-gemma4.md)[[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md). This is the runtime the rest of this series focuses on.

## Decision table

| Your situation | Runtime | Why |
|---|---|---|
| "Let me just try Gemma 4" | **Ollama** | one command, auto-fit | 
| Mac / CPU / Raspberry-Pi / custom quant | **llama.cpp** | widest hardware, GGUF control |
| Homelab API, many users, agents/tools | **vLLM** | throughput, OpenAI API, reliable tool-calling |
| GB10 / GX10, max local speed | **vLLM + NVFP4** | ~52 tok/s on the 26B-A4B MoE (guide `02`) |
| Need function calling *today* | **vLLM** or llama.cpp | Ollama parser bug at launch [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) |

## Quantization by runtime (preview of guide `03`)

- **Ollama / llama.cpp:** GGUF — `Q4_K_M` (~14–18 GB) is the default sweet spot, `Q8_0` for higher
  fidelity [[g06]](../sources/g06-unsloth-gemma4.md)[[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md).
- **vLLM:** **NVFP4** (Blackwell) or **FP8** — NVFP4 gives the best memory/bandwidth on the GB10,
  with the SM121 kernel caveat from guide `02` [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md).

## Where to go next

- **Guide `02`** — stand up the 26B-A4B NVFP4 endpoint on the GX10.
- **Guide `03`** — quantization + memory math across formats.

## Sources cited

- [[g05]](../sources/g05-vllm-recipe-gemma4.md) Gemma 4 vLLM recipe
- [[g06]](../sources/g06-unsloth-gemma4.md) Unsloth — run Gemma 4 locally
- [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) Run Gemma 4 locally (DEV)
- [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) 52 tok/s NVFP4 on DGX Spark
- [[g10]](../sources/g10-shamily-gemma4-llama-dgx-spark.md) gemma4-llama-dgx-spark (GB10 llama.cpp)
