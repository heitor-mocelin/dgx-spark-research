# 03 · Quantization & memory

> How small can you make Gemma 4 without making it dumb, and how much context fits once you do.
> The two knobs: **weight format** (bytes per parameter) and **KV cache** (bytes per token of
> context).

## Plain-language on-ramp

A model is billions of numbers. Store each in 16 bits (BF16) and it's big and slow to read; store
each in 4 bits and it's ~4× smaller and faster, with a little accuracy loss. The **format** you pick
depends on your runtime:

- **llama.cpp / Ollama → GGUF.** `Q4_K_M` is the everyday sweet spot; `Q8_0` if you want more
  fidelity.
- **vLLM → NVFP4 or FP8.** On the GB10's Blackwell GPU, **NVFP4** (4-bit) is the bandwidth win;
  FP8 (8-bit) is the safe, broadly-supported fallback.

Separately, the **KV cache** (the model's memory of the conversation) grows with context length and
concurrency — and Gemma 4 is built to keep it small.

## Weight formats and footprints

| Variant | BF16 | GGUF Q4_K_M | NVFP4 (vLLM) | Source |
|---|---|---|---|---|
| E2B | ~5 GB | ~1.5 GB | — | [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) |
| E4B | ~9 GB | ~3 GB | — | [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) |
| 26B-A4B (MoE) | ~50 GB | **~14 GB** | **~16.5 GB** | [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md)[[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) |
| 31B (dense) | ~62 GB | ~18 GB | ~18 GB | [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) |

All fit the GB10's 128 GB many times over — **capacity is never the issue here, bandwidth is**
(guide `00`). The point of quantizing isn't to *fit*; it's to move fewer bytes per token so decode
is faster, and to free memory for KV cache.

### The Gemma-4 MoE quantization wrinkle
Quantizing the **26B-A4B** to NVFP4 needed a **custom modelopt plugin** — Gemma 4's *fused 3D expert
tensor format* isn't handled by stock NVIDIA tooling, which is why the working checkpoint is the
community `bg-digitalservices` one plus a model-code patch (guide `02`) [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md). GGUF quantization
(llama.cpp/Ollama) of the MoE is more mature via Unsloth's pre-built files [[g06]](../sources/g06-unsloth-gemma4.md).

### Quality vs size
`Q4_K_M` / NVFP4 retain most quality for a ~4× shrink — the 26B-A4B at Q4 still posts AIME 88.3% /
GPQA 82.3% [[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md). Step up to `Q8_0` / FP8 only if you measure a regression on *your* tasks; on
the GB10 the bandwidth cost of 8-bit roughly doubles per-token reads, so 4-bit is usually the right
call for the MoE.

## KV cache — and why Gemma 4 sips it

KV-cache size scales with **context length × layers × KV-heads × bytes × concurrent sequences**.
Gemma 4 attacks this structurally [[g01]](../sources/g01-gemma4-model-card.md)[[g03]](../sources/g03-hf-blog-welcome-gemma4.md):

- **Shared KV cache** — late layers reuse earlier layers' K/V, eliminating redundant KV storage.
- **Hybrid attention** — most layers are *local sliding-window*, which bounds their KV growth;
  only the global layers carry full-context KV (and use *pruned* RoPE).

The practical payoff on the GX10: with NVFP4 weights at ~16.5 GB, you have **~82 GB for KV** [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md)
— enough for **128K context across many concurrent sessions**. You can also quantize the KV cache to
**FP8** in vLLM (`--kv-cache-dtype fp8`) to roughly double tokens-per-GB again, mirroring the Qwen
setup. Context length trades against batch width: longer `--max-model-len` reserves more KV per
sequence, so set it to what you need, not the 256K max.

## A memory plan for the GX10

| Goal | Weights | KV strategy | Notes |
|---|---|---|---|
| Max speed, agent serving | 26B-A4B **NVFP4** | FP8 KV, `max-model-len` 32–128K | the guide-02 recipe |
| Max quality, low concurrency | 31B `Q8_0` / FP8 | smaller batch | slow (~7 tok/s) — accept it or don't |
| Long-context single user | 26B-A4B NVFP4 | FP8 KV, `max-model-len` 256K | shared-KV makes this cheap |
| Edge / second device | E4B `Q4_K_M` | default | ~3 GB, runs almost anywhere |

## Where to go next

- **Guide `04`** — multimodal, thinking mode, and tool-calling.
- **Guide `05`** — measure the speed/quality you actually get.

## Sources cited

- [[g01]](../sources/g01-gemma4-model-card.md) Gemma 4 model card • [[g03]](../sources/g03-hf-blog-welcome-gemma4.md) HF Welcome Gemma 4
- [[g06]](../sources/g06-unsloth-gemma4.md) Unsloth GGUFs • [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) Run locally (DEV)
- [[g09]](../sources/g09-ai-muninn-dgxspark-nvfp4-52.md) NVFP4 on DGX Spark • [[g14]](../sources/g14-lushbinary-gemma4-benchmarks.md) Benchmarks (Lushbinary)
