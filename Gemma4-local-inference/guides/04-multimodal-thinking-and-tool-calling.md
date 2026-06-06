# 04 · Multimodal, thinking mode & tool-calling

> The three capabilities that make Gemma 4 an *agent* and not just a chatbot — and the runtime
> caveats that decide whether they actually work.

## Plain-language on-ramp

Gemma 4 can **see** (images, and audio on the small models), **think** (structured reasoning before
answering), and **act** (call your functions/tools). All three are exposed through vLLM's
OpenAI-compatible API. The catch: tool-calling was buggy in some runtimes at launch, so the runtime
you pick matters (guide `01`).

## Multimodal

Gemma 4 natively processes **text + images**; the small **E2B / E4B / 12B** also handle **audio**
[[g01]](../sources/g01-gemma4-model-card.md)[[g05]](../sources/g05-vllm-recipe-gemma4.md). Two design notes worth knowing:

- **Encoder-free 12B.** The 12B "unified" variant (`Gemma4UnifiedForConditionalGeneration`) has *no*
  separate vision/audio tower — raw pixel patches and audio frames are projected straight into the
  language model [[g05]](../sources/g05-vllm-recipe-gemma4.md). Simpler pipeline, fewer moving parts.
- **Dynamic vision resolution.** You choose how many tokens an image costs: **70 / 140 / 280 / 560 /
  1120** per image [[g01]](../sources/g01-gemma4-model-card.md)[[g05]](../sources/g05-vllm-recipe-gemma4.md). Fewer tokens = faster + cheaper KV; more = finer detail. Set it per
  request, or cap it server-side:

```bash
vllm serve <gemma4-model> --mm-processor-kwargs '{"max_soft_tokens": 560}'
```

Video is supported via a vLLM frame-extraction pipeline [[g05]](../sources/g05-vllm-recipe-gemma4.md). Send images the standard OpenAI
way (`image_url` content parts) against your endpoint.

## Thinking mode

Gemma 4 supports **structured reasoning**: it emits a thought segment delimited by
`<|channel>thought\n … <channel|>` before the final answer [[g05]](../sources/g05-vllm-recipe-gemma4.md). This improves hard
math/coding/science tasks (the AIME/GPQA numbers in guide `00` are *with* reasoning), at the cost of
extra tokens — so latency and KV use go up. Treat it like Qwen's thinking mode: **on** for hard
problems, **off** for fast tool-routing or simple chat. Toggle it through the model's chat template /
request parameters exposed by vLLM; verify the exact knob against the recipe for your build [[g05]](../sources/g05-vllm-recipe-gemma4.md).

> When you parse outputs yourself, strip the `<|channel>…<channel|>` thought span before showing the
> user — same discipline as stripping `<think>` for Qwen.

## Tool-calling / function calling

Gemma 4 uses a **custom tool-call protocol with dedicated special tokens** (not the Hermes/Qwen
formats) [[g05]](../sources/g05-vllm-recipe-gemma4.md). Through vLLM's OpenAI API you pass `tools` and `tool_choice` as usual and vLLM
maps Gemma 4's protocol to standard `tool_calls`.

> ⚠️ **Runtime matters.** Gemma 4's hybrid-attention architecture exposed bugs in **Ollama's**
> tool-call parser at launch — for reliable function calling use **vLLM** (or raw llama.cpp) until
> those are fixed [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md). This is the single biggest reason to serve agents on vLLM rather than Ollama.

This is exactly your OpenClaw use case: if you point the agent gateway at a Gemma 4 endpoint, **serve
it on vLLM** so tool calls parse correctly — the same lesson as choosing `qwen3_coder` for Qwen3.6 in
the sibling subproject.

## Putting it together

A single vLLM endpoint gives you all three over the OpenAI API: send image parts, let the model think
when it needs to, and expose your tools. Tune the vision token budget and thinking mode per workload —
they're the main levers on latency/KV for multimodal-agentic use.

## Where to go next

- **Guide `05`** — benchmark the speed cost of thinking mode and vision tokens on your box.

## Sources cited

- [[g01]](../sources/g01-gemma4-model-card.md) Gemma 4 model card
- [[g05]](../sources/g05-vllm-recipe-gemma4.md) Gemma 4 vLLM recipe (multimodal, thinking, tools)
- [[g08]](../sources/g08-run-locally-ollama-llamacpp-vllm.md) Run Gemma 4 locally (tool-call caveats)
