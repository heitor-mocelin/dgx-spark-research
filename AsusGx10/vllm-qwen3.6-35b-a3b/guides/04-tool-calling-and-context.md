# 04 · Tool-calling & long context

> **Why these two together:** they're what makes your model *useful as an agent* rather than a
> chatbot. Tool-calling lets the model act; long context lets it remember. Both are configured
> at serve time, and both interact with the memory/batching budget from earlier guides.

---

## Plain-language on-ramp

**Tool-calling** is how the model asks your software to *do* something — call a function, query
an API, run a shell command — instead of just writing prose. The model emits a structured
"please call `get_weather(city=…)`" message; your client runs it and feeds the result back.

The wrinkle: each model family writes those call requests in its *own* text format. vLLM needs a
matching **parser** to turn that model-specific text into the clean OpenAI `tool_calls` JSON your
client expects. Pick the wrong parser and tool-calling silently breaks. For Qwen3.6 the right one
is **`qwen3_coder`** — which is exactly what you run.

**Long context** is how many tokens of conversation/history the server will hold. Bigger context
= more memory reserved for the KV cache = fewer concurrent requests. So context length is a
*budget* you trade against batch width (guide `01`).

---

## Tool-calling in depth

### The flags
Two serve-time flags switch it on [[008]](../sources/008-vllm-tool-calling.md):

```bash
--enable-auto-tool-choice --tool-call-parser <parser>
```

`--enable-auto-tool-choice` lets the model decide *when* to call a tool; `--tool-call-parser`
tells vLLM how to decode that model's tool-call syntax.

### `tool_choice` semantics
The OpenAI-compatible API supports, per request [[008]](../sources/008-vllm-tool-calling.md):

- **`auto`** — model decides whether/which tool to call (needs `--enable-auto-tool-choice`).
- **`required`** — force a tool call (vLLM ≥ 0.8.3).
- **`none`** — never call a tool.
- **named** — force one specific function.

### Choosing the parser (the part that bites people)
The parser is **model-specific**, and Qwen has several [[008]](../sources/008-vllm-tool-calling.md)[[009]](../sources/009-vllm-recipes-qwen35-qwen36.md):

- **`qwen3_coder`** — the parser for Qwen3.6. The official recipe says explicitly: *"To enable
  tool calling, add `--enable-auto-tool-choice --tool-call-parser qwen3_coder`"* [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md). **This is
  your setting.** ✅
- **`qwen3_xml`** — an alternative Qwen3 parser (XML-style tool encoding); a fallback if you hit
  parsing edge cases with `qwen3_coder`.
- **`hermes`** — the older Hermes-style parser that worked for **Qwen2.5** [[008]](../sources/008-vllm-tool-calling.md). Using it for
  Qwen3.6 is the classic mistake — it can appear to work then mis-parse arguments. **Don't.**

### Reasoning vs tool output
Qwen3.6 is a *thinking* model: it emits chain-of-thought before its answer. Separate that from
the final/tool output with `--reasoning-parser qwen3` [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md). To turn thinking off from the
server side (instead of per request) [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md):

```bash
--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
```

Whether to leave thinking on is a quality-vs-latency call: reasoning improves hard tasks but adds
tokens (and TTFT/TPOT cost — guide `03`). For tool-routing-heavy agent loops, many operators
disable it for speed and re-enable per-request for hard problems.

---

## Long context in depth

### Where the memory goes
KV-cache size scales with **context length × layers × KV-heads × bytes-per-element × concurrent
sequences**. Two things shrink it:

- **FP8 KV cache** (`--kv-cache-dtype fp8`) ≈ halves bytes-per-element vs BF16 (guide `02`) [[002]](../sources/002-vllm-quantized-kv-cache.md).
- **MoE active-param efficiency** leaves more of your 128 GB free for KV in the first place.

Concretely, the near-twin GB10 build (Qwen3.5-35B-A3B) reports **~28.6 GB free for KV after
loading ~70 GB of BF16 weights, supporting ~374K tokens** of cache [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md). With **NVFP4 weights**
(~20 GB instead of ~70 GB — guide `00`) you free up *even more* for KV. The point: on this box,
**32k context is conservative** — you have substantial headroom.

### Going beyond native context (YaRN)
Qwen3.5/3.6 natively support very long contexts (the recipe shows `--max-model-len 262144` for
the larger model) [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md). To exceed the native window, use **RoPE scaling (YaRN)** [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md):

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
vllm serve ... \
  --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn",
    "factor": 4.0, "original_max_position_embeddings": 262144, ...}}}' \
  --max-model-len 1010000
```

YaRN trades a little short-context quality for a much larger window — enable it only when you
actually need the length.

### The context ↔ batching trade
A larger `--max-model-len` makes vLLM reserve KV for longer sequences, which **reduces how many
sequences fit at once** → less batching → lower aggregate throughput. So don't set context higher
than your workload needs. Mitigations:

- **Prefix caching** for shared long system prompts/tool schemas (reuse, don't re-store) [[001]](../sources/001-vllm-optimization-tuning.md).
- **NVFP4 KV cache** (emerging) halves KV again vs FP8 → ~2× the context or batch at equal memory
  [[019]](../sources/019-nvidia-nvfp4-kv-cache-long-context.md). Watch for it landing in your stack.

---

## On *your* GX10

Your tool-calling config matches the official recipe exactly:

```
--enable-auto-tool-choice --tool-call-parser qwen3_coder
# (pair with --reasoning-parser qwen3 if not already set)
```

This is the validated path for Qwen3.6-35B-A3B [[009]](../sources/008-vllm-tool-calling.md). And you have a **live, real-world test
harness for it**: the OpenClaw agent gateway uses this very endpoint for tool-calling. That's a
gift — agent tool-call reliability *is* your acceptance test.

Context: you run **32k** with FP8 KV. Given the headroom above, the questions for Phase 3:

1. **Tool-call reliability sweep** — run your OpenClaw agent's real tool-using tasks and measure
   the parse/exec success rate with `qwen3_coder`; spot-check `qwen3_xml` if any calls mis-parse.
2. **Reasoning on vs off** — measure the latency cost (guide `03`) and quality delta of
   `enable_thinking` for your agent loops.
3. **Context headroom** — measure how many concurrent 32k sessions fit before preemption, and
   whether bumping `--max-model-len` (e.g. 64k) is "free" given your KV headroom or starts costing
   batch width.
4. **Prefix-caching gain** on the long shared agent system prompt + tool schema (also in guide `01`).

### Decision table

| You want… | Do this | Source |
|---|---|---|
| Tool calls on Qwen3.6 | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) |
| Tool calls mis-parsing | try `--tool-call-parser qwen3_xml`; **never** `hermes` on Qwen3.6 | [[008]](../sources/008-vllm-tool-calling.md) |
| Faster agent routing | `enable_thinking: false` (re-enable per-request for hard tasks) | [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) |
| Longer context, same memory | FP8 KV now; NVFP4 KV when available | [[002]](../sources/002-vllm-quantized-kv-cache.md)[[019]](../sources/019-nvidia-nvfp4-kv-cache-long-context.md) |
| Context beyond native window | YaRN via `--hf-overrides` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) |

---

## Where to go next

- **Guide `05` — Benchmarking:** turn every "measure this" above into a repeatable run.

## Sources cited

- [[001]](../sources/001-vllm-optimization-tuning.md) Optimization and Tuning (vLLM)
- [[002]](../sources/002-vllm-quantized-kv-cache.md) Quantized KV Cache (vLLM)
- [[008]](../sources/008-vllm-tool-calling.md) Tool Calling (vLLM)
- [[009]](../sources/009-vllm-recipes-qwen35-qwen36.md) Qwen3.5/3.6 usage guide (vLLM Recipes)
- [[014]](../sources/014-community-adadrag-qwen35-dgx-spark.md) Qwen3.5-35B-A3B on DGX Spark (community)
- [[019]](../sources/019-nvidia-nvfp4-kv-cache-long-context.md) NVFP4 KV cache for long context (NVIDIA)
