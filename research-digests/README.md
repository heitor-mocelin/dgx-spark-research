# research-digests/

Auto-generated, **local-first** literature digests — produced entirely on-device by the
homelab's local model (Qwen3.6-35B-A3B via vLLM on the GX10, orchestrated by OpenClaw). No
cloud model and no human-in-the-loop in the generation path.

## Digests

| File | Topic | Papers | Generated |
|------|-------|--------|-----------|
| `inference-major-discoveries.md` | Major discoveries in efficient LLM inference | ~48 (arXiv, 8 subtopics) | 2026-06-05 |

## How they're generated

A deterministic **map-reduce**, all reasoning on the local model
([`generator/build_digest.py`](generator/build_digest.py)):

1. **Scrape** — query the arXiv Atom API per subtopic (phrase-matched, relevance-sorted).
2. **Distill** — each paper → bullet contributions (one local call).
3. **Synthesize** — one local call per subtopic.
4. **Reduce** — one local call for the executive summary.
5. **Assemble** — the Markdown document + `papers.json` / `distilled.json` provenance sidecars.

**Thinking must be disabled.** Qwen3.6 is a reasoning model; left on, its chain-of-thought
leaks into the output ("The user wants me to…"). The generator calls the vLLM OpenAI endpoint
directly with `chat_template_kwargs: {enable_thinking: false}`, which suppresses it cleanly —
more reliable here than routing through the agent wrapper, and still 100% on-device.

### Reproduce

```bash
# anywhere that can reach the GX10 vLLM endpoint (defaults to http://172.27.27.210:8000)
VLLM_URL=http://172.27.27.210:8000/v1/chat/completions MODEL=qwen36-moe \
  OUT_DIR=. python3 generator/build_digest.py     # writes inference-major-discoveries.md
```

## Caveats (read before citing)

- **Automated selection, not curation.** arXiv relevance ranking is imperfect — an occasional
  off-topic paper slips through. Treat each digest as a *literature map*, not a peer-reviewed
  survey, and not citation-ranked toward seminal works.
- **Abstracts only.** Distillation is from abstracts, not full text.
- **Model-generated.** Bullets and syntheses are produced by the local model; verify specific
  claims/numbers against the linked arXiv sources before relying on them.
