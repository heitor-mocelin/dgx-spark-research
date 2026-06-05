# research-digests/

Auto-generated, **local-first** literature digests — produced entirely on-device by the
homelab's local model (Qwen3.6-35B-A3B via vLLM on the GX10, orchestrated by OpenClaw). No
cloud model and no human-in-the-loop in the generation path.

## Digests

| File | Topic | Papers | Generated |
|------|-------|--------|-----------|
| `inference-major-discoveries.md` | Major discoveries in efficient LLM inference | ~48 (arXiv, 8 subtopics) | 2026-06-05 |

## How they're generated

A deterministic **map-reduce** (see [`generator/`](generator/)):

1. **Scrape** — [`arxiv_fetch.py`](generator/arxiv_fetch.py) queries the arXiv Atom API per
   subtopic (phrase-matched, relevance-sorted) and parses titles + abstracts.
2. **Distill** — [`run.py`](generator/run.py) sends each paper to the local model in a single
   short call → bullet contributions.
3. **Synthesize** — one local call per subtopic.
4. **Reduce** — one local call for the executive summary.
5. **Assemble** — the Markdown document + `papers.json` / `distilled.json` provenance sidecars.

Every model turn is short and uses a fresh session key by design — this keeps each call under
the local server's context budget and sidesteps a known CLI-compaction failure mode on long,
many-tool-call agent turns.

### Reproduce

```bash
# on the OpenClaw LXC (needs the vLLM server up and `openclaw` configured)
mkdir -p ~/inference-research && cp generator/*.py ~/inference-research/
cd ~/inference-research && python3 run.py        # writes inference-major-discoveries.md
```

## Caveats (read before citing)

- **Automated selection, not curation.** arXiv relevance ranking is imperfect — an occasional
  off-topic paper slips through. Treat each digest as a *literature map*, not a peer-reviewed
  survey, and not citation-ranked toward seminal works.
- **Abstracts only.** Distillation is from abstracts, not full text.
- **Model-generated.** Bullets and syntheses are produced by the local model; verify specific
  claims/numbers against the linked arXiv sources before relying on them.
