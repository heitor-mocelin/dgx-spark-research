# sources/

Cited corpus for Gemma 4 local inference. Every non-trivial guide claim traces to a source
here (or a measurement). Files are `gNN-slug.md`, each fetched on-device (OpenClaw LXC:
trafilatura for HTML, curl-raw for GitHub/HF raw) with a YAML provenance header:

```yaml
---
id: gNN
title: "<title>"
url: "<canonical URL>"
publisher: "<site / org>"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/<trafilatura|curl-raw>"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [..]
---
```

Gemma 4 weights are **gated** (accept the license on Hugging Face / Kaggle) — these files are
documentation/benchmark text, not weights. No paywalled full text is redistributed; sources are
cited by URL + retrieval date with short attributed excerpts.
