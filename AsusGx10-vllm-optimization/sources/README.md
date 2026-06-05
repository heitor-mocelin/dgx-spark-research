# sources/

The cited research corpus backing the guides. Every non-trivial claim in a guide should
trace to a source here (or to a measurement on the hardware).

## Layout

Each source is saved as a Markdown file (HTML fetched and converted to Markdown), named:

```
NNN-short-slug.md
```

where `NNN` is a zero-padded ordinal in retrieval order (e.g. `001-vllm-quickstart.md`).

## Provenance front matter

Each saved source begins with a YAML front-matter block recording where it came from and
when, so claims remain auditable:

```yaml
---
id: 001
title: "<page or document title>"
url: "<canonical source URL>"
publisher: "<site / org, e.g. vLLM docs, NVIDIA, GitHub>"
author: "<if known, else omit>"
retrieved: "2026-06-05"          # UTC date the page was fetched
fetched_by: "openclaw/mcp-server-fetch"
license_note: "<redistribution note if relevant, else 'reference only — cited by URL'>"
topics: [throughput, nvfp4, ...]  # which guide priorities it supports
---
```

## Notes

- Sources are fetched on-device by the OpenClaw agent (`mcp-server-fetch`) and synced here.
- Large pages are captured in bounded chunks; the `url` always points to the canonical
  original so readers can verify in full.
- No paywalled or license-restricted full text is redistributed — those are cited by URL
  and retrieval date with only short, attributed excerpts.
