#!/usr/bin/env python3
"""
Autonomous, local-first research digest builder (runs unattended on the OpenClaw LXC).

Map-reduce over arXiv, all reasoning done by the local Qwen3.6 model via `openclaw agent`:
  1. scrape arXiv per subtopic (deterministic)         -> papers.json
  2. distill each paper to bullet contributions (1 short local call each)
  3. synthesize each subtopic (1 short local call each)
  4. reduce to an executive summary (1 short local call)
  5. assemble inference-major-discoveries.md

Short, fresh-session-key calls keep every model turn under the context budget (avoids the
known CLI-compaction failure). Incremental state is saved so a crash is recoverable.
"""
import json, subprocess, sys, time, re, os
from datetime import datetime, timezone
from arxiv_fetch import fetch

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "inference-major-discoveries.md")
PAPERS_JSON = os.path.join(HERE, "papers.json")
DISTILLED_JSON = os.path.join(HERE, "distilled.json")

TOPICS = [
    ("Quantization for LLM inference",
     ['"post-training quantization" language model', '"low-bit quantization" llm inference']),
    ("KV cache optimization",
     ['"KV cache" compression', '"KV cache" quantization']),
    ("Speculative decoding",
     ['"speculative decoding"', '"speculative sampling" language model']),
    ("Efficient attention & kernels",
     ['"FlashAttention"', '"efficient attention" inference transformer']),
    ("Serving systems & batching",
     ['"LLM serving" throughput', '"continuous batching" language model']),
    ("Mixture-of-Experts inference",
     ['"mixture of experts" inference', '"expert parallelism" inference']),
    ("Long-context inference",
     ['"long-context" inference efficient', '"long context" KV cache']),
    ("Pruning, sparsity & distillation",
     ['"model pruning" large language model', '"knowledge distillation" efficient inference']),
]
PER_QUERY = 6
PER_TOPIC = 6

LOGPFX = re.compile(r'^\s*\[(agents?|ws|gateway|engine)[/\]]')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def clean(out):
    lines = [l for l in out.splitlines() if not LOGPFX.match(l) and not l.startswith('[')]
    return "\n".join(lines).strip()

def agent(prompt, key, timeout=200):
    """One short local-model turn; strips log lines; retries once with a fresh key."""
    for attempt in (1, 2):
        try:
            r = subprocess.run(
                ["openclaw", "agent", "--local", "--agent", "main",
                 "--session-key", f"{key}-{attempt}", "--thinking", "off", "-m", prompt],
                capture_output=True, text=True, timeout=timeout)
            txt = clean(r.stdout)
            if r.returncode == 0 and len(txt) > 20:
                return txt
            log(f"  agent ret--rc={r.returncode} len={len(txt)} key={key} attempt={attempt}")
        except subprocess.TimeoutExpired:
            log(f"  agent TIMEOUT key={key} attempt={attempt}")
        time.sleep(2)
    return ""

def arxiv_id(raw):
    m = re.search(r'(\d{4}\.\d{4,5})', raw)
    return m.group(1) if m else raw.strip()

# ---- 1. scrape -------------------------------------------------------------------------
def gather():
    topics = []
    seen = set()
    for name, queries in TOPICS:
        papers = []
        for q in queries:
            try:
                for p in fetch(q, PER_QUERY):
                    aid = arxiv_id(p["id"])
                    if aid in seen:
                        continue
                    seen.add(aid)
                    p["aid"] = aid
                    papers.append(p)
                    if len(papers) >= PER_TOPIC:
                        break
            except Exception as e:
                log(f"  fetch error '{q}': {e}")
            time.sleep(3)  # arXiv rate-limit courtesy
            if len(papers) >= PER_TOPIC:
                break
        log(f"topic '{name}': {len(papers)} papers")
        topics.append({"name": name, "papers": papers})
    json.dump(topics, open(PAPERS_JSON, "w"), ensure_ascii=False, indent=1)
    return topics

# ---- 2-3. distill + synthesize ---------------------------------------------------------
DISTILL = ("From the title and abstract, output ONLY 2-4 markdown bullets ('- ') naming the "
           "paper's concrete contributions/discoveries (methods, mechanisms, measured results, "
           "numbers). No preamble, do not restate the title.\n\nTitle: {t}\nAbstract: {a}")
SYNTH = ("Below are distilled contributions from several papers on '{topic}' (efficient LLM "
         "inference). Write a concise synthesis: 1 short paragraph on the MAJOR discoveries and "
         "themes, then a '- ' list of the 3-5 most important specific advances. Technical and "
         "specific; do not invent citations.\n\n{body}")
REDUCE = ("Below are per-topic syntheses of major discoveries in efficient LLM inference. Write a "
          "tight executive summary (2-3 paragraphs) of the biggest cross-cutting discoveries and "
          "where the field is heading. Technical, specific, no fluff.\n\n{body}")

def process(topics):
    for ti, topic in enumerate(topics):
        for pi, p in enumerate(topic["papers"]):
            log(f"distill [{topic['name']}] {pi+1}/{len(topic['papers'])}: {p['title'][:60]}")
            b = agent(DISTILL.format(t=p["title"], a=p["abstract"][:1600]), f"infd-{ti}-{pi}")
            p["bullets"] = b if b else f"- {p['abstract'][:240]}…"
            json.dump(topics, open(DISTILLED_JSON, "w"), ensure_ascii=False, indent=1)
        body = "\n\n".join(f"{p['title']}\n{p['bullets']}" for p in topic["papers"])
        log(f"synthesize topic: {topic['name']}")
        topic["synthesis"] = agent(SYNTH.format(topic=topic["name"], body=body[:6000]), f"infs-{ti}") \
            or "_(synthesis unavailable)_"
        json.dump(topics, open(DISTILLED_JSON, "w"), ensure_ascii=False, indent=1)
    body = "\n\n".join(f"## {t['name']}\n{t['synthesis']}" for t in topics)
    log("reduce: executive summary")
    summary = agent(REDUCE.format(body=body[:8000]), "infx") or "_(executive summary unavailable)_"
    return topics, summary

# ---- 5. assemble -----------------------------------------------------------------------
def assemble(topics, summary):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = sum(len(t["papers"]) for t in topics)
    out = []
    out.append("# Major Discoveries in Efficient LLM Inference\n")
    out.append(f"> Auto-generated **{now}** by the local **Qwen3.6-35B-A3B** model on the GX10 "
               f"(OpenClaw LXC), from **{n} arXiv papers** across **{len(topics)} subtopics**. "
               "Distillation and synthesis are model-generated from abstracts — verify against "
               "the linked sources before citing.\n")
    out.append("## Executive summary\n\n" + summary + "\n")
    for t in topics:
        out.append(f"\n## {t['name']}\n\n{t['synthesis']}\n\n### Papers\n")
        for p in t["papers"]:
            auth = ", ".join(p.get("authors", [])[:3]) + (" et al." if len(p.get("authors", [])) > 3 else "")
            yr = p.get("published", "")[:4]
            out.append(f"- **{p['title']}** ({yr}) — {auth}. [arXiv:{p['aid']}](https://arxiv.org/abs/{p['aid']})")
            for line in p["bullets"].splitlines():
                line = line.strip()
                if line:
                    out.append(f"  {line if line.startswith('-') else '- ' + line}")
        out.append("")
    out.append("\n## Method & caveats\n")
    out.append("- **Source:** arXiv API, queried per subtopic with phrase matches (relevance-sorted). "
               "Not citation-ranked; abstracts only (not full text); selection is automated, not curated.")
    out.append("- **Reasoning:** every distillation/synthesis is one short local-model call — no cloud, "
               "no human-in-the-loop. Treat as a literature map, not a peer-reviewed survey.")
    out.append("- **Provenance:** `papers.json` (raw metadata) and `distilled.json` (per-paper bullets) "
               "sit beside this file.")
    open(DOC, "w").write("\n".join(out) + "\n")
    log(f"WROTE {DOC} ({n} papers)")

def main():
    log("=== inference research digest: START ===")
    topics = gather()
    topics, summary = process(topics)
    assemble(topics, summary)
    log("=== DONE ===")

if __name__ == "__main__":
    main()
