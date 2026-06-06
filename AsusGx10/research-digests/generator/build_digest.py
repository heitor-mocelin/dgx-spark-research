#!/usr/bin/env python3
"""
Build a local-first literature digest end-to-end:
  scrape arXiv -> distill each paper -> synthesize each subtopic -> reduce -> assemble.

All reasoning runs on the local vLLM endpoint (GX10) with Qwen3.6 thinking DISABLED via
chat_template_kwargs (a reasoning model otherwise leaks chain-of-thought into the output).
No cloud, no human-in-the-loop. Outputs inference-major-discoveries.md + JSON provenance.

Env: VLLM_URL (default http://172.27.27.210:8000/v1/chat/completions), MODEL (default qwen36-moe).
"""
import json, re, os, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from datetime import datetime, timezone

OUT_DIR = os.environ.get("OUT_DIR", ".")
ENDPOINT = os.environ.get("VLLM_URL", "http://172.27.27.210:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "qwen36-moe")
DOC = os.path.join(OUT_DIR, "inference-major-discoveries.md")
NS = {'a': 'http://www.w3.org/2005/Atom'}

TOPICS = [
    ("Quantization for LLM inference", ['"post-training quantization" language model', '"low-bit quantization" llm inference']),
    ("KV cache optimization", ['"KV cache" compression', '"KV cache" quantization']),
    ("Speculative decoding", ['"speculative decoding"', '"speculative sampling" language model']),
    ("Efficient attention & kernels", ['"FlashAttention"', '"efficient attention" inference transformer']),
    ("Serving systems & batching", ['"LLM serving" throughput', '"continuous batching" language model']),
    ("Mixture-of-Experts inference", ['"mixture of experts" inference', '"expert parallelism" inference']),
    ("Long-context inference", ['"long-context" inference efficient', '"long context" KV cache']),
    ("Pruning, sparsity & distillation", ['"model pruning" large language model', '"knowledge distillation" efficient inference']),
]
PER_QUERY, PER_TOPIC = 6, 6

def arxiv(query, n):
    params = urllib.parse.urlencode({'search_query': f'all:{query}', 'start': 0,
        'max_results': n, 'sortBy': 'relevance', 'sortOrder': 'descending'})
    req = urllib.request.Request(f'http://export.arxiv.org/api/query?{params}',
                                 headers={'User-Agent': 'inference-research/0.2'})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=40).read())
    out = []
    for e in root.findall('a:entry', NS):
        t, s, i = e.find('a:title', NS), e.find('a:summary', NS), e.find('a:id', NS)
        if None in (t, s, i):
            continue
        m = re.search(r'(\d{4}\.\d{4,5})', i.text)
        p = e.find('a:published', NS)
        out.append({'aid': m.group(1) if m else i.text.strip(),
            'title': ' '.join(t.text.split()), 'abstract': ' '.join(s.text.split()),
            'published': (p.text[:10] if p is not None else ''),
            'authors': [a.find('a:name', NS).text for a in e.findall('a:author', NS)][:6]})
    return out

def chat(system, user, max_tokens=350, temp=0.2):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            txt = re.sub(r'(?s)<think>.*?</think>', '', d["choices"][0]["message"]["content"]).strip()
            if len(txt) > 20:
                return txt
        except Exception as e:
            print(f"  api err ({attempt}): {e}", flush=True); time.sleep(2)
    return ""

DSYS = ("You output ONLY markdown bullet points naming a paper's concrete contributions/discoveries "
        "(methods, mechanisms, measured results, numbers). No preamble, no reasoning, no restating the task.")
SSYS = "You write concise, technical research syntheses. Output only the requested prose and bullets — no preamble."

def main():
    topics, seen = [], set()
    for name, queries in TOPICS:
        papers = []
        for q in queries:
            try:
                for p in arxiv(q, PER_QUERY):
                    if p["aid"] in seen:
                        continue
                    seen.add(p["aid"]); papers.append(p)
                    if len(papers) >= PER_TOPIC:
                        break
            except Exception as e:
                print(f"  fetch err '{q}': {e}", flush=True)
            time.sleep(3)
            if len(papers) >= PER_TOPIC:
                break
        print(f"topic '{name}': {len(papers)} papers", flush=True)
        topics.append({"name": name, "papers": papers})
    json.dump(topics, open(os.path.join(OUT_DIR, "papers.json"), "w"), ensure_ascii=False, indent=1)

    for t in topics:
        for p in t["papers"]:
            print(f"distill [{t['name']}] {p['title'][:55]}", flush=True)
            b = chat(DSYS, f"Title: {p['title']}\nAbstract: {p['abstract'][:1600]}", 350)
            # normalize bullet markers, drop any stray meta lines
            lines = []
            for ln in (b or f"- {p['abstract'][:240]}…").splitlines():
                ln = re.sub(r'^[\*\-•]\s+', '', ln.strip())
                if ln and not re.match(r'(?i)^(the user|let me|i need|i should|here)', ln):
                    lines.append(f"- {ln}")
            p["bullets"] = "\n".join(lines) or f"- {p['abstract'][:240]}…"
        body = "\n\n".join(f"{p['title']}\n{p['bullets']}" for p in t["papers"])
        print(f"synthesize: {t['name']}", flush=True)
        t["synthesis"] = chat(SSYS, f"Distilled contributions from papers on '{t['name']}' (efficient LLM "
            f"inference):\n\n{body[:7000]}\n\nWrite 1 short paragraph on the major discoveries/themes, then a "
            "'- ' list of the 3-5 most important specific advances. Be specific; do not invent citations.", 550) or "_(unavailable)_"
        json.dump(topics, open(os.path.join(OUT_DIR, "distilled.json"), "w"), ensure_ascii=False, indent=1)

    body = "\n\n".join(f"## {t['name']}\n{t['synthesis']}" for t in topics)
    print("reduce: executive summary", flush=True)
    summary = chat(SSYS, "Per-topic syntheses of major discoveries in efficient LLM inference:\n\n"
        f"{body[:18000]}\n\nWrite a tight executive summary (3-4 paragraphs) of the biggest CROSS-CUTTING "
        "discoveries spanning all areas, and where the field is heading. Technical and specific.", 850) or "_(unavailable)_"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = sum(len(t["papers"]) for t in topics)
    out = ["# Major Discoveries in Efficient LLM Inference\n",
      f"> Auto-generated **{now}** by the local **Qwen3.6-35B-A3B** model on the GX10 (vLLM endpoint, "
      f"thinking disabled), from **{n} arXiv papers** across **{len(topics)} subtopics**. Distillation and "
      "synthesis are model-generated from abstracts — verify against the linked sources before citing.\n",
      "## Executive summary\n\n" + summary + "\n"]
    for t in topics:
        out.append(f"\n## {t['name']}\n\n{t['synthesis']}\n\n### Papers\n")
        for p in t["papers"]:
            auth = ", ".join(p.get("authors", [])[:3]) + (" et al." if len(p.get("authors", [])) > 3 else "")
            out.append(f"- **{p['title']}** ({p.get('published','')[:4]}) — {auth}. "
                       f"[arXiv:{p['aid']}](https://arxiv.org/abs/{p['aid']})")
            for ln in p["bullets"].splitlines():
                out.append(f"  {ln}")
        out.append("")
    out += ["\n## Method & caveats\n",
      "- **Source:** arXiv API, phrase-matched per subtopic (relevance-sorted). Not citation-ranked; "
      "abstracts only; automated selection, not curation.",
      "- **Reasoning:** fully local — the GX10 vLLM model with thinking disabled, no cloud, no human-in-the-loop. "
      "A literature map, not a peer-reviewed survey.",
      "- **Provenance:** `papers.json` and `distilled.json` sit beside this file."]
    open(DOC, "w").write("\n".join(out) + "\n")
    print(f"WROTE {DOC}: {n} papers", flush=True)

if __name__ == "__main__":
    main()
