#!/usr/bin/env python3
"""Aggregate coding-model bench JSONs -> combined.json + FINDINGS-coding-models.md."""
import json, sys, os

RC = sys.argv[1] if len(sys.argv) > 1 else "/home/user/results-coding"
OUT_MD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RC, "FINDINGS-coding-models.md")

# key -> (display, params, role, quant)
META = {
    "qwen3-coder-30b":        ("Qwen3-Coder-30B-A3B-Instruct", "30B / 3B MoE", "flagship coding MoE", "NVFP4 (ig1)"),
    "qwen25-coder-32b":       ("Qwen2.5-Coder-32B-Instruct",   "32B dense",    "dense flagship coder", "FP8 (RedHatAI)"),
    "qwen25-coder-14b":       ("Qwen2.5-Coder-14B-Instruct",   "14B dense",    "mid dense coder",      "FP8 (RedHatAI)"),
    "qwen25-coder-7b":        ("Qwen2.5-Coder-7B-Instruct",    "7B dense",     "small dense coder",    "FP8 (RedHatAI)"),
    "deepseek-coder-v2-lite": ("DeepSeek-Coder-V2-Lite-Instruct","16B / 2.4B MoE","coding MoE",         "FP8 (RedHatAI)"),
    "devstral-24b":           ("Devstral-Small-2-24B",         "24B dense",    "agentic coder",        "NVFP4 (Firworks)"),
    "codestral-22b":          ("Codestral-22B-v0.1",           "22B dense",    "FIM / autocomplete",   "FP8 (TechxGenus)"),
}
ORDER = list(META.keys())

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default

rows = []
for k in ORDER:
    perf = load(f"{RC}/{k}.perf.json")
    he   = load(f"{RC}/{k}.he-score.json")
    ctx  = load(f"{RC}/{k}.ctx.json")
    disp, params, role, quant = META[k]
    r = {"key": k, "display": disp, "params": params, "role": role, "quant": quant, "status": "ok"}
    if perf and (perf.get("serve_failed") or perf.get("download_failed")):
        r["status"] = "serve_failed" if perf.get("serve_failed") else "download_failed"
    else:
        r["pass1"]   = g(he, "summary", "pass@1")
        r["passed"]  = g(he, "summary", "passed"); r["total"] = g(he, "summary", "total")
        r["decode"]  = g(perf, "decode_tps", "mean")
        r["ttft"]    = g(perf, "ttft_ms", "mean")
        r["c1"]      = g(perf, "concurrency", "1", "agg_tps")
        r["c32"]     = g(perf, "concurrency", "32", "agg_tps")
        r["ctx"]     = g(ctx, "context_sweep", default={})
    rows.append(r)

json.dump(rows, open(f"{RC}/combined.json", "w"), indent=2)

def cell(v, suf=""):
    return f"{v}{suf}" if v is not None else "—"

md = []
md.append("# Coding models on the DGX Spark (GB10 / NVFP4) — findings\n")
md.append("Measured on a single **ASUS GX10 / NVIDIA GB10** (DGX2, 128 GB unified), vLLM nightly, "
          "each model served standalone on `:8001` at util 0.85 with the proven sm_121 FP4 env. "
          "**Ready-made quants only** (NVFP4 where it exists, else reputable FP8). One model at a time (OOM-safe).\n")
md.append("Battery per model: HumanEval pass@1 (greedy, instruct-style) · single-stream decode tok/s + TTFT · "
          "concurrency sweep (c=1/4/16/32) · context-length sweep. See `coding-models/` for the harness.\n")
md.append("## Summary\n")
md.append("| Model | Params | Quant | HumanEval pass@1 | Decode tok/s | TTFT | Agg tok/s @c=32 |")
md.append("|---|---|---|---|---|---|---|")
for r in rows:
    if r["status"] != "ok":
        md.append(f"| {r['display']} | {r['params']} | {r['quant']} | _{r['status']}_ | — | — | — |")
        continue
    p1 = f"**{r['pass1']}%** ({r['passed']}/{r['total']})" if r.get("pass1") is not None else "—"
    md.append(f"| {r['display']} | {r['params']} | {r['quant']} | {p1} | "
              f"{cell(r.get('decode'))} | {cell(r.get('ttft'),' ms')} | {cell(r.get('c32'))} |")
md.append("\n## Context-length sweep (TTFT / decode tok/s by real prompt tokens)\n")
for r in rows:
    if r["status"] != "ok" or not r.get("ctx"):
        continue
    md.append(f"### {r['display']}\n")
    md.append("| Prompt tokens | TTFT | Decode tok/s |")
    md.append("|---|---|---|")
    for _, v in sorted(r["ctx"].items(), key=lambda kv: (kv[1].get("prompt_tokens") or int(kv[0]))):
        if "error" in v:
            continue
        md.append(f"| {cell(v.get('prompt_tokens'))} | {cell(v.get('ttft_ms'),' ms')} | {cell(v.get('decode_tps'))} |")
    md.append("")
open(OUT_MD, "w").write("\n".join(md) + "\n")
print(f"[aggregate] wrote {RC}/combined.json and {OUT_MD}")
print(f"[aggregate] {sum(1 for r in rows if r['status']=='ok')}/{len(rows)} models ok")
