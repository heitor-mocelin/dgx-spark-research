#!/usr/bin/env python3
"""Concurrency sweep against the live vLLM endpoint (no restart needed): maps the
throughput-vs-latency curve and finds the saturation knee. Reuses the streaming client."""
import json, time, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://172.27.27.210:8000/v1/chat/completions"
MODEL = "qwen36-moe"
PROMPT = ("Explain in detail how memory bandwidth limits autoregressive decoding throughput on a "
          "unified-memory system, and how batching and quantization change that picture.")
OUT = 192

def req(_):
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": OUT, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": PROMPT}]}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; ct = 0
    with urllib.request.urlopen(r, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "): continue
            d = line[6:]
            if d == "[DONE]": break
            try: o = json.loads(d)
            except: continue
            ch = o.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content") and ttft is None:
                ttft = time.perf_counter() - t0
            if o.get("usage"): ct = o["usage"].get("completion_tokens", 0)
    tot = time.perf_counter() - t0
    return {"ttft": ttft or tot, "tot": tot, "ct": ct or OUT}

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs)-1, int(round(p/100*(len(xs)-1))))]

print(f"{'conc':>5} {'reqs':>5} {'agg tok/s':>10} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8}")
rows = []
for c in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]:
    n = max(8, min(2*c, 96))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        rs = list(ex.map(req, range(n)))
    wall = time.perf_counter() - t0
    toks = sum(r["ct"] for r in rs)
    ttfts = [r["ttft"]*1000 for r in rs]
    itls = [(r["tot"]-r["ttft"])/max(1, r["ct"]-1)*1000 for r in rs]
    row = {"conc": c, "reqs": n, "agg_tok_s": round(toks/wall, 1),
           "ttft_p50_ms": round(pct(ttfts,50),1), "ttft_p99_ms": round(pct(ttfts,99),1),
           "itl_p50_ms": round(pct(itls,50),1)}
    rows.append(row)
    print(f"{c:>5} {n:>5} {row['agg_tok_s']:>10} {row['ttft_p50_ms']:>9} {row['ttft_p99_ms']:>9} {row['itl_p50_ms']:>8}")
    time.sleep(2)
json.dump(rows, open("/root/inference-research/concurrency_sweep.json", "w"), indent=1)
print("\nsaved -> concurrency_sweep.json")
