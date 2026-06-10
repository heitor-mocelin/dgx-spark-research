#!/usr/bin/env python3
"""Reflex-model benchmark probe. Hits an OpenAI-compatible vLLM endpoint and measures:
   - correctness smoke (non-null coherent output)
   - TTFT (time-to-first-token), streaming, N runs averaged
   - single-stream decode tok/s, N runs averaged
   - concurrency sweep aggregate tok/s (c=1,4,16,32)
Outputs a JSON blob to stdout (and --out file). Designed to run on the DGX (localhost) to keep TTFT clean.
"""
import argparse, json, time, sys, statistics, threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DECODE_PROMPT = "Write a detailed, well-structured explanation of how the TCP three-way handshake works, step by step."
TTFT_PROMPT   = "In one sentence, what is a firewall?"

def _post(base, model, prompt, max_tokens, stream, extra_body, timeout=300):
    url = base.rstrip("/") + "/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if extra_body:
        body.update(extra_body)
    return requests.post(url, json=body, stream=stream, timeout=timeout)

def stream_run(base, model, prompt, max_tokens, extra_body):
    """Return (ttft_s, total_s, completion_tokens, text)."""
    t0 = time.perf_counter()
    r = _post(base, model, prompt, max_tokens, True, extra_body)
    r.raise_for_status()
    ttft = None; text = []; ctoks = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        ch = obj.get("choices") or []
        if ch:
            delta = ch[0].get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(delta)
        if obj.get("usage"):
            ctoks = obj["usage"].get("completion_tokens")
    total = time.perf_counter() - t0
    txt = "".join(text)
    if ctoks is None:
        ctoks = max(1, len(txt) // 4)  # rough fallback
    return ttft if ttft is not None else total, total, ctoks, txt

def blocking_run(base, model, prompt, max_tokens, extra_body):
    """Non-stream; return (total_s, completion_tokens)."""
    t0 = time.perf_counter()
    r = _post(base, model, prompt, max_tokens, False, extra_body)
    r.raise_for_status()
    obj = r.json()
    total = time.perf_counter() - t0
    ctoks = (obj.get("usage") or {}).get("completion_tokens", 0)
    return total, ctoks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="1,4,16,32")
    ap.add_argument("--extra-body", default="")  # JSON, e.g. '{"chat_template_kwargs":{"enable_thinking":false}}'
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    extra = json.loads(a.extra_body) if a.extra_body else {}
    res = {"label": a.label, "model": a.model, "base": a.base, "ts": int(time.time()),
           "runs": a.runs, "max_tokens": a.max_tokens, "extra_body": extra}

    # 1) correctness smoke
    try:
        ttft, total, ctoks, txt = stream_run(a.base, a.model, "Reply with exactly: OK", 16, extra)
        res["smoke"] = {"ok": bool(txt.strip()), "text": txt.strip()[:200], "ctoks": ctoks}
    except Exception as e:
        res["smoke"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(res, indent=2)); _save(a, res); sys.exit(0)
    if not res["smoke"]["ok"]:
        print(json.dumps(res, indent=2)); _save(a, res); sys.exit(0)

    # 2) TTFT (N runs)
    ttfts = []
    for _ in range(a.runs):
        try:
            t, _, _, _ = stream_run(a.base, a.model, TTFT_PROMPT, 32, extra); ttfts.append(t*1000)
        except Exception as e:
            res.setdefault("errors", []).append(f"ttft: {e}")
    if ttfts:
        res["ttft_ms"] = {"mean": round(statistics.mean(ttfts),1),
                          "stdev": round(statistics.pstdev(ttfts),1) if len(ttfts)>1 else 0.0,
                          "runs": [round(x,1) for x in ttfts]}

    # 3) single-stream decode tok/s (N runs): tok/s = ctoks / (total - ttft)
    decs = []
    for _ in range(a.runs):
        try:
            t_ttft, total, ctoks, _ = stream_run(a.base, a.model, DECODE_PROMPT, a.max_tokens, extra)
            gen = max(1e-6, total - t_ttft)
            decs.append(ctoks/gen)
        except Exception as e:
            res.setdefault("errors", []).append(f"decode: {e}")
    if decs:
        res["decode_tps"] = {"mean": round(statistics.mean(decs),1),
                             "stdev": round(statistics.pstdev(decs),1) if len(decs)>1 else 0.0,
                             "runs": [round(x,1) for x in decs]}

    # 4) concurrency sweep aggregate tok/s
    sweep = {}
    for c in [int(x) for x in a.concurrency.split(",") if x]:
        try:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                futs = [ex.submit(blocking_run, a.base, a.model, DECODE_PROMPT, a.max_tokens, extra) for _ in range(c)]
                tot_tok = 0
                for f in as_completed(futs):
                    _, ct = f.result(); tot_tok += ct
            wall = time.perf_counter() - t0
            sweep[str(c)] = {"agg_tps": round(tot_tok/max(1e-6,wall),1), "tot_tokens": tot_tok, "wall_s": round(wall,2)}
        except Exception as e:
            sweep[str(c)] = {"error": f"{type(e).__name__}: {e}"}
    res["concurrency"] = sweep

    print(json.dumps(res, indent=2)); _save(a, res)

def _save(a, res):
    if a.out:
        with open(a.out, "w") as f: json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
