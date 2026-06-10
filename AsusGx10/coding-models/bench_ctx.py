#!/usr/bin/env python3
"""Context-length sweep: TTFT (prefill) + decode tok/s at increasing input sizes.
Builds a code-ish prompt of ~N tokens, asks a tiny question, streams the reply.
Records the server's actual prompt_tokens so the x-axis is real, not estimated."""
import argparse, json, time, requests
FILLER = "def helper_{i}(x):\n    return x * 2  # padding line {i} for context-length benchmarking\n"

def make_prompt(approx_tokens):
    chars = approx_tokens * 4  # ~4 chars/token
    buf = []
    n = 0
    i = 0
    while n < chars:
        s = FILLER.format(i=i); buf.append(s); n += len(s); i += 1
    return "Here is some Python code:\n\n" + "".join(buf) + "\n\nIn one sentence, what do these functions do?"

def stream_once(base, model, prompt, max_tokens):
    t0 = time.perf_counter()
    r = requests.post(base.rstrip("/") + "/v1/chat/completions",
                      json={"model": model, "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                            "stream_options": {"include_usage": True}}, stream=True, timeout=600)
    r.raise_for_status()
    ttft = ct = pt = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        o = json.loads(d)
        ch = o.get("choices") or []
        if ch and ch[0].get("delta", {}).get("content") and ttft is None:
            ttft = time.perf_counter() - t0
        if o.get("usage"):
            ct = o["usage"].get("completion_tokens"); pt = o["usage"].get("prompt_tokens")
    total = time.perf_counter() - t0
    return ttft, total, ct, pt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--model", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--ctx", default="1024,8192,32768")
    ap.add_argument("--max-tokens", type=int, default=64); ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"label": a.label, "model": a.model, "ts": int(time.time()), "context_sweep": {}}
    for c in [int(x) for x in a.ctx.split(",") if x]:
        try:
            ttft, total, ct, pt = stream_once(a.base, a.model, make_prompt(c), a.max_tokens)
            gen = max(1e-6, total - (ttft or 0))
            res["context_sweep"][str(c)] = {"prompt_tokens": pt, "ttft_ms": round((ttft or 0) * 1000, 1),
                                            "decode_tps": round((ct or 0) / gen, 1), "completion_tokens": ct}
            print(f"  ctx~{c}: prompt_tokens={pt} ttft={round((ttft or 0)*1000)}ms decode={round((ct or 0)/gen,1)}tps", flush=True)
        except Exception as e:
            res["context_sweep"][str(c)] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  ctx~{c}: ERROR {e}", flush=True)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
