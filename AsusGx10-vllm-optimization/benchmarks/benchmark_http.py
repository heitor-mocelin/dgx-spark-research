#!/usr/bin/env python3
"""Dependency-free streaming benchmark for an OpenAI-compatible vLLM endpoint.
Measures TTFT, inter-token latency (ITL), and throughput in single-stream and concurrent
regimes — no docker/vllm CLI needed (hits the HTTP API directly). Forces fixed output length
with ignore_eos for comparable runs."""
import json, time, sys, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://172.27.27.210:8000/v1/chat/completions"
MODEL = "qwen36-moe"
PROMPT = ("Explain, in detail, how memory bandwidth limits autoregressive decoding throughput "
          "on a unified-memory system, and how batching and quantization change that picture. "
          "Be thorough and technical.")
OUT_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 256

def one_request():
    body = json.dumps({
        "model": MODEL, "temperature": 0.0, "max_tokens": OUT_TOKENS,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; ptoks = ctoks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if obj.get("usage"):
                ptoks = obj["usage"].get("prompt_tokens", 0)
                ctoks = obj["usage"].get("completion_tokens", 0)
    total = time.perf_counter() - t0
    return {"ttft": ttft or total, "total": total, "ctoks": ctoks or OUT_TOKENS, "ptoks": ptoks}

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

def single_stream(n=5):
    print(f"\n=== single-stream (n={n}, out={OUT_TOKENS} tok, ignore_eos) ===")
    rs = [one_request() for _ in range(n)]
    ttfts = [r["ttft"] for r in rs]
    itls = [(r["total"] - r["ttft"]) / max(1, r["ctoks"] - 1) * 1000 for r in rs]
    tputs = [r["ctoks"] / r["total"] for r in rs]
    print(f"  TTFT   p50={pct(ttfts,50)*1000:7.1f} ms   p99={pct(ttfts,99)*1000:7.1f} ms")
    print(f"  ITL    p50={pct(itls,50):7.1f} ms   p99={pct(itls,99):7.1f} ms   (= {1000/statistics.mean(itls):.1f} tok/s/stream)")
    print(f"  decode tok/s/stream  mean={statistics.mean(tputs):6.1f}")
    return {"ttft_p50_ms": pct(ttfts,50)*1000, "itl_p50_ms": pct(itls,50), "tok_s_stream": statistics.mean(tputs)}

def concurrent(c=32, n=64):
    print(f"\n=== concurrent (concurrency={c}, requests={n}, out={OUT_TOKENS} tok) ===")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        rs = list(ex.map(lambda _: one_request(), range(n)))
    wall = time.perf_counter() - t0
    toks = sum(r["ctoks"] for r in rs)
    ttfts = [r["ttft"] for r in rs]
    print(f"  aggregate output throughput = {toks/wall:7.1f} tok/s   ({n} reqs in {wall:.1f}s)")
    print(f"  TTFT under load  p50={pct(ttfts,50)*1000:7.1f} ms   p99={pct(ttfts,99)*1000:7.1f} ms")
    return {"agg_tok_s": toks/wall, "ttft_p50_ms": pct(ttfts,50)*1000, "ttft_p99_ms": pct(ttfts,99)*1000,
            "concurrency": c, "requests": n, "wall_s": wall}

if __name__ == "__main__":
    res = {"single_stream": single_stream(5), "concurrent": concurrent(32, 64)}
    json.dump(res, open("/root/inference-research/baseline_bench.json", "w"), indent=1)
    print("\nsaved -> /root/inference-research/baseline_bench.json")
