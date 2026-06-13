#!/usr/bin/env python3
"""Cluster benchmark + telemetry capture for the 2x DGX Spark cluster.

Runs single-stream (N reps, with TTFT) and a duration-based concurrency sweep against a
vLLM OpenAI endpoint, and captures per-phase Prometheus telemetry on BOTH nodes:
GPU temp, memory(HBM/junction) temp, power, util, SM clock, board temp, and the RoCE
interconnect throughput (Gb/s TX/RX). Saves raw JSON for the article findings.

stdlib only. Run from a host that can reach the vLLM endpoint AND Prometheus (e.g. alpha).

Usage:
  python3 cluster_bench.py --endpoint http://172.27.27.210:8000 --model qwen36-moe \
    --label "qwen36-moe TP=2 cluster" --out /root/results/qwen36_tp2.json \
    [--sweep 1,4,16,32] [--reps 5] [--max-tokens 256] [--load-seconds 30] [--mode cluster]
"""
import json, time, sys, argparse, threading, urllib.request, urllib.parse

PROM = "http://172.27.27.212:9090"
CAGE = 'device=~"en(p1|P2p1)s0f[01]np[01]"'   # ConnectX-7 cage netdevs (excl mgmt enP7s7)
PROMPT = ("Write a thorough, well-structured technical explanation of how tensor parallelism "
          "and pipeline parallelism distribute a large transformer model across multiple GPUs, "
          "including the communication patterns and their bandwidth costs.")

def prom_q(expr, t=None):
    u = f"{PROM}/api/v1/query?query=" + urllib.parse.quote(expr)
    if t: u += f"&time={t:.0f}"
    try:
        r = json.load(urllib.request.urlopen(u, timeout=12))["data"]["result"]
        return float(r[0]["value"][1]) if r else None
    except Exception:
        return None

def _g(v): return round(v / 1e9, 3) if v is not None else None

def telemetry(t0, t1):
    """avg/max over [t0,t1] for temps/power/util + interconnect, per node."""
    d = max(int(t1 - t0), 8)
    w = f"{d}s"
    out = {"window_s": d}
    for inst, tag in (("gx10", "dgx1"), ("gx10b", "dgx2")):
        n = {}
        for key, m in (("gpu_temp_c", "DCGM_FI_DEV_GPU_TEMP"),
                       ("mem_temp_c", "DCGM_FI_DEV_MEMORY_TEMP"),
                       ("power_w", "DCGM_FI_DEV_POWER_USAGE"),
                       ("gpu_util_pct", "DCGM_FI_DEV_GPU_UTIL"),
                       ("sm_clock_mhz", "DCGM_FI_DEV_SM_CLOCK")):
            n[key + "_avg"] = prom_q(f'avg_over_time({m}{{instance="{inst}"}}[{w}])', t1)
            n[key + "_max"] = prom_q(f'max_over_time({m}{{instance="{inst}"}}[{w}])', t1)
        n["board_temp_c_max"] = prom_q(f'max_over_time((max(node_hwmon_temp_celsius{{instance="{inst}"}}))[{w}:5s])', t1)
        # RoCE/RDMA bypasses node_network -> use InfiniBand HW counters (bytes)
        n["link_tx_gbps_avg"] = _g(prom_q(f'8*sum(increase(node_infiniband_port_data_transmitted_bytes_total{{instance="{inst}"}}[{w}]))/{d}', t1))
        n["link_rx_gbps_avg"] = _g(prom_q(f'8*sum(increase(node_infiniband_port_data_received_bytes_total{{instance="{inst}"}}[{w}]))/{d}', t1))
        n["link_tx_gbps_max"] = _g(prom_q(f'max_over_time((8*sum(rate(node_infiniband_port_data_transmitted_bytes_total{{instance="{inst}"}}[30s])))[{w}:10s])', t1))
        out[tag] = n
    return out

def chat(endpoint, model, max_tokens, stream=False):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.7, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(endpoint + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    if not stream:
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=600))
        dt = time.time() - t0
        ct = r.get("usage", {}).get("completion_tokens", 0)
        return {"completion_tokens": ct, "elapsed_s": dt, "tok_s": ct / dt if dt else 0}
    # streaming: capture TTFT + decode rate
    t0 = time.time(); ttft = None; ntok = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: obj = json.loads(payload)
            except Exception: continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None: ttft = time.time() - t0
                ntok += 1
    total = time.time() - t0
    decode = (ntok - 1) / (total - ttft) if (ttft and total > ttft and ntok > 1) else (ntok / total if total else 0)
    return {"ttft_s": round(ttft, 4) if ttft else None, "tokens": ntok,
            "total_s": round(total, 4), "decode_tok_s": round(decode, 2)}

def single_stream(endpoint, model, reps, max_tokens):
    runs = [chat(endpoint, model, max_tokens, stream=True) for _ in range(reps)]
    dec = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
    ttfts = [r["ttft_s"] for r in runs if r.get("ttft_s")]
    return {"runs": runs,
            "decode_tok_s_mean": round(sum(dec) / len(dec), 2) if dec else None,
            "decode_tok_s_max": max(dec) if dec else None,
            "ttft_s_mean": round(sum(ttfts) / len(ttfts), 4) if ttfts else None}

def load_phase(endpoint, model, conc, seconds, max_tokens):
    stop = time.time() + seconds
    results = []; lock = threading.Lock()
    def worker():
        while time.time() < stop:
            try: r = chat(endpoint, model, max_tokens, stream=False)
            except Exception: r = {"completion_tokens": 0, "elapsed_s": 0, "tok_s": 0}
            with lock: results.append(r)
    th = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    toks = sum(r["completion_tokens"] for r in results)
    return {"concurrency": conc, "requests": len(results), "wall_s": round(wall, 2),
            "total_tokens": toks, "agg_tok_s": round(toks / wall, 1) if wall else 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", default="1,4,8,16")   # c>=32 can time out the cross-node RPC and crash the engine
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--load-seconds", type=int, default=30)
    ap.add_argument("--mode", default="cluster")
    a = ap.parse_args()
    sweep = [int(x) for x in a.sweep.split(",") if x.strip()]
    rec = {"label": a.label or a.model, "model": a.model, "endpoint": a.endpoint,
           "mode": a.mode, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "max_tokens": a.max_tokens, "phases": {}}

    # idle baseline telemetry (10s)
    ti = time.time(); time.sleep(10)
    rec["phases"]["idle"] = {"telemetry": telemetry(ti, time.time())}
    print("[idle] captured")

    # single-stream
    p0 = time.time()
    ss = single_stream(a.endpoint, a.model, a.reps, a.max_tokens)
    ss["telemetry"] = telemetry(p0, time.time())
    rec["phases"]["single_stream"] = ss
    print(f"[single-stream] decode {ss['decode_tok_s_mean']} tok/s  ttft {ss['ttft_s_mean']}s  "
          f"gpu {ss['telemetry']['dgx1']['gpu_temp_c_max']}/{ss['telemetry']['dgx2']['gpu_temp_c_max']}C  "
          f"link {ss['telemetry']['dgx1']['link_tx_gbps_max']}Gb/s")

    # concurrency sweep
    rec["phases"]["sweep"] = []
    for c in sweep:
        p = time.time()
        lp = load_phase(a.endpoint, a.model, c, a.load_seconds, a.max_tokens)
        lp["telemetry"] = telemetry(p, time.time())
        rec["phases"]["sweep"].append(lp)
        t1 = lp["telemetry"]["dgx1"]; t2 = lp["telemetry"]["dgx2"]
        print(f"[c={c:>3}] agg {lp['agg_tok_s']:>7} tok/s | "
              f"gpu {t1['gpu_temp_c_max']}/{t2['gpu_temp_c_max']}C mem {t1['mem_temp_c_max']}/{t2['mem_temp_c_max']}C | "
              f"pwr {t1['power_w_avg']}/{t2['power_w_avg']}W | link {t1['link_tx_gbps_max']}->/{t2['link_tx_gbps_max']}-> Gb/s")

    rec["peak_agg_tok_s"] = max((s["agg_tok_s"] for s in rec["phases"]["sweep"]), default=0)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f: json.dump(rec, f, indent=2)
    print("WROTE", a.out, "| peak agg", rec["peak_agg_tok_s"], "tok/s")

if __name__ == "__main__":
    main()
