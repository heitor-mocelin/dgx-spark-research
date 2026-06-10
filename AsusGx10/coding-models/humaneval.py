#!/usr/bin/env python3
"""HumanEval pass@1 for an OpenAI-compatible endpoint, instruct-style (greedy).
  phase=gen   : prompt each task, extract code block -> completions.json   (needs network to endpoint)
  phase=score : exec completion+test in a subprocess (run me in --network none for safety) -> pass@1
"""
import argparse, json, os, gzip, urllib.request, subprocess, sys, time, re
HE_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

def load_problems(path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(HE_URL, path)
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m[0] if m else text).strip()

def gen(a):
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    probs = load_problems(a.data)
    if a.limit:
        probs = probs[: a.limit]
    sess = requests.Session()
    def one(p):
        prompt = ("Complete the following Python function. Return ONLY the complete function "
                  "definition inside a single ```python code block, with no explanation.\n\n" + p["prompt"])
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": a.max_tokens, "temperature": 0.0}
        r = sess.post(a.base.rstrip("/") + "/v1/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return {"task_id": p["task_id"], "entry_point": p["entry_point"], "test": p["test"],
                "prompt": p["prompt"], "code": extract_code(txt)}
    comps = [None] * len(probs)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:   # problems are independent -> generate concurrently
        futs = {ex.submit(one, p): i for i, p in enumerate(probs)}
        for f in as_completed(futs):
            comps[futs[f]] = f.result(); done += 1
            if done % 40 == 0:
                print(f"  gen {done}/{len(probs)} ({int(time.time()-t0)}s)", flush=True)
    json.dump(comps, open(a.out, "w"))
    print(f"[gen] wrote {len(comps)} completions in {int(time.time()-t0)}s -> {a.out}")

def run_prog(program, timeout):
    try:
        p = subprocess.run([sys.executable, "-c", program], capture_output=True, timeout=timeout)
        return p.returncode == 0
    except Exception:
        return False

def score(a):
    comps = json.load(open(a.inp))
    passed = 0
    for c in comps:
        prog = c["code"] + "\n" + c["test"] + f"\ncheck({c['entry_point']})\n"
        ok = run_prog(prog, a.timeout)
        if not ok:  # model may have returned only the body -> retry with the original signature prefix
            prog2 = c["prompt"] + "\n" + c["code"] + "\n" + c["test"] + f"\ncheck({c['entry_point']})\n"
            ok = run_prog(prog2, a.timeout)
        c["passed"] = ok
        passed += 1 if ok else 0
    n = len(comps)
    summary = {"label": a.label, "pass@1": round(100 * passed / max(1, n), 1), "passed": passed, "total": n}
    print(json.dumps(summary))
    json.dump({"summary": summary, "detail": [{"task_id": c["task_id"], "passed": c["passed"]} for c in comps]},
              open(a.out, "w"), indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    g = sub.add_parser("gen");   g.add_argument("--base", default="http://localhost:8001")
    g.add_argument("--model", required=True); g.add_argument("--out", required=True)
    g.add_argument("--data", default="/work/HumanEval.jsonl.gz"); g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-tokens", type=int, default=512); g.add_argument("--concurrency", type=int, default=16)
    s = sub.add_parser("score"); s.add_argument("--inp", required=True); s.add_argument("--out", required=True)
    s.add_argument("--label", default=""); s.add_argument("--timeout", type=int, default=15)
    a = ap.parse_args()
    (gen if a.phase == "gen" else score)(a)
