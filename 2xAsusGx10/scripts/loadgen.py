#!/usr/bin/env python3
"""Sustained load generator for interconnect measurement. Usage: loadgen.py <seconds> <concurrency>"""
import sys, json, urllib.request, threading, time
secs = int(sys.argv[1]); conc = int(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else "qwen36-moe"
EP = "http://172.27.27.210:8000/v1/chat/completions"
body = json.dumps({"model": model,
                   "messages": [{"role": "user", "content": "Explain tensor and pipeline parallelism in depth, with concrete examples of the communication patterns."}],
                   "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}}).encode()
stop = time.time() + secs
def w():
    while time.time() < stop:
        try:
            urllib.request.urlopen(urllib.request.Request(EP, data=body, headers={"Content-Type": "application/json"}), timeout=300).read()
        except Exception:
            pass
ts = [threading.Thread(target=w) for _ in range(conc)]
[t.start() for t in ts]; [t.join() for t in ts]
print("loadgen done")
