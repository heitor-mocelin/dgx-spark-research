#!/usr/bin/env python3
"""Chart for the small reflex-model study: (L) single-stream decode tok/s + TTFT;
   (R) decode tok/s vs bytes-read-per-token against the 273 GB/s roofline.
   Run: python3 plot_small.py  ->  small-models-bench.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (name, format, bytes_per_token_GB, ttft_ms, decode_tps, peak_W, agg_c32)
DATA = [
    ("Qwen3-4B",     "NVFP4", 2.25, 32,  64.4, 46.8, 1680),
    ("Gemma-4-E4B",  "NVFP4", 2.50, 40,  53.5, 34.6, 1498),
    ("Llama-3.1-8B", "NVFP4", 4.50, 39,  41.7, 49.2, 1204),
    ("Qwen3-8B",     "NVFP4", 4.60, 52,  39.6, 49.2, 1174),
    ("Phi-4-mini",   "BF16",  7.60, 78,  24.0, 39.3,  800),
    ("Ministral-8B", "BF16", 16.00,137,  14.6, 41.1,  412),
]
BW = 273.0  # GB/s
col = {"NVFP4": "#1f77b4", "BF16": "#d62728"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Left: decode tok/s bars (sorted), TTFT annotated ---
d = sorted(DATA, key=lambda x: x[4])
names = [r[0] for r in d]
tps   = [r[4] for r in d]
cols  = [col[r[1]] for r in d]
y = np.arange(len(names))
ax1.barh(y, tps, color=cols)
ax1.set_yticks(y); ax1.set_yticklabels(names)
for i, r in enumerate(d):
    ax1.text(r[4] + 0.7, i, f"{r[4]:.1f} tok/s  ·  TTFT {r[3]} ms", va="center", fontsize=9)
ax1.set_xlabel("Single-stream decode (tok/s) — higher = snappier")
ax1.set_title("Reflex models: single-user speed on the GX10 (GB10)")
ax1.set_xlim(0, max(tps) * 1.35)
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=col["NVFP4"], label="NVFP4 (4-bit)"),
                    Patch(color=col["BF16"], label="BF16 (16-bit)")], loc="lower right")

# --- Right: decode tok/s vs bytes/token + roofline ---
xs = np.linspace(1.5, 17, 200)
ax2.plot(xs, BW / xs, "--", color="gray", label=f"roofline = {BW:.0f} GB/s ÷ bytes/token")
for r in DATA:
    ax2.scatter(r[2], r[4], color=col[r[1]], s=90, zorder=3)
    ax2.annotate(r[0], (r[2], r[4]), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax2.set_xlabel("Bytes read per token  =  active params × bytes/param  (GB)")
ax2.set_ylabel("Single-stream decode (tok/s)")
ax2.set_title("Decode speed is bandwidth-bound: fewer bytes/token → more tok/s")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

fig.suptitle("Small instruction-tuned models on the ASUS GX10 / NVIDIA GB10 — measured 2026-06",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("small-models-bench.png", dpi=130, bbox_inches="tight")
print("wrote small-models-bench.png")
