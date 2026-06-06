import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BW = 273e9            # GB10 memory bandwidth, bytes/s
BPP = 0.5625          # NVFP4 bytes/param (4.5 bits)
def ceil(active):     # single-stream decode ceiling, weights-only
    return BW / (active * BPP)

# (name, active_params, measured_single_stream_or_None, is_moe)
models = [
    ("Qwen3.6-35B-A3B",            3.0e9, 75,  True),
    ("Gemma-4-26B-A4B",            3.8e9, 52,  True),
    ("Nemotron-3-Nano-30B-A3B",    3.0e9, None, True),
    ("Nemotron-3-Super-120B-A12B", 12e9,  None, True),
    ("Gemma-4-31B",                31e9,  6.9, False),
    ("Qwen3-32B",                  32e9,  None, False),
    ("Llama-3.3-70B",              70e9,  None, False),
]

x = np.logspace(9, 11, 240)
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.plot(x/1e9, ceil(x),       color="#1f4eb0", lw=2.2, label="Peak ceiling (273 GB/s)")
ax.plot(x/1e9, 0.8*ceil(x),   color="#1f4eb0", lw=1.4, ls="--", alpha=0.7, label="Realistic (~80% of peak BW)")

for name, act, meas, moe in models:
    c = ceil(act)
    ax.plot(act/1e9, c, "o", color="#1f4eb0", ms=8, zorder=5)
    dy = 10 if name != "Nemotron-3-Nano-30B-A3B" else -16
    ax.annotate(name, (act/1e9, c), textcoords="offset points", xytext=(7, dy), fontsize=8.2)
    if meas:
        ax.plot(act/1e9, meas, "^", color="#c0392b", ms=11, zorder=6)
        ax.annotate(f"{meas} measured", (act/1e9, meas), textcoords="offset points",
                    xytext=(7, -13), fontsize=8, color="#c0392b")

ax.plot([], [], "o", color="#1f4eb0", label="Theoretical (per model · by active params)")
ax.plot([], [], "^", color="#c0392b", label="Measured single-stream")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Active parameters (billions)  →  bandwidth cost per token")
ax.set_ylabel("Single-stream decode (tok/s)")
ax.set_title("NVFP4 decode roofline on the GX10 (GB10, 273 GB/s)\n"
             r"ceiling $=$ bandwidth $\div$ (active_params $\times$ 0.5625 B/param)")
ax.grid(True, which="both", alpha=0.28)
ax.legend(loc="upper right", fontsize=8.6, framealpha=0.95)
fig.text(0.012, 0.012, "Measured points sit ~50-58% of the realistic ceiling — the constant 'everything-but-weights' "
         "overhead (KV reads, attention, Marlin FP4->BF16, per-token).", fontsize=7.2, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("/root/gemma-research/roofline-nvfp4-gx10.png", dpi=135)
print("saved /root/gemma-research/roofline-nvfp4-gx10.png")
