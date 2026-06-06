import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BW, BPP = 273e9, 0.5625
def ceil(a): return BW/(a*BPP)

# name, active, measured_single (None=failed), is_moe, cited(bool)
M = [
    ("Qwen3.6-35B-A3B",            3.0e9, 75.0, True, False),
    ("Nemotron-3-Nano-30B-A3B",    3.0e9, 54.1, True, False),
    ("Qwen3-Next-80B-A3B",         3.0e9, 35.5, True, False),
    ("Gemma-4-26B-A4B (cited)",    3.8e9, 52.0, True, True),
    ("Nemotron-3-Super-120B-A12B", 12e9,  14.6, True, False),
    ("Gemma-4-31B",                31e9,  6.8,  False, False),
    ("Qwen3-32B",                  32e9,  11.0, False, False),
    ("Llama-3.3-70B",              70e9,  5.4,  False, False),
]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.4))

# --- Panel 1: roofline + measured ---
x = np.logspace(9, 11, 240)
ax.plot(x/1e9, ceil(x),     color="#1f4eb0", lw=2.2, label="Peak ceiling (273 GB/s)")
ax.plot(x/1e9, 0.8*ceil(x), color="#1f4eb0", lw=1.4, ls="--", alpha=.7, label="Realistic (80% BW)")
for name, a, meas, moe, cited in M:
    ax.plot(a/1e9, ceil(a), "o", color="#1f4eb0", ms=7, zorder=5)
    if meas:
        mk = "s" if cited else "^"
        ax.plot(a/1e9, meas, mk, color=("#888" if cited else "#c0392b"), ms=10, zorder=6)
        off = {"Qwen3.6-35B-A3B": (8, 5), "Nemotron-3-Nano-30B-A3B": (8, -11),
               "Qwen3-Next-80B-A3B": (-118, -3)}.get(name, (7, -3))
        ax.annotate(name.replace(" (cited)", ""), (a/1e9, meas), textcoords="offset points",
                    xytext=off, fontsize=7.6, color="#333")
ax.plot([], [], "o", color="#1f4eb0", label="Theoretical (per model)")
ax.plot([], [], "^", color="#c0392b", label="Measured single-stream")
ax.plot([], [], "s", color="#888", label="Cited (deploy failed)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Active parameters (billions)"); ax.set_ylabel("Single-stream decode (tok/s)")
ax.set_title("NVFP4 decode roofline — predicted vs measured (GX10)")
ax.grid(True, which="both", alpha=.28); ax.legend(loc="upper right", fontsize=8)

# --- Panel 2: efficiency vs active params (the refuted 'constant' hypothesis) ---
pts = [(a, 100*meas/(0.8*ceil(a)), name, cited) for name, a, meas, moe, cited in M if meas]
bx.axhspan(51, 58, color="#f0c040", alpha=.18, label="original ~55% guess")
for a, eff, name, cited in pts:
    bx.plot(a/1e9, eff, ("s" if cited else "o"), color=("#888" if cited else "#c0392b"), ms=11, zorder=5)
    bx.annotate(name.replace(" (cited)", ""), (a/1e9, eff), textcoords="offset points",
                xytext=(8, 3), fontsize=7.8)
# trend (large models approach 100%)
bx.set_xscale("log"); bx.set_ylim(30, 105)
bx.set_xlabel("Active parameters (billions)")
bx.set_ylabel("Efficiency: measured ÷ realistic ceiling (%)")
bx.set_title("Efficiency is NOT constant — it rises with active params")
bx.grid(True, which="both", alpha=.28); bx.legend(loc="lower right", fontsize=8)
bx.text(0.03, 0.04, "Big/dense weight-read dominates fixed per-token overhead → near-roofline.\n"
        "Small-active MoEs pay more overhead; Gemma-31B is an architecture outlier.",
        transform=bx.transAxes, fontsize=7.4, color="#555", va="bottom")

fig.tight_layout()
fig.savefig("/root/gemma-research/roofline-measured-gx10.png", dpi=130)
print("saved roofline-measured-gx10.png")
