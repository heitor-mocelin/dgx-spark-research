# bootstrap/ — out-of-box setup script

[`dgx-spark-bootstrap.sh`](dgx-spark-bootstrap.sh) takes a DGX Spark (GB10) from **bare device →
a serving vLLM endpoint** for the most efficient NVFP4 model from [this study](../FINDINGS.md), in
one command. Built for a **factory-reset / fresh box**, and designed not to break.

## Quick start
```bash
# on the DGX (or copy the script over)
./dgx-spark-bootstrap.sh                 # interactive: pick nano | qwen | llama
MODEL=nano ./dgx-spark-bootstrap.sh      # non-interactive (automation / AI agents)
DRY_RUN=1 ./dgx-spark-bootstrap.sh       # detect + plan only, deploy nothing
```
Useful env overrides: `MODEL` · `PORT` (8000) · `DOCKER` (`sudo docker` if not in the docker group) ·
`GPU_MEM_UTIL` per-model defaults · `MODELS_DIR` · `LOGDIR`.

## The 3 models it offers (with the trade-off shown at the prompt)
| Choice | Model | Pick it for | Measured |
|---|---|---|---|
| **nano** | Nemotron-3-Nano-30B-A3B (NVFP4) | **most efficient** — throughput-per-watt | 54 single · **1215 agg @ 44 W** |
| **qwen** | Qwen3.6-35B-A3B (NVFP4) | **fastest single-user** (production model) | **75 single** · 951 agg |
| **llama** | Llama-3.3-70B (NVFP4, dense) | highest roofline % but slow | 5.4 single (98% of ceiling) |

Serve commands are **frozen from the configs that deployed healthy in the benchmark matrix** —
including the production Qwen `qwen3_coder` tool-calling flags and a conservative `gpu-memory-utilization`
for the 40 GB Llama (the [memory-OOM lesson](../FINDINGS.md)).

## What it does (phased — mirrors NVIDIA's dgx-spark-playbooks structure)
0. **Preflight** — detect OS / GPU / driver / docker / toolkit / Secure Boot / disk; flag deviations.
1. **Dependencies** — on **DGX-OS**, use the factory stack as-is; on **generic Ubuntu 24.04**, install
   the missing pieces (docker → nvidia-container-toolkit → driver) from NVIDIA's documented steps.
2. **Model selection** — interactive menu (or `MODEL=`).
3. **Fetch** — download the NVFP4 checkpoint (idempotent; skips if present).
4. **Deploy** — the frozen `docker run` for that model.
5. **Verify** — wait on `/health`, list models, run a smoke `chat/completions` (records tok/s).
6. **Report** — write the machine + human summary.

## Logging stack (built for humans *and* AI agents)
Every run writes to `~/.dgx-bootstrap/logs/<UTC>/` (the path is printed at start):
- **`run.log`** — complete transcript: `[time] [LEVEL] [phase] …`, every command + output + exit code.
- **`report.json`** — machine-readable summary an AI agent can ingest directly: `status`, `os`,
  `gpu`, `driver`, `secure_boot`, `docker`, `model`, `endpoint`, `smoke_tok_s`, `warnings[]`,
  `errors[]`. Printed to stdout at the end of *every* run.
- **`diagnostics/`** — on **any failure**, a `trap` auto-collects the troubleshooting snapshot
  (`nvidia-smi -q`, `lsmod | grep nvidia`, `mokutil --sb-state`, `docker logs`, `journalctl -k`
  NVRM/Xid, `df`/`free`/`uname`/os-release) and tells you where it is. Paste `report.json` to an
  agent, or attach the bundle for support.

## Variations it detects & handles
| Detected | Action |
|---|---|
| Not DGX-OS (generic Ubuntu) | switch to the bare-metal install path |
| GPU not GB10/Blackwell | WARN (NVFP4 configs are `sm_121`-tuned) |
| Driver missing | install (generic) / ERROR (DGX-OS, unexpected) |
| **Secure Boot ON + driver not loaded** | WARN + the MOK re-enroll / disable guidance (see below) |
| Driver ≠ tested `580.159.03` | WARN, continue |
| Disk < 60 GB free (configurable) | ERROR, halt |
| Checkpoint already on disk | skip download (idempotent) |

## Tested vs modeled (honest scope)
- ✅ **Serve commands** — frozen from runs that deployed healthy in the matrix.
- ✅ **Preflight + logging + report** — validated live on this GX10 (`DRY_RUN`).
- ⚠️ **Bare-metal install path** — *modeled on NVIDIA's documented driver/toolkit steps*, **not
  executed on a fresh box** during development (no clean GB10 was available). It's heavily logged so
  a first real run is fully debuggable; reconcile the driver step with the current
  [DGX Spark docs](https://docs.nvidia.com/dgx/dgx-spark/) if it changes.

## Secure Boot note (from a real incident)
If the GPU is missing after a reboot and `modprobe nvidia` says **"Key was rejected by service"**,
Secure Boot is rejecting the DKMS module's signing key. Fix (needs a monitor once): either
`sudo mokutil --import /var/lib/shim-signed/mok/MOK.der` + reboot + enroll at the blue MOK screen,
or disable Secure Boot in firmware. A **factory reset** restores a clean, enrolled state.
