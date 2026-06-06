#!/usr/bin/env bash
#
# dgx-spark-bootstrap.sh — out-of-box: bare device → a serving vLLM endpoint for the most
# efficient NVFP4 model from this study, on an NVIDIA DGX Spark (GB10 / sm_121).
#
# Design: phased (mirrors NVIDIA's dgx-spark-playbooks structure), environment-adaptive
# (uses the factory DGX-OS stack if present, otherwise installs deps on generic Ubuntu 24.04),
# idempotent, and HEAVILY logged for human + AI-agent troubleshooting.
#
# Every run writes to  ~/.dgx-bootstrap/logs/<UTC>/ :
#     run.log          full transcript (every command + output + exit code)
#     report.json      machine-readable summary (status, versions, model, endpoint, warnings[])
#     diagnostics/      auto-collected snapshot on ANY failure (nvidia-smi, dmesg, docker logs, …)
#
# The vLLM serve commands are FROZEN from configs that deployed healthy in this project's
# benchmark matrix. The bare-metal install path is MODELED on NVIDIA's documented steps
# (not executed on a fresh box during development) — it is clearly logged so a first run is debuggable.
#
# Usage:
#   ./dgx-spark-bootstrap.sh                 # interactive model pick
#   MODEL=nano   ./dgx-spark-bootstrap.sh    # non-interactive (nano|qwen|llama)
#   DRY_RUN=1    ./dgx-spark-bootstrap.sh    # detect + plan only, deploy nothing
set -uo pipefail

# ---------------------------------------------------------------- config / tunables
DOCKER="${DOCKER:-docker}"                       # set to "sudo docker" if not in the docker group
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
DRY_RUN="${DRY_RUN:-0}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"                 # refuse to run below this much free disk
TESTED_DRIVER="580.159.03"                       # the driver version this study validated

# ---------------------------------------------------------------- logging stack
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="${LOGDIR:-$HOME/.dgx-bootstrap/logs/$TS}"
mkdir -p "$LOGDIR/diagnostics"
RUNLOG="$LOGDIR/run.log"; REPORT="$LOGDIR/report.json"; WARN_F="$LOGDIR/.warnings"; ERR_F="$LOGDIR/.errors"
: > "$WARN_F"; : > "$ERR_F"
# capture ALL stdout+stderr to console AND run.log
exec > >(tee -a "$RUNLOG") 2>&1

PHASE="init"
ts() { date -u +%H:%M:%S; }
log()  { printf '[%s] [INFO]  [%s] %s\n' "$(ts)" "$PHASE" "$*"; }
step() { PHASE="$1"; printf '\n[%s] [STEP]  ===== %s =====\n' "$(ts)" "$2"; }
warn() { printf '[%s] [WARN]  [%s] %s\n' "$(ts)" "$PHASE" "$*"; echo "$*" >> "$WARN_F"; }
err()  { printf '[%s] [ERROR] [%s] %s\n' "$(ts)" "$PHASE" "$*"; echo "$*" >> "$ERR_F"; }
run()  { printf '[%s] [RUN]   [%s] $ %s\n' "$(ts)" "$PHASE" "$*"; "$@"; }

# report.json fields (populated as we go)
declare -A R=( [status]=started [phase_failed]="" [os]="?" [is_dgx_os]="?" [gpu]="?" [driver]="?" \
  [secure_boot]="?" [docker]="?" [toolkit]="?" [free_gb]="?" [model]="?" [served_name]="?" [endpoint]="?" [smoke_tok_s]="?" )

emit_report() {
  python3 - "$REPORT" "$WARN_F" "$ERR_F" \
    "${R[status]}" "${R[phase_failed]}" "${R[os]}" "${R[is_dgx_os]}" "${R[gpu]}" "${R[driver]}" \
    "${R[secure_boot]}" "${R[docker]}" "${R[toolkit]}" "${R[free_gb]}" "${R[model]}" \
    "${R[served_name]}" "${R[endpoint]}" "${R[smoke_tok_s]}" "$TS" <<'PY'
import json, sys
report, warn_f, err_f = sys.argv[1], sys.argv[2], sys.argv[3]
keys = ["status","phase_failed","os","is_dgx_os","gpu","driver","secure_boot","docker","toolkit",
        "free_gb","model","served_name","endpoint","smoke_tok_s","run_id"]
d = dict(zip(keys, sys.argv[4:]))
d["warnings"] = [l.strip() for l in open(warn_f) if l.strip()]
d["errors"]   = [l.strip() for l in open(err_f) if l.strip()]
json.dump(d, open(report, "w"), indent=2)
print(json.dumps(d, indent=2))
PY
}

collect_diagnostics() {
  local d="$LOGDIR/diagnostics"
  log "collecting diagnostics bundle → $d"
  { nvidia-smi -q || echo "nvidia-smi failed"; }            > "$d/nvidia-smi.txt" 2>&1
  { lsmod | grep -i nvidia || echo "no nvidia modules"; }   > "$d/lsmod-nvidia.txt" 2>&1
  { mokutil --sb-state 2>&1 || true; }                      > "$d/secureboot.txt" 2>&1
  { $DOCKER ps -a 2>&1 || true; }                           > "$d/docker-ps.txt" 2>&1
  { $DOCKER logs --tail 80 vllm 2>&1 || true; }             > "$d/vllm-logs.txt" 2>&1
  { journalctl -k -b 0 2>&1 | grep -iE "nvrm|xid|nvidia|oom" | tail -60 || true; } > "$d/kernel-nvidia.txt" 2>&1
  { df -h /; free -g; uname -a; cat /etc/os-release; } > "$d/system.txt" 2>&1
}

on_error() {
  local rc=$1
  R[status]=failed; R[phase_failed]="$PHASE"
  err "FAILED in phase '$PHASE' (exit $rc)"
  collect_diagnostics
  emit_report
  echo
  echo "❌ Bootstrap FAILED in phase '$PHASE'."
  echo "   Full transcript : $RUNLOG"
  echo "   Machine report  : $REPORT   (paste this to an AI agent)"
  echo "   Diagnostics     : $LOGDIR/diagnostics/"
  exit "$rc"
}
trap 'on_error $?' ERR

echo "=== DGX Spark bootstrap — run $TS ==="
echo "Logs for this run: $LOGDIR"
echo "  • live transcript : $RUNLOG"
echo "  • report.json     : $REPORT"
echo "  • diagnostics (on failure): $LOGDIR/diagnostics/"
echo

# ---------------------------------------------------------------- Phase 0: preflight
step preflight "Phase 0 — environment detection"
. /etc/os-release 2>/dev/null || true
R[os]="${PRETTY_NAME:-unknown}"; log "OS: ${R[os]}"
if [ -f /etc/dgx-release ] || echo "${R[os]}" | grep -qi "dgx"; then R[is_dgx_os]=yes; else R[is_dgx_os]=no; fi
log "DGX-OS: ${R[is_dgx_os]}"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  R[gpu]="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  R[driver]="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  log "GPU: ${R[gpu]} (driver ${R[driver]})"
  echo "${R[gpu]}" | grep -qiE "GB10|GB200|Blackwell" || warn "GPU '${R[gpu]}' is not a recognized GB10/Blackwell — NVFP4 configs are tuned for sm_121"
  [ "${R[driver]}" = "$TESTED_DRIVER" ] || warn "driver ${R[driver]} differs from the study-validated $TESTED_DRIVER (usually fine)"
else
  R[gpu]="NONE"; R[driver]="NONE"
  warn "nvidia-smi cannot talk to the driver — GPU not ready"
fi

R[secure_boot]="$(mokutil --sb-state 2>/dev/null | tr -d '\n' || echo unknown)"; log "Secure Boot: ${R[secure_boot]}"
if echo "${R[secure_boot]}" | grep -qi enabled && [ "${R[driver]}" = "NONE" ]; then
  warn "Secure Boot ENABLED and the NVIDIA module isn't loaded — if a DKMS module signature was rejected ('Key was rejected by service'), re-enroll the MOK (sudo mokutil --import /var/lib/shim-signed/mok/MOK.der + reboot + enroll at the blue screen) or disable Secure Boot in firmware. See bootstrap/README."
fi

command -v docker  >/dev/null 2>&1 && R[docker]="$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ,)"  || R[docker]=MISSING
{ command -v nvidia-ctk >/dev/null 2>&1 || [ -f /etc/docker/daemon.json ] && grep -q nvidia /etc/docker/daemon.json 2>/dev/null; } && R[toolkit]=present || R[toolkit]=MISSING
R[free_gb]="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
log "docker: ${R[docker]} | nvidia-container-toolkit: ${R[toolkit]} | free disk: ${R[free_gb]} GB"
[ "${R[free_gb]:-0}" -ge "$MIN_FREE_GB" ] || { err "only ${R[free_gb]} GB free; need >= $MIN_FREE_GB GB"; false; }

# ---------------------------------------------------------------- Phase 1: dependencies
step deps "Phase 1 — dependency resolution"
need_install=0
[ "${R[docker]}" = MISSING ] && need_install=1
[ "${R[toolkit]}" = MISSING ] && need_install=1
[ "${R[driver]}" = NONE ] && need_install=1
if [ "$need_install" = 0 ]; then
  log "stack present (docker + toolkit + driver) — using it as-is."
elif [ "${R[is_dgx_os]}" = yes ]; then
  err "On DGX-OS but part of the NVIDIA stack is missing (docker=${R[docker]} toolkit=${R[toolkit]} driver=${R[driver]}). This is unexpected on a factory image — not auto-installing; inspect ${RUNLOG}."
  false
else
  warn "Generic OS (not DGX-OS) with missing pieces — running the MODELED bare-metal install (per NVIDIA's documented steps; not executed on a fresh box during dev)."
  log "Dependency manifest to install: $([ "${R[driver]}" = NONE ] && echo 'nvidia-driver(+cuda) ')$([ "${R[docker]}" = MISSING ] && echo 'docker ')$([ "${R[toolkit]}" = MISSING ] && echo 'nvidia-container-toolkit')"
  if [ "$DRY_RUN" = 1 ]; then
    log "DRY_RUN: skipping installs."
  else
    if [ "${R[docker]}" = MISSING ]; then run curl -fsSL https://get.docker.com | sh; run sudo usermod -aG docker "$USER" || true; fi
    if [ "${R[toolkit]}" = MISSING ]; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      run sudo apt-get update && run sudo apt-get install -y nvidia-container-toolkit
      run sudo nvidia-ctk runtime configure --runtime=docker && run sudo systemctl restart docker
    fi
    [ "${R[driver]}" = NONE ] && warn "NVIDIA driver install on generic Ubuntu for GB10 is hardware-specific — install NVIDIA's DGX-Spark/Grace-Blackwell driver per https://docs.nvidia.com/dgx/dgx-spark/ then re-run this script."
    [ "${R[driver]}" = NONE ] && { err "driver still absent — cannot proceed"; false; }
  fi
fi

# ---------------------------------------------------------------- Phase 2: model selection
step model "Phase 2 — choose the model"
choose() {
  cat <<'MENU'
Pick the model to serve:
  1) nano  — Nemotron-3-Nano-30B-A3B (NVFP4, MoE 3B active)
             MOST EFFICIENT: ~1215 tok/s aggregate at the lowest power (44 W), 54 tok/s single-stream.
             Best when you serve many requests and care about throughput-per-watt.  (~19 GB)
  2) qwen  — Qwen3.6-35B-A3B (NVFP4, MoE 3B active)
             FASTEST SINGLE-USER: 75 tok/s single-stream, 951 aggregate; this project's production model.
             Best for snappy 1:1 chat / agent tool-calling (qwen3_coder parser baked in).  (~22 GB)
  3) llama — Llama-3.3-70B-Instruct (NVFP4, dense 70B)
             HIGHEST ROOFLINE EFFICIENCY (98%) but SLOW: 5.4 tok/s single-stream. Frontier dense
             quality; pick only if you accept low speed for a big dense model.  (~40 GB)
MENU
}
MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
  if [ -t 0 ]; then choose; read -rp "Enter nano | qwen | llama [nano]: " MODEL; MODEL="${MODEL:-nano}"
  else MODEL=nano; warn "non-interactive and no MODEL set → defaulting to 'nano'"; fi
fi
case "$MODEL" in
  nano)  REPO="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"; DIR="nemotron-nano-30b-a3b-nvfp4"; SERVED="nemotron-nano"; UTIL=0.85
         EXTRA=(--quantization modelopt --dtype auto --kv-cache-dtype fp8 --max-model-len 32768 --enable-chunked-prefill --trust-remote-code) ;;
  qwen)  REPO="nvidia/Qwen3.6-35B-A3B-NVFP4"; DIR="qwen36-35b-moe-nvfp4"; SERVED="qwen36-moe"; UTIL=0.90
         EXTRA=(--quantization modelopt --dtype auto --kv-cache-dtype fp8 --max-model-len 32768 --max-num-batched-tokens 4096 --max-num-seqs 128 --enable-chunked-prefill --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder) ;;
  llama) REPO="nvidia/Llama-3.3-70B-Instruct-NVFP4"; DIR="llama33-70b-nvfp4"; SERVED="llama-70b"; UTIL=0.70   # conservative: 40 GB model, the OOM lesson
         EXTRA=(--quantization modelopt --dtype auto --kv-cache-dtype fp8 --max-model-len 32768 --enable-chunked-prefill --trust-remote-code) ;;
  *) err "unknown MODEL '$MODEL' (want nano|qwen|llama)"; false ;;
esac
R[model]="$MODEL ($REPO)"; R[served_name]="$SERVED"
log "selected: $MODEL → $REPO  (served as '$SERVED', gpu-mem-util $UTIL)"
[ "$DRY_RUN" = 1 ] && { log "DRY_RUN: plan complete, deploying nothing."; R[status]=dry-run; emit_report; exit 0; }

# ---------------------------------------------------------------- Phase 3: fetch
step fetch "Phase 3 — download checkpoint"
mkdir -p "$MODELS_DIR"
if [ -d "$MODELS_DIR/$DIR" ] && [ -n "$(ls -A "$MODELS_DIR/$DIR" 2>/dev/null)" ]; then
  log "checkpoint already present: $MODELS_DIR/$DIR ($(du -sh "$MODELS_DIR/$DIR" 2>/dev/null | cut -f1))"
else
  log "downloading $REPO → $MODELS_DIR/$DIR (several GB; logged)…"
  run $DOCKER run --rm --entrypoint python3 -v "$MODELS_DIR":/models -v "$HF_CACHE":/root/.cache/huggingface "$IMAGE" \
    -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', local_dir='/models/$DIR')"
  log "downloaded ($(du -sh "$MODELS_DIR/$DIR" 2>/dev/null | cut -f1))"
fi

# ---------------------------------------------------------------- Phase 4: deploy
step deploy "Phase 4 — launch vLLM"
run $DOCKER rm -f "$SERVED" >/dev/null 2>&1 || true
run $DOCKER run -d --name "$SERVED" --runtime=nvidia --gpus=all --shm-size=16g -p "${PORT}:8000" \
  -v "$MODELS_DIR":/models \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FP8_MOE_BACKEND=flashinfer_cutlass \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CUTE_DSL_ARCH=sm_121a \
  "$IMAGE" --model "/models/$DIR" --served-model-name "$SERVED" --gpu-memory-utilization "$UTIL" "${EXTRA[@]}"
R[endpoint]="http://localhost:${PORT}/v1"

step verify "Phase 5 — wait for health + smoke test"
log "waiting for /health (cold start can be 5–7 min)…"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { err "health timeout after ${HEALTH_TIMEOUT}s"; false; }
  sleep 5
done
log "healthy. served models: $(curl -fsS http://localhost:${PORT}/v1/models 2>/dev/null | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin).get("data",[])])' 2>/dev/null)"
t0=$(date +%s%3N)
ntok=$(curl -fsS "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}]}" 2>/dev/null \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["usage"]["completion_tokens"])' 2>/dev/null || echo 0)
t1=$(date +%s%3N); dt=$(( t1 - t0 )); [ "$dt" -gt 0 ] && [ "$ntok" -gt 0 ] && R[smoke_tok_s]="$(awk "BEGIN{printf \"%.1f\", $ntok*1000/$dt}")"
log "smoke test: $ntok tokens in ${dt}ms (~${R[smoke_tok_s]} tok/s)"

# ---------------------------------------------------------------- Phase 6: report
step report "Phase 6 — report"
R[status]=success
emit_report
echo
echo "✅ DGX Spark ready. '$SERVED' serving at ${R[endpoint]}"
echo "   Logs: $LOGDIR   (report.json + run.log)"
echo "   Test: curl ${R[endpoint]}/models"
