#!/usr/bin/env bash
#
# tune.sh — sweep one serve knob, relaunching + benchmarking at each value (guide 05).
#
# For each value it: sets the knob, runs launch.sh (server restart, ~5–6 min cold start each!),
# waits for health, runs benchmark.sh, and appends a row to a summary CSV. This is the
# automation behind the guide-05 experiment table.
#
# ⚠️  This RESTARTS the production server repeatedly — only run inside a maintenance window.
#
# Usage:
#   SWEEP_VAR=MAX_NUM_BATCHED_TOKENS SWEEP_VALUES="2048 4096 8192 16384" ./tune.sh
#   SWEEP_VAR=MAX_NUM_SEQS SWEEP_VALUES="64 128 256" ./tune.sh
#   DRY_RUN=1 ./tune.sh        # show the plan, change nothing
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SWEEP_VAR="${SWEEP_VAR:-MAX_NUM_BATCHED_TOKENS}"
SWEEP_VALUES="${SWEEP_VALUES:-2048 4096 8192 16384}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
SUMMARY="${SUMMARY:-${RESULTS_DIR}/sweep-${SWEEP_VAR}.csv}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[tune] %s\n' "$*" >&2; }
mkdir -p "$RESULTS_DIR"

log "sweep ${SWEEP_VAR} over: ${SWEEP_VALUES}"
if [ "$DRY_RUN" = "1" ]; then
  for v in $SWEEP_VALUES; do log "  would set ${SWEEP_VAR}=${v} -> launch -> benchmark"; done
  exit 0
fi

read -r -p "This restarts the vLLM server $(set -- $SWEEP_VALUES; echo $#) times (downtime). Continue? [y/N] " ans
[ "${ans:-N}" = "y" ] || { log "aborted."; exit 1; }

echo "stamp,${SWEEP_VAR},pass,result_json" > "$SUMMARY"
for v in $SWEEP_VALUES; do
  log "=== ${SWEEP_VAR}=${v} ==="
  export "${SWEEP_VAR}=${v}"
  RESULTS_DIR="$RESULTS_DIR" "${HERE}/launch.sh"            # restart with this value
  LABEL="${SWEEP_VAR,,}${v}" RESULTS_DIR="$RESULTS_DIR" "${HERE}/benchmark.sh"
  for f in "${RESULTS_DIR}/"*"${SWEEP_VAR,,}${v}"-*.json; do
    [ -f "$f" ] || continue
    pass="$(basename "$f" | sed -E 's/.*-(single-stream|saturated)\.json/\1/')"
    echo "$(date +%Y%m%dT%H%M%S),${v},${pass},${f}" >> "$SUMMARY"
  done
done
log "sweep complete. Summary -> ${SUMMARY}. Restore last-known-good with ./rollback.sh if needed."
