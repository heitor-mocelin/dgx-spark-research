#!/usr/bin/env bash
#
# rollback.sh — restore the previous vLLM container backed up by launch.sh.
#
# launch.sh renames the running 'vllm' to 'vllm_prebackup' before replacing it; this reverses
# that: stop the current container, restore the backup, wait for health. Use when a tuning
# experiment regressed and you want the last-known-good server back fast.
#
# Mirrors the intent of the device's existing /home/user/vllm-rollback.sh.
#
# Usage:  ./rollback.sh        |        DRY_RUN=1 ./rollback.sh
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
CONTAINER="${CONTAINER:-vllm}"
BACKUP="${BACKUP:-vllm_prebackup}"
PORT="${PORT:-8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[rollback] %s\n' "$*" >&2; }

if ! $DOCKER ps -a --format '{{.Names}}' | grep -qx "$BACKUP"; then
  log "ERROR: no backup container '$BACKUP' to roll back to."; exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN — would: rm -f $CONTAINER ; rename $BACKUP -> $CONTAINER ; start $CONTAINER"; exit 0
fi

log "removing current '$CONTAINER' (if any) and restoring '$BACKUP'"
$DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true
$DOCKER rename "$BACKUP" "$CONTAINER"
$DOCKER start "$CONTAINER"

log "waiting for /health (timeout ${HEALTH_TIMEOUT}s)…"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { log "TIMEOUT — check: $DOCKER logs $CONTAINER"; exit 1; }
  sleep 5
done
log "rollback complete — '$CONTAINER' healthy at http://localhost:${PORT}/v1"
