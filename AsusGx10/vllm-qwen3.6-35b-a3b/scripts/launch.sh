#!/usr/bin/env bash
#
# launch.sh — start (or restart) the vLLM server for Qwen3.6-35B-A3B (NVFP4) on the GX10/GB10.
#
# Mirrors the validated production command, but every knob is an overridable env var so the
# same script drives the guide-05 tuning sweeps. Backs up the running container before
# replacing it (rollback.sh reverses that), and waits on /health before returning.
#
# Usage:
#   ./launch.sh                         # launch with the current production defaults
#   MAX_NUM_BATCHED_TOKENS=8192 ./launch.sh   # override one knob
#   DRY_RUN=1 ./launch.sh               # print the docker command, change nothing
#
# See guides/01–04 for what each knob does, and scripts/README.md for the full variable list.
set -euo pipefail

# ---- configurable knobs (defaults = current production config) --------------------------
DOCKER="${DOCKER:-sudo docker}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
CONTAINER="${CONTAINER:-vllm}"
BACKUP="${BACKUP:-vllm_prebackup}"
MODELS_DIR="${MODELS_DIR:-/home/user/models}"
MODEL_PATH="${MODEL_PATH:-/models/qwen36-35b-moe-nvfp4}"
SERVED_NAME="${SERVED_NAME:-qwen36-moe}"
PORT="${PORT:-8000}"
SHM_SIZE="${SHM_SIZE:-16g}"

QUANTIZATION="${QUANTIZATION:-modelopt}"
DTYPE="${DTYPE:-auto}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Extra serve flags appended verbatim (e.g. EXTRA_ARGS="--speculative-config '{...}'")
EXTRA_ARGS="${EXTRA_ARGS:-}"

# vLLM/FlashInfer env passed into the container (see guide 02 for the sm_121 backstory)
ENV_VLLM_USE_FLASHINFER_MOE_FP4="${ENV_VLLM_USE_FLASHINFER_MOE_FP4:-0}"
ENV_VLLM_FP8_MOE_BACKEND="${ENV_VLLM_FP8_MOE_BACKEND:-flashinfer_cutlass}"
ENV_FLASHINFER_DISABLE_VERSION_CHECK="${ENV_FLASHINFER_DISABLE_VERSION_CHECK:-1}"
ENV_CUTE_DSL_ARCH="${ENV_CUTE_DSL_ARCH:-sm_121a}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"   # cold start is ~5–6 min (weights + torch.compile)
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[launch] %s\n' "$*" >&2; }

build_cmd() {
  # shellcheck disable=SC2206
  cmd=( $DOCKER run -d --name "$CONTAINER" --runtime=nvidia --gpus=all --shm-size="$SHM_SIZE"
    -p "${PORT}:8000" -v "${MODELS_DIR}:/models"
    -e VLLM_USE_FLASHINFER_MOE_FP4="$ENV_VLLM_USE_FLASHINFER_MOE_FP4"
    -e VLLM_FP8_MOE_BACKEND="$ENV_VLLM_FP8_MOE_BACKEND"
    -e FLASHINFER_DISABLE_VERSION_CHECK="$ENV_FLASHINFER_DISABLE_VERSION_CHECK"
    -e CUTE_DSL_ARCH="$ENV_CUTE_DSL_ARCH"
    "$IMAGE"
    --model "$MODEL_PATH" --quantization "$QUANTIZATION" --dtype "$DTYPE"
    --kv-cache-dtype "$KV_CACHE_DTYPE" --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS" --enable-chunked-prefill --enable-prefix-caching
    --served-model-name "$SERVED_NAME"
    --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER"
    --reasoning-parser "$REASONING_PARSER" )
  [ -n "$EXTRA_ARGS" ] && cmd+=( $EXTRA_ARGS )
}

backup_existing() {
  if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    log "existing container '$CONTAINER' found — backing up as '$BACKUP'"
    $DOCKER rm -f "$BACKUP" >/dev/null 2>&1 || true
    $DOCKER stop "$CONTAINER" >/dev/null 2>&1 || true
    $DOCKER rename "$CONTAINER" "$BACKUP"
  fi
}

wait_healthy() {
  log "waiting for /health on :$PORT (timeout ${HEALTH_TIMEOUT}s; cold start ~5–6 min)…"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  until curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    [ "$(date +%s)" -ge "$deadline" ] && { log "TIMEOUT waiting for health"; $DOCKER logs --tail 40 "$CONTAINER" >&2 || true; return 1; }
    sleep 5
  done
  log "server healthy. Model served as '$SERVED_NAME' at http://localhost:${PORT}/v1"
}

main() {
  build_cmd
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN — command that would run:"; printf '  %q ' "${cmd[@]}"; echo; exit 0
  fi
  backup_existing
  log "launching ${IMAGE} (model=${MODEL_PATH}, mnbt=${MAX_NUM_BATCHED_TOKENS}, seqs=${MAX_NUM_SEQS}, ctx=${MAX_MODEL_LEN})"
  "${cmd[@]}"
  wait_healthy
}

main "$@"
