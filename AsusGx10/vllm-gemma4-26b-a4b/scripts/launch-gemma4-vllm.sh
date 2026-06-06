#!/usr/bin/env bash
#
# launch-gemma4-vllm.sh — serve Gemma 4 26B-A4B (NVFP4 MoE) on the GX10/GB10 via vLLM.
#
# Mirrors the validated DGX Spark recipe (ai-muninn, sources/g09): community NVFP4 checkpoint +
# the gemma4 model-code patch, served on an OpenAI-compatible endpoint. Every knob is an env var.
# Backs up an existing container; waits on /health.
#
#   ./launch-gemma4-vllm.sh                 # defaults (26B-A4B NVFP4, 128K ctx)
#   MAX_MODEL_LEN=32768 ./launch-gemma4-vllm.sh
#   DRY_RUN=1 ./launch-gemma4-vllm.sh
#
# NOTE: as of writing, the official NVIDIA NVFP4 checkpoint exists only for the 31B DENSE (slow on
# GB10, ~7 tok/s). The MoE uses the community bg-digitalservices quant + a patched gemma4.py, because
# stock modelopt can't quantize Gemma 4's fused 3D expert tensor format. See guides/02.
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
IMAGE="${IMAGE:-vllm/vllm-openai:cu130-nightly}"   # Gemma 4 needs a recent nightly (sm_121 NVFP4 fixes)
CONTAINER="${CONTAINER:-gemma4-nvfp4}"
BACKUP="${BACKUP:-gemma4_prebackup}"
HF_REPO="${HF_REPO:-bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma4-26b-a4b-nvfp4}"
SERVED_NAME="${SERVED_NAME:-gemma-4-26b}"
PORT="${PORT:-8001}"                                # 8001 so it can run alongside the Qwen server on 8000
QUANTIZATION="${QUANTIZATION:-modelopt}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
VISION_MAX_SOFT_TOKENS="${VISION_MAX_SOFT_TOKENS:-560}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
DRY_RUN="${DRY_RUN:-0}"
PATCH="${MODEL_DIR}/gemma4_patched.py"

log() { printf '[gemma4] %s\n' "$*" >&2; }

ensure_model() {
  if [ ! -d "$MODEL_DIR" ]; then
    log "downloading $HF_REPO -> $MODEL_DIR (gated: accept the Gemma license + set HF_TOKEN)"
    [ "$DRY_RUN" = 1 ] || huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR"
  fi
}

build_cmd() {
  cmd=( $DOCKER run -d --name "$CONTAINER" --runtime=nvidia --gpus=all -p "${PORT}:8000"
    -v "${MODEL_DIR}:/models/gemma4" )
  # mount the patched model file over vLLM's gemma4.py if the checkpoint ships one (needed for NVFP4 scale keys)
  [ -f "$PATCH" ] && cmd+=( -v "${PATCH}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py" )
  cmd+=( "$IMAGE" --model /models/gemma4 --served-model-name "$SERVED_NAME"
    --quantization "$QUANTIZATION" --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
    --mm-processor-kwargs "{\"max_soft_tokens\": ${VISION_MAX_SOFT_TOKENS}}" )
  [ -n "$EXTRA_ARGS" ] && cmd+=( $EXTRA_ARGS )
}

backup_existing() {
  if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    log "backing up existing '$CONTAINER' -> '$BACKUP'"
    $DOCKER rm -f "$BACKUP" >/dev/null 2>&1 || true
    $DOCKER stop "$CONTAINER" >/dev/null 2>&1 || true
    $DOCKER rename "$CONTAINER" "$BACKUP"
  fi
}

wait_healthy() {
  log "waiting for /health on :$PORT (cold start a few min)…"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  until curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    [ "$(date +%s)" -ge "$deadline" ] && { log "TIMEOUT"; $DOCKER logs --tail 40 "$CONTAINER" >&2 || true; return 1; }
    sleep 5
  done
  log "healthy — '$SERVED_NAME' at http://localhost:${PORT}/v1"
}

ensure_model
build_cmd
if [ "$DRY_RUN" = 1 ]; then log "DRY_RUN — command:"; printf '  %q ' "${cmd[@]}"; echo; exit 0; fi
backup_existing
log "launching $IMAGE (26B-A4B NVFP4, ctx=$MAX_MODEL_LEN, port=$PORT)"
"${cmd[@]}"
wait_healthy
