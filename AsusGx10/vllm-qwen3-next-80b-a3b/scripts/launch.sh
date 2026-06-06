#!/usr/bin/env bash
# launch.sh — serve Qwen3-Next-80B-A3B (NVFP4, hybrid Gated-DeltaNet MoE) on the GX10 via vLLM.
# Weights: nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 (~48 GB). Needs --trust-remote-code.
set -euo pipefail
DOCKER="${DOCKER:-docker}"; IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
CONTAINER="${CONTAINER:-qwen3-next}"
MODEL_DIR="${MODEL_DIR:-/home/user/models/qwen3-next-80b-a3b-nvfp4}"
SERVED_NAME="${SERVED_NAME:-qwen3-next}"; PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"   # safe at ~48 GB; use ~0.55 for 60 GB+ models
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"

$DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true
$DOCKER run -d --name "$CONTAINER" --runtime=nvidia --gpus=all --shm-size=16g -p "${PORT}:8000" \
  -v "$(dirname "$MODEL_DIR")":/models \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FP8_MOE_BACKEND=flashinfer_cutlass \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CUTE_DSL_ARCH=sm_121a \
  "$IMAGE" --model "/models/$(basename "$MODEL_DIR")" --served-model-name "$SERVED_NAME" \
  --quantization modelopt --dtype auto --kv-cache-dtype fp8 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-chunked-prefill --trust-remote-code

echo "[qwen3-next] waiting for /health on :$PORT (cold start ~5-7 min)…"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "TIMEOUT"; $DOCKER logs --tail 40 "$CONTAINER"; exit 1; }
  sleep 5
done
echo "[qwen3-next] healthy → http://localhost:${PORT}/v1 (served as $SERVED_NAME)"
