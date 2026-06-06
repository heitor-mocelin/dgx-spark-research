#!/usr/bin/env bash
# launch.sh — serve Nemotron-3-Super-120B-A12B (NVFP4, hybrid LatentMoE) on the GX10 via vLLM.
# Weights: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (~75 GB). Needs --trust-remote-code.
set -euo pipefail
DOCKER="${DOCKER:-docker}"; IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
CONTAINER="${CONTAINER:-nemotron-super}"
MODEL_DIR="${MODEL_DIR:-/home/user/models/nemotron-super-120b-a12b-nvfp4}"
SERVED_NAME="${SERVED_NAME:-nemotron-super}"; PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"   # 75 GB load + compile → allow extra

$DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true
$DOCKER run -d --name "$CONTAINER" --runtime=nvidia --gpus=all --shm-size=16g -p "${PORT}:8000" \
  -v "$(dirname "$MODEL_DIR")":/models \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FP8_MOE_BACKEND=flashinfer_cutlass \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CUTE_DSL_ARCH=sm_121a \
  "$IMAGE" --model "/models/$(basename "$MODEL_DIR")" --served-model-name "$SERVED_NAME" \
  --quantization modelopt --dtype auto --kv-cache-dtype fp8 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-chunked-prefill --trust-remote-code
# For latency: add  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'  (native MTP heads)

echo "[nemotron-super] waiting for /health on :$PORT (75 GB load, cold start may exceed 6 min)…"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "TIMEOUT"; $DOCKER logs --tail 40 "$CONTAINER"; exit 1; }
  sleep 5
done
echo "[nemotron-super] healthy → http://localhost:${PORT}/v1 (served as $SERVED_NAME)"
