#!/usr/bin/env bash
# serve_coding.sh KEY [PORT] [UTIL] — launch one coding model with GB10/sm_121-proven flags, wait /health.
# NVFP4 -> --quantization modelopt + Marlin-fallback env (same path proven for qwen36-moe).
# FP8   -> quant auto-detected from the model's compressed-tensors/fp8 config.
set -uo pipefail
KEY="${1:?usage: serve_coding.sh KEY [PORT] [UTIL]}"; PORT="${2:-8001}"; UTIL="${3:-0.85}"
NAME="cm-${KEY}"
COMMON="--gpus all --runtime nvidia --ipc host --shm-size 16g -p ${PORT}:8000 \
  -v /home/user/models:/models -v /home/user/.cache/huggingface:/root/.cache/huggingface"
FP4ENV="-e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FP8_MOE_BACKEND=flashinfer_cutlass"
IMG_N="vllm/vllm-openai:nightly"
IMG_C="vllm/vllm-openai:cu130-nightly"
SERVE="--max-num-seqs 64 --gpu-memory-utilization ${UTIL}"

docker rm -f "$NAME" >/dev/null 2>&1
case "$KEY" in
  qwen3-coder-30b)        # NVFP4 MoE (flagship)
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG_N \
      --model /models/qwen3-coder-30b-a3b-nvfp4 --served-model-name qwen3-coder-30b \
      --kv-cache-dtype fp8 --max-model-len 131072 $SERVE ;;
  qwen25-coder-32b)       # FP8 dense (reliable; the lone NVFP4 quant had ~351 downloads)
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG_C \
      --model /models/qwen25-coder-32b-fp8 --served-model-name qwen25-coder-32b \
      --kv-cache-dtype fp8 --max-model-len 32768 $SERVE ;;
  qwen25-coder-14b)       # FP8 dense
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG_C \
      --model /models/qwen25-coder-14b-fp8 --served-model-name qwen25-coder-14b \
      --kv-cache-dtype fp8 --max-model-len 32768 $SERVE ;;
  qwen25-coder-7b)        # FP8 dense
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG_C \
      --model /models/qwen25-coder-7b-fp8 --served-model-name qwen25-coder-7b \
      --kv-cache-dtype fp8 --max-model-len 32768 $SERVE ;;
  deepseek-coder-v2-lite) # FP8 MoE w/ MLA -> NO fp8 KV (MLA + fp8 KV crashes engine init on this build)
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG_N \
      --model /models/deepseek-coder-v2-lite-fp8 --served-model-name deepseek-coder-v2-lite \
      --max-model-len 131072 --trust-remote-code $SERVE ;;
  devstral-24b)           # NVFP4, mistral arch, 128k
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG_N \
      --model /models/devstral-24b-nvfp4 --served-model-name devstral-24b \
      --kv-cache-dtype fp8 --max-model-len 131072 $SERVE ;;
  codestral-22b)          # FP8 base/FIM -> force a Mistral [INST] chat template so chat works
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG_C \
      --model /models/codestral-22b-fp8 --served-model-name codestral-22b \
      --chat-template /models/codestral-chat.jinja \
      --kv-cache-dtype fp8 --max-model-len 32768 $SERVE ;;
  *) echo "unknown key: $KEY"; exit 2 ;;
esac

echo "[serve] $NAME on :$PORT (util $UTIL). Logs: docker logs -f $NAME"
for i in $(seq 1 120); do                         # up to 10 min for cold start
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://localhost:${PORT}/health" 2>/dev/null)
  [ "$code" = "200" ] && { echo "[serve] HEALTHY after $((i*5))s"; exit 0; }
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[serve] CONTAINER DIED. Last 40 log lines:"; docker logs --tail 40 "$NAME" 2>&1; exit 1
  fi
  sleep 5
done
echo "[serve] TIMEOUT. Last 40 lines:"; docker logs --tail 40 "$NAME" 2>&1; exit 1
