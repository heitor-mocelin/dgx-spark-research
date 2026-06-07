#!/usr/bin/env bash
# serve_small.sh KEY [PORT] [UTIL] — launch one small model on cu130-nightly, wait for /health.
# Encodes per-model configs from the research digests. Prints log location; tails logs on failure.
set -uo pipefail
KEY="${1:?usage: serve_small.sh KEY [PORT] [UTIL]}"; PORT="${2:-8001}"; UTIL="${3:-0.85}"
IMG="vllm/vllm-openai:cu130-nightly"
NAME="sm-${KEY}"
COMMON="--gpus all --runtime nvidia --ipc host --shm-size 16g -p ${PORT}:8000 \
  -v /home/user/models:/models -v /home/user/.cache/huggingface:/root/.cache/huggingface"
# FP4 path enablers proven on this box
FP4ENV="-e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_DISABLE_VERSION_CHECK=1"
SERVE="--max-num-seqs 64 --gpu-memory-utilization ${UTIL}"

docker rm -f "$NAME" >/dev/null 2>&1
case "$KEY" in
  gemma-4-e4b)
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG \
      --model /models/gemma4-e4b-nvfp4a16 --served-model-name gemma-4-e4b \
      --quantization compressed-tensors --kv-cache-dtype fp8 --max-model-len 16384 $SERVE ;;
  qwen3-4b)
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG \
      --model /models/qwen3-4b-instruct-nvfp4 --served-model-name qwen3-4b \
      --quantization compressed-tensors --max-model-len 32768 \
      --enable-auto-tool-choice --tool-call-parser hermes $SERVE ;;
  qwen3-8b)
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG \
      --model /models/qwen3-8b-nvfp4 --served-model-name qwen3-8b \
      --quantization modelopt --max-model-len 32768 \
      --enable-auto-tool-choice --tool-call-parser hermes $SERVE ;;
  llama31-8b)
    docker run -d --name "$NAME" $COMMON $FP4ENV $IMG \
      --model /models/llama31-8b-nvfp4 --served-model-name llama31-8b \
      --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser llama3_json $SERVE ;;
  phi4-mini)
    docker run -d --name "$NAME" $COMMON -e VLLM_DISABLE_COMPILE_CACHE=1 $IMG \
      --model /models/phi4-mini-fp8 --served-model-name phi4-mini \
      --tokenizer /models/phi4-mini-base --max-model-len 32768 \
      --enable-auto-tool-choice --tool-call-parser phi4_mini_json $SERVE ;;
  phi4-mini-bf16)
    docker run -d --name "$NAME" $COMMON $IMG \
      --model /models/phi4-mini-base --served-model-name phi4-mini \
      --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser phi4_mini_json $SERVE ;;
  gpt-oss-20b)
    docker run -d --name "$NAME" $COMMON $FP4ENV -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 $IMG \
      --model /models/gpt-oss-20b --served-model-name gpt-oss-20b \
      --quantization mxfp4 --kv-cache-dtype fp8 --max-model-len 32768 $SERVE ;;
  ministral-8b)
    docker run -d --name "$NAME" $COMMON $IMG \
      --model /models/ministral-8b --served-model-name ministral-8b \
      --tokenizer-mode mistral --config-format mistral --load-format mistral \
      --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser mistral $SERVE ;;
  *) echo "unknown key: $KEY"; exit 2 ;;
esac

echo "[serve] $NAME launching on :$PORT (util $UTIL). Logs: docker logs -f $NAME"
# wait for /health up to 480s
for i in $(seq 1 96); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://localhost:${PORT}/health" 2>/dev/null)
  if [ "$code" = "200" ]; then echo "[serve] HEALTHY after $((i*5))s"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[serve] CONTAINER DIED. Last 40 log lines:"; docker logs --tail 40 "$NAME" 2>&1; exit 1
  fi
  sleep 5
done
echo "[serve] TIMEOUT waiting for health. Last 40 log lines:"; docker logs --tail 40 "$NAME" 2>&1; exit 1
