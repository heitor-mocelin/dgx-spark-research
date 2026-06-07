#!/usr/bin/env bash
# Standalone sweep over the remaining small models. One model at a time on :8001 (OOM-safe).
# Per-model: teardown -> serve(health wait) -> correctness-gated benchmark -> JSON. Logs each step.
set -uo pipefail
mkdir -p /home/user/results
IMG=vllm/vllm-openai:cu130-nightly

teardown(){ docker ps -q --filter name=sm- | xargs -r docker rm -f >/dev/null 2>&1; sleep 3; }
BENCH(){ # key served extra
  docker run --rm --network host --entrypoint python3 -v /home/user:/host "$IMG" \
    /host/bench_small.py --base http://localhost:8001 --model "$2" --label "$1" \
    --out /host/results/"$1".json ${3:+--extra-body "$3"} 2>/dev/null
}
run_model(){ # key served extra
  local key="$1" served="$2" extra="${3:-}"
  echo "[$(date -u +%H:%M:%S)] ===== $key ====="
  teardown
  if bash /home/user/serve_small.sh "$key" 8001 0.85; then
    BENCH "$key" "$served" "$extra"
    echo "[$(date -u +%H:%M:%S)] $key DONE"
    return 0
  else
    echo "[$(date -u +%H:%M:%S)] $key SERVE-FAILED"
    printf '{"label":"%s","serve_failed":true}\n' "$key" > /home/user/results/"$key".json
    return 1
  fi
}

run_model qwen3-4b    qwen3-4b   ""
run_model llama31-8b  llama31-8b ""
run_model qwen3-8b    qwen3-8b   '{"chat_template_kwargs":{"enable_thinking":false}}'
# Phi: FP8 first, fall back to BF16 if the sm_121 FP8 kernel bug bites
if ! run_model phi4-mini phi4-mini ""; then
  echo "[$(date -u +%H:%M:%S)] phi4-mini FP8 failed -> trying BF16"
  run_model phi4-mini-bf16 phi4-mini ""
fi
run_model gpt-oss-20b  gpt-oss-20b ""
run_model ministral-8b ministral-8b ""
teardown
echo "[$(date -u +%H:%M:%S)] ===== SWEEP COMPLETE ====="
