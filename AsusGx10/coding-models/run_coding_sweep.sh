#!/usr/bin/env bash
# Unattended coding-model bench-loop. One model at a time (OOM-safe):
#   teardown -> download(if absent) -> serve@native-ctx -> perf -> HumanEval(gen+score) -> ctx-sweep -> JSON.
# Honest "what runs on this GB10": each model uses whatever ready-made quant exists; format is in serve_coding.sh.
set -uo pipefail
CMDIR_H=/home/user/coding-bench/AsusGx10/coding-models      # host path (for bash serve_coding.sh)
CMDIR_C=/host/coding-bench/AsusGx10/coding-models           # container path (-v /home/user:/host)
RC_H=/home/user/results-coding
RC_C=/host/results-coding
IMG=vllm/vllm-openai:nightly
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# key | hf_repo | localdir | ctx_targets
MODELS=(
"qwen3-coder-30b|ig1/Qwen3-Coder-30B-A3B-Instruct-NVFP4|qwen3-coder-30b-a3b-nvfp4|1024,8192,32768,98304"
"qwen25-coder-32b|RedHatAI/Qwen2.5-Coder-32B-Instruct-FP8-dynamic|qwen25-coder-32b-fp8|1024,8192,24576"
"qwen25-coder-14b|RedHatAI/Qwen2.5-Coder-14B-Instruct-FP8-dynamic|qwen25-coder-14b-fp8|1024,8192,24576"
"qwen25-coder-7b|RedHatAI/Qwen2.5-Coder-7B-Instruct-FP8-dynamic|qwen25-coder-7b-fp8|1024,8192,24576"
"deepseek-coder-v2-lite|RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8|deepseek-coder-v2-lite-fp8|1024,8192,32768,98304"
"devstral-24b|Firworks/Devstral-Small-2-24B-Instruct-2512-nvfp4|devstral-24b-nvfp4|1024,8192,32768,98304"
"codestral-22b|TechxGenus/Codestral-22B-v0.1-FP8|codestral-22b-fp8|1024,8192,24576"
)

dl(){ local repo="$1" dir="$2"
  if [ -d "/home/user/models/$dir" ] && [ -n "$(ls -A /home/user/models/$dir 2>/dev/null)" ]; then
    log "  present: $dir"; return 0; fi
  log "  downloading $repo -> $dir"
  docker run --rm -v /home/user/models:/models --entrypoint bash $IMG \
    -c "hf download $repo --local-dir /models/$dir" >/dev/null 2>&1 || { log "  DOWNLOAD FAILED: $repo"; return 1; }
}
pyhost(){ docker run --rm --network host --entrypoint python3 -v /home/user:/host $IMG "$@"; }
pynone(){ docker run --rm --network none --entrypoint python3 -v /home/user:/host $IMG "$@"; }

for entry in "${MODELS[@]}"; do
  IFS='|' read -r key repo dir ctx <<< "$entry"
  log "===== $key ====="
  if [ -s "$RC_H/$key.he-score.json" ] && [ -s "$RC_H/$key.perf.json" ] && [ -s "$RC_H/$key.ctx.json" ]; then
    log "  $key already complete -> skip"; continue; fi
  docker ps -q --filter name=cm- | xargs -r docker rm -f >/dev/null 2>&1; sleep 3
  if ! dl "$repo" "$dir"; then printf '{"label":"%s","download_failed":true}\n' "$key" > "$RC_H/$key.perf.json"; continue; fi
  if bash "$CMDIR_H/serve_coding.sh" "$key" 8001 0.85; then
    log "  perf"
    pyhost "$CMDIR_C/bench_small.py" --base http://localhost:8001 --model "$key" --label "$key" --out "$RC_C/$key.perf.json" >/dev/null 2>&1 || log "  perf ERR"
    log "  humaneval"
    if pyhost "$CMDIR_C/humaneval.py" gen --base http://localhost:8001 --model "$key" --out "$RC_C/$key.he.json" --data "$RC_C/HumanEval.jsonl.gz" >/dev/null 2>&1; then
      pynone "$CMDIR_C/humaneval.py" score --inp "$RC_C/$key.he.json" --out "$RC_C/$key.he-score.json" --label "$key" >/dev/null 2>&1 || log "  score ERR"
    else log "  humaneval gen ERR"; fi
    log "  ctx-sweep"
    pyhost "$CMDIR_C/bench_ctx.py" --base http://localhost:8001 --model "$key" --label "$key" --ctx "$ctx" --out "$RC_C/$key.ctx.json" >/dev/null 2>&1 || log "  ctx ERR"
    log "  $key DONE"
  else
    log "  $key SERVE-FAILED"; printf '{"label":"%s","serve_failed":true}\n' "$key" > "$RC_H/$key.perf.json"
  fi
done
docker ps -q --filter name=cm- | xargs -r docker rm -f >/dev/null 2>&1
log "===== CODING SWEEP COMPLETE ====="
