#!/usr/bin/env bash
# run_one.sh KEY [CTX] [EXTRA_SERVE_ARGS...] — full battery for a single already-downloaded model.
set -uo pipefail
KEY="$1"; CTX="${2:-1024,8192,32768,98304}"
CMDIR_H=/home/user/coding-bench/AsusGx10/coding-models
CMDIR_C=/host/coding-bench/AsusGx10/coding-models
RC_C=/host/results-coding
IMG=vllm/vllm-openai:nightly
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
pyhost(){ docker run --rm --network host --entrypoint python3 -v /home/user:/host $IMG "$@"; }
pynone(){ docker run --rm --network none --entrypoint python3 -v /home/user:/host $IMG "$@"; }

docker ps -q --filter name=cm- | xargs -r docker rm -f >/dev/null 2>&1; sleep 5   # ensure GPU released
log "===== $KEY ====="
if bash "$CMDIR_H/serve_coding.sh" "$KEY" 8001 0.85; then
  log "perf";       pyhost "$CMDIR_C/bench_small.py" --base http://localhost:8001 --model "$KEY" --label "$KEY" --out "$RC_C/$KEY.perf.json" >/dev/null 2>&1 || log "perf ERR"
  log "humaneval";  if pyhost "$CMDIR_C/humaneval.py" gen --base http://localhost:8001 --model "$KEY" --out "$RC_C/$KEY.he.json" --data "$RC_C/HumanEval.jsonl.gz" >/dev/null 2>&1; then
                      pynone "$CMDIR_C/humaneval.py" score --inp "$RC_C/$KEY.he.json" --out "$RC_C/$KEY.he-score.json" --label "$KEY" >/dev/null 2>&1 || log "score ERR"
                    else log "humaneval gen ERR"; fi
  log "ctx";        pyhost "$CMDIR_C/bench_ctx.py" --base http://localhost:8001 --model "$KEY" --label "$KEY" --ctx "$CTX" --out "$RC_C/$KEY.ctx.json" >/dev/null 2>&1 || log "ctx ERR"
  docker rm -f "cm-$KEY" >/dev/null 2>&1
  log "$KEY DONE"
else
  log "$KEY SERVE-FAILED"
fi
