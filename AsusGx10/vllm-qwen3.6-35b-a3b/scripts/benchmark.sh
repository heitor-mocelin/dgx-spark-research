#!/usr/bin/env bash
#
# benchmark.sh — measure the running vLLM server with `vllm bench serve` (guide 05).
#
# Runs two passes against the live endpoint and writes JSON + a metadata sidecar per run:
#   1. single-stream  (--max-concurrency 1)        -> latency truth (TTFT, ITL/TPOT)
#   2. saturated      (--request-rate inf)         -> throughput truth (tok/s)
# Always pin/record the image + serve args so runs are comparable across configs.
#
# Usage:
#   ./benchmark.sh                                  # default 2048-in / 512-out, 500 prompts
#   INPUT_LEN=1024 OUTPUT_LEN=1024 NUM_PROMPTS=1000 ./benchmark.sh
#   LABEL=mnbt8192 ./benchmark.sh                   # tag the run (tune.sh sets this)
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
CONTAINER="${CONTAINER:-vllm}"
BENCH_EXEC="${BENCH_EXEC:-$DOCKER exec $CONTAINER}"   # use the container's vllm CLI
SERVED_NAME="${SERVED_NAME:-qwen36-moe}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"

DATASET="${DATASET:-random}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
PERCENTILES="${PERCENTILES:-50,99}"
LABEL="${LABEL:-run}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
STAMP="$(date +%Y%m%dT%H%M%S)"

log() { printf '[bench] %s\n' "$*" >&2; }
mkdir -p "$RESULTS_DIR"

# Record the exact stack under test (image + serve args) next to the results.
META="${RESULTS_DIR}/${STAMP}-${LABEL}.meta"
$DOCKER inspect --format '{{.Config.Image}}{{"\n"}}{{range .Args}}{{.}} {{end}}' "$CONTAINER" \
  > "$META" 2>/dev/null || log "warn: could not inspect $CONTAINER for metadata"
{ echo "stamp=$STAMP label=$LABEL input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS"; } >> "$META"

run_bench() {  # $1=mode-label  $2..=extra flags
  local mode="$1"; shift
  local out="${RESULTS_DIR}/${STAMP}-${LABEL}-${mode}.json"
  log "pass=${mode} input=${INPUT_LEN} output=${OUTPUT_LEN} prompts=${NUM_PROMPTS} -> ${out}"
  # shellcheck disable=SC2086
  $BENCH_EXEC vllm bench serve \
    --backend openai-chat --base-url "$BASE_URL" --endpoint /v1/chat/completions \
    --model "$SERVED_NAME" --dataset-name "$DATASET" \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --percentile-metrics "$PERCENTILES" \
    --save-result --result-filename "$out" "$@" \
    || log "warn: bench pass '${mode}' returned non-zero (check flags vs installed vLLM)"
}

# Warm-up note: discard the first run after a cold start before trusting numbers (guide 05).
run_bench single-stream --max-concurrency 1 --request-rate inf
run_bench saturated     --request-rate inf

log "done. JSON + .meta in ${RESULTS_DIR}/ (prefix ${STAMP}-${LABEL})."
if command -v jq >/dev/null 2>&1; then
  for f in "${RESULTS_DIR}/${STAMP}-${LABEL}"-*.json; do
    [ -f "$f" ] || continue
    log "$(basename "$f"): $(jq -rc '{tput:.output_throughput, ttft_p99:.p99_ttft_ms, itl_p99:.p99_itl_ms}' "$f" 2>/dev/null)"
  done
fi
