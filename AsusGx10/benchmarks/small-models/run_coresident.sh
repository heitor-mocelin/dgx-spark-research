#!/usr/bin/env bash
# Co-residency pass: each working small model on :8001 (util 0.25) WHILE production Qwen runs on :8000.
# Validates memory coexistence + clean co-resident perf; plus an under-load interference test on the winner.
set -uo pipefail
IMG=vllm/vllm-openai:cu130-nightly
COMMON="--gpus all --runtime nvidia --ipc host --shm-size 16g -p 8001:8000 \
  -v /home/user/models:/models -v /home/user/.cache/huggingface:/root/.cache/huggingface"
FP4ENV="-e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_DISABLE_VERSION_CHECK=1"
U=0.25
teardown(){ docker ps -aq --filter name=sm- | xargs -r docker rm -f >/dev/null 2>&1; sleep 3; }
MEM(){ free -g | awk '/Mem:/{printf "[mem] %dG used / %dG total\n",$3,$2}'; }
wait_health(){ for i in $(seq 1 96); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://localhost:8001/health 2>/dev/null)" = "200" ] && { echo "[ok] $1 healthy $((i*5))s"; return 0; }
    docker ps --format '{{.Names}}'|grep -q "^$1$" || { echo "[DIED] $1"; docker logs --tail 20 "$1" 2>&1; return 1; }
    sleep 5; done; echo "[TIMEOUT] $1"; return 1; }
BENCH(){ docker run --rm --network host --entrypoint python3 -v /home/user:/host "$IMG" \
  /host/bench_small.py --base http://localhost:8001 --model "$2" --label "$1" --out /host/results/"$3" ${4:+--extra-body "$4"} 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print('   TTFT',d.get('ttft_ms',{}).get('mean'),'ms  decode',d.get('decode_tps',{}).get('mean'),'tok/s  smoke',d.get('smoke',{}).get('ok'))" 2>/dev/null; }

echo "[$(date -u +%H:%M:%S)] waiting for Qwen :8000 cold start (up to ~10min)..."
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://localhost:8000/health 2>/dev/null)" = "200" ] && { echo "[ok] Qwen healthy after $((i*5))s"; break; }
  sleep 5
done
curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -q qwen36-moe || { echo "QWEN NOT UP — abort"; exit 1; }
echo "[$(date -u +%H:%M:%S)] Qwen present on :8000. Starting co-resident pass (small models @ util $U)."

run(){ # label name modelpath served serveflags resultfile extrabody
  echo "[$(date -u +%H:%M:%S)] === co: $1 ==="; teardown
  docker run -d --name "$2" $COMMON $FP4ENV $IMG --model "$3" --served-model-name "$4" \
    --max-model-len 16384 --gpu-memory-utilization $U --max-num-seqs 64 $5 >/dev/null
  if wait_health "$2"; then MEM; BENCH "$1" "$4" "$6" "${7:-}"; else
    printf '{"label":"%s","coresident":true,"serve_failed":true}\n' "$1" > /home/user/results/"$6"; fi
}

run gemma-4-e4b  sm-gemma /models/gemma4-e4b-nvfp4a16     gemma-4-e4b  "--quantization compressed-tensors --kv-cache-dtype fp8" gemma-4-e4b.coresident.json
run qwen3-4b     sm-q4    /models/qwen3-4b-instruct-nvfp4  qwen3-4b     "--quantization compressed-tensors" qwen3-4b.coresident.json

# --- interference test on the winner (qwen3-4b still up as sm-q4) ---
echo "[$(date -u +%H:%M:%S)] === co+LOAD: qwen3-4b while Qwen is hammered ==="
( end=$((SECONDS+150)); while [ $SECONDS -lt $end ]; do
    for j in $(seq 1 8); do curl -s --max-time 40 http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen36-moe","messages":[{"role":"user","content":"Write a long technical essay about computer networking."}],"max_tokens":256}' >/dev/null & done; wait
  done ) & FLOOD=$!
sleep 3; MEM
BENCH qwen3-4b-underload qwen3-4b qwen3-4b.coresident-underload.json
kill $FLOOD 2>/dev/null; wait 2>/dev/null

run qwen3-8b     sm-q8    /models/qwen3-8b-nvfp4           qwen3-8b     "--quantization modelopt" qwen3-8b.coresident.json '{"chat_template_kwargs":{"enable_thinking":false}}'
run llama31-8b   sm-l8    /models/llama31-8b-nvfp4         llama31-8b   "" llama31-8b.coresident.json
run phi4-mini    sm-phi   /models/phi4-mini-base          phi4-mini    "" phi4-mini.coresident.json
run ministral-8b sm-min   /models/ministral-8b            ministral-8b "--tokenizer-mode mistral --config-format mistral --load-format mistral" ministral-8b.coresident.json
teardown
echo "[$(date -u +%H:%M:%S)] === CORESIDENT SWEEP COMPLETE ==="