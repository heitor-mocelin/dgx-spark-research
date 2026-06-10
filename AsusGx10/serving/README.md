# Serving multiple models on one DGX Spark (GB10) + routing by model name

How to run a **generalist** model and a **coding** model (plus a small auxiliary) **resident at the same time**
on a single GB10 Spark, load-balance each across multiple Sparks, and have an agent framework use the coder
**only when needed** — without paying a cold-start penalty per task.

> Hosts are written `spark-1` / `spark-2` (your inference boxes) and `router-host` (where the router +
> agent run). Replace them with your IPs/DNS names. Served model names (`qwen36-moe`, `qwen3-coder-30b`,
> `qwen3-4b`) are examples — use your own.

## Why "both resident", not "load on demand"

vLLM serves **one model per process**, and a GB10 cold start is **~5–6 min** (weights + `torch.compile`).
So "load the coder only when a coding request arrives" means a 5-minute stall on that request — unworkable.
And vLLM sleep/wake gives little on the Spark's **unified** memory (no separate CPU RAM to park weights in).

**The workable pattern:** keep every model resident, each in its own container on its own port, and **route by
model name** at a thin proxy. "Used when needed" becomes "always loaded, selected per request" — instant.

```
                       router (LiteLLM) ── model name ──► the right container
 agent  ──qwen36-moe──►  generalist 35B   :8000   (LB across spark-1, spark-2)
        ──qwen3-coder──►  coder 30B        :8002   (LB across spark-1, spark-2)
        ──qwen3-4b────►  aux 4B           :8001   (LB across spark-1, spark-2)
```

## Memory budgeting (the part that bites)

GB10 has ~121 GiB **unified** memory. Each vLLM container reserves `--gpu-memory-utilization × total` for
**weights + a fixed KV-cache pool**. The util values across all containers must sum to leave ~15+ GiB of host
headroom. `nvidia-smi` memory queries return **N/A** on this unified-memory part — read the real pool from each
container's startup log instead:

```
GPU KV cache size: 198,640 tokens
Maximum concurrency for 131,072 tokens per request: 1.52x
```

A worked three-model layout that fits (~17 GiB headroom):

| Container | Model | util | Notes |
|---|---|---|---|
| `vllm` :8000 | generalist 35B MoE (NVFP4) | **0.40** | ~20 GiB weights + big KV pool |
| `vllm-coder` :8002 | coder 30B MoE (NVFP4) | **0.22** | ~17 GiB weights; see the util floor below |
| `sm-q4` :8001 | aux 4B | **0.15** + `--kv-cache-dtype fp8` | slimmed from 0.35 to free ~24 GiB |

### Gotchas we hit (each cost real time)

1. **Coder util has a hard floor for long context.** At `--gpu-memory-utilization 0.20` with
   `--max-model-len 131072`, the coder's KV pool (4.56 GiB) **can't hold even one full-context request**
   (needs 6.0 GiB) and vLLM **refuses to start**:
   `ValueError: ... 6.0 GiB KV cache is needed, which is larger than the available KV cache memory (4.56 GiB)`.
   Fix: raise util (0.22 worked) or lower `--max-model-len`. Rule: **KV pool must be ≥ one max-len sequence.**
2. **A small aux model is easy to massively over-provision.** Our 4B at util 0.35 held a ~42 GiB KV pool
   (271k tokens) it never needed. Dropping it to **util 0.15 + `--kv-cache-dtype fp8`** freed ~24 GiB — enough
   to fund the entire coder. Right-size aux models first; that's usually where the headroom is.
3. **`--restart unless-stopped` hides the real crash.** A container that fails KV allocation will silently
   **crash-loop** (we saw `RestartCount: 21`), and you only see the generic "Engine core init failed". Bring a
   new model up with **`--restart no` first**, confirm `/health`, *then* `docker update --restart unless-stopped`.
4. **Don't force `--quantization`.** NVFP4 from `llm-compressor` declares `compressed-tensors` in its config;
   forcing `--quantization modelopt` errors with a method mismatch. **Omit it** and let vLLM auto-detect.

### Adding the coder (genericized run command)

```bash
docker run -d --name vllm-coder --restart no --runtime=nvidia --gpus all --shm-size=16g -p 8002:8000 \
  -v /home/user/models:/models \
  -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FP8_MOE_BACKEND=flashinfer_cutlass \
  vllm/vllm-openai:nightly \
  --model /models/qwen3-coder-30b-a3b-nvfp4 --served-model-name qwen3-coder-30b \
  --kv-cache-dtype fp8 --max-model-len 131072 --gpu-memory-utilization 0.22 \
  --max-num-seqs 128 --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
# wait for /health, then: docker update --restart unless-stopped vllm-coder
```

### Slimming the aux model to make room

Recreate it with a lower util + fp8 KV (preserve the old one as a rollback):

```bash
docker stop sm-q4 && docker rename sm-q4 sm-q4_pre-coder
docker run -d --name sm-q4 ... --gpu-memory-utilization 0.15 --kv-cache-dtype fp8 ...   # else identical
```

## Routing by model name (LiteLLM)

A thin **LiteLLM** proxy presents **one endpoint** and routes by the request's `model` field; listing each model
twice (once per Spark) makes it **load-balance with failover**. See [`litellm-config.example.yaml`](./litellm-config.example.yaml).

Install (any host with Python; no Docker needed):

```bash
python3 -m venv /opt/litellm && /opt/litellm/bin/pip install 'litellm[proxy]'
/opt/litellm/bin/litellm --config /etc/litellm/config.yaml --host 0.0.0.0 --port 4000   # systemd-ize this
```

LiteLLM passes through **tool calls** and the generalist's **`reasoning` field** unchanged — both verified.
`routing_strategy: simple-shuffle` balances the two backends per model; `cooldown_time` parks a dead Spark and
shifts traffic to the other (the redundancy you don't get pointing an agent at one box directly).

## Agent integration: use the coder *when needed*

The router makes every model reachable by name; the agent still has to **choose** the coder for coding work.
Two clean patterns:

- **Delegation / sub-agents.** If your agent framework supports a separate model for delegated sub-tasks, point
  that at the coder. Then the generalist stays the orchestrator and **spawns coder-powered sub-agents** for
  coding. (Example: the Nous **Hermes** agent exposes `delegation.model` + `orchestrator_enabled: true` —
  set `delegation.model` to the coder's name and `base_url` to the router. Note its `subagent_auto_approve`
  flag gates whether spawns need manual approval.)
- **A dedicated coding persona/sub-agent** bound to the coder model, invoked explicitly for dev work.

Verify routing by tailing the router's logs while you give the agent a coding task — you should see the coder's
model name being hit.

## Rollback discipline

Never `docker rm` a working model to change it — **`stop` + `rename` to `*_pre-*`**, run the new one, validate,
then remove the backup. Applies to the coder, the slimmed aux, and the agent config (keep a `config.yaml.bak-*`).
