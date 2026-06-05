---
id: 012
title: "vLLM Issue #33333: NVFP4 MoE backend FLASHINFER_CUTLASS unsupported on SM120 device"
url: "https://github.com/vllm-project/vllm/issues/33333"
publisher: "GitHub (vLLM)"
retrieved: "2026-06-05"
fetched_by: "openclaw-lxc/gh-cli"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [backends, moe, nvfp4, sm121, marlin]
---

author:	shahizat
association:	none
edited:	false
status:	none
--
The same issue occurred on the machine with a 5090 GPU.
--
author:	helperlings
association:	none
edited:	true
status:	none
--
I am facing the same issue. I installed vLLM with pip. Both version 1.14.1 as well as 1.15.0 are affected. I have a RTX Pro 6000 Workstation Edition.

AlmaLinux 10.1 
Kernel 6.12.0-124.27.1.el10_1.x86_64_v2
Driver Version: 590.48.01      
CUDA Version: 13.1
--
author:	eugr
association:	none
edited:	false
status:	none
--
Looks like this commit broke compatibility, as this model works with NVIDIA provided container that has an older vLLM version: https://github.com/vllm-project/vllm/commit/42135d689830c0e764d925b6454bc68ba6c6cab4#diff-3886f75aa77ead65142d91249abd431fb740df5242672c064bb8018dae15b171
--
author:	helperlings
association:	none
edited:	false
status:	none
--
I had a look at this commit and am gonna be honest that I do not have the slightest idea how to fix this myself.
--
author:	danisereb
association:	contributor
edited:	false
status:	none
--
Maybe it's related to this issue:
https://github.com/flashinfer-ai/flashinfer/issues/2077

And maybe this PR:
https://github.com/vllm-project/vllm/pull/33417
--
author:	pcgeek86
association:	none
edited:	false
status:	none
--
I just reported this same exact issue on the vLLM forums, with a bunch of system details. I have a dual-GPU setup with an RTX 5080 / 5070 Ti totaling 32 GB of VRAM. This should be enough to run the Nemotron 3 Nano ▶️ NVFP4 ◀️  model. It appears the `.safetensors` files are only about 20 GB in total. I'm trying to use Docker Desktop (WSL) on Windows 11.

I was excited to try running this model, now that I've got a couple of Blackwell GPUs running in the same system. Looks like there is still more work to be done.

https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4

https://discuss.vllm.ai/t/running-nvfp4-nemotron-model-on-win11-wsl-rtx-5080-5070-ti/2310
--
author:	pcgeek86
association:	none
edited:	false
status:	none
--
> I had a look at this commit and am gonna be honest that I do not have the slightest idea how to fix this myself.

Same here, there is _way too much_ going on in that commit for me to even remotely know where to start looking.

@eugr do you know which NVIDIA container image currently works with Nemotron 3 Nano NVFP4? I would love to try it out, even if it's a slightly older version of vLLM.
--
author:	pcgeek86
association:	none
edited:	false
status:	none
--
I got it working with NVIDIA's latest vLLM container image on the NVIDIA container registry.

```
docker run --gpus all `
    --rm `
    -v "C:\git\NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4:/model" `
    -p 8000:8000 `
    --env "VLLM_USE_FLASHINFER_MOE_FP4=1" `
    --env "VLLM_FLASHINFER_MOE_BACKEND=throughput" `
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID `
    --env "CUDA_VISIBLE_DEVICES=0,1" `
    --gpus=all `
    --ipc=host `
    nvcr.io/nvidia/vllm:26.01-py3 `
    vllm serve `
    --model /model `
    --served-model-name nemotron `
    --max-model-len 131072 `
    --max-num-seqs 8 `
    --kv-cache-dtype fp8 `
    --trust-remote-code `
    --reasoning-parser-plugin "/model/nano_v3_reasoning_parser.py" `
    --reasoning-parser nano_v3 `
    --tensor-parallel-size 2
```

Loading the model and compiling **takes several minutes**, so be patient. Look at the timestamps in my logs below, and you'll see loading the `safetensors` files takes a while, and `torch.compile` took over 5 minutes on my system.

```
(Worker_TP0 pid=271) INFO 02-02 13:17:21 [gpu_model_runner.py:3562] Starting to load model /model...
(Worker_TP0 pid=271) INFO 02-02 13:17:21 [modelopt.py:982] Using flashinfer-cutlass for NVFP4 GEMM
(Worker_TP0 pid=271) INFO 02-02 13:17:21 [layer.py:372] Enabled separate cuda stream for MoE shared_experts
(Worker_TP1 pid=272) INFO 02-02 13:17:21 [modelopt.py:982] Using flashinfer-cutlass for NVFP4 GEMM
(Worker_TP1 pid=272) INFO 02-02 13:17:33 [nvfp4_moe_support.py:42] Using FlashInfer kernels for ModelOptNvFp4FusedMoE.
(Worker_TP1 pid=272) INFO 02-02 13:17:33 [modelopt.py:1191] Using FlashInfer CUTLASS kernels for ModelOptNvFp4FusedMoE.
(Worker_TP0 pid=271) INFO 02-02 13:17:33 [nvfp4_moe_support.py:42] Using FlashInfer kernels for ModelOptNvFp4FusedMoE.
(Worker_TP0 pid=271) INFO 02-02 13:17:33 [modelopt.py:1191] Using FlashInfer CUTLASS kernels for ModelOptNvFp4FusedMoE.
(Worker_TP0 pid=271) INFO 02-02 13:17:33 [cuda.py:351] Using FLASHINFER attention backend out of potential backends: ('FLASHINFER', 'TRITON_ATTN')
Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:17<01:10, 17.52s/it]
Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:37<00:57, 19.22s/it]
Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:58<00:39, 19.71s/it]
Loading safetensors checkpoint shards:  80% Completed | 4/5 [01:18<00:19, 19.86s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [01:35<00:00, 18.80s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [01:35<00:00, 19.05s/it]
(Worker_TP0 pid=271)
(Worker_TP0 pid=271) INFO 02-02 13:19:09 [default_loader.py:308] Loading weights took 95.26 seconds
(Worker_TP0 pid=271) INFO 02-02 13:19:10 [gpu_model_runner.py:3659] Model loading took 9.8291 GiB memory and 108.345254 seconds
(Worker_TP0 pid=271) WARNING 02-02 13:19:10 [vllm.py:1403] Current vLLM config is not set.
(Worker_TP0 pid=271) INFO 02-02 13:19:10 [scheduler.py:230] Chunked prefill is enabled with max_num_batched_tokens=2048.
(Worker_TP1 pid=272) WARNING 02-02 13:19:32 [vllm.py:1403] Current vLLM config is not set.
(Worker_TP1 pid=272) INFO 02-02 13:19:32 [scheduler.py:230] Chunked prefill is enabled with max_num_batched_tokens=2048.
(Worker_TP0 pid=271) INFO 02-02 13:19:36 [backends.py:643] Using cache directory: /root/.cache/vllm/torch_compile_cache/90ba35ecfd/rank_0_0/backbone for vLLM's torch.compile
(Worker_TP0 pid=271) INFO 02-02 13:19:36 [backends.py:703] Dynamo bytecode transform time: 3.99 s
(Worker_TP0 pid=271) INFO 02-02 13:19:38 [backends.py:261] Cache the graph of compile range (1, 2048) for later use
(Worker_TP1 pid=272) INFO 02-02 13:19:44 [backends.py:261] Cache the graph of compile range (1, 2048) for later use
(Worker_TP1 pid=272) /usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py:321: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
(Worker_TP1 pid=272)   warnings.warn(
(Worker_TP0 pid=271) /usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py:321: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
(Worker_TP0 pid=271)   warnings.warn(
(EngineCore_DP0 pid=214) INFO 02-02 13:20:32 [shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(EngineCore_DP0 pid=214) INFO 02-02 13:21:32 [shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(EngineCore_DP0 pid=214) INFO 02-02 13:22:32 [shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(EngineCore_DP0 pid=214) INFO 02-02 13:23:32 [shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(EngineCore_DP0 pid=214) INFO 02-02 13:24:32 [shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(Worker_TP0 pid=271) INFO 02-02 13:25:03 [backends.py:278] Compiling a graph for compile range (1, 2048) takes 325.75 s
(Worker_TP0 pid=271) INFO 02-02 13:25:03 [monitor.py:34] torch.compile takes 329.74 s in total
(Worker_TP0 pid=271) INFO 02-02 13:25:04 [gpu_worker.py:375] Available KV cache memory: 4.02 GiB
(EngineCore_DP0 pid=214) WARNING 02-02 13:25:05 [kv_cache_utils.py:1033] Add 1 padding layers, may waste at most 4.35% KV cache memory
(EngineCore_DP0 pid=214) INFO 02-02 13:25:05 [kv_cache_utils.py:1291] GPU KV cache size: 183,744 tokens
(EngineCore_DP0 pid=214) INFO 02-02 13:25:05 [kv_cache_utils.py:1296] Maximum concurrency for 131,072 tokens per request: 6.17x
(Worker_TP1 pid=272) 2026-02-02 13:25:05,231 - INFO - autotuner.py:256 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP0 pid=271) 2026-02-02 13:25:05,231 - INFO - autotuner.py:256 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
```

Here's what it looks like while loading the `.safetensors` model shard files:

<img width="1734" height="927" alt="Image" src="https://github.com/user-attachments/assets/01171744-478e-40d6-9f60-43fce3f69966" />

Here's what it looks like when everything is loaded and ready to go:

<img width="2222" height="1711" alt="Image" src="https://github.com/user-attachments/assets/439187a2-d788-4081-be8c-caf4f9e6ddb0" />

Here's what it looks like running inference against the model:

```
(Worker_TP0 pid=271) 2026-02-02 13:25:11,802 - INFO - autotuner.py:262 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP1 pid=272) 2026-02-02 13:25:11,804 - INFO - autotuner.py:262 - flashinfer.jit: [Autotuner]: Autotuning process ends
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 5/5 [00:00<00:00,  8.66it/s]
Capturing CUDA graphs (decode, FULL): 100%|██████████| 4/4 [00:13<00:00,  3.32s/it]
(Worker_TP0 pid=271) INFO 02-02 13:25:26 [gpu_model_runner.py:4587] Graph capturing finished in 15 secs, took 0.04 GiB
(EngineCore_DP0 pid=214) INFO 02-02 13:25:26 [core.py:259] init engine (profile, create kv cache, warmup model) took 354.15 seconds
(EngineCore_DP0 pid=214) WARNING 02-02 13:25:27 [interface.py:465] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
(APIServer pid=1) INFO 02-02 13:25:27 [api_server.py:1102] Supported tasks: ['generate']
(APIServer pid=1) INFO 02-02 13:25:27 [api_server.py:1428] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:38] Available routes are:
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /pause, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /resume, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /is_paused, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/audio/translations, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /classify, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/embeddings, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /score, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/score, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /rerank, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v1/rerank, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /v2/rerank, Methods: POST
(APIServer pid=1) INFO 02-02 13:25:27 [launcher.py:46] Route: /pooling, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
(APIServer pid=1) INFO 02-02 13:28:29 [chat_utils.py:590] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=1) INFO:     127.0.0.1:55092 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1) INFO 02-02 13:28:58 [loggers.py:248] Engine 000: Avg prompt throughput: 3.9 tokens/s, Avg generation throughput: 32.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO:     127.0.0.1:47910 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1) INFO 02-02 13:29:08 [loggers.py:248] Engine 000: Avg prompt throughput: 3.9 tokens/s, Avg generation throughput: 33.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 02-02 13:29:18 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%

```

<img width="2208" height="344" alt="Image" src="https://github.com/user-attachments/assets/e19438f6-057e-4494-acc9-d6b90760aa42" />

Have fun!
--
author:	robertgshaw2-redhat
association:	collaborator
edited:	true
status:	none
--
apologies for the breakage in 0.15, this was related to a mistake I made in some refactoring. We have fixed the issue here (https://github.com/vllm-project/vllm/pull/33417) and will be doing a patch release 0.15.1 in the coming days after we fix some other issues with 0.15.0

in the meantime, the nightly images and wheels should be good to go. you can see how to install these in our docs

thanks!
--
author:	robertgshaw2-redhat
association:	collaborator
edited:	false
status:	none
--
please reopen the issue if you have any additional problems. thanks for using vLLM!
--
author:	shahizat
association:	none
edited:	false
status:	none
--
@robertgshaw2-redhat np at all, thanks for your help and involvement. @eugr, @johnnynunez, and I were aware that Nvidia NGC containers `nvcr.io/nvidia/vllm:26.01-py3 `and `nvcr.io/nvidia/vllm:25.12.post1-py3` were working fine with Nemotron-3-Nano-30B-A3B-NVFP4. Happy to see the fix in the main branch as well.
--
author:	eugr
association:	none
edited:	false
status:	none
--
@shahizat - actually NGC containers were working because the breakage was introduced in 0.15.0, so no fix was needed for those as they are 0.13.0 and below. 
--
author:	shahizat
association:	none
edited:	false
status:	none
--
@eugr yes, I know. The NGC containers use an outdated version of vLLM, which is why they were working. NVIDIA is making an effort to deliver stable containers through NGC for edge devices like Jetson Thor and DGX Spark, rather than relying on the latest and greatest releases of vllm. This is simply my personal choice to build everything from source, others may not need the latest updates from the main branch, which is also understandable. 
--
author:	johnnynunez
association:	contributor
edited:	false
status:	none
--
> apologies for the breakage in 0.15, this was related to a mistake I made in some refactoring. We have fixed the issue here ([#33417](https://github.com/vllm-project/vllm/pull/33417)) and will be doing a patch release 0.15.1 in the coming days after we fix some other issues with 0.15.0
> 
> in the meantime, the nightly images and wheels should be good to go. you can see how to install these in our docs
> 
> thanks!

is there nightly builds with cu130?
--
author:	shahizat
association:	none
edited:	false
status:	none
--
```
uv venv .vllm --python 3.12
source .vllm/bin/activate
 
uv pip install -U vllm \
    --torch-backend=cu130 \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130
```
Output:

` ValueError: NvFp4 MoE backend 'FLASHINFER_CUTLASS' does not support the deployment configuration since kernel does not support current device.`
--
author:	johnnynunez
association:	contributor
edited:	false
status:	none
--
> ```
> uv venv .vllm --python 3.12
> source .vllm/bin/activate
>  
> uv pip install -U vllm \
>     --torch-backend=cu130 \
>     --extra-index-url https://wheels.vllm.ai/nightly/cu130
> ```
> 
> Output:
> 
> ` ValueError: NvFp4 MoE backend 'FLASHINFER_CUTLASS' does not support the deployment configuration since kernel does not support current device.`

it should be fixed: https://github.com/vllm-project/vllm/blob/bcd2f74c0d1e85a2da4dcb41849ad75a7e3fdaf4/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py#L86
--
author:	johnnynunez
association:	contributor
edited:	false
status:	none
--
with 0.15.rc1 is working
--
author:	pcgeek86
association:	none
edited:	false
status:	none
--
vLLM 0.15.1 is working great for me. I'm getting 75 tokens per second, roughly speaking. I posted all of my details, similar to earlier in this thread, over in the NVIDIA Nemotron forums.

https://forums.developer.nvidia.com/t/nvidia-nemotron-3-nano-nvfp4-extremely-slow-on-dual-blackwell-32gb-vram-system/359970/2

There might still be a problem with NVIDIA's container image, but I have not tested it again. At the moment, I am unblocked using `vLLM:latest`.

```pwsh
docker run `
    --gpus all `
    -v "C:\git\NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4:/model" `
    -p 8000:8000 `
    --env "VLLM_USE_FLASHINFER_MOE_FP4=1" `
    --env "VLLM_FLASHINFER_MOE_BACKEND=throughput" `
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID `
    --env "CUDA_VISIBLE_DEVICES=0,1" `
    --ipc=host `
    vllm/vllm-openai:latest `
    /model `
    --served-model-name nemotron `
    --max-model-len 10000 `
    --max-num-seqs 8 `
    --kv-cache-dtype fp8 `
    --trust-remote-code `
    --reasoning-parser-plugin "/model/nano_v3_reasoning_parser.py" `
    --reasoning-parser nano_v3 `
    --tensor-parallel-size 2
```
--
author:	eugr
association:	none
edited:	false
status:	none
--
@pcgeek86 - 75 t/s seems to be very slow for dedicated GPU and a model with 3B active parameters. I'm getting 83 t/s on AWQ quants with the original Qwen3-30B-A3B on a single DGX Spark which has much slower memory throughput than your dGPUs...
--
author:	pcgeek86
association:	none
edited:	false
status:	none
--
Hmmmm, what would you expect to get with a dual-GPU setup like this running the NVFP4 model? This is the first time I'm experimenting with it, so I am just figuring it out and seeing what I get.
--
author:	eugr
association:	none
edited:	false
status:	none
--
Probably around 200 t/s... You can try to run this one for comparison: QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ
--
author:	eugr
association:	none
edited:	false
status:	none
--
@pcgeek86 - for example, this is what I'm getting on my single RTX4090:

| model                                      |            test |                t/s |       peak t/s |   peak t/s (req) |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------------------------------|----------------:|-------------------:|---------------:|-----------------:|----------------:|----------------:|----------------:|
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit |          pp2048 | 14615.15 ± 2240.97 |                |                  |  148.34 ± 24.68 |  143.89 ± 24.68 |  148.44 ± 24.67 |
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit |            tg32 |      197.36 ± 1.92 |  203.85 ± 1.97 |    203.85 ± 1.97 |                 |                 |                 |
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit |  pp2048 @ d4096 |   13766.84 ± 53.04 |                |                  |   450.75 ± 1.72 |   446.30 ± 1.72 |   450.85 ± 1.72 |
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit |    tg32 @ d4096 |      187.39 ± 1.46 |  193.61 ± 1.49 |    193.61 ± 1.49 |                 |                 |                 |
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit |  pp2048 @ d8192 |  11641.60 ± 277.23 |                |                  |  884.56 ± 21.32 |  880.11 ± 21.32 |  884.65 ± 21.33 |
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit |    tg32 @ d8192 |     196.84 ± 23.46 | 203.38 ± 24.30 |   203.38 ± 24.30 |                 |                 |                 |
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit | pp2048 @ d12000 |  10311.30 ± 115.90 |                |                  | 1367.01 ± 15.43 | 1362.56 ± 15.43 | 1367.09 ± 15.43 |
| cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit |   tg32 @ d12000 |     187.73 ± 17.55 | 193.92 ± 18.13 |   193.92 ± 18.13 |                 |                 |                 |

llama-benchy (0.3.0)
date: 2026-02-07 18:03:01 | latency mode: api
--
