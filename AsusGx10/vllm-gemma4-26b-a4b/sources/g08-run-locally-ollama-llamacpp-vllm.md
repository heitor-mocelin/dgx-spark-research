---
id: g08
title: "How to Run Gemma 4 Locally with Ollama, llama.cpp, and vLLM"
url: "https://dev.to/dmaxdev/how-to-run-gemma-4-locally-with-ollama-llamacpp-and-vllm-3n44"
publisher: "DEV Community"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [local, ollama, llamacpp, vllm]
---

## TL;DR

Google Gemma 4 dropped on April 2 under Apache 2.0 and it's genuinely good: the 31B dense model hit #3 on the Arena AI leaderboard, beating models 20x its size. You can run it locally with Ollama in about two minutes, or go the llama.cpp / vLLM route if you want more control. But there are real bugs right now, especially on Apple Silicon and with tool calling. This guide covers all three options, what hardware you actually need, and the workarounds for the issues I've hit so far.

## Why Gemma 4 Is Worth Running Locally

I've been running local models since the Llama 2 days, and Gemma 4 is the first time an open model has made me reconsider whether I need API access to frontier models for everyday coding tasks.

Look at the benchmarks. Gemma 4 31B scores 89.2% on AIME 2026 (math), 80.0% on LiveCodeBench v6 (coding), and 84.3% on GPQA Diamond (science). Gemma 3 scored 20.8%, 29.1%, and 42.4% on those same tests. Every metric roughly tripled in one generation.

The family comes in four sizes:

| Model | Parameters | Active Params | Min VRAM (Q4) | Best For |
|---|---|---|---|---|
| E2B | 2.3B | 2.3B | ~1.5 GB | Mobile, Raspberry Pi |
| E4B | 4.5B | 4.5B | ~3 GB | Quick local tasks |
| 26B MoE | 26B | 3.8B | ~14 GB | Best bang per VRAM GB |
| 31B Dense | 31B | 31B | ~18 GB | Maximum quality |

The 26B MoE model is the sleeper hit here. It only activates 3.8B parameters per token but delivers reasoning quality close to the full 31B, and it fits in 14 GB of VRAM at Q4 quantization. If you're on a 16 GB GPU or a MacBook Pro with 18 GB unified memory, go with that one.

All four variants ship under Apache 2.0. No usage restrictions, no commercial limitations, no weird "you can't use this to compete with Google" clauses that plagued earlier open model releases. (If you're on a Mac and want to explore Apple's built-in local AI too, see my Apfel review — different beast, but it's free and already on your machine.)

## Option 1: Ollama (Easiest)

Ollama is the fastest way to get Gemma 4 running. Two commands and you're chatting.

### Install Ollama

On macOS:


```
brew install ollama
```


On Linux:


```
curl -fsSL https://ollama.com/install.sh | sh
```


On Windows, download the installer from ollama.com.

You need Ollama v0.20.0 or later for Gemma 4 support. Check with:


```
ollama --version
```


### Pull and Run a Model

```
# The 26B MoE — best quality-to-VRAM ratio
ollama run gemma4:26b
# The small but capable 4B
ollama run gemma4:4b
# The full 31B dense (need 20+ GB VRAM)
ollama run gemma4:31b
# Tiny model for edge devices
ollama run gemma4:2b
```


That's it. Ollama handles downloading the GGUF, quantization selection, and memory management automatically. By default it picks a quantization that fits your available memory.

### Pick Your Quantization

If you want more control over the quality/memory tradeoff:


```
# Higher quality, more memory
ollama run gemma4:26b-q8_0
# Lower memory, slightly less quality
ollama run gemma4:26b-q4_K_M
# Middle ground
ollama run gemma4:26b-q5_K_M
```


For the 31B model, Q4_K_M is the sweet spot. It keeps quality high while fitting in ~18 GB. Going to Q8 pushes you to ~28 GB, which means you need a 32 GB GPU or Mac with 32+ GB unified memory.

### Use the API

Ollama exposes an OpenAI-compatible API on port 11434:


```
curl http://localhost:11434/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "gemma4:26b",
"messages": [{"role": "user", "content": "Write a Python function to merge two sorted arrays"}]
}'
```


This works with any OpenAI SDK client. Just point the base URL to `http://localhost:11434/v1`

.


```
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:11434/v1",
api_key="ollama" # required but ignored
)
response = client.chat.completions.create(
model="gemma4:26b",
messages=[{"role": "user", "content": "Explain quicksort in 3 sentences"}]
)
print(response.choices[0].message.content)
```


### Known Ollama Issues (April 2026)

I'm flagging these because they burned me:

Tool calling is broken in Ollama v0.20.0. The tool call parser crashes, and streaming drops tool calls entirely. If you need function calling, use vLLM instead for now.

If you're on an M-series Mac, don't set

`OLLAMA_FLASH_ATTENTION=1`

. The 31B model will hang once your prompt exceeds ~500 tokens. Ollama's defaults work fine without it.Some general knowledge prompts cause the model to spit out an infinite stream of

`<unused24>`

tokens. Tokenizer bug. If it happens, stop generation and rephrase your prompt. A fix is being tracked in llama.cpp issue #21321.

## Option 2: llama.cpp (More Control)

If you want raw performance, custom quantization, or you're deploying on hardware Ollama doesn't support well, llama.cpp gives you full control.

### Build llama.cpp

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON # or -DGGML_METAL=ON for Mac
cmake --build build --config Release -j$(nproc)
```


For CPU-only (no GPU acceleration):


```
cmake -B build
cmake --build build --config Release -j$(nproc)
```


### Download a GGUF Model

Grab a pre-quantized model from Hugging Face. Unsloth provides well-tested GGUFs:


```
# 31B Q4_K_M — ~18 GB, good quality
huggingface-cli download unsloth/gemma-4-31B-it-GGUF \
gemma-4-31B-it-Q4_K_M.gguf \
--local-dir ./models
# 26B MoE Q4_K_M — ~14 GB
huggingface-cli download unsloth/gemma-4-26B-MoE-it-GGUF \
gemma-4-26B-MoE-it-Q4_K_M.gguf \
--local-dir ./models
```


### Run Inference

```
./build/bin/llama-cli \
-m ./models/gemma-4-31B-it-Q4_K_M.gguf \
-p "Write a Rust function that implements a thread-safe LRU cache" \
-n 512 \
-ngl 99 # offload all layers to GPU
```


The `-ngl 99`

flag offloads all layers to your GPU. If you don't have enough VRAM, lower this number and llama.cpp will split layers between GPU and CPU. For the 31B Q4 model, I'd start with `-ngl 40`

on a 16 GB GPU and adjust from there.

### Run as a Server

```
./build/bin/llama-server \
-m ./models/gemma-4-31B-it-Q4_K_M.gguf \
--host 0.0.0.0 \
--port 8080 \
-ngl 99 \
-c 8192
```


This gives you an OpenAI-compatible API at `http://localhost:8080/v1`

. Same client code as the Ollama example above, just change the port.

### Performance Tips for llama.cpp

- Gemma 4 advertises 256K context, but on consumer hardware you're realistically looking at ~20K tokens before memory pressure kills throughput. Qwen 3.5 27B manages ~190K on the same hardware, a 10x difference. Set
`-c`

conservatively. (Compression techniques like Google's TurboQuant may help here eventually.) - On Mac, use
`-DGGML_METAL=ON`

during build. Metal acceleration gives 2-3x speedup over CPU on M-series chips. - Increasing
`-b`

(batch size) can improve throughput for server workloads. I use`-b 512`

for my setup.

## Option 3: vLLM (Production Serving)

vLLM is the right choice if you're serving Gemma 4 to multiple users or building it into a production pipeline. It handles batching, paged attention, and continous batching automatically.

### Install and Run

The easiest path is Docker:


```
docker run --gpus all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
-p 8000:8000 \
vllm/vllm-openai:latest \
--model google/gemma-4-31b-it \
--max-model-len 8192 \
--gpu-memory-utilization 0.9
```


Or install directly:


```
pip install vllm>=0.20.0
vllm serve google/gemma-4-31b-it \
--max-model-len 8192 \
--gpu-memory-utilization 0.9
```


This starts an OpenAI-compatible API on port 8000.

### The vLLM Performance Bug

Fair warning: there's a known performance issue with Gemma 4 on vLLM right now. The E4B model generates at only ~9 tokens/s on an RTX 4090. That's terrible for a 4B parameter model.

The root cause is Gemma 4's hybrid attention architecture. It uses 50 sliding-window layers plus 10 global attention layers, each with different head dimensions. vLLM's FlashAttention implementation can't handle this dual-dimension layout, so it falls back to a much slower Triton attention kernel.

The vLLM team is tracking this in issue #38887. Until it's fixed, you'll get better throughput from llama.cpp for single-user workloads. vLLM still wins when you're serving multiple concurrent users because of its batching, but the per-request latency is worse than it should be.

### Multi-GPU Setup

For the 31B model on multiple GPUs:


```
vllm serve google/gemma-4-31b-it \
--tensor-parallel-size 2 \
--max-model-len 16384 \
--gpu-memory-utilization 0.9
```


Two 16 GB GPUs can serve the 31B model comfortably at BF16, which avoids any quantization quality loss.

## Which Model Should You Pick?

After a week of running all four variants, here's my take:

Most people should start with the 26B MoE. It activates only 3.8B parameters but delivers 82.3% on GPQA and 77.1% on LiveCodeBench. It fits on a single 16 GB GPU at Q4. For coding assistance, general Q&A, and document analysis, it handles all of those well.

The 31B dense is worth the VRAM if you have it. The jump from 26B MoE to 31B dense is noticeable on hard math and complex multi-step reasoning. If you have 24 GB VRAM (RTX 3090/4090) or 32+ GB unified memory on a Mac, run this one.

I reach for the E4B when I want speed. Quick code completions, simple questions where I want sub-second responses. At ~3 GB VRAM, it runs comfortably alongside everything else on my machine.

The E2B? It runs on a Raspberry Pi, which is cool, but the quality gap to E4B is too large for anything beyond simple tasks.

## Hardware Cheat Sheet

Here's what actually works based on my testing and community reports:

| Hardware | Best Model | Quantization | Tokens/s |
|---|---|---|---|
| RTX 4090 (24 GB) | 31B Dense | Q4_K_M | ~35 t/s |
| RTX 3090 (24 GB) | 31B Dense | Q4_K_M | ~25 t/s |
| RTX 4070 Ti (16 GB) | 26B MoE | Q4_K_M | ~30 t/s |
| Mac M3 Pro (18 GB) | 26B MoE | Q4_K_M | ~15 t/s |
| Mac M2 Ultra (64 GB) | 31B Dense | Q8_0 | ~20 t/s |
| RTX 3060 (12 GB) | E4B | Q8_0 | ~45 t/s |
| Raspberry Pi 5 (8 GB) | E2B | Q4 | ~3 t/s |

These numbers are from llama.cpp with full GPU offloading. Ollama performance is within 5-10% of these.

## Connecting Gemma 4 to Your Editor

Once you have a local Gemma 4 instance running (Ollama, llama.cpp server, or vLLM), you can use it as a coding assistant in most editors.

**VS Code with Continue:**


```
{
"models": [
{
"title": "Gemma 4 26B Local",
"provider": "ollama",
"model": "gemma4:26b"
}
]
}
```


**Neovim with avante.nvim or codecompanion.nvim:**

Point the OpenAI-compatible endpoint to your local server. Both plugins accept a custom base URL.

**Any tool that supports OpenAI API:**


```
Base URL: http://localhost:11434/v1 (Ollama)
Base URL: http://localhost:8080/v1 (llama.cpp)
Base URL: http://localhost:8000/v1 (vLLM)
API Key: "not-needed" (any string works)
Model: gemma4:26b
```


## FAQ

### How much VRAM do I need to run Gemma 4?

It depends on the model variant. The E2B runs in under 1.5 GB. The E4B needs about 3 GB at Q4. The 26B MoE needs ~14 GB at Q4. The 31B dense needs ~18 GB at Q4_K_M. On Macs, unified memory counts as VRAM, so a 16 GB MacBook can run the 26B MoE.

### Can I run Gemma 4 on CPU only?

Yes, but it's slow. llama.cpp supports CPU inference natively. Expect 2-5 tokens per second for the 26B model on a modern desktop CPU. The E4B at ~8-12 tokens per second on CPU is usable for simple tasks.

### Is Gemma 4 better than Llama 3 for coding?

On LiveCodeBench v6, Gemma 4 31B scores 80.0% versus Llama 3.3 70B's score in the low 60s. Gemma 4 is smaller and faster while producing better code. The 26B MoE at 77.1% also beats Llama 3.3 70B while using a fraction of the memory. And with Meta pivoting toward closed models with Muse Spark, Gemma 4 might be the best open alternative for a while.

### Does Gemma 4 support vision and audio?

The E2B and E4B variants support multimodal input: images and audio. The larger 26B and 31B models are text-only. If you need local vision capabilities, the E4B is your best option in the Gemma 4 family.

### Why is Gemma 4 tool calling broken in Ollama?

Gemma 4's hybrid attention architecture (mixing sliding-window and global attention layers with different head dimensions) exposed bugs in Ollama's tool call parser and streaming implementation. The Ollama team is working on a fix. For now, use vLLM or raw llama.cpp if you need function calling.

## Bottom Line

I've tried every major open model release since Llama 2, and Gemma 4's 26B MoE is the first one where I stopped reaching for API keys during normal coding work. 14 GB of VRAM, no license restrictions, and benchmark scores that would've been frontier-tier eighteen months ago. The tooling has rough edges right now. Tool calling in Ollama is broken, vLLM has a performance regression, and Apple Silicon users need to dodge a Flash Attention bug. Those will get fixed. The model quality won't go backwards. Start with `ollama run gemma4:26b`

and see where it gets you.

## Top comments (1)

I found Ollama likes to use a context window of 4096 by default. Because I'm working with opencode and larger development projects that is just not enough. I had to save a version of the Gemma4 model with a 32K context window explicitly (see dev.to/grovertek/running-gemma-4-l... for details). Then calling it via opencode worked as expected - other than the periodic tool call issues you mentioned.
