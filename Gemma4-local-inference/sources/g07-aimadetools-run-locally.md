---
id: g07
title: "How to Run Gemma 4 Locally — Complete Setup Guide (2026)"
url: "https://www.aimadetools.com/blog/how-to-run-gemma-4-locally/"
publisher: "aimadetools (blog)"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only"
topics: [local, ollama, llamacpp, vllm]
---

Google’s Gemma 4 family includes four models that run on everything from a Raspberry Pi to a single GPU. This guide covers three ways to run them locally: Ollama (easiest), llama.cpp (most control), and vLLM (best for serving).

## Before you start: pick your model

| Model | Min RAM (Q4) | Min VRAM | Speed | Quality |
|---|---|---|---|---|
| E2B | 2 GB | 2 GB | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| E4B | 4 GB | 4 GB | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| 26B (MoE) | 8 GB | 8 GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| 31B (Dense) | 16 GB | 24 GB | ⚡⚡ | ⭐⭐⭐⭐⭐ |

**Most people should start with the 26B MoE model.** It only activates 3.8B parameters per inference, so it runs at near-8B speeds while delivering near-30B quality. If you have a laptop with 8 GB RAM, you can run it.

If you’re on constrained hardware (Raspberry Pi, old laptop), the E2B model at Q4 quantization fits in 2 GB. See our guide on running AI on a Raspberry Pi for more on that setup.

## Method 1: Ollama (recommended)

Ollama is the fastest way to get Gemma 4 running. One command to install, one command to run.

### Install Ollama

```
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
# Windows — download from ollama.com
```


### Run Gemma 4

```
# 26B MoE — best balance of speed and quality
ollama run gemma4:26b
# Edge model — fastest, runs on anything
ollama run gemma4:e2b
# Dense model — highest quality
ollama run gemma4:31b
# With specific quantization
ollama run gemma4:26b-q4_K_M
```


### Use as an API

Ollama exposes an OpenAI-compatible API on port 11434:

```
curl http://localhost:11434/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "gemma4:26b",
"messages": [{"role": "user", "content": "Explain MoE architecture in 3 sentences"}]
}'
```


This works with any tool that supports the OpenAI API format — including Continue.dev for VS Code integration and most AI coding tools.

## Method 2: llama.cpp (most control)

If you need fine-grained control over quantization, context length, or batch size, llama.cpp gives you direct access.

### Build llama.cpp

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)
# With CUDA support (NVIDIA GPUs)
make -j$(nproc) GGML_CUDA=1
# With Metal support (Apple Silicon)
make -j$(nproc) GGML_METAL=1
```


### Download the model

```
# Download GGUF quantized model from Hugging Face
# Q4_K_M is the best balance of size and quality
huggingface-cli download google/gemma-4-26b-GGUF gemma-4-26b-Q4_K_M.gguf
```


### Run inference

```
./llama-cli \
-m gemma-4-26b-Q4_K_M.gguf \
-c 8192 \
-n 512 \
--temp 0.7 \
-p "Write a Python function that validates email addresses"
```


### Run as a server

```
./llama-server \
-m gemma-4-26b-Q4_K_M.gguf \
-c 8192 \
--host 0.0.0.0 \
--port 8080
```


This exposes an OpenAI-compatible API, just like Ollama but with more configuration options.

## Method 3: vLLM (best for serving)

For production serving with high throughput, vLLM handles batching and memory management automatically.

```
pip install vllm
# Serve the model
vllm serve google/gemma-4-26b \
--max-model-len 8192 \
--gpu-memory-utilization 0.9
```


vLLM requires a GPU with enough VRAM for the full model. For the 26B model at FP16, that’s about 26 GB — an A100 40GB or two consumer GPUs. If you don’t have that kind of hardware locally, cloud GPU providers offer A100 and H100 instances on demand for a few dollars per hour. For quantized inference, use llama.cpp instead.

## Quantization options

Quantization reduces model size and memory usage at the cost of some quality. Here’s how the 26B model performs at different quantization levels:

| Quantization | Size | RAM needed | Quality loss | Speed |
|---|---|---|---|---|
| FP16 | 52 GB | 52 GB | None | Baseline |
| Q8_0 | 26 GB | 28 GB | Minimal | ~Same |
| Q5_K_M | 18 GB | 20 GB | Very small | Faster |
Q4_K_M | 14 GB | 16 GB | Small | Faster |
| Q3_K_M | 11 GB | 13 GB | Noticeable | Fastest |
| Q2_K | 8 GB | 10 GB | Significant | Fastest |

**Q4_K_M is the sweet spot** for most users. The quality loss is barely noticeable in conversation and coding tasks, while cutting memory usage by 70%.

If you’re running without a GPU, Q4_K_M on CPU is still usable — expect 5-15 tokens per second on a modern laptop.

## Performance tips

### Use GPU offloading

If you have a GPU but not enough VRAM for the full model, offload some layers to GPU and keep the rest in RAM:

```
# llama.cpp: offload 20 layers to GPU
./llama-cli -m gemma-4-26b-Q4_K_M.gguf -ngl 20 -p "Hello"
# Ollama: set GPU layers in Modelfile
echo 'PARAMETER num_gpu 20' >> Modelfile
```


### Adjust context length

The default 256K context window uses a lot of memory. If you don’t need it, reduce it:

```
# llama.cpp: use 4K context (saves ~2GB RAM)
./llama-cli -m gemma-4-26b-Q4_K_M.gguf -c 4096
# Ollama: set in Modelfile
echo 'PARAMETER num_ctx 4096' >> Modelfile
```


### Use Flash Attention

Both llama.cpp and vLLM support Flash Attention, which significantly reduces memory usage for long contexts:

```
# llama.cpp
./llama-cli -m gemma-4-26b-Q4_K_M.gguf -fa
# vLLM enables it automatically
```


## Docker setup

For a reproducible environment:

```
FROM ollama/ollama:latest
RUN ollama pull gemma4:26b
EXPOSE 11434
CMD ["ollama", "serve"]
```


```
docker build -t gemma4-local .
docker run -d --gpus all -p 11434:11434 gemma4-local
```


## Comparing local AI runtimes

Not sure which runtime to use? Here’s the quick version:

| Ollama | llama.cpp | vLLM | |
|---|---|---|---|
| Setup time | 2 min | 10 min | 5 min |
| GPU required | No | No | Yes |
| Quantization | Auto | Full control | Limited |
| API | OpenAI-compat | OpenAI-compat | OpenAI-compat |
| Best for | Getting started | Tweaking | Production |

For a deeper comparison, see our Ollama vs llama.cpp vs vLLM guide.

## What’s next

Once you have Gemma 4 running locally, you can:

**Use it for coding**: Connect it to VS Code via Continue.dev. See best AI models for coding locally.**Compare it**: See how it stacks up against Llama 4 and Qwen 3.5.**Build with it**: Create a local AI docs chatbot powered by Gemma 4.**Go offline**: Set up a fully offline AI stack for air-gapped environments.

Gemma 4 26B running locally on a laptop is genuinely useful for daily coding and writing tasks. The MoE architecture makes it feel like a much larger model than the hardware suggests. If you’ve been waiting for open models to be “good enough” — they are now.

*Related: Run Gemma 4 Locally*
