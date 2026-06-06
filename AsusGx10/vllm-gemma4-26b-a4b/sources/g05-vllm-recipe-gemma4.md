---
id: g05
title: "Gemma 4 Usage Guide"
url: "https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html"
publisher: "vLLM Recipes"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [vllm, local, serving, multimodal]
---

# Gemma 4 Usage Guide¶

Gemma 4 is Google's most capable open model family, featuring a unified multimodal architecture that natively processes text, images, and audio. Gemma 4 models support advanced capabilities including structured thinking/reasoning, function calling with a custom tool-use protocol, and dynamic vision resolution — all available through vLLM's OpenAI-compatible API.

Gemma 4 models are supported on NVIDIA GPUs, AMD GPUs, Google Cloud TPUs and Intel Xeon 6 CPUs. TPU support is provided through vLLM TPU. For detailed TPU deployment guides, see the Trillium and Ironwood recipes.

## Supported Models¶

### Dense Models¶

| Model | Parameters | Min NVIDIA GPUs (BF16) | Min AMD GPUs (BF16) | Min TPUs | Min Xeon 6 CPUs | HuggingFace |
|---|---|---|---|---|---|---|
| Gemma 4 E2B IT | effective 2B | 1× (24 GB+) | 1× MI300X/MI325X/MI350X/MI355X | - | 1× NUMA node | google/gemma-4-E2B-it |
| Gemma 4 E4B IT | effective 4B | 1× (24 GB+) | 1× MI300X/MI325X/MI350X/MI355X | - | 1× NUMA node | google/gemma-4-E4B-it |
| Gemma 4 12B IT | 12B | 1× (40 GB+) | 1× MI300X/MI325X/MI350X/MI355X | - | - | google/gemma-4-12B-it |
| Gemma 4 31B IT | 31B | 1× (80 GB) | 1× MI300X/MI325X/MI350X/MI355X | 4× Trillium / 1× Ironwood | - | google/gemma-4-31B-it |

### Mixture-of-Experts (MoE) Models¶

| Model | Total / Active Params | Min NVIDIA GPUs (BF16) | Min AMD GPUs (BF16) | Min TPUs | Min Xeon 6 CPUs | HuggingFace |
|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B IT | 26B / 4B active | 1× (80 GB) | 1× MI300X/MI325X/MI350X/MI355X | 4× Trillium / 1× Ironwood | 2× NUMA node | google/gemma-4-26B-A4B-it |

### Key Architecture Features¶

**Multimodal**: Natively processes text and images (video supported via a custom vLLM processing pipeline that extracts frames; smaller gemma4-E2B and gemma-4-E4B also support audio).**MoE**: 128 fine-grained experts with top-8 routing and custom GELU-activated FFN (26B-A4B)**Encoder-Free (Unified)**: The 12B variant (`Gemma4UnifiedForConditionalGeneration`

) has no vision tower or audio encoder. Raw pixel patches are projected directly into LM space via Dense+LayerNorm with factorized 2D positional embeddings; raw audio waveform frames are projected through a simple multimodal embedder. All modalities (image, video, audio) are supported.**Dual Attention**: Alternating sliding-window (local) and global attention with different head dimensions**Thinking Mode**: Structured reasoning via`<|channel>thought\n...<channel|>`

delimiters**Function Calling**: Custom tool-call protocol with dedicated special tokens**Dynamic Vision Resolution**: Per-request configurable vision token budget (70, 140, 280, 560, 1120 tokens)

## Installing vLLM¶

### pip (NVIDIA CUDA)¶

```
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
--extra-index-url https://wheels.vllm.ai/nightly/cu129 \
--extra-index-url https://download.pytorch.org/whl/cu129 \
--index-strategy unsafe-best-match
```


### pip (AMD ROCm: MI300X, MI325X, MI350X, MI355X)¶

**Note:** The vLLM nightly wheel for ROCm requires Python 3.12, ROCm 7.2.1, glibc ≥ 2.35 (Ubuntu 22.04+)

```
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --pre \
--extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm721 --upgrade
```


### pip (Intel Xeon 6 CPUs)¶

For Intel and AMD x86 CPUs, follow the CPU pre-built wheels installation instructions.

### Docker¶

```
docker pull vllm/vllm-openai:latest # For CUDA 12.9
docker pull vllm/vllm-openai:latest-cu130 # For CUDA 13.0
docker pull vllm/vllm-openai-rocm:latest # For AMD GPUs
docker pull vllm/vllm-tpu:gemma4 # For Cloud TPUs
docker pull vllm/vllm-openai-cpu:latest-x86_64 # For Intel Xeon 6
```


## Running Gemma 4¶

### Quick Start (Single GPU)¶

### Multi-GPU Deployment¶

## 31B Dense on 2× A100/H100 (TP2, BF16)

## 26B MoE on 1× A100/H100 (BF16)

## E2B, E4B, 31B Dense or 26B MoE on 1× MI300X/MI325X/MI350X/MI355X (BF16)

where MODEL is any of the gemma4 models.### Docker Deployment¶

```
docker run -itd --name gemma4 \
--ipc=host \
--network host \
--shm-size 16G \
--gpus all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
vllm/vllm-openai:latest \
--model google/gemma-4-31B-it \
--tensor-parallel-size 2 \
--max-model-len 32768 \
--gpu-memory-utilization 0.90 \
--host 0.0.0.0 \
--port 8000
```


### Cloud TPU Deployment via Docker¶

```
docker run -itd --name gemma4-tpu \
--privileged \
--network host \
--shm-size 16G \
-v /dev/shm:/dev/shm \
-e HF_TOKEN=$HF_TOKEN \
vllm/vllm-tpu:gemma4 \
--model google/gemma-4-31B-it \
--tensor-parallel-size 8 \
--max-model-len 16384 \
--disable_chunked_mm_input \
--host 0.0.0.0 \
--port 8000
```


For detailed deployment guides and configurations, see the TPU recipes for Trillium and Ironwood

### AMD GPU Deployment (MI300X, MI325X, MI350X, MI355X) via Docker¶

Launch the ROCm vLLM Docker container where

```
docker run -itd --name gemma4-rocm \
--ipc=host \
--network=host \
--privileged \
--cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd \
--device=/dev/dri \
--group-add=video \
--cap-add=SYS_PTRACE \
--security-opt=seccomp=unconfined \
--shm-size 16G \
-v ~/.cache/huggingface:/root/.cache/huggingface \
vllm/vllm-openai-rocm:latest \
--model <MODEL> \
--host 0.0.0.0 \
--port 8000
```


### Intel Xeon 6 Deployment via Docker¶

Launch the x86 CPU vLLM Docker container, replacing `<MODEL>`

with the desired Google Gemma 4 model:

```
docker run -itd --name gemma4-cpu \
--network=host \
--shm-size 16G \
-v ~/.cache/huggingface:/root/.cache/huggingface \
vllm/vllm-openai-cpu:latest-x86_64 \
--model <MODEL> \
--host 0.0.0.0 \
--port 8000
```


For additional Intel Xeon 6 deployment details, see the Intel Software Catalog entries for Gemma 4 E4B IT and Gemma 4 26B-A4B IT.

### Configuration Tips¶

- Set
`--max-model-len`

to match your actual workload. The default context length can be very large; reducing it saves memory for KV cache. - Use
`--gpu-memory-utilization 0.90`

to`0.95`

to maximize KV cache capacity. - For image-only workloads (no audio), pass
`--limit-mm-per-prompt.audio 0`

to skip audio encoder memory allocation (tower-based models) or audio embedder allocation (12B encoder-free model). - For text-only workloads, pass
`--limit-mm-per-prompt '{"image": 0, "audio": 0}'`

to skip multimodal profiling entirely. - Use
`--async-scheduling`

for better overall throughput by overlapping scheduling with decoding.

## Text Generation¶

### Online Serving (OpenAI SDK)¶

```
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{"role": "user", "content": "Write a poem about the ocean."}
],
max_tokens=512,
temperature=0.7
)
print(response.choices[0].message.content)
```


### Online Serving (cURL)¶

```
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "google/gemma-4-31B-it",
"messages": [
{"role": "user", "content": "Explain quantum entanglement in simple terms."}
],
"max_tokens": 512,
"temperature": 0.7
}'
```


### Offline Inference¶

```
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model_path = "google/gemma-4-31B-it"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
model=model_path,
tensor_parallel_size=2,
max_model_len=8192,
gpu_memory_utilization=0.90,
trust_remote_code=True
)
messages = [
{"role": "user", "content": "What are the three laws of thermodynamics?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=1024))
print(outputs[0].outputs[0].text)
```


## Image Understanding¶

Gemma 4 natively understands images via its custom vision encoder with configurable resolution (utilizing native vision blocks).

### Single Image (OpenAI SDK)¶

```
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{
"role": "user",
"content": [
{
"type": "image_url",
"image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}
},
{
"type": "text",
"text": "Describe this image in detail."
}
]
}
],
max_tokens=1024
)
print(response.choices[0].message.content)
```


### Multiple Images¶

```
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{
"role": "user",
"content": [
{
"type": "image_url",
"image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}
},
{
"type": "image_url",
"image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"}
},
{
"type": "text",
"text": "What are the key similarities and differences between these two images?"
}
]
}
],
max_tokens=1024
)
```


### Dynamic Vision Resolution¶

Gemma 4 supports per-request dynamic vision token budgets. Higher token counts produce more detailed image understanding at the cost of more compute.

Supported values: **70**, **140**, **280** (default), **560**, **1120** tokens per image.

To configure the default at server launch:

To override per-request (offline inference):

```
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor
model_path = "google/gemma-4-31B-it"
processor = AutoProcessor.from_pretrained(model_path)
llm = LLM(
model=model_path,
max_model_len=8192,
trust_remote_code=True,
limit_mm_per_prompt={"image": 4},
# Set the maximum capacity the vision tower will allocate for
hf_overrides={
"vision_config": {"default_output_length": 1120},
"vision_soft_tokens_per_image": 1120
},
# Default token budget when no per-request override is given
mm_processor_kwargs={"max_soft_tokens": 280}
)
image = Image.open("photo.jpg").convert("RGB")
messages = [{"role": "user", "content": [
{"type": "image"},
{"type": "text", "text": "Describe this image in detail."}
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Override to 1120 tokens for this specific request
outputs = llm.generate(
{
"prompt": prompt,
"multi_modal_data": {"image": image},
"mm_processor_kwargs": {"max_soft_tokens": 1120}
},
sampling_params=SamplingParams(temperature=0.0, max_tokens=512),
)
print(outputs[0].outputs[0].text)
```


## Audio Understanding¶

Gemma 4 supports audio understanding across multiple variants. The E2B and E4B models use a conformer-based audio encoder, while the 12B (encoder-free) model projects raw 16 kHz waveform frames directly into LM space through a lightweight multimodal embedder with no audio tower required.

ℹ️

NoteAudio support requires the`vllm[audio]`

extras:`uv pip install "vllm[audio]"`


### Launch Server with Audio Support¶

```
vllm serve google/gemma-4-31B-it \
--max-model-len 8192 \
--limit-mm-per-prompt '{"image": 4, "audio": 1}'
```


### Audio Transcription (OpenAI SDK)¶

```
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
response = client.chat.completions.create(
model="google/gemma-4-E2B-it",
messages=[
{
"role": "user",
"content": [
{
"type": "audio_url",
"audio_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/2/22/Beatbox_by_Wikipedia_user_Wikipedia_Brown.ogg"}
},
{
"type": "text",
"text": "Provide a verbatim, word-for-word transcription of the audio."
}
]
}
],
max_tokens=512
)
print(response.choices[0].message.content)
```


### Audio Transcription (cURL)¶

```
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "google/gemma-4-E4B-it",
"messages": [
{
"role": "user",
"content": [
{"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}},
{"type": "text", "text": "Transcribe this audio."}
]
}
],
"max_tokens": 512
}'
```


## Video Understanding¶

Video understanding is supported via a custom processing pipeline (available in this vLLM branch) that extracts video frames and pairs them with text prompts for the vision tower.

### Launch Server with Video Support¶

```
vllm serve google/gemma-4-E2B-it \
--max-model-len 8192 \
--limit-mm-per-prompt '{"image": 4, "video": 1}'
```


### Video Inference (OpenAI SDK Style)¶

```
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
response = client.chat.completions.create(
model="google/gemma-4-E2B-it",
messages=[
{
"role": "user",
"content": [
{
"type": "video_url",
"video_url": {"url": "https://example.com/sample_video.mp4"}
},
{
"type": "text",
"text": "Summarize what happens in this video."
}
]
}
],
max_tokens=1024
)
print(response.choices[0].message.content)
```


### Offline Inference (Video)¶

```
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_video
from transformers import AutoProcessor
model_path = "google/gemma-4-E2B-it"
processor = AutoProcessor.from_pretrained(model_path)
llm = LLM(
model=model_path,
max_model_len=8192,
trust_remote_code=True,
limit_mm_per_prompt={"image": 4, "video": 1}
)
video_url = "https://example.com/sample_video.mp4"
video_data = fetch_video(video_url)
messages = [{"role": "user", "content": [
{"type": "video"},
{"type": "text", "text": "Summarize what happens in this video."}
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate(
{
"prompt": prompt,
"multi_modal_data": {"video": [video_data]}
},
sampling_params=SamplingParams(temperature=0.0, max_tokens=1024),
)
print(outputs[0].outputs[0].text)
```


## Thinking / Reasoning Mode¶

Gemma 4 supports structured thinking, where the model can reason step-by-step before producing a final answer. The reasoning process is exposed via the `reasoning`

field in the API response.

ℹ️

NoteThe example chat template file is included in the official container and can also be downloaded from the vLLM repository.

### Launch Server with Thinking Support¶

```
vllm serve google/gemma-4-31B-it \
--max-model-len 16384 \
--enable-auto-tool-choice \
--reasoning-parser gemma4 \
--tool-call-parser gemma4 \
--chat-template examples/tool_chat_template_gemma4.jinja
```


If you want to default to thinking enabled for all requests, add the argument `--default-chat-template-kwargs '{"enable_thinking": true}'`

to the above command.

### Thinking Mode (OpenAI SDK)¶

```
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{"role": "user", "content": "A snail is at the bottom of a 20-foot well. Each day it climbs 3 feet, but at night it slides back 2 feet. How many days will it take to reach the top?"}
],
max_tokens=4096,
extra_body={
"chat_template_kwargs": {"enable_thinking": True}
}
)
message = response.choices[0].message
# The thinking process is in reasoning
if hasattr(message, "reasoning") and message.reasoning:
print("=== Thinking ===")
print(message.reasoning)
print("\n=== Answer ===")
print(message.content)
```


### Thinking Mode (cURL)¶

```
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "google/gemma-4-31B-it",
"messages": [
{"role": "user", "content": "What is the derivative of x^3 * ln(x)?"}
],
"max_tokens": 4096,
"chat_template_kwargs": {"enable_thinking": true}
}'
```


ℹ️

NoteThinking mode produces additional tokens for the reasoning chain. Increase`--max-model-len`

and`max_tokens`

accordingly to accommodate longer outputs. The`thought\n`

role label inside the channel delimiters is automatically stripped by the reasoning parser.

## Function Calling / Tool Use¶

Gemma 4 supports function calling with a dedicated tool-call protocol using custom special tokens (`<|tool_call>`

, `<tool_call|>`

, etc.).

ℹ️

NoteThe example chat template file is included in the official container and can also be downloaded from the vLLM repository.

### Launch Server with Tool Calling¶

```
vllm serve google/gemma-4-31B-it \
--max-model-len 8192 \
--enable-auto-tool-choice \
--tool-call-parser gemma4 \
--reasoning-parser gemma4 \
--chat-template examples/tool_chat_template_gemma4.jinja
```


### Tool Calling (OpenAI SDK)¶

```
from openai import OpenAI
import json
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
# Define tools
tools = [
{
"type": "function",
"function": {
"name": "get_weather",
"description": "Get the current weather for a location",
"parameters": {
"type": "object",
"properties": {
"location": {
"type": "string",
"description": "City name, e.g. 'San Francisco, CA'"
},
"unit": {
"type": "string",
"enum": ["celsius", "fahrenheit"],
"description": "Temperature unit"
}
},
"required": ["location"]
}
}
}
]
# Step 1: Send user message with tools
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{"role": "user", "content": "What is the weather in Tokyo today?"}
],
tools=tools,
max_tokens=1024
)
message = response.choices[0].message
# Step 2: Process tool calls
if message.tool_calls:
tool_call = message.tool_calls[0]
print(f"Tool: {tool_call.function.name}")
print(f"Args: {tool_call.function.arguments}")
# Step 3: Feed back tool result and get final answer
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{"role": "user", "content": "What is the weather in Tokyo today?"},
message, # assistant's tool call message
{
"role": "tool",
"tool_call_id": tool_call.id,
"content": json.dumps({"temperature": 22, "condition": "Partly cloudy", "unit": "celsius"})
}
],
tools=tools,
max_tokens=1024
)
print(f"\nFinal answer: {response.choices[0].message.content}")
```


### Tool Calling with Thinking¶

Gemma 4 can combine thinking mode with tool calling — the model reasons about which tool to use before making the call:

```
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{"role": "user", "content": "I need to know the weather in Tokyo and then calculate the wind chill factor."}
],
tools=tools,
max_tokens=4096,
extra_body={
"chat_template_kwargs": {"enable_thinking": True}
}
)
```


## Multimodal + Tool Calling¶

Gemma 4 can combine vision understanding with tool calling — for example, identifying a city from an image and then looking up its weather:

```
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{
"role": "user",
"content": [
{
"type": "image_url",
"image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Christ_the_Redeemer_-_Rio_de_Janeiro%2C_Brazil.jpg/800px-Christ_the_Redeemer_-_Rio_de_Janeiro%2C_Brazil.jpg"}
},
{
"type": "text",
"text": "What city is shown in this image? What is the current weather there?"
}
]
}
],
tools=tools,
max_tokens=1024
)
```


## Structured Outputs¶

Gemma 4 supports structured output generation via vLLM's guided decoding engine, which constrains the model to produce valid JSON matching a provided schema. This is useful for extracting structured data, building reliable pipelines, and integrating with typed APIs.

### JSON Schema (OpenAI SDK)¶

```
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
json_schema = {
"type": "object",
"properties": {
"city": {"type": "string"},
"country": {"type": "string"},
"population": {"type": "integer"},
"landmarks": {
"type": "array",
"items": {"type": "string"}
}
},
"required": ["city", "country", "population", "landmarks"]
}
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{
"role": "system",
"content": "Extract city information as structured JSON."
},
{
"role": "user",
"content": "Tell me about Paris, France."
}
],
response_format={
"type": "json_schema",
"json_schema": {
"name": "city-info",
"schema": json_schema
}
},
max_tokens=512
)
import json
data = json.loads(response.choices[0].message.content)
print(data)
# {"city": "Paris", "country": "France", "population": 2161000, "landmarks": ["Eiffel Tower", "Louvre Museum", ...]}
```


### Pydantic Models (OpenAI SDK)¶

```
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI
client = OpenAI(
base_url="http://localhost:8000/v1",
api_key="EMPTY"
)
class WeatherReport(BaseModel):
air_temperature: Optional[float] = Field(None, description="Temperature in Fahrenheit")
wind_speed: Optional[float] = Field(None, description="Wind speed in mph")
comments_or_answer: str = Field(description="Comments or answer to the user's question")
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{
"role": "system",
"content": (
"Extract the weather information. Output JSON with these fields:\n"
"- air_temperature: float, converted to Fahrenheit\n"
"- wind_speed: float, converted to mph\n"
"- comments_or_answer: string, answer the user's question"
)
},
{
"role": "user",
"content": "The current weather in Seattle is 22.0°C with a wind speed of 6.0 km/h."
}
],
response_format={
"type": "json_schema",
"json_schema": {
"name": "weather-report",
"schema": WeatherReport.model_json_schema()
}
},
max_tokens=256
)
```


⚠️

Important: Schema Descriptions Are Not Visible to the ModelStructured output in vLLM works at two separate layers:


Constrained decoding(`response_format`

) — forces the output to match the JSON schemastructurally(correct keys, types, required fields). The modeldoes not seethe schema or its field descriptions.Prompt / system message— the model reads and reasons about the instructions. This is where it learnswhat valuesto produce (e.g., unit conversions, formatting rules).If you only use

`response_format`

without describing the schema in the prompt, the model will produce structurally valid JSON but won't follow semantic instructions embedded in`Field(description=...)`

.Always include output instructions in the system messageand use`response_format`

for structural enforcement.

### Structured Outputs with Thinking¶

Structured outputs can be combined with thinking mode. The model reasons step-by-step before producing the constrained JSON output:

```
response = client.chat.completions.create(
model="google/gemma-4-31B-it",
messages=[
{
"role": "system",
"content": (
"Analyze the text and extract entities. Output JSON with:\n"
"- people: list of person names mentioned\n"
"- organizations: list of organization names\n"
"- locations: list of location names\n"
"- summary: one-sentence summary of the text"
)
},
{
"role": "user",
"content": "Dr. Elena Torres, lead researcher at the Riverside Institute, presented her findings on marine biodiversity at the annual symposium in Cape Marina. The Oceanic Wildlife Fund and the Global Conservation Alliance both pledged support."
}
],
response_format={
"type": "json_schema",
"json_schema": {
"name": "entity-extraction",
"schema": {
"type": "object",
"properties": {
"people": {"type": "array", "items": {"type": "string"}},
"organizations": {"type": "array", "items": {"type": "string"}},
"locations": {"type": "array", "items": {"type": "string"}},
"summary": {"type": "string"}
},
"required": ["people", "organizations", "locations", "summary"]
}
}
},
max_tokens=4096,
extra_body={
"chat_template_kwargs": {"enable_thinking": True}
}
)
message = response.choices[0].message
if hasattr(message, "reasoning") and message.reasoning:
print("=== Thinking ===")
print(message.reasoning)
print("\n=== Structured Output ===")
print(message.content)
```


## Offline Inference (Multimodal)¶

For batch processing without a running server:

```
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor
model_path = "google/gemma-4-31B-it"
processor = AutoProcessor.from_pretrained(model_path)
llm = LLM(
model=model_path,
tensor_parallel_size=2,
max_model_len=8192,
gpu_memory_utilization=0.90,
trust_remote_code=True,
limit_mm_per_prompt={"image": 4, "audio": 1}
)
# Text + Image example
image = Image.open("photo.jpg").convert("RGB")
messages = [
{
"role": "user",
"content": [
{"type": "image"},
{"type": "text", "text": "Describe this image in detail."}
]
}
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
outputs = llm.generate(
{"prompt": prompt, "multi_modal_data": {"image": image}},
sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```


## Speculative Decoding (MTP)¶

Gemma 4 supports Multi-Token Prediction (MTP) speculative decoding using lightweight assistant models that share KV cache with the target model, enabling faster token generation with no quality loss.

### Available Assistant Models¶

| Target Model | Assistant Model | Centroids Masking |
|---|---|---|
| Gemma 4 E2B IT | google/gemma-4-E2B-it-assistant | Yes |
| Gemma 4 E4B IT | google/gemma-4-E4B-it-assistant | Yes |
| Gemma 4 12B IT | google/gemma-4-12B-it-assistant | No |
| Gemma 4 26B-A4B IT | google/gemma-4-26B-A4B-it-assistant | No |
| Gemma 4 31B IT | google/gemma-4-31B-it-assistant | No |

The E2B and E4B assistant models use **centroids masking** — a sparse logit computation that replaces the full vocabulary dot product (~262K tokens) with a centroid-based selection of ~4K candidate tokens. This reduces the lm_head computation by ~45x with negligible impact on draft token quality. Centroids masking is enabled automatically when the assistant checkpoint includes the centroid weights (`use_ordered_embeddings: true`

); no user configuration is needed.

### Online Serving¶

```
vllm serve google/gemma-4-31B-it \
--tensor-parallel-size 2 \
--max-model-len 8192 \
--speculative-config '{"model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 4}'
```


### Offline Inference¶

```
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model_path = "google/gemma-4-E4B-it"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
model=model_path,
speculative_config={
"model": "google/gemma-4-E4B-it-assistant",
"num_speculative_tokens": 4,
},
max_model_len=8192,
trust_remote_code=True,
)
messages = [{"role": "user", "content": "What are the three laws of thermodynamics?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=1024))
print(outputs[0].outputs[0].text)
```


### Recommended Settings¶

| Target Model | Recommended `num_speculative_tokens` |
TP |
|---|---|---|
| E2B | 2 | 1 |
| E4B | 4 | 1 |
| 12B | 4–8 | 1 |
| 26B-A4B | 4 | 2 |
| 31B | 4–8 | 2 |

Higher `num_speculative_tokens`

increases draft overhead per cycle. The optimal value depends on the target model speed — slower targets (31B) benefit from more speculative tokens, while faster targets (E2B) prefer fewer.

ℹ️

NoteThese recommendations were benchmarked on NVIDIA A100 and H100 servers. Optimal settings may vary on different hardware platforms — experimentation is recommended.

## Quantized Models (QAT W4A16)¶

Google provides official QAT (Quantization-Aware Training) W4A16 checkpoints for the Gemma 4 dense model family. These use 4-bit integer weights with 16-bit activations (group_size=32) in `compressed-tensors`

format, delivering significant memory savings and throughput improvements with minimal quality loss.

### Available Checkpoints¶

| Base Model | QAT W4A16 Checkpoint | Memory Savings |
|---|---|---|
| google/gemma-4-E2B-it | google/gemma-4-E2B-it-qat-w4a16-ct | ~26% (9.8→7.3 GB) |
| google/gemma-4-E4B-it | google/gemma-4-E4B-it-qat-w4a16-ct | ~36% (15.2→9.8 GB) |
| google/gemma-4-12B-it | google/gemma-4-12B-it-qat-w4a16-ct | ~64% (22.8→8.3 GB) |
| google/gemma-4-31B-it | google/gemma-4-31B-it-qat-w4a16-ct | ~66% (59.0→19.8 GB) |

ℹ️

NoteThe 26B-A4B MoE model is not included — its small expert dimensions (704) cause excessive quality loss with 4-bit quantization. For the MoE model, use`--quantization int8_per_channel_weight_only`

(online, no checkpoint needed) which provides ~47% memory savings with negligible quality impact.

### Serving a QAT W4A16 Checkpoint¶

No `--quantization`

flag is needed — vLLM auto-detects the quantization config from the checkpoint:

```
vllm serve google/gemma-4-31B-it-qat-w4a16-ct \
--max-model-len 32768 \
--gpu-memory-utilization 0.90
```


### 26B MoE: Int8 Per-Channel Quantization¶

The 26B-A4B MoE model uses online int8 per-channel weight-only quantization instead of W4A16 — its small expert dimensions (128 experts × 704 intermediate size) are sensitive to 4-bit quantization:

```
vllm serve google/gemma-4-26B-A4B-it \
--quantization int8_per_channel_weight_only \
--max-model-len 32768 \
--gpu-memory-utilization 0.90
```


### Quantized Models (QAT Mobile — Mixed Int2/4/8)¶

Google also provides **mobile-optimized QAT checkpoints** for the E2B and E4B models. Unlike the uniform W4A16 variants above, these use a mixed-precision `compressed-tensors`

scheme with per-layer bitwidth assignment — including int2 embeddings and lm_head, which are typically left unquantized:

| Component | Weight Bits | Notes |
|---|---|---|
| Embeddings + lm_head | 2-bit |
Channel-quantized, pack-quantized format |
| Audio tower linears | 2-bit |
W2A8 (int2 weights, int8 activations) |
| LLM attention + MLP | 4-bit (E4B) / 2–4-bit (E2B) |
E2B drops deeper MLP layers (15–34) to int2 |
| Per-layer gates/projections + vision tower | 8-bit |
W8A8 (int8 weights, int8 activations) |

| Base Model | Mobile QAT Checkpoint | Weight Memory | Compression vs BF16 |
|---|---|---|---|
| google/gemma-4-E2B-it | google/gemma-4-E2B-it-qat-mobile-ct | 2.7 GB | 3.6× smaller |
| google/gemma-4-E4B-it | google/gemma-4-E4B-it-qat-mobile-ct | 3.7 GB | 4.1× smaller |

**Performance (A100-80GB, 1024in/1024out text requests):** E4B mobile delivers **up to 1.5× higher output throughput** vs BF16 at low concurrency (1–16 requests) where decode is memory-bandwidth-bound. The speedup tapers above ~64 concurrent requests as the workload becomes compute-bound.

The primary value proposition is **footprint**: 3.6–4.1× smaller weights free VRAM for KV cache or allow deployment on much smaller GPUs. Multimodal correctness (text, vision, audio) tracks BF16 quality.

## Benchmarking¶

### Launch Server for Benchmarking¶

When benchmarking, disable prefix caching to get consistent measurements:

```
vllm serve google/gemma-4-31B-it \
--tensor-parallel-size 2 \
--max-model-len 32768 \
--no-enable-prefix-caching \
--limit-mm-per-prompt '{"image": 0, "audio": 0}' \
--async-scheduling
```


### Text Benchmark¶

```
# Prompt-heavy benchmark (8k input / 1k output)
vllm bench serve \
--model google/gemma-4-31B-it \
--dataset-name random \
--random-input-len 8000 \
--random-output-len 1000 \
--request-rate 10000 \
--num-prompts 16 \
--ignore-eos
```


### Benchmark Configurations¶

Test different workloads by adjusting input/output lengths:

**Prompt-heavy**: 8000 input / 1000 output**Decode-heavy**: 1000 input / 8000 output**Balanced**: 1000 input / 1000 output

Test different batch sizes by changing `--num-prompts`

:

- Batch sizes: 1, 16, 32, 64, 128, 256

### Interpreting Results¶

```
============ Serving Benchmark Result ============
Successful requests: N
Benchmark duration (s): xxx.xx
Total input tokens: xxxxx
Total generated tokens: xxxxx
Request throughput (req/s): xxx.xx
Output token throughput (tok/s): xxx.xx
Total Token throughput (tok/s): xxx.xx
---------------Time to First Token----------------
Mean TTFT (ms): xxx.xx
Median TTFT (ms): xxx.xx
P99 TTFT (ms): xxx.xx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms): xxx.xx
Median TPOT (ms): xxx.xx
P99 TPOT (ms): xxx.xx
==================================================
```


Key metrics:
- **TTFT** (Time to First Token): Latency until the first output token. Critical for interactive applications.
- **TPOT** (Time per Output Token): Per-token generation latency after the first token.
- **Output token throughput**: Overall generation rate (tokens/second).

## Advanced Configuration¶

### Throughput vs. Latency Tuning¶

| Goal | Tensor Parallelism | Batch Size (`--max-num-seqs` ) |
Notes |
|---|---|---|---|
Max throughput |
Minimum (1-2) | High (256-512) | Best tokens/s per GPU |
Min latency |
High (4-8) | Low (8-16) | Best per-request TTFT/TPOT |
Balanced |
2 | 128 | Good for mixed workloads |

### Memory Optimization¶

**Reduce context length**:`--max-model-len 8192`

if your workload doesn't need long contexts**FP8 KV cache**:`--kv-cache-dtype fp8`

to reduce KV cache memory by ~50%**Limit multimodal inputs**:`--limit-mm-per-prompt '{"image": 2, "audio": 1}'`

to cap per-request memory

### Server Flags Reference¶

| Flag | Description | Recommended |
|---|---|---|
`--reasoning-parser gemma4` |
Enable Gemma 4 thinking/reasoning parser | Required for thinking mode |
`--tool-call-parser gemma4` |
Enable Gemma 4 tool call parser | Required for function calling |
`--enable-auto-tool-choice` |
Auto-detect tool calls in output | Required for function calling |
`--chat-template examples/tool_chat_template_gemma4.jinja` |
Override the model's default chat template to one optimized for reasoning and tool calling with vLLM | |
`--mm-processor-kwargs '{"max_soft_tokens": N}'` |
Set default vision token budget | 280 (default), up to 1120 |
`--async-scheduling` |
Overlap scheduling with decoding | Recommended for throughput |
`--gpu-memory-utilization 0.90` |
GPU memory fraction for model + KV cache | 0.85-0.95 |
`--limit-mm-per-prompt '{"image": N, "audio": M}'` |
Max multimodal inputs per request | Depends on workload |

### Full-Featured Server Launch¶

This command enables all Gemma 4 capabilities (text, image, audio, thinking, and tool calling):

```
vllm serve google/gemma-4-31B-it \
--tensor-parallel-size 2 \
--max-model-len 16384 \
--gpu-memory-utilization 0.90 \
--enable-auto-tool-choice \
--reasoning-parser gemma4 \
--tool-call-parser gemma4 \
--chat-template examples/tool_chat_template_gemma4.jinja \
--limit-mm-per-prompt '{"image": 4, "audio": 1}' \
--async-scheduling \
--host 0.0.0.0 \
--port 8000
```


## Deploy on Modal¶

Modal lets you run this recipe on cloud GPUs with a single command — no infrastructure setup required.

The deployment script is `gemma4-modal.py`

in this directory.
