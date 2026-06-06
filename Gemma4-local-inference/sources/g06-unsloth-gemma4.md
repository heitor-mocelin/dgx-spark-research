---
id: g06
title: "Gemma 4 — How to Run Locally (Unsloth)"
url: "https://docs.unsloth.ai/models/gemma-4"
publisher: "Unsloth docs"
retrieved: "2026-06-06"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [local, gguf, llamacpp, quantization]
---

# ✨Gemma 4 - How to Run Locally

Run Google’s new Gemma 4 models locally, including E2B, E4B, 26B A4B, and 31B.

Gemma 4 is Google DeepMind’s new family of open models, including **12B**, **E2B**,** E4B**,** 26B-A4B**, and **31B.** The multimodal, hybrid-thinking models support 140+ languages, up to **256K context**, and have dense and MoE variants. Gemma 4 is Apache-2.0 licensed and can run on your local device.

**Gemma-4-12B** is new and features unified text, image and audio support. It runs on **8GB** RAM (4-bit) or 14GB (8-bit). **Gemma-4-E2B** and **E4B** also support image and audio. Run on **5GB RAM** (4-bit) or 15GB (full 16-bit).

Run Gemma 4Fine-tune Gemma 4Gemma 4 QAT

**Gemma-4-26B-A4B** runs on **18GB** (4-bit) or 28GB (8-bit). **Gemma-4-31B** needs **20GB RAM** (4-bit) or 34GB (8-bit).

You can now run all GGUFs, MLX and fine-tune Gemma 4 in Unsloth Studio (see right).

**QAT** variants of Gemma 4 reduce memory requirements around 3x while preserving model quality.

**Jun 5: **Gemma 4 QAT is released.

**Jun 2: **Gemma 4 12B Unified is released.

**Apr 20: **We conducted Gemma 4 GGUF Benchmarks to help you pick the best quant.

### Usage Guide

Gemma 4 excels at reasoning, coding, tool use, long-context and agentic workflows, and multimodal tasks. The smaller E2B and E4B variants are designed for phones and laptops, while the larger models target medium-high CPU /VRAM systems such as PCs with NVIDIA RTX GPUs.

**E2B**

Dense + PLE (128K context) Support: Text, Image, Audio

For phone / edge inference, ASR, speech translation

**E4B**

Dense + PLE (128K context) Support: Text, Image, Audio

Small model for laptops and fast local multimodal use

**12B Unified**

Dense (256K context) Support: Text, Image, Audio

Medium model for laptops and local multimodal use

**26B-A4B**

MoE (256K context) Support: Text, Image

Best speed / quality tradeoff for computer use

**31B**

Dense (256K context) Support: Text, Image

Strongest performance at slower inference

**See Gemma 4: ****Performance benchmarks**** and ****GGUF benchmarks****.**

**Should I pick 26B-A4B or 31B?**

**26B-A4B**- balances speed and accuracy. Its MoE design makes it faster than 31B, with 4B active parameters. Pick it if RAM is limited and you are fine trading a bit of quality for speed.**31B**- currently the strongest Gemma 4 model. Pick it for maximum quality if you have enough memory and can accept slightly slower speeds.

### Hardware requirements

**Table: Gemma 4 Inference GGUF recommended hardware requirements** (units = total memory: RAM + VRAM, or unified memory). You can use Gemma 4 on MacOS, NVIDIA RTX GPUs etc.

**E2B**

4 GB

5–8 GB

10 GB

**E4B**

5.5–6 GB

9–12 GB

16 GB

**12B Unified**

7–8 GB

13–14 GB

25 GB

**26B A4B**

16–18 GB

28–30 GB

52 GB

**31B**

17–20 GB

34–38 GB

62 GB

As a rule of thumb, your total available memory should at least exceed the size of the quantized model you download. If it does not, llama.cpp can still run using partial RAM / disk offload, but generation will be slower. You will also need more compute, depending on the context window you use.

### Recommended Settings

It is recommended to use Google's default Gemma 4 parameters:

`temperature = 1.0`

`top_p = 0.95`

`top_k = 64`


Recommended practical defaults for local inference:

Keep

**repetition/presence penalty**disabled or 1.0 unless you see looping.The End of Sentence token is

`<turn|>`


Gemma 4's max context is **128K** for **E2B** /** E4B** and `262,144`

for **12B** / **26B A4B **/** 31B**.

#### Thinking Mode

Compared to older Gemma chat templates, Gemma 4 uses the standard

, **system**

, and **assistant**

roles and adds explicit thinking control.**user**

**How to enable thinking:**

Add the token

at the **<|think|>****start of the system prompt**.

**Thinking enabled**

**Thinking disabled**

**Output behavior:**

When thinking is enabled, the model outputs its internal reasoning channel before the final answer.

When thinking is disabled, the larger models may still emit an **empty thought block** before the final answer.

**For example using "**What is the capital of France?":

**then it outputs with:**

**Multi-turn chat rule:**

For multi-turn conversations, **only keep the final visible answer in chat history**. Do **not** feed prior thought blocks back into the next turn.

**How to disable thinking:**

Note `llama-cli`

might not work reliably, so use `llama-server`

for disabling reasoning:

To disable thinking / reasoning, use `--chat-template-kwargs '{"enable_thinking":false}'`


If you're on **Windows** Powershell, use: `--chat-template-kwargs "{\"enable_thinking\":false}"`


Use 'true' and 'false' interchangeably.

## Run Gemma 4 Tutorials

Because Gemma 4 GGUFs comes in several sizes, the recommended starting point for the small models is 8-bit and the larger models is **Dynamic**** 4-bit**. Gemma 4 GGUFs or MLX:

🦥 Unsloth Studio Guide🦙 Llama.cpp Guide

**You can run and train Gemma 4 for free with a UI in our ****Unsloth Studio**✨** notebook:**

### 🦥 Unsloth Studio Guide

Gemma 4 can now be run and fine-tuned in Unsloth Studio, our new open-source web UI for local AI. Unsloth Studio lets you run models locally on **MacOS, Windows**, Linux and:

Search, download, run GGUFs and safetensor models

**Self-healing**tool calling +**web search****Code execution**(Python, Bash)Automatic inference parameter tuning (temp, top-p, etc.)

Fast CPU + GPU inference via llama.cpp

Train LLMs 2x faster with 70% less VRAM


#### Search and download Gemma 4

On first launch you will need to create a password to secure your account and sign in again.

Then go to the Studio Chat tab and search for Gemma 4 in the search bar and download your desired model and quant. Unsloth supports the latest Gemma-4-12B Unified model.

#### Run Gemma 4

Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings. You can run GGUFs and MLX files.

For more information, you can view our Unsloth Studio inference guide.

### 🦙 Llama.cpp Guide

For this guide we will be utilizing Dynamic 4-bit for the 12B, 26B-A4B and 31B, and 8-bit for E2B and E4B. See: Gemma 4 GGUF collection

For these tutorials, we will using llama.cpp for fast local inference, especially if you have a CPU.

Obtain the latest `llama.cpp`

**on** **GitHub here**. You can follow the build instructions below as well. Change `-DGGML_CUDA=ON`

to `-DGGML_CUDA=OFF`

if you don't have a GPU or just want CPU inference. **For Apple Mac / Metal devices**, set `-DGGML_CUDA=OFF`

then continue as usual - Metal support is on by default.

If you want to use `llama.cpp`

directly to load models, you can follow commands below, according to each model. `UD-Q4_K_XL`

is the quantization type. You can also download via Hugging Face (step 3). This is similar to `ollama run`

. Use `export LLAMA_CACHE="folder"`

to force `llama.cpp`

to save to a specific location. There is no need to set context length as llama.cpp automatically uses the exact amount required.

To disable thinking / reasoning, use: `--chat-template-kwargs '{"enable_thinking":false}'`


**Windows** Powershell: `--chat-template-kwargs "{\"enable_thinking\":false}"`


Use '`true`

' and '`false`

' interchangeably.

**12B:**

**26B-A4B:**

**31B:**

**E4B:**

**E2B:**

Download the model via (after installing `pip install huggingface_hub hf_transfer`

). You can choose `UD-Q4_K_XL`

or other quantized versions like `Q8_0`

. If downloads get stuck, see: Hugging Face Hub, XET debugging

Then run the model in conversation mode (with vision `mmproj-F16`

):

### MLX Dynamic Quants

We also uploaded dynamic 4bit and 8bit quants as a first trial for MacOS device! The MLX quants support **vision.**

All MLX quants now work in Unsloth Studio!

To try them out use:

### Ollama Guide

Ollama now supports Unsloth GGUFs well now. Use `curl -fsSL https://ollama.com/install.sh | sh`

to install Ollama on Linux or `irm https://ollama.com/install.ps1 | iex`

for Windows.
To use a single quant file (under 50GB) use:

For multiple shards like larger BF16 shards do:

If you see `Error: 500 Internal Server Error: unable to load model`

update Ollama via `curl -fsSL https://ollama.com/install.sh | sh`

or use the Powershell one.

## Gemma 4 Best Practices

### Prompting examples

#### Simple reasoning prompt

#### OCR / document prompt

For OCR, use a **high visual token budget** like **560** or **1120**.

#### Multi-modal comparison prompt

#### Audio ASR prompt

#### Audio translation prompt

### Multi-modal Settings

For best results with multimodal prompts, put multimodal content first:

Put

**image and/or audio before text**.For video, pass a sequence of frames first, then the instruction.


#### Variable image resolution

Gemma 4 supports multiple visual token budgets:

`70`

`140`

`280`

`560`

`1120`


Use them like this:

**70 / 140**: classification, captioning, fast video understanding**280 / 560**: general multimodal chat, charts, screens, UI reasoning**1120**: OCR, document parsing, handwriting, small text

#### Audio and video limits

**Audio**is available on**12B**,**E2B**and**E4B**only.Audio supports a maximum of

**30 seconds**.Video supports a maximum of

**60 seconds**assuming**1 frame per second**processing.

#### Audio prompt templates

**ASR prompt**

**Speech translation prompt**

## 📊 Benchmarks

### Unsloth GGUF Benchmarks

We conducted Mean KL Divergence benchmarks for Gemma 4 GGUFs across providers to help you pick the best quant (lower is better).

KL Divergence puts all Unsloth GGUFs on the SOTA Pareto frontier

KLD shows how well a quantized model matches the original BF16 output distribution, indicating retained accuracy.


### Official Gemma Benchmarks

**Text/Code Benchmarks**

MMLU Pro

85.2%

82.6%

77.2%

69.4%

60.0%

67.6%

AIME 2026 no tools

89.2%

88.3%

77.5%

42.5%

37.5%

20.8%

LiveCodeBench v6

80.0%

77.1%

72.0%

52.0%

44.0%

29.1%

Codeforces ELO

2150

1718

1659

940

633

110

GPQA Diamond

84.3%

82.3%

78.8%

58.6%

43.4%

42.4%

Tau2

76.9%

68.2%

69.0%

42.2%

24.5%

16.2%

HLE no tools

19.5%

8.7%

5.2%

-

-

-

HLE with search

26.5%

17.2%

-

-

-

-

BigBench Extra Hard

74.4%

64.8%

53.0%

33.1%

21.9%

19.3%

MMMLU

88.4%

86.3%

83.4%

76.6%

67.4%

70.7%

**Vision Benchmarks**

MMMU Pro

76.9%

73.8%

69.1%

52.6%

44.2%

49.7%

OmniDocBench 1.5 (lower is better)

0.131

0.149

0.164

0.181

0.290

0.365

MATH-Vision

85.6%

82.4%

79.7%

59.5%

52.4%

46.0%

MedXPertQA MM

61.3%

58.1%

48.7%

28.7%

23.5%

-

**Audio Benchmarks**

CoVoST

-

-

38.5*

35.54

33.47

-

FLEURS (lower is better)

-

-

0.069*

0.08

0.09

-

**Long Context**

MRCR v2 8 needle 128k (average)

66.4%

44.1%

43.4%

25.4%

19.1%

13.5%

Last updated

Was this helpful?
