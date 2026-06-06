---
id: 019
title: "Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache"
url: "https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/"
publisher: "NVIDIA Technical Blog"
retrieved: "2026-06-05"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [quantization, nvfp4, context, memory, kv-cache]
---

Quantization is one of the strongest levers for large-scale inference. By reducing the precision of weights, activations, and KV cache, we can reduce the memory footprint and compute cost—directly improving throughput, latency, and achievable context length.

This blog introduces NVFP4 KV cache quantization, a new KV format that enables significant performance gains on NVIDIA Blackwell GPUs. NVFP4 cuts KV cache memory footprint by up to 50% and can effectively double context budgets, unlocking larger batch sizes, longer sequences, and higher cache-hit rates. These gains come with <1% accuracy loss across code-generation, knowledge, and long-context benchmarks.

In the sections that follow, we will explore how this optimization delivers tangible gains for inference workloads and strengthens the stacking effects of the NVIDIA extreme co-design stack.

## What is KV cache?

Large language models (LLMs) rely on an autoregressive process of generating tokens one by one based on all previous tokens. This process allows for consideration of the sequence’s full context, which is at the heart of why LLMs perform so well at natural language modeling tasks. This same behavior results in significant compute inefficiencies as models attempt to recalculate each preceding token’s attention projection, known as the key and value tensors, each time a new token is generated.

Figure 1 below provides a simplified representation of the attention computations with and without KV cache. Since the previous tokens’ attention values are masked from attending to future tokens, the key and value vectors for all past tokens (including the original input sequence) never change. As a result, recomputing them and redoing the associated matrix-multiply-add (MMA) operations for every new token is redundant and wastes computation.

KV cache was introduced to relieve the compute bottleneck created by having to regenerate key and value vectors for every previously seen token. By paying a cost in memory footprint and bandwidth, those K/V tensors are stored once and then fetched directly during attention, rather than recomputed. In practice, the cache sits behind a fixed-size memory pool, as shown in Figure 2 below.

When that pool fills, the KV cache manager evicts portions of older context. If a future request references an evicted span, the system takes a cache miss and is forced to recompute the missing K/V tensors. The net effect is that the actual performance gain hinges on cache-hit rate: High hit rates preserve the intended compute savings, while lower hit rates push the model back toward the very recomputation path the KV cache was meant to eliminate.

During inference, this cache is populated and used across two distinct phases. In the prefill phase, the model ingests the entire input sequence, running large, highly parallel matrix multiply and accumulate (MMA) operations to compute attention, and storing the resulting key and value vectors for all input tokens into the KV cache. The model then enters the decode phase, where it generates new tokens one by one; each step requires a full forward pass, but the attention blocks now fetch key and value vectors for all previous tokens from the KV cache, compute the current token’s key and value vectors, and append them back into the cache so they can be reused on the next decoding step.

## Optimizing KV cache with NVFP4

One of the latest opportunities to optimize KV cache performance is through NVFP4 and the NVIDIA TensorRT Model Optimizer. This new feature allows for the quantization of the KV cache from its native 16-bit precision down to 4-bit.

The quantization of KV cache is not entirely new, as FP8 KV caches are well utilized in production; however, the increasing size of models and scale of inference deployments means that the KV cache can still result in significant bottlenecks during prefill and decode. The quantization of KV cache helps alleviate the burden on multiple components of the inference pipeline, impacting compute, memory capacity, and memory bandwidth:

**Memory capacity:**NVFP4 KV cache reduces the memory footprint of the KV cache by about 50% compared to FP8 KV cache. This enables larger context lengths, batch sizes, and user concurrency.**Memory bandwidth:**During the decode phase, which involves many read/writes of KV cache and puts significant pressure on memory bandwidth, smaller KV cache consumes less memory bandwidth.

The current implementation of NVFP4 KV cache requires that values be dequantized from NVFP4 to FP8 before attention and context matrix math. The new token’s key and value vectors are quantized to NVFP4 before being appended to the KV cache (Figure 3).

The `quantize`

API from Model Optimizer can be used to perform post-training quantization (PTQ) or quantization aware training (QAT). To enable NVFP4 KV cache during PTQ or QAT the same `quantize`

API can be used—and it only requires changing the quantization configuration.

The code snippet below prepares the model for quantization to NVFP4 KV cache on top of FP8 weights and activations. To also get the benefit of 4-bit math, the model weights could be compressed to NVFP4 by changing `quant_cfg`

to `mtq.NVFP4_DEFAULT_CFG`

.

```
# configure fp8 quantization and fp4 for KV cache
quant_cfg = mtq.FP8_DEFAULT_CFG
quant_cfg["quant_cfg"].update(mtq.NVFP4_KV_CFG["quant_cfg"])
# Define forward loop for calibration with
def forward_loop(model):
for data in calib_set:
model(data)
# Quantize the modelmodel = mtq.quantize(model, quant_cfg, forward_loop)
# Model is ready for Post Training Quantization (PTQ) deployment
# (Optional) Quantization-aware training (QAT)
Train quantized model further for improving accuracy
# adjust training parameters, e.g., lr, schedule, epochs
# HuggingFace and Megatron models supported
train(model, train_loader, optimizer, scheduler, ...)
```

## How KV cache impacts performance

As mentioned above, KV cache eliminates redundant recomputation for previously processed tokens, at the cost of memory. Compressing KV cache to NVFP4 reduces this cost by 50% and doubles the content budget over the current standard FP8 KV cache, allowing models to hold double the context for inference. This benefits use cases that leverage textbook-scale sources and deep-reasoning—which would otherwise quickly exhaust KV cache memory budgets.

### Higher hit rates save prefill compute

During prefill, latency is heavily impacted by how much of the incoming request’s context is already resident in the KV cache. NVFP4 improves this by delivering higher effective cache-hit rates than FP8 since the 4-bit footprint allows approximately 2x more context to remain on-device. This reduces evictions and preserves larger spans of previously processed tokens. When the model can retrieve these KV entries directly instead of recomputing them, prefill experiences fewer stalls and higher sustained ingestion throughput, resulting in up to 3x better time-to-first-token (TTFT) latency.

As the KV cache grows, it captures more K/V tensors and naturally drives higher hit rates. This leads to a plateau effect where the latency and hit-rate delta between NVFP4 and FP8 narrows (Figure 4 above)—highly model and context-length dependent. But an ever-inflating, unoptimized KV cache consumes an increasing share of the HBM budget. NVFP4 restores efficiency by making KV caching dramatically more HBM-effective, freeing budget for model weights, and enabling stronger stacking benefits with other co-designed components across the stack—NVLink, kernel optimizations, and Wide Expert Parallelism.

### How NVFP4 KV cache impacts accuracy

We observe an accuracy loss of less than 1%, compared to BF16 and FP8 baselines, on modern LLM benchmarks such as LiveCodeBench, MMLU-PRO, MBPP, and Ruler 64K. In particular, near parity on LiveCodeBench shows that the quantization preserves precise multi-step code generation, where small numerical errors can easily turn into syntax, compilation, or logic failures.

Likewise, maintaining performance on Ruler 64K demonstrates robustness for long-context reasoning over 64K-token sequences, a setting where quantization noise typically accumulates. Together, these results indicate that the proposed format delivers efficiency gains without sacrificing end-to-end capability on challenging code and long-context workloads.

Another critical insight is how NVFP4 compares to MXFP4 for KV cache quantization. Figure 6 shows the impact on MMLU model accuracy scores across BF16, FP8, NVFP4, and MXFP4. For the model tested, Llama 3.3 70B, we observe 5% higher accuracy when the KV cache is in NVFP4 versus MXFP4. These benefits come from NVFP4’s more granular block scaling and higher precision E4M3 FP8 scaling factors, which together allow for lower quantization error during the dequantization step.

## Looking forward

NVFP4 KV cache is one more practical step in the broader software–hardware co‑design of the NVIDIA inference stack. As the ecosystem around it matures, it can be combined with KV‑aware routing and offload in NVIDIA Dynamo and stacked with large‑scale expert parallelism in NVIDIA TensorRT‑LLM’s Wide‑EP to improve utilization across big MoE deployments.

On the hardware side, tighter KV cache optimization can better exploit the NVL72 scale‑up domain and NVLink fabric for multi‑agent inference and long‑context deep‑reasoning workloads. Together, these pieces make it more feasible to serve larger experts, longer sequences, and higher concurrency without giving up accuracy.

To start applying these techniques, we recommend leveraging the Model Optimizer code samples and notebooks as a base recipe for custom quantization workflows.

*Kai Xu, Shengliang Xu, Tian Zheng, and Asma Kuriparambil Thekkumpate contributed to the engineering efforts described in this blog.*
