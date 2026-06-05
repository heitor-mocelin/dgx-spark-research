---
id: 003
title: "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"
url: "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/"
publisher: "NVIDIA Technical Blog"
retrieved: "2026-06-05"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [quantization, nvfp4, blackwell]
---

To get the most out of AI, optimizations are critical. When developers think about optimizing AI models for inference, model compression techniques—such as quantization, distillation, and pruning—typically come to mind. The most common of the three, without a doubt, is quantization. This is typically due to its post-optimization task-specific accuracy performance and broad choice of supported frameworks and techniques.

Yet the main challenge with model quantization is the potential loss of model intelligence or task-specific accuracy, particularly when transitioning from higher precision data types like FP32 down to the latest FP4 format. NVIDIA Blackwell provides maximum flexibility with support for FP64, FP32/TF32, FP16/BF16, INT8/FP8, FP6, and FP4 data formats. Figure 1 compares the smallest supported floating-point data type and corresponding dense/sparse performance across NVIDIA Ampere, Hopper, and Blackwell GPUs, showcasing the evolution of performance and data type support across GPU generations.

The latest fifth-generation NVIDIA Blackwell Tensor Cores pave the way for various ultra-low precision formats, enabling both research and real-world scenarios. Table 1 compares the three primary 4-bit floating point formats supported in NVIDIA Blackwell—FP4, MXFP4, and NVFP4—highlighting key differences in structure, memory usage, and accuracy. It illustrates how NVFP4 builds on the simplicity of earlier formats while maintaining model accuracy.

Feature | FP4 (E2M1) | MXFP4 | NVFP4 |
FormatStructure | 4 bits (1 sign, 2 exponent, 1 mantissa) plus software scaling factor | 4 bits (1 sign, 2 exponent, 1 mantissa) plus 1 shared power-of-two scale per 32 value block | 4 bits (1 sign, 2 exponent, 1 mantissa) plus 1 shared FP8 scale per 16 value block |
Accelerated Hardware Scaling | No | Yes | Yes |
Memory | Up to 4x less memory than FP16 | ||
Accuracy | Risk of noticeable accuracy drop compared to FP8 | Risk of noticeable accuracy drop compared to FP8 | Lower risk of noticeable accuracy drop particularly for larger models |

*Table 1. Comparison of Blackwell-supported 4-bit floating point formats*

This post introduces NVFP4, a state-of-the-art data type, and explains how it was purpose-built to help developers scale more efficiently on Blackwell, with the best accuracy at ultra-low precision.

## What is NVFP4?

NVFP4 is an innovative 4-bit floating point format introduced with the NVIDIA Blackwell GPU architecture. NVFP4 builds on the concept of low-bit “micro” floating-point formats and grants greater flexibility to developers by providing an additional format to choose from.

The structure of NVFP4 is similar to most floating-point 4-bit formats (E2M1), meaning that it has 1 sign bit, 2 exponent bits, and 1 mantissa bit. The value in the format ranges approximately -6 to 6. For example, the values in the range could include 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6 (same for the negative range).

One of the key challenges in ultra-low precision formats is maintaining numerical accuracy across a wide dynamic range of tensor values. NVFP4 addresses this concern with two architectural innovations that make it highly effective for AI inference:

- High-precision scale encoding
- A two-level micro-block scaling strategy

This strategy applies a fine-grained E4M3 scaling factor to each 16-value *micro-block*, a compact subset of the larger tensor, while also leveraging a second-level FP32 scalar applied per tensor. Together, these two levels of scaling enable more accurate value representation and significantly reduce quantization error (Figure 2).

## High-precision scaling: Encoding more signal, less error

To get value out of the shared micro-block scaling, NVFP4 encodes blocks using E4M3 FP8 precision. NVFP4 uses the E4M3 FP8 format variant that enables non-power-of-two scaling factors with fractional precision. This added flexibility enables more accurate encoding of the tensor’s actual distribution. Figure 3 shows an example of a full precision input matrix and the resulting quantized matrices using E8M0 and E4M3 scaling.

NVFP4’s more precise scaling factor with E4M3 results in a reduced range of scale values. This is counteracted by utilizing a second-level scaling factor. This second-level scaling factor is done at a per-tensor level with FP32 (illustrated in Figure 2), which adjusts the original tensor’s distribution such that the micro-blocks can be effectively encoded using E4M3 scale factors.

The animation in Figure 4 is a numberline representation of the matrix conversion in Figure 3. This example maps the original full precision values (represented by yellow circles) to their corresponding position along the dynamic range of the quantized datatype. The figure of merit is the average mean squared error (MSE) of the mappings from the original values to their representations in the quantized datatypes E8M0 and E4M3. Lower MSE is better with 0.08 average for E4M3.

What makes E4M3 “better on average” is that it picks that one fractional scale so that, when the squared (or absolute) errors over all 16 values are summed, the total error is generally smaller than an E8M0‐quantized block. In other words:

**E8M0**= Snaps the scale factor to nearest 2ⁿ, which can create a large quantization error for the block maximum (amax) and can often lead to larger overall quantization errors for blocks.**E4M3**= Finds one scale factor that makes the block errors collectively as small as possible—often improving accuracy for the block maximum (amax)—though some values might be slightly less accurate, the block as a whole retains higher fidelity.

You might ask yourself, why would we ever want to use E8M0? The answer is, when simplicity is the highest priority. E8M0 scale factors have slightly reduced computational complexity (that is, they don’t require an extra per-tensor software scaling factor) and can be adequate for activations and weights that are less sensitive to the precision of scale factors. E4M3 adjusts its scaling factor to each small block of values, allowing for finer fit across wider ranges of inputs. That additional flexibility is what translates to a lower overall rounding error

NVIDIA Blackwell fifth-generation Tensor Core architecture implements NVFP4 and can automatically handle the microscaled FP4 data including the grouping of elements, dynamic scaling, and 4-bit matrix operations.

## Micro-block scaling for efficient model compression

Another key component of NVFP4 is the block floating-point representation, where micro-blocks share a common scaling factor. By reducing the group size from 32 elements to 16 values per block, NVFP4 enables finer-grained scaling than MXFP4.

Large tensors in AI models often mix large and small numbers, and a single “umbrella” scaling can lead to significant quantization errors that degrade model performance. The tighter grouping in NVFP4 offers twice as many opportunities to match the local dynamic range of the data, significantly reducing those errors.

To better understand how NVFP4 improves quantization accuracy, it helps to compare it directly to MXFP4, its predecessor. Both formats rely on grouped value blocks and shared scale factors, but a key innovation in NVFP4 lies in its smaller block size and robust scaling. By cutting the block size in half—from 32 values to 16—NVFP4 enables more localized adaptation to the data’s dynamic range. This makes it easier to preserve small-but-important differences in model weights or activations. Figure 5 illustrates how this works in practice.

How does it work? Inside each 16-value quantized block, every 4-bit encoded value \(x_q\) (between the range of -6 to +6) is reconstructed using:

\(x = x_q \times s\)

In this equation, \(s\) is a higher precision FP8 (E4M3) scaling factor. By recomputing \(s\) for each group of 16 elements, NVFP4 minimizes quantization error at 4-bit precision, while still significantly reducing memory and compute complexity compared to higher-precision formats. This structure makes NVFP4 excel at preserving model intelligence.

## NVFP4 versus FP8: Model performance and memory efficiency

Quantization benefits are driven by two factors: reduced memory burden and simplified compute operations. These two factors reduce pressure on memory bandwidth which can improve output token throughput. It can also improve overall end-to-end latency performance as a result of simplified attention layer computations which yield direct benefits during prefill. For a deep dive into these metrics and how they contribute to the overall inference performance story, see LLM Inference Benchmarking: Fundamental Concepts.

### Model performance

Inference performance optimizations must strive to preserve model intelligence, a balance that NVFP4 is designed to deliver. Figure 6 illustrates this point by comparing the accuracy of DeepSeek-R1-0528 across seven different evaluations, highlighting the minimal accuracy difference between the FP8 and NVFP4 quantized versions of the model.

The analysis showcases the 1% or less accuracy degradation on key language modeling tasks for DeepSeek-R1-0528, when quantized from its original FP8 format to NVFP4 using post-training quantization (PTQ). In the case of AIME 2024, NVFP4 is even 2% better in accuracy.

### Memory

FP8 is supported by Hopper and Blackwell, and has enabled significant benefits in memory and latency/throughput over the previously smallest supported 16-bit floating-point datatypes, FP16/BF16. Now, NVFP4 offers an accurate and compact data type for AI workloads on Blackwell. NVFP4 stores one 4-bit value plus minor overhead of one FP8 scale per 16 values (4.5 bits per value) and one FP32 per tensor second-level scaling factor. This reduces the model memory footprint by approximately 3.5x relative to FP16, and approximately 1.8x compared to FP8.

When this analysis is extended to an NVIDIA GB300 NVL72 rack-scale system, which contains 36 Grace Blackwell Ultra Superchips, each with one NVIDIA Grace CPU and two NVIDIA Blackwell Ultra GPUs, the total memory budget increases to 40 TB per system. This HBM and Grace memory budget partnered with the memory size and accuracy advantages of NVFP4 provide significant benefits for large scale AI inference deployments, particularly in overcoming the challenges posed by test-time scaling.

## FP4 energy efficiency

Reducing precision not only speeds up inference and reduces memory footprints, but also improves performance per watt. Each 4-bit operation requires less energy for data movement and arithmetic than a higher-precision data type. Innovations such as liquid cooling and FP4 support in the Blackwell Tensor Core architecture enable Blackwell and Blackwell Ultra to deliver substantial energy efficiency gains, up to 25x and 50x, respectively, compared to an NVIDIA H100 Tensor Core baseline as shown in Figure 7.

## Get started with NVFP4

The inference ecosystem is rapidly embracing NVFP4 precision to meet the escalating demands of AI. If you’re looking to quantize your model to NVFP4, NVIDIA TensorRT Model Optimizer and LLM Compressor both offer streamlined workflows to do so. It is easier than ever to apply PTQ, QAT, and other advanced quantization techniques to quantize your model to NVFP4.

Once quantized, the NVFP4 model can be easily exported to a Unified Hugging Face Checkpoint and deployed on NVIDIA TensorRT-LLM and vLLM, which offers early NVFP4 support, with upcoming support in SGLang. These frameworks are part of a rapidly expanding ecosystem embracing NVFP4 precision. TensorRT Model Optimizer also supports quantization of non-LLM models and exporting to ONNX format. You don’t have to start from scratch, either: Hugging Face already hosts NVFP4 prequantized checkpoints ready for deployment, including some of the most popular: DeepSeek-R1-0528, Llama 3, and FLUX.1-dev.

Whether you’re optimizing from scratch or adopting prequantized models, NVFP4 is gaining momentum across real-world deployments—with more tutorials and code samples coming soon. Stay tuned.

Learn how NVIDIA Blackwell NVL72 runs 10x faster and delivers 1/10 the token cost for MoE models in this blog.
