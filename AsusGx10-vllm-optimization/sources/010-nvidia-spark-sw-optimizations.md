---
id: 010
title: "New Software and Model Optimizations Supercharge NVIDIA DGX Spark"
url: "https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/"
publisher: "NVIDIA Technical Blog"
retrieved: "2026-06-05"
fetched_by: "openclaw-lxc/trafilatura"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [platform, throughput, quantization, nvfp4]
---

Since its release, NVIDIA has continued to push performance of the Grace Blackwell-powered DGX Spark through continuous software optimization and close collaboration with software partners and the open-source community. These efforts are delivering meaningful gains across inference, training and creative workflows.

At CES 2026, the latest DGX Spark software release, combined with new model updates, and open-source libraries provide significant performance improvements for both DGX Spark and OEM GB10-based systems.

## Scaling large models locally with unified memory and NVFP4

DGX Spark is designed for working locally with large models, featuring 128GB of unified memory in a compact desktop form factor. Two DGX Spark systems can be connected to deliver 256GB of combined memory, enabling developers to run even larger models locally.

The systems connect using ConnectX-7 networking, providing 200 Gbps of bandwidth for fast, low-latency multi-node workloads.

Support for the NVIDIA NVFP4 data format enables next-generation models to dramatically reduce memory footprint while boosting throughput. For example, running the Qwen-235B model using NVFP4 precision and speculative decoding delivers up to a 2.6x performance increase compared with FP8 execution on the same dual DGX Spark configuration.

With FP8 precision, the model saturates the combined memory of two systems, limiting multitasking and overall responsiveness. Quantizing to NVFP4 reduces memory usage by approximately 40% while maintaining high accuracy, allowing developers to achieve FP8-equivalent results with significantly higher performance and enough free memory to run multiple other workloads simultaneously. The result is a noticeably more responsive and productive local AI development experience.

**Open-source collaboration drives additional performance gains**

NVIDIA’s collaborations with open-source software partners continues to push performance further. Llama.cpp updates deliver an average 35% performance uplift when running mixture-of-experts (MoE) models on DGX Spark – improving both throughput and efficiency for popular open-source workflows.

**A powerful desktop platform for creators**

While DGX Spark is an exceptional platform for AI developers, creators can also take advantage of its desktop-class capabilities.

By offloading AI workloads to DGX Spark, creators free up their laptop or PC to remain responsive while content is being generated. With 128GB of unified memory, DGX Spark can run large models such as GPT-OSS-120B or FLUX 2 (90GB) at full precision, enabling the highest-quality outputs without compromise.

Leading diffusion models, including FLUX.2 from Black Forest Labs and Qwen-Image from Alibaba, leverage NVFP4 to reduce memory footprint while delivering higher performance.

AI video generation is particularly well-suited for DGX Spark, as it demands both significant memory and compute. The new LTX-2 audio-video generation model from Lightricks, featuring NVFP8-optimized weights, delivers substantial performance gains over the previous generation, making high-quality video generation practical on the desktop.

**DGX Spark now included in the NVIDIA-Certified Systems program**

The NVIDIA-Certified Systems program validates system performance across a wide range of accelerated graphics, compute and AI workloads. NVIDIA-Certified Systems provide a trusted foundation for AI development, desktop inference, data science, design, and content creation workloads, while also augmenting data center, and cloud resources.

DGX Spark and OEM GB10-based systems are now included in the program, with DGX Spark and partner systems currently in testing.

**New playbooks to help you get started faster**

To help developers get productive immediately, we’re releasing a new set of DGX Spark playbooks that showcase what’s possible with the Blackwell GPU. These playbooks focus on practical, hands-on workflows you can try right away, including:

**Nemotron 3 Nano**: Run NVIDIA’s efficient 30B-parameter MoE model locally for LLM experimentation.**Live VLM WebUI**: Stream webcam input into vision-language models for real-time analysis, with GPU utilization.**Isaac Sim / Lab**: Build and train robotics applications using GPU-accelerated simulation and reinforcement learning.**SGLang**and**vLLM**serving playbooks: Now include a clear model support matrix showing tested and supported models and quantization options.- GPU-accelerated
**quantitative finance**and**genomics**playbooks: Workflows with minimal code changes compared to CPU implementations. **Fine-tune with PyTorch**: Distributed fine-tuning across two DGX Spark systems for LLMs up to 70B parameters using FSDP and LoRA.**Speculative Decoding**: A new EAGLE-3 with GPT-OSS-120B example uses a built-in drafting head instead of a separate draft model, simplifying deployment and increasing token acceptance rates.

Each playbook is designed to be straightforward and dependable, with clear steps, practical troubleshooting guidance, and configurations validated on the latest DGX OS, so you can spend less time setting up and more time building.

**Access your DGX Spark from anywhere with NVIDIA Brev**

With NVIDIA Brev, your DGX Spark is accessible from anywhere through a secure connection. Brev enables developers to easily spin up AI cloud instances and take advantage of Launchables, using a single click to set up AI environments. At CES, updates to Brev demonstrated the ability to register local compute, such as DGX Spark. Once registered with Brev, you can access your DGX Spark from anywhere. You can also securely share access with your team.

Brev enables hybrid deployment between local and cloud models. Using a router layer, you can keep sensitive tasks, such as email or proprietary data processing, on a local open model running on DGX Spark, while routing general reasoning to frontier models in the cloud. See the NVIDIA LLM Router developer example for implementation details.

Brev support for local compute will be previewed at CES with official support coming in the spring 2026.

**Bring your own agent to life**

Ready to take it further? NVIDIA and Hugging Face have partnered to show how you can build a personal desktop AI companion. Using DGX Spark with Reachy Mini, you can create a private AI assistant that processes your data privately. Check out the NVIDIA and Hugging Face tutorial to get started.

Join the DGX Spark developer community, and start your AI-building journey today.
