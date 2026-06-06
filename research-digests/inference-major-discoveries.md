# Major Discoveries in Efficient LLM Inference

> Auto-generated **2026-06-06 00:28 UTC** by the local **Qwen3.6-35B-A3B** model on the GX10 (vLLM endpoint, thinking disabled), from **48 arXiv papers** across **8 subtopics**. Distillation and synthesis are model-generated from abstracts — verify against the linked sources before citing.

## Executive summary

The field has shifted from isolated component optimization to holistic, system-aware architectures that treat inference as a unified pipeline rather than a sequence of independent bottlenecks. A critical cross-cutting discovery is the decoupling of latency from pure language modeling capability, evidenced by the finding that draft model latency, not accuracy, is the primary bottleneck in speculative decoding, and that prefix-aware batching significantly reduces iteration-level bubbles in serving systems. This systemic view reveals that efficiency gains are maximized when hardware-software co-design addresses end-to-end constraints—such as memory bandwidth and KV cache management—simultaneously, rather than optimizing single operators like attention kernels or quantization schemes in isolation.

Furthermore, the interaction between model dynamics and structural constraints is now understood as a fundamental determinant of efficiency. Research demonstrates that quantization, pruning, and KV cache compression are not merely lossy compression problems but are deeply tied to how models form and retain knowledge circuits. For instance, hallucinations emerge concurrently with factual knowledge acquisition, and aggressive KV cache eviction can cause system prompt leakage if not managed with redundancy-aware strategies. This necessitates methods that preserve salient weights and cross-block correlations, such as channel equalization in AWEQ or vector-quantized interfaces in MoE routing, to maintain performance fidelity while drastically reducing resource consumption.

Looking forward, the trajectory points toward adaptive, hybrid, and retrieval-augmented inference systems that dynamically adjust to workload characteristics and hardware constraints. The convergence of speculative decoding with retrieval techniques (RASD), the use of evolutionary search for task-driven KV cache allocation (EvolKV), and the integration of auxiliary tasks like past-token prediction for long-context stability indicate a move away from static, one-size-fits-all models. Future systems will likely prioritize lightweight, predictive mechanisms—such as expert caching and token compression—that enable efficient inference on constrained devices while maintaining the reasoning capabilities and long-context retention required for complex, real-world applications.


## Quantization for LLM inference

Research on LLM inference efficiency and dynamics reveals that quantization performance is fundamentally constrained by the interaction between activation and weight distributions, as well as the structural dependencies across model blocks. While post-training quantization (PTQ) methods like SEPTQ, AWEQ, and FPTQ demonstrate that static, fine-grained, and equalization-based strategies can achieve high-fidelity low-bit inference without retraining, these technical advances must account for the complex learning dynamics of the models. Specifically, the emergence of hallucinations during factual acquisition and the corruption of parametric memory during fine-tuning highlight that quantization and adaptation are not merely optimization problems but are deeply tied to how models form and retain knowledge circuits.

- AWEQ introduces channel equalization to transfer quantization difficulty from activations to weights, enabling robust W8A8 and ultra-low-bit inference without training overhead.
- FPTQ achieves state-of-the-art W4A8 performance by combining fine-grained weight quantization with layerwise activation strategies and logarithmic equalization to mitigate degradation in intractable layers.
- SEPTQ simplifies PTQ into a two-step process using static global importance scoring and column-by-column updates, addressing the high costs of QAT and performance drops in existing PTQ methods.
- Multi-block fine-tuning strategies for weight-only quantization reveal that capturing cross-block weight correlations and minimizing downstream pre-activation errors can improve quantization effectiveness, though benefits vary by model architecture.
- Learning dynamics analysis identifies three distinct phases in factual recall, linking performance plateaus to attention circuit formation and showing that hallucinations emerge concurrently with factual knowledge acquisition.

### Papers

- **How do language models learn facts? Dynamics, curricula and hallucinations** (2025) — Nicolas Zucchet, Jörg Bornschein, Stephanie Chan et al.. [arXiv:2503.21676](https://arxiv.org/abs/2503.21676)
  - Identified three distinct phases of learning in language models on factual recall tasks, characterized by an initial performance plateau before the acquisition of precise factual knowledge.
  - Mechanistically linked the performance plateau to the formation of attention-based circuits that support recall.
  - Demonstrated that imbalanced training data distributions result in shorter learning plateaus compared to balanced distributions.
  - Showed that hallucinations emerge simultaneously with the acquisition of factual knowledge.
  - Found that fine-tuning to integrate new knowledge is challenging because it quickly corrupts the model's existing parametric memories.
- **SEPTQ: A Simple and Effective Post-Training Quantization Paradigm for Large Language Models** (2026) — Han Liu, Haotian Gao, Xiaotong Zhang et al.. [arXiv:2604.10091](https://arxiv.org/abs/2604.10091)
  - Proposes SEPTQ, a simple and effective post-training quantization (PTQ) paradigm for large language models that simplifies the quantization procedure into only two steps.
  - Introduces a method to calculate importance scores for each element in the weight matrix to determine quantization locations in a static global manner.
  - Utilizes a mask matrix representing important locations to quantize and update associated weights column-by-column to obtain the appropriate quantized weight matrix.
  - Demonstrates effectiveness and efficiency by addressing the high training costs of quantization-aware training (QAT) and the performance degradation of existing PTQ methods under low-bit settings.
  - Validates the approach through experiments on various datasets across a suite of models ranging from millions to billions of parameters at different quantization bit-levels.
- **AWEQ: Post-Training Quantization with Activation-Weight Equalization for Large Language Models** (2023) — Baisong Li, Xingwang Wang, Haixiao Xu. [arXiv:2311.01305](https://arxiv.org/abs/2311.01305)
  - Introduces AWEQ, a post-training quantization method for large language models that requires no additional training overhead.
  - Proposes a channel equalization technique that transfers the difficulty of activation quantization to weights to balance quantization difficulties.
  - Refines the equalization method to mitigate quantization bias error and ensure model robustness.
  - Achieves high performance in both ultra-low-bit quantization and 8-bit weight and activation (W8A8) quantization.
  - Demonstrates superior performance compared to all existing post-training quantization methods on popular models such as LLaMA and OPT.
- **FPTQ: Fine-grained Post-Training Quantization for Large Language Models** (2023) — Qingyuan Li, Yifan Zhang, Liang Li et al.. [arXiv:2308.15987](https://arxiv.org/abs/2308.15987)
  - Proposes FPTQ, a novel fine-grained post-training quantization method for large language models that achieves W4A8 (4-bit weights, 8-bit activations) precision.
  - Combines the I/O utilization benefits of 4-bit weight quantization with the acceleration advantages of 8-bit matrix computation to address deployment challenges.
  - Introduces layerwise activation quantization strategies featuring a novel logarithmic equalization technique to mitigate performance degradation in intractable layers.
  - Integrates fine-grained weight quantization with the proposed activation strategies to eliminate the necessity for further fine-tuning.
  - Achieves state-of-the-art W4A8 quantized performance on standard benchmarks for BLOOM, LLaMA, and LLaMA-2 models.
- **Interactions Across Blocks in Post-Training Quantization of Large Language Models** (2024) — Khasmamad Shabanovi, Lukas Wiest, Vladimir Golkov et al.. [arXiv:2411.03934](https://arxiv.org/abs/2411.03934)
  - Identifies that deriving local quantization objectives from global task loss involves simplifications of assuming substructure independence and ignoring subsequent block knowledge.
  - Introduces two multi-block fine-tuning strategies for weight-only quantization of large language models:
  - A method that captures weight correlations across blocks by jointly optimizing multiple quantized blocks.
  - A method that incorporates knowledge of subsequent blocks by minimizing error in downstream pre-activations rather than focusing solely on the quantized block.
  - Demonstrates that the effectiveness of these multi-block strategies varies by model, showing significant benefits for some models while having no impact on others.
- **Unforgettable Generalization in Language Models** (2024) — Eric Zhang, Leshem Chosen, Jacob Andreas. [arXiv:2409.02228](https://arxiv.org/abs/2409.02228)
  - Investigates the behavior of transformer language models when tasks are "forgotten" via fine-tuning on randomized labels, noting that models generate near-random predictions for individual examples in the forgetting training set.
  - Identifies extreme variability in generalization across different tasks, where some tasks (e.g., entailment classification) show robust generalization of forgetting to new instances, while others (e.g., physical commonsense reasoning, scientific question answering) retain accurate performance on new examples similar to the training set.
  - Determines that dataset difficulty is not predictive of whether a behavior can be forgotten.
  - Finds that generalization in forgetting is weakly predicted by the confidence of the model's initial task predictions and the variability of the model's representations of the training data, with low confidence and low variability associated with greater generalization.
  - Demonstrates that random-label forgetting is insensitive to the semantic contents of the training set, evidenced by models trained on science questions with random labels continuing to answer other science questions accurately while producing random labels on entailment classification tasks.


## KV cache optimization

Research into KV cache optimization reveals a divergence between general-purpose compression and specialized reasoning needs, highlighting that aggressive compression can severely degrade instruction-following capabilities and cause system prompt leakage if eviction policies are not carefully managed. While methods like PolyKV and EvolKV demonstrate that significant memory reductions are possible through asymmetric quantization and evolutionary layer-wise allocation without substantial perplexity loss, R-KV emphasizes that reasoning models require redundancy-aware strategies to maintain performance where standard baselines fail. Furthermore, EVICPRESS illustrates that joint optimization of compression and eviction across storage tiers is critical for minimizing latency while preserving generation quality in serving environments.

- PolyKV achieves a 97.7% memory reduction (19.8 GB to 0.45 GB) for 15 concurrent agents on Llama-3-8B using asymmetric compression (int8 Keys, 3-bit Values via FWHT and Lloyd-Max), maintaining +0.57% perplexity degradation.
- R-KV enables reasoning models to achieve 105% of full KV cache performance using only 16% of the cache, reducing memory by 90% and increasing throughput by 6.6X compared to standard chain-of-thought.
- EvolKV utilizes evolutionary search for layer-wise, task-driven compression, surpassing heuristic baselines by up to 7 percentage points on GSM8K and achieving superior code completion performance with only 1.5% of the KV cache budget.
- EVICPRESS introduces a unified utility function to jointly optimize lossy compression and adaptive eviction across storage tiers, improving KV-cache hit rates and reducing latency without degrading generation quality.
- The Pitfalls of KV Cache Compression identifies that compression methods, instruction order, and KV eviction bias contribute to system prompt leakage, proposing modified eviction policies to mitigate rapid degradation in multi-instruction prompting.

### Papers

- **PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference** (2026) — Ishan Patel, Ishan Joshi. [arXiv:2604.24971](https://arxiv.org/abs/2604.24971)
  - Introduces PolyKV, a system enabling multiple concurrent inference agents to share a single, asymmetrically compressed KV cache pool by writing a compressed cache once and injecting it into N independent agent contexts via HuggingFace DynamicCache objects.
  - Implements asymmetric compression where Keys are quantized at int8 (q8_0) to preserve softmax stability, while Values are compressed using TurboQuant MSE, involving a Fast Walsh-Hadamard Transform (FWHT) rotation followed by 3-bit Lloyd-Max quantization with centroids tuned to N(0,1).
  - Achieves a stable 2.91x compression ratio across all evaluated configurations, including model scales (SmolLM2-1.7B-Instruct and Llama-3-8B-Instruct), context lengths (600-7,194 tokens), and up to 15 concurrent agents.
  - Reduces KV cache memory from 19.8 GB to 0.45 GB (a 97.7% reduction) on Llama-3-8B with 15 agents sharing a 4K-token context.
  - Maintains only +0.57% perplexity degradation and a mean BERTScore F1 of 0.928 in the 15-agent, 4K-token scenario.
  - Demonstrates that perplexity delta does not grow with agent count and improves as context length increases, inverting to -0.26% at 1,851 coherent tokens.
- **Lossless KV Cache Compression to 2%** (2024) — Zhen Yang, J. N. Han, Kan Wu et al.. [arXiv:2410.15252](https://arxiv.org/abs/2410.15252)
  - Introduces Cross-Layer Latent Attention (CLLA), a novel architecture designed to compress the Key-Value (KV) cache to less than 2% of its original size.
  - Integrates attention head/dimension reduction, layer sharing, and quantization techniques into a unified compression framework.
  - Achieves lossless performance on most tasks while maintaining minimal KV cache usage.
- **The Pitfalls of KV Cache Compression** (2025) — Alex Chen, Renato Geh, Aditya Grover et al.. [arXiv:2510.00231](https://arxiv.org/abs/2510.00231)
  - Evaluates five KV cache compression methods (StreamingLLM, SnapKV, TOVA, H2O, and K-Norm) on Llama3.1 8B and Qwen2.5 14B models under multi-instruction prompting using the IFEval benchmark.
  - Demonstrates that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM.
  - Empirically highlights system prompt leakage as a case study to show the impact of compression on leakage and general instruction-following.
  - Identifies three specific factors contributing to system prompt leakage: compression method, instruction order, and KV eviction bias.
  - Proposes simple changes to KV cache eviction policies to reduce the impact of these factors and improve overall performance in multi-instruction tasks.
- **R-KV: Redundancy-aware KV Cache Compression for Reasoning Models** (2025) — Zefan Cai, Wen Xiao, Hanshi Sun et al.. [arXiv:2505.24133](https://arxiv.org/abs/2505.24133)
  - Proposes R-KV, a redundancy-aware KV cache compression method specifically designed for reasoning models to address excessive output lengths and reasoning failures in existing compression approaches.
  - Achieves nearly 100% of full KV cache performance using only 10% of the KV cache, significantly outperforming existing baselines which reach only 60% performance.
  - Attains 105% of full KV cache performance while utilizing only 16% of the KV cache.
  - Reduces memory usage by 90% and increases inference throughput by 6.6X compared to standard chain-of-thought reasoning.
  - Demonstrates consistent outperformance over existing KV cache compression baselines across two mathematical reasoning datasets.
- **EvolKV: Evolutionary KV Cache Compression for LLM Inference** (2025) — Bohan Yu, Yekun Chai. [arXiv:2509.08315](https://arxiv.org/abs/2509.08315)
  - Proposes EvolKV, an adaptive framework for layer-wise, task-driven KV cache compression that jointly optimizes memory efficiency and task performance.
  - Reformulates cache allocation as a multi-objective optimization problem and leverages evolutionary search to dynamically configure layer budgets while directly maximizing downstream performance.
  - Demonstrates outperformance of all baseline methods across a wide range of KV cache budgets on 11 tasks, particularly on long-context tasks.
  - Surpasses heuristic baselines by up to 7 percentage points on the GSM8K benchmark.
  - Achieves superior performance over the full KV cache setting on code completion while utilizing only 1.5% of the original KV cache budget.
- **EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving** (2025) — Shaoting Feng, Yuhan Liu, Hanchen Li et al.. [arXiv:2512.14946](https://arxiv.org/abs/2512.14946)
  - Proposes EVICPRESS, a KV-cache management system that jointly optimizes lossy compression and adaptive eviction across multiple storage tiers to minimize average generation latency without degrading quality.
  - Introduces a unified utility function that quantifies the combined impact of quality and delay for lossy compression or eviction decisions across all contexts.
  - Implements a profiling module that periodically updates utility function scores for all possible eviction-compression configurations for all contexts.
  - Utilizes a fast heuristic to rearrange KV caches across storage tiers based on the utility function scores to maximize overall system utility.
  - Achieves higher KV-cache hit rates on fast devices (resulting in lower delay) and preserves high generation quality compared to baselines that only evict or only compress KV cache.


## Speculative decoding

Research on speculative decoding has evolved from establishing baseline acceleration frameworks to optimizing architectural compatibility, hardware efficiency, and hybrid model utilization. Major themes include the critical role of draft model latency over language modeling capability, the significant impact of model composition patterns (parallel vs. sequential hybrids) on acceptance rates, and the necessity of specialized memory management and retrieval integration to overcome domain-specific and hardware constraints.

- Component-aware self-speculative decoding exploits internal architectural heterogeneity in hybrid models, revealing that parallel hybrids (e.g., Falcon-H1) achieve significantly higher acceptance rates ($\alpha = 0.68$) than sequential hybrids (e.g., Qwen3.5, $\alpha = 0.038$), with performance strongly correlated to perplexity degradation ratios.
- The "Decoding Speculative Decoding" study identifies draft model latency as the primary bottleneck rather than language modeling capability, enabling the design of hardware-efficient draft models that achieve 111% higher throughput across LLaMA and SFT variants.
- Speculative Speculative Decoding (SSD) and Saguaro parallelize speculation and verification operations, eliminating drafting overhead when verification outcomes are predicted correctly, resulting in a 30% average speedup over optimized baselines and up to 5x over autoregressive decoding.
- RASD integrates retrieval techniques with model-based speculation using tree pruning and longest prefix matching, achieving state-of-the-art acceleration across diverse tasks by addressing out-of-domain inefficiencies and low acceptance lengths.
- SpecMemo introduces device-aware inference for memory-constrained environments, reducing generation-memory by 65% while maintaining 96% throughput on mobile GPUs, and enables big-model inference through novel batched speculative decoding across multiple constrained GPUs.

### Papers

- **Component-Aware Self-Speculative Decoding in Hybrid Language Models** (2026) — Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó. [arXiv:2605.01106](https://arxiv.org/abs/2605.01106)
  - Introduces component-aware self-speculative decoding, the first method to exploit internal architectural heterogeneity in hybrid language models by isolating the SSM/linear-attention subgraph as a zero-cost internal draft.
  - Evaluates the method on Falcon-H1 (parallel Mamba-2 + attention) and Qwen3.5 (sequential interleaved linear and attention layers), comparing them against a pure Transformer control (Qwen2.5).
  - Achieves an acceptance rate of $\alpha = 0.68$ at draft length $k=2$ under greedy decoding for parallel hybrids (Falcon-H1).
  - Yields an acceptance rate of $\alpha = 0.038$ for sequential hybrids (Qwen3.5), representing an 18x performance gap compared to parallel hybrids.
  - Demonstrates scale invariance, where Falcon-H1 at 3B parameters reproduces the acceptance rates observed at 0.5B.
  - Establishes a predictive relationship between perplexity degradation ratios and speculative viability: a 3.15x ratio maps to $\alpha = 0.37$ at $k=4$, while an 81.96x ratio maps to $\alpha = 0.019$.
  - Finds that for sequential hybrids, generic LayerSkip achieves 12x higher acceptance rates than the proposed component-aware strategy.
  - Concludes that the composition pattern of hybrid models, rather than just the presence of alternative components, determines the viability of component-level self-speculation.
- **Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation** (2022) — Heming Xia, Tao Ge, Peiyi Wang et al.. [arXiv:2203.16487](https://arxiv.org/abs/2203.16487)
  - Proposes Speculative Decoding (SpecDec), a framework that applies speculative execution to accelerate autoregressive (AR) decoding in sequence-to-sequence (seq2seq) models.
  - Introduces Spec-Drafter, an independent model specifically optimized for efficient and accurate token drafting.
  - Introduces Spec-Verification, a reliable method for efficiently verifying drafted tokens within the decoding paradigm.
  - Achieves approximately $5\times$ speedup for popular Transformer architectures on machine translation and abstractive summarization tasks.
  - Maintains generation quality comparable to beam search decoding while achieving the stated speedup.
  - Demonstrates that the draft-then-verify paradigm can exceed the previously assumed $1.4\times$ to $2\times$ speedup limit.
  - Identifies and demonstrates three additional practical advantages of SpecDec for accelerating generative models in real-world applications.
- **Decoding Speculative Decoding** (2024) — Minghao Yan, Saurabh Agarwal, Shivaram Venkataraman. [arXiv:2402.01528](https://arxiv.org/abs/2402.01528)
  - Conducted a detailed study comprising over 350 experiments with LLaMA-65B and OPT-66B to delineate factors affecting speculative decoding performance.
  - Identified that speculative decoding performance depends heavily on the latency of the draft model.
  - Demonstrated that the draft model's capability in language modeling does not correlate strongly with its performance in speculative decoding.
  - Designed hardware-efficient draft models based on the identified design space.
  - Achieved 111% higher throughput with the newly designed draft model compared to existing draft models.
  - Validated that the approach generalizes to all LLaMA models (1, 2, 3.1) and supervised fine-tuned models.
- **Speculative Speculative Decoding** (2026) — Tanishq Kumar, Tri Dao, Avner May. [arXiv:2603.03251](https://arxiv.org/abs/2603.03251)
  - Introduces Speculative Speculative Decoding (SSD), a method that parallelizes the speculation and verification operations inherent in standard speculative decoding.
  - Implements a mechanism where the draft model predicts likely verification outcomes during ongoing verification to pre-emptively prepare speculations, eliminating drafting overhead when outcomes match predictions.
  - Identifies three key challenges presented by SSD and proposes principled methods to solve each.
  - Develops Saguaro, an optimized SSD algorithm.
  - Achieves an average speedup of 30% compared to optimized speculative decoding baselines.
  - Achieves up to 5x speedup compared to autoregressive decoding using open source inference engines.
- **RASD: Retrieval-Augmented Speculative Decoding** (2025) — Guofeng Quan, Wenfeng Feng, Chuzhan Hao et al.. [arXiv:2503.03434](https://arxiv.org/abs/2503.03434)
  - Proposes RASD (Retrieval-Augmented Speculative Decoding), a method that integrates retrieval techniques with model-based speculative decoding to address out-of-domain inefficiencies and low acceptance lengths.
  - Introduces a tree pruning method based on the draft model's probability distribution to construct an optimal retrieval tree.
  - Employs a longest prefix matching algorithm to merge the draft model's generated tree with the retrieval tree, creating a unified tree for verification.
  - Achieves state-of-the-art inference acceleration across tasks including DocQA, Summary, Code, and In-Domain QA.
  - Demonstrates strong scalability by seamlessly integrating with various speculative decoding approaches, including both generation-based and retrieval-based methods.
- **SpecMemo: Speculative Decoding is in Your Pocket** (2025) — Selin Yildirim, Deming Chen. [arXiv:2506.01986](https://arxiv.org/abs/2506.01986)
  - Introduces SpecMemo, a device-aware inference engine designed to enable speculative decoding on memory-constrained devices like mobile GPUs by controlling memory allocations at finer levels.
  - Theoretically models the memory footprint of speculative decoding to determine a lower bound on the required memory budget while retaining speedup.
  - Empirically balances minimizing redundant memory allocations for rejected candidate tokens with maintaining competitive performance gains.
  - Maintains 96% of overall throughput from speculative decoding on MT-Bench while reducing generation-memory by 65% on a single Nvidia Titan RTX.
  - Facilitates big-model inference by distributing the Llama-2-70B-Chat model across multiple constrained GPUs.
  - Provides novel batched speculative decoding to increase the usability of multiple small server GPUs, demonstrating a 2x speedup over distributed and batched vanilla decoding.


## Efficient attention & kernels

Research on efficient attention mechanisms centers on optimizing the IO-bound and compute-intensive nature of FlashAttention through hardware-aware algorithmic refinements, quantization, and architectural fusion. Major themes include reducing memory bandwidth requirements via tiling and IO-awareness, minimizing computational overhead by fusing operations like softmax and exponentiation, and enabling lower-precision inference through INT8 quantization and token compression. These approaches collectively aim to preserve numerical stability and accuracy while significantly accelerating inference and reducing hardware resource consumption.

- INT-FlashAttention introduces the first fully INT8 quantized attention operator, achieving 72% faster inference and an 82% reduction in quantization error compared to FP16/FP8 FlashAttention.
- SystolicAttention fuses the entire FlashAttention algorithm within a single systolic array, achieving 1.77x and 4.83x higher FLOPs/s utilization than AWS Neuron-v2 and Google TPUv5e, respectively.
- FLASH-D simplifies the FlashAttention formulation by hiding softmax division within non-linear evaluations, resulting in a 22.8% area reduction and 20.3% power savings in 28nm implementations.
- Low-Cost FlashAttention utilizes fused exponential and multiplication hardware operators (ExpMul), yielding a 28.8% average area reduction and 17.6% power reduction in 28nm ASIC technology.
- Representation Shift enables training-free, model-agnostic token compression compatible with fused attention kernels, providing speedups of up to 5.5% in video-text retrieval and 4.4% in video QA.

### Papers

- **INT-FlashAttention: Enabling Flash Attention for INT8 Quantization** (2024) — Shimao Chen, Zirui Liu, Zhiying Wu et al.. [arXiv:2409.16997](https://arxiv.org/abs/2409.16997)
  - Introduces INT-FlashAttention, the first INT8 quantization architecture compatible with the forward workflow of FlashAttention.
  - Implements a prototype using fully INT8 activations and general matrix-multiplication (GEMM) kernels, creating the first attention operator with fully INT8 input.
  - Provides a general token-level post-training quantization framework that supports data formats such as INT4.
  - Achieves 72% faster inference speed compared to standard FlashAttention with FP16 and FP8 data formats.
  - Reduces quantization error by 82% compared to standard FlashAttention with FP16 and FP8 data formats.
- **SystolicAttention: Fusing FlashAttention within a Single Systolic Array** (2025) — Jiawei Lin, Yuanlong Li, Guokai Chen et al.. [arXiv:2507.11331](https://arxiv.org/abs/2507.11331)
  - Proposes FSA, an enhanced systolic array architecture that executes the entire FlashAttention algorithm without external vector units.
  - Introduces SystolicAttention, an optimized kernel enabling fine-grained, element-wise overlapping of FlashAttention operations to maximize array utilization.
  - Preserves the original floating-point operation order of FlashAttention while integrating softmax and matrix multiplication within the systolic array.
  - Achieves 1.77x higher attention FLOPs/s utilization compared to AWS Neuron-v2.
  - Achieves 4.83x higher attention FLOPs/s utilization compared to Google TPUv5e.
  - Synthesizes FSA in 16 nm technology at 1.5 GHz with only a 12% area overhead compared to a standard weight-stationary systolic array.
- **FLASH-D: FlashAttention with Hidden Softmax Division** (2025) — Kosmas Alexandridis, Vasileios Titopoulos, Giorgos Dimitrakopoulos. [arXiv:2505.14201](https://arxiv.org/abs/2505.14201)
  - Presents FLASH-D, a mathematically equivalent and simplified formulation of the FlashAttention kernel that preserves essential properties for efficient tiled implementation.
  - Introduces a mechanism to hide softmax division within other non-linear function evaluations.
  - Achieves inherently numerically stable computation of exponentials, eliminating the need for maximum value subtraction.
  - Reduces computational cost without introducing numerical approximations to the FlashAttention kernel.
  - Demonstrates hardware implementation results at 28nm showing a 22.8% average reduction in area compared to state-of-the-art parallel hardware architectures.
  - Demonstrates a 20.3% average reduction in power consumption compared to state-of-the-art parallel hardware architectures without any performance penalty.
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (2022) — Tri Dao, Daniel Y. Fu, Stefano Ermon et al.. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
  - Proposes FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes between GPU high bandwidth memory (HBM) and on-chip SRAM.
  - Analyzes IO complexity to demonstrate that FlashAttention requires fewer HBM accesses than standard attention and is optimal for a range of SRAM sizes.
  - Extends FlashAttention to block-sparse attention, creating an approximate attention algorithm faster than existing methods.
  - Achieves a 15% end-to-end wall-clock speedup on BERT-large (sequence length 512) compared to the MLPerf 1.1 training speed record.
  - Achieves a 3x speedup on GPT-2 (sequence length 1K).
  - Achieves a 2.4x speedup on long-range arena (sequence length 1K-4K).
  - Yields 0.7 better perplexity on GPT-2 and a 6.4-point lift on long-document classification.
  - Enables the first Transformers to achieve better-than-chance performance on the Path-X challenge.
- **Representation Shift: Unifying Token Compression with FlashAttention** (2025) — Joonmyung Choi, Sanghyeok Lee, Byungoh Ko et al.. [arXiv:2508.00367](https://arxiv.org/abs/2508.00367)
  - Proposes Representation Shift, a training-free, model-agnostic metric that measures the degree of change in each token's representation to enable token compression compatible with FlashAttention.
  - Eliminates the need for attention maps and retraining by integrating token compression directly with fused attention kernels like FlashAttention.
  - Generalizes the proposed method beyond Transformers to apply to CNNs and state space models.
  - Achieves significant speedups of up to 5.5% in video-text retrieval and 4.4% in video QA through effective token compression.
- **Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators** (2025) — Kosmas Alexandridis, Vasileios Titopoulos, Giorgos Dimitrakopoulos. [arXiv:2505.14314](https://arxiv.org/abs/2505.14314)
  - Proposes fused exponential and multiplication hardware operators (ExpMul) to optimize the floating-point-based FlashAttention kernel.
  - Achieves an average area reduction of 28.8% compared to state-of-the-art hardware architectures with separate exponential and vector multiplication operators.
  - Achieves an average power reduction of 17.6% compared to state-of-the-art hardware architectures with separate exponential and vector multiplication operators.
  - Demonstrates these improvements through implementation in 28nm ASIC technology.


## Serving systems & batching

Research on LLM serving systems highlights a shift from isolated optimization to holistic, workload-aware architectures that address end-to-end bottlenecks, including KV cache management, memory bandwidth, and scheduling inefficiencies. Key themes include the critical role of prefix-aware batching to minimize iteration-level bubbles, the strategic disaggregation of attention computation to balance memory and compute loads, and the necessity of hardware-software co-design to maximize resource utilization under bursty, real-world traffic patterns.

- AlignedServe introduces prefix-aware batching and a GPU-Prefetch-For-GPU architecture, achieving up to 1.98x higher decoding throughput and 7.4x latency reduction by grouping requests with similar KV-cache lengths and minimizing CPU-to-GPU transfer latency.
- ScaleLLM delivers a 4.3x speedup over vLLM and 1.5x higher throughput by optimizing end-to-end efficiency across system-wide bottlenecks rather than focusing solely on inference.
- Adrenaline boosts memory capacity utilization by 2.28x through attention disaggregation, offloading memory-intensive decoding attention to the compute-intensive prefill phase while increasing decoding batch sizes.
- ADOR provides an automated design exploration framework that identifies hardware architectures tailored for heterogeneous dataflows, achieving 2.51x higher QoS and 4.01x better area efficiency than NVIDIA A100 at high batch sizes.
- BurstGPT reveals that realistic workload burstiness and concurrency variations significantly impact efficiency, stability, and reliability, demonstrating that generalizations in KV cache management and scheduling require evaluation under diverse, real-world traffic patterns.

### Papers

- **On the throughput of the common target area for robotic swarm strategies -- extended version** (2022) — Yuri Tavares dos Passos, Xavier Duquesne, Leandro Soriano Marcolino. [arXiv:2201.09335](https://arxiv.org/abs/2201.09335)
  - Proposes "common target area throughput" as a measure for evaluating access efficiency in robotic swarms as the number of robots rises.
  - Introduces "target area asymptotic throughput" to provide a finite metric for comparing algorithms, avoiding the infinity issues associated with arrival time per robot.
  - Formally evaluates three theoretical strategies for accessing a circular target area:
  - Forming parallel queues towards the target area.
  - Forming a hexagonal packing through a corridor leading to the target.
  - Executing multiple curved trajectories towards the boundary of the target area.
  - Calculates both fixed-time throughput and asymptotic throughput for the three proposed strategies.
  - Corroborates analytical results with simulations, demonstrating that strategies with higher throughput correspond to lower arrival times per number of robots.
  - Concludes that throughput is a suitable metric for comparing congestion algorithms even in the absence of closed asymptotic equations.
- **ADOR: A Design Exploration Framework for LLM Serving with Enhanced Latency and Throughput** (2025) — Junsoo Kim, Hunjong Lee, Geonwoo Ko et al.. [arXiv:2503.04253](https://arxiv.org/abs/2503.04253)
  - Proposes ADOR, an automated design exploration framework that identifies and recommends hardware architectures tailored for Large Language Model (LLM) serving.
  - Utilizes predefined architecture templates specialized for heterogeneous dataflows to optimize the balance between throughput and latency.
  - Achieves a 2.51x higher Quality-of-Service (QoS) compared to the NVIDIA A100 at high batch sizes.
  - Demonstrates a 4.01x improvement in area efficiency relative to the NVIDIA A100 at high batch sizes.
- **Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation** (2025) — Yunkai Liang, Zhangyu Chen, Pengfei Zuo et al.. [arXiv:2503.20552](https://arxiv.org/abs/2503.20552)
  - Proposes Adrenaline, an attention disaggregation and offloading mechanism that moves part of the attention computation from the memory-intensive decoding phase to the compute-intensive prefill phase.
  - Introduces three key techniques: low-latency decoding synchronization, resource-efficient prefill colocation, and load-aware offloading scheduling.
  - Achieves improved memory capacity and bandwidth utilization in prefill instances by offloading memory-bound decoding attention computation.
  - Increases decoding batch sizes to enhance compute utilization in decoding instances.
  - Achieves 2.28x higher memory capacity utilization (based on the provided text fragment).
- **BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems** (2024) — Yuxin Wang, Yuhan Chen, Zeyu Li et al.. [arXiv:2401.17644](https://arxiv.org/abs/2401.17644)
  - Presents BurstGPT, an LLM serving workload dataset comprising 10.31 million traces from regional Azure OpenAI GPT services collected over 213 days.
  - Captures user request concurrency characteristics, revealing burstiness variations and diversified concurrency patterns across different services and model types.
  - Analyzes user conversation patterns by measuring counts and intervals within conversations to support service optimizations.
  - Examines model response lengths to demonstrate statistical relations between requests and their auto-regressive responses.
  - Documents system response failures in conversation and API services, highlighting intensive resource needs and limited availability of LLM services.
  - Demonstrates through demo evaluation that frequent variations in BurstGPT reveal declines in efficiency, stability, or reliability in realistic LLM serving scenarios.
  - Identifies that the generalization of KV cache management, scheduling, and disaggregation optimizations can be improved under realistic workload evaluation.
- **ScaleLLM: A Resource-Frugal LLM Serving Framework by Optimizing End-to-End Efficiency** (2024) — Yuhang Yao, Han Jin, Alay Dilipbhai Shah et al.. [arXiv:2408.00008](https://arxiv.org/abs/2408.00008)
  - Identifies major bottlenecks impacting end-to-end latency in LLM serving systems, revealing that efficiency challenges extend beyond LLM inference to require a holistic system view.
  - Proposes ScaleLLM, a resource-frugal serving framework designed to optimize end-to-end efficiency by addressing identified system-wide bottlenecks.
  - Achieves a 4.3x speedup over vLLM under conditions of 64 concurrent requests.
  - Outperforms state-of-the-art serving systems with 1.5x higher throughput.
- **AlignedServe: Orchestrating Prefix-aware Batching to Build a High-throughput and Computing-efficient LLM Serving System** (2026) — Fengyao Bai, Hongbin Zhang, Zhitao Chen et al.. [arXiv:2605.23389](https://arxiv.org/abs/2605.23389)
  - Proposes AlignedServe, an LLM serving framework centered on prefix-aware batching that groups requests with similar KV-cache lengths to reduce iteration-level bubbles.
  - Utilizes large CPU memory to maintain a sufficient pool of in-flight requests for effective batching.
  - Implements a batch-level scheduling policy to minimize batch-level bubbles.
  - Introduces a GPU-Prefetch-For-GPU architecture where one GPU prefetches KV cache for another to reduce CPU-to-GPU transfer latency.
  - Achieves up to 1.98 times improvement in decoding throughput and up to 7.4 times reduction in latency compared to state-of-the-art systems.


## Mixture-of-Experts inference

Research on Mixture-of-Experts (MoE) systems highlights a divergence between inference optimization and theoretical generalization, where recent work prioritizes memory efficiency and routing stability over raw throughput gains in standard hardware. Key themes include the development of lightweight, predictive caching mechanisms for single-GPU deployment, the stabilization of expert routing through Lipschitz regularization and vector-quantized interfaces, and the identification of fundamental tradeoffs between stability and generalization floors. Theoretical analyses further reveal that sample complexity in MoE models is heavily dictated by identifiability conditions, with linear experts requiring exponentially more data than non-linear structures, while empirical studies suggest that naive conditional routing often fails to yield inference speedups on modern hardware without specialized architectural adjustments.

- ExpertFlow introduces a transformer-based routing path predictor and predictive expert cache that reduces GPU memory usage by up to 93.72% and improves inference throughput by up to 10x on single-GPU environments.
- STEM-GNN stabilizes GNN inference via a vector-quantized token interface and Lipschitz-regularized heads, achieving a robust balance of clean performance, robustness, and stability against feature and edge corruptions.
- Post-training expert pruning and skipping techniques enable task-agnostic and task-specific sparsification, reducing model size and increasing inference speed without requiring specialized hardware.
- Theoretical convergence analysis establishes that estimating experts satisfying "strong identifiability" requires polynomially many data points, whereas linear experts violating this condition necessitate exponentially many due to intrinsic parameter interactions.
- Tensor-variate MoE models preserve multi-dimensional data structure for regression tasks, demonstrating efficiency and effectiveness in real-time myographic control of robotic hands with limited training data.

### Papers

- **ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling** (2024) — Xin He, Shunkang Zhang, Kaijie Tang et al.. [arXiv:2410.17954](https://arxiv.org/abs/2410.17954)
  - Proposes ExpertFlow, a lightweight Mixture-of-Experts (MoE) inference system designed for memory-constrained single-GPU environments.
  - Introduces a transformer-based routing path predictor that estimates expert usage across all MoE layers in a single forward pass.
  - Implements a token scheduler that groups tokens with similar predicted routes to improve expert utilization.
  - Develops a predictive expert cache that loads only required experts and corrects mispredictions at runtime.
  - Achieves a reduction in GPU memory usage by up to 93.72% compared to baselines.
  - Improves inference throughput by up to 10x over strong offloading baselines on a single GPU.
- **Generalizing GNNs with Tokenized Mixture of Experts** (2026) — Xiaoguang Guo, Zehong Wang, Jiazheng Li et al.. [arXiv:2602.09258](https://arxiv.org/abs/2602.09258)
  - Identifies a fundamental tradeoff in static GNN inference where improving stability by reducing reliance on shift-sensitive features creates an irreducible worst-case generalization floor.
  - Proposes STEM-GNN, a pretrain-then-finetune framework featuring a mixture-of-experts encoder for diverse computation paths.
  - Introduces a vector-quantized token interface to stabilize encoder-to-head signals and mitigate routing fluctuations.
  - Implements a Lipschitz-regularized head to bound output amplification and prevent perturbation-induced instability.
  - Achieves a stronger three-way balance of clean performance, robustness, and stability across nine node, link, and graph benchmarks.
  - Demonstrates improved robustness to degree and homophily shifts, as well as feature and edge corruptions, while maintaining competitive performance on clean graphs.
- **Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models** (2024) — Xudong Lu, Qi Liu, Yuhui Xu et al.. [arXiv:2402.14800](https://arxiv.org/abs/2402.14800)
  - Proposes post-training approaches for task-agnostic and task-specific expert pruning and skipping in Mixture-of-Experts (MoE) Large Language Models.
  - Introduces plug-and-play expert-level sparsification techniques designed to enhance deployment efficiency without relying on specifically designed hardware.
  - Demonstrates that the proposed methods simultaneously reduce model sizes and increase inference speed while maintaining satisfactory performance across a wide range of tasks.
- **Convergence Rates for Softmax Gating Mixture of Experts** (2025) — Huy Nguyen, Nhat Ho, Alessandro Rinaldo. [arXiv:2503.03213](https://arxiv.org/abs/2503.03213)
  - Performs a convergence analysis of parameter estimation and expert estimation for Mixture of Experts (MoE) models equipped with standard softmax gating, dense-to-sparse gating, and hierarchical softmax gating.
  - Establishes that estimating experts satisfying the proposed "strong identifiability" condition, such as commonly used two-layer feed-forward networks, requires polynomially many data points.
  - Demonstrates that estimating linear experts, which violate the strong identifiability condition, necessitates exponentially many data points due to intrinsic parameter interactions.
  - Expresses the intrinsic parameter interactions causing exponential sample complexity in the language of partial differential equations.
  - Provides theoretical insights into the design of sample-efficient expert structures.
- **Tensor-variate Mixture of Experts for Proportional Myographic Control of a Robotic Hand** (2019) — Noémie Jaquier, Robert Haschke, Sylvain Calinon. [arXiv:1902.11104](https://arxiv.org/abs/1902.11104)
  - Proposes a tensor-variate mixture-of-experts model for regression of tensor-valued data that preserves the underlying structure of multi-dimensional data, avoiding the dimensionality increase and overfitting associated with classical vector-flattening methods.
  - Demonstrates the model's efficiency and effectiveness in scenarios with limited training data through evaluation on artificially generated data.
  - Validates the approach via offline and real-time experiments for recognizing hand movements from tactile myography to enable proportional myographic control of a robotic hand.
- **Mixture-of-Experts Models in Vision: Routing, Optimization, and Generalization** (2026) — Adam Rokah, Daniel Veress, Caleb Caulk et al.. [arXiv:2601.15021](https://arxiv.org/abs/2601.15021)
  - Compares dense, SoftMoE, and SparseMoE classifier heads on the CIFAR10 dataset under comparable model capacity.
  - Demonstrates that both MoE variants achieve slightly higher validation accuracy than the dense baseline.
  - Shows that regularization maintains balanced expert utilization in MoE models, avoiding expert collapse.
  - Computes Hessian-based sharpness metrics (largest eigenvalue and trace of the loss Hessian) on training and test data at convergence.
  - Finds that SoftMoE exhibits higher sharpness than Dense and SparseMoE, which lie in a similar curvature regime.
  - Observes that despite differences in curvature, all models achieve comparable generalization performance.
  - Identifies qualitative differences in non-local behavior under finite parameter perturbations between dense and MoE models via loss surface perturbation analysis.
  - Shows that naively implemented conditional routing does not yield inference speedups on modern hardware at the studied scale.


## Long-context inference

Recent research on long-context inference demonstrates that efficiency and stability can be achieved through hierarchical memory routing, sparse attention gating, and specialized auxiliary training objectives. Key innovations include Memory-Keyed Attention (MKA) and Gated Sparse Attention (GSA), which leverage dynamic routing and adaptive sparsity to reduce latency and eliminate attention sinks without sacrificing perplexity. Concurrently, methodological advances in diffusion policies and video generation utilize past-token prediction and context-causal attention to preserve temporal dependencies and scene consistency. Theoretical work further clarifies that long-context learning is statistically viable without requiring mixing properties, while comprehensive surveys map the architectural and infrastructural landscape of extending LLM context windows to millions of tokens.

- Memory-Keyed Attention (MKA) and its FastMKA variant integrate multi-level KV caches with dynamic routing, achieving up to 5x faster training throughput and 1.8x lower evaluation latency compared to baselines.
- Gated Sparse Attention (GSA) combines sparse mechanisms with gated attention to deliver a 12-16x speedup at 128K context, reduces attention to the first token from 47% to under 4%, and cuts loss spikes by 98%.
- Past-Token Prediction (PTP) serves as an auxiliary task for diffusion policies, explicitly regularizing historical information retention and enabling self-verification at test time with minimal visual reliance.
- Long Context Tuning (LCT) expands video diffusion models to learn scene-level consistency by extending full attention across shots and using context-causal attention for efficient auto-regressive generation.
- Theoretical analysis establishes sample complexity bounds for long-context linear system identification, proving that learning is not hindered by slow mixing and identifying statistical advantages of shorter contexts in misspecified stable systems.

### Papers

- **MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning** (2026) — Dong Liu, Yanxuan Yu, Ben Lengerich et al.. [arXiv:2603.20586](https://arxiv.org/abs/2603.20586)
  - Proposes Memory-Keyed Attention (MKA), a hierarchical attention mechanism that integrates multi-level KV caches (local, session, and long-term) and dynamically routes attention across them.
  - Introduces Route-Fused MKA (FastMKA), a broadcast-routed variant that fuses memory sources before attention computation to improve efficiency.
  - Achieves comparable perplexity to Multi-Latent Attention (MLA) while delivering up to 5x faster training throughput.
  - Reduces evaluation latency by 1.8x compared to baseline methods.
- **Learning Long-Context Diffusion Policies via Past-Token Prediction** (2025) — Marcel Torne, Andy Tang, Yuejiang Liu et al.. [arXiv:2505.09561](https://arxiv.org/abs/2505.09561)
  - Proposes Past-Token Prediction (PTP), an auxiliary task where the policy learns to predict past action tokens alongside future ones to explicitly regularize the retention of historical information.
  - Identifies that recent diffusion policies often fail to capture essential dependencies between past and future actions, contrasting with the traditional "copycat problem" of over-relying on prior actions.
  - Demonstrates that PTP significantly improves temporal modeling in the policy head with minimal reliance on visual representations.
  - Introduces a multistage training strategy that pre-trains the visual encoder with short contexts and fine-tunes the policy head using cached long-context embeddings to reduce memory and computational overhead.
  - Extends PTP into a self-verification mechanism at test time that enables the policy to score and select candidate actions.
- **Thus Spake Long-Context Large Language Model** (2025) — Xiaoran Liu, Ruixiao Li, Mianqiu Huang et al.. [arXiv:2502.17129](https://arxiv.org/abs/2502.17129)
  - Provides a comprehensive survey of long-context Large Language Model (LLM) technologies, covering the full spectrum from architecture and infrastructure to training and evaluation.
  - Illustrates the lifecycle of long-context LLMs through four specific perspectives: architecture, infrastructure, training, and evaluation.
  - Documents the breakthrough extension of LLM context lengths to millions of tokens over the past two years.
  - Identifies and presents 10 unanswered questions currently facing the field of long-context LLMs.
  - Offers a systematic introduction to research on long-context LLMs, including associated video and GitHub resources.
- **Gated Sparse Attention: Combining Computational Efficiency with Training Stability for Long-Context Language Models** (2026) — Alfred Shen, Aaron Shen. [arXiv:2601.15305](https://arxiv.org/abs/2601.15305)
  - Proposes Gated Sparse Attention (GSA), an architecture combining sparse attention mechanisms with gated attention variants to address computational efficiency and training stability.
  - Incorporates a gated lightning indexer using sigmoid activations to produce bounded, interpretable selection scores.
  - Implements an adaptive sparsity controller that modulates the number of attended tokens based on local uncertainty.
  - Introduces dual gating at the value and output stages of the attention mechanism.
  - Establishes theoretical foundations including complexity analysis, expressiveness results, and convergence guarantees.
  - Achieves a 12-16x speedup at 128K context compared to standard attention, matching the efficiency of sparse-only baselines.
  - Improves perplexity from 6.03 to 5.70 in experiments with 1.7B parameter models trained on 400B tokens.
  - Nearly doubles RULER scores at 128K context.
  - Reduces attention to the first token (a proxy for attention sinks) from 47% to under 4%.
  - Reduces loss spikes by 98%, significantly improving training stability.
- **Long Context Tuning for Video Generation** (2025) — Yuwei Guo, Ceyuan Yang, Ziyan Yang et al.. [arXiv:2503.10589](https://arxiv.org/abs/2503.10589)
  - Introduces Long Context Tuning (LCT), a training paradigm that expands the context window of pre-trained single-shot video diffusion models to learn scene-level consistency directly from data.
  - Expands full attention mechanisms from individual shots to encompass all shots within a scene.
  - Incorporates interleaved 3D position embedding and an asynchronous noise strategy to enable joint and auto-regressive shot generation without additional parameters.
  - Enables fine-tuning of models with bidirectional attention using context-causal attention to facilitate auto-regressive generation with efficient KV-cache.
  - Demonstrates that single-shot models after LCT can produce coherent multi-shot scenes.
  - Identifies emerging capabilities in LCT-enhanced models, including compositional generation and interactive shot extension.
- **Long-Context Linear System Identification** (2024) — Oğuz Kaan Yüksel, Mathieu Even, Nicolas Flammarion. [arXiv:2410.05690](https://arxiv.org/abs/2410.05690)
  - Establishes a sample complexity bound for long-context linear system identification that matches the i.i.d. parametric rate up to logarithmic factors for a broad class of systems.
  - Extends previous theoretical results by addressing systems with dependencies beyond the first order.
  - Identifies a "learning-without-mixing" phenomenon, demonstrating that learning long-context linear autoregressive models is not hindered by slow mixing properties associated with extended context windows.
  - Demonstrates that rank-regularized estimators improve the dependence of sample complexity rates on dimensionality in the context of shared low-rank representations.
  - Shows that shorter context lengths offer statistical advantages in strictly stable systems when the context length is misspecified.


## Pruning, sparsity & distillation

The reviewed literature highlights a shift from static model compression toward dynamic, data-centric, and hybrid optimization strategies for efficient and capable LLM inference. Major themes include the critical role of data quality and composition—specifically leveraging negative examples, evolving instruction complexity, and balancing instruction types—over simple scale. Concurrently, technical advances in quantization and distillation demonstrate that preserving salient weights and integrating metric learning principles can maintain reasoning capacity and output fidelity where naive binarization or standard distillation fails.

- Integrating negative (failed) interaction trajectories into agent fine-tuning significantly improves performance in reasoning and QA tasks by providing a better trade-off between valuable information and error correction compared to using only successful trajectories.
- The Evol-Instruct method automates the generation of high-complexity instructions via LLM self-improvement, enabling models like WizardLM to surpass human-created instruction sets and achieve near-ChatGPT performance on complex tasks.
- PB-LLM introduces a hybrid quantization approach that preserves linguistic reasoning by storing only salient weights in higher-bit formats while binarizing the remainder, utilizing Hessian-guided reconstruction and quantization-aware training to minimize error.
- Triplet Loss distillation enhances student model mimicry by applying metric learning to reduce the distance between similar teacher-student outputs and increase the distance between dissimilar ones, improving output distinction.
- Instruction mixing analysis reveals that while specific instruction types (NLP, coding, chat) benefit certain applications, they can negatively impact others, necessitating balanced mixture strategies for optimal general performance.

### Papers

- **Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents** (2024) — Renxi Wang, Haonan Li, Xudong Han et al.. [arXiv:2402.11651](https://arxiv.org/abs/2402.11651)
  - Propose a fine-tuning strategy that integrates negative (failed) interaction trajectories into the training of Large Language Models as agents, addressing the scarcity and cost issues associated with using only successful trajectories.
  - Demonstrate that adding a specific prefix or suffix to indicate whether a trajectory is successful or unsuccessful during training significantly improves model performance.
  - Show empirical improvements in model performance across mathematical reasoning, multi-hop question answering, and strategic question answering tasks.
  - Provide analysis indicating that the proposed method achieves a better trade-off between valuable information and errors in unsuccessful trajectories compared to discarding them.
  - Present the first demonstration of the value of negative trajectories and their application in agent-tuning scenarios.
- **Demystifying Instruction Mixing for Fine-tuning Large Language Models** (2023) — Renxi Wang, Haonan Li, Minghao Wu et al.. [arXiv:2312.10793](https://arxiv.org/abs/2312.10793)
  - Categorizes instructions into three primary types: NLP downstream tasks, coding, and general chat.
  - Explores the effects of instruction tuning on different combinations of datasets on LLM performance.
  - Identifies that certain instruction types are more advantageous for specific applications but can negatively impact other areas.
  - Provides insights into instruction mixtures to lay the foundations for future research.
- **WizardLM: Empowering large pre-trained language models to follow complex instructions** (2023) — Can Xu, Qingfeng Sun, Kai Zheng et al.. [arXiv:2304.12244](https://arxiv.org/abs/2304.12244)
  - Proposes Evol-Instruct, a method that uses large language models to rewrite initial instructions step-by-step into more complex ones, replacing manual human creation.
  - Develops WizardLM by fine-tuning LLaMA with a mixture of generated instruction data produced by Evol-Instruct.
  - Demonstrates through human evaluations that instructions generated by Evol-Instruct are superior to human-created ones on a complexity-balanced test bed and Vicuna's testset.
  - Shows that WizardLM outputs are preferred over OpenAI ChatGPT outputs in human evaluations focusing on high-complexity instructions.
  - Achieves more than 90% capacity of ChatGPT in GPT-4 automatic evaluations across 17 out of 29 skills.
- **PB-LLM: Partially Binarized Large Language Models** (2023) — Yuzhang Shang, Zhihang Yuan, Qiang Wu et al.. [arXiv:2310.00034](https://arxiv.org/abs/2310.00034)
  - Proposes Partially-Binarized LLM (PB-LLM), a novel approach that filters and allocates a small ratio of salient weights to higher-bit storage while binarizing the remaining weights to maintain linguistic reasoning capacity.
  - Identifies the ineffectiveness of naive binarization algorithms for LLMs and establishes the critical role of salient weights in achieving successful low-bit quantization.
  - Introduces a Post-Training Quantization (PTQ) method that reconstructs the binarized weight matrix guided by the Hessian matrix, based on GPTQ concepts, to recover reasoning capacity.
  - Develops a Quantization-Aware Training (QAT) strategy that freezes salient weights during training and derives optimal scaling factors to minimize quantization error.
  - Proposes a specific scaling mechanism for residual binarized weights derived from the optimal scaling strategy within the QAT framework.
- **Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models** (2023) — Neha Sengupta, Sunil Kumar Sahu, Bokang Jia et al.. [arXiv:2308.16149](https://arxiv.org/abs/2308.16149)
  - Introduction of Jais and Jais-chat, open generative large language models based on the GPT-3 decoder-only architecture with 13 billion parameters.
  - Pretraining on a mixture of Arabic and English texts, including source code in various programming languages.
  - Demonstration of superior knowledge and reasoning capabilities in Arabic compared to existing open Arabic and multilingual models by a sizable margin.
  - Competitive performance in English relative to English-centric open models of similar size, despite training on significantly less English data.
  - Release of two open versions: the foundation Jais model and the instruction-tuned Jais-chat variant.
- **Triplet Loss for Knowledge Distillation** (2020) — Hideki Oki, Motoshi Abe, Junichi Miyao et al.. [arXiv:2004.08116](https://arxiv.org/abs/2004.08116)
  - Proposes a knowledge distillation method that integrates metric learning concepts to enhance the similarity between teacher and student model outputs.
  - Utilizes pairs or triplets of training samples to guide the student model in mimicking the teacher model.
  - Applies the metric learning mechanism of reducing the distance between similar outputs and increasing the distance between dissimilar outputs to minimize differences between teacher and student predictions.
  - Leverages the ability of metric learning to clarify differences between outputs for different objects, enabling the student model to better distinguish between them.


## Method & caveats

- **Source:** arXiv API, phrase-matched per subtopic (relevance-sorted). Not citation-ranked; abstracts only; automated selection, not curation.
- **Reasoning:** fully local — the GX10 vLLM model, thinking disabled, no cloud, no human-in-the-loop. A literature map, not a peer-reviewed survey.
- **Provenance:** `papers.json` and `distilled_clean.json` sit beside this file.
