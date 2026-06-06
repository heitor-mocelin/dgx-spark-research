---
id: 011
title: "vLLM PR #40082: Integrate FlashInfer b12x MoE and FP4 GEMM kernels for SM120/121"
url: "https://github.com/vllm-project/vllm/pull/40082"
publisher: "GitHub (vLLM)"
retrieved: "2026-06-05"
fetched_by: "openclaw-lxc/gh-cli"
license_note: "reference only — cited by URL; short excerpts attributed"
topics: [backends, moe, nvfp4, sm121, flashinfer]
---

author:	gemini-code-assist
association:	contributor
edited:	false
status:	commented
--
## Code Review

This pull request introduces support for the FlashInfer CuteDSL fused MoE kernel on SM12x architectures, adding the FlashInferCuteDSLSM12xExperts class and necessary backend configurations. Feedback identifies several issues: in-place modification of a2_gscale causes permanent side effects on model parameters, and performing MMA layout conversions during every forward pass adds unnecessary overhead. Additionally, suggestions were made to ensure correct data types for activation scale placeholders and routing weights when interfacing with the FlashInfer kernel.
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
cc: @mgoin  @pavanimajety  -- can one of you please review and mark this PR as ready?
cc: @johnnynunez 
--
author:	pavanimajety
association:	member
edited:	false
status:	none
--
Thanks for the PR, @meena-at-work! Any reason we need a new backend?  Any chance we can reuse https://github.com/vllm-project/vllm/pull/40082/changes#diff-a2596f35b4e543a919c6600cb9d03b542453cc80f01b2aaf02d735fb8186271cR96-R101? It would be easier to manage on framework end. Perhaps in future, we should also consider this architecture switch at Flashinfer level when possible. 
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
> Thanks for the PR, @meena-at-work! Any reason we need a new backend? Any chance we can reuse https://github.com/vllm-project/vllm/pull/40082/changes#diff-a2596f35b4e543a919c6600cb9d03b542453cc80f01b2aaf02d735fb8186271cR96-R101? It would be easier to manage on framework end. 

These use CuTe DSL kernels generated agentically (https://github.com/lukealonso/b12x), which have gained significant community traction. The FlashInfer team considered them distinct enough  from the existing CuTe DSL path to warrant a separate backend. I'll rename the user-visible name to make that clearer (e.g. flashinfer_cutedsl_b12x or similar).  

> Perhaps in future, we should also consider this architecture switch at Flashinfer level when possible.
Definitely.

I agree with the long-term direction of pushing the SM12x dispatch into FlashInfer itself — at that point this vLLM-side class could be folded back into FLASHINFER_CUTEDSL.

--
author:	pavanimajety
association:	member
edited:	false
status:	none
--
@meena-at-work Could you please add accuracy and perf benchmarks as well for how they compare against any available backends? Thanks!!  
--
author:	eugr
association:	none
edited:	false
status:	none
--
@meena-at-work - how do you run it on DGX Spark? Does it work with Nemotron models?
I've just tried to build fresh vLLM with this PR applied (and I always use Flashinfer from main, so dependencies are satisfied), but I get:

```
ValueError: NvFp4 MoE backend 'FLASHINFER_B12X' does not support the deployment configuration since kernel does not support no act_and_mul MLP layer.
```

Launching like this:

```bash
CUTE_DSL_ARCH=sm_121a vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.7 \
  --max-model-len 262144 \
  --max-num-seqs 10 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8888 \
  --enable-auto-tool-choice \
  --load-format instanttensor \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --mamba_ssm_cache_dtype float32 \
  --moe-backend flashinfer_b12x
```
--
author:	meena-at-work
association:	contributor
edited:	true
status:	none
--
Hi @eugr  -- You'd need to use the top of tree flashinfer to resolve that issue with B12x.

Nemotron 3 super work is still WIP, and is here:  https://github.com/askliar/vllm/commits/askliar/b12x-with-tinygemm/
--
author:	eugr
association:	none
edited:	false
status:	none
--
> You'd need to use the top of tree flashinfer to resolve that issue with B12x.

Yes, that's what I'm using. My latest build includes flashinfer up to this commit: https://github.com/flashinfer-ai/flashinfer/commit/fb3bb44c63d594f1452eea1af18b81d7b7693d48
--
author:	eugr
association:	none
edited:	false
status:	none
--
@meena-at-work: testing with a different model (RedHatAI/Qwen3-30B-A3B-NVFP4) gets me beyond that error, but then it fails with:

```
(EngineCore pid=87)   File "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/warp/mma.py", line 139, in __post_init__
(EngineCore pid=87)     raise OpError(
(EngineCore pid=87) cutlass.cute.nvgpu.common.OpError: OpError: expects arch to be one of ['sm_120a'], but got Arch.sm_121a
```

If I patch cutlass files, like @askliar suggested in his PR, then it just fails with exceptions.

Patch:

```bash
#!/bin/bash

set -e

CUTEDSL_ROOT="/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/"

# warp/mma.py: Add sm_121a alongside sm_120a
sed -i "s/if not arch == Arch.sm_120a:/if arch not in (Arch.sm_120a, Arch.sm_121a):/" $CUTEDSL_ROOT/warp/mma.py

# tcgen05/mma.py: Add sm_120a and sm_121a to arch list
sed -i "/Arch.sm_103a,/a\\        Arch.sm_120a,\n        Arch.sm_121a," $CUTEDSL_ROOT/tcgen05/mma.py

# tcgen05/copy.py: Add sm_120f family
sed -i "s/arch.is_family_of(Arch.sm_110f)/arch.is_family_of(Arch.sm_110f) or arch.is_family_of(Arch.sm_120f)/" $CUTEDSL_ROOT/tcgen05/copy.py
```

Result:

```
(EngineCore pid=103)       %1403 = "llvm.sub"(%1394, %1402) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32                                                                                                                                                  11:28:18 [36/1878]
(EngineCore pid=103)       "llvm.br"(%1396, %1403, %1382, %1354, %1355, %1381)[^bb141] : (i32, i32, i1, i32, i32, i32) -> ()
(EngineCore pid=103)     ^bb160:  // pred: ^bb141
(EngineCore pid=103)       %1404 = "llvm.add"(%1347, %39) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
(EngineCore pid=103)       %1405 = "llvm.icmp"(%1404, %29) <{predicate = 0 : i64}> : (i32, i32) -> i1
(EngineCore pid=103)       %1406 = "llvm.select"(%1405, %28, %1404) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i32, i32) -> i32
(EngineCore pid=103)       "llvm.cond_br"(%1405)[^bb161, ^bb162] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
(EngineCore pid=103)     ^bb161:  // pred: ^bb160
(EngineCore pid=103)       %1407 = "llvm.xor"(%1348, %39) : (i32, i32) -> i32
(EngineCore pid=103)       "llvm.br"(%1407)[^bb163] : (i32) -> ()
(EngineCore pid=103)     ^bb162:  // pred: ^bb160
(EngineCore pid=103)       "llvm.br"(%1348)[^bb163] : (i32) -> ()
(EngineCore pid=103)     ^bb163(%1408: i32):  // 2 preds: ^bb161, ^bb162
(EngineCore pid=103)       "llvm.br"()[^bb164] : () -> ()
(EngineCore pid=103)     ^bb164:  // pred: ^bb163
(EngineCore pid=103)       %1409 = "llvm.add"(%1406, %39) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
(EngineCore pid=103)       %1410 = "llvm.icmp"(%1409, %29) <{predicate = 0 : i64}> : (i32, i32) -> i1
(EngineCore pid=103)       %1411 = "llvm.select"(%1410, %28, %1409) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i32, i32) -> i32
(EngineCore pid=103)       "llvm.cond_br"(%1410)[^bb165, ^bb166] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
(EngineCore pid=103)     ^bb165:  // pred: ^bb164
(EngineCore pid=103)       %1412 = "llvm.xor"(%1408, %39) : (i32, i32) -> i32
(EngineCore pid=103)       "llvm.br"(%1412)[^bb167] : (i32) -> ()
(EngineCore pid=103)     ^bb166:  // pred: ^bb164
(EngineCore pid=103)       "llvm.br"(%1408)[^bb167] : (i32) -> ()
(EngineCore pid=103)     ^bb167(%1413: i32):  // 2 preds: ^bb165, ^bb166
(EngineCore pid=103)       "llvm.br"()[^bb168] : () -> ()
(EngineCore pid=103)     ^bb168:  // pred: ^bb167
(EngineCore pid=103)       %1414 = "llvm.getelementptr"(%78, %1411) <{elem_type = i64, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
(EngineCore pid=103)       "llvm.inline_asm"(%1414, %1413, %27) <{asm_dialect = 0 : i64, asm_string = "{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, $2; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}", con
straints = "r,r,n", has_side_effects}> : (!llvm.ptr<3>, i32, i32) -> ()
(EngineCore pid=103)       %1415 = "nvvm.elect.sync"() : () -> i1
(EngineCore pid=103)       "llvm.cond_br"(%1415)[^bb169, ^bb170] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
(EngineCore pid=103)     ^bb169:  // pred: ^bb168
(EngineCore pid=103)       %1416 = "llvm.getelementptr"(%16, %1411) <{elem_type = i64, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
(EngineCore pid=103)       "nvvm.mbarrier.txn"(%1416, %31) <{kind = #nvvm.mbar_txn_kind<arrive_expect_tx>, scope = #nvvm.mbar_scope<cta>, space = #nvvm.mbar_space<cta>}> : (!llvm.ptr<3>, i32) -> ()
(EngineCore pid=103)       "llvm.br"()[^bb170] : () -> ()
(EngineCore pid=103)     ^bb170:  // 2 preds: ^bb168, ^bb169
(EngineCore pid=103)       "llvm.br"()[^bb138] : () -> ()
(EngineCore pid=103)     ^bb171:  // pred: ^bb138
(EngineCore pid=103)       "llvm.return"() : () -> ()
(EngineCore pid=103)     }) {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 288, 1, 1>} : () -> ()
(EngineCore pid=103)   }) {compute_targets = [#cuda.compute_target<sass, conditional, [sm_121]>]} : () -> ()
```
--
author:	meena-at-work
association:	contributor
edited:	true
status:	none
--
@eugr  -- You need specific version of cute-dsl libraries: 

nvidia-cutlass-dsl==4.4.2 nvidia-cutlass-dsl-libs-base==4.4.2 nvidia-cutlass-dsl-libs-cu13==4.4.2
--
author:	eugr
association:	none
edited:	true
status:	none
--
> @eugr -- You need specific version of cute-dsl libraries:
> 
> nvidia-cutlass-dsl==4.4.2 nvidia-cutlass-dsl-libs-base==4.4.2 nvidia-cutlass-dsl-libs-cu13==4.4.2

Yep, that's the versions I have:

```
root@spark3:/workspace/vllm# uv pip freeze | grep cutlass
Using Python 3.12.3 environment at: /usr
nvidia-cutlass-dsl==4.4.2
nvidia-cutlass-dsl-libs-base==4.4.2
nvidia-cutlass-dsl-libs-cu13==4.4.2
```

BTW, I've noticed that Flashinfer also bundles CUTLASS - I wonder if I need to patch that one too?

```
root@spark3:/workspace/vllm# find /usr/local/lib/python3.12/dist-packages/ -name tcgen05 -print
/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05
/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05
```

EDIT: patched those as well, no difference.
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
Hi @eugr  -- For the Qwen workload, I only needed to patch the warp/mma.py files (and not tcgen05). Can you try with that?

--
author:	eugr
association:	none
edited:	false
status:	none
--
@meena-at-work - same thing :(

Launching vLLM like this: 

```bash
CUTE_DSL_ARCH=sm_121a vllm serve RedHatAI/Qwen3-30B-A3B-NVFP4 \
  --load-format instanttensor \
  --gpu-memory-utilization 0.7 \
  --host 0.0.0.0 --port 8888 \
  --attention-backend flashinfer
```

A bit more relevant part of the exception:

```
EngineCore pid=104)   ptxas application ptx input, line 1008; error   : Unexpected instruction types specified for '_mma'
(EngineCore pid=104)   ptxas application ptx input, line 1009; error   : Unexpected instruction types specified for '_mma'
(EngineCore pid=104)   ptxas application ptx input, line 1010; error   : Unexpected instruction types specified for '_mma'
(EngineCore pid=104)   ptxas application ptx input, line 1011; error   : Unexpected instruction types specified for '_mma'
(EngineCore pid=104)   ptxas application ptx input, line 1012; error   : Unexpected instruction types specified for '_mma'
(EngineCore pid=104)   ptxas fatal   : Ptx assembly aborted due to errors
(EngineCore pid=104)
(EngineCore pid=104) error: unknown: An error happened while serializing the module.
(EngineCore pid=104)  note: unknown: see current operation:
(EngineCore pid=104)   "gpu.module"() <{sym_name = "kernels", targets = [#nvvm.target<chip = "sm_121a", flags = {"ptx-cmd-options" = []}>]}> ({
(EngineCore pid=104)     "llvm.mlir.global"() <{addr_space = 3 : i32, alignment = 1024 : i64, dso_local, global_type = !llvm.array<0 x i8>, linkage = #llvm.linkage<external>, sym_name = "__dynamic_shmem__0", visibility_ = 0 : i64}> ({
(EngineCore pid=104)     }) : () -> ()
(EngineCore pid=104)     "llvm.func"() <{CConv = #llvm.cconv<ccc>, arg_attrs = [{llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}, {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}
, {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}, {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}, {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16
x i64>)>)>, nvvm.grid_constant}, {}, {}, {}, {}, {}, {}, {}, {}, {}], function_type = !llvm.func<void (ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i3
2, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, struct<(ptr, ptr)>, struct<(ptr, ptr)>, i32, i32, struct<(i32, i32, i8, i8)>, struct<(i32, i32, i8, i8)>, struct<(i32, i
32, i8, i8)>, struct<(ptr<1>, struct<(struct<()>, struct<()>)>)>)>, linkage = #llvm.linkage<external>, sym_name = "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120DenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc11638401_TVLayoutDst11638401_Valuetypef4E2M
1FN_tensor00odiv21_0", visibility_ = 0 : i64}> ({
(EngineCore pid=104)     ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg2: !llvm.ptr, %arg3: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg4: !llvm.ptr, %arg5: !llvm.struct<(struct<()>, struct<(struc
t<(i32, i32)>, struct<()>)>)>, %arg6: !llvm.ptr, %arg7: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg8: !llvm.ptr, %arg9: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg10: !llvm.struct<(ptr, ptr)>, %arg11: !llvm.struct<(ptr, pt
r)>, %arg12: i32, %arg13: i32, %arg14: !llvm.struct<(i32, i32, i8, i8)>, %arg15: !llvm.struct<(i32, i32, i8, i8)>, %arg16: !llvm.struct<(i32, i32, i8, i8)>, %arg17: !llvm.struct<(ptr<1>, struct<(struct<()>, struct<()>)>)>):
(EngineCore pid=104)       %0 = "llvm.mlir.constant"() <{value = 2 : i64}> : () -> i64
(EngineCore pid=104)       %1 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
(EngineCore pid=104)       %2 = "llvm.mlir.constant"() <{value = 0 : i64}> : () -> i64
(EngineCore pid=104)       %3 = "llvm.mlir.undef"() : () -> vector<4xi32>
(EngineCore pid=104)       %4 = "llvm.mlir.undef"() : () -> vector<64xf32>
(EngineCore pid=104)       %5 = "llvm.mlir.zero"() : () -> !llvm.struct<(ptr, ptr<3>, i16, i64, struct<()>)>
(EngineCore pid=104)       %6 = "llvm.mlir.zero"() : () -> !llvm.struct<(ptr, i64, struct<()>)>
(EngineCore pid=104)       %7 = "llvm.mlir.constant"() <{value = 896 : i64}> : () -> i64
(EngineCore pid=104)       %8 = "llvm.mlir.constant"() <{value = 0 : i16}> : () -> i16
(EngineCore pid=104)       %9 = "llvm.mlir.undef"() : () -> !llvm.struct<(ptr<3>, struct<(struct<()>, struct<()>)>)>
```
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
@eugr  -- is it possible for you to try manually uninstalling and reinstalliing the nvidia-cutlass-dsl python modules? I hit the crash on the vLLM nightly 13.0 container, and:

1. manually installing: nvidia-cutlass-dsl==4.4.2 nvidia-cutlass-dsl-libs-base==4.4.2 nvidia-cutlass-dsl-libs-cu13==4.4.2
2. patching warp/mma.py as mentioned above

resolved the crash.
--
author:	eugr
association:	none
edited:	false
status:	none
--
@meena-at-work - tried it yesterday, it didn't work. I wonder if it's related to me running CUDA 13.2 base image...
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
I've successfully run it on CUDA 13.2, so that's probably not it.
--
author:	eugr
association:	none
edited:	true
status:	none
--
I wonder if it is related to me building both flashinfer and vllm with 12.1a as the only architecture. Could be some compille-time guards that don't match runtime guards. That happened before.
--
author:	johnnynunez
association:	contributor
edited:	false
status:	none
--
> I wonder if it is related to me building both flashinfer and vllm with 12.1a as the only architecture. Could be some compille-time guards that don't match runtime guards. That happened before.

are you using flashinfer 0.6.8 or from main? Because 0.6.8.post1 doesn't have dgx spark support for b12x while future 0.6.9 yes
--
author:	eugr
association:	none
edited:	false
status:	none
--
Always from main.  
--
author:	meena-at-work
association:	contributor
edited:	true
status:	none
--
## Performance results on DGX Spark (SM121/GB10) — 2026-04-24

**Setup:** `vllm/vllm-openai:cu130-nightly-aarch64`, FlashInfer 0.6.9rc1, `nvidia/Qwen3-30B-A3B-NVFP4`, `vllm bench throughput`, batch_size=1 (num_prompts=1), input_len=1024, output_len=128, TP=1, MTP=0.

**Throughput (batch_size=1, decode-bound):**

| GEMM backend | MoE backend | Output tok/s | vs INT4 baseline |
|---|---|---|---|
| int4-marlin | marlin | 70.01 | — (reference) |
| **flashinfer-b12x** | **flashinfer_b12x** | **69.60** | **-0.6%** |
| marlin | throughput | 68.42 | -2.3% |
| flashinfer-cutlass | flashinfer_b12x | 67.65 | -3.4% |
| flashinfer-cudnn | throughput | 66.31 | -5.3% |
| flashinfer-cutlass | throughput | 65.65 | -6.2% |

b12x (GEMM + MoE) reaches **69.60 tok/s** — within 0.6% of INT4+Marlin and +6% over the flashinfer-cutlass baseline (65.65).

**Accuracy (MMLU, 50 samples/subject, 57 subjects):**

| Backend | MMLU | Humanities | Social Sciences | STEM | Other |
|---|---|---|---|---|---|
| flashinfer-cutlass (control) | 0.7789 | 0.7631 | 0.8433 | 0.7442 | 0.7862 |
| b12x | 0.7786 | 0.7523 | 0.8450 | 0.7505 | 0.7846 |

All differences within per-category standard error (~±0.014). No accuracy regression.

@pavanimajety  -- can you please review this now?
--
author:	mergify
association:	contributor
edited:	false
status:	none
--
Hi @meena-at-work, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit>=4.5.1
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
--
author:	mergify
association:	contributor
edited:	false
status:	none
--
Hi @meena-at-work, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit>=4.5.1
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
--
author:	tonyliu312
association:	contributor
edited:	false
status:	none
--
From a sm_121 (DGX Spark) deployment angle — this PR is the path forward for NVFP4 on Blackwell-edge / RTX-50-series workstations.

A few notes that may help unblock review:

**1. The int4-marlin reference number @meena-at-work measured (70.01 tok/s) silently depends on #40923.** On a stock vLLM build without #40923, the Marlin path for sm_12x emits no native cubin — `MARLIN_ARCHS = "8.0+PTX"` JIT-promotes to `sm_120/sm_121` PTX-as-cubin, which produces silently-wrong outputs on Marlin-MoE (we verified this on dual DGX Spark TP=2 with V4-Flash: gibberish without #40923, coherent with). So the "70 vs 69.6" headline comparison materially understates b12x's advantage if the Marlin baseline being compared against was a broken-PTX-JIT build. Worth verifying whether @meena-at-work's image had `MARLIN_ARCHS` patched to include `12.0;12.1` — if not, the b12x number is the only trustworthy reference. (#40923 is a 3-line CMakeLists patch and rebases clean on current main; cross-confirmed by @idonati on 8× DGX Spark TP=8 at #40899 and #40923 reproduction).

**2. Independent dual-Spark TP=2 validation offer.** We're running a 2× GB10 TP=2 cluster on RoCE multi-rail, currently sustaining V4-Flash production. Happy to apply this branch + run `vllm bench throughput` on `nvidia/Qwen3-30B-A3B-NVFP4` to give a TP=2 confirmation point alongside @meena-at-work's TP=1 numbers. Drop a note if useful — the bench is fast (under 10 min once the model is staged) and TP=2 exercises the inter-actor channel paths that single-GPU TP=1 doesn't.

**3. On the four open gemini-code-assist comments (Apr 17).** Three of them (process_weights timing, x_sf_placeholder dtype, token_final_scales float32 cast) look like they may already be partially addressed in subsequent commits (the `process_weights_after_loading` block already pre-computes the MMA-layout views once, per the Apr-22+ diff). The `a2_gscale.fill_(1.0)` one is more nuanced — the in-place flag-setting reads as intentional per the inline comment ("force to 1.0 so the kernel uses its own per-block dynamic scale"), but the bot's concern about cross-backend pollution is real. A `replace_parameter`-style approach (analogous to `_upcast_e8m0_to_fp32` recently landed in #40860 / #40899) would be safer — happy to push a small follow-up if @meena-at-work prefers.

For sm_12x users, this PR is the unlock — without it default `--moe-backend` lands on flashinfer-cutlass (65.6 tok/s) instead of b12x (69.6 tok/s), which is a real 6% perf gap that compounds across the 2026 GB10 install base. Worth landing.

— @tonyliu312

--
author:	pavanimajety
association:	member
edited:	false
status:	commented
--
Thanks @meena-at-work  for the PR! A couple of things worth addressing. 
--
author:	mergify
association:	contributor
edited:	false
status:	none
--
Hi @meena-at-work, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit>=4.5.1
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
--
author:	mergify
association:	contributor
edited:	false
status:	none
--
Hi @meena-at-work, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit>=4.5.1
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
--
author:	AethoceSora
association:	none
edited:	false
status:	none
--
> > I wonder if it is related to me building both flashinfer and vllm with 12.1a as the only architecture. Could be some compille-time guards that don't match runtime guards. That happened before.
> 
> are you using flashinfer 0.6.8 or from main? Because 0.6.8.post1 doesn't have dgx spark support for b12x while future 0.6.9 yes

I submitted PR #40998. Upgrading FlashInfer may help integrate the FlashInfer B12x MoE and FP4 GEMM kernels for SM120/121.
@pavanimajety 


--
author:	AethoceSora
association:	none
edited:	false
status:	none
--
> warp/mma.py must be patched to accept sm_121a (default only allows sm_120a)

A corresponding fix has already been proposed in the CUTLASS project:
[fix: Use is_family_of() for SM12x arch guard in MmaSM120BlockScaledOp(#3082)](https://github.com/NVIDIA/cutlass/pull/3082)

We will need to wait for the maintainers to review and approve the merge:
https://github.com/NVIDIA/cutlass/pull/3082#issuecomment-4332742305

--
author:	AethoceSora
association:	none
edited:	false
status:	none
--
> @meena-at-work Could you please add accuracy and perf benchmarks as well for how they compare against any available backends? Thanks!!

Tested under a relatively heavy setup on 4× RTX 5090 (sm120)
16 concurrent requests, each with 16K input and 16K output tokens.

Switching from the CUTLASS path (FLASHINFER_CUTLASS + FlashInferCutlassNvFp4LinearKernel) 
to the B12x path (FLASHINFER_B12X + FlashInferB12xNvFp4LinearKernel) , the improvement is quite clear. 

Output token throughput: increases from ~1755 tok/s to ~1974 tok/s
Time to First Token (TTFT, P99): drops from ~14.2s to ~13.3s
Time per Output Token (TPOT, P99): improves from 8.99 ms to 7.98 ms
Inter-token Latency (ITL, P99): decreases from 10.96 ms to 10.32 ms

In this long-context NVFP4 MoE workload, B12x makes better use of the hardware and provides speedup over the CUTLASS backend.

<details>
<summary>FLASHINFER_CUTLASS + FlashInferCutlassNvFp4LinearKernel</summary>

```
============ Serving Benchmark Result ============
Successful requests:                     16        
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  149.32    
Total input tokens:                      262311    
Total generated tokens:                  262144    
Request throughput (req/s):              0.11      
Output token throughput (tok/s):         1755.60   
Peak output token throughput (tok/s):    2031.00   
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          3512.32   
---------------Time to First Token----------------
Mean TTFT (ms):                          7752.98   
Median TTFT (ms):                        7761.01   
P99 TTFT (ms):                           14205.68  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.62      
Median TPOT (ms):                        8.62      
P99 TPOT (ms):                           8.99      
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.63      
Median ITL (ms):                         8.23      
P99 ITL (ms):                            10.96     
==================================================
```

</details>

<details>
<summary>FLASHINFER_B12X + FlashInferB12xNvFp4LinearKernel</summary>

```
============ Serving Benchmark Result ============
Successful requests:                     16        
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  132.81    
Total input tokens:                      262311    
Total generated tokens:                  262144    
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         1973.83   
Peak output token throughput (tok/s):    2326.00   
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          3948.91   
---------------Time to First Token----------------
Mean TTFT (ms):                          7298.08   
Median TTFT (ms):                        7315.46   
P99 TTFT (ms):                           13284.81  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.64      
Median TPOT (ms):                        7.64      
P99 TPOT (ms):                           7.98      
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.64      
Median ITL (ms):                         7.27      
P99 ITL (ms):                            10.32     
==================================================
```

</details>


--
author:	pavanimajety
association:	member
edited:	false
status:	none
--
@robertgshaw2-redhat Could you take a look for another set of eyes? Thanks!
--
author:	pavanimajety
association:	member
edited:	false
status:	approved
--
LGTM, thanks for the PR
--
author:	johnnynunez
association:	contributor
edited:	false
status:	none
--
> > warp/mma.py must be patched to accept sm_121a (default only allows sm_120a)
> 
> A corresponding fix has already been proposed in the CUTLASS project: [fix: Use is_family_of() for SM12x arch guard in MmaSM120BlockScaledOp(#3082)](https://github.com/NVIDIA/cutlass/pull/3082)
> 
> We will need to wait for the maintainers to review and approve the merge: [NVIDIA/cutlass#3082 (comment)](https://github.com/NVIDIA/cutlass/pull/3082#issuecomment-4332742305)

+viz @depaulmillz 
--
author:	eugr
association:	none
edited:	false
status:	none
--
@pavanimajety - JFYI, if this PR gets merged as is, it will cause errors loading NVFP4 models on DGX Spark, because of CUTLASS guard that excludes sm121 from required mma ops as @johnnynunez mentioned above. 

So, this backend should: 1) either not be selected automatically on sm121a until upstream cutlass fixes it; 2) or include necessary CUTLASS patches in it. 
--
author:	meena-at-work
association:	contributor
edited:	true
status:	none
--

> JFYI, if this PR gets merged as is, it will cause errors loading NVFP4 models on DGX Spark, because of CUTLASS guard that excludes sm121 from required mma ops as @johnnynunez mentioned above.

Thanks for catching this @eugr . I've removed b12x from being auto-selected.

--
author:	pfn
association:	none
edited:	false
status:	none
--
> @pavanimajety - JFYI, if this PR gets merged as is, it will cause errors loading NVFP4 models on DGX Spark, because of CUTLASS guard that excludes sm121 from required mma ops as @johnnynunez mentioned above.
> 
> So, this backend should: 1) either not be selected automatically on sm121a until upstream cutlass fixes it; 2) or include necessary CUTLASS patches in it.

https://github.com/NVIDIA/cutlass/pull/3082#issuecomment-4337996130

This seems like a non issue very soon now?
--
author:	johnnynunez
association:	contributor
edited:	false
status:	none
--
> > @pavanimajety - JFYI, if this PR gets merged as is, it will cause errors loading NVFP4 models on DGX Spark, because of CUTLASS guard that excludes sm121 from required mma ops as @johnnynunez mentioned above.
> > So, this backend should: 1) either not be selected automatically on sm121a until upstream cutlass fixes it; 2) or include necessary CUTLASS patches in it.
> 
> [NVIDIA/cutlass#3082 (comment)](https://github.com/NVIDIA/cutlass/pull/3082#issuecomment-4337996130)
> 
> This seems like a non issue very soon now?

you can try with cutlass v4.5-dev, it is on pypi
--
author:	pavanimajety
association:	member
edited:	true
status:	none
--
@meena-at-work What is the minimum flashinfer version required for this? Please also fix your DCO. 
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
> @meena-at-work What is the minimum flashinfer version required for this? Please also fix your DCO.

@pavanimajety  -- minimum flashinfer required for this is 0.6.9.
--
author:	eugr
association:	none
edited:	false
status:	none
--
> This seems like a non issue very soon now?

Just too many moving parts. It has to be merged in CUTLASS, then flashinfer/vllm need to switch to that specific cutlass version.
--
author:	pavanimajety
association:	member
edited:	false
status:	changes requested
--
Requires https://github.com/vllm-project/vllm/pull/40998.
--
author:	AethoceSora
association:	none
edited:	false
status:	none
--
> > > @pavanimajety - JFYI, if this PR gets merged as is, it will cause errors loading NVFP4 models on DGX Spark, because of CUTLASS guard that excludes sm121 from required mma ops as @johnnynunez mentioned above.
> > > So, this backend should: 1) either not be selected automatically on sm121a until upstream cutlass fixes it; 2) or include necessary CUTLASS patches in it.
> > 
> > 
> > [NVIDIA/cutlass#3082 (comment)](https://github.com/NVIDIA/cutlass/pull/3082#issuecomment-4337996130)
> > This seems like a non issue very soon now?
> 
> you can try with cutlass v4.5-dev, it is on pypi

After verification, nvidia-cutlass-dsl==4.5.0.dev0 does not seem to resolve this issue.

It is possible that the fix will be included in a future stable release, such as nvidia-cutlass-dsl 4.5.0.
--
author:	AethoceSora
association:	none
edited:	false
status:	none
--
> @meena-at-work - same thing :(
> 
> Launching vLLM like this:
> 
> ```shell
> CUTE_DSL_ARCH=sm_121a vllm serve RedHatAI/Qwen3-30B-A3B-NVFP4 \
>   --load-format instanttensor \
>   --gpu-memory-utilization 0.7 \
>   --host 0.0.0.0 --port 8888 \
>   --attention-backend flashinfer
> ```
> 
> A bit more relevant part of the exception:
> 
> ```
> EngineCore pid=104)   ptxas application ptx input, line 1008; error   : Unexpected instruction types specified for '_mma'
> (EngineCore pid=104)   ptxas application ptx input, line 1009; error   : Unexpected instruction types specified for '_mma'
> (EngineCore pid=104)   ptxas application ptx input, line 1010; error   : Unexpected instruction types specified for '_mma'
> (EngineCore pid=104)   ptxas application ptx input, line 1011; error   : Unexpected instruction types specified for '_mma'
> (EngineCore pid=104)   ptxas application ptx input, line 1012; error   : Unexpected instruction types specified for '_mma'
> (EngineCore pid=104)   ptxas fatal   : Ptx assembly aborted due to errors
> (EngineCore pid=104)
> (EngineCore pid=104) error: unknown: An error happened while serializing the module.
> (EngineCore pid=104)  note: unknown: see current operation:
> (EngineCore pid=104)   "gpu.module"() <{sym_name = "kernels", targets = [#nvvm.target<chip = "sm_121a", flags = {"ptx-cmd-options" = []}>]}> ({
> (EngineCore pid=104)     "llvm.mlir.global"() <{addr_space = 3 : i32, alignment = 1024 : i64, dso_local, global_type = !llvm.array<0 x i8>, linkage = #llvm.linkage<external>, sym_name = "__dynamic_shmem__0", visibility_ = 0 : i64}> ({
> (EngineCore pid=104)     }) : () -> ()
> (EngineCore pid=104)     "llvm.func"() <{CConv = #llvm.cconv<ccc>, arg_attrs = [{llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}, {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}
> , {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}, {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16 x i64>)>)>, nvvm.grid_constant}, {}, {llvm.align = 64 : i64, llvm.byval = !llvm.struct<(struct<(array<16
> x i64>)>)>, nvvm.grid_constant}, {}, {}, {}, {}, {}, {}, {}, {}, {}], function_type = !llvm.func<void (ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i3
> 2, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, ptr, struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, struct<(ptr, ptr)>, struct<(ptr, ptr)>, i32, i32, struct<(i32, i32, i8, i8)>, struct<(i32, i32, i8, i8)>, struct<(i32, i
> 32, i8, i8)>, struct<(ptr<1>, struct<(struct<()>, struct<()>)>)>)>, linkage = #llvm.linkage<external>, sym_name = "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120DenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc11638401_TVLayoutDst11638401_Valuetypef4E2M
> 1FN_tensor00odiv21_0", visibility_ = 0 : i64}> ({
> (EngineCore pid=104)     ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg2: !llvm.ptr, %arg3: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg4: !llvm.ptr, %arg5: !llvm.struct<(struct<()>, struct<(struc
> t<(i32, i32)>, struct<()>)>)>, %arg6: !llvm.ptr, %arg7: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg8: !llvm.ptr, %arg9: !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>, %arg10: !llvm.struct<(ptr, ptr)>, %arg11: !llvm.struct<(ptr, pt
> r)>, %arg12: i32, %arg13: i32, %arg14: !llvm.struct<(i32, i32, i8, i8)>, %arg15: !llvm.struct<(i32, i32, i8, i8)>, %arg16: !llvm.struct<(i32, i32, i8, i8)>, %arg17: !llvm.struct<(ptr<1>, struct<(struct<()>, struct<()>)>)>):
> (EngineCore pid=104)       %0 = "llvm.mlir.constant"() <{value = 2 : i64}> : () -> i64
> (EngineCore pid=104)       %1 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
> (EngineCore pid=104)       %2 = "llvm.mlir.constant"() <{value = 0 : i64}> : () -> i64
> (EngineCore pid=104)       %3 = "llvm.mlir.undef"() : () -> vector<4xi32>
> (EngineCore pid=104)       %4 = "llvm.mlir.undef"() : () -> vector<64xf32>
> (EngineCore pid=104)       %5 = "llvm.mlir.zero"() : () -> !llvm.struct<(ptr, ptr<3>, i16, i64, struct<()>)>
> (EngineCore pid=104)       %6 = "llvm.mlir.zero"() : () -> !llvm.struct<(ptr, i64, struct<()>)>
> (EngineCore pid=104)       %7 = "llvm.mlir.constant"() <{value = 896 : i64}> : () -> i64
> (EngineCore pid=104)       %8 = "llvm.mlir.constant"() <{value = 0 : i16}> : () -> i16
> (EngineCore pid=104)       %9 = "llvm.mlir.undef"() : () -> !llvm.struct<(ptr<3>, struct<(struct<()>, struct<()>)>)>
> ```



> @eugr -- is it possible for you to try manually uninstalling and reinstalliing the nvidia-cutlass-dsl python modules? I hit the crash on the vLLM nightly 13.0 container, and:
> 
> 1. manually installing: nvidia-cutlass-dsl==4.4.2 nvidia-cutlass-dsl-libs-base==4.4.2 nvidia-cutlass-dsl-libs-cu13==4.4.2
> 2. patching warp/mma.py as mentioned above
> 
> resolved the crash.



> I wonder if it is related to me building both flashinfer and vllm with 12.1a as the only architecture. Could be some compille-time guards that don't match runtime guards. That happened before.


After further investigation, the root cause is now clear.

The issue was caused by manually installing `nvidia-cutlass-dsl`, `nvidia-cutlass-dsl-libs-base`, and `nvidia-cutlass-dsl-libs-cu13` together.

`nvidia-cutlass-dsl-libs-base` and `nvidia-cutlass-dsl-libs-cu13` are CUDA-version-specific variants of the same underlying package. They both unpack files into the same paths, but with different contents. If both are installed, pip resolves the file conflicts based on install order, which can vary across upgrade paths and therefore leads to non-deterministic behavior.

The expected dependency behavior is:

- Installing `nvidia-cutlass-dsl` pulls in `nvidia-cutlass-dsl-libs-base`.
- Installing `nvidia-cutlass-dsl[cu13]` pulls in `nvidia-cutlass-dsl-libs-cu13` instead.

This means we should not install both library variants at the same time.

I also confirmed that `flashinfer-python` uses its own `cu13` extra to decide whether to depend on `nvidia-cutlass-dsl` with the `cu13` extra.

Therefore, the correct fix is to select the FlashInfer dependency based on the CUDA environment used to build vLLM:

- For CUDA 12 builds, depend on `flashinfer-python`.
- For CUDA 13 builds, depend on `flashinfer-python[cu13]`.

With this setup, the correct CUTLASS DSL library variant will be installed transitively, and we avoid conflicting installations of both CUDA variants.

I have also pushed a new commit to PR #40998  , which should address this issue as part of that PR.

--
author:	AethoceSora
association:	none
edited:	true
status:	none
--
> Please update to Flashinfer version that has flashinfer-jit-cache built with cutlass 4.5

I believe all prerequisites are now in place for merging both PR #40998 (`[CI/Build] chore(deps): bump flashinfer to v0.6.11`) and PR #40082 (`Integrate flashinfer b12x MoE and FP4 GEMM kernels for SM120/121`).

The relevant dependency and compatibility blockers have been resolved:

- `nvidia-cutlass-dsl` 4.5.0 stable has been released, which fixes the issue where `sm_121a` devices could not use `MmaSM120BlockScaledOp`.
- FlashInfer v0.6.11 has been released with `nvidia-cutlass-dsl` 4.5.0 as its dependency.
- Several issues that affected the use of the b12x backend on SM12x devices have been fixed and included in FlashInfer v0.6.11.

Relevant FlashInfer fixes:

- https://github.com/flashinfer-ai/flashinfer/pull/3246
- https://github.com/flashinfer-ai/flashinfer/pull/3191

Relevant Cutlass fixes:
- https://github.com/NVIDIA/cutlass/commit/cb37157db50d0528c4aea99feb37946ec278e3d9#diff-353c270c8ced4d47cd9fd493beca9ce6f72e3161fa00aa2894c1b07794bc881aR227

Given the above, the dependency stack should now be ready for enabling the FlashInfer b12x MoE and FP4 GEMM kernels for SM120/121 in vLLM.

Could you please review and merge this PR when you get a chance? Thanks!
cc @pavanimajety 

--
author:	idonati
association:	none
edited:	false
status:	none
--

## Tried the full prerequisite stack on 8× DGX Spark (GB10, sm_121a) — still fails

Following @AethoceSora's 2026-05-08 dependency-stack message, I rebuilt our 8-node DGX Spark cluster (GB10, sm_121a) with the exact stack named there and ran `festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8` (1023 GB raw, NVFP4 weights + MXFP8 attn-QKV, TP=8, EP=8).

**Stack** (all on every node):

- `vllm @ meena-at-work:integrate-flashinfer-b12x-moe` commit `7a16942`
- `flashinfer-python==0.6.11`, `flashinfer-cubin==0.6.11`, `flashinfer-jit-cache==0.6.11+cu130`
- `nvidia-cutlass-dsl==4.5.0` **stable** (not 4.5.0.dev0; AethoceSora's identified fix)
- `nvidia-cutlass-dsl-libs-base==4.5.0`, `nvidia-cutlass-dsl-libs-cu13==4.5.0`
- Phoenix-Shen's V-padding patch applied to `mimo_v2.py`
- NCCL 2.30.4 (clears the Ray Compiled-DAG wedge from #40969)
- `--moe-backend flashinfer_b12x`, `VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x`, `CUTE_DSL_ARCH=sm_121a`, `--attention-backend TRITON_ATTN`, `--enforce-eager`, `--enable-expert-parallel`

### Booted fine through the entire setup

- V-padding fallback engaged on all 8 ranks (`mimo_v2.py:313 FA2: using MiMo V-padding fallback for unequal K/V (pad v_head_dim=128 -> head_dim=192)`)
- `FLASHINFER_B12X` selected by the dispatcher on all 8 ranks (`nvfp4.py:225 Using 'FLASHINFER_B12X' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'EMULATION']`)
- EP=8 sharding correct (48/384 experts per rank)
- Weights loaded across all 8 ranks (slowest 684 s / fastest 462 s, 71.24 GiB per rank)
- `MoEPrepareAndFinalizeNoDPEPModular` selected (`nvfp4.py:504`)

### Then the FP4 MoE kernel JIT failed

First worker into the MoE kernel hits the same `_mma` ptxas error we and shadowlilac-oss have been seeing all along:

```
WorkerProc hit an exception.
File ".../nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/dsl.py", line 1385, in compile_and_jit
File ".../nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/compiler.py", line 145, in compile
  pm.run(module.operation)
cutlass._mlir._mlir_libs._site_initialize.<locals>.MLIRError: Failure while executing pass pipeline:
error: unknown: NVPTX compiler invocation failed, error log: ptxas application ptx input, line 969; error : Unexpected instruction types specified for '_mma'
  ptxas application ptx input, line 970; error : Unexpected instruction types specified for '_mma'
  ...
  ptxas application ptx input, line 1111; error : Unexpected instruction types specified for '_mma'
```

Same failure signature (different PTX line numbers) as @AethoceSora's 2026-05-04 repro on Qwen3-30B-A3B-NVFP4.

### What cutlass-dsl 4.5.0 stable actually fixed (and what it didn't)

I inspected the 4.5.0 stable wheel under `/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/`:

- ✅ `cute/nvgpu/warp/mma.py`: `MmaSM120BlockScaledOp.admissible_archs = ["sm_120a", "sm_121a"]` — **passes** for sm_121a, so the python-level guard from #2800 is gone.
- ❌ But the actual MLIR pipeline still emits `nvvm.mma.block_scale {aType=e2m1, bType=e2m1, blockScaleFormat=ue4m3, scaleVecSize=x4, shape=<m=16,n=8,k=64>, cType=f32}` with `nvvm.target chip="sm_121a"`, and ptxas in CUDA 13.0 (Aug 2025 build) rejects the lowered PTX.

### A separate small bug in 4.5.0

While there, `base_dsl/runtime/cuda.py:_get_gpu_arch_info` has no `(12, 1)` entry in the `gpu_arch_map`:

```python
gpu_arch_map = {
    ...
    (12, 0): ("Blackwell", "sm_120a", ["sm_120a"]),  # RTX PRO 6000 / RTX 50 Series
}
return gpu_arch_map.get(
    (major, minor), ("Unknown", f"sm_{major}{minor}", [f"sm_{major}{minor}"])
)
```

For a GB10 (compute capability 12.1), this returns `("Unknown", "sm_121", ["sm_121"])` — note `sm_121` *without* the `a` suffix. We're working around it by exporting `CUTE_DSL_ARCH=sm_121a`, so admissibility passes, but anyone relying on autodetection on GB10 silently gets a non-`a` arch string that won't load the right device template.

### Sanity check on ptxas

Took a minimal repro out of the toolkit:

```ptx
.version 8.8
.target sm_121a
.address_size 64
.visible .entry test() {
  ...
  mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
    {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3}, sa, {0,0}, sb, {0,0};
  ret;
}
```

```bash
$ ptxas test_mma.ptx -o test_mma.cubin --gpu-name=sm_121a; echo exit=$?
exit=0
$ ptxas test_mma.ptx -o test_mma.cubin --gpu-name=sm_120a; echo exit=$?
exit=0
```

So ptxas accepts that exact instruction with `.target sm_121a` standalone. **The cute-dsl-emitted PTX must be lowering to a slightly different encoding that's malformed specifically for the sm_121a target.** The instructions ptxas flags as `_mma` (with underscore) suggest the LLVM NVPTX backend is emitting an LLVM intrinsic that lowers to PTX `_mma` (intrinsic name) rather than the user-facing `mma.sync` mnemonic — and that path has a bug somewhere in the sm_121a sub-target.

### And `flashinfer_cutlass` (the precompiled cubin path that bypasses cute-dsl JIT) is also broken on sm_121 — different reason

Tested `--moe-backend flashinfer_cutlass` (precompiled cubins from `flashinfer-cubin==0.6.11`, the version AethoceSora says is built with cutlass 4.5). Boot completes, API comes up, `/v1/models` works. Smoke test:

```bash
$ curl http://192.168.0.10:5001/v1/chat/completions -d '{"model":"mimo-v2.5-pro","messages":[{"role":"user","content":"What is 2+2? One sentence answer only."}],"max_tokens":64,"temperature":0}'
"reasoning":"あるいあるいあるいあるいあるいあるい..." (64 tokens of degenerate repetition)
```

Identical degenerate `あるい` repetition we and shadowlilac-oss have been getting on every FP4-MoE path. **Reason confirmed by inspecting the wheel:**

```bash
$ find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin -name "*.cubin" | wc -l
12681
$ find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin -name "*.cubin" | grep -oE '[Ss]m[_]?[0-9]+[afp]?' | sort -u
Sm100a
Sm100f
Sm103a
sm100f
$ find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin -name "*.cubin" | grep -iE "sm[_]?12[01]" | wc -l
0
```

**flashinfer-cubin 0.6.11 ships 12,681 cubins for Sm100a / Sm100f / Sm103a — zero cubins for sm_120 or sm_121.** AethoceSora's "FlashInfer v0.6.11 has been released with `nvidia-cutlass-dsl` 4.5.0 as its dependency" is technically true for the build dep, but the wheel itself doesn't actually contain any sm_120/121 binaries. The kernel loader on sm_121a hardware presumably falls through to *some* sm_100 cubin via family compatibility, which compiles fine but produces wrong numerics for GB10's actual tile/cluster/TMA layout. Same `あるい` signature we've been chasing for two weeks.

So on consumer Blackwell, both PR #40082's path (`flashinfer_b12x` cute-dsl JIT — ptxas error) and the precompiled `flashinfer_cutlass` path (no sm_120/121 cubins — wrong numerics) are blocked, for two independent upstream reasons.

### Family-target workaround result (b12x path): also fails

Tested whether retargeting CUTE_DSL emission to `sm_120f` (Blackwell family target, forward-compatible with sm_121a hardware per ptxas help) bypasses the bug. Confirmed at the PTX level that hand-written `mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` compiles cleanly with `.target sm_120f` and the resulting cubin loads on sm_121a hardware.

Patched `cute/nvgpu/warp/mma.py`:
```python
admissible_archs = ["sm_120a", "sm_120f", "sm_121a"]
```
and set `CUTE_DSL_ARCH=sm_120f`. Rebuilt + redistributed to all 8 nodes. Boot proceeded identically:
- V-padding fallback engaged on all 8 ranks ✓
- `FLASHINFER_B12X` selected on all 8 ranks ✓
- All weights loaded (471 s fastest / 694 s slowest, 71.24 GiB / rank) ✓
- Same ptxas failure on first MoE JIT, same error message:

```
"gpu.module"() <{sym_name = "kernels", targets = [#nvvm.target<chip = "sm_120f", flags = {"ptx-cmd-options" = []}>]}> ({
...
ptxas application ptx input, line 969; error : Unexpected instruction types specified for '_mma'
```

The MLIR target chip *did* switch from `sm_121a` to `sm_120f` (so our admissibility patch took effect), but ptxas rejects the lowered PTX regardless. **This makes the bug fundamentally in cute-dsl's MLIR→PTX lowering pipeline, not in target-arch admissibility or sm_121a-specific code paths.** The hand-written equivalent compiles; the cute-dsl-emitted version doesn't, on either `sm_120a`, `sm_120f`, or `sm_121a` targets when reached via this codepath.

That makes this a hard cute-dsl bug. Nothing the vLLM or recipe layer can fix.

### Asks

1. **For @AethoceSora**: when you wrote "the dependency stack should now be ready" on 5/8, did you actually verify end-to-end inference on a sm_121a (or sm_120a) box, or was that an inference from the cutlass-dsl 4.5.0 release notes + flashinfer 0.6.11 dep bump? Two separate upstream gaps both block consumer Blackwell here: cutlass-dsl's MLIR lowering on the b12x path, and flashinfer-cubin's missing sm_120/121 cubins on the precompiled path. Neither is resolved by 4.5.0 + 0.6.11 alone.
2. **For NVIDIA / cute-dsl maintainers**: the cute-dsl MLIR→PTX lowering of `nvvm.mma.block_scale {aType=e2m1, bType=e2m1, blockScaleFormat=ue4m3, scaleVecSize=x4, shape=<m=16,n=8,k=64>, cType=f32}` produces malformed PTX on `chip="sm_120f"` and `chip="sm_121a"` — ptxas reports `Unexpected instruction types specified for '_mma'` on ~50 instructions per kernel. Hand-written `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` compiles fine on all three targets, so the lowering pipeline is producing something the user-facing PTX assembler can't handle. Happy to share the full IR dump (~2000 lines) if helpful.
3. **For @yzh119 / flashinfer**: would you consider adding sm_120a / sm_121a (or family-compatible sm_120f) to the flashinfer-cubin build matrix? Right now FP4 GEMM and MoE cubins ship for Sm100a / Sm100f / Sm103a only — every consumer Blackwell card (RTX 50 series, RTX PRO 6000, DGX Spark) is falling through to a wrong-arch cubin and producing degenerate output. We'd be happy to test a candidate wheel.
4. **Smaller bug**: the `gpu_arch_map` entry for `(12, 1)` is a one-line fix in `base_dsl/runtime/cuda.py:_get_gpu_arch_info`. Worth slipping into the next 4.5.x bump.
5. **For PR landing**: this PR is well-structured and the vLLM-side dispatcher / backend code is correct. It just can't actually run on sm_121 until *both* upstream gaps close. Suggest landing with a guard that prints a clear "MoE backend FLASHINFER_B12X selected, but cutlass-dsl 4.5.0 doesn't yet emit valid PTX for sm_120/121 — see X for status" warning when device capability matches the gap.

Hardware/repro details:

- 8× NVIDIA GB10 (sm_121a), Spark cluster, NCCL 2.30.4, Ray, vLLM
- Same repro pattern as @AethoceSora's 5/4 message
- Image build (Dockerfile + patches) and full launcher script available on request

--
author:	AethoceSora
association:	none
edited:	false
status:	none
--
> ## Tried the full prerequisite stack on 8× DGX Spark (GB10, sm_121a) — still fails
> Following @AethoceSora's 2026-05-08 dependency-stack message, I rebuilt our 8-node DGX Spark cluster (GB10, sm_121a) with the exact stack named there and ran `festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8` (1023 GB raw, NVFP4 weights + MXFP8 attn-QKV, TP=8, EP=8).
> 
> **Stack** (all on every node):
> 
> * `vllm @ meena-at-work:integrate-flashinfer-b12x-moe` commit `7a16942`
> * `flashinfer-python==0.6.11`, `flashinfer-cubin==0.6.11`, `flashinfer-jit-cache==0.6.11+cu130`
> * `nvidia-cutlass-dsl==4.5.0` **stable** (not 4.5.0.dev0; AethoceSora's identified fix)
> * `nvidia-cutlass-dsl-libs-base==4.5.0`, `nvidia-cutlass-dsl-libs-cu13==4.5.0`
> * Phoenix-Shen's V-padding patch applied to `mimo_v2.py`
> * NCCL 2.30.4 (clears the Ray Compiled-DAG wedge from [[Bug]: DeepSeek-V4-Flash hangs after ~6 requests with cudagraph_mode=FULL_AND_PIECEWISE + chunked prefill on SM 12.x (GB10) #40969](https://github.com/vllm-project/vllm/issues/40969))
> * `--moe-backend flashinfer_b12x`, `VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x`, `CUTE_DSL_ARCH=sm_121a`, `--attention-backend TRITON_ATTN`, `--enforce-eager`, `--enable-expert-parallel`
> 
> ### Booted fine through the entire setup
> * V-padding fallback engaged on all 8 ranks (`mimo_v2.py:313 FA2: using MiMo V-padding fallback for unequal K/V (pad v_head_dim=128 -> head_dim=192)`)
> * `FLASHINFER_B12X` selected by the dispatcher on all 8 ranks (`nvfp4.py:225 Using 'FLASHINFER_B12X' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'EMULATION']`)
> * EP=8 sharding correct (48/384 experts per rank)
> * Weights loaded across all 8 ranks (slowest 684 s / fastest 462 s, 71.24 GiB per rank)
> * `MoEPrepareAndFinalizeNoDPEPModular` selected (`nvfp4.py:504`)
> 
> ### Then the FP4 MoE kernel JIT failed
> First worker into the MoE kernel hits the same `_mma` ptxas error we and shadowlilac-oss have been seeing all along:
> 
> ```
> WorkerProc hit an exception.
> File ".../nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/dsl.py", line 1385, in compile_and_jit
> File ".../nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/compiler.py", line 145, in compile
>   pm.run(module.operation)
> cutlass._mlir._mlir_libs._site_initialize.<locals>.MLIRError: Failure while executing pass pipeline:
> error: unknown: NVPTX compiler invocation failed, error log: ptxas application ptx input, line 969; error : Unexpected instruction types specified for '_mma'
>   ptxas application ptx input, line 970; error : Unexpected instruction types specified for '_mma'
>   ...
>   ptxas application ptx input, line 1111; error : Unexpected instruction types specified for '_mma'
> ```
> 
> Same failure signature (different PTX line numbers) as @AethoceSora's 2026-05-04 repro on Qwen3-30B-A3B-NVFP4.
> 
> ### What cutlass-dsl 4.5.0 stable actually fixed (and what it didn't)
> I inspected the 4.5.0 stable wheel under `/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/`:
> 
> * ✅ `cute/nvgpu/warp/mma.py`: `MmaSM120BlockScaledOp.admissible_archs = ["sm_120a", "sm_121a"]` — **passes** for sm_121a, so the python-level guard from [[bug] AssertionError with prompt_logprobs and logits_processors both set #2800](https://github.com/vllm-project/vllm/issues/2800) is gone.
> * ❌ But the actual MLIR pipeline still emits `nvvm.mma.block_scale {aType=e2m1, bType=e2m1, blockScaleFormat=ue4m3, scaleVecSize=x4, shape=<m=16,n=8,k=64>, cType=f32}` with `nvvm.target chip="sm_121a"`, and ptxas in CUDA 13.0 (Aug 2025 build) rejects the lowered PTX.
> 
> ### A separate small bug in 4.5.0
> While there, `base_dsl/runtime/cuda.py:_get_gpu_arch_info` has no `(12, 1)` entry in the `gpu_arch_map`:
> 
> ```python
> gpu_arch_map = {
>     ...
>     (12, 0): ("Blackwell", "sm_120a", ["sm_120a"]),  # RTX PRO 6000 / RTX 50 Series
> }
> return gpu_arch_map.get(
>     (major, minor), ("Unknown", f"sm_{major}{minor}", [f"sm_{major}{minor}"])
> )
> ```
> 
> For a GB10 (compute capability 12.1), this returns `("Unknown", "sm_121", ["sm_121"])` — note `sm_121` _without_ the `a` suffix. We're working around it by exporting `CUTE_DSL_ARCH=sm_121a`, so admissibility passes, but anyone relying on autodetection on GB10 silently gets a non-`a` arch string that won't load the right device template.
> 
> ### Sanity check on ptxas
> Took a minimal repro out of the toolkit:
> 
> ```
> .version 8.8
> .target sm_121a
> .address_size 64
> .visible .entry test() {
>   ...
>   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
>     {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3}, sa, {0,0}, sb, {0,0};
>   ret;
> }
> ```
> 
> ```shell
> $ ptxas test_mma.ptx -o test_mma.cubin --gpu-name=sm_121a; echo exit=$?
> exit=0
> $ ptxas test_mma.ptx -o test_mma.cubin --gpu-name=sm_120a; echo exit=$?
> exit=0
> ```
> 
> So ptxas accepts that exact instruction with `.target sm_121a` standalone. **The cute-dsl-emitted PTX must be lowering to a slightly different encoding that's malformed specifically for the sm_121a target.** The instructions ptxas flags as `_mma` (with underscore) suggest the LLVM NVPTX backend is emitting an LLVM intrinsic that lowers to PTX `_mma` (intrinsic name) rather than the user-facing `mma.sync` mnemonic — and that path has a bug somewhere in the sm_121a sub-target.
> 
> ### And `flashinfer_cutlass` (the precompiled cubin path that bypasses cute-dsl JIT) is also broken on sm_121 — different reason
> Tested `--moe-backend flashinfer_cutlass` (precompiled cubins from `flashinfer-cubin==0.6.11`, the version AethoceSora says is built with cutlass 4.5). Boot completes, API comes up, `/v1/models` works. Smoke test:
> 
> ```shell
> $ curl http://192.168.0.10:5001/v1/chat/completions -d '{"model":"mimo-v2.5-pro","messages":[{"role":"user","content":"What is 2+2? One sentence answer only."}],"max_tokens":64,"temperature":0}'
> "reasoning":"あるいあるいあるいあるいあるいあるい..." (64 tokens of degenerate repetition)
> ```
> 
> Identical degenerate `あるい` repetition we and shadowlilac-oss have been getting on every FP4-MoE path. **Reason confirmed by inspecting the wheel:**
> 
> ```shell
> $ find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin -name "*.cubin" | wc -l
> 12681
> $ find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin -name "*.cubin" | grep -oE '[Ss]m[_]?[0-9]+[afp]?' | sort -u
> Sm100a
> Sm100f
> Sm103a
> sm100f
> $ find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin -name "*.cubin" | grep -iE "sm[_]?12[01]" | wc -l
> 0
> ```
> 
> **flashinfer-cubin 0.6.11 ships 12,681 cubins for Sm100a / Sm100f / Sm103a — zero cubins for sm_120 or sm_121.** AethoceSora's "FlashInfer v0.6.11 has been released with `nvidia-cutlass-dsl` 4.5.0 as its dependency" is technically true for the build dep, but the wheel itself doesn't actually contain any sm_120/121 binaries. The kernel loader on sm_121a hardware presumably falls through to _some_ sm_100 cubin via family compatibility, which compiles fine but produces wrong numerics for GB10's actual tile/cluster/TMA layout. Same `あるい` signature we've been chasing for two weeks.
> 
> So on consumer Blackwell, both PR #40082's path (`flashinfer_b12x` cute-dsl JIT — ptxas error) and the precompiled `flashinfer_cutlass` path (no sm_120/121 cubins — wrong numerics) are blocked, for two independent upstream reasons.
> 
> ### Family-target workaround result (b12x path): also fails
> Tested whether retargeting CUTE_DSL emission to `sm_120f` (Blackwell family target, forward-compatible with sm_121a hardware per ptxas help) bypasses the bug. Confirmed at the PTX level that hand-written `mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` compiles cleanly with `.target sm_120f` and the resulting cubin loads on sm_121a hardware.
> 
> Patched `cute/nvgpu/warp/mma.py`:
> 
> ```python
> admissible_archs = ["sm_120a", "sm_120f", "sm_121a"]
> ```
> 
> and set `CUTE_DSL_ARCH=sm_120f`. Rebuilt + redistributed to all 8 nodes. Boot proceeded identically:
> 
> * V-padding fallback engaged on all 8 ranks ✓
> * `FLASHINFER_B12X` selected on all 8 ranks ✓
> * All weights loaded (471 s fastest / 694 s slowest, 71.24 GiB / rank) ✓
> * Same ptxas failure on first MoE JIT, same error message:
> 
> ```
> "gpu.module"() <{sym_name = "kernels", targets = [#nvvm.target<chip = "sm_120f", flags = {"ptx-cmd-options" = []}>]}> ({
> ...
> ptxas application ptx input, line 969; error : Unexpected instruction types specified for '_mma'
> ```
> 
> The MLIR target chip _did_ switch from `sm_121a` to `sm_120f` (so our admissibility patch took effect), but ptxas rejects the lowered PTX regardless. **This makes the bug fundamentally in cute-dsl's MLIR→PTX lowering pipeline, not in target-arch admissibility or sm_121a-specific code paths.** The hand-written equivalent compiles; the cute-dsl-emitted version doesn't, on either `sm_120a`, `sm_120f`, or `sm_121a` targets when reached via this codepath.
> 
> That makes this a hard cute-dsl bug. Nothing the vLLM or recipe layer can fix.
> 
> ### Asks
> 1. **For @AethoceSora**: when you wrote "the dependency stack should now be ready" on 5/8, did you actually verify end-to-end inference on a sm_121a (or sm_120a) box, or was that an inference from the cutlass-dsl 4.5.0 release notes + flashinfer 0.6.11 dep bump? Two separate upstream gaps both block consumer Blackwell here: cutlass-dsl's MLIR lowering on the b12x path, and flashinfer-cubin's missing sm_120/121 cubins on the precompiled path. Neither is resolved by 4.5.0 + 0.6.11 alone.
> 2. **For NVIDIA / cute-dsl maintainers**: the cute-dsl MLIR→PTX lowering of `nvvm.mma.block_scale {aType=e2m1, bType=e2m1, blockScaleFormat=ue4m3, scaleVecSize=x4, shape=<m=16,n=8,k=64>, cType=f32}` produces malformed PTX on `chip="sm_120f"` and `chip="sm_121a"` — ptxas reports `Unexpected instruction types specified for '_mma'` on ~50 instructions per kernel. Hand-written `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` compiles fine on all three targets, so the lowering pipeline is producing something the user-facing PTX assembler can't handle. Happy to share the full IR dump (~2000 lines) if helpful.
> 3. **For @yzh119 / flashinfer**: would you consider adding sm_120a / sm_121a (or family-compatible sm_120f) to the flashinfer-cubin build matrix? Right now FP4 GEMM and MoE cubins ship for Sm100a / Sm100f / Sm103a only — every consumer Blackwell card (RTX 50 series, RTX PRO 6000, DGX Spark) is falling through to a wrong-arch cubin and producing degenerate output. We'd be happy to test a candidate wheel.
> 4. **Smaller bug**: the `gpu_arch_map` entry for `(12, 1)` is a one-line fix in `base_dsl/runtime/cuda.py:_get_gpu_arch_info`. Worth slipping into the next 4.5.x bump.
> 5. **For PR landing**: this PR is well-structured and the vLLM-side dispatcher / backend code is correct. It just can't actually run on sm_121 until _both_ upstream gaps close. Suggest landing with a guard that prints a clear "MoE backend FLASHINFER_B12X selected, but cutlass-dsl 4.5.0 doesn't yet emit valid PTX for sm_120/121 — see X for status" warning when device capability matches the gap.
> 
> Hardware/repro details:
> 
> * 8× NVIDIA GB10 (sm_121a), Spark cluster, NCCL 2.30.4, Ray, vLLM
> * Same repro pattern as @AethoceSora's 5/4 message
> * Image build (Dockerfile + patches) and full launcher script available on request

https://github.com/vllm-project/vllm/pull/40082#issuecomment-4349406309
--
author:	idonati
association:	none
edited:	false
status:	none
--
Quick follow-up — big movement on this PR's blockers thanks to @depaulmillz over on [NVIDIA/cutlass#3227](https://github.com/NVIDIA/cutlass/issues/3227):

### 1. The `_mma` ptxas error: resolved (root cause was a cutlass-dsl wheel-collision)

The cute-dsl JIT failure was **not** an MLIR lowering bug — it was a packaging issue. `nvidia-cutlass-dsl-libs-base==4.5.0` and `nvidia-cutlass-dsl-libs-cu13==4.5.0` ship 168 overlapping `.py` files at the same paths inside the package (including `cute/nvgpu/warp/mma.py`) with **different content / different SHA256**:

- `libs-base` ships a variant using `CuTeDSL._get_dsl()` + string-literal `admissible_archs` → emits malformed PTX that ptxas rejects as `_mma`
- `libs-cu13` ships the correct variant using `BaseDSL._get_dsl()` + enum `admissible_archs` → emits valid PTX

A clean `pip install nvidia-cutlass-dsl[cu13]` resolves install order so `libs-cu13` wins. But `--force-reinstall` chains, direct-wheel-file installs, or any Dockerfile layer that previously installed `libs-base` alone can leave the buggy version on disk. Our v6 image had the bad version; v7 (built via `pip uninstall -y nvidia-cutlass-dsl* && rm -rf /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl* && pip install --upgrade "nvidia-cutlass-dsl[cu13]>=4.5.0"`) has the correct version.

```
v6 (failing) mma.py sha256:  e49c79fd43e28b00442b10c294277702ad3fc0c973c72b84cd00e3dcfab14f48
v7 (working) mma.py sha256:  c9171964a5a4a37171631428a7e3871cd5679755729fba740f0ae475e5c6f329
```

With the right mma.py on disk, the cute-dsl JIT compiles cleanly through `compile_and_jit`. **No more `_mma` ptxas error.** Full discussion + suggested wheel-packaging fix in NVIDIA/cutlass#3227.

So my earlier ask 1 — about whether @AethoceSora had verified end-to-end — turned out to be misdirected: the dep stack you listed *was* correct, the failure mode I was hitting was a separate file-layering trap underneath. Apologies for the noise.

### 2. New blocker (EP-slicing): `b_w13` / `b_down` aren't sliced to local experts before the `flashinfer_b12x_fused_moe` kernel call

With JIT working, the kernel call fails at warmup. Looks like the EP-aware caller isn't pre-slicing `b_w13` / `b_down` to local experts:

```
ValueError: Mismatched b_w13.shape[2] on argument #9 when calling: `__call__(
  a_input:                Tensor([8192, 6144], bfloat16),
  topk_ids:               Tensor([65536], int32),
  topk_weights:           Tensor([65536], float32),
  packed_a:               Tensor([65536, 3072, 48], float4_e2m1fnx2),
  ...
  b_w13:                  Tensor([4096, 3072, 384], float4_e2m1fnx2),
  b_down:                 Tensor([6144, 1024, 384], float4_e2m1fnx2),
  row_counts:             Tensor([48], int32),
  weight_expert_ids:      Tensor([48], int32),
  global_to_local_expert: Tensor([384], int32),
  ...
)`, expected to be 384
```

`packed_a.shape[-1] = 48`, `row_counts.shape[0] = 48`, `weight_expert_ids.shape[0] = 48` ← all sliced to local experts (384 global / 8 EP ranks = 48 local). But `b_w13.shape[-1] = 384` and `b_down.shape[-1] = 384` are still global. Kernel internal check fires.

Stack lands in:
- `multiproc_executor.py:957 worker_busy_loop`
- → `gpu_worker.py:392 determine_available_memory`
- → `gpu_model_runner.py:5948 profile_run`
- → cute-dsl b12x `__call__`

So this triggers before any actual inference. Triggered the same way on all 8 ranks simultaneously. Filed separately at [flashinfer-ai/flashinfer#3294 (comment)](https://github.com/flashinfer-ai/flashinfer/issues/3294#issuecomment-4426701270) asking @kahyunnam whether the kernel expects pre-sliced weights or handles slicing internally. Want it filed against vllm#40082 too, or fine to let it land on the FlashInfer side?

### 3. Other findings still standing

- `gpu_arch_map` missing `(12, 1)` entry: still a one-line miss in `nvidia-cutlass-dsl-libs-cu13` 4.5.0's `base_dsl/runtime/cuda.py`.
- `flashinfer-cubin 0.6.11` having no `Sm120`-tagged cubins: per @kahyunnam, the kernels in `flashinfer-jit-cache` are *labeled* `sm_120` by cuobjdump but are actually `sm_120f` family-target, expected to work on sm_121 hardware. So my earlier "no cubins for consumer Blackwell" claim was wrong — the kernels are there, the family-target naming was just confusing. The `あるい` garbage path is now a correctness question on the kernels themselves (separate FlashInfer thread).

### Net

PR #40082's vLLM-side dispatcher / backend code is correct. Once #3 (EP-slicing) closes, we should have working FP4 MoE on sm_121a end-to-end with this PR. Happy to test a fix branch on our cluster the moment one drops.

--
author:	idonati
association:	none
edited:	false
status:	none
--
@meena-at-work — quick FYI, the EP shape mismatch I posted [above](https://github.com/vllm-project/vllm/pull/40082#issuecomment-4427370553) looks like the same pattern that landed on the SGLang side as [sgl-project/sglang#24576](https://github.com/sgl-project/sglang/pull/24576) (still open, opened 5/7). Same numbers: `num_experts=384, ep_size=8, num_local_experts=48`. Same symptom: weights still sized at global E (384) while the routing/state tensors are at local E (48). The sgl fix is a one-line swap in `ModelOptNvFp4FusedMoEMethod.process_weights_after_loading`:

```diff
-                or existing_params.num_experts != layer.num_experts
+                or existing_params.num_experts != layer.num_local_experts
...
-                    num_experts=layer.num_experts,  # global num experts
+                    num_experts=layer.num_local_experts,
```

For the b12x path on our side, I traced it down to `launch_sm120_moe`'s `_get_weight_views()` at `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py:263`:

```python
# Permute [E, w1_rows, k//2] -> [w1_rows, k//2, E]
w13 = w1_fp4.permute(1, 2, 0)
```

so the `w1_fp4` (= `w1` in `FlashInferB12xExperts.apply`) is expected to have `shape[0] == E` for whatever expert-count the rest of the call uses. The kernel's other arrays (`packed_a`, `row_counts`, `weight_expert_ids`, `compact_topk_ids` — all `[state_E, ...]`) are sized by `num_local_experts=48`, but our `w1.shape[0] = num_experts=384`. The kernel's spec validator picks up `E` from `packed_a.shape[2] = 48` first, then fails when `b_w13.shape[2] = 384` doesn't match.

So the question is what `FlashInferB12xExperts.apply()` should receive as `w1`/`w2` on EP runs:

1. **Pre-sliced to `num_local_experts`** (vLLM's FusedMoE layer slices `w13_weight`/`w2_weight` to `[num_local_experts, ...]` before calling the experts module). Mirrors the sgl#24576 fix. Most consistent with `state_E` semantics in the kernel.
2. **Global `num_experts`** (current vLLM behavior; the b12x kernel would have to be the one that gathers/scatters via `global_to_local_expert`). Doesn't match the current `_get_weight_views()` permutation though.

If the answer is (1), the integration-side fix in vLLM looks roughly like:

```python
# vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
def apply(self, ..., w1, w2, ..., global_num_experts, ...):
    if w1.shape[0] != self.num_local_experts:
        w1 = w1[:self.num_local_experts]
        w2 = w2[:self.num_local_experts]
    # ... existing call ...
```

though that's just the kernel call — the matching slice would need to apply to `self.w1_scale`, `self.w2_scale`, `self.g1_alphas`, `self.g2_alphas`, and `self.a2_gscale` as well (they're all `[num_experts, ...]` today and feed `w1_weight_sf` / `w2_weight_sf` / `w1_alpha` / etc). I can attempt this patch on our cluster as an experiment if you want a data point, but happy to wait if you'd rather do it properly upstream — I'd rather not ship a workaround that papers over a different convention you're using.

Will report back once we hear from @kahyunnam on [flashinfer-ai/flashinfer#3294](https://github.com/flashinfer-ai/flashinfer/issues/3294#issuecomment-4426701270) too — they have the diagnostic dump request.

--
author:	idonati
association:	none
edited:	false
status:	none
--
Quick concrete data on the shape mismatch — turns out the `flashinfer_b12x_fused_moe` call coming through `FlashInferB12xExperts.apply()` is **already** receiving local-sized weights:

```
[B12X-PATCH] w1.shape=(48, 4096, 3072)
             w2.shape=(48, 6144, 1024)
             global_num_experts=384
             self.num_local_experts=48
             w1_scale.shape=(48, 4096, 384)
             w1_sf_mma.shape=(32, 4, 32, 4, 96, 48)
             g1_alphas.shape=(48,)
             a2_gscale.numel=48
```

(Diagnostic added at the top of `FlashInferB12xExperts.apply()` to dump shapes on the first call per instance.)

Per-expert dim 0 is 48 (= local), `self.num_local_experts=48`, scales/alphas are local. After `permute(1,2,0)` in `_get_weight_views` this should produce `b_w13.shape = (4096, 3072, 48)` — matching `packed_a.shape[2] = 48`.

But the kernel still fails with `b_w13: Tensor([4096, 3072, 384])` — the `384` (= `global_num_experts`) coming from somewhere else, **not from this `apply()` call**. The `[B12X-PATCH]` line and the `Mismatched b_w13.shape[2]` exception are coming from different worker ranks (TP0 logged once, TP3 hit the kernel error), and only one rank logged — so either:

1. A different MoE code path bypasses `FlashInferB12xExperts.apply()` for some layers (shared/routed experts? `mimo_v2` has `MimoV2SparseMoeBlock` with optional `shared_expert` and routed experts that might use different expert classes). The non-b12x path could be running the kernel with global weights.
2. Some allgather/expert-redistribute step expands `b_w13` from `(48, ...)` to `(384, ...)` between `apply()` and the kernel ffi call.

Either way, the EP-slicing isn't the bug at the `apply()` boundary — it's already correct there.

Could you (or anyone reading) confirm whether `FlashInferB12xExperts` is the only code path that calls `flashinfer.b12x_fused_moe` on a `mimo_v2` (or similar dual-MoE-block) model? If there's a second entry point on the `shared_experts` / fall-back-experts / fused MoE path that doesn't go through this class, that's where to look next.

Patch I'm running is just a no-op diagnostic at the top of `apply()` plus a conditional slice that didn't fire (since `w1.shape[0] == self.num_local_experts == 48`); happy to share the full patch + reproducer if useful. Tag: `local/vllm-mimo-b12x:v7w` plus the runtime patch — equivalent to `pip install nvidia-cutlass-dsl[cu13]>=4.5.0` (the depaulmillz/NVIDIA/cutlass#3227 fix) + the v0.6.11 flashinfer stack + Phoenix-Shen V-padding patch.

Steps so far that all worked:
- ✅ cute-dsl JIT (NVIDIA/cutlass#3227 wheel-layering fix)
- ✅ V-padding fallback (Phoenix-Shen)
- ✅ FLASHINFER_B12X backend selection
- ✅ EP=8 weight load (71.24 GiB / rank, 48 local experts / rank)
- ✅ `FlashInferB12xExperts.apply()` receives local-sized weights from vLLM correctly
- ❌ Some _other_ kernel call (probably shared/routed-expert path on mimo_v2) hits the shape mismatch

--
author:	idonati
association:	none
edited:	false
status:	none
--
Another data point, narrowing this down further:

I added two log lines inside `FlashInferB12xExperts.apply()` — one at the very top after `top_k = topk_ids.shape[1]`, another immediately before `flashinfer_b12x_fused_moe(...)`. After fixing the patch delivery so it reaches all 8 worker containers (the `docker cp -` form I had needed to be `docker exec -i ... tee`), I see:

```
[B12X-PATCH]            ←  fires 8 times (once per TP/EP rank)
[B12X-PATCH-PRE-FFI]    ←  fires 0 times  (never reaches the kernel call site)
Mismatched b_w13.shape[2] ... expected to be 384   ←  fires
```

All 8 ranks log the same:
```
w1.shape=(48, 4096, 3072)
w2.shape=(48, 6144, 1024)
global_num_experts=384
self.num_local_experts=48
w1_scale.shape=(48, 4096, 384)        # last 384 = K/16 scale factors, not experts
w1_sf_mma.shape=(32, 4, 32, 4, 96, 48) # last 48 = num_groups = num_local_experts
g1_alphas.shape=(48,)
a2_gscale.numel=48
```

So:
1. `apply()` enters and runs the first log block on every rank — vLLM IS giving us local weights.
2. Whatever happens between that log and the `flashinfer_b12x_fused_moe(` call does not appear to be our slice logic — there's no slice to do (`w1.shape[0]==48==num_local_experts`), and the slice block in my patch is just `if w1.shape[0] > _state_E: ... else: _s = 0; _e = _state_E` with no operation in the else.
3. The kernel error fires anyway.

Either:
- The "Mismatched" exception is coming from a `flashinfer_b12x_fused_moe` invocation that **doesn't pass through our `apply()`** at all — there's a pre-warm / compile / autotune path that calls the kernel ahead of the modular MoE forward.
- Or my second log (immediately before the FFI call) is raising in `%s` formatting and being silently swallowed by the surrounding `try/except`, and our `apply()` *does* reach the kernel — in which case the global 384 in `b_w13.shape[2]` would have to come from inside `b12x_fused_moe` → `launch_sm120_moe` (where `_get_weight_views` does `w1.permute(1,2,0)` — that should produce dim2=48 from our `w1.shape[0]=48`, so something downstream from there is expanding it).

Two questions for you / @kahyunnam:

1. Is there a separate pre-flight code path that constructs and invokes `b12x_fused_moe` outside of `FlashInferB12xExperts.apply()` — for example a cache-warmup, an autotuner over expert counts, or a `_dummy_run` that synthesizes its own MoE inputs? (The error traceback bottoms out in `gpu_model_runner.profile_run() → self._dummy_run(...)`, so `_dummy_run` is the entrypoint — but that should ultimately just call the model's forward and route through MoE.apply, right?)

2. If `apply()` *does* reach the kernel call with `w1.shape=(48, 4096, 3072)`, where would `b_w13.shape = (4096, 3072, 384)` come from on the receiving side? The `_get_weight_views` permute is `permute(1, 2, 0)` which gives `(4096, 3072, 48)` from our local `w1`, not 384.

Happy to drop a `sys.stderr.write` (no `try/except`, no logger) right before the `flashinfer_b12x_fused_moe(` call to definitively answer #2 — will do that on the next iteration. Just thought you'd want this data point now.

(All running on `local/vllm-mimo-b12x:v7w` + the runtime patch added in the launcher; cute-dsl wheel-collision fixed per [NVIDIA/cutlass#3227](https://github.com/NVIDIA/cutlass/issues/3227); model = `festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8`; `-tp 8 --enable-expert-parallel`, EP=8.)

--
author:	idonati
association:	none
edited:	false
status:	none
--
Found the architectural mismatch. In `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py:438-450`:

```python
b_w13_fake = cute.runtime.make_fake_compact_tensor(
    ab_dtype,
    (w1_rows, k, weight_E),           # ← weight_E = num_experts (global, 384)
    stride_order=(1, 0, 2),
    assumed_align=16,
)
...
b_down_fake = cute.runtime.make_fake_compact_tensor(
    ab_dtype,
    (k, n, weight_E),                  # ← weight_E (global) again
    stride_order=(1, 0, 2),
    assumed_align=16,
)
```

The fake tensors used to JIT-compile the static kernel set the expert-dim to `weight_E = num_experts` (global, 384 for our case). The runtime validator then expects `b_w13.shape[2] == 384` regardless of how many experts are actually local on this rank. Same for `b_down.shape[2]`.

But the EP-aware buffers in the SAME workspace are sized with `state_E = num_local_experts` (48):

```python
row_counts:              [state_E]                       # 48
weight_expert_ids:       [state_E]                       # 48
compact_topk_ids:        [state_E]                       # 48
packed_input:            [state_E, max_rows, k//2]       # 48
token_map / token_weights: [state_E, max_rows]
global_to_local_expert:  [weight_E]                      # 384  (the map)
```

So the kernel's design is: "weights are addressed globally (`weight_E`), routing/state is local (`state_E`), and `global_to_local_expert` bridges the two." That requires the caller to **pass global-shaped `w1` / `w2`** even on EP runs.

But on consumer Blackwell (DGX Spark, RTX PRO 6000, RTX 50 series), the FP4 weight per global expert is large (we're at ~1.18 GB per expert for MiMo-V2.5-Pro), so passing 384 global experts per rank is ~454 GB just for `w1` + similar for `w2` — well above GB10's 128 GB unified memory. EP slicing is necessary precisely because we *can't* hold the global weight set on each rank.

So there's a genuine architectural conflict:

- **vLLM** (correctly per its EP design) slices `w13_weight` / `w2_weight` to `num_local_experts` on each rank before handing them to `FlashInferB12xExperts.apply()`. Verified: `w1.shape == (48, 4096, 3072)` on all 8 ranks.
- **b12x kernel** (per `moe_dispatch.py:438-450`) is JIT-compiled to expect `b_w13.shape[2] == weight_E == num_experts` globally — 384 in our case.

Two paths to a fix:

1. **kernel-side**: change `b_w13_fake` / `b_down_fake` to use `state_E` (= local), and have the kernel index `b_w13[..., row_counts.expert_id]` (a local index in `[0, state_E)`) rather than via `global_to_local_expert` lookup. This is the EP-correct design but a non-trivial kernel rewrite, since the `weight_expert_ids` buffer currently holds global indices.
2. **vLLM-side**: don't slice for EP when the b12x backend is selected — pass full global `w13_weight` to `apply()`. Memory-prohibitive on consumer Blackwell, would essentially disable EP for this backend (the whole point of EP is to *not* keep global weights per rank). Possibly OK for small experts but not for our 1B-per-expert MoE.

(1) seems to match what the kernel's `state_E` plumbing *is already doing* for the activation side — `packed_a`, `row_counts`, `weight_expert_ids`, `compact_topk_ids`, `token_map`, `token_weights` are all `state_E`. The two weight tensors are the only ones still on `weight_E`. So the spec might have been an oversight rather than intentional. cc @meena-at-work — any reason `b_w13` / `b_down` need to be `weight_E` rather than `state_E`?

3. **workaround (us, in-place)**: zero-pad local `w1` / `w2` to `weight_E` size before calling. Memory unfeasible at our scale (~568 GB per rank).

For now this PR can land safely with a guard that disables `flashinfer_b12x` autoselection whenever `num_local_experts < num_experts` (i.e., any EP > 1 deployment) on consumer Blackwell, with a clear log line pointing at the upstream FlashInfer item for tracking. Will file the kernel issue separately on FlashInfer side now; just wanted to drop this here first since it closes the loop on what I was tracking on this PR.

--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
@idonati  -- can you use TP (instead of EP) to run your workload? That should enable you to use sharded weights.
--
author:	ECMGit
association:	contributor
edited:	true
status:	none
--
Hi @meena-at-work I tested this PR on DGX Spark (GB10, sm_121a) against
`nvidia/Qwen3.6-35B-A3B-2.0GB-per-token-CT`, a mixed-precision
compressed-tensors checkpoint with NVFP4 W4A16 experts **plus** a few
FP8 experts. Hit two engine-init blockers that aren't exercised by this
PR's tests. Tiny fixes; happy to send as follow-ons.

### 1. FP8 dispatcher rejects `flashinfer_b12x`

`--moe-backend` is consumed by both the NVFP4 oracle (which this PR
extends — works) and the FP8 oracle (which is unchanged). On a mixed
checkpoint, `map_fp8_backend` then raises:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for FP8 MoE.
Expected one of ['triton', 'deep_gemm', 'cutlass', 'flashinfer_trtllm',
                 'flashinfer_cutlass', 'marlin', 'aiter'].
```

Extending the FP8 allow-list would be wrong — b12x is NVFP4-only by
construction (the class docstring is explicit). Suggest a
`VLLM_FP8_MOE_BACKEND` env that overrides **only** the FP8 dispatcher.
Then `--moe-backend=flashinfer_b12x VLLM_FP8_MOE_BACKEND=flashinfer_cutlass`
routes NVFP4 experts to b12x and FP8 experts to FI cutlass. Unset
behavior unchanged.

```python
# vllm/model_executor/layers/fused_moe/oracle/fp8.py
 def map_fp8_backend(runner_backend: MoEBackend) -> Fp8MoeBackend:
+    effective = envs.VLLM_FP8_MOE_BACKEND or runner_backend
     mapping = {
         "triton": Fp8MoeBackend.TRITON,
         ...
     }
-    if backend := mapping.get(runner_backend):
+    if backend := mapping.get(effective):
         return backend
+    src = ("VLLM_FP8_MOE_BACKEND env var"
+           if envs.VLLM_FP8_MOE_BACKEND else "--moe-backend")
     raise ValueError(
-        f"moe_backend='{runner_backend}' is not supported for FP8 MoE. "
+        f"FP8 MoE backend='{effective}' (from {src}) is not supported. "
         f"Expected one of {list(mapping.keys())}."
     )
```

Plus a one-line `VLLM_FP8_MOE_BACKEND` registration in `vllm/envs.py`
alongside the existing `VLLM_FLASHINFER_MOE_BACKEND`.

### 2. `FlashInferB12xExperts._supports_quant_scheme` rejects W4A16

Once (1) is unblocked, the NVFP4 oracle calls
`FlashInferB12xExperts._supports_quant_scheme(weight_key, activation_key)`.
W4A16 NVFP4 checkpoints have `activation_key is None` (BF16
activations, no static activation-quant declared), so the check fails:

```
ValueError: NvFp4 MoE backend 'FLASHINFER_B12X' does not support the deployment
configuration since kernel does not support quantization scheme
QuantKey(u8, scale(f8e4m3fn, static, GroupShape(row=1, col=16)),
         scale2(f32, static, per_tensor), symmetric) x None.
```

But the kernel itself handles BF16 input fine — the class docstring
already says *"Input quantization (BF16→FP4) is performed inside the
kernel so BF16 hidden states are passed directly."* So this is an
over-strict metadata gate, not a kernel limitation. One-line loosening:

```python
# vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
-        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)
+        return (weight_key, activation_key) in (
+            (kNvfp4Static, kNvfp4Dynamic),
+            (kNvfp4Static, None),
+        )
```

------------
Let me know if you need me to add a PR on top of it to make these change.
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
@ECMGit , This PR is specifically for NVFP4 (W4A4) GEMM and MoE. @askliar has some WIP patches for W4A16 in https://github.com/askliar/vllm/tree/askliar/b12x-with-tinygemm.
--
author:	gbanyan
association:	none
edited:	false
status:	none
--
@idonati picking up your TP-vs-EP thread with a Spark/SM121 TP=1 data point — turns out the kernel **does** fail on TP=1 too, just farther in. Posting raw observations in case it's useful before you file the FlashInfer-side issue.

**Hardware/setup**
- DGX Spark (GB10, sm_121a), single GPU
- Sehyo `Qwen3.5-122B-A10B-NVFP4` (compressed-tensors W4A4 NVFP4, 128 experts) + z-lab `Qwen3.5-122B-A10B-DFlash` drafter, `--num_speculative_tokens 15`
- `tensor-parallel-size 1`, no `--enable-expert-parallel` → `num_local_experts == num_experts == 128`. So no EP slice; the `weight_E vs state_E` mismatch you documented shouldn't apply, and the kernel doesn't raise it.
- vLLM PR #40082 @ `7a16942d`, FlashInfer 0.6.11, `nvidia-cutlass-dsl[cu13]>=4.5.0` clean install per NVIDIA/cutlass#3227, `--moe-backend flashinfer_b12x --attention-backend flash_attn`.

**Independent confirmation of #3227**
Before the clean cu13 install: `pip list` showed only `nvidia-cutlass-dsl` + `nvidia-cutlass-dsl-libs-base` (no `libs-cu13`); pip warned `Skipping nvidia-cutlass-dsl-libs-cu13 as it is not installed`. After the single-RUN clean uninstall + `pip install "nvidia-cutlass-dsl[cu13]>=4.5.0"`, all three at 4.5.0 and `warp/mma.py` SHA changed (`4fc8898d…` → `c9171964…`). Wheel-collision is real and reproducible; ptxas `_mma` rejection is gone after the fix.

**What works on TP=1**
- vLLM oracle picks `FLASHINFER_B12X` for MoE and `FlashInferB12xNvFp4LinearKernel` for linear.
- Weight load and `process_weights_after_loading` complete cleanly; `MoEPrepareAndFinalizeNoDPEPModular` initialized.
- cuTeDSL JIT-compile of `launch_sm120_dynamic_moe` returns a callable.

**What fails on TP=1**
First MoE forward inside `profile_run() → _dummy_run(...)`:

```
File ".../vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py", line 207, in apply
    flashinfer_b12x_fused_moe(...)
File ".../flashinfer/fused_moe/cute_dsl/b12x_moe.py", line 139, in b12x_fused_moe
    return launch_sm120_moe(...)
File ".../flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py", line 1742, in launch_sm120_moe
    return launch_sm120_dynamic_moe(...)
File ".../flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py", line 1542, in launch_sm120_dynamic_moe
    compiled(...)
File ".../nvidia_cutlass_dsl/.../tvm_ffi_provider.py", line 593, in __call__
    return tvm_ffi.Function.__call__(self, *args)
File "python/tvm_ffi/cython/function.pxi", line 929, in tvm_ffi.core.Function.__call__
RuntimeError: CUDA Error: cudaErrorInvalidValue
```

So the kernel JIT-compiles, gets invoked, and CUDA rejects an argument at launch. The `tvm_ffi` wrapper doesn't surface which arg — happy to retry with `enforce_eager=True` to bypass torch.compile/cudagraph capture, or with spec-decode disabled, if either would help isolate. The full log is local; can share more on request.

**Open question**
Is there a reference TP=1 b12x-MoE-active run on any sm_120/121 hardware? @meena-at-work's measurements in the PR description had b12x on linear only with MoE held on `cutlass`, so I'm not sure there's a single-GPU baseline with `launch_sm120_dynamic_moe` actually invoked. If this is the first TP=1 attempt full stop, then `cudaErrorInvalidValue` could be something the kernel has never exercised — happy to dig further with `enforce_eager=True` or spec-decode disabled if it helps.

--
author:	lukealonso
association:	none
edited:	false
status:	none
--
@gbanyan @ECMGit So the exercise here is going to be figuring out what got lost in translation between https://github.com/lukealonso/b12x and the resulting flashinfer port/copy  and the previous integration PR https://github.com/vllm-project/vllm/pull/39634 . I'll take a look today.
--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
@mgoin  @pavanimajety  -- with the merge of flashinfer 0.6.11.post2 into vLLM, this PR is now unblocked. I've rebased the branch to the latest vLLM, and am able to successfully run with this backend.

Could you please review the PR so it can be merged?
--
author:	alexbi29
association:	none
edited:	false
status:	none
--
Tested this PR against a real compressed-tensors W4A4 NVFP4 checkpoint (`RedHatAI/Qwen3-Coder-Next-NVFP4`, 512 experts, TP=2, SM120) and hit two bugs in `process_weights_after_loading` that produce either a crash or garbled output. Sharing findings + fixes in case they're useful before this merges.

---

### Bug 1 — `convert_to_nvfp4_moe_kernel_format` asserts B12X out

`FLASHINFER_B12X` is added to `FLASHINFER_NVFP4_MOE_BACKENDS`, so it falls into the generic `elif` branch in `convert_to_nvfp4_moe_kernel_format` and calls `prepare_nvfp4_moe_layer_for_fi_or_cutlass`. That function has an explicit assert that B12X is not in its supported list → `AssertionError` at model load.

**Fix** (`oracle/nvfp4.py`): add a no-op branch before the generic FLASHINFER branch:

```python
elif nvfp4_backend == NvFp4MoeBackend.FLASHINFER_B12X:
    # FlashInferB12xExperts.process_weights_after_loading handles all prep.
    pass
elif nvfp4_backend in FLASHINFER_NVFP4_MOE_BACKENDS or ...:
    ...
```

---

### Bug 2 — `process_weights_after_loading` produces garbled output

After fixing Bug 1, the model loads but generates garbage. Two sub-issues in `FlashInferB12xExperts.process_weights_after_loading`:

**2a. Missing gate/up reorder.** Compressed-tensors checkpoints store FC1 weights as `[gate, up]` = `[w1, w3]`, but the SM12x kernel expects `[up, gate]` = `[w3, w1]` (same convention as `FLASHINFER_CUTLASS`). Without the reorder, the kernel computes `silu(up) * gate` instead of `silu(gate) * up`.

**Fix**: apply `reorder_w1w3_to_w3w1` directly on `layer.w13_weight` and `layer.w13_weight_scale` when `moe_config.is_act_and_mul`:

```python
if self.moe_config.is_act_and_mul:
    reordered_w, reordered_s = reorder_w1w3_to_w3w1(
        layer.w13_weight, layer.w13_weight_scale
    )
    layer.w13_weight.data = reordered_w
    layer.w13_weight_scale.data = reordered_s
```

**2b. Missing `swizzle_blockscale` before `convert_sf_to_mma_layout`.** Checkpoint block scales are in plain (row-major) layout, but `convert_sf_to_mma_layout` expects swizzled input (the output of `fp4_quantize(..., is_sf_swizzled_layout=True)`). Internally, `_get_weight_views` calls `convert_sf_from_mma_layout` (the inverse) to recover the 2D swizzled format for TMA; if the input to `convert_sf_to_mma_layout` was plain, the recovered format is also plain — and the kernel misinterprets it as swizzled, producing noise.

**Fix**: call `swizzle_blockscale` first, then use the padded dimensions for `convert_sf_to_mma_layout`:

```python
w1_scale_swizzled = swizzle_blockscale(self.w1_scale)   # plain → swizzled
E, M_pad, K_sf_pad = w1_scale_swizzled.shape
self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
    w1_scale_swizzled.reshape(E * M_pad, K_sf_pad),
    m=M_pad, k=K_sf_pad * 16, num_groups=E,
)
# same for w2
```

This matches what `prepare_nvfp4_moe_layer_for_fi_or_cutlass` already does for the CUTLASS/CUTEDSL paths (see `flashinfer_fp4_moe.py` lines 122–144).

---

### Separate FlashInfer-side bug (filed as flashinfer-ai/flashinfer#3359)

There's also a bug in `moe_dispatch.py` where both dispatch call sites compile a 42-parameter kernel but omit `max_active_clusters` (`mac`) when invoking it:
```python
compiled, mac = _get_dynamic_kernel(...)
compiled(*runtime_args, current_cuda_stream())   # mac missing → TypeError
```
Fix: `compiled(*runtime_args, mac, current_cuda_stream())`. Both static/micro and dynamic paths are affected. Filed upstream; workaround is to patch the installed wheel.

---

### Results after all three fixes

`RedHatAI/Qwen3-Coder-Next-NVFP4`, TP=2, SM120, FlashInfer 0.6.11.post3:

```
avg: 189.5 tok/s  median: 189.8  min: 188.6  max: 190.3  (5 runs, 2048 tokens)
```

Happy to send these as follow-on commits or a separate small PR if that's easier than modifying this one.
--
author:	pavanimajety
association:	member
edited:	false
status:	approved
--

--
author:	meena-at-work
association:	contributor
edited:	false
status:	none
--
@WoosukKwon @mgoin @youkaichao --  can one of  you please review the PR?
--
