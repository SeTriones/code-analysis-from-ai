# SGLang MoE Runner Architecture

This document details the Mixture-of-Experts (MoE) runner modes supported in SGLang and their application scenarios.

## Supported MoE Runner Modes

SGLang supports **9 modes** (backends) for the MoE runner, defined in `MoeRunnerBackend`:

1.  **`auto`** (Default)
2.  **`deep_gemm`**
3.  **`triton`**
4.  **`triton_kernel`**
5.  **`flashinfer_trtllm`**
6.  **`flashinfer_cutlass`**
7.  **`flashinfer_mxfp4`**
8.  **`flashinfer_cutedsl`**
9.  **`cutlass`**

## Application Scenarios

These modes can be selected via the `--moe-runner-backend` server argument.

| Mode | Scenario & Application | Parallelism (MoE FFN) | Supported Quantization |
| :--- | :--- | :--- | :--- |
| **`auto`** | **General Usage.** This is the default setting. SGLang automatically selects the most appropriate backend based on your hardware (e.g., NVIDIA vs. AMD), installed libraries (`flashinfer`, `deep_gemm`), and model configuration (quantization type). | Auto-selected | Auto-selected |
| **`deep_gemm`** | **DeepSeek Models (Hopper+).** Specifically optimized for DeepSeek-V3 and DeepSeek-R1 models, typically delivering the highest performance on NVIDIA Hopper (H100/H200) and Blackwell GPUs. Requires the `deep_gemm` library. | Expert Parallel (EP) via DeepEP | FP8 (W8A8) |
| **`triton`** | **Standard / Portable.** A general-purpose backend using Triton kernels. It serves as a robust baseline and is often used when specific high-performance kernels (like FlashInfer) are unavailable or incompatible with the current setup. | Tensor Parallel (TP) | FP16/BF16, FP8 (W8A8), INT8 (W8A8), INT8 (W8A16), INT4 (W4A16) |
| **`triton_kernel`** | **Optimized Triton.** A variation of the Triton backend using specific optimized kernels (likely from `sgl-kernel` or internal implementations) for better performance than the standard Triton path in some cases. | Tensor Parallel (TP) | FP16/BF16, FP8, INT8, INT4 |
| **`flashinfer_trtllm`** | **High-Performance FP8/FP4.** Leverages FlashInfer with TensorRT-LLM kernels. This is critical for running quantized models like **DeepSeek-V3 in FP4**, providing significant speedups on supported hardware. | Tensor Parallel (TP) | FP8 (W8A8), FP4 (NVFP4) |
| **`flashinfer_cutlass`** | **FlashInfer + Cutlass.** Used when FlashInfer is available but Cutlass kernels are preferred for specific matrix multiplications, often for specific quantization formats or attention mechanisms coupled with MoE. | Expert Parallel (EP) + Tensor Parallel (TP) | FP4 (NVFP4) |
| **`flashinfer_mxfp4`** | **MXFP4 Quantization.** Specifically designed for models quantized using the MXFP4 (Microscaling formats) standard. | Expert Parallel (EP) + Tensor Parallel (TP) | MXFP4 |
| **`flashinfer_cutedsl`** | **Experimental / CuTe.** A backend using FlashInfer with CuTeDSL (CUDA Template Library), offering high-performance primitives for specific GPU architectures. | Expert Parallel (EP) + Tensor Parallel (TP) | FP4 (NVFP4) |
| **`cutlass`** | **FP8 / GEMM.** A generic backend using Cutlass kernels. It is often used for **FP8 MoE** implementations (`Fp8MoEMethod`) when FlashInfer is not used or for specific GEMM (General Matrix Multiply) operations. | Tensor Parallel (TP) | FP8 (W8A8), FP4 (NVFP4) |

## Configuration

You can explicitly set the backend when launching the server:

```bash
python -m sglang.launch_server --model-path <path> --moe-runner-backend deep_gemm
```

(Replace `deep_gemm` with your desired mode).
