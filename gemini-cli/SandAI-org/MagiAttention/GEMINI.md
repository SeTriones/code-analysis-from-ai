# MagiAttention

## Project Overview

**MagiAttention** is a distributed attention mechanism (Context Parallelism strategy) designed for training Large Language Models (LLMs) with **ultra-long contexts** and **heterogeneous mask types**. It aims to achieve linear scalability with respect to context-parallel size.

### Key Features
*   **Flexible Flash Attention (FFA):** A generalized kernel for irregular attention mask patterns.
*   **Linear Scalability:** Efficient computation load balancing and zero-redundant communication.
*   **Communication Primitives:** Custom `GroupCast` and `GroupReduce` based on All-to-All-v.
*   **Integration:** Supports PyTorch Native (FSDP), Megatron-LM, and Hugging Face Transformers.

### Tech Stack
*   **Languages:** Python, C++, CUDA (kernels).
*   **Frameworks:** PyTorch, CUTLASS.
*   **Hardware:** Optimized for NVIDIA Hopper GPUs (e.g., H100).
*   **Build System:** `setuptools`, `torch.utils.cpp_extension`.

### Directory Structure
*   `magi_attention/`: Main Python package.
    *   `api/`: Public API exposed to users (e.g., `calc_attn`, `dispatch`).
    *   `csrc/`: C++ and CUDA source code for custom kernels and communication.
    *   `common/`: Shared utilities and data structures (`AttnRanges`, `AttnMaskType`).
    *   `functional/`: Functional interfaces for kernels.
*   `examples/`: Integration examples for various frameworks.
*   `tests/`: Unit tests (pytest).
*   `tools/`: Build and codestyle tools.

## Building and Running

### Prerequisites
*   **OS:** Linux (Darwin/macOS is detected but project is CUDA/Linux heavy).
*   **Hardware:** NVIDIA Hopper GPU recommended.
*   **Software:**
    *   CUDA Toolkit (12.8+ recommended).
    *   PyTorch (NGC container recommended).
    *   `nvidia-nvshmem` (often required for communication kernels).

### Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install from Source:**
    ```bash
    # Initialize submodules (Cutlass, etc.)
    git submodule update --init --recursive
    
    # Install with pip (avoids build isolation for torch extension visibility)
    pip install --no-build-isolation .
    ```

### Running Tests

The project uses `pytest`. Ensure you have a GPU environment set up for most tests.

```bash
# Run all tests
pytest tests/

# Run with coverage
make coverage
```

### Common Commands (Makefile)

*   `make format`: Formats C++/CUDA code using `clang-format`.
*   `make format-check`: Checks C++/CUDA code formatting without modifying.
*   `make build`: Builds the wheel package.
*   `make clean`: Cleans build artifacts (`build/`, `dist/`, `*.egg-info`).

## Development Conventions

### Code Style
*   **C++/CUDA:** strictly follows `clang-format`.
    *   Run `make format` before committing.
    *   A `.clang-format` file is present in the root.
*   **Python:** Follows standard PEP 8, enforced via `pre-commit` hooks.
*   **Pre-commit:**
    *   Install hooks: `pre-commit install`
    *   Run manually: `pre-commit run -a`

### Contribution Workflow
1.  **Fork** the repository.
2.  **Create** a feature branch (`git checkout -b feature/MyFeature`).
3.  **Commit** changes.
4.  **Verify** with `pytest` and `make format-check`.
5.  **Push** and open a **Pull Request**.

### Key API Entry Points (`magi_attention.api`)
*   `calc_attn`: Main function for distributed attention computation.
*   `dispatch`: Dispatches data for context parallelism.
*   `flex_flash_attn_func`: Direct interface to the Flexible Flash Attention kernel.
*   `undispatch`: Gathers data back after attention.
