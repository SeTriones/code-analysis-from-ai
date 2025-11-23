# FlashInfer

## Project Overview

FlashInfer is a kernel library and generator for Large Language Model (LLM) serving. It provides high-performance implementations of GPU kernels such as FlashAttention, SparseAttention, PageAttention, and Sampling. It is designed to be integrated into LLM serving systems like vLLM, MLC-LLM, and SGLang.

### Key Features
*   **Efficient Attention:** Optimized kernels for sparse/dense KV-storage on CUDA and Tensor Cores.
*   **Cascade Attention:** Hierarchical KV-Cache support.
*   **JIT Compilation:** Customizable attention variants via Just-In-Time compilation.
*   **Framework Support:** PyTorch, TVM, and C++ APIs.

### Architecture
*   **`include/`**: Framework-agnostic CUDA kernel definitions (header-only).
*   **`csrc/`**: PyTorch bindings and operator registration.
*   **`flashinfer/`**: The core Python package.
*   **`benchmarks/`**: Performance benchmarking scripts.
*   **`tests/`**: Unit tests.

For detailed implementation specifics, including supported backends and selection strategies, refer to [arch.md](arch.md).

## Building and Running

### Installation

**From PyPI (User):**
```bash
pip install flashinfer-python
```

**From Source (Developer):**
Recommended for development to ensure changes are reflected immediately.
```bash
git clone --recursive https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
pip install --no-build-isolation -e . -v
```
*Note: `--no-build-isolation` is recommended to avoid dependency conflicts with `torch`.*

### Running Tests

Tests are located in the `tests/` directory and use `pytest`.

**Run all tests:**
```bash
pytest tests/
```

**Run specific kernel tests:**
Use the scripts provided in `scripts/` for partitioned testing (useful for CI):
```bash
bash scripts/task_jit_run_tests_part1.sh
```

### Benchmarking

Benchmark scripts are located in `benchmarks/`.

**Run the main benchmark:**
```bash
python benchmarks/flashinfer_benchmark.py
```

**Run specific operator benchmarks:**
```bash
python benchmarks/bench_batch_decode.py
```

## Development Conventions

### Code Structure rules
*   **Kernels (`include/`):** Write raw CUDA code here. Do NOT include Torch headers.
*   **Bindings (`csrc/`):** Write PyTorch bindings and registration here. This is where Torch headers are allowed.
*   **Python (`flashinfer/`):** Python interface exposed to users.

### Linting and Formatting

**Python:**
The project uses `ruff` for linting and `mypy` for type checking. Configuration is in `pyproject.toml`.
```bash
bash scripts/task_lint.sh
bash scripts/task_mypy.sh
```

**C++:**
C++ code is checked using `cpplint`.
```bash
bash scripts/task_cpplint.sh
```

### Versioning
Follows a "right-shifted" scheme (`major.minor.patch[.post1]`).
*   **Major:** Incompatible API changes.
*   **Minor:** Significant backwards-compatible features.
*   **Patch:** Bug fixes and small features.
