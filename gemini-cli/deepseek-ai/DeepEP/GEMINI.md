# DeepEP - Expert Parallel Communication Library

## Project Overview
DeepEP is a high-performance communication library designed for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides optimized all-to-all GPU kernels (dispatch and combine) supporting high throughput and low latency, critical for large-scale model training and inference (e.g., DeepSeek-V3).

**Key Features:**
- **Optimized Kernels:** High-throughput kernels for asymmetric-domain bandwidth forwarding (NVLink to RDMA).
- **Low-Latency Mode:** Pure RDMA kernels for latency-sensitive inference decoding.
- **Hardware Support:** Optimized for NVIDIA Ampere (SM80) and Hopper (SM90) architectures.
- **Precision:** Supports FP8 low-precision operations.

## Architecture
The project consists of a Python package (`deep_ep`) backed by high-performance C++/CUDA extensions (`deep_ep_cpp`).

- **`csrc/`**: Contains the C++ and CUDA source code for the kernels.
  - `deep_ep.cpp`: Pybind11 bindings.
  - `kernels/`: Implementation of intranode, internode, and low-latency kernels.
- **`deep_ep/`**: Python wrapper package providing the user-facing API (`Buffer` class, etc.).
- **`tests/`**: Python scripts for testing and benchmarking.
- **`third-party/`**: Contains NVSHMEM patches and documentation.

## Building and Installation

### Prerequisites
- **OS:** Linux (implied by typical HPC/GPU setups).
- **Hardware:** NVIDIA GPUs (Ampere SM80+, Hopper SM90+ recommended).
- **Software:**
  - Python 3.8+
  - CUDA 11.0+ (SM80) / 12.3+ (SM90)
  - PyTorch 2.1+
  - NVSHMEM (installed separately).

### Installation Steps

1.  **Install NVSHMEM:**
    DeepEP requires NVSHMEM. Follow the instructions in `third-party/README.md` or use a pre-installed version.

2.  **Build & Install DeepEP:**
    Use the standard Python `setup.py` process.

    ```bash
    # Set NVSHMEM location (Required)
    export NVSHMEM_DIR=/path/to/nvshmem

    # Install
    python setup.py install
    ```

    **Environment Variables for Build:**
    - `NVSHMEM_DIR`: Path to NVSHMEM installation.
    - `DISABLE_SM90_FEATURES`: Set to `1` to disable Hopper-specific features (required for SM90 devices or CUDA 11).
    - `TORCH_CUDA_ARCH_LIST`: Target architectures (e.g., `"9.0"` for H800, `"8.0"` for A100).
    - `DISABLE_AGGRESSIVE_PTX_INSTRS`: Set to `1` to disable aggressive PTX instructions if you encounter stability issues (default behavior varies by arch).
    - `TOPK_IDX_BITS`: Bit width for top-k indices (32 or 64).

### Development Build
For development, you can build in-place and symlink the shared object:

```bash
export NVSHMEM_DIR=/path/to/nvshmem
python setup.py build

# Symlink the generated .so file (adjust path based on python version/arch)
ln -s build/lib.linux-x86_64-cpython-*/deep_ep_cpp.*.so deep_ep/
```

## Running Tests
Tests are located in the `tests/` directory and typically run using `torch.multiprocessing` or distributed launchers.

```bash
# Intranode tests
python tests/test_intranode.py

# Internode tests (requires multi-node setup or appropriate local simulation)
python tests/test_internode.py

# Low-latency kernel tests
python tests/test_low_latency.py
```

*Note: You may need to adjust `tests/utils.py` `init_dist` function to match your specific cluster/distributed environment configuration.*

## Development Conventions

- **Code Style:**
  - Python code is linted/formatted using `ruff` and `yapf` (configuration in `pyproject.toml`).
  - C++ code generally follows standard CUDA/C++ practices found in `csrc/`.
- **Git:**
  - The repository uses standard git workflows.
  - `install.sh` is a convenience script for cleaning, building wheels, and installing.
- **CMake:**
  - A `csrc/CMakeLists.txt` exists primarily for debugging purposes. The production build system is `setup.py`.

## Key Files
- `README.md`: Main documentation and usage examples.
- `setup.py`: Build configuration and dependency management.
- `deep_ep/buffer.py`: Core Python logic for managing communication buffers.
- `csrc/deep_ep.cpp`: Entry point for C++ bindings.
