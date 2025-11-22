# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Testing
```bash
# Build the extension (requires NVSHMEM_DIR environment variable)
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py build

# Create symbolic links for built SO files
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# Run test cases (modify tests/utils.py init_dist for your cluster settings)
python tests/test_intranode.py
python tests/test_internode.py
python tests/test_low_latency.py

# Install the package
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py install
```

### Environment Variables for Building
- `NVSHMEM_DIR`: Path to NVSHMEM installation (required for internode/low-latency features)
- `DISABLE_SM90_FEATURES`: 0/1 to disable SM90 features (required for SM90 devices or CUDA 11)
- `TORCH_CUDA_ARCH_LIST`: Target architectures (e.g., "9.0")
- `DISABLE_AGGRESSIVE_PTX_INSTRS`: 0/1 to disable aggressive PTX instructions

## Architecture Overview

DeepEP is a CUDA-based communication library for Mixture-of-Experts (MoE) models providing three main types of all-to-all communication kernels:

### Core Components

1. **Python Layer (`deep_ep/`)**
   - `buffer.py`: Main Buffer class handling communication buffers and dispatch/combine operations
   - `utils.py`: EventOverlap class for communication-computation overlap
   - `__init__.py`: Exports Buffer, EventOverlap, and Config classes

2. **C++/CUDA Layer (`csrc/`)**
   - `deep_ep.cpp`: PyBind11 bindings and main Buffer class implementation
   - `kernels/`: CUDA kernels organized by functionality
     - `intranode.cu`: NVLink-based communication for single node
     - `internode.cu`: RDMA-based communication for multi-node
     - `internode_ll.cu`: Low-latency RDMA kernels
     - `layout.cu`: Token layout computation for dispatch operations
     - `runtime.cu`: Runtime utilities and memory management

3. **Key Architecture Concepts**
   - **Dispatch**: Scattering tokens to experts based on top-k indices
   - **Combine**: Gathering results from experts back to original positions
   - **Normal Mode**: High-throughput kernels for training/inference prefilling
   - **Low-Latency Mode**: Pure RDMA kernels for inference decoding with hook-based overlapping
   - **Buffer Management**: Queued communication buffers with NVLink and RDMA regions

### Communication Modes

1. **Intranode**: Uses NVLink for high-bandwidth communication within a single node
2. **Internode**: Uses RDMA over InfiniBand for multi-node communication
3. **Low-Latency**: Pure RDMA with minimal latency for inference decoding

### Dependencies and Integration

- **NVSHMEM**: Required for internode and low-latency functionality
- **PyTorch**: Primary integration framework via torch.distributed
- **CUDA**: Core GPU computation and memory management
- **RDMA/InfiniBand**: Network communication infrastructure

### Testing Framework

Tests are located in `tests/` and require cluster configuration in `tests/utils.py`. The `init_dist` function should be modified according to your specific cluster settings.

### Important Implementation Details

- The library uses aggressive PTX optimizations that may be platform-specific
- Buffer management uses queues to save memory but adds complexity
- Low-latency mode supports hook-based communication-computation overlap without SM occupation
- FP8 support is available for SM90 (Hopper) architectures only

### Build Configuration

The build system automatically detects CUDA architecture and enables appropriate features:
- SM90 (Hopper): FP8 support, aggressive PTX instructions enabled by default
- SM80 (Ampere): Limited features, no FP8 support
- NVSHMEM dependency can be disabled for intranode-only builds