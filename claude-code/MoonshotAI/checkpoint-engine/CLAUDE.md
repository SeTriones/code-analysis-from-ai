# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Checkpoint-engine is a lightweight middleware designed for **reinforcement learning with large language models**. Its primary purpose is to enable efficient in-place weight updates in LLM inference engines during RL training loops.

**Key Use Case**: In reinforcement learning scenarios, model weights are frequently updated based on training feedback, and these updated weights need to be deployed to inference engines that are actively serving requests. Traditional approaches require stopping inference, reloading weights from disk, and restarting - causing significant downtime.

**Solution**: Checkpoint-engine provides a decoupled architecture where:
- Inference engines continue serving requests without interruption  
- Weight updates are streamed directly from training systems or storage to inference engines
- Updates can be applied to hundreds of GPUs across multiple nodes in ~20 seconds (tested with 1T parameter models)

**Target Scenarios**:
1. **RL Training**: Continuous weight updates during RLHF, PPO, or other RL training of LLMs
2. **Dynamic Model Serving**: Hot-swapping model versions without downtime
3. **Multi-tenant Inference**: Different model variants served by the same infrastructure

## Development Commands

### Installation
```bash
# Basic installation
pip install checkpoint-engine

# With P2P support (includes mooncake-transfer-engine)
pip install 'checkpoint-engine[p2p]'
```

### Testing
```bash
# Run correctness test (requires 8 GPUs)
torchrun --nproc-per-node 8 tests/test_update.py
```

### Running Examples
```bash
# Basic weight update example (requires 8 GPUs)
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /path/to/model

# Save global metadata for reuse by new instances
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path /path/to/model \
    --sleep-time 300 --save-metas-file global_metas.pkl

# Load from existing metadata
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

## Architecture

### Core Components

1. **ParameterServer** (checkpoint_engine/ps.py): The main weight update service with two implementations:
   - **Broadcast**: Fast synchronous weight updates for multiple inference instances
   - **P2P**: Asynchronous updates for dynamically added instances using mooncake-transfer-engine

2. **VllmColocateWorkerExtension** (checkpoint_engine/worker.py): vLLM worker extension for coordinating with the parameter server

3. **Weight Update Pipeline**: Three-stage process (H2D → Broadcast → Reload) with overlapped communication and copy for optimal performance

## ParameterServer Architecture

### Class Structure and Key Components

**Initialization & Configuration**:
- Environment-based setup using `RANK`, `WORLD_SIZE`, and `MASTER_ADDR`
- GPU management with automatic local rank detection (`rank % gpu_count`)
- Process group management supporting both automatic and manual initialization
- Optional RDMA-based P2P communication via `P2PStore`

**Key Data Structures**:
- `_memory_pool`: Maps checkpoint names to lists of `MemoryBuffer` objects
- `_current_global_parameter_metas`: Maps owner ranks to `MemoryBufferMetaList`
- `_zmq_ctx`: ZMQ context for IPC communication with inference engines
- Memory alignment to 256 bytes for optimal GPU access

**Memory Management**:
- Uses pinned memory buffers for efficient CPU-GPU transfers
- Double-buffered GPU memory allocation (bucket_size * 2)
- Dynamic bucket sizing based on 90% of available GPU memory

### Broadcast Method Implementation (_update_per_bucket)

**Phase 1: Setup & Memory Allocation**
1. Calculate optimal bucket size based on available GPU memory
2. Generate H2D buckets from global parameter metadata
3. Allocate double-buffered GPU memory for overlap operations
4. Optional H2D buffer allocation for parallel host-to-device transfers

**Phase 2: Coordination Setup**
1. Start request thread for inference engine communication via ZMQ
2. Create IPC socket for memory sharing coordination
3. Send tensor handle via IPC for shared memory access

**Phase 3: Bucket-by-Bucket Processing**
1. For each owner rank's buckets:
   - Copy parameters from memory pool to H2D buffer (if owner)
   - Broadcast buffer content using NCCL (`dist.broadcast`)
   - Coordinate with inference engines via ZMQ sockets
   - Send tensor metadata for parameter reconstruction
2. Use double buffering (`gidx % 2`) to overlap operations

**Key Mechanisms**:
- Memory-efficient broadcasting with rank-specific NCCL sources
- Double buffering to overlap computation and communication
- H2D optimization with fallback when memory constrained
- ZMQ-based IPC coordination for tensor handle sharing

### P2P Method Implementation (_update_per_bucket_p2p)

**Phase 1: P2P Infrastructure Setup**
1. Validate P2P store availability and rank membership
2. Initialize temporary process group for participating ranks only
3. Allocate double-buffered GPU memory
4. Register IPC buffer with RDMA-capable P2P store

**Phase 2: RDMA-based Data Transfer**
1. For each bucket:
   - Rank 0 performs RDMA reads from parameter owners using `batch_transfer_sync_read`
   - Collected data broadcast to all participating ranks via NCCL
   - ZMQ coordination for tensor reconstruction metadata
2. Cleanup: Unregister buffers and destroy temporary process group

**Key Differences from Broadcast**:
- Selective rank participation (dynamic inference instance addition)
- RDMA optimization for direct memory access between nodes
- Temporary process groups for rank subsets
- Designed for disaggregated deployment scenarios

### Coordination Mechanisms

**ZMQ-based IPC Coordination**:
- Each rank binds to unique IPC socket (`ipc://@checkpoint-engine-{uuid}.sock`)
- Tensor handle sharing using `reduce_tensor()` for memory sharing
- Request-response protocol for coordinating tensor updates

**Multi-threading Architecture**:
- Request thread handles inference engine communication
- Main thread manages parameter transfers and synchronization
- ThreadPoolExecutor for parallel operations (up to 32 workers)

**Communication Protocols**:
- Metadata gathering via `dist.all_gather_object()` for global parameter view
- NCCL broadcast with rank-specific sources for parameter distribution
- HTTP API for checkpoint management and health checks

## ParameterServer Method Reference

### Public API Methods

**Core Lifecycle**:
- `__init__(*, auto_pg: bool = False)`: Initialize parameter server with distributed environment setup
- `init_process_group(*, master_port, timeout)`: Set up NCCL process group for distributed communication
- `init_process_group_for_ranks(ranks, *, master_port, timeout)`: Create process group for rank subset (P2P mode)

**Checkpoint Management**:
- `register_checkpoint(checkpoint_name, *, files=[], named_tensors={})`: Load and register model weights from files or tensors
- `unregister_checkpoint(checkpoint_name)`: Remove checkpoint and free resources
- `gather_metas(checkpoint_name)`: Synchronize parameter metadata across all ranks (required before updates)
- `update(checkpoint_name, req_func, *, ranks=[])`: Distribute weights to inference engines (broadcast if ranks=[], P2P if ranks specified)

**Metadata Access**:
- `get_metas()`: Access current global parameter metadata (read-only)
- `load_metas(metas)`: Load previously saved metadata for state restoration

### Private Implementation Methods

**Memory Management**:
- `_get_bucket_size(*, disable_h2d_buffer)`: Calculate optimal bucket size based on available GPU memory
- `_copy_to_buffer(checkpoint_name, bucket, buffer, owner_rank)`: Copy parameters to staging buffers (local or P2P)
- `_register_parameters_to_p2p_store(checkpoint_name)`: Register memory buffers for RDMA access
- `_unregister_parameters_from_p2p_store(checkpoint_name)`: Cleanup P2P memory registrations

**Distribution Implementation**:
- `_update_per_bucket(checkpoint_name, req_func)`: Broadcast method implementation with NCCL collectives
- `_update_per_bucket_p2p(checkpoint_name, req_func, ranks)`: P2P method with RDMA transfers
- `_get_addr_ptrs(owner_rank)`: Retrieve P2P store addresses and memory pointers

**Utilities**:
- `_logger_rank0(msg)`: Rank 0 only logging to avoid spam
- `_zmq_socket_path` (property): Generate unique IPC socket path for ZMQ coordination

### Standalone Utility Functions

**Memory and Layout**:
- `_align_size(dtype, shape)`: Calculate 256-byte aligned tensor size
- `_to_named_tensor(metas, offset)`: Convert metadata to IPC communication format
- `_gen_h2d_buckets(global_metas, bucket_size)`: Generate optimized transfer buckets

**Checkpoint Loading**:
- `_load_checkpoint_file(file_path)`: Load individual safetensors/numpy files with memory mapping
- `_concat_tp_weights(tp_weights, tp_concat_dim, tp_size)`: Reconstruct tensor parallel weights
- `_load_checkpoint(files)`: High-level multi-file checkpoint loading with TP reconstruction
- `_register_checkpoint(...)`: Convert loaded parameters into optimized memory buffers

**Hardware Discovery**:
- `_get_physical_gpu_id(rank)`: Get GPU UUID via nvidia-smi
- `_get_ip()`: Determine machine IP for network communication
- `_get_rdma_devices()`: Discover available RDMA devices
- `_get_my_rdma_device(local_rank, gpu_count, devices)`: Assign RDMA device based on GPU topology
- `_get_master_port(master_port)`: Determine distributed communication port (avoids torchrun conflicts)

**Integration APIs**:
- `request_inference_to_update(url, socket_paths, timeout)`: HTTP request to trigger inference engine weight reload
- `_init_api(ps)`: Create FastAPI application with REST endpoints for parameter server control
- `run_from_cli()`: CLI entry point that starts parameter server with uvicorn

### P2PStore Class (RDMA Support)

- `__init__()`: Initialize mooncake TransferEngine for RDMA communication
- `register_named_tensors(named_tensors)`: Register tensor buffers for remote RDMA access  
- `unregister_named_tensors(names)`: Cleanup RDMA tensor registrations
- `batch_transfer_sync_read(target_hostname, buf_ptrs, remote_ptrs, lens)`: Perform bulk RDMA reads

### REST API Endpoints (via FastAPI)

- `POST /v1/checkpoints/{name}/files`: Register checkpoint from file paths
- `DELETE /v1/checkpoints/{name}`: Unregister checkpoint
- `GET /v1/healthz`: Health check endpoint
- `POST /v1/checkpoints/{name}/gather-metas`: Gather parameter metadata
- `POST /v1/checkpoints/{name}/update`: Trigger weight distribution

### Key Features

- Supports various data types: BF16, FP16, FP8, Float32
- Optimized for large models (tested up to 1T parameters)
- GPU memory-aware pipelining with automatic fallback to serial execution
- RDMA support for P2P transfers between nodes
- Integration with vLLM inference engine via ZeroMQ sockets

### vLLM Integration

Requires vLLM with `/collective_rpc` API endpoint. Start vLLM with:
```bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension \
    --load-format dummy --tensor-parallel-size=8
```

### FP8 Support

FP8 quantization requires applying the patch in `patches/vllm_fp8.patch` to vLLM. Only tested with DeepSeek-V3.1 and Kimi-K2 models.