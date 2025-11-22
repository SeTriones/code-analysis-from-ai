# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup
```bash
# Install SGLang with all dependencies
pip install -e "python/[all]"

# Install for specific hardware configurations
pip install -e "python/[all]"        # NVIDIA GPU
pip install -e "python/[all_hip]"    # AMD GPU  
pip install -e "python/[all_cpu]"    # CPU
pip install -e "python/[all_npu]"    # Ascend NPU

# Install development dependencies
pip install -e "python/[dev]"
```

### Building and Testing
```bash
# Format code (requires isort and black)
make format

# Run test suites
cd test/lang && python run_suite.py --suite per-commit
cd test/srt && python run_suite.py --suite per-commit

# Run individual tests with pytest
pytest test/lang/test_*.py
pytest test/srt/test_*.py

# Run kernel tests
cd sgl-kernel && pytest tests/
```

### Development Tools
```bash
# Benchmark performance
python python/sglang/bench_offline_throughput.py
python python/sglang/bench_serving.py

# Check environment
python python/sglang/check_env.py

# Launch server for testing
python python/sglang/launch_server.py --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

## Architecture Overview

SGLang is a fast serving framework for large language models with a modular architecture:

### Core Components

1. **Frontend Language** (`python/sglang/lang/`)
   - Intuitive programming interface for LLM applications
   - Supports chained generation calls, control flow, multi-modal inputs
   - Key files: `compiler.py`, `interpreter.py`, `ir.py`, `tracer.py`

2. **SRT Backend** (`python/sglang/srt/`)
   - SGLang Runtime - efficient model serving engine
   - Features: RadixAttention (prefix caching), zero-overhead scheduler, continuous batching
   - Key directories: `layers/`, `managers/`, `mem_cache/`, `distributed/`

3. **Kernel Library** (`sgl-kernel/`)
   - High-performance CUDA kernels for optimized inference
   - Custom attention, MoE, quantization, and sampling kernels
   - Built with CMake and CUDA

4. **Router** (`sgl-router/`)
   - Rust-based load balancer with prefill-decode disaggregation
   - Multiple routing algorithms and cache-aware policies

### Key Features

- **RadixAttention**: Automatic prefix caching for efficient repeated text generation
- **Continuous Batching**: Dynamic batching with zero overhead
- **Structured Outputs**: Constrained generation with JSON schema support
- **Multi-Modal Support**: Vision and language models (LLaVA, etc.)
- **Quantization**: FP4/FP8/INT4/AWQ/GPTQ support
- **Parallelism**: Tensor/pipeline/expert/data parallelism
- **Speculative Decoding**: Faster inference with draft models

### Model Support

The framework supports extensive model families:
- **Generative Models**: Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral
- **Embedding Models**: e5-mistral, gte, mcdse
- **Reward Models**: Skywork
- **Multi-Modal Models**: LLaVA series, Qwen-VL, InternVL

### Development Patterns

1. **Model Integration**: Add new models in `python/sglang/srt/configs/` following existing patterns
2. **Kernel Development**: Add CUDA kernels in `sgl-kernel/csrc/` with corresponding Python bindings
3. **Testing**: Use pytest framework with test suites organized in `test/lang/` and `test/srt/`
4. **Benchmarking**: Use provided benchmark scripts in `benchmark/` directory

### Configuration

- Model configurations: `python/sglang/srt/configs/model_config.py`
- Server arguments: `python/sglang/srt/server_args.py`
- Global settings: `python/sglang/global_config.py`

### Build System

The project uses a multi-package structure:
- Main package: `python/` (setuptools)
- Kernel library: `sgl-kernel/` (scikit-build-core with CMake)
- Router: `sgl-router/` (setuptools-rust with PyO3 bindings)

### Python Module Structure

The `python/sglang/` directory contains the following modules:

**Core Modules:**
- `lang/` - Frontend language interface with compiler, interpreter, and backend integrations
- `srt/` - SGLang Runtime backend with layers, managers, and model implementations
- `test/` - Test suite for various components including attention and ops

**Language Module (`lang/`):**
- `backend/` - Backend integrations (Anthropic, OpenAI, LiteLLM, VertexAI)
- `compiler.py`, `interpreter.py`, `ir.py`, `tracer.py` - Core language components
- `api.py`, `chat_template.py`, `choices.py` - API and interface utilities

**SRT Module (`srt/`) - Detailed Structure:**

**Core Infrastructure:**
- `configs/` - Model configurations (ChatGLM, DBRX, DeepSeekVL2, InternVL, etc.) and device settings
- `managers/` - Core scheduling, tokenization, session management, and cache control
- `mem_cache/` - Advanced memory management with radix trees, chunk caching, and storage backends
- `models/` - Extensive model implementations (50+ models including Llama, Qwen, DeepSeek, Gemma, etc.)
- `layers/` - Neural network components with attention, MoE, quantization, and linear layers

**Distributed Computing:**
- `distributed/` - Multi-GPU and multi-node distributed training/inference
  - `device_communicators/` - Hardware-specific communication (CUDA, NPU, HPU, XPU)
  - Communication operations and parallel state management
- `disaggregation/` - Prefill-decode disaggregation with various backends
  - `ascend/`, `mooncake/`, `nixl/` - Platform-specific transfer engines
  - `base/`, `common/`, `fake/` - Abstract and utility implementations

**Serving and APIs:**
- `entrypoints/` - HTTP server, OpenAI API compatibility, and engine interfaces
  - `openai/` - Complete OpenAI API implementation (chat, completions, embedding, etc.)
- `connector/` - External service connectors (Redis, S3) with serialization
- `sampling/` - Advanced sampling with penalty libraries and custom processors

**Advanced Features:**
- `constrained/` - Grammar-based constrained generation (Outlines, XGrammar, LLaMA Guidance)
- `function_call/` - Function calling capabilities with format detectors
- `speculative/` - Speculative decoding with EAGLE implementations
- `lora/` - LoRA fine-tuning support with memory management
- `multimodal/` - Multi-modal processing with various model processors

**Optimization Layers:**
- `layers/attention/` - Multiple attention backends (FlashAttention, FlashInfer, Triton, etc.)
- `layers/moe/` - Mixture of Experts implementations (Cutlass, Triton, DeepEP)
- `layers/quantization/` - Comprehensive quantization support (AWQ, GPTQ, FP8, INT8, etc.)

**Utilities and Support:**
- `model_executor/` - Model execution with CUDA/NPU graph runners
- `model_loader/` - Efficient model loading and weight management
- `debug_utils/` - Debugging and comparison utilities
- `metrics/` - Performance monitoring and timing
- `eplb/` - Expert placement and load balancing for MoE models

**Supporting Modules:**
- `eval/` - Evaluation scripts for different benchmarks
- `bench_*.py` - Benchmarking utilities
- `launch_server.py` - Server launch script
- `global_config.py` - Global configuration settings

## Server Architecture Details

### KV Cache Memory Management

SGLang implements a sophisticated three-tier KV cache memory management system:

#### **Memory Allocation Hierarchy**

1. **ReqToTokenPool** (`mem_cache/memory_pool.py`)
   - Maps requests to their token locations
   - Size: `(size, max_context_len)` tensor
   - Uses free list for allocation

2. **TokenToKVPoolAllocator** (`mem_cache/allocator.py`)
   - Manages indices to KV cache data
   - Multiple allocator types:
     - `TokenToKVPoolAllocator`: Token-level allocation
     - `PagedTokenToKVPoolAllocator`: Page-aligned allocation
     - `SWATokenToKVPoolAllocator`: Sliding Window Attention

3. **KVCache Storage**
   - Physical KV cache storage implementations:
     - `MHATokenToKVPool`: Multi-head attention
     - `MLATokenToKVPool`: Multi-head Latent Attention
     - `SWAKVPool`: Separate pools for full/SWA layers
     - `DoubleSparseTokenToKVPool`: Sparse attention models

#### **Request Lifecycle**

1. **Allocation** (`managers/schedule_batch.py`):
   - Request slot allocation from `ReqToTokenPool`
   - Token slot allocation from `TokenToKVPoolAllocator`
   - Prefix matching via `RadixCache` for reuse

2. **Execution**:
   - KV cache populated during model forward pass
   - Radix cache locks active requests (`lock_ref > 0`)

3. **Release/Reuse**:
   - Immediate free for non-cacheable requests
   - Storage in radix tree for future reuse
   - LRU eviction when memory pressure is high

#### **Advanced Features**

- **Radix Cache**: Prefix caching with radix tree structure
- **Multi-tier Storage**: Host-device cache hierarchy (HiCache)
- **Memory Optimization**: Page alignment, CPU offloading, custom pools
- **Leak Detection**: Automatic memory leak detection and reporting

### KV Cache Allocation and Freeing Lifecycle

#### **1. Request Arrival and Initial Processing**
```python
# Step 1: Request reception and prefix matching
req_pool_indices = self.alloc_req_slots(bs)  # From ReqToTokenPool
ret = self.tree_cache.match_prefix(req.origin_input_ids)  # Check radix cache
if ret.device_indices is not None:
    req.prefix_indices = ret.device_indices  # Reuse existing KV cache
```

#### **2. KV Cache Allocation**
```python
# Step 2: Allocate token slots
out_cache_loc = self.token_to_kv_pool_allocator.alloc(num_tokens)

# Step 3: Update ReqToTokenPool mapping
self.req_to_token_pool.write(
    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
    out_cache_loc[pt : pt + extend_lens[i]],  # KV cache indices
)
```

#### **3. KV Cache Population**
```python
# During model forward pass
layer.set_kv_buffer(
    locs=forward_batch.out_cache_loc,  # Token locations from ReqToTokenPool
    cache_k=new_k,
    cache_v=new_v,
)
```

#### **4. Request Processing**
```python
# Incremental decoding - update mapping for new tokens
self.req_to_token_pool.write(
    (self.req_pool_indices, locs),
    self.out_cache_loc.to(torch.int32)
)
```

#### **5. KV Cache Freeing**
```python
# Request completion
token_indices = self.req_to_token_pool.req_to_token[
    req.req_pool_idx, : seq_lens_cpu[idx]
]
self.token_to_kv_pool_allocator.free(token_indices)  # Free KV slots
self.req_to_token_pool.free(req.req_pool_idx)  # Free request slot
```

#### **6. Optional Radix Cache Storage**
```python
# Store completed request for future reuse
self.tree_cache.insert(req.origin_input_ids, req)
req.finished = True  # Don't free KV cache
```

#### **7. Memory Eviction (When Needed)**
```python
# LRU-based eviction when memory pressure is high
if node.lock_ref == 0:  # Node not in use
    kv_indices = self.req_to_token_pool.req_to_token[
        node.req.req_pool_idx, : node.value_len
    ]
    self.token_to_kv_pool_allocator.free(kv_indices)
    self.req_to_token_pool.free(node.req.req_pool_idx)
```

### Key Data Flow

- **Allocation**: Request → ReqToTokenPool slot → TokenToKVPool slots → Update mapping
- **Freeing**: Complete → Get indices → Free KV slots → Free request slot

### Special Cases

- **Prefix Reuse**: No allocation for matched prefixes
- **Chunked Processing**: Incremental allocation for streaming
- **Eviction**: LRU-based eviction under memory pressure
- **Disaggregated Serving**: Separate prefill/decode handling

## LayerCommunicator Class

The `LayerCommunicator` class in SGLang manages communication patterns for tensor parallelism and data parallelism within transformer layers, optimizing data movement and computation overlap.

### Overview

**Location**: `python/sglang/srt/layers/communicator.py`

**Purpose**: Handles efficient communication between GPUs in distributed inference scenarios, managing tensor distribution patterns, all-reduce operations, and communication-computation overlap.

### Key Concepts

#### Scatter Modes
- `SCATTERED`: Each rank has its portion of data
- `TP_ATTN_FULL`: All ranks in tensor parallel group have full attention data
- `FULL`: All ranks have complete data

#### Communication Functions
The class uses three specialized communication function objects:
1. **Simple Communication**: Basic tensor redistribution
2. **All-Reduce with LayerNorm**: Fused operations for efficiency
3. **Summable Tensor Pair**: Handles (hidden_states, residual) pairs

### Class Methods

#### `__init__(layer_scatter_modes, input_layernorm, post_attention_layernorm, allow_reduce_scatter, is_last_layer)`
- Initializes communicator with layer-specific scatter modes
- Sets up communication function objects based on modes
- Configures reduce-scatter support and layer position

#### `prepare_attn(hidden_states, residual, forward_batch, qaunt_format)`
- Prepares tensors for attention computation
- Applies input layer norm with optional all-reduce fusion
- Handles MXFP4 quantization for ROCm devices
- Ensures proper tensor distribution across ranks

#### `prepare_mlp(hidden_states, residual, forward_batch)`
- Prepares tensors for MLP/MoE computation
- Handles post-attention layer norm
- Manages all-reduce operations fused with normalization
- Optimizes for sparse (MoE) vs dense layers

#### `postprocess_layer(hidden_states, residual, forward_batch)`
- Processes layer output after MLP/MoE computation
- Manages summation of hidden states and residual
- Handles final communication patterns for layer output
- Supports reduce-scatter optimization

#### `should_use_reduce_scatter(forward_batch)`
- Determines if reduce-scatter should replace all-reduce
- Checks: permission, function type, and padding mode
- Optimizes communication for padded batches

#### `should_fuse_mlp_allreduce_with_next_layer(forward_batch)`
- Decides whether to fuse MLP all-reduce with next layer's input norm
- Considers: layer position, TP size, FlashInfer availability, batch size
- Avoids fusion with EAGLE speculative decoding

### Usage Pattern

```python
# In transformer layer forward pass:
# 1. Prepare for attention
hidden_states, residual = layer_comm.prepare_attn(
    hidden_states, residual, forward_batch
)

# 2. Attention computation
attn_output = self_attn(hidden_states)

# 3. Prepare for MLP
mlp_input, mlp_residual = layer_comm.prepare_mlp(
    attn_output, residual, forward_batch
)

# 4. MLP computation
mlp_output = mlp(mlp_input)

# 5. Postprocess layer
hidden_states, residual = layer_comm.postprocess_layer(
    mlp_output, mlp_residual, forward_batch
)
```

### Performance Optimizations

1. **Communication Fusion**: Combines all-reduce with layer norm
2. **Dynamic Pattern Selection**: Chooses optimal communication based on layer type
3. **Hardware Awareness**: Leverages FlashInfer for GPU optimization
4. **Overlap Maximization**: Strategic yielding for computation-communication overlap
5. **Memory Efficiency**: Reduce-scatter when beneficial

### Integration

The LayerCommunicator is essential for:
- Tensor parallel inference efficiency
- MoE model performance
- Multi-GPU communication optimization
- Hardware-specific acceleration

## Operations Strategy for MoE Models

The `operations_strategy.py` file defines execution strategies for MoE (Mixture of Experts) models, specifically optimized for Tensor Parallelism with Blocking Overlap (TBO) to maximize GPU utilization.

### Overview

**Location**: `python/sglang/srt/operations_strategy.py`

**Purpose**: Defines optimized operation sequences for MoE model layers, strategically placing yield points to overlap computation with communication and balance GPU resources.

### Key Components

#### OperationsStrategy Class
A dataclass containing:
- `operations`: List of operations to execute in sequence
- `deep_gemm_num_sms`: Number of SMs allocated for deep GEMM operations
- `tbo_delta_stages`: Number of delta stages for TBO synchronization

**Key Methods**:
- `concat()`: Combines multiple strategies into one
- `init_new_tbo()`: Creates TBO strategies for specific model types

### Model-Specific Strategies

#### DeepSeek MoE Strategy
1. **Prefill Mode** (`_compute_moe_deepseek_blog_prefill`)
   - Reserves SMs for DeepEP communication
   - Sequence: attention prep → attention → MoE gate → dispatch → yield → experts → yield → shared experts → output
   - 2 strategic yield points for optimal overlap

2. **Decode Mode** (`_compute_moe_deepseek_blog_decode`)
   - More aggressive yielding (4 points)
   - 2 delta stages for better TBO synchronization
   - Distributed operations with fine-grained overlap

#### Qwen3 MoE Strategy
Similar to DeepSeek but with optimizations:
- Fewer yield points in decode mode
- No shared experts architecture
- Slightly different operation ordering

### Key Features

1. **Tensor Parallelism with Blocking Overlap (TBO)**
   - Strategic `YieldOperation()` placement for computation-communication overlap
   - Delta stages control synchronization granularity
   - Maximizes GPU utilization during communication

2. **Dynamic Resource Management**
   - Allocates SMs between computation (deep_gemm) and communication (DeepEP)
   - Calculated based on device properties: `total_sms - deepep_sms`

3. **Mode-Specific Optimization**
   - Different strategies for prefill (EXTEND) vs decode (DECODE) phases
   - Prefill: Focus on throughput with fewer yields
   - Decode: Focus on latency with more fine-grained yielding

4. **Hardware-Aware Design**
   - Considers GPU SM count for resource allocation
   - Optimized for NVIDIA GPUs with CUDA
   - Balances load across available resources

### Usage Example

```python
# Create strategy for a layer
strategy = OperationsStrategy.init_new_tbo(
    layers=model_layers,
    forward_mode=ForwardMode.EXTEND  # or DECODE
)

# Execute operations
for op in strategy.operations:
    if isinstance(op, YieldOperation):
        yield  # Allow other operations to run
    else:
        result = op()
```

### Operation Flow (DeepSeek Decode)
1. Communication preparation for attention
2. Attention preparation → yield
3. Attention core computation
4. MLP communication prep
5. MoE gate computation → expert selection
6. Token dispatch A → shared experts → yield
7. Token dispatch B → expert computation → combine → yield
8. Final combine → yield → output → postprocess

### Performance Benefits

1. **Improved Utilization**: Overlaps communication with computation
2. **Better Balance**: Dynamic SM allocation based on workload
3. **Reduced Latency**: Strategic yielding minimizes stalls
4. **Scalability**: Designed for large-scale tensor parallel deployments

### Integration Points

- Used by MoE model layers during forward pass
- Integrates with DeepEP for expert communication
- Works with LayerCommunicator for tensor parallelism
- Supports FlashInfer optimizations where available

## SGLang Server Command-Line Arguments

SGLang provides over 180 command-line arguments to configure the server behavior. The arguments are defined in `python/sglang/srt/server_args.py` and can be viewed using `python -m sglang.launch_server --help`.

### Model and Tokenizer Arguments

- **`--model-path, --model`** (required): Path to model weights (local folder or Hugging Face repo ID)
- **`--tokenizer-path`**: Path to the tokenizer
- **`--tokenizer-worker-num`**: Number of tokenizer manager workers
- **`--tokenizer-mode`**: Tokenizer mode (auto/slow)
- **`--skip-tokenizer-init`**: Skip tokenizer initialization
- **`--load-format`**: Model weight format (auto/pt/safetensors/npcache/dummy/gguf/bitsandbytes/layered)
- **`--trust-remote-code`**: Allow custom model definitions from Hub
- **`--context-length`**: Model's maximum context length
- **`--revision`**: Specific model version (branch/tag/commit)

### Server Configuration

- **`--host`**: HTTP server host
- **`--port`**: HTTP server port
- **`--device`**: Device to use (cuda/xpu/hpu/npu/cpu)
- **`--tensor-parallel-size, --tp-size`**: Tensor parallelism size
- **`--pipeline-parallel-size, --pp-size`**: Pipeline parallelism size
- **`--data-parallel-size, --dp-size`**: Data parallelism size
- **`--dist-init-addr`**: Distributed backend initialization address
- **`--nnodes`**: Number of nodes in multi-node setup
- **`--node-rank`**: Node rank in multi-node setup

### Memory and Performance

- **`--mem-fraction-static`**: Memory fraction for static allocation
- **`--max-running-requests`**: Maximum concurrent requests
- **`--max-queued-requests`**: Maximum queued requests
- **`--max-total-tokens`**: Maximum tokens in memory pool
- **`--chunked-prefill-size`**: Maximum tokens per prefill chunk (-1 = disable)
- **`--max-prefill-tokens`**: Maximum tokens in prefill batch
- **`--schedule-policy`**: Request scheduling policy (lpm/random/fcfs/dfs-weight/lof)
- **`--schedule-conservativeness`**: Scheduling conservativeness (higher = more conservative)
- **`--page-size`**: Tokens per page in paged attention

### Attention and KV Cache

- **`--attention-backend`**: Attention kernel backend
- **`--prefill-attention-backend`**: Prefill-specific attention backend
- **`--decode-attention-backend`**: Decode-specific attention backend
- **`--kv-cache-dtype`**: KV cache data type (auto/fp8_e5m2/fp8_e4m3)
- **`--disable-radix-cache`**: Disable prefix caching
- **`--hybrid-kvcache-ratio`**: Mix ratio between uniform and hybrid KV buffers
- **`--enable-hierarchical-cache`**: Enable hierarchical caching
- **`--hicache-ratio`**: Host KV cache size ratio to device
- **`--hicache-storage-backend`**: Storage backend (file/mooncake/hf3fs/nixl)

### Quantization

- **`--dtype`**: Data type for weights/activations (auto/half/float16/bfloat16/float32)
- **`--quantization`**: Quantization method
- **`--quantization-param-path`**: Path to KV cache scaling factors
- **`--torchao-config`**: TorchAO optimization config (int8dq/int8wo/int4wo/fp8wo)

### Kernel Backends

- **`--sampling-backend`**: Sampling kernel backend (flashinfer/pytorch)
- **`--grammar-backend`**: Grammar-guided decoding backend
- **`--mm-attention-backend`**: Multimodal attention backend (sdpa/fa3/triton_attn)
- **`--disable-cuda-graph`**: Disable CUDA graph optimization
- **`--enable-torch-compile`**: Enable torch.compile optimization

### LoRA Support

- **`--enable-lora`**: Enable LoRA support
- **`--lora-paths`**: List of LoRA adapters to load
- **`--max-lora-rank`**: Maximum LoRA adapter rank
- **`--lora-target-modules`**: Target modules for LoRA
- **`--max-loras-per-batch`**: Maximum LoRAs per batch
- **`--max-loaded-loras`**: Maximum loaded LoRAs in CPU

### Speculative Decoding

- **`--speculative-algorithm`**: Algorithm (EAGLE/EAGLE3/NEXTN/STANDALONE)
- **`--speculative-draft-model-path`**: Draft model path
- **`--speculative-num-steps`**: Number of draft steps
- **`--speculative-eagle-topk`**: Top-k for EAGLE sampling
- **`--speculative-accept-threshold-single`**: Single token acceptance threshold
- **`--speculative-accept-threshold-acc`**: Accumulative acceptance threshold

### MoE and Expert Parallelism

- **`--expert-parallel-size, --ep-size, --ep`**: Expert parallelism size
- **`--moe-a2a-backend`**: MoE A2A backend (none/deepep)
- **`--moe-runner-backend`**: MoE runner backend
- **`--enable-flashinfer-allreduce-fusion`**: Enable all-reduce fusion
- **`--deepep-mode`**: DeepEP mode (normal/low_latency/auto)
- **`--enable-eplb`**: Enable EPLB load balancing
- **`--ep-num-redundant-experts`**: Number of redundant experts

### Data Parallel Attention

- **`--enable-dp-attention`**: Enable data parallel for attention
- **`--enable-dp-lm-head`**: Enable vocabulary parallel across DP groups
- **`--load-balance-method`**: Load balancing strategy (round_robin/shortest_queue/minimum_tokens)

### Multimodal and Tool Use

- **`--enable-multimodal`**: Enable multimodal functionality
- **`--tool-call-parser`**: Tool call parser (none/llama3_tool_call/qwen_tool_call/etc)
- **`--tool-server`**: Tool server URLs
- **`--reasoning-parser`**: Reasoning model parser (deepseek_r1/none)
- **`--mm-attention-backend`**: Multimodal attention backend

### Disaggregation

- **`--disaggregation-mode`**: PD disaggregation mode (prefill/decode)
- **`--disaggregation-transfer-backend`**: Transfer backend
- **`--disaggregation-bootstrap-port`**: Bootstrap server port
- **`--disaggregation-decode-tp`**: Decode TP size (prefill server only)
- **`--disaggregation-decode-dp`**: Decode DP size (prefill server only)

### Logging and Monitoring

- **`--log-level`**: Logging level
- **`--log-requests`**: Log request metadata and I/O
- **`--log-requests-level`**: Request logging verbosity (0-3)
- **`--enable-metrics`**: Enable Prometheus metrics
- **`--show-time-cost`**: Show custom timing marks
- **`--crash-dump-folder`**: Folder for crash dumps
- **`--collect-tokens-histogram`**: Collect token statistics

### Optimization Features

- **`--enable-mixed-chunk`**: Mix prefill and decode in batch
- **`--enable-two-batch-overlap`**: Enable two micro-batch overlap
- **`--num-continuous-decode-steps`**: Continuous decode steps (default: 1)
- **`--disable-overlap-schedule`**: Disable CPU-GPU overlap
- **`--cuda-graph-max-bs`**: Maximum CUDA graph batch size
- **`--enable-nan-detection`**: Enable NaN detection for debugging
- **`--enable-memory-saver`**: Enable memory saving features

### API Configuration

- **`--api-key`**: Server API key
- **`--served-model-name`**: Model name for API responses
- **`--chat-template`**: Chat template path or name
- **`--completion-template`**: Completion template
- **`--file-storage-path`**: File storage path
- **`--enable-cache-report`**: Report cached tokens in usage stats

### Debug and Development

- **`--skip-server-warmup`**: Skip server warmup
- **`--warmups`**: Custom warmup functions
- **`--watchdog-timeout`**: Watchdog timeout in seconds
- **`--random-seed`**: Random seed for reproducibility
- **`--delete-ckpt-after-loading`**: Delete checkpoint after loading
- **`--allow-auto-truncate`**: Auto-truncate long inputs
- **`--debug-tensor-dump-output-folder`**: Tensor dump output folder

### Usage Examples

```bash
# Basic server launch
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000

# Multi-GPU inference
python -m sglang.launch_server --model-path meta-llama/Llama-2-70b-chat-hf --tp-size 4

# Enable LoRA support
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --enable-lora --lora-paths adapter1=path/to/adapter1

# Quantized model
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --quantization fp8

# Speculative decoding
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --speculative-algorithm EAGLE --speculative-draft-model-path draft-model

# View all options
python -m sglang.launch_server --help
```

For the complete and most up-to-date list of arguments, always refer to the help output or the source code in `server_args.py`.

## KV Cache Transfer in Prefill-Decode Disaggregation

SGLang implements a sophisticated KV cache transfer mechanism for disaggregated serving where prefill and decode instances run separately. This section details the complete transfer process from prefill to decode instances.

### Architecture Overview

The disaggregation system consists of:

1. **Prefill Instances**: Handle prompt processing and KV cache generation
2. **Decode Instances**: Handle token generation using transferred KV cache
3. **Bootstrap Server**: Coordinates registration and connection between instances
4. **Transfer Backends**: Handle the actual data movement (Mooncake, Ascend, etc.)

### Core Components

#### Base Classes (`disaggregation/base/conn.py`)

- **`KVArgs`**: Contains KV cache configuration including data pointers, lengths, device info
- **`KVPoll`**: Status constants (Failed=0, Bootstrapping=1, WaitingForInput=2, Transferring=3, Success=4)
- **`BaseKVManager`**: Abstract manager for transfer state management
- **`BaseKVSender`**: Abstract prefill-side sender interface
- **`BaseKVReceiver`**: Abstract decode-side receiver interface
- **`BaseKVBootstrapServer`**: Abstract bootstrap server interface

#### Transfer Backends

**Mooncake Backend** (`disaggregation/mooncake/`):
- Uses RDMA/InfiniBand for high-speed transfers
- Supports batch memory registration and transfers
- Hardware-accelerated data movement
- Session-based connection management

**Ascend Backend** (`disaggregation/ascend/`):
- NPU-specific transfer engine
- Centralized storage coordination
- Optimized for Ascend hardware

### Bootstrap Server

The **Bootstrap Server** is a coordination service that enables communication between prefill and decode instances in SGLang's disaggregated serving architecture.

#### Purpose and Design

The Bootstrap Server acts as a **service discovery and connection broker** between:
- **Prefill instances** (handle prompt processing and KV cache generation)
- **Decode instances** (handle token generation using transferred KV cache)

#### Key Functions

1. **Service Registration**
   ```python
   # Prefill instances register themselves
   payload = {
       "role": "Prefill",
       "attn_tp_size": self.attn_tp_size,
       "attn_tp_rank": self.attn_tp_rank,
       "attn_dp_size": self.attn_dp_size,
       "attn_dp_rank": self.attn_dp_rank,
       "pp_size": self.pp_size,
       "pp_rank": self.pp_rank,
       "rank_ip": self.local_ip,
       "rank_port": self.rank_port,
   }
   ```

2. **Connection Discovery**
   ```python
   # Decode instances query for prefill connection info
   url = f"http://{bootstrap_addr}/route?engine_rank={target_tp_rank}&target_dp_group={target_dp_group}&target_pp_rank={target_pp_rank}"
   response = requests.get(url)
   bootstrap_info = response.json()  # Returns IP, port, and connection details
   ```

3. **Parallel Configuration Coordination**
   ```python
   # Synchronize parallel configuration between instances
   prefill_parallel_info = {
       "prefill_attn_tp_size": self.attn_tp_size,
       "prefill_dp_size": self.dp_size,
       "prefill_pp_size": self.pp_size,
   }
   ```

#### Implementation Details

**HTTP REST API** (`disaggregation/mooncake/conn.py:1572-1665`):
- **PUT `/route`**: Prefill instances register their connection details
- **GET `/route`**: Decode instances query for prefill instance locations
- **GET `/health`**: Health check endpoint for monitoring

**Registration Table Structure**:
```python
# Hierarchical storage: [dp_group][tp_rank][pp_rank] -> connection_info
self.prefill_port_table: Dict[int, Dict[int, Dict[int, Dict[str, Union[str, int]]]]] = {}

# Example entry:
prefill_port_table[dp_group][tp_rank][pp_rank] = {
    "rank_ip": "192.168.1.100",
    "rank_port": 12345,
}
```

#### Why It's Needed

**Problem**: Dynamic Service Discovery
In disaggregated serving:
- Prefill and decode instances may be on different nodes
- Instances need to find each other dynamically
- Multiple parallel configurations (TP, DP, PP) need coordination

**Solution**: Centralized Coordination
The Bootstrap Server provides:
1. **Service Registry**: Central location for instance registration
2. **Load Balancing**: Distributes connections across available instances
3. **Configuration Validation**: Ensures compatible parallel configurations
4. **Health Monitoring**: Tracks instance availability

#### Lifecycle Example

```python
# 1. Prefill instance starts and registers
prefill_instance.register_to_bootstrap(ip="192.168.1.100", port=12345)

# 2. Decode instance queries for prefill location
decode_instance.query_bootstrap(target_tp_rank=0, target_dp_group=0)
# Returns: {"rank_ip": "192.168.1.100", "rank_port": 12345}

# 3. Direct KV cache transfer between instances
decode_instance.connect_to_prefill("192.168.1.100:12345")
prefill_instance.transfer_kv_cache(decode_instance)
```

#### Key Design Benefits

1. **Decoupling**: Instances don't need to know each other's locations in advance
2. **Scalability**: New instances can join dynamically
3. **Fault Tolerance**: Failed instances can be detected and replaced
4. **Multi-tenancy**: Supports multiple parallel configurations simultaneously

The Bootstrap Server is essentially SGLang's **"service mesh control plane"** for disaggregated serving, enabling flexible and scalable deployment of prefill and decode instances across distributed infrastructure.

### Detailed Transfer Process

#### Phase 1: Bootstrap and Registration

1. **Prefill Instance Startup**:
   ```python
   # MooncakeKVManager.__init__ for PREFILL mode
   self.register_buffer_to_engine()  # Register KV cache buffers with transfer engine
   self._register_to_bootstrap()     # Register with bootstrap server
   ```

2. **Bootstrap Server Registration** (`mooncake/conn.py:1021-1065`):
   ```python
   # Prefill registers its connection info
   payload = {
       "role": "Prefill",
       "attn_tp_size": self.attn_tp_size,
       "attn_tp_rank": self.attn_tp_rank,
       "attn_dp_size": self.attn_dp_size,
       "attn_dp_rank": self.attn_dp_rank,
       "pp_size": self.pp_size,
       "pp_rank": self.pp_rank,
       "rank_ip": self.local_ip,
       "rank_port": self.rank_port,
   }
   ```

3. **Decode Instance Connection**:
   ```python
   # MooncakeKVReceiver.__init__
   bootstrap_info = self._get_bootstrap_info_from_server(target_tp_rank, target_dp_group, target_pp_rank)
   self._register_kv_args()  # Send decode instance KV buffer info to prefill
   ```

#### Phase 2: KV Cache Transfer Registration

1. **KV Args Registration** (`mooncake/conn.py:1420-1450`):
   ```python
   # Decode instance sends its KV buffer configuration to prefill
   packed_kv_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs)
   packed_aux_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs)

   sock.send_multipart([
       "None".encode("ascii"),           # Special registration message
       self.kv_mgr.local_ip.encode("ascii"),
       str(self.kv_mgr.rank_port).encode("ascii"),
       self.session_id.encode("ascii"),
       packed_kv_data_ptrs,             # Target buffer pointers
       packed_aux_data_ptrs,            # Auxiliary buffer pointers
       dst_tp_rank,                     # Target tensor parallel rank
       dst_attn_tp_size,                # Target attention TP size
       dst_kv_item_len,                 # KV item length on decode side
   ])
   ```

2. **Prefill Bootstrap Thread** (`mooncake/conn.py:810-843`):
   ```python
   # Prefill receives and stores decode instance configuration
   if room == "None":
       self.decode_kv_args_table[mooncake_session_id] = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
   ```

#### Phase 3: Transfer Request Initialization

1. **Transfer Request Setup** (`mooncake/conn.py:1473-1490`):
   ```python
   # Decode instance sends transfer request with KV indices
   sock.send_multipart([
       str(self.bootstrap_room).encode("ascii"),    # Unique request identifier
       self.kv_mgr.local_ip.encode("ascii"),       # Decode instance IP
       str(self.kv_mgr.rank_port).encode("ascii"), # Decode instance port
       self.session_id.encode("ascii"),            # Mooncake session ID
       kv_indices.tobytes() if not is_dummy else b"",  # Target KV indices
       str(aux_index).encode("ascii") if not is_dummy else b"",  # Auxiliary index
       str(self.required_dst_info_num).encode("ascii"),  # Expected responses
   ])
   ```

2. **Prefill Request Processing** (`mooncake/conn.py:830-842`):
   ```python
   # Prefill creates transfer info and updates status
   self.transfer_infos[room][mooncake_session_id] = TransferInfo.from_zmq(waiting_req_bytes)
   if len(self.transfer_infos[room]) == required_dst_info_num:
       self.update_status(room, KVPoll.WaitingForInput)
   ```

#### Phase 4: KV Cache Data Transfer

1. **Transfer Execution** (`mooncake/conn.py:315-418`):
   ```python
   # Main KV cache transfer function
   def send_kvcache(self, mooncake_session_id, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, executor):
       # Group contiguous indices for efficient transfer
       prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(prefill_kv_indices, dst_kv_indices)

       # Handle different model architectures (MLA vs standard)
       if self.is_mla_backend:
           # Single buffer per layer for MLA models
           layers_params = [(src_kv_ptrs[i], dst_kv_ptrs[i], kv_item_len) for i in range(layers_per_pp_stage)]
       else:
           # Separate K and V buffers for standard models
           layers_params = [(src_k_ptrs[i], dst_k_ptrs[i], kv_item_len) for i in range(layers_per_pp_stage)] + \
                          [(src_v_ptrs[i], dst_v_ptrs[i], kv_item_len) for i in range(layers_per_pp_stage)]
   ```

2. **Tensor Parallel Handling** (`mooncake/conn.py:420-583`):
   ```python
   # For different TP sizes between prefill and decode
   def send_kvcache_slice(self, mooncake_session_id, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, ...):
       # Calculate head distribution across TP ranks
       src_heads_per_rank = num_kv_heads
       dst_heads_per_rank = num_kv_heads * self.attn_tp_size // dst_attn_tp_size

       # Determine slice parameters based on TP configuration
       if self.attn_tp_size > dst_attn_tp_size:
           # Multiple prefill ranks → single decode rank
           dst_head_start_offset = local_tp_rank_in_group * src_heads_per_rank
       else:
           # Single prefill rank → multiple decode ranks
           src_head_start_offset = dst_tp_rank_in_group * dst_heads_per_rank
   ```

3. **Transfer Worker Execution** (`mooncake/conn.py:668-802`):
   ```python
   # Background transfer worker processes transfer chunks
   def transfer_worker(self, queue, executor):
       while True:
           kv_chunk = queue.get()
           for req in self.transfer_infos[kv_chunk.room].values():
               if not req.is_dummy:
                   # Execute actual KV cache transfer
                   ret = self.send_kvcache(req.mooncake_session_id, kv_chunk.prefill_kv_indices,
                                         target_rank_registration_info.dst_kv_ptrs, chunked_dst_kv_indice, executor)

                   # Send auxiliary data on last chunk
                   if kv_chunk.is_last and self.pp_group.is_last_rank:
                       ret = self.send_aux(req, kv_chunk.prefill_aux_index, target_rank_registration_info.dst_aux_ptrs)
   ```

#### Phase 5: Auxiliary Data Transfer

1. **Auxiliary Data Handling** (`mooncake/conn.py:584-630`):
   ```python
   # Transfer auxiliary data (non-KV cache data)
   def send_aux(self, req, prefill_aux_index, dst_aux_ptrs):
       transfer_blocks = []
       for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
           length = prefill_aux_item_lens[i]
           src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
           dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
           transfer_blocks.append((src_addr, dst_addr, length))
       return self._transfer_data(req.mooncake_session_id, transfer_blocks)
   ```

2. **TCP Fallback for Auxiliary Data** (`mooncake/conn.py:606-654`):
   ```python
   # Fallback to TCP for auxiliary data when needed
   def send_aux_tcp(self, req, prefill_aux_index, dst_aux_ptrs):
       for i in range(len(prefill_aux_ptrs)):
           data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)
           self.send_aux_data_to_endpoint(remote=req.endpoint, dst_port=req.dst_port,
                                        room=req.room, buffer_index=i, aux_index=req.dst_aux_index, data=data)
   ```

#### Phase 6: Status Synchronization and Completion

1. **Status Updates** (`mooncake/conn.py:655-667`):
   ```python
   # Sync transfer completion status to decode instance
   def sync_status_to_decode_endpoint(self, remote, dst_port, room, status, prefill_rank):
       self._connect(format_tcp_address(remote, dst_port)).send_multipart([
           str(room).encode("ascii"),
           str(status).encode("ascii"),
           str(prefill_rank).encode("ascii"),
       ])
   ```

2. **Decode Status Processing** (`mooncake/conn.py:869-898`):
   ```python
   # Decode instance processes status updates
   def decode_thread():
       while True:
           msg = self.server_socket.recv_multipart()
           (bootstrap_room, status, prefill_rank) = msg
           if status == KVPoll.Success:
               self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
               if arrived_response_num == expected_response_num:
                   self.update_status(bootstrap_room, KVPoll.Success)
   ```

### Key Data Structures

#### TransferInfo (`mooncake/conn.py:82-112`)
Contains per-request transfer metadata:
- `room`: Unique request identifier
- `endpoint`: Decode instance IP address
- `mooncake_session_id`: Transfer session identifier
- `dst_kv_indices`: Target KV cache locations
- `dst_aux_index`: Auxiliary data target index
- `is_dummy`: Whether this is a dummy request for TP coordination

#### KVArgsRegisterInfo (`mooncake/conn.py:116-140`)
Contains decode instance buffer configuration:
- `dst_kv_ptrs`: KV cache buffer pointers
- `dst_aux_ptrs`: Auxiliary buffer pointers
- `dst_tp_rank`: Target tensor parallel rank
- `dst_attn_tp_size`: Target attention TP size
- `dst_kv_item_len`: KV item length

#### TransferKVChunk (`mooncake/conn.py:72-78`)
Represents a chunk of KV cache to transfer:
- `room`: Request identifier
- `prefill_kv_indices`: Source KV cache indices
- `index_slice`: Slice of the total transfer
- `is_last`: Whether this is the final chunk
- `prefill_aux_index`: Auxiliary data source index

### Transfer Backends Implementation

#### Mooncake Transfer Engine (`mooncake/transfer_engine.py`)

```python
class MooncakeTransferEngine:
    def __init__(self, hostname, gpu_id, ib_device):
        # Initialize with RDMA/InfiniBand support
        self.engine = TransferEngine()
        self.initialize(hostname=hostname, device_name=ib_device)

    def batch_transfer_sync(self, session_id, buffers, peer_buffer_addresses, lengths):
        # High-performance batch transfer using RDMA
        return self.engine.batch_transfer_sync_write(session_id, buffers, peer_buffer_addresses, lengths)

    def batch_register(self, ptrs, lengths):
        # Register memory regions for RDMA access
        return self.engine.batch_register_memory(ptrs, lengths)
```

#### Memory Registration Process

1. **Buffer Registration** (`mooncake/conn.py:285-297`):
   ```python
   def register_buffer_to_engine(self):
       # Register KV cache buffers with transfer engine
       if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
           self.engine.batch_register(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)

       # Register auxiliary buffers
       if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
           self.engine.batch_register(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens)
   ```

### Error Handling and Recovery

#### Failure Detection (`mooncake/conn.py:899-952`)
```python
def heartbeat_checker():
    # Monitor prefill instance health
    for bootstrap_addr in addresses:
        response = session.get(f"http://{bootstrap_addr}/health", timeout=(2, 3))
        if response.status_code != 200:
            self.heartbeat_failures[bootstrap_addr] += 1
            if self.heartbeat_failures[bootstrap_addr] >= self.max_failures:
                self._handle_node_failure(bootstrap_addr)
```

#### Transfer Failure Handling (`mooncake/conn.py:740-761`)
```python
if ret != 0:  # Transfer failed
    with self.session_lock:
        self.session_failures[req.mooncake_session_id] += 1
        if self.session_failures[req.mooncake_session_id] >= 1:
            self.failed_sessions.add(req.mooncake_session_id)
    self.record_failure(kv_chunk.room, f"Failed to send kv chunk of {kv_chunk.room}")
    self.update_status(kv_chunk.room, KVPoll.Failed)
```

### Performance Optimizations

1. **Batch Transfers**: Group contiguous memory regions for efficient transfer
2. **Concurrent Processing**: Multi-threaded transfer workers with thread pools
3. **Memory Registration**: Pre-register buffers for RDMA efficiency
4. **Session Reuse**: Cache transfer sessions to avoid setup overhead
5. **Hardware Acceleration**: Leverage RDMA/InfiniBand for high-speed transfers

### Configuration Options

Key environment variables for tuning:
- `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`: Transfer worker thread count
- `SGLANG_DISAGGREGATION_QUEUE_SIZE`: Transfer queue count
- `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`: Bootstrap timeout (default: 300s)
- `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`: Transfer completion timeout (default: 300s)
- `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`: Health check interval (default: 5s)
- `SGLANG_MOONCAKE_CUSTOM_MEM_POOL`: Enable custom memory pool optimization

### Integration with SGLang Runtime

The disaggregation system integrates seamlessly with SGLang's core components:

1. **Memory Management**: Uses existing KV cache allocators and memory pools
2. **Scheduling**: Coordinates with request scheduling and batching
3. **Model Execution**: Transparent to model forward passes
4. **Tensor Parallelism**: Handles different TP sizes between prefill and decode
5. **Pipeline Parallelism**: Supports PP stages in prefill instances

This comprehensive KV cache transfer mechanism enables efficient disaggregated serving while maintaining high performance and reliability through sophisticated error handling, optimization strategies, and seamless integration with SGLang's runtime architecture.

## Speculative Decoding Implementation

SGLang implements **EAGLE** (Enhanced Approximate Generation with Leveraged features) speculative decoding, which uses a smaller "draft model" to generate multiple candidate tokens in parallel, then a larger "target model" to verify and accept/reject them.

### Supported Algorithms

From `python/sglang/srt/speculative/spec_info.py`:
- **EAGLE**: Standard EAGLE speculative decoding with tree-based speculation
- **EAGLE3**: Enhanced EAGLE variant with improved acceptance rates
- **STANDALONE**: Independent speculative worker for distributed setups
- **NONE**: Disabled speculative decoding

### Core Data Structures

#### **EagleDraftInput** (`python/sglang/srt/speculative/eagle_utils.py:56-119`)
Contains draft model inputs and outputs for the speculation process:
```python
@dataclass
class EagleDraftInput:
    # Draft model outputs (for verification)
    topk_p: torch.Tensor          # shape: (b, topk) - top-k probabilities
    topk_index: torch.Tensor      # shape: (b, topk) - top-k token indices
    hidden_states: torch.Tensor   # shape: (b, hidden_size) - hidden representations

    # Verification results (from target model)
    verified_id: torch.Tensor     # shape: (b,) - accepted tokens
    accept_length: torch.Tensor   # shape: (b,) - acceptance length per sequence
    accept_length_cpu: List[int]  # CPU copy of acceptance lengths

    # Attention metadata
    kv_indptr: torch.Tensor       # shape: (b+1,) - KV cache pointers
    kv_indices: torch.Tensor      # KV cache indices for attention
```

#### **EagleVerifyInput** (`python/sglang/srt/speculative/eagle_utils.py:246-284`)
Contains target model verification inputs with tree structure:
```python
@dataclass
class EagleVerifyInput:
    draft_token: torch.Tensor           # Draft tokens to verify
    custom_mask: torch.Tensor           # Attention mask for tree structure
    positions: torch.Tensor             # Position encodings
    retrive_index: torch.Tensor         # Tree traversal indices
    retrive_next_token: torch.Tensor    # Next token in tree structure
    retrive_next_sibling: torch.Tensor  # Sibling nodes in tree
    spec_steps: int                     # Number of speculative steps
    topk: int                          # Top-k for draft generation
    draft_token_num: int               # Total number of draft tokens
```

#### **EagleVerifyOutput** (`python/sglang/srt/speculative/eagle_utils.py:232-242`)
Results from target model verification:
```python
@dataclass
class EagleVerifyOutput:
    draft_input: EagleDraftInput              # Next draft input
    logits_output: LogitsProcessorOutput      # Target model logits
    verified_id: torch.Tensor                 # Accepted tokens
    accept_length_per_req_cpu: List[int]      # Acceptance lengths
    accepted_indices: torch.Tensor            # Indices of accepted tokens
```

### Speculative Decoding Process Flow

#### **Phase 1: Draft Generation**

1. **Tree Construction** (`python/sglang/srt/speculative/build_eagle_tree.py:51-98`):
   ```python
   def build_tree_kernel_efficient(verified_id, score_list, token_list, parents_list, ...):
       # Build speculation tree from draft model outputs
       parent_list, top_scores_index, draft_tokens = build_tree_kernel_efficient_preprocess(...)
       tree_mask = torch.full((num_verify_tokens * bs,), True, dtype=torch.bool, device=device)
   ```

2. **Token Selection** (`python/sglang/srt/speculative/eagle_utils.py:1096-1142`):
   ```python
   def select_top_k_tokens(topk_p, topk_index, hidden_states, scores, topk):
       if i == 0:  # First step after extend
           input_ids = topk_index.flatten()
           hidden_states = hidden_states.repeat_interleave(topk, dim=0)
       else:  # Later decode steps
           expand_scores = torch.mul(scores.unsqueeze(2), topk_p.reshape(-1, topk, topk))
           topk_cs_p, topk_cs_index = fast_topk(expand_scores.flatten(start_dim=1), topk)
   ```

#### **Phase 2: Target Model Verification**

1. **Batch Preparation** (`python/sglang/srt/speculative/eagle_utils.py:286-318`):
   ```python
   def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
       batch.input_ids = self.draft_token
       batch.out_cache_loc = batch.alloc_token_slots(len(batch.input_ids))
       # Allocate KV cache slots for all draft tokens
   ```

2. **Verification Process** (`python/sglang/srt/speculative/eagle_utils.py:358-517`):
   ```python
   def verify(self, batch, logits_output, token_to_kv_pool_allocator, ...):
       # Apply sampling (greedy vs probabilistic)
       if is_all_greedy:
           target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
           verify_tree_greedy(predict, accept_index, accept_length, candidates, ...)
       else:
           # Probabilistic verification with rejection sampling
           tree_speculative_sampling_target_only(predict, accept_index, ...)
   ```

#### **Phase 3: Acceptance/Rejection**

1. **Token Acceptance** (`python/sglang/srt/speculative/eagle_utils.py:529-548`):
   ```python
   for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
       for j, idx in enumerate(accept_index_row):
           if idx == -1: break
           id = predict_cpu[idx]
           req.output_ids.append(id)  # Accept token
           req.check_finished()       # Check if sequence finished
   ```

2. **KV Cache Management** (`python/sglang/srt/speculative/eagle_utils.py:569-634`):
   ```python
   # Free KV cache for rejected tokens
   accept_index = accept_index[accept_index != -1]
   verified_id = predict[accept_index]
   evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
   evict_mask[accept_index] = False
   token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
   ```

### Key Optimizations

#### **1. Tree-based Speculation**
- Uses tree structure to explore multiple token paths simultaneously
- `retrive_next_token` and `retrive_next_sibling` tensors encode the tree topology
- Allows parallel verification of multiple candidate sequences
- Builds speculation trees using `build_tree_kernel_efficient` for optimal token selection

#### **2. CUDA Graph Optimization**
- `EAGLEDraftCudaGraphRunner` and `EAGLEDraftExtendCudaGraphRunner` classes
- Pre-compiled CUDA graphs for draft model inference
- Reduces kernel launch overhead for repeated inference patterns
- Supports both decode and extend modes with separate graph runners

#### **3. Efficient Memory Management**
- Allocates KV cache for all draft tokens upfront during verification
- Immediately frees rejected tokens after verification to reclaim memory
- Page-aligned allocation for better memory efficiency
- Supports both paged and non-paged KV cache allocation strategies

#### **4. Advanced Rejection Sampling**
- Two-threshold system: `threshold_single` and `threshold_acc` for acceptance control
- Probabilistic acceptance based on draft vs target model probability ratios
- Uses `tree_speculative_sampling_target_only` kernel for efficient sampling
- Fallback to greedy verification on AMD/HIP builds where sampling kernels unavailable

#### **5. Grammar-Aware Speculation**
- Supports structured output with grammar constraints during speculation
- `traverse_tree` function performs DFS to validate tokens against grammar rules
- `generate_token_bitmask` creates vocabulary masks for valid tokens
- Integrates with SGLang's constrained generation backends

### Integration with SGLang Runtime

The speculative decoding system integrates seamlessly with SGLang's core components:

1. **Forward Modes**: Uses `ForwardMode.DRAFT_EXTEND` and `ForwardMode.TARGET_VERIFY`
2. **Batch Processing**: Works with `ScheduleBatch` for request management and scheduling
3. **Memory Management**: Integrates with existing KV cache allocators and memory pools
4. **Attention Backends**: Compatible with FlashAttention, FlashInfer, and other attention backends
5. **Distributed Inference**: Supports tensor parallelism with draft model workers
6. **CUDA Optimization**: Leverages CUDA graphs and custom kernels for performance

### Performance Benefits

The implementation achieves **1.5-3x speedup** for generation-heavy workloads by accepting multiple tokens per iteration while maintaining **identical output quality** to standard autoregressive decoding. Key performance characteristics:

- **Throughput**: Significantly higher tokens/second for long generation tasks
- **Latency**: Lower time-to-first-token for multi-turn conversations
- **Memory Efficiency**: Optimized KV cache usage with immediate cleanup
- **Scalability**: Works with distributed serving and disaggregated architectures

### Configuration Options

Key server arguments for speculative decoding:
- `--speculative-algorithm`: Algorithm choice (EAGLE/EAGLE3/STANDALONE)
- `--speculative-draft-model-path`: Path to the draft model
- `--speculative-num-steps`: Number of draft steps per iteration
- `--speculative-eagle-topk`: Top-k for EAGLE token selection
- `--speculative-accept-threshold-single`: Single token acceptance threshold
- `--speculative-accept-threshold-acc`: Accumulative acceptance threshold