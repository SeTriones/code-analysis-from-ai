# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepGEMM is a high-performance CUDA library for General Matrix Multiplications (GEMMs) optimized for NVIDIA SM90/SM100 architectures. It supports FP8 and BF16 data types for both normal and Mix-of-Experts (MoE) grouped scenarios. The library uses Just-In-Time (JIT) compilation for kernels and is designed for simplicity and performance.

## Development Commands

### Setup and Installation
```bash
# Clone with submodules (required)
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Development build (links includes and builds JIT module)
./develop.sh

# Installation
./install.sh
```

### Testing
```bash
# Run all GEMM implementation tests
python tests/test_layout.py
python tests/test_bf16.py
python tests/test_fp8.py
python tests/test_lazy_init.py
```

### Building
```bash
# Build only (creates development environment)
python setup.py build

# Create wheel package
python setup.py bdist_wheel
```

## Architecture

### Core Components

1. **Python Interface** (`deep_gemm/`):
   - `__init__.py`: Main API entry point, exports all GEMM functions and utilities
   - `utils/`: Layout utilities and math functions
   - `testing/`: Benchmarking and numerical testing utilities

2. **C++/CUDA Source** (`csrc/`):
   - `python_api.cpp`: Main Python binding interface
   - `jit/`: JIT compilation system for runtime kernel generation
   - `jit_kernels/`: Kernel templates and compilation logic
   - `apis/`: C++ API implementations
   - `utils/`: C++ utility functions

3. **CUDA Headers** (`deep_gemm/include/`):
   - `deep_gemm/impls/`: Architecture-specific GEMM implementations:
     - `sm90_fp8_gemm_1d2d.cuh`: SM90 FP8 kernels (1D warp, 2D thread)
     - `sm100_fp8_gemm_1d1d.cuh`/`sm100_fp8_gemm_1d2d.cuh`: SM100 FP8 kernels
     - `sm90_bf16_gemm.cuh`/`sm100_bf16_gemm.cuh`: BF16 kernels
   - `deep_gemm/common/`: Shared utilities and scheduler implementations

### Key GEMM Functions

The library provides several categories of GEMM operations:

1. **Standard Dense GEMMs**:
   - `fp8_gemm_nt`, `fp8_gemm_nn`, `fp8_gemm_tn`, `fp8_gemm_tt`
   - `bf16_gemm_nt`, `bf16_gemm_nn`, `bf16_gemm_tn`, `bf16_gemm_tt`

2. **Grouped GEMMs (Contiguous Layout)**:
   - `m_grouped_fp8_gemm_nt_contiguous`, `m_grouped_fp8_gemm_nn_contiguous`
   - `m_grouped_bf16_gemm_nt_contiguous`
   - `k_grouped_fp8_gemm_tn_contiguous` (for MoE weight backward)

3. **Grouped GEMMs (Masked Layout)**:
   - `m_grouped_fp8_gemm_nt_masked`
   - `m_grouped_bf16_gemm_nt_masked`

### JIT Compilation System

DeepGEMM implements a sophisticated JIT compilation system that generates optimized CUDA kernels at runtime based on input matrix dimensions and hardware capabilities.

#### Core Components

1. **Compiler Interface** (`csrc/jit/compiler.hpp`):
   - Abstract `Compiler` base class with two concrete implementations
   - `NVCCCompiler`: Uses NVCC for compilation (default, higher performance)
   - `NVRTCCompiler`: Uses NVRTC for compilation (faster, optional via `DG_JIT_USE_NVRTC=1`)

2. **Kernel Cache** (`csrc/jit/cache.hpp`):
   - `KernelRuntimeCache`: In-memory cache for compiled kernels
   - Filesystem-based persistent cache in `$HOME/.deep_gemm/cache/`
   - Cache invalidation based on library version and compiler signature

3. **Kernel Runtime** (`csrc/jit/kernel_runtime.hpp`):
   - Manages loaded CUDA kernels and their execution
   - Supports both CUDA driver API and runtime API (CUDA 12.8+)

4. **Kernel Templates** (`csrc/jit_kernels/impls/`):
   - Architecture-specific kernel implementations (SM90/SM100)
   - Code generation using `fmt::format` with template parameters
   - Runtime instantiation of template kernels

#### JIT Compilation Process

1. **Kernel Selection**:
   - Input parameters (matrix dimensions, data types) are analyzed
   - Heuristics in `csrc/jit_kernels/heuristics/` select optimal kernel configuration
   - Architecture-specific constraints are applied

2. **Code Generation**:
   - Kernel templates are instantiated with specific template parameters
   - C++ code is generated using string formatting with proper includes
   - Template parameters include tile sizes, data types, and kernel configurations

3. **Compilation**:
   - **NVCC Path**: Uses external `nvcc` process with optimized flags
   - **NVRTC Path**: Uses in-process compilation with PCH support (12.8+)
   - Both paths generate CUBIN files with architecture-specific optimizations

4. **Caching**:
   - Kernel signature includes: name, library version, compiler signature, flags, and code
   - Cache directory structure: `$DG_JIT_CACHE_DIR/cache/kernel.name.<hash>/`
   - Atomic file operations ensure cache consistency

5. **Kernel Loading**:
   - CUBIN files are loaded using CUDA driver/runtime API
   - Kernel handles are cached in memory for subsequent calls
   - Support for cluster launch configurations (CUDA 9.0+)

#### Key Features

- **Dual Compiler Support**: NVCC for maximum performance, NVRTC for faster compilation
- **Smart Caching**: Persistent filesystem cache with in-memory runtime cache
- **Architecture Optimization**: Generates code specific to SM90/SM100 capabilities
- **Template Instantiation**: Runtime generation of template kernels with optimal parameters
- **Debugging Support**: Comprehensive environment variables for debugging and profiling
- **Atomic Operations**: Thread-safe cache operations using atomic file renames

## Dynamic Kernel Generation

DeepGEMM's JIT system dynamically generates specialized CUDA kernels at runtime based on input matrix dimensions, data types, and hardware capabilities. This enables optimal performance for each specific GEMM operation.

### Kernel Generation Process

#### 1. Configuration Selection (`csrc/jit_kernels/heuristics/common.hpp`)

The `get_best_config()` function analyzes input parameters and selects optimal kernel configurations:

**Input Analysis:**
- Matrix dimensions (M, N, K)
- Data types (FP8, BF16 for inputs/outputs)
- Memory layout (NT, NN, TN, TT)
- GEMM type (Normal, M-grouped, K-grouped, Masked)
- Hardware architecture (SM90/SM100)

**Block Size Selection:**
```cpp
// Dynamic M/N block selection based on input size and GEMM type
const auto& block_ms = gemm_type == GemmType::MGroupedContiguous ?
    std::vector{get_mk_alignment_for_contiguous_layout()} : std::vector{64, 128, 256};
std::vector<int> block_ns;
for (int i = 16; i <= 256; i += 16)
    block_ns.push_back(i);

// Fixed K block based on data type
const auto& block_k = 128 / static_cast<int>(c10::elementSize(ab_dtype));
```

**Optimization Heuristics:**
- Minimizes wave count (number of kernel launches per SM)
- Maximizes last wave utilization
- Prefers larger block sizes when wave counts are equal
- Considers hardware-specific constraints (shared memory, registers)

**Multicast Configuration:**
- Determines optimal TMA multicast usage (1x or 2x)
- Chooses between A or B matrix broadcasting
- Only enabled for matrices ≥ 512 in target dimension

**Pipeline Stage Selection:**
- Maximizes number of stages (up to 12) within shared memory limits
- Balances memory throughput and latency hiding
- Considers data type and block size constraints

#### 2. Template Instantiation (`csrc/jit_kernels/impls/`)

Each architecture-specific kernel implementation uses template parameters for specialization:

**Template Structure:**
```cpp
static std::string generate_impl(const Args& args) {
    return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<
        {}, {}, {},  // M, N, K dimensions
        {},          // Number of groups
        {}, {}, {},  // Block M, N, K sizes
        {},          // Swizzle mode
        {}, {},      // Number of stages, last stages
        {}, {},      // TMA threads, math threads
        {}, {},      // Multicast config
        {}, {}       // Number of SMs, GEMM type
    >);
}};
)",
    get_compiled_dim(args.m, 'm', args.compiled_dims),  // Runtime M dimension
    get_compiled_dim(args.n, 'n', args.compiled_dims),  // Runtime N dimension
    get_compiled_dim(args.k, 'k', args.compiled_dims),  // Runtime K dimension
    args.num_groups,                                    // Number of groups
    args.gemm_config.block_m,                          // Template block M
    args.gemm_config.block_n,                          // Template block N
    args.gemm_config.block_k,                          // Template block K
    args.gemm_config.smem_config.swizzle_cd_mode,      // Memory swizzling
    args.gemm_config.num_stages,                       // Pipeline stages
    args.gemm_config.num_last_stages,                 // Last stage count
    args.gemm_config.thread_config.num_tma_threads,   // TMA thread count
    args.gemm_config.thread_config.num_math_threads,  // Math thread count
    args.gemm_config.multicast_config.num_multicast,   // Multicast factor
    args.gemm_config.multicast_config.is_multicast_on_a, // Multicast target
    args.gemm_config.num_sms,                          // SM utilization
    to_string(args.gemm_config.gemm_type));            // GEMM type
}
```

**Template Parameters:**
- **Compile-time Constants**: Block sizes, data types, pipeline stages
- **Runtime Values**: Matrix dimensions, tensor pointers, tensor maps
- **Architecture Features**: TMA thread counts, multicast configurations

#### 3. Code Generation

**String Formatting:**
- Uses `fmt::format` for template instantiation
- Converts C++ enums and types to string representations
- Handles architecture-specific parameter formatting

**Type Conversion Utilities:**
```cpp
static std::string to_string(const cute::UMMA::Major& major) {
    switch (major) {
        case cute::UMMA::Major::K:  return "cute::UMMA::Major::K";
        case cute::UMMA::Major::MN: return "cute::UMMA::Major::MN";
    }
}

static std::string to_string(const at::ScalarType& dtype) {
    switch (dtype) {
        case torch::kInt:           return "int";
        case torch::kFloat:         return "float";
        case torch::kBFloat16:      return "cutlass::bfloat16_t";
        // ... more types
    }
}
```

#### 4. TMA Descriptor Generation

**Tensor Memory Accelerator (TMA) Setup:**
- Generates TMA descriptors for each matrix (A, B, C, D, scaling factors)
- Handles different memory layouts and strides
- Configures swizzling modes for optimal memory access patterns

```cpp
const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
    SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
    config.block_k,
    static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
    config.smem_config.swizzle_a_mode);
```

#### 5. Kernel Compilation and Execution

**Compilation Flow:**
```cpp
// Generate specialized kernel code
const auto& code = SM90FP8Gemm1D2DRuntime::generate(args);

// Compile with NVCC or NVRTC
const auto& runtime = compiler->build("sm90_fp8_gemm_1d2d", code);

// Launch the compiled kernel
SM90FP8Gemm1D2DRuntime::launch(runtime, args);
```

### Key Advantages

1. **Perfect Specialization**: Each kernel is optimized for exact input dimensions and hardware
2. **Template Metaprogramming**: Leverages C++ template system for compile-time optimization
3. **Runtime Flexibility**: Supports arbitrary matrix sizes without precompiled kernels
4. **Hardware Optimization**: Generates code specific to SM90/SM100 capabilities
5. **Memory Layout Awareness**: Optimizes for different memory layouts (NT, NN, TN, TT)

### Generated Kernel Characteristics

- **Block Size Optimization**: Dynamic tile sizes based on matrix dimensions
- **Pipeline Tuning**: Optimal number of stages for memory latency hiding
- **Thread Configuration**: Balanced TMA vs math thread allocation
- **Memory Patterns**: Swizzling modes for conflict-free shared memory access
- **Multicast Support**: TMA multicast for large matrix dimensions
- **Shared Memory Management**: Optimal usage within hardware limits

This dynamic generation approach allows DeepGEMM to achieve performance comparable to expert-tuned libraries while maintaining flexibility for arbitrary input sizes and configurations.

## NVCC vs NVRTC Compilation

DeepGEMM provides two distinct compilation paths for JIT kernel generation, each with different tradeoffs in terms of performance, compilation speed, and functionality.

### Compilation Mechanisms

#### NVCC (NVIDIA CUDA Compiler)

**Process:**
```cpp
// External process compilation
const auto& command = fmt::format("{} {} -o {} {}",
    nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
const auto& [return_code, output] = call_external_command(command);
```

**Key Characteristics:**
- **External Process**: Spawns separate `nvcc` process for each compilation
- **Filesystem I/O**: Writes temporary files, reads output from stdout/stderr
- **Full Optimizations**: Access to complete NVCC optimization pipeline
- **Host Compiler Integration**: Uses system compiler for host code generation

**Compilation Flags:**
```bash
-std=c++20 --gpu-architecture=sm_{} \
--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi \
-cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda
```

#### NVRTC (NVIDIA Runtime Compiler)

**Process:**
```cpp
// In-process compilation
nvrtcProgram program;
DG_NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
const auto& compile_result = nvrtcCompileProgram(program, num_options, option_cstrs);

// Direct CUBIN access
DG_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
DG_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));
```

**Key Characteristics:**
- **In-Process**: Compilation occurs within the same process
- **Memory-Based**: No temporary files, direct memory access
- **Limited Optimizations**: Subset of NVCC optimizations
- **API-Based**: Programmatic interface with fine-grained control

**Compilation Flags:**
```bash
-std=c++20 --gpu-architecture=sm_{} -default-device \
--pch  # Precompiled Headers (12.8+)
```

### Key Differences

| Aspect | NVCC | NVRTC |
|--------|------|-------|
| **Process Model** | External process | In-process library |
| **Compilation Speed** | Slower (process overhead) | Faster (no process spawn) |
| **Performance** | Higher (full optimizations) | Lower (limited optimizations) |
| **Memory Usage** | Higher (process isolation) | Lower (shared memory space) |
| **Precompiled Headers** | Not supported | Supported (12.8+) |
| **Debugging** | Easier (traditional tools) | Harder (API-based) |
| **Error Reporting** | Stdout/stderr capture | Programmatic log access |

### Performance Tradeoffs

#### NVCC Advantages
1. **Maximum Performance**: Full optimization pipeline including:
   - Advanced register allocation
   - Instruction scheduling
   - Memory coalescing optimization
   - Loop unrolling and vectorization

2. **Mature Tooling**: Integration with:
   - CUDA profiler tools
   - Traditional debugging workflows
   - IDE support and syntax highlighting

3. **Architecture Features**: Better support for:
   - Latest SM90/SM100 features
   - Complex tensor core operations
   - Advanced memory patterns

#### NVRTC Advantages
1. **Compilation Speed**: Significant speedup from:
   - No process spawning overhead
   - Precompiled headers (PCH) support
   - Direct memory operations
   - Reduced filesystem I/O

2. **Resource Efficiency**: Lower overhead from:
   - Shared memory space
   - No temporary file management
   - Reduced system calls

3. **Integration Benefits**: Better for:
   - Embedded applications
   - Runtime-sensitive workloads
   - Memory-constrained environments

### Version Requirements

**Minimum Supported Versions:**
- **NVCC**: 12.3+ (recommended 12.9+ for best performance)
- **NVRTC**: 12.3+ (recommended 12.8+ for PCH support)

**Feature Support:**
```cpp
// NVCC 12.9+ - Architecture family suffix support
const auto& arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);

// NVRTC 12.8+ - Precompiled Headers
if (major > 12 or minor >= 8) {
    pch_flags = "--pch ";  // Vital for compilation speed
}
```

### Usage Scenarios

#### Use NVCC When:
- **Maximum performance is critical** (production workloads)
- **Running on varied hardware configurations**
- **Complex kernels requiring advanced optimizations**
- **Debugging and development workflows**
- **Long-running applications with infrequent compilation**

#### Use NVRTC When:
- **Compilation latency is critical** (interactive applications)
- **Memory-constrained environments**
- **Frequent kernel compilation** (exploratory workloads)
- **Embedded or containerized deployments**
- **Development and prototyping phases**

### Selection Mechanism

```cpp
// Compiler selection based on environment variable
static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();  // Faster compilation
    } else {
        return std::make_shared<NVCCCompiler>();   // Better performance
    }
});
```

**Environment Variable Control:**
```bash
# Use NVRTC for faster compilation (may have performance impact)
export DG_JIT_USE_NVRTC=1

# Use NVCC for maximum performance (default)
export DG_JIT_USE_NVRTC=0
```

### Practical Considerations

**Cache Behavior:**
- Both compilers use the same caching mechanism
- Cache keys include compiler signature to avoid cross-compiler issues
- Performance differences primarily affect first-time compilation

**Debugging Support:**
```bash
# Enable detailed compilation output for both compilers
export DG_JIT_DEBUG=1
export DG_JIT_PRINT_COMPILER_COMMAND=1

# Enable PTXAS verbose output (NVCC only)
export DG_JIT_PTXAS_VERBOSE=1
```

**Recommendation:**
- **Development/Prototyping**: Use NVRTC for faster iteration
- **Production/Benchmarking**: Use NVCC for maximum performance
- **Hybrid Approach**: Switch based on workload characteristics

### Environment Variables

Key environment variables for development:

- `DG_JIT_DEBUG`: Enable JIT debugging output
- `DG_JIT_CACHE_DIR`: Set cache directory (default: `$HOME/.deep_gemm`)
- `DG_JIT_USE_NVRTC`: Use NVRTC instead of NVCC (faster compilation)
- `DG_JIT_PTXAS_VERBOSE`: Show detailed PTXAS output
- `DG_PRINT_CONFIGS`: Print selected kernel configurations

### Dependencies

- **CUDA Toolkit**: 12.3+ for SM90, 12.9+ for SM100
- **PyTorch**: 2.1+ for Python bindings
- **CUTLASS**: 4.0+ (included as submodule)
- **fmt**: Formatting library (included as submodule)

### Architecture Support

- **SM90**: Hopper architecture with NT memory layout support
- **SM100**: Blackwell architecture with full memory layout support (NT, TN, NN, TT)

### Code Style and Patterns

- The library avoids heavy template usage compared to CUTLASS
- Kernel implementations focus on readability and performance
- Uses CuTe library concepts but with simplified abstractions
- JIT compilation allows runtime optimization based on input shapes

### Testing Strategy

Tests are located in `tests/` and cover:
- Layout correctness
- Numerical accuracy for FP8/BF16
- Lazy initialization behavior
- Benchmarking capabilities

Use the provided test scripts to verify correctness during development.